use crate::neuron::{Activation, BatchLayerCache};
use crate::TensorHandle;
use cblas::{Layout, Transpose};
use std::cell::RefCell;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_setzero_ps,
    _mm256_storeu_ps,
    _mm256_sub_ps,
};

use crate::conv1d_backward::backward_direct;
pub use crate::grouped_backward::backward_direct as grouped_conv1d_backward;

pub struct BackwardWorkspace {
    pub(crate) weights: Vec<f32>,
    positions: Vec<i32>,

    weight_grads: Vec<f32>,
    bias_grads: Vec<f32>,

    col: Vec<f32>,
    col_grads: Vec<f32>,
    group_grad: Vec<f32>,

    first_weight_grads: Vec<f32>,
    first_bias_grads: Vec<f32>,

    second_weight_grads: Vec<f32>,
    second_bias_grads: Vec<f32>,

    hidden_grads: Vec<f32>,

    embedding_grads: Vec<f32>,

    depthwise_weights_kmajor: Vec<f32>,
    depthwise_weight_grads_kmajor: Vec<f32>,

    mixer_global_grads: Vec<f32>,
    mixer_score_grads: Vec<f32>,
}

impl BackwardWorkspace {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            positions: Vec::new(),

            weight_grads: Vec::new(),
            bias_grads: Vec::new(),

            col: Vec::new(),
            col_grads: Vec::new(),
            group_grad: Vec::new(),

            first_weight_grads: Vec::new(),
            first_bias_grads: Vec::new(),

            second_weight_grads: Vec::new(),
            second_bias_grads: Vec::new(),

            hidden_grads: Vec::new(),

            embedding_grads: Vec::new(),

            depthwise_weights_kmajor: Vec::new(),
            depthwise_weight_grads_kmajor: Vec::new(),

            mixer_global_grads: Vec::new(),
            mixer_score_grads: Vec::new(),
        }
    }
}

thread_local! {
    static BACKWARD_WORKSPACE:
        RefCell<BackwardWorkspace> =
            RefCell::new(
                BackwardWorkspace::new()
            );
}

#[inline]
pub fn make_conv_positions(
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
) -> Vec<i32> {
    let total =
        output_length * kernel_size;

    let mut positions =
        Vec::with_capacity(total);

    for out_pos in 0..output_length {
        let base =
            out_pos * stride;

        for k in 0..kernel_size {
            let pos =
                if causal {
                    base as i32
                        + k as i32
                        - (kernel_size - 1) as i32
                } else {
                    base as i32
                        + k as i32
                        - padding as i32
                };

            positions.push(pos);
        }
    }

    positions
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_f32_slice_avx2(
    dst: &mut [f32],
    src: &[f32],
) { unsafe {
    debug_assert_eq!(dst.len(), src.len());

    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let a = _mm256_loadu_ps(
            dst.as_ptr().add(i)
        );

        let b = _mm256_loadu_ps(
            src.as_ptr().add(i)
        );

        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_add_ps(a, b),
        );

        i += 8;
    }

    while i < len {
        dst[i] += src[i];
        i += 1;
    }
}}

#[inline]
pub fn add_f32_slice_simd(
    dst: &mut [f32],
    src: &[f32],
) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                add_f32_slice_avx2(dst, src);
            }

            return;
        }
    }

    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_add_f32_slice_avx2(
    dst: &mut [f32],
    a: &[f32],
    b: &[f32],
) { unsafe {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let x = _mm256_loadu_ps(
            a.as_ptr().add(i)
        );

        let y = _mm256_loadu_ps(
            b.as_ptr().add(i)
        );

        let old = _mm256_loadu_ps(
            dst.as_ptr().add(i)
        );

        let result = _mm256_fmadd_ps(
            x,
            y,
            old,
        );

        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            result,
        );

        i += 8;
    }

    while i < len {
        dst[i] += a[i] * b[i];
        i += 1;
    }
}}

#[inline]
fn mul_add_f32_slice_simd(
    dst: &mut [f32],
    a: &[f32],
    b: &[f32],
) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                mul_add_f32_slice_avx2(
                    dst,
                    a,
                    b,
                );
            }

            return;
        }
    }

    for i in 0..dst.len() {
        dst[i] += a[i] * b[i];
    }
}

pub fn conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    let weights_len =
        out_channels
            * kernel_size
            * in_channels;

    workspace.weights.resize(
        weights_len,
        0.0,
    );

    crate::handle_data_slice(
        weight_handles,
        &mut workspace.weights,
    );

    let (
        input_grads,
        weight_grads,
        bias_grads,
    ) = backward_direct(
        input,
        grad,
        output,
        batch_size,
        input_length,
        output_length,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        causal,
        &workspace.weights,
        activation,
    );

    crate::add_handle_grad_slices_2(
        weight_handles,
        &weight_grads,
        bias_handles,
        &bias_grads,
    );

    input_grads
}

pub fn depthwise_conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    debug_assert_eq!(
        input.len(),
        batch_size * input_length * in_channels
    );

    debug_assert_eq!(
        output.len(),
        batch_size * output_length * in_channels
    );

    debug_assert_eq!(
        grad.len(),
        output.len()
    );

    debug_assert_eq!(
        weight_handles.len(),
        in_channels * kernel_size
    );

    debug_assert_eq!(
        bias_handles.len(),
        in_channels
    );

    workspace.weights.resize(
        weight_handles.len(),
        0.0,
    );

    crate::handle_data_slice(
        weight_handles,
        &mut workspace.weights,
    );

    workspace.positions =
        make_conv_positions(
            output_length,
            kernel_size,
            stride,
            padding,
            causal,
        );

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    workspace.weight_grads.resize(
        in_channels * kernel_size,
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        in_channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    // K-major temporary layout:
    //
    // [k0 c0, k0 c1, ..., k0 cN,
    //  k1 c0, k1 c1, ..., k1 cN, ...]
    let depthwise_size =
        in_channels * kernel_size;

    workspace.depthwise_weights_kmajor.resize(
        depthwise_size,
        0.0,
    );

    workspace.depthwise_weight_grads_kmajor.resize(
        depthwise_size,
        0.0,
    );

    workspace.depthwise_weight_grads_kmajor.fill(0.0);

    for c in 0..in_channels {
        for k in 0..kernel_size {
            workspace.depthwise_weights_kmajor[
                k * in_channels + c
                ] =
                workspace.weights[
                    c * kernel_size + k
                    ];
        }
    }

    let mut input_grads =
        vec![0.0; input.len()];

    for b in 0..batch_size {
        let input_batch_base =
            b * input_length * in_channels;

        let output_batch_base =
            b * output_length * in_channels;

        for out_pos in 0..output_length {
            let grad_base =
                output_batch_base
                    + out_pos * in_channels;

            // Bias gradient:
            //
            // bias_grad[c] += grad[c]
            add_f32_slice_simd(
                &mut workspace.bias_grads,
                &grad[
                    grad_base..grad_base + in_channels
                    ],
            );

            let position_base =
                out_pos * kernel_size;

            for k in 0..kernel_size {
                let src_pos =
                    workspace.positions[
                        position_base + k
                        ];

                if src_pos < 0
                    || src_pos as usize >= input_length
                {
                    continue;
                }

                let src_base =
                    input_batch_base
                        + src_pos as usize
                        * in_channels;

                let weight_base =
                    k * in_channels;

                mul_add_f32_slice_simd(
                    &mut workspace.depthwise_weight_grads_kmajor[
                        weight_base
                            ..weight_base + in_channels
                        ],
                    &input[
                        src_base..src_base + in_channels
                        ],
                    &grad[
                        grad_base..grad_base + in_channels
                        ],
                );

                mul_add_f32_slice_simd(
                    &mut input_grads[
                        src_base..src_base + in_channels
                        ],
                    &workspace.depthwise_weights_kmajor[
                        weight_base
                            ..weight_base + in_channels
                        ],
                    &grad[
                        grad_base..grad_base + in_channels
                        ],
                );
            }
        }
    }

    // Convert k-major gradients back to the original
    // channel-major parameter layout.
    for c in 0..in_channels {
        for k in 0..kernel_size {
            workspace.weight_grads[
                c * kernel_size + k
                ] =
                workspace.depthwise_weight_grads_kmajor[
                    k * in_channels + c
                    ];
        }
    }

    crate::add_handle_grad_slices(
        weight_handles,
        &workspace.weight_grads,
    );

    crate::add_handle_grad_slices(
        bias_handles,
        &workspace.bias_grads,
    );

    input_grads
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn channel_scale_backward_avx2(
    input: &[f32],
    grad: &[f32],
    input_grads: &mut [f32],
    scale_grads: &mut [f32],
    bias_grads: &mut [f32],
    scales: &[f32],
    channels: usize,
) { unsafe {
    let positions =
        input.len() / channels;

    for position in 0..positions {
        let offset =
            position * channels;

        let mut c = 0;

        while c + 8 <= channels {
            let x =
                _mm256_loadu_ps(
                    input.as_ptr()
                        .add(offset + c)
                );

            let g =
                _mm256_loadu_ps(
                    grad.as_ptr()
                        .add(offset + c)
                );

            let scale =
                _mm256_loadu_ps(
                    scales.as_ptr()
                        .add(c)
                );

            let old_dx =
                _mm256_loadu_ps(
                    input_grads.as_ptr()
                        .add(offset + c)
                );

            let old_dscale =
                _mm256_loadu_ps(
                    scale_grads.as_ptr()
                        .add(c)
                );

            let old_dbias =
                _mm256_loadu_ps(
                    bias_grads.as_ptr()
                        .add(c)
                );

            let dx =
                _mm256_fmadd_ps(
                    g,
                    scale,
                    old_dx,
                );

            let dscale =
                _mm256_fmadd_ps(
                    g,
                    x,
                    old_dscale,
                );

            let dbias =
                _mm256_add_ps(
                    old_dbias,
                    g,
                );

            _mm256_storeu_ps(
                input_grads.as_mut_ptr()
                    .add(offset + c),
                dx,
            );

            _mm256_storeu_ps(
                scale_grads.as_mut_ptr()
                    .add(c),
                dscale,
            );

            _mm256_storeu_ps(
                bias_grads.as_mut_ptr()
                    .add(c),
                dbias,
            );

            c += 8;
        }

        // Scalar tail.
        while c < channels {
            let x =
                input[offset + c];

            let g =
                grad[offset + c];

            input_grads[offset + c] +=
                g * scales[c];

            scale_grads[c] +=
                g * x;

            bias_grads[c] +=
                g;

            c += 1;
        }
    }
}}

#[inline]
fn channel_scale_backward_simd(
    input: &[f32],
    grad: &[f32],
    input_grads: &mut [f32],
    scale_grads: &mut [f32],
    bias_grads: &mut [f32],
    scales: &[f32],
    channels: usize,
) {
    debug_assert_eq!(
        input.len(),
        grad.len()
    );

    debug_assert_eq!(
        input.len(),
        input_grads.len()
    );

    debug_assert_eq!(
        scale_grads.len(),
        channels
    );

    debug_assert_eq!(
        bias_grads.len(),
        channels
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                channel_scale_backward_avx2(
                    input,
                    grad,
                    input_grads,
                    scale_grads,
                    bias_grads,
                    scales,
                    channels,
                );
            }

            return;
        }
    }

    let positions =
        input.len() / channels;

    for position in 0..positions {
        let offset =
            position * channels;

        for c in 0..channels {
            let x =
                input[offset + c];

            let g =
                grad[offset + c];

            input_grads[offset + c] +=
                g * scales[c];

            scale_grads[c] +=
                g * x;

            bias_grads[c] +=
                g;
        }
    }
}

pub fn channel_scale_backward(
    input: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    channels: usize,
    scale_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    debug_assert_eq!(
        input.len(),
        grad.len()
    );

    debug_assert_eq!(
        input.len() % batch_size,
        0
    );

    debug_assert_eq!(
        scale_handles.len(),
        channels
    );

    debug_assert_eq!(
        bias_handles.len(),
        channels
    );

    workspace.weights.resize(
        channels,
        0.0,
    );

    crate::handle_data_slice(
        scale_handles,
        &mut workspace.weights,
    );

    workspace.weight_grads.resize(
        channels,
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    let mut input_grads =
        vec![0.0; input.len()];

    let sequence_size =
        input.len() / batch_size;

    debug_assert_eq!(
        sequence_size % channels,
        0
    );

    for b in 0..batch_size {
        let base =
            b * sequence_size;

        channel_scale_backward_simd(
            &input[
                base..base + sequence_size
                ],
            &grad[
                base..base + sequence_size
                ],
            &mut input_grads[
                base..base + sequence_size
                ],
            &mut workspace.weight_grads,
            &mut workspace.bias_grads,
            &workspace.weights,
            channels,
        );
    }

    crate::add_handle_grad_slices_2(
        scale_handles,
        &workspace.weight_grads,
        bias_handles,
        &workspace.bias_grads,
    );

    input_grads
}

pub fn low_rank_pointwise_backward(
    input: &[f32],
    hidden: &[f32],
    output: &[f32],
    grad: &mut [f32],
    rows: usize,
    in_channels: usize,
    rank: usize,
    out_channels: usize,
    first_weights: &[f32],
    second_weights: &[f32],
    first_weight_handles: &[TensorHandle],
    first_bias_handles: &[TensorHandle],
    second_weight_handles: &[TensorHandle],
    second_bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {

    // ============================================================
    // Backprop through SECOND activation.
    //
    // grad:
    // dL/d(output)
    //
    // becomes:
    // dL/d(output_pre)
    // ============================================================

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    // ============================================================
    // Allocate/reset second-layer gradients.
    // ============================================================

    workspace.second_weight_grads.resize(
        out_channels * rank,
        0.0,
    );

    workspace.second_weight_grads.fill(0.0);

    workspace.second_bias_grads.resize(
        out_channels,
        0.0,
    );

    workspace.second_bias_grads.fill(0.0);

    // ============================================================
    // dW2 = grad^T @ hidden
    //
    // grad   [rows, out_channels]
    // hidden [rows, rank]
    //
    // dW2    [out_channels, rank]
    // ============================================================

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            out_channels as i32,
            rank as i32,
            rows as i32,
            1.0,
            grad,
            out_channels as i32,
            hidden,
            rank as i32,
            0.0,
            &mut workspace.second_weight_grads,
            rank as i32,
        );
    }

    // ============================================================
    // db2 = sum over rows
    // ============================================================

    for row in 0..rows {
        let base =
            row * out_channels;

        for oc in 0..out_channels {
            workspace.second_bias_grads[oc] +=
                grad[base + oc];
        }
    }

    // ============================================================
    // dHidden = grad @ W2
    //
    // grad [rows, out_channels]
    // W2   [out_channels, rank]
    //
    // dHidden [rows, rank]
    // ============================================================

    workspace.hidden_grads.resize(
        rows * rank,
        0.0,
    );

    workspace.hidden_grads.fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            rank as i32,
            out_channels as i32,
            1.0,
            grad,
            out_channels as i32,
            second_weights,
            rank as i32,
            0.0,
            &mut workspace.hidden_grads,
            rank as i32,
        );
    }

    // ============================================================
    // Backprop through FIRST activation.
    //
    // hidden contains:
    //
    // activation(first_pre + b1)
    //
    // so activation.backward(hidden[i], ...)
    // gives dL/d(first_pre).
    // ============================================================

    for i in 0..workspace.hidden_grads.len() {
        activation.backward(
            hidden[i],
            &mut workspace.hidden_grads[i],
        );
    }

    // ============================================================
    // Allocate/reset first-layer gradients.
    // ============================================================

    workspace.first_weight_grads.resize(
        rank * in_channels,
        0.0,
    );

    workspace.first_weight_grads.fill(0.0);

    workspace.first_bias_grads.resize(
        rank,
        0.0,
    );

    workspace.first_bias_grads.fill(0.0);

    // ============================================================
    // dW1 = dHidden^T @ input
    //
    // dHidden [rows, rank]
    // input   [rows, in_channels]
    //
    // dW1     [rank, in_channels]
    // ============================================================

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            rank as i32,
            in_channels as i32,
            rows as i32,
            1.0,
            &workspace.hidden_grads,
            rank as i32,
            input,
            in_channels as i32,
            0.0,
            &mut workspace.first_weight_grads,
            in_channels as i32,
        );
    }

    // ============================================================
    // db1 = sum over rows
    // ============================================================

    for row in 0..rows {
        let base =
            row * rank;

        for r in 0..rank {
            workspace.first_bias_grads[r] +=
                workspace.hidden_grads[
                    base + r
                    ];
        }
    }

    // ============================================================
    // dInput = dHidden @ W1
    //
    // dHidden [rows, rank]
    // W1      [rank, in_channels]
    //
    // dInput  [rows, in_channels]
    // ============================================================

    let mut input_grads =
        vec![
            0.0;
            rows * in_channels
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            in_channels as i32,
            rank as i32,
            1.0,
            &workspace.hidden_grads,
            rank as i32,
            first_weights,
            in_channels as i32,
            0.0,
            &mut input_grads,
            in_channels as i32,
        );
    }

    // ============================================================
    // Accumulate parameter gradients.
    // ============================================================

    crate::add_handle_grad_slices(
        first_weight_handles,
        &workspace.first_weight_grads,
    );

    crate::add_handle_grad_slices(
        first_bias_handles,
        &workspace.first_bias_grads,
    );

    crate::add_handle_grad_slices(
        second_weight_handles,
        &workspace.second_weight_grads,
    );

    crate::add_handle_grad_slices(
        second_bias_handles,
        &workspace.second_bias_grads,
    );

    input_grads
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_backward_group_avx2(
    input: &[f32],
    grad: &[f32],
    output_grad: &mut [f32],
    gamma: &[f32],
    weight_grads: &mut [f32],
    bias_grads: &mut [f32],
    mean: f32,
    inv_std: f32,
) {
    let channels =
        input.len();

    debug_assert_eq!(
        grad.len(),
        channels
    );

    debug_assert_eq!(
        output_grad.len(),
        channels
    );

    debug_assert_eq!(
        gamma.len(),
        channels
    );

    debug_assert_eq!(
        weight_grads.len(),
        channels
    );

    debug_assert_eq!(
        bias_grads.len(),
        channels
    );

    // ------------------------------------------------------------
    // First pass:
    //
    // dxhat = dy * gamma
    //
    // We need:
    //
    //   mean(dxhat)
    //   mean(dxhat * xhat)
    //
    // because:
    //
    //   dx = inv_std *
    //        (dxhat
    //         - mean(dxhat)
    //         - xhat * mean(dxhat * xhat))
    // ------------------------------------------------------------

    let mut grad_gamma_sum_0 =
        _mm256_setzero_ps();

    let mut grad_gamma_sum_1 =
        _mm256_setzero_ps();

    let mut grad_gamma_xhat_sum_0 =
        _mm256_setzero_ps();

    let mut grad_gamma_xhat_sum_1 =
        _mm256_setzero_ps();

    let mean_vec =
        _mm256_set1_ps(mean);

    let inv_std_vec =
        _mm256_set1_ps(inv_std);

    let mut c =
        0usize;

    // ------------------------------------------------------------
    // 16 channels per iteration.
    // ------------------------------------------------------------

    while c + 16 <= channels {
        let x0 =
            _mm256_loadu_ps(
                input.as_ptr().add(c)
            );

        let x1 =
            _mm256_loadu_ps(
                input.as_ptr().add(c + 8)
            );

        let dy0 =
            _mm256_loadu_ps(
                grad.as_ptr().add(c)
            );

        let dy1 =
            _mm256_loadu_ps(
                grad.as_ptr().add(c + 8)
            );

        let g0 =
            _mm256_loadu_ps(
                gamma.as_ptr().add(c)
            );

        let g1 =
            _mm256_loadu_ps(
                gamma.as_ptr().add(c + 8)
            );

        let xhat0 =
            _mm256_mul_ps(
                _mm256_sub_ps(
                    x0,
                    mean_vec,
                ),
                inv_std_vec,
            );

        let xhat1 =
            _mm256_mul_ps(
                _mm256_sub_ps(
                    x1,
                    mean_vec,
                ),
                inv_std_vec,
            );

        let dxhat0 =
            _mm256_mul_ps(
                dy0,
                g0,
            );

        let dxhat1 =
            _mm256_mul_ps(
                dy1,
                g1,
            );

        grad_gamma_sum_0 =
            _mm256_add_ps(
                grad_gamma_sum_0,
                dxhat0,
            );

        grad_gamma_sum_1 =
            _mm256_add_ps(
                grad_gamma_sum_1,
                dxhat1,
            );

        grad_gamma_xhat_sum_0 =
            _mm256_add_ps(
                grad_gamma_xhat_sum_0,
                _mm256_mul_ps(
                    dxhat0,
                    xhat0,
                ),
            );

        grad_gamma_xhat_sum_1 =
            _mm256_add_ps(
                grad_gamma_xhat_sum_1,
                _mm256_mul_ps(
                    dxhat1,
                    xhat1,
                ),
            );

        c += 16;
    }

    // ------------------------------------------------------------
    // Remaining complete SIMD vector.
    // ------------------------------------------------------------

    while c + 8 <= channels {
        let x =
            _mm256_loadu_ps(
                input.as_ptr().add(c)
            );

        let dy =
            _mm256_loadu_ps(
                grad.as_ptr().add(c)
            );

        let g =
            _mm256_loadu_ps(
                gamma.as_ptr().add(c)
            );

        let xhat =
            _mm256_mul_ps(
                _mm256_sub_ps(
                    x,
                    mean_vec,
                ),
                inv_std_vec,
            );

        let dxhat =
            _mm256_mul_ps(
                dy,
                g,
            );

        grad_gamma_sum_0 =
            _mm256_add_ps(
                grad_gamma_sum_0,
                dxhat,
            );

        grad_gamma_xhat_sum_0 =
            _mm256_add_ps(
                grad_gamma_xhat_sum_0,
                _mm256_mul_ps(
                    dxhat,
                    xhat,
                ),
            );

        c += 8;
    }

    // ------------------------------------------------------------
    // Reduce SIMD accumulators.
    // ------------------------------------------------------------

    let mut tmp =
        [0.0f32; 8];

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        grad_gamma_sum_0,
    );

    let mut grad_gamma_sum =
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        grad_gamma_sum_1,
    );

    grad_gamma_sum +=
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        grad_gamma_xhat_sum_0,
    );

    let mut grad_gamma_xhat_sum =
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        grad_gamma_xhat_sum_1,
    );

    grad_gamma_xhat_sum +=
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

    // ------------------------------------------------------------
    // Scalar tail.
    // ------------------------------------------------------------

    while c < channels {
        let dy =
            grad[c];

        let g =
            gamma[c];

        let xhat =
            (input[c] - mean)
                * inv_std;

        let dxhat =
            dy * g;

        grad_gamma_sum +=
            dxhat;

        grad_gamma_xhat_sum +=
            dxhat * xhat;

        c += 1;
    }

    let channels_f32 =
        channels as f32;

    let mean_dyg =
        grad_gamma_sum
            / channels_f32;

    let mean_dyg_xhat =
        grad_gamma_xhat_sum
            / channels_f32;

    let mean_dyg_vec =
        _mm256_set1_ps(
            mean_dyg
        );

    let mean_dyg_xhat_vec =
        _mm256_set1_ps(
            mean_dyg_xhat
        );

    // ------------------------------------------------------------
    // Second pass:
    //
    //   dgamma
    //   dbeta
    //   dx
    // ------------------------------------------------------------

    c = 0;

    while c + 8 <= channels {
        let x =
            _mm256_loadu_ps(
                input.as_ptr().add(c)
            );

        let dy =
            _mm256_loadu_ps(
                grad.as_ptr().add(c)
            );

        let g =
            _mm256_loadu_ps(
                gamma.as_ptr().add(c)
            );

        let xhat =
            _mm256_mul_ps(
                _mm256_sub_ps(
                    x,
                    mean_vec,
                ),
                inv_std_vec,
            );

        // dgamma += dy * xhat
        let dgamma =
            _mm256_mul_ps(
                dy,
                xhat,
            );

        let old_dgamma =
            _mm256_loadu_ps(
                weight_grads
                    .as_ptr()
                    .add(c)
            );

        _mm256_storeu_ps(
            weight_grads
                .as_mut_ptr()
                .add(c),
            _mm256_add_ps(
                old_dgamma,
                dgamma,
            ),
        );

        // dbeta += dy
        let old_dbeta =
            _mm256_loadu_ps(
                bias_grads
                    .as_ptr()
                    .add(c)
            );

        _mm256_storeu_ps(
            bias_grads
                .as_mut_ptr()
                .add(c),
            _mm256_add_ps(
                old_dbeta,
                dy,
            ),
        );

        // dxhat = dy * gamma
        let dxhat =
            _mm256_mul_ps(
                dy,
                g,
            );

        // dx =
        //     inv_std *
        //     (
        //         dxhat
        //         - mean(dxhat)
        //         - xhat * mean(dxhat * xhat)
        //     )
        let centered_grad =
            _mm256_sub_ps(
                _mm256_sub_ps(
                    dxhat,
                    mean_dyg_vec,
                ),
                _mm256_mul_ps(
                    xhat,
                    mean_dyg_xhat_vec,
                ),
            );

        let dx =
            _mm256_mul_ps(
                inv_std_vec,
                centered_grad,
            );

        _mm256_storeu_ps(
            output_grad
                .as_mut_ptr()
                .add(c),
            dx,
        );

        c += 8;
    }

    // ------------------------------------------------------------
    // Scalar tail.
    // ------------------------------------------------------------

    while c < channels {
        let x =
            input[c];

        let dy =
            grad[c];

        let g =
            gamma[c];

        let xhat =
            (x - mean)
                * inv_std;

        // dgamma
        weight_grads[c] +=
            dy * xhat;

        // dbeta
        bias_grads[c] +=
            dy;

        // dxhat
        let dxhat =
            dy * g;

        // dx
        output_grad[c] =
            inv_std
                * (
                dxhat
                    - mean_dyg
                    - xhat
                    * mean_dyg_xhat
            );

        c += 1;
    }
}

pub fn layer_norm_backward(
    input: &[f32],
    grad: &[f32],
    batch_size: usize,
    channels: usize,
    means: &[f32],
    inv_stds: &[f32],
    gamma_handles: &[TensorHandle],
    beta_handles: &[TensorHandle],
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    debug_assert!(
        batch_size > 0
    );

    debug_assert!(
        channels > 0
    );

    debug_assert_eq!(
        input.len(),
        grad.len()
    );

    debug_assert_eq!(
        input.len() % batch_size,
        0
    );

    debug_assert_eq!(
        gamma_handles.len(),
        channels
    );

    debug_assert_eq!(
        beta_handles.len(),
        channels
    );

    let sequence_size =
        input.len() / batch_size;

    debug_assert_eq!(
        sequence_size % channels,
        0
    );

    let positions =
        sequence_size / channels;

    let group_count =
        batch_size * positions;

    debug_assert_eq!(
        means.len(),
        group_count
    );

    debug_assert_eq!(
        inv_stds.len(),
        group_count
    );

    workspace.weights.resize(
        channels,
        0.0,
    );

    crate::handle_data_slice(
        gamma_handles,
        &mut workspace.weights,
    );

    workspace.weight_grads.resize(
        channels,
        0.0,
    );

    workspace.weight_grads.fill(
        0.0
    );

    workspace.bias_grads.resize(
        channels,
        0.0,
    );

    workspace.bias_grads.fill(
        0.0
    );

    let gamma =
        &workspace.weights;

    let mut input_grads =
        vec![0.0f32; input.len()];

    for group in 0..group_count {
        let base =
            group * channels;

        let input_group =
            &input[
                base..base + channels
                ];

        let grad_group =
            &grad[
                base..base + channels
                ];

        let output_grad_group =
            &mut input_grads[
                base..base + channels
                ];

        let mean =
            means[group];

        let inv_std =
            inv_stds[group];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    layer_norm_backward_group_avx2(
                        input_group,
                        grad_group,
                        output_grad_group,
                        gamma,
                        &mut workspace.weight_grads,
                        &mut workspace.bias_grads,
                        mean,
                        inv_std,
                    );
                }

                continue;
            }
        }

        // --------------------------------------------------------
        // Scalar fallback.
        //
        // dxhat = dy * gamma
        //
        // dx =
        //     inv_std *
        //     (
        //         dxhat
        //         - mean(dxhat)
        //         - xhat * mean(dxhat * xhat)
        //     )
        // --------------------------------------------------------

        let channels_f32 =
            channels as f32;

        let mut grad_gamma_sum =
            0.0f32;

        let mut grad_gamma_xhat_sum =
            0.0f32;

        for c in 0..channels {
            let x =
                input_group[c];

            let dy =
                grad_group[c];

            let xhat =
                (x - mean)
                    * inv_std;

            let dxhat =
                dy * gamma[c];

            grad_gamma_sum +=
                dxhat;

            grad_gamma_xhat_sum +=
                dxhat * xhat;
        }

        let mean_dyg =
            grad_gamma_sum
                / channels_f32;

        let mean_dyg_xhat =
            grad_gamma_xhat_sum
                / channels_f32;

        for c in 0..channels {
            let x =
                input_group[c];

            let dy =
                grad_group[c];

            let xhat =
                (x - mean)
                    * inv_std;

            // dgamma
            workspace.weight_grads[c] +=
                dy * xhat;

            // dbeta
            workspace.bias_grads[c] +=
                dy;

            // dxhat
            let dxhat =
                dy * gamma[c];

            // dx
            output_grad_group[c] =
                inv_std
                    * (
                    dxhat
                        - mean_dyg
                        - xhat
                        * mean_dyg_xhat
                );
        }
    }

    crate::add_handle_grad_slices_2(
        gamma_handles,
        &workspace.weight_grads,
        beta_handles,
        &workspace.bias_grads,
    );

    input_grads
}

pub fn weight_tying_backward(
    input: &[f32],
    grad: &[f32],
    batch_size: usize,
    embedding_dim: usize,
    vocab_size: usize,
    embeddings: &crate::embeddings::Embeddings,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        batch_size * embedding_dim,
        "Invalid WeightTying input size"
    );

    assert_eq!(
        grad.len(),
        batch_size * vocab_size,
        "Invalid WeightTying gradient size"
    );

    let weights =
        embeddings.flat_values();

    debug_assert_eq!(
        weights.len(),
        vocab_size * embedding_dim
    );

    workspace.embedding_grads.resize(
        vocab_size * embedding_dim,
        0.0,
    );
    workspace.embedding_grads.fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            vocab_size as i32,
            embedding_dim as i32,
            batch_size as i32,
            1.0,
            grad,
            vocab_size as i32,
            input,
            embedding_dim as i32,
            0.0,
            &mut workspace.embedding_grads,
            embedding_dim as i32,
        );
    }

    embeddings.accumulate_flat_grads(
        &workspace.embedding_grads
    );

    let mut input_grads =
        vec![
            0.0;
            batch_size * embedding_dim
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            batch_size as i32,
            embedding_dim as i32,
            vocab_size as i32,
            1.0,
            grad,
            vocab_size as i32,
            &weights,
            embedding_dim as i32,
            0.0,
            &mut input_grads,
            embedding_dim as i32,
        );
    }

    input_grads
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn global_mixer_read_softmax_backward_avx2(
    probs: &[f32],
    score_grads: &mut [f32],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
) { unsafe {
    let rows =
        batch_size * positions;

    for row in 0..rows {
        let base =
            row * global_dim;

        // sum(probs * dR)
        let mut dot0 =
            _mm256_setzero_ps();

        let mut dot1 =
            _mm256_setzero_ps();

        let mut g = 0usize;

        while g + 16 <= global_dim {
            let p0 =
                _mm256_loadu_ps(
                    probs.as_ptr()
                        .add(base + g),
                );

            let dg0 =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g),
                );

            let p1 =
                _mm256_loadu_ps(
                    probs.as_ptr()
                        .add(base + g + 8),
                );

            let dg1 =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g + 8),
                );

            dot0 =
                _mm256_fmadd_ps(
                    p0,
                    dg0,
                    dot0,
                );

            dot1 =
                _mm256_fmadd_ps(
                    p1,
                    dg1,
                    dot1,
                );

            g += 16;
        }

        while g + 8 <= global_dim {
            let p =
                _mm256_loadu_ps(
                    probs.as_ptr()
                        .add(base + g),
                );

            let dg =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g),
                );

            dot0 =
                _mm256_fmadd_ps(
                    p,
                    dg,
                    dot0,
                );

            g += 8;
        }

        let mut tmp =
            [0.0f32; 8];

        _mm256_storeu_ps(
            tmp.as_mut_ptr(),
            dot0,
        );

        let mut dot =
            tmp[0]
                + tmp[1]
                + tmp[2]
                + tmp[3]
                + tmp[4]
                + tmp[5]
                + tmp[6]
                + tmp[7];

        _mm256_storeu_ps(
            tmp.as_mut_ptr(),
            dot1,
        );

        dot +=
            tmp[0]
                + tmp[1]
                + tmp[2]
                + tmp[3]
                + tmp[4]
                + tmp[5]
                + tmp[6]
                + tmp[7];

        while g < global_dim {
            dot +=
                probs[base + g]
                    * score_grads[base + g];

            g += 1;
        }

        let dot_v =
            _mm256_set1_ps(dot);

        g = 0;

        while g + 8 <= global_dim {
            let p =
                _mm256_loadu_ps(
                    probs.as_ptr()
                        .add(base + g),
                );

            let dg =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g),
                );

            let result =
                _mm256_mul_ps(
                    p,
                    _mm256_sub_ps(
                        dg,
                        dot_v,
                    ),
                );

            _mm256_storeu_ps(
                score_grads.as_mut_ptr()
                    .add(base + g),
                result,
            );

            g += 8;
        }

        while g < global_dim {
            score_grads[base + g] =
                probs[base + g]
                    * (
                    score_grads[base + g]
                        - dot
                );

            g += 1;
        }
    }
}}

#[inline]
fn global_mixer_read_softmax_backward_scalar(
    probs: &[f32],
    score_grads: &mut [f32],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
) {
    let rows =
        batch_size * positions;

    for row in 0..rows {
        let base =
            row * global_dim;

        let mut dot =
            0.0f32;

        for g in 0..global_dim {
            dot +=
                probs[base + g]
                    * score_grads[base + g];
        }

        for g in 0..global_dim {
            score_grads[base + g] =
                probs[base + g]
                    * (
                    score_grads[base + g]
                        - dot
                );
        }
    }
}


// ============================================================
// GlobalMixer backward
// ============================================================

pub fn global_mixer_backward(
    input: &[f32],
    write_probs: &[f32],
    global_vectors: &[f32],
    read_probs: &[f32],
    mut grad: Vec<f32>,
    batch_size: usize,
    positions: usize,
    channels: usize,
    global_dim: usize,
    write_weight_handles: &[TensorHandle],
    write_bias_handles: &[TensorHandle],
    read_weight_handles: &[TensorHandle],
    read_bias_handles: &[TensorHandle],
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    let rows =
        batch_size * positions;

    debug_assert_eq!(
        input.len(),
        rows * channels,
    );

    debug_assert_eq!(
        grad.len(),
        rows * channels,
    );

    debug_assert_eq!(
        write_probs.len(),
        rows * global_dim,
    );

    debug_assert_eq!(
        read_probs.len(),
        rows * global_dim,
    );

    debug_assert_eq!(
        global_vectors.len(),
        batch_size *
            global_dim *
            channels,
    );

    debug_assert_eq!(
        write_weight_handles.len(),
        global_dim * channels,
    );

    debug_assert_eq!(
        read_weight_handles.len(),
        global_dim * channels,
    );

    debug_assert_eq!(
        write_bias_handles.len(),
        global_dim,
    );

    debug_assert_eq!(
        read_bias_handles.len(),
        global_dim,
    );

    #[cfg(target_arch = "x86_64")]
    let use_avx2 =
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma");

    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    // ------------------------------------------------------------
    // Important:
    //
    // grad initially contains dY.
    //
    // Since:
    //
    //     Y = X + Message
    //
    // it already contains the residual dX.
    //
    // We therefore accumulate every mixer contribution directly
    // into grad and avoid making another full-size copy.
    //
    // Also, dGlobal and dR MUST be computed before grad is mutated
    // by dX_read/dX_write.
    // ------------------------------------------------------------

    // ============================================================
    // 1. dGlobal = R^T dY
    //
    // dGlobal[b,g,c] =
    //     sum_p R[b,p,g] * dY[b,p,c]
    //
    // Small per-batch matrix -> SIMD/direct loops.
    // ============================================================

    workspace.mixer_global_grads.resize(
        batch_size *
            global_dim *
            channels,
        0.0,
    );

    workspace
        .mixer_global_grads
        .fill(0.0);

    for b in 0..batch_size {
        let grad_base =
            b * positions * channels;

        let probs_base =
            b * positions * global_dim;

        let global_base =
            b * global_dim * channels;

        for g in 0..global_dim {
            let dst_base =
                global_base +
                    g * channels;

            let mut c = 0usize;

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    while c + 8 <= channels {
                        let mut acc0 =
                            _mm256_setzero_ps();

                        let mut acc1 =
                            _mm256_setzero_ps();

                        let mut p = 0usize;

                        while p + 1 < positions {
                            let r0 =
                                _mm256_set1_ps(
                                    read_probs[
                                        probs_base
                                            + p * global_dim
                                            + g
                                        ],
                                );

                            let r1 =
                                _mm256_set1_ps(
                                    read_probs[
                                        probs_base
                                            + (p + 1)
                                            * global_dim
                                            + g
                                        ],
                                );

                            let dy0 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_base
                                                + p * channels
                                                + c,
                                        ),
                                );

                            let dy1 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_base
                                                + (p + 1)
                                                * channels
                                                + c,
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    dy0,
                                    r0,
                                    acc0,
                                );

                            acc1 =
                                _mm256_fmadd_ps(
                                    dy1,
                                    r1,
                                    acc1,
                                );

                            p += 2;
                        }

                        while p < positions {
                            let r =
                                _mm256_set1_ps(
                                    read_probs[
                                        probs_base
                                            + p * global_dim
                                            + g
                                        ],
                                );

                            let dy =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_base
                                                + p * channels
                                                + c,
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    dy,
                                    r,
                                    acc0,
                                );

                            p += 1;
                        }

                        _mm256_storeu_ps(
                            workspace
                                .mixer_global_grads
                                .as_mut_ptr()
                                .add(
                                    dst_base + c
                                ),
                            _mm256_add_ps(
                                acc0,
                                acc1,
                            ),
                        );

                        c += 8;
                    }
                }
            }

            while c < channels {
                let mut sum =
                    0.0f32;

                for p in 0..positions {
                    sum +=
                        read_probs[
                            probs_base
                                + p * global_dim
                                + g
                            ]
                            * grad[
                            grad_base
                                + p * channels
                                + c
                            ];
                }

                workspace
                    .mixer_global_grads[
                    dst_base + c
                    ] = sum;

                c += 1;
            }
        }
    }

    // ============================================================
    // 2. dR = dY * Global^T
    //
    // dR[b,p,g] =
    //     sum_c dY[b,p,c] * Global[b,g,c]
    //
    // Small dot products -> AVX2.
    // ============================================================

    workspace.mixer_score_grads.resize(
        rows * global_dim,
        0.0,
    );

    workspace
        .mixer_score_grads
        .fill(0.0);

    for b in 0..batch_size {
        let grad_base =
            b * positions * channels;

        let global_base =
            b * global_dim * channels;

        let score_base =
            b * positions * global_dim;

        for p in 0..positions {
            let grad_row =
                grad_base +
                    p * channels;

            let score_row =
                score_base +
                    p * global_dim;

            for g in 0..global_dim {
                let global_row =
                    global_base +
                        g * channels;

                let mut sum =
                    0.0f32;

                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        let mut acc0 =
                            _mm256_setzero_ps();

                        let mut acc1 =
                            _mm256_setzero_ps();

                        let mut c = 0usize;

                        while c + 16 <= channels {
                            let dy0 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_row + c
                                        ),
                                );

                            let gv0 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        ),
                                );

                            let dy1 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_row + c + 8
                                        ),
                                );

                            let gv1 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_row + c + 8
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    dy0,
                                    gv0,
                                    acc0,
                                );

                            acc1 =
                                _mm256_fmadd_ps(
                                    dy1,
                                    gv1,
                                    acc1,
                                );

                            c += 16;
                        }

                        while c + 8 <= channels {
                            let dy =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_row + c
                                        ),
                                );

                            let gv =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    dy,
                                    gv,
                                    acc0,
                                );

                            c += 8;
                        }

                        let mut tmp =
                            [0.0f32; 8];

                        _mm256_storeu_ps(
                            tmp.as_mut_ptr(),
                            acc0,
                        );

                        sum =
                            tmp[0]
                                + tmp[1]
                                + tmp[2]
                                + tmp[3]
                                + tmp[4]
                                + tmp[5]
                                + tmp[6]
                                + tmp[7];

                        _mm256_storeu_ps(
                            tmp.as_mut_ptr(),
                            acc1,
                        );

                        sum +=
                            tmp[0]
                                + tmp[1]
                                + tmp[2]
                                + tmp[3]
                                + tmp[4]
                                + tmp[5]
                                + tmp[6]
                                + tmp[7];

                        while c < channels {
                            sum +=
                                grad[
                                    grad_row + c
                                    ]
                                    * global_vectors[
                                    global_row + c
                                    ];

                            c += 1;
                        }
                    }
                } else {
                    for c in 0..channels {
                        sum +=
                            grad[
                                grad_row + c
                                ]
                                * global_vectors[
                                global_row + c
                                ];
                    }
                }

                workspace
                    .mixer_score_grads[
                    score_row + g
                    ] = sum;
            }
        }
    }

    // ============================================================
    // 3. Read softmax backward
    //
    // R = softmax_global(read_scores)
    // ============================================================

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            global_mixer_read_softmax_backward_avx2(
                read_probs,
                &mut workspace.mixer_score_grads,
                batch_size,
                positions,
                global_dim,
            );
        }
    } else {
        global_mixer_read_softmax_backward_scalar(
            read_probs,
            &mut workspace.mixer_score_grads,
            batch_size,
            positions,
            global_dim,
        );
    }

    // ============================================================
    // 4. dW_read = dS_read^T X
    //
    //     [G,R] * [R,C] -> [G,C]
    //
    // W_read is stored as [G,C].
    //
    // THIS is the corrected orientation that fixes the old bug.
    // ============================================================

    workspace.weight_grads.resize(
        global_dim * channels,
        0.0,
    );

    workspace
        .weight_grads
        .fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            global_dim as i32,
            channels as i32,
            rows as i32,
            1.0,
            &workspace.mixer_score_grads,
            global_dim as i32,
            input,
            channels as i32,
            0.0,
            &mut workspace.weight_grads,
            channels as i32,
        );
    }

    // ============================================================
    // 5. db_read
    // ============================================================

    workspace.bias_grads.resize(
        global_dim,
        0.0,
    );

    workspace
        .bias_grads
        .fill(0.0);

    for row in 0..rows {
        let base =
            row * global_dim;

        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let mut g = 0usize;

                while g + 8 <= global_dim {
                    let old =
                        _mm256_loadu_ps(
                            workspace
                                .bias_grads
                                .as_ptr()
                                .add(g),
                        );

                    let value =
                        _mm256_loadu_ps(
                            workspace
                                .mixer_score_grads
                                .as_ptr()
                                .add(base + g),
                        );

                    _mm256_storeu_ps(
                        workspace
                            .bias_grads
                            .as_mut_ptr()
                            .add(g),
                        _mm256_add_ps(
                            old,
                            value,
                        ),
                    );

                    g += 8;
                }

                while g < global_dim {
                    workspace.bias_grads[g] +=
                        workspace.mixer_score_grads[
                            base + g
                            ];

                    g += 1;
                }
            }
        } else {
            for g in 0..global_dim {
                workspace.bias_grads[g] +=
                    workspace.mixer_score_grads[
                        base + g
                        ];
            }
        }
    }

    crate::add_handle_grad_slices_2(
        read_weight_handles,
        &workspace.weight_grads,
        read_bias_handles,
        &workspace.bias_grads,
    );

    // ============================================================
    // 6. dX_read = dS_read W_read
    //
    //     [R,G] * [G,C] -> [R,C]
    //
    // Accumulate into grad because grad already contains dX_residual.
    // ============================================================

    workspace.weights.resize(
        global_dim * channels,
        0.0,
    );

    crate::handle_data_slice(
        read_weight_handles,
        &mut workspace.weights,
    );

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            channels as i32,
            global_dim as i32,
            1.0,
            &workspace.mixer_score_grads,
            global_dim as i32,
            &workspace.weights,
            channels as i32,
            1.0,
            &mut grad,
            channels as i32,
        );
    }

    // ============================================================
    // 7. dA = X dGlobal^T
    //
    // dA[b,p,g] =
    //     sum_c X[b,p,c] * dGlobal[b,g,c]
    //
    // Small dot products -> AVX2.
    //
    // We can now safely use input_grads/grad because dGlobal and dR
    // have already consumed the original dY.
    // ============================================================

    for b in 0..batch_size {
        let input_base =
            b * positions * channels;

        let global_base =
            b * global_dim * channels;

        let score_base =
            b * positions * global_dim;

        for p in 0..positions {
            let input_row =
                input_base +
                    p * channels;

            let score_row =
                score_base +
                    p * global_dim;

            for g in 0..global_dim {
                let global_row =
                    global_base +
                        g * channels;

                let mut sum =
                    0.0f32;

                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        let mut acc0 =
                            _mm256_setzero_ps();

                        let mut acc1 =
                            _mm256_setzero_ps();

                        let mut c = 0usize;

                        while c + 16 <= channels {
                            let x0 =
                                _mm256_loadu_ps(
                                    input.as_ptr()
                                        .add(
                                            input_row + c
                                        ),
                                );

                            let dg0 =
                                _mm256_loadu_ps(
                                    workspace
                                        .mixer_global_grads
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        ),
                                );

                            let x1 =
                                _mm256_loadu_ps(
                                    input.as_ptr()
                                        .add(
                                            input_row + c + 8
                                        ),
                                );

                            let dg1 =
                                _mm256_loadu_ps(
                                    workspace
                                        .mixer_global_grads
                                        .as_ptr()
                                        .add(
                                            global_row + c + 8
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    x0,
                                    dg0,
                                    acc0,
                                );

                            acc1 =
                                _mm256_fmadd_ps(
                                    x1,
                                    dg1,
                                    acc1,
                                );

                            c += 16;
                        }

                        while c + 8 <= channels {
                            let x =
                                _mm256_loadu_ps(
                                    input.as_ptr()
                                        .add(
                                            input_row + c
                                        ),
                                );

                            let dg =
                                _mm256_loadu_ps(
                                    workspace
                                        .mixer_global_grads
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    x,
                                    dg,
                                    acc0,
                                );

                            c += 8;
                        }

                        let mut tmp =
                            [0.0f32; 8];

                        _mm256_storeu_ps(
                            tmp.as_mut_ptr(),
                            acc0,
                        );

                        sum =
                            tmp[0]
                                + tmp[1]
                                + tmp[2]
                                + tmp[3]
                                + tmp[4]
                                + tmp[5]
                                + tmp[6]
                                + tmp[7];

                        _mm256_storeu_ps(
                            tmp.as_mut_ptr(),
                            acc1,
                        );

                        sum +=
                            tmp[0]
                                + tmp[1]
                                + tmp[2]
                                + tmp[3]
                                + tmp[4]
                                + tmp[5]
                                + tmp[6]
                                + tmp[7];

                        while c < channels {
                            sum +=
                                input[
                                    input_row + c
                                    ]
                                    * workspace
                                    .mixer_global_grads[
                                    global_row + c
                                    ];

                            c += 1;
                        }
                    }
                } else {
                    for c in 0..channels {
                        sum +=
                            input[
                                input_row + c
                                ]
                                * workspace
                                .mixer_global_grads[
                                global_row + c
                                ];
                    }
                }

                workspace.mixer_score_grads[
                    score_row + g
                    ] = sum;
            }
        }
    }

    // ============================================================
    // 8. Write softmax backward
    //
    // A = softmax_positions(write_scores)
    //
    // The reduction is over positions for each g, which is
    // strided in memory. Keep it scalar rather than introducing
    // a transpose just for SIMD.
    // ============================================================

    for b in 0..batch_size {
        let score_base =
            b * positions * global_dim;

        for g in 0..global_dim {
            let mut dot =
                0.0f32;

            for p in 0..positions {
                let idx =
                    score_base +
                        p * global_dim +
                        g;

                dot +=
                    write_probs[idx]
                        * workspace
                        .mixer_score_grads[idx];
            }

            for p in 0..positions {
                let idx =
                    score_base +
                        p * global_dim +
                        g;

                workspace.mixer_score_grads[idx] =
                    write_probs[idx]
                        * (
                        workspace
                            .mixer_score_grads[idx]
                            - dot
                    );
            }
        }
    }

    // ============================================================
    // 9. dW_write = dS_write^T X
    //
    //     [G,R] * [R,C] -> [G,C]
    //
    // Correct [G,C] output layout.
    // ============================================================

    workspace.weight_grads.fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            global_dim as i32,
            channels as i32,
            rows as i32,
            1.0,
            &workspace.mixer_score_grads,
            global_dim as i32,
            input,
            channels as i32,
            0.0,
            &mut workspace.weight_grads,
            channels as i32,
        );
    }

    // ============================================================
    // 10. db_write
    // ============================================================

    workspace.bias_grads.fill(0.0);

    for row in 0..rows {
        let base =
            row * global_dim;

        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let mut g = 0usize;

                while g + 8 <= global_dim {
                    let old =
                        _mm256_loadu_ps(
                            workspace
                                .bias_grads
                                .as_ptr()
                                .add(g),
                        );

                    let value =
                        _mm256_loadu_ps(
                            workspace
                                .mixer_score_grads
                                .as_ptr()
                                .add(base + g),
                        );

                    _mm256_storeu_ps(
                        workspace
                            .bias_grads
                            .as_mut_ptr()
                            .add(g),
                        _mm256_add_ps(
                            old,
                            value,
                        ),
                    );

                    g += 8;
                }

                while g < global_dim {
                    workspace.bias_grads[g] +=
                        workspace.mixer_score_grads[
                            base + g
                            ];

                    g += 1;
                }
            }
        } else {
            for g in 0..global_dim {
                workspace.bias_grads[g] +=
                    workspace.mixer_score_grads[
                        base + g
                        ];
            }
        }
    }

    crate::add_handle_grad_slices_2(
        write_weight_handles,
        &workspace.weight_grads,
        write_bias_handles,
        &workspace.bias_grads,
    );

    // ============================================================
    // 11. Load W_write
    // ============================================================

    crate::handle_data_slice(
        write_weight_handles,
        &mut workspace.weights,
    );

    // ============================================================
    // 12. dX_write = dS_write W_write
    //
    //     [R,G] * [G,C] -> [R,C]
    //
    // Accumulate into existing grad.
    // ============================================================

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            channels as i32,
            global_dim as i32,
            1.0,
            &workspace.mixer_score_grads,
            global_dim as i32,
            &workspace.weights,
            channels as i32,
            1.0,
            &mut grad,
            channels as i32,
        );
    }

    // ============================================================
    // 13. dX_global = A dGlobal
    //
    // For each batch:
    //
    //     [P,G] * [G,C] -> [P,C]
    //
    // This is essentially a small matrix-vector accumulation with
    // very favorable contiguous channel accesses, so use the old
    // handwritten AVX2 approach rather than launching another
    // collection of tiny GEMMs.
    // ============================================================

    for b in 0..batch_size {
        let grad_base =
            b * positions * channels;

        let probs_base =
            b * positions * global_dim;

        let global_base =
            b * global_dim * channels;

        for p in 0..positions {
            let dst_base =
                grad_base +
                    p * channels;

            let probs_row =
                probs_base +
                    p * global_dim;

            let mut c = 0usize;

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    while c + 8 <= channels {
                        let mut acc0 =
                            _mm256_setzero_ps();

                        let mut acc1 =
                            _mm256_setzero_ps();

                        let mut g = 0usize;

                        while g + 1 < global_dim {
                            let a0 =
                                _mm256_set1_ps(
                                    write_probs[
                                        probs_row + g
                                        ],
                                );

                            let a1 =
                                _mm256_set1_ps(
                                    write_probs[
                                        probs_row + g + 1
                                        ],
                                );

                            let gv0 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_base
                                                + g * channels
                                                + c,
                                        ),
                                );

                            let gv1 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_base
                                                + (g + 1)
                                                * channels
                                                + c,
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    gv0,
                                    a0,
                                    acc0,
                                );

                            acc1 =
                                _mm256_fmadd_ps(
                                    gv1,
                                    a1,
                                    acc1,
                                );

                            g += 2;
                        }

                        while g < global_dim {
                            let a =
                                _mm256_set1_ps(
                                    write_probs[
                                        probs_row + g
                                        ],
                                );

                            let gv =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_base
                                                + g * channels
                                                + c,
                                        ),
                                );

                            acc0 =
                                _mm256_fmadd_ps(
                                    gv,
                                    a,
                                    acc0,
                                );

                            g += 1;
                        }

                        let old =
                            _mm256_loadu_ps(
                                grad.as_ptr()
                                    .add(
                                        dst_base + c
                                    ),
                            );

                        _mm256_storeu_ps(
                            grad.as_mut_ptr()
                                .add(
                                    dst_base + c
                                ),
                            _mm256_add_ps(
                                old,
                                _mm256_add_ps(
                                    acc0,
                                    acc1,
                                ),
                            ),
                        );

                        c += 8;
                    }
                }
            }

            while c < channels {
                let mut sum =
                    0.0f32;

                for g in 0..global_dim {
                    sum +=
                        write_probs[
                            probs_row + g
                            ]
                            * global_vectors[
                            global_base
                                + g * channels
                                + c
                            ];
                }

                grad[dst_base + c] +=
                    sum;

                c += 1;
            }
        }
    }

    grad
}

pub fn backward_layers_batch(
    layers: &[BatchLayerCache],
    grad: Vec<f32>,
    batch_size: usize,
) -> Vec<f32> {
    BACKWARD_WORKSPACE.with(|cell| {
        let mut workspace =
            cell.borrow_mut();

        backward_layers_batch_inner(
            layers,
            grad,
            batch_size,
            &mut workspace,
        )
    })
}

fn backward_layers_batch_inner(
    layers: &[BatchLayerCache],
    mut grad: Vec<f32>,
    batch_size: usize,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    for layer in layers.iter().rev() {
        grad = backward_layer_batch(
            layer,
            grad,
            batch_size,
            workspace,
        );
    }

    grad
}

fn backward_layer_batch(
    layer: &BatchLayerCache,
    mut grad: Vec<f32>,
    batch_size: usize,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    match layer {
        BatchLayerCache::Dense {
            input_size,
            output_size,
            input,
            activation_output,
            weights,
            weight_handles,
            bias_handles,
            activation,
        } => {
            if let Some(output) =
                activation_output
            {
                for b in 0..batch_size {
                    let base =
                        b * *output_size;

                    for o in 0..*output_size {
                        activation.backward(
                            output[base + o],
                            &mut grad[base + o],
                        );
                    }
                }
            }

            workspace.weight_grads.resize(
                output_size * input_size,
                0.0,
            );
            workspace.weight_grads.fill(0.0);

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    *output_size as i32,
                    *input_size as i32,
                    batch_size as i32,
                    1.0,
                    &grad,
                    *output_size as i32,
                    input,
                    *input_size as i32,
                    0.0,
                    &mut workspace.weight_grads,
                    *input_size as i32,
                );
            }

            workspace.bias_grads.resize(
                *output_size,
                0.0,
            );
            workspace.bias_grads.fill(0.0);

            for b in 0..batch_size {
                let base =
                    b * *output_size;

                for o in 0..*output_size {
                    workspace.bias_grads[o] +=
                        grad[base + o];
                }
            }

            let mut input_grads =
                vec![
                    0.0;
                    batch_size * *input_size
                ];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    batch_size as i32,
                    *input_size as i32,
                    *output_size as i32,
                    1.0,
                    &grad,
                    *output_size as i32,
                    weights,
                    *input_size as i32,
                    0.0,
                    &mut input_grads,
                    *input_size as i32,
                );
            }

            crate::add_handle_grad_slices(
                weight_handles,
                &workspace.weight_grads,
            );

            crate::add_handle_grad_slices(
                bias_handles,
                &workspace.bias_grads,
            );

            input_grads
        }

        BatchLayerCache::Conv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::Residual {
            inner,
        } => {
            let skip_grad = grad.clone();

            let mut result =
                backward_layers_batch_inner(
                    inner,
                    grad,
                    batch_size,
                    workspace,
                );

            debug_assert_eq!(
                result.len(),
                skip_grad.len()
            );

            for i in 0..result.len() {
                result[i] +=
                    skip_grad[i];
            }

            result
        }

        BatchLayerCache::DepthwiseConv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            depthwise_conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::GroupedConv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            out_channels,
            groups,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            grouped_conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *out_channels,
                *groups,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
            )
        }

        BatchLayerCache::LowRankPointwise {
            input,
            hidden,
            output,
            rows,
            in_channels,
            rank,
            out_channels,
            first_weights,
            second_weights,
            first_weight_handles,
            first_bias_handles,
            second_weight_handles,
            second_bias_handles,
            activation,
        } => {
            low_rank_pointwise_backward(
                input,
                hidden,
                output,
                &mut grad,
                *rows,
                *in_channels,
                *rank,
                *out_channels,
                first_weights,
                second_weights,
                first_weight_handles,
                first_bias_handles,
                second_weight_handles,
                second_bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::ChannelScale {
            input,
            channels,
            scale_handles,
            bias_handles,
        } => {
            channel_scale_backward(
                input,
                &mut grad,
                batch_size,
                *channels,
                scale_handles,
                bias_handles,
                workspace,
            )
        }

        BatchLayerCache::LayerNorm {
            input,
            means,
            inv_stds,
            channels,
            gamma_handles,
            beta_handles,
        } => {
            layer_norm_backward(
                input,
                &grad,
                batch_size,
                *channels,
                means,
                inv_stds,
                gamma_handles,
                beta_handles,
                workspace,
            )
        }

        BatchLayerCache::WeightTying {
            input,
            embeddings,
            batch_size,
            embedding_dim,
            vocab_size,
        } => {
            weight_tying_backward(
                input,
                &grad,
                *batch_size,
                *embedding_dim,
                *vocab_size,
                embeddings,
                workspace,
            )
        }

        BatchLayerCache::GlobalMixer {
            input,
            write_probs,
            global_vectors,
            read_probs,
            positions,
            channels,
            global_dim,
            write_weight_handles,
            write_bias_handles,
            read_weight_handles,
            read_bias_handles,
        } => {
            global_mixer_backward(
                input,
                write_probs,
                global_vectors,
                read_probs,
                grad,
                batch_size,
                *positions,
                *channels,
                *global_dim,
                write_weight_handles,
                write_bias_handles,
                read_weight_handles,
                read_bias_handles,
                workspace,
            )
        }
    }
}
