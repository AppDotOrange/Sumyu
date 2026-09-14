use crate::embeddings::Embeddings;
use crate::neuron::{
    Activation,
    BatchLayerCache,
    Layer,
};
use crate::parameters::ParameterStore;

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

use crate::forwards::{
    global_mixer_position_features,
    GLOBAL_MIXER_POS_FEATURES,
};


// ============================================================================
// Shared backward workspace
// ============================================================================

pub struct BackwardWorkspace {
    pub(crate) weights: Vec<f32>,
    positions: Vec<i32>,

    weight_grads: Vec<f32>,
    bias_grads: Vec<f32>,

    // Retained as reusable scratch for compatibility with existing code.
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

    mixer_pos_grads_write: Vec<f32>,
    mixer_pos_grads_read: Vec<f32>,
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

            mixer_pos_grads_write: Vec::new(),
            mixer_pos_grads_read: Vec::new(),
        }
    }
}

thread_local! {
    static BACKWARD_WORKSPACE: RefCell<BackwardWorkspace> =
        RefCell::new(BackwardWorkspace::new());
}


// ============================================================================
// Small helpers
// ============================================================================

#[inline]
fn accumulate_parameter_grads(
    params: &mut ParameterStore,
    range: crate::parameters::ParamRange,
    local_grads: &[f32],
) {
    debug_assert_eq!(range.len, local_grads.len());

    add_f32_slice_simd(
        params.grads_mut(range),
        local_grads,
    );
}


// ============================================================================
// Convolution position helper
// ============================================================================

#[inline]
pub fn make_conv_positions(
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
) -> Vec<i32> {
    let total = output_length * kernel_size;

    let mut positions = Vec::with_capacity(total);

    for out_pos in 0..output_length {
        let base = out_pos * stride;

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


// ============================================================================
// SIMD helpers
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_f32_slice_avx2(
    dst: &mut [f32],
    src: &[f32],
) {
    debug_assert_eq!(dst.len(), src.len());

    let len = dst.len();
    let mut i = 0usize;

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
}

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
) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    let len = dst.len();
    let mut i = 0usize;

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
}

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


// ============================================================================
// Conv1D backward wrapper
// ============================================================================

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
    weights: &[f32],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    let weights_len =
        out_channels
            * kernel_size
            * in_channels;

    debug_assert_eq!(
        weights.len(),
        weights_len,
    );

    let (
        input_grads,
        weight_grads,
        bias_grads,
    ) = backward_direct(
        input,
        output,
        grad,
        batch_size,
        input_length,
        output_length,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        causal,
        weights,
        activation,
    );

    workspace.weight_grads.clear();
    workspace.bias_grads.clear();

    // These are only scratch references for callers/debugging.
    workspace.weight_grads.extend_from_slice(
        &weight_grads
    );

    workspace.bias_grads.extend_from_slice(
        &bias_grads
    );

    input_grads
}


// ============================================================================
// Depthwise Conv1D backward
// ============================================================================

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
    weights: &[f32],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
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
        weights.len(),
        in_channels * kernel_size
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

    let depthwise_size =
        in_channels * kernel_size;

    workspace.weight_grads.resize(
        depthwise_size,
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        in_channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    workspace.depthwise_weights_kmajor.resize(
        depthwise_size,
        0.0,
    );

    workspace.depthwise_weight_grads_kmajor.resize(
        depthwise_size,
        0.0,
    );

    workspace.depthwise_weight_grads_kmajor.fill(
        0.0
    );

    for c in 0..in_channels {
        for k in 0..kernel_size {
            workspace.depthwise_weights_kmajor[
                k * in_channels + c
                ] =
                weights[
                    c * kernel_size + k
                    ];
        }
    }

    let mut input_grads =
        vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let input_batch_base =
            b * input_length * in_channels;

        let output_batch_base =
            b * output_length * in_channels;

        for out_pos in 0..output_length {
            let grad_base =
                output_batch_base
                    + out_pos * in_channels;

            add_f32_slice_simd(
                &mut workspace.bias_grads,
                &grad[
                    grad_base
                        ..grad_base + in_channels
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
                    &mut workspace
                        .depthwise_weight_grads_kmajor[
                        weight_base
                            ..weight_base + in_channels
                        ],
                    &input[
                        src_base
                            ..src_base + in_channels
                        ],
                    &grad[
                        grad_base
                            ..grad_base + in_channels
                        ],
                );

                mul_add_f32_slice_simd(
                    &mut input_grads[
                        src_base
                            ..src_base + in_channels
                        ],
                    &workspace
                        .depthwise_weights_kmajor[
                        weight_base
                            ..weight_base + in_channels
                        ],
                    &grad[
                        grad_base
                            ..grad_base + in_channels
                        ],
                );
            }
        }
    }

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

    (
        input_grads,
        workspace.weight_grads.clone(),
        workspace.bias_grads.clone(),
    )
}


// ============================================================================
// ChannelScale backward
// ============================================================================

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
) {
    let positions =
        input.len() / channels;

    for position in 0..positions {
        let offset =
            position * channels;

        let mut c = 0usize;

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

        while c < channels {
            let x = input[offset + c];
            let g = grad[offset + c];

            input_grads[offset + c] +=
                g * scales[c];

            scale_grads[c] +=
                g * x;

            bias_grads[c] +=
                g;

            c += 1;
        }
    }
}

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
    grad: &[f32],
    batch_size: usize,
    channels: usize,
    scales: &[f32],
    workspace: &mut BackwardWorkspace,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    debug_assert_eq!(
        input.len(),
        grad.len()
    );

    debug_assert_eq!(
        input.len() % batch_size,
        0
    );

    debug_assert_eq!(
        scales.len(),
        channels
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
        vec![0.0f32; input.len()];

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
            scales,
            channels,
        );
    }

    (
        input_grads,
        workspace.weight_grads.clone(),
        workspace.bias_grads.clone(),
    )
}


// ============================================================================
// Low-rank pointwise backward
// ============================================================================

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
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> (
    Vec<f32>, // input gradients
    Vec<f32>, // first weight gradients
    Vec<f32>, // first bias gradients
    Vec<f32>, // second weight gradients
    Vec<f32>, // second bias gradients
) {
    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    // ------------------------------------------------------------
    // Second layer gradients
    // ------------------------------------------------------------

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

    for row in 0..rows {
        let base = row * out_channels;

        for oc in 0..out_channels {
            workspace.second_bias_grads[oc] +=
                grad[base + oc];
        }
    }

    // ------------------------------------------------------------
    // dHidden = dOutput * W2
    // ------------------------------------------------------------

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

    // First activation backward.
    for i in 0..workspace.hidden_grads.len() {
        activation.backward(
            hidden[i],
            &mut workspace.hidden_grads[i],
        );
    }

    // ------------------------------------------------------------
    // First layer gradients
    // ------------------------------------------------------------

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

    for row in 0..rows {
        let base = row * rank;

        for r in 0..rank {
            workspace.first_bias_grads[r] +=
                workspace.hidden_grads[
                    base + r
                    ];
        }
    }

    // ------------------------------------------------------------
    // dInput = dHidden * W1
    // ------------------------------------------------------------

    let mut input_grads =
        vec![0.0f32; rows * in_channels];

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

    (
        input_grads,
        workspace.first_weight_grads.clone(),
        workspace.first_bias_grads.clone(),
        workspace.second_weight_grads.clone(),
        workspace.second_bias_grads.clone(),
    )
}

// ============================================================================
// LayerNorm backward
// ============================================================================

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

    debug_assert_eq!(grad.len(), channels);
    debug_assert_eq!(output_grad.len(), channels);
    debug_assert_eq!(gamma.len(), channels);

    let mut grad_sum0 =
        _mm256_setzero_ps();

    let mut grad_sum1 =
        _mm256_setzero_ps();

    let mut grad_xhat_sum0 =
        _mm256_setzero_ps();

    let mut grad_xhat_sum1 =
        _mm256_setzero_ps();

    let mean_vec =
        _mm256_set1_ps(mean);

    let inv_std_vec =
        _mm256_set1_ps(inv_std);

    let mut c = 0usize;

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

        grad_sum0 =
            _mm256_add_ps(
                grad_sum0,
                dxhat0,
            );

        grad_sum1 =
            _mm256_add_ps(
                grad_sum1,
                dxhat1,
            );

        grad_xhat_sum0 =
            _mm256_add_ps(
                grad_xhat_sum0,
                _mm256_mul_ps(
                    dxhat0,
                    xhat0,
                ),
            );

        grad_xhat_sum1 =
            _mm256_add_ps(
                grad_xhat_sum1,
                _mm256_mul_ps(
                    dxhat1,
                    xhat1,
                ),
            );

        c += 16;
    }

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

        grad_sum0 =
            _mm256_add_ps(
                grad_sum0,
                dxhat,
            );

        grad_xhat_sum0 =
            _mm256_add_ps(
                grad_xhat_sum0,
                _mm256_mul_ps(
                    dxhat,
                    xhat,
                ),
            );

        c += 8;
    }

    let mut tmp =
        [0.0f32; 8];

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        grad_sum0,
    );

    let mut grad_sum =
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
        grad_sum1,
    );

    grad_sum +=
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
        grad_xhat_sum0,
    );

    let mut grad_xhat_sum =
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
        grad_xhat_sum1,
    );

    grad_xhat_sum +=
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

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

        grad_sum +=
            dxhat;

        grad_xhat_sum +=
            dxhat * xhat;

        c += 1;
    }

    let channels_f32 =
        channels as f32;

    let mean_dyg =
        grad_sum / channels_f32;

    let mean_dyg_xhat =
        grad_xhat_sum
            / channels_f32;

    let mean_dyg_vec =
        _mm256_set1_ps(
            mean_dyg
        );

    let mean_dyg_xhat_vec =
        _mm256_set1_ps(
            mean_dyg_xhat
        );

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

        let dgamma =
            _mm256_mul_ps(
                dy,
                xhat,
            );

        let old_dgamma =
            _mm256_loadu_ps(
                weight_grads.as_ptr().add(c)
            );

        _mm256_storeu_ps(
            weight_grads.as_mut_ptr().add(c),
            _mm256_add_ps(
                old_dgamma,
                dgamma,
            ),
        );

        let old_dbeta =
            _mm256_loadu_ps(
                bias_grads.as_ptr().add(c)
            );

        _mm256_storeu_ps(
            bias_grads.as_mut_ptr().add(c),
            _mm256_add_ps(
                old_dbeta,
                dy,
            ),
        );

        let dxhat =
            _mm256_mul_ps(
                dy,
                g,
            );

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
            output_grad.as_mut_ptr().add(c),
            dx,
        );

        c += 8;
    }

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

        weight_grads[c] +=
            dy * xhat;

        bias_grads[c] +=
            dy;

        let dxhat =
            dy * g;

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
    gamma: &[f32],
    workspace: &mut BackwardWorkspace,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    debug_assert!(batch_size > 0);
    debug_assert!(channels > 0);
    debug_assert_eq!(input.len(), grad.len());
    debug_assert_eq!(gamma.len(), channels);

    debug_assert_eq!(
        input.len() % batch_size,
        0
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

        let channels_f32 =
            channels as f32;

        let mut grad_sum =
            0.0f32;

        let mut grad_xhat_sum =
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

            grad_sum +=
                dxhat;

            grad_xhat_sum +=
                dxhat * xhat;
        }

        let mean_dyg =
            grad_sum
                / channels_f32;

        let mean_dyg_xhat =
            grad_xhat_sum
                / channels_f32;

        for c in 0..channels {
            let x =
                input_group[c];

            let dy =
                grad_group[c];

            let xhat =
                (x - mean)
                    * inv_std;

            workspace.weight_grads[c] +=
                dy * xhat;

            workspace.bias_grads[c] +=
                dy;

            let dxhat =
                dy * gamma[c];

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

    (
        input_grads,
        workspace.weight_grads.clone(),
        workspace.bias_grads.clone(),
    )
}


// ============================================================================
// Weight tying backward
// ============================================================================

pub fn weight_tying_backward(
    input: &[f32],
    grad: &[f32],
    batch_size: usize,
    embedding_dim: usize,
    vocab_size: usize,
    embeddings: &Embeddings,
    params: &ParameterStore,
    workspace: &mut BackwardWorkspace,
) -> (
    Vec<f32>,
    Vec<f32>,
) {
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

    let embedding_range =
        embeddings.parameter_range();

    let weights =
        params.values(embedding_range);

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

    let mut input_grads =
        vec![
            0.0f32;
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
            weights,
            embedding_dim as i32,
            0.0,
            &mut input_grads,
            embedding_dim as i32,
        );
    }

    (
        input_grads,
        workspace.embedding_grads.clone(),
    )
}


// ============================================================================
// GlobalMixer helpers
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn global_mixer_read_softmax_backward_avx2(
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

        let mut dot0 =
            _mm256_setzero_ps();

        let mut dot1 =
            _mm256_setzero_ps();

        let mut g = 0usize;

        while g + 16 <= global_dim {
            let p0 =
                _mm256_loadu_ps(
                    probs.as_ptr()
                        .add(base + g)
                );

            let dg0 =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g)
                );

            let p1 =
                _mm256_loadu_ps(
                    probs.as_ptr()
                        .add(base + g + 8)
                );

            let dg1 =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g + 8)
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
                        .add(base + g)
                );

            let dg =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g)
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
                        .add(base + g)
                );

            let dg =
                _mm256_loadu_ps(
                    score_grads.as_ptr()
                        .add(base + g)
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
}

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

fn global_mixer_positional_backward(
    score_grads: &[f32],
    position_features: &[[f32; GLOBAL_MIXER_POS_FEATURES]],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
    output_grads: &mut [f32],
) {
    debug_assert_eq!(
        score_grads.len(),
        batch_size
            * positions
            * global_dim
    );

    debug_assert_eq!(
        output_grads.len(),
        global_dim
            * GLOBAL_MIXER_POS_FEATURES
    );

    output_grads.fill(0.0);

    for g in 0..global_dim {
        let weight_base =
            g * GLOBAL_MIXER_POS_FEATURES;

        let mut grad0 =
            0.0f32;

        let mut grad1 =
            0.0f32;

        let mut grad2 =
            0.0f32;

        let mut grad3 =
            0.0f32;

        for b in 0..batch_size {
            let score_base =
                b * positions * global_dim;

            for p in 0..positions {
                let ds =
                    score_grads[
                        score_base
                            + p * global_dim
                            + g
                        ];

                let f =
                    position_features[p];

                grad0 +=
                    ds * f[0];

                grad1 +=
                    ds * f[1];

                grad2 +=
                    ds * f[2];

                grad3 +=
                    ds * f[3];
            }
        }

        output_grads[weight_base] =
            grad0;

        output_grads[weight_base + 1] =
            grad1;

        output_grads[weight_base + 2] =
            grad2;

        output_grads[weight_base + 3] =
            grad3;
    }
}


// ============================================================================
// GlobalMixer backward
// ============================================================================

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
    write_weights: &[f32],
    read_weights: &[f32],
    workspace: &mut BackwardWorkspace,
) -> (
    Vec<f32>,
    GlobalMixerGradients,
) {
    let rows =
        batch_size * positions;

    let position_features =
        global_mixer_position_features(
            positions
        );

    debug_assert_eq!(
        input.len(),
        rows * channels
    );

    debug_assert_eq!(
        grad.len(),
        rows * channels
    );

    debug_assert_eq!(
        write_probs.len(),
        rows * global_dim
    );

    debug_assert_eq!(
        read_probs.len(),
        rows * global_dim
    );

    debug_assert_eq!(
        global_vectors.len(),
        batch_size
            * global_dim
            * channels
    );

    debug_assert_eq!(
        write_weights.len(),
        global_dim * channels
    );

    debug_assert_eq!(
        read_weights.len(),
        global_dim * channels
    );

    #[cfg(target_arch = "x86_64")]
    let use_avx2 =
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma");

    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let mut mixer_grads =
        GlobalMixerGradients::default();

    // ========================================================================
    // 1. dGlobal = R^T dY
    // ========================================================================

    workspace.mixer_global_grads.resize(
        batch_size
            * global_dim
            * channels,
        0.0,
    );

    workspace.mixer_global_grads.fill(0.0);

    for b in 0..batch_size {
        let grad_base =
            b * positions * channels;

        let probs_base =
            b * positions * global_dim;

        let global_base =
            b * global_dim * channels;

        for g in 0..global_dim {
            let dst_base =
                global_base
                    + g * channels;

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
                                        ]
                                );

                            let r1 =
                                _mm256_set1_ps(
                                    read_probs[
                                        probs_base
                                            + (p + 1)
                                            * global_dim
                                            + g
                                        ]
                                );

                            let dy0 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_base
                                                + p * channels
                                                + c
                                        )
                                );

                            let dy1 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_base
                                                + (p + 1)
                                                * channels
                                                + c
                                        )
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
                                        ]
                                );

                            let dy =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_base
                                                + p * channels
                                                + c
                                        )
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

    // ========================================================================
    // 2. dR = dY Global^T
    // ========================================================================

    workspace.mixer_score_grads.resize(
        rows * global_dim,
        0.0,
    );

    workspace.mixer_score_grads.fill(0.0);

    for b in 0..batch_size {
        let grad_base =
            b * positions * channels;

        let global_base =
            b * global_dim * channels;

        let score_base =
            b * positions * global_dim;

        for p in 0..positions {
            let grad_row =
                grad_base
                    + p * channels;

            let score_row =
                score_base
                    + p * global_dim;

            for g in 0..global_dim {
                let global_row =
                    global_base
                        + g * channels;

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
                                        )
                                );

                            let gv0 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        )
                                );

                            let dy1 =
                                _mm256_loadu_ps(
                                    grad.as_ptr()
                                        .add(
                                            grad_row + c + 8
                                        )
                                );

                            let gv1 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_row + c + 8
                                        )
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
                                        )
                                );

                            let gv =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        )
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

    // ========================================================================
    // 3. Read softmax backward
    // ========================================================================

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

    // ========================================================================
    // 4. Read positional gradients
    // ========================================================================

    workspace.mixer_pos_grads_read.resize(
        global_dim
            * GLOBAL_MIXER_POS_FEATURES,
        0.0,
    );

    global_mixer_positional_backward(
        &workspace.mixer_score_grads,
        &position_features,
        batch_size,
        positions,
        global_dim,
        &mut workspace.mixer_pos_grads_read,
    );

    mixer_grads.read_positional_weights =
        workspace
            .mixer_pos_grads_read
            .clone();

    // ========================================================================
    // 5. Read weight gradient
    // ========================================================================

    workspace.weight_grads.resize(
        global_dim * channels,
        0.0,
    );

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

    mixer_grads.read_weights =
        workspace.weight_grads.clone();

    // ========================================================================
    // 6. Read bias gradient
    // ========================================================================

    workspace.bias_grads.resize(
        global_dim,
        0.0,
    );

    workspace.bias_grads.fill(0.0);

    for row in 0..rows {
        let base =
            row * global_dim;

        add_f32_slice_simd(
            &mut workspace.bias_grads,
            &workspace.mixer_score_grads[
                base..base + global_dim
                ],
        );
    }

    mixer_grads.read_biases =
        workspace.bias_grads.clone();

    // ========================================================================
    // 7. dX_read
    //
    // Add it after all original dY-dependent quantities above have already
    // been computed.
    // ========================================================================

    let mut read_input_grads =
        vec![0.0f32; rows * channels];

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
            read_weights,
            channels as i32,
            0.0,
            &mut read_input_grads,
            channels as i32,
        );
    }

    add_f32_slice_simd(
        &mut grad,
        &read_input_grads,
    );

    // ========================================================================
    // 8. dA = X dGlobal^T
    // ========================================================================

    for b in 0..batch_size {
        let input_base =
            b * positions * channels;

        let global_base =
            b * global_dim * channels;

        let score_base =
            b * positions * global_dim;

        for p in 0..positions {
            let input_row =
                input_base
                    + p * channels;

            let score_row =
                score_base
                    + p * global_dim;

            for g in 0..global_dim {
                let global_row =
                    global_base
                        + g * channels;

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
                                        )
                                );

                            let dg0 =
                                _mm256_loadu_ps(
                                    workspace
                                        .mixer_global_grads
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        )
                                );

                            let x1 =
                                _mm256_loadu_ps(
                                    input.as_ptr()
                                        .add(
                                            input_row + c + 8
                                        )
                                );

                            let dg1 =
                                _mm256_loadu_ps(
                                    workspace
                                        .mixer_global_grads
                                        .as_ptr()
                                        .add(
                                            global_row + c + 8
                                        )
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
                                        )
                                );

                            let dg =
                                _mm256_loadu_ps(
                                    workspace
                                        .mixer_global_grads
                                        .as_ptr()
                                        .add(
                                            global_row + c
                                        )
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

    // ========================================================================
    // 9. Write softmax backward
    // ========================================================================

    for b in 0..batch_size {
        let score_base =
            b * positions * global_dim;

        for g in 0..global_dim {
            let mut dot =
                0.0f32;

            for p in 0..positions {
                let idx =
                    score_base
                        + p * global_dim
                        + g;

                dot +=
                    write_probs[idx]
                        * workspace
                        .mixer_score_grads[idx];
            }

            for p in 0..positions {
                let idx =
                    score_base
                        + p * global_dim
                        + g;

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

    // ========================================================================
    // 10. Write positional gradients
    // ========================================================================

    workspace.mixer_pos_grads_write.resize(
        global_dim
            * GLOBAL_MIXER_POS_FEATURES,
        0.0,
    );

    global_mixer_positional_backward(
        &workspace.mixer_score_grads,
        &position_features,
        batch_size,
        positions,
        global_dim,
        &mut workspace.mixer_pos_grads_write,
    );

    mixer_grads.write_positional_weights =
        workspace
            .mixer_pos_grads_write
            .clone();

    // ========================================================================
    // 11. Write weight gradient
    // ========================================================================

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

    mixer_grads.write_weights =
        workspace.weight_grads.clone();

    // ========================================================================
    // 12. Write bias gradient
    // ========================================================================

    workspace.bias_grads.fill(0.0);

    for row in 0..rows {
        let base =
            row * global_dim;

        add_f32_slice_simd(
            &mut workspace.bias_grads,
            &workspace.mixer_score_grads[
                base..base + global_dim
                ],
        );
    }

    mixer_grads.write_biases =
        workspace.bias_grads.clone();

    // ========================================================================
    // 13. dX_write
    // ========================================================================

    let mut write_input_grads =
        vec![0.0f32; rows * channels];

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
            write_weights,
            channels as i32,
            0.0,
            &mut write_input_grads,
            channels as i32,
        );
    }

    add_f32_slice_simd(
        &mut grad,
        &write_input_grads,
    );

    // ========================================================================
    // 14. dX_global = A dGlobal
    // ========================================================================

    for b in 0..batch_size {
        let grad_base =
            b * positions * channels;

        let probs_base =
            b * positions * global_dim;

        let global_base =
            b * global_dim * channels;

        for p in 0..positions {
            let dst_base =
                grad_base
                    + p * channels;

            let probs_row =
                probs_base
                    + p * global_dim;

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
                                        ]
                                );

                            let a1 =
                                _mm256_set1_ps(
                                    write_probs[
                                        probs_row + g + 1
                                        ]
                                );

                            let gv0 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_base
                                                + g * channels
                                                + c
                                        )
                                );

                            let gv1 =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_base
                                                + (g + 1)
                                                * channels
                                                + c
                                        )
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
                                        ]
                                );

                            let gv =
                                _mm256_loadu_ps(
                                    global_vectors
                                        .as_ptr()
                                        .add(
                                            global_base
                                                + g * channels
                                                + c
                                        )
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
                                    )
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

    (
        grad,
        mixer_grads,
    )
}


#[derive(Default)]
pub struct GlobalMixerGradients {
    pub write_weights: Vec<f32>,
    pub write_biases: Vec<f32>,

    pub read_weights: Vec<f32>,
    pub read_biases: Vec<f32>,

    pub write_positional_weights: Vec<f32>,
    pub read_positional_weights: Vec<f32>,
}


// ============================================================================
// Main backward dispatcher
// ============================================================================

pub fn backward_layers_batch(
    model_layers: &[Layer],
    caches: &[BatchLayerCache],
    params: &mut ParameterStore,
    grad: Vec<f32>,
    batch_size: usize,
) -> Vec<f32> {
    BACKWARD_WORKSPACE.with(|cell| {
        let mut workspace =
            cell.borrow_mut();

        backward_layers_batch_inner(
            model_layers,
            caches,
            params,
            grad,
            batch_size,
            &mut workspace,
        )
    })
}

fn backward_layers_batch_inner(
    model_layers: &[Layer],
    caches: &[BatchLayerCache],
    params: &mut ParameterStore,
    mut grad: Vec<f32>,
    batch_size: usize,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    debug_assert_eq!(
        model_layers.len(),
        caches.len(),
    );

    for (layer, cache) in
        model_layers.iter()
            .zip(caches.iter())
            .rev()
    {
        grad = backward_layer_batch(
            layer,
            cache,
            params,
            grad,
            batch_size,
            workspace,
        );
    }

    grad
}


// ============================================================================
// Per-layer backward
// ============================================================================

fn backward_layer_batch(
    layer: &Layer,
    cache: &BatchLayerCache,
    params: &mut ParameterStore,
    mut grad: Vec<f32>,
    batch_size: usize,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    match (layer, cache) {

        // ====================================================================
        // Dense
        // ====================================================================

        (
            Layer::Dense(layer),
            BatchLayerCache::Dense {
                input_size,
                output_size,
                input,
                activation_output,
                activation,
            },
        ) => {
            debug_assert_eq!(
                *input_size,
                layer.input_size
            );

            debug_assert_eq!(
                *output_size,
                layer.output_size
            );

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

            let weights =
                params.values(layer.weights);

            workspace.weight_grads.resize(
                *output_size * *input_size,
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

                add_f32_slice_simd(
                    &mut workspace.bias_grads,
                    &grad[
                        base..base + *output_size
                        ],
                );
            }

            let mut input_grads =
                vec![
                    0.0f32;
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

            accumulate_parameter_grads(
                params,
                layer.weights,
                &workspace.weight_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.biases,
                &workspace.bias_grads,
            );

            input_grads
        }

        // ====================================================================
        // Conv1D
        // ====================================================================

        (
            Layer::Conv1D(layer),
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
                activation,
            },
        ) => {
            let weights =
                params.values(layer.weights);

            let input_grads =
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
                    weights,
                    activation,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                layer.weights,
                &workspace.weight_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.biases,
                &workspace.bias_grads,
            );

            input_grads
        }

        // ====================================================================
        // Residual
        // ====================================================================

        (
            Layer::Residual(inner_layers),
            BatchLayerCache::Residual {
                inner,
            },
        ) => {
            let skip_grad =
                grad.clone();

            let mut result =
                backward_layers_batch_inner(
                    &*inner_layers.layers,
                    inner,
                    params,
                    grad,
                    batch_size,
                    workspace,
                );

            debug_assert_eq!(
                result.len(),
                skip_grad.len()
            );

            add_f32_slice_simd(
                &mut result,
                &skip_grad,
            );

            result
        }

        // ====================================================================
        // Depthwise Conv1D
        // ====================================================================

        (
            Layer::DepthwiseConv1D(layer),
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
                activation,
            },
        ) => {
            let weights =
                params.values(layer.weights);

            let (
                input_grads,
                weight_grads,
                bias_grads,
            ) =
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
                    weights,
                    activation,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                layer.weights,
                &weight_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.biases,
                &bias_grads,
            );

            input_grads
        }

        // ====================================================================
        // Grouped Conv1D
        // ====================================================================

        (
            Layer::GroupedConv1D(layer),
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
                activation,
            },
        ) => {
            let weights =
                params.values(layer.weights);

            let (
                input_grads,
                weight_grads,
                bias_grads,
            ) =
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
                    weights,
                    activation,
                );

            accumulate_parameter_grads(
                params,
                layer.weights,
                &weight_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.biases,
                &bias_grads,
            );

            input_grads
        }

        // ====================================================================
        // Low-rank pointwise
        // ====================================================================

        (
            Layer::LowRankPointwise(layer),
            BatchLayerCache::LowRankPointwise {
                input,
                hidden,
                output,
                rows,
                in_channels,
                rank,
                out_channels,
                activation,
                ..
            },
        ) => {
            let first_weights =
                params.values(
                    layer.first_weights
                );

            let second_weights =
                params.values(
                    layer.second_weights
                );

            let (
                input_grads,
                first_weight_grads,
                first_bias_grads,
                second_weight_grads,
                second_bias_grads,
            ) =
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
                    activation,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                layer.first_weights,
                &first_weight_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.first_biases,
                &first_bias_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.second_weights,
                &second_weight_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.second_biases,
                &second_bias_grads,
            );

            input_grads
        }

        // ====================================================================
        // ChannelScale
        // ====================================================================

        (
            Layer::ChannelScale(layer),
            BatchLayerCache::ChannelScale {
                input,
                channels,
            },
        ) => {
            let scales =
                params.values(
                    layer.scales
                );

            let (
                input_grads,
                scale_grads,
                bias_grads,
            ) =
                channel_scale_backward(
                    input,
                    &grad,
                    batch_size,
                    *channels,
                    scales,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                layer.scales,
                &scale_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.biases,
                &bias_grads,
            );

            input_grads
        }

        // ====================================================================
        // LayerNorm
        // ====================================================================

        (
            Layer::LayerNorm(layer),
            BatchLayerCache::LayerNorm {
                input,
                means,
                inv_stds,
                channels,
            },
        ) => {
            let gamma =
                params.values(
                    layer.gamma
                );

            let (
                input_grads,
                gamma_grads,
                beta_grads,
            ) =
                layer_norm_backward(
                    input,
                    &grad,
                    batch_size,
                    *channels,
                    means,
                    inv_stds,
                    gamma,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                layer.gamma,
                &gamma_grads,
            );

            accumulate_parameter_grads(
                params,
                layer.beta,
                &beta_grads,
            );

            input_grads
        }

        // ====================================================================
        // Weight tying
        // ====================================================================

        (
            Layer::WeightTying(layer),
            BatchLayerCache::WeightTying {
                input,
                embeddings,
                batch_size: cached_batch_size,
                embedding_dim,
                vocab_size,
            },
        ) => {
            debug_assert_eq!(
                *cached_batch_size,
                batch_size
            );

            let (
                input_grads,
                embedding_grads,
            ) =
                weight_tying_backward(
                    input,
                    &grad,
                    *cached_batch_size,
                    *embedding_dim,
                    *vocab_size,
                    embeddings,
                    params,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                embeddings.parameter_range(),
                &embedding_grads,
            );

            let _ = layer;

            input_grads
        }

        // ====================================================================
        // GlobalMixer
        // ====================================================================

        (
            Layer::GlobalMixer(layer),
            BatchLayerCache::GlobalMixer {
                input,
                write_probs,
                global_vectors,
                read_probs,
                positions,
                channels,
                global_dim,
            },
        ) => {
            let write_weights =
                params.values(
                    layer.write_weights
                );

            let read_weights =
                params.values(
                    layer.read_weights
                );

            let (
                input_grads,
                mixer_grads,
            ) =
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
                    write_weights,
                    read_weights,
                    workspace,
                );

            accumulate_parameter_grads(
                params,
                layer.write_weights,
                &mixer_grads.write_weights,
            );

            accumulate_parameter_grads(
                params,
                layer.write_biases,
                &mixer_grads.write_biases,
            );

            accumulate_parameter_grads(
                params,
                layer.read_weights,
                &mixer_grads.read_weights,
            );

            accumulate_parameter_grads(
                params,
                layer.read_biases,
                &mixer_grads.read_biases,
            );

            accumulate_parameter_grads(
                params,
                layer.write_positional_weights,
                &mixer_grads.write_positional_weights,
            );

            accumulate_parameter_grads(
                params,
                layer.read_positional_weights,
                &mixer_grads.read_positional_weights,
            );

            input_grads
        }

        _ => {
            panic!(
                "Layer/cache mismatch during backward pass"
            );
        }
    }
}
