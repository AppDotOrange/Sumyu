// NOT USED
// THIS FILE WAS A FAILED ATTEMPT
// REAL KERNEL IS IN backwards.rs USING BLAS

use crate::neuron::Activation;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256,
    _mm256_add_ps,
    _mm256_blendv_ps,
    _mm256_cmp_ps,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_setzero_ps,
    _mm256_storeu_ps,
    _CMP_LE_OQ,
};

const CHANNEL_TILE: usize = 8;

#[derive(Clone, Copy)]
struct OutputSpan {
    origin: isize,
    k_start: usize,
    k_end: usize,
}

#[inline(always)]
fn make_span(
    output_position: usize,
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
) -> OutputSpan {
    let left_padding =
        if causal {
            kernel_size - 1
        } else {
            padding
        };

    /*
     * source = output * stride - left_padding + k
     */
    let origin =
        output_position as isize
            * stride as isize
            - left_padding as isize;

    /*
     * Need:
     *
     *     0 <= origin + k < input_length
     */
    let k_start_i =
        (-origin)
            .max(0)
            .min(kernel_size as isize);

    let k_end_i =
        (input_length as isize - origin)
            .clamp(
                0,
                kernel_size as isize,
            );

    let k_start =
        k_start_i as usize;

    let k_end =
        k_end_i as usize;

    /*
     * These are the exact invariants relied on by the
     * pointer arithmetic in the AVX2 kernel.
     */
    if k_start < k_end {
        debug_assert!(
            origin + k_start as isize >= 0
        );

        debug_assert!(
            origin + (k_start as isize)
            < input_length as isize
        );

        debug_assert!(
            (origin + (k_end - 1) as isize)
            < input_length as isize
        );
    }

    OutputSpan {
        origin,
        k_start,
        k_end,
    }
}

fn build_spans(
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
) -> Vec<OutputSpan> {
    let mut spans =
        Vec::with_capacity(
            output_length,
        );

    for output_position in
        0..output_length
    {
        spans.push(
            make_span(
                output_position,
                input_length,
                kernel_size,
                stride,
                padding,
                causal,
            ),
        );
    }

    spans
}

/* ============================================================
 * LeakyReLU derivative
 * ============================================================
 */

#[inline(always)]
fn apply_leaky_scalar(
    dy: f32,
    activation: f32,
    slope: f32,
) -> f32 {
    if activation <= 0.0 {
        dy * slope
    } else {
        dy
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_leaky_avx2(
    dy: __m256,
    activation: __m256,
    slope: __m256,
) -> __m256 {
    let zero =
        _mm256_setzero_ps();

    let one =
        _mm256_set1_ps(1.0);

    let mask =
        _mm256_cmp_ps(
            activation,
            zero,
            _CMP_LE_OQ,
        );

    let multiplier =
        _mm256_blendv_ps(
            one,
            slope,
            mask,
        );

    _mm256_mul_ps(
        dy,
        multiplier,
    )
}

/* ============================================================
 * Fused generic AVX2 backward
 *
 * One traversal computes:
 *
 *     dB
 *     dW
 *     dX
 *
 * Layout:
 *
 *     input / grad / activation:
 *         [batch][position][channel]
 *
 *     internal weights:
 *         [kernel][channel]
 *
 * This deliberately stays output-major because that gives the
 * best locality for dY and preserves excellent batch scaling.
 * ============================================================
 */

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn backward_fused_avx2<
    const LEAKY: bool,
>(
    input: *const f32,
    grad: *const f32,
    activation: *const f32,

    input_grad: *mut f32,

    weight_grad_kmajor: *mut f32,
    bias_grad: *mut f32,

    batch_size: usize,
    input_length: usize,
    output_length: usize,
    channels: usize,

    kernel_size: usize,
    spans: &[OutputSpan],

    weights_kmajor: *const f32,

    activation_slope: f32,
) {
    debug_assert_eq!(
        spans.len(),
        output_length,
    );

    let full_channels =
        channels / CHANNEL_TILE;

    let simd_channels =
        full_channels
            * CHANNEL_TILE;

    let slope =
        _mm256_set1_ps(
            activation_slope,
        );

    /*
     * Reuse this accumulator allocation for every channel
     * block instead of allocating once per block.
     */
    let mut dw =
        vec![
            _mm256_setzero_ps();
            kernel_size
        ];

    for block in 0..full_channels {
        let c_base =
            block * CHANNEL_TILE;

        /*
         * Reset dW accumulators.
         */
        let zero =
            _mm256_setzero_ps();

        for accumulator in
            dw.iter_mut()
        {
            *accumulator = zero;
        }

        let mut db =
            _mm256_setzero_ps();

        for batch in 0..batch_size {
            let input_batch =
                input.add(
                    batch
                        * input_length
                        * channels,
                );

            let input_grad_batch =
                input_grad.add(
                    batch
                        * input_length
                        * channels,
                );

            let grad_batch =
                grad.add(
                    batch
                        * output_length
                        * channels,
                );

            let activation_batch =
                activation.add(
                    batch
                        * output_length
                        * channels,
                );

            for output_position in
                0..output_length
            {
                let span =
                    spans[
                        output_position
                        ];

                if span.k_start
                    == span.k_end
                {
                    continue;
                }

                let output_base =
                    output_position
                        * channels
                        + c_base;

                /*
                 * dY is loaded exactly once.
                 *
                 * For LeakyReLU the derivative is also applied
                 * exactly once.
                 */
                let mut dy =
                    _mm256_loadu_ps(
                        grad_batch.add(
                            output_base
                        )
                    );

                if LEAKY {
                    let y =
                        _mm256_loadu_ps(
                            activation_batch
                                .add(
                                    output_base
                                )
                        );

                    dy =
                        apply_leaky_avx2(
                            dy,
                            y,
                            slope,
                        );
                }

                /*
                 * dB
                 */
                db =
                    _mm256_add_ps(
                        db,
                        dy,
                    );

                /*
                 * The first valid input position for this
                 * output is:
                 *
                 *     origin + k_start
                 *
                 * Since k increases by one, source also
                 * increases by exactly one.
                 *
                 * This removes an integer add involving `k`
                 * from every iteration.
                 */
                let mut source_position =
                    (span.origin
                        + span.k_start
                        as isize)
                        as usize;

                let mut k =
                    span.k_start;

                while k < span.k_end {
                    debug_assert!(
                        source_position
                            < input_length
                    );

                    let input_base =
                        source_position
                            * channels
                            + c_base;

                    /*
                     * dW[k] += X * dY
                     */
                    let x =
                        _mm256_loadu_ps(
                            input_batch.add(
                                input_base
                            )
                        );

                    dw[k] =
                        _mm256_fmadd_ps(
                            x,
                            dy,
                            dw[k],
                        );

                    /*
                     * dX += W[k] * dY
                     *
                     * W is contiguous in the internal
                     * [kernel][channel] representation.
                     */
                    let w =
                        _mm256_loadu_ps(
                            weights_kmajor.add(
                                k * channels
                                    + c_base
                            )
                        );

                    let dx_ptr =
                        input_grad_batch
                            .add(
                                input_base
                            );

                    let old_dx =
                        _mm256_loadu_ps(
                            dx_ptr
                        );

                    let new_dx =
                        _mm256_fmadd_ps(
                            w,
                            dy,
                            old_dx,
                        );

                    _mm256_storeu_ps(
                        dx_ptr,
                        new_dx,
                    );

                    source_position +=
                        1;

                    k += 1;
                }
            }
        }

        /*
         * One bias store per channel block.
         */
        _mm256_storeu_ps(
            bias_grad.add(c_base),
            db,
        );

        /*
         * One dW store per kernel tap.
         */
        for k in 0..kernel_size {
            _mm256_storeu_ps(
                weight_grad_kmajor.add(
                    k * channels
                        + c_base,
                ),
                dw[k],
            );
        }
    }

    /* ========================================================
     * Scalar channel remainder
     * ========================================================
     */

    let mut scalar_dw =
        vec![
            0.0f32;
            kernel_size
        ];

    for c in simd_channels..channels {
        for value in
            scalar_dw.iter_mut()
        {
            *value = 0.0;
        }

        let mut db =
            0.0f32;

        for batch in 0..batch_size {
            let input_batch =
                input.add(
                    batch
                        * input_length
                        * channels,
                );

            let input_grad_batch =
                input_grad.add(
                    batch
                        * input_length
                        * channels,
                );

            let grad_batch =
                grad.add(
                    batch
                        * output_length
                        * channels,
                );

            let activation_batch =
                activation.add(
                    batch
                        * output_length
                        * channels,
                );

            for output_position in
                0..output_length
            {
                let span =
                    spans[
                        output_position
                        ];

                if span.k_start
                    == span.k_end
                {
                    continue;
                }

                let output_base =
                    output_position
                        * channels
                        + c;

                let mut dy =
                    *grad_batch.add(
                        output_base
                    );

                if LEAKY {
                    dy =
                        apply_leaky_scalar(
                            dy,
                            *activation_batch
                                .add(
                                    output_base
                                ),
                            activation_slope,
                        );
                }

                db += dy;

                let mut source_position =
                    (
                        span.origin
                            + span.k_start
                            as isize
                    ) as usize;

                let mut k =
                    span.k_start;

                while k < span.k_end {
                    debug_assert!(
                        source_position
                            < input_length
                    );

                    let input_base =
                        source_position
                            * channels
                            + c;

                    let x =
                        *input_batch.add(
                            input_base
                        );

                    scalar_dw[k] +=
                        x * dy;

                    let w =
                        *weights_kmajor.add(
                            k * channels + c
                        );

                    *input_grad_batch
                        .add(input_base)
                        += w * dy;

                    source_position +=
                        1;

                    k += 1;
                }
            }
        }

        *bias_grad.add(c) =
            db;

        for k in 0..kernel_size {
            *weight_grad_kmajor.add(
                k * channels + c,
            ) = scalar_dw[k];
        }
    }
}

/* ============================================================
 * Scalar fallback
 * ============================================================
 */

fn backward_scalar(
    input: &[f32],
    grad: &[f32],
    activation_output: &[f32],

    input_grad: &mut [f32],
    weight_grad: &mut [f32],
    bias_grad: &mut [f32],

    batch_size: usize,
    input_length: usize,
    output_length: usize,
    channels: usize,

    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,

    weights: &[f32],

    activation_mode: &Activation,
) {
    let left_padding =
        if causal {
            kernel_size - 1
        } else {
            padding
        };

    for batch in 0..batch_size {
        for output_position in
            0..output_length
        {
            let origin =
                output_position
                    .saturating_mul(stride)
                    as isize
                    - left_padding as isize;

            let k_start =
                (-origin)
                    .max(0)
                    .min(kernel_size as isize)
                    as usize;

            let k_end =
                (input_length as isize
                    - origin)
                    .clamp(
                        0,
                        kernel_size as isize,
                    ) as usize;

            let output_base =
                (
                    batch
                        * output_length
                        + output_position
                ) * channels;

            for c in 0..channels {
                let mut dy =
                    grad[
                        output_base + c
                        ];

                if let Activation::LeakyReLU {
                    slope,
                } = activation_mode
                {
                    dy =
                        apply_leaky_scalar(
                            dy,
                            activation_output[
                                output_base + c
                                ],
                            *slope,
                        );
                }

                bias_grad[c] += dy;

                for k in
                    k_start..k_end
                {
                    let source_position =
                        (
                            origin
                                + k as isize
                        ) as usize;

                    let input_base =
                        (
                            batch
                                * input_length
                                + source_position
                        ) * channels
                            + c;

                    let weight_index =
                        c * kernel_size
                            + k;

                    input_grad[
                        input_base
                        ] +=
                        weights[
                            weight_index
                            ] * dy;

                    weight_grad[
                        weight_index
                        ] +=
                        input[
                            input_base
                            ] * dy;
                }
            }
        }
    }
}

/* ============================================================
 * Public entry point
 * ============================================================
 */

pub fn backward_direct(
    input: &[f32],
    grad: &[f32],
    activation_output: &[f32],

    batch_size: usize,
    input_length: usize,
    output_length: usize,

    channels: usize,

    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,

    weights: &[f32],

    activation: &Activation,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    assert!(
        kernel_size > 0,
        "kernel_size must be > 0"
    );

    assert!(
        stride > 0,
        "stride must be > 0"
    );

    assert!(
        channels > 0,
        "channels must be > 0"
    );

    let expected_input =
        batch_size
            .checked_mul(input_length)
            .and_then(|x| {
                x.checked_mul(channels)
            })
            .expect(
                "input shape overflow"
            );

    let expected_output =
        batch_size
            .checked_mul(output_length)
            .and_then(|x| {
                x.checked_mul(channels)
            })
            .expect(
                "output shape overflow"
            );

    let expected_weights =
        channels
            .checked_mul(kernel_size)
            .expect(
                "weight shape overflow"
            );

    assert_eq!(
        input.len(),
        expected_input,
        "input length mismatch"
    );

    assert_eq!(
        grad.len(),
        expected_output,
        "gradient length mismatch"
    );

    assert_eq!(
        activation_output.len(),
        expected_output,
        "activation length mismatch"
    );

    assert_eq!(
        weights.len(),
        expected_weights,
        "weight length mismatch"
    );

    let mut input_grad =
        vec![
            0.0f32;
            expected_input
        ];

    let mut weight_grad =
        vec![
            0.0f32;
            expected_weights
        ];

    let mut bias_grad =
        vec![
            0.0f32;
            channels
        ];

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!(
            "avx2"
        ) && std::arch::is_x86_feature_detected!(
            "fma"
        ) {
            /*
             * Convert public:
             *
             *     [channel][kernel]
             *
             * to:
             *
             *     [kernel][channel]
             *
             * so every AVX2 weight load is contiguous.
             */
            let mut weights_kmajor =
                vec![
                    0.0f32;
                    expected_weights
                ];

            for c in 0..channels {
                let source_base =
                    c * kernel_size;

                for k in 0..kernel_size {
                    weights_kmajor[
                        k * channels + c
                        ] =
                        weights[
                            source_base + k
                            ];
                }
            }

            let mut weight_grad_kmajor =
                vec![
                    0.0f32;
                    expected_weights
                ];

            /*
             * Geometry is calculated once, outside the hot
             * gradient loops.
             */
            let spans =
                build_spans(
                    input_length,
                    output_length,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                );

            debug_assert_eq!(
                spans.len(),
                output_length
            );

            /*
             * Validate the generated spans in debug builds.
             *
             * This makes bad geometry fail immediately instead
             * of turning into silent memory corruption.
             */
            for span in &spans {
                if span.k_start
                    < span.k_end
                {
                    let first =
                        span.origin
                            + span.k_start
                            as isize;

                    let last =
                        span.origin
                            + (span.k_end - 1)
                            as isize;

                    debug_assert!(
                        first >= 0
                    );

                    debug_assert!(
                        last >= 0
                    );

                    debug_assert!(
                        first
                            < input_length as isize
                    );

                    debug_assert!(
                        last
                            < input_length as isize
                    );
                }
            }

            match activation {
                Activation::None => unsafe {
                    backward_fused_avx2::<false>(
                        input.as_ptr(),
                        grad.as_ptr(),
                        activation_output.as_ptr(),

                        input_grad.as_mut_ptr(),

                        weight_grad_kmajor
                            .as_mut_ptr(),
                        bias_grad.as_mut_ptr(),

                        batch_size,
                        input_length,
                        output_length,
                        channels,

                        kernel_size,
                        &spans,

                        weights_kmajor.as_ptr(),

                        0.0,
                    );
                },

                Activation::LeakyReLU {
                    slope,
                } => unsafe {
                    backward_fused_avx2::<true>(
                        input.as_ptr(),
                        grad.as_ptr(),
                        activation_output.as_ptr(),

                        input_grad.as_mut_ptr(),

                        weight_grad_kmajor
                            .as_mut_ptr(),
                        bias_grad.as_mut_ptr(),

                        batch_size,
                        input_length,
                        output_length,
                        channels,

                        kernel_size,
                        &spans,

                        weights_kmajor.as_ptr(),

                        *slope,
                    );
                },
            }

            /*
             * Convert:
             *
             *     [kernel][channel]
             *
             * ->
             *
             *     [channel][kernel]
             */
            for c in 0..channels {
                let destination_base =
                    c * kernel_size;

                for k in 0..kernel_size {
                    weight_grad[
                        destination_base + k
                        ] =
                        weight_grad_kmajor[
                            k * channels + c
                            ];
                }
            }

            return (
                input_grad,
                weight_grad,
                bias_grad,
            );
        }
    }

    backward_scalar(
        input,
        grad,
        activation_output,

        &mut input_grad,
        &mut weight_grad,
        &mut bias_grad,

        batch_size,
        input_length,
        output_length,
        channels,

        kernel_size,
        stride,
        padding,
        causal,

        weights,

        activation,
    );

    (
        input_grad,
        weight_grad,
        bias_grad,
    )
}
/*
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

    /*
     * Load depthwise weights.
     *
     * Public weight layout:
     *
     *     [channel][kernel]
     */
    workspace.weights.resize(
        weight_handles.len(),
        0.0,
    );

    crate::handle_data_slice(
        weight_handles,
        &mut workspace.weights,
    );

    /*
     * IMPORTANT:
     *
     * Do NOT call activation.backward() here.
     *
     * The direct kernel handles:
     *
     *     None
     *     LeakyReLU
     *
     * itself, using `output` as the activation output.
     */

    let (
        input_grads,
        weight_grads,
        bias_grads,
    ) =
        crate::depthwise_backward::backward_direct(
            input,
            grad,
            output,

            batch_size,
            input_length,
            output_length,

            in_channels,

            kernel_size,
            stride,
            padding,
            causal,

            &workspace.weights,

            activation,
        );

    /*
     * Accumulate parameter gradients into the actual
     * tensor handles.
     */
    crate::add_handle_grad_slices_2(
        weight_handles,
        &weight_grads,

        bias_handles,
        &bias_grads,
    );

    input_grads
}
*/
