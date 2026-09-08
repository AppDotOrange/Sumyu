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

const OC_TILE: usize = 8;
const IC_SIMD: usize = 8;

#[derive(Clone, Copy)]
struct KernelSpan {
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
) -> KernelSpan {
    let left_padding = if causal {
        kernel_size - 1
    } else {
        padding
    };

    let origin =
        output_position as isize * stride as isize
            - left_padding as isize;

    let k_start = (-origin)
        .max(0)
        .min(kernel_size as isize) as usize;

    let k_end = (input_length as isize - origin)
        .clamp(0, kernel_size as isize) as usize;

    KernelSpan {
        origin,
        k_start,
        k_end,
    }
}

#[inline]
fn build_spans(
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
) -> Vec<KernelSpan> {
    (0..output_length)
        .map(|output_position| {
            make_span(
                output_position,
                input_length,
                kernel_size,
                stride,
                padding,
                causal,
            )
        })
        .collect()
}


/* ============================================================
 * Leaky ReLU
 * ============================================================
 *
 * The forward kernel uses:
 *
 *     if y <= 0.0 {
 *         y *= slope;
 *     }
 *
 * Therefore the backward derivative uses exactly the same
 * condition:
 *
 *     if y <= 0.0 {
 *         dy *= slope;
 *     }
 *
 * `activation` is the OUTPUT of the forward activation.
 *
 * For the normal case:
 *
 *     activation = None
 *
 * the compiler eliminates all activation logic when LEAKY=false.
 * ============================================================
 */

#[inline(always)]
fn apply_leaky_scalar(
    value: f32,
    activation: f32,
    slope: f32,
) -> f32 {
    if activation <= 0.0 {
        value * slope
    } else {
        value
    }
}

/* ============================================================
 * dX
 *
 * One output position + one output-channel tile.
 *
 * dX[src,c] += Σ_oc dY[out,oc] * W[oc,k,c]
 *
 * W layout:
 *     [oc][k][ic]
 *
 * X / dX layout:
 *     [position][ic]
 *
 * LEAKY:
 *     dY is transformed exactly once when entering this kernel.
 *     The transformed values remain cached in `dy`.
 * ============================================================
 */

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dinput<const OC: usize, const LEAKY: bool>(
    grad_row: *const f32,
    activation_row: *const f32,

    weights: *const f32,
    input_grad_batch: *mut f32,

    in_channels: usize,
    kernel_size: usize,

    span: KernelSpan,

    activation_slope: f32,
) {
    debug_assert!(OC > 0 && OC <= OC_TILE);

    let kernel_width =
        kernel_size * in_channels;

    /*
     * dY is reused for every k and every IC vector.
     *
     * For LeakyReLU, apply the derivative exactly once here.
     */
    let mut dy =
        [0.0f32; OC_TILE];

    for oc in 0..OC {
        let value =
            *grad_row.add(oc);

        if LEAKY {
            dy[oc] =
                apply_leaky_scalar(
                    value,
                    *activation_row.add(oc),
                    activation_slope,
                );
        } else {
            dy[oc] = value;
        }
    }

    let simd_end =
        in_channels / IC_SIMD * IC_SIMD;

    for k in span.k_start..span.k_end {
        let src_position =
            (span.origin + k as isize) as usize;

        let dinput_row =
            input_grad_batch.add(
                src_position * in_channels
            );

        let weight_k_offset =
            k * in_channels;

        /*
         * Main SIMD body.
         */
        let mut c = 0;

        while c < simd_end {
            let mut acc =
                _mm256_loadu_ps(
                    dinput_row.add(c)
                );

            for oc in 0..OC {
                let weights_ptr =
                    weights.add(
                        oc * kernel_width
                            + weight_k_offset
                            + c
                    );

                let w =
                    _mm256_loadu_ps(
                        weights_ptr
                    );

                acc =
                    _mm256_fmadd_ps(
                        w,
                        _mm256_set1_ps(dy[oc]),
                        acc,
                    );
            }

            _mm256_storeu_ps(
                dinput_row.add(c),
                acc,
            );

            c += IC_SIMD;
        }

        /*
         * IC remainder.
         */
        while c < in_channels {
            let mut value =
                *dinput_row.add(c);

            for oc in 0..OC {
                value +=
                    dy[oc]
                        * *weights.add(
                        oc * kernel_width
                            + weight_k_offset
                            + c
                    );
            }

            *dinput_row.add(c) =
                value;

            c += 1;
        }
    }
}


/* ============================================================
 * dW
 *
 * One OC tile + one kernel position + one IC SIMD vector.
 *
 * dW[oc,k,c] += X[src,c] * dY[out,oc]
 *
 * Activation derivative is applied when dY is loaded.
 * ============================================================
 */

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dweight<const OC: usize, const LEAKY: bool>(
    input: *const f32,
    grad: *const f32,
    activation: *const f32,

    weight_grad: *mut f32,

    batch_size: usize,
    input_length: usize,
    output_length: usize,

    in_channels: usize,
    out_channels: usize,

    kernel_size: usize,
    oc_base: usize,

    spans: &[KernelSpan],

    activation_slope: f32,
) {
    debug_assert!(OC > 0 && OC <= OC_TILE);
    debug_assert_eq!(spans.len(), output_length);

    let kernel_width =
        kernel_size * in_channels;

    let simd_end =
        in_channels / IC_SIMD * IC_SIMD;

    for k in 0..kernel_size {
        let weight_k_offset =
            k * in_channels;

        let mut c = 0;

        while c < simd_end {
            /*
             * One YMM accumulator per output channel.
             *
             * OC=8:
             *
             *     8 accumulators × 8 f32
             *
             * = 64 dW values accumulated before stores.
             */
            let mut acc: [__m256; OC_TILE] =
                [_mm256_setzero_ps(); OC_TILE];

            for batch in 0..batch_size {
                let input_batch =
                    input.add(
                        batch
                            * input_length
                            * in_channels
                    );

                let grad_batch =
                    grad.add(
                        batch
                            * output_length
                            * out_channels
                    );

                let activation_batch =
                    activation.add(
                        batch
                            * output_length
                            * out_channels
                    );

                for output_position in 0..output_length {
                    let span =
                        spans[output_position];

                    if k < span.k_start
                        || k >= span.k_end
                    {
                        continue;
                    }

                    let src_position =
                        (span.origin
                            + k as isize)
                            as usize;

                    let x =
                        _mm256_loadu_ps(
                            input_batch.add(
                                src_position
                                    * in_channels
                                    + c
                            )
                        );

                    let dy_ptr =
                        grad_batch.add(
                            output_position
                                * out_channels
                                + oc_base
                        );

                    let activation_ptr =
                        activation_batch.add(
                            output_position
                                * out_channels
                                + oc_base
                        );

                    for oc in 0..OC {
                        let mut dy =
                            *dy_ptr.add(oc);

                        if LEAKY {
                            dy =
                                apply_leaky_scalar(
                                    dy,
                                    *activation_ptr.add(oc),
                                    activation_slope,
                                );
                        }

                        acc[oc] =
                            _mm256_fmadd_ps(
                                x,
                                _mm256_set1_ps(dy),
                                acc[oc],
                            );
                    }
                }
            }

            /*
             * One store per dW vector.
             */
            for oc in 0..OC {
                _mm256_storeu_ps(
                    weight_grad.add(
                        (oc_base + oc)
                            * kernel_width
                            + weight_k_offset
                            + c
                    ),
                    acc[oc],
                );
            }

            c += IC_SIMD;
        }

        /*
         * Scalar IC remainder.
         */
        let mut c =
            simd_end;

        while c < in_channels {
            let mut acc =
                [0.0f32; OC_TILE];

            for batch in 0..batch_size {
                let input_batch =
                    input.add(
                        batch
                            * input_length
                            * in_channels
                    );

                let grad_batch =
                    grad.add(
                        batch
                            * output_length
                            * out_channels
                    );

                let activation_batch =
                    activation.add(
                        batch
                            * output_length
                            * out_channels
                    );

                for output_position in 0..output_length {
                    let span =
                        spans[output_position];

                    if k < span.k_start
                        || k >= span.k_end
                    {
                        continue;
                    }

                    let src_position =
                        (span.origin
                            + k as isize)
                            as usize;

                    let x =
                        *input_batch.add(
                            src_position
                                * in_channels
                                + c
                        );

                    let dy_ptr =
                        grad_batch.add(
                            output_position
                                * out_channels
                                + oc_base
                        );

                    let activation_ptr =
                        activation_batch.add(
                            output_position
                                * out_channels
                                + oc_base
                        );

                    for oc in 0..OC {
                        let mut dy =
                            *dy_ptr.add(oc);

                        if LEAKY {
                            dy =
                                apply_leaky_scalar(
                                    dy,
                                    *activation_ptr.add(oc),
                                    activation_slope,
                                );
                        }

                        acc[oc] +=
                            x * dy;
                    }
                }
            }

            for oc in 0..OC {
                *weight_grad.add(
                    (oc_base + oc)
                        * kernel_width
                        + weight_k_offset
                        + c
                ) =
                    acc[oc];
            }

            c += 1;
        }
    }
}


/* ============================================================
 * dB
 *
 * Output-channel dimension is contiguous.
 *
 * Linear:
 *     dB += dY
 *
 * Leaky:
 *     dB += dY * derivative(Y)
 *
 * The activation derivative is vectorized.
 * ============================================================
 */

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dbias_block<const LEAKY: bool>(
    grad: *const f32,
    activation: *const f32,

    bias_grad: *mut f32,

    batch_size: usize,
    output_length: usize,
    out_channels: usize,

    oc_base: usize,

    activation_slope: f32,
) {
    let rows =
        batch_size * output_length;

    let mut acc =
        _mm256_setzero_ps();

    if LEAKY {
        let slope =
            _mm256_set1_ps(
                activation_slope
            );

        let one =
            _mm256_set1_ps(1.0);

        let zero =
            _mm256_setzero_ps();

        for row in 0..rows {
            let dy =
                _mm256_loadu_ps(
                    grad.add(
                        row * out_channels
                            + oc_base
                    )
                );

            let y =
                _mm256_loadu_ps(
                    activation.add(
                        row * out_channels
                            + oc_base
                    )
                );

            let mask =
                _mm256_cmp_ps(
                    y,
                    zero,
                    _CMP_LE_OQ,
                );

            let multiplier =
                _mm256_blendv_ps(
                    one,
                    slope,
                    mask,
                );

            let dy =
                _mm256_mul_ps(
                    dy,
                    multiplier,
                );

            acc =
                _mm256_add_ps(
                    acc,
                    dy,
                );
        }
    } else {
        for row in 0..rows {
            acc =
                _mm256_add_ps(
                    acc,
                    _mm256_loadu_ps(
                        grad.add(
                            row * out_channels
                                + oc_base
                        )
                    ),
                );
        }
    }

    _mm256_storeu_ps(
        bias_grad.add(oc_base),
        acc,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dbias_tail<const LEAKY: bool>(
    grad: *const f32,
    activation: *const f32,

    bias_grad: *mut f32,

    batch_size: usize,
    output_length: usize,
    out_channels: usize,

    oc_base: usize,
    count: usize,

    activation_slope: f32,
) {
    let rows =
        batch_size * output_length;

    for oc in 0..count {
        let mut sum =
            0.0f32;

        for row in 0..rows {
            let mut dy =
                *grad.add(
                    row * out_channels
                        + oc_base
                        + oc
                );

            if LEAKY {
                dy =
                    apply_leaky_scalar(
                        dy,
                        *activation.add(
                            row * out_channels
                                + oc_base
                                + oc
                        ),
                        activation_slope,
                    );
            }

            sum += dy;
        }

        *bias_grad.add(
            oc_base + oc
        ) = sum;
    }
}


/* ============================================================
 * AVX2 backward
 * ============================================================
 */

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn backward_avx2<const LEAKY: bool>(
    input: &[f32],
    grad: &[f32],
    activation: &[f32],

    input_grad: &mut [f32],
    weight_grad: &mut [f32],
    bias_grad: &mut [f32],

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

    activation_slope: f32,
) {
    let spans =
        build_spans(
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            causal,
        );

    let input_ptr =
        input.as_ptr();

    let grad_ptr =
        grad.as_ptr();

    let activation_ptr =
        activation.as_ptr();

    let input_grad_ptr =
        input_grad.as_mut_ptr();

    let weight_grad_ptr =
        weight_grad.as_mut_ptr();

    let bias_grad_ptr =
        bias_grad.as_mut_ptr();

    /*
     * ========================================================
     * dB
     * ========================================================
     */

    let full_oc =
        out_channels / OC_TILE;

    for block in 0..full_oc {
        dbias_block::<LEAKY>(
            grad_ptr,
            activation_ptr,

            bias_grad_ptr,

            batch_size,
            output_length,
            out_channels,

            block * OC_TILE,

            activation_slope,
        );
    }

    let oc_tail =
        out_channels % OC_TILE;

    if oc_tail != 0 {
        dbias_tail::<LEAKY>(
            grad_ptr,
            activation_ptr,

            bias_grad_ptr,

            batch_size,
            output_length,
            out_channels,

            full_oc * OC_TILE,
            oc_tail,

            activation_slope,
        );
    }

    /*
     * ========================================================
     * dW
     * ========================================================
     */

    for block in 0..full_oc {
        dweight::<OC_TILE, LEAKY>(
            input_ptr,
            grad_ptr,
            activation_ptr,

            weight_grad_ptr,

            batch_size,
            input_length,
            output_length,

            in_channels,
            out_channels,

            kernel_size,

            block * OC_TILE,

            &spans,

            activation_slope,
        );
    }

    /*
     * Output-channel tail.
     */
    if oc_tail != 0 {
        let oc_base =
            full_oc * OC_TILE;

        match oc_tail {
            1 => dweight::<1, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            2 => dweight::<2, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            3 => dweight::<3, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            4 => dweight::<4, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            5 => dweight::<5, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            6 => dweight::<6, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            7 => dweight::<7, LEAKY>(
                input_ptr,
                grad_ptr,
                activation_ptr,

                weight_grad_ptr,

                batch_size,
                input_length,
                output_length,

                in_channels,
                out_channels,

                kernel_size,
                oc_base,

                &spans,

                activation_slope,
            ),

            _ => unreachable!(),
        }
    }

    /*
     * ========================================================
     * dX
     * ========================================================
     *
     * Forward-like traversal:
     *
     * batch
     *   output position
     *     OC tile
     *       valid K
     *         SIMD IC
     */

    for batch in 0..batch_size {
        let input_grad_batch =
            input_grad_ptr.add(
                batch
                    * input_length
                    * in_channels
            );

        let grad_batch =
            grad_ptr.add(
                batch
                    * output_length
                    * out_channels
            );

        let activation_batch =
            activation_ptr.add(
                batch
                    * output_length
                    * out_channels
            );

        for output_position in 0..output_length {
            let span =
                spans[output_position];

            if span.k_start == span.k_end {
                continue;
            }

            let grad_row =
                grad_batch.add(
                    output_position
                        * out_channels
                );

            let activation_row =
                activation_batch.add(
                    output_position
                        * out_channels
                );

            for block in 0..full_oc {
                dinput::<OC_TILE, LEAKY>(
                    grad_row.add(
                        block * OC_TILE
                    ),

                    activation_row.add(
                        block * OC_TILE
                    ),

                    weights.as_ptr().add(
                        block
                            * OC_TILE
                            * kernel_size
                            * in_channels
                    ),

                    input_grad_batch,

                    in_channels,
                    kernel_size,

                    span,

                    activation_slope,
                );
            }

            if oc_tail != 0 {
                let oc_base =
                    full_oc * OC_TILE;

                let weights_base =
                    oc_base
                        * kernel_size
                        * in_channels;

                match oc_tail {
                    1 => dinput::<1, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    2 => dinput::<2, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    3 => dinput::<3, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    4 => dinput::<4, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    5 => dinput::<5, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    6 => dinput::<6, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    7 => dinput::<7, LEAKY>(
                        grad_row.add(oc_base),
                        activation_row.add(oc_base),

                        weights.as_ptr()
                            .add(weights_base),

                        input_grad_batch,

                        in_channels,
                        kernel_size,

                        span,

                        activation_slope,
                    ),

                    _ => unreachable!(),
                }
            }
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
    activation: &[f32],

    input_grad: &mut [f32],
    weight_grad: &mut [f32],
    bias_grad: &mut [f32],

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

    activation_mode: &Activation,
) {
    let kernel_width =
        kernel_size * in_channels;

    for batch in 0..batch_size {
        for output_position in 0..output_length {
            let span =
                make_span(
                    output_position,
                    input_length,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                );

            let grad_base =
                (batch * output_length
                    + output_position)
                    * out_channels;

            for oc in 0..out_channels {
                let mut dy =
                    grad[grad_base + oc];

                /*
                 * Match forward exactly:
                 *
                 * if y <= 0.0 { y *= slope; }
                 *
                 * The derivative is therefore slope for
                 * activated output <= 0.
                 */
                match activation_mode {
                    Activation::None => {}

                    Activation::LeakyReLU { slope } => {
                        dy =
                            apply_leaky_scalar(
                                dy,
                                activation[
                                    grad_base + oc
                                    ],
                                *slope,
                            );
                    }
                }

                bias_grad[oc] +=
                    dy;

                for k in span.k_start..span.k_end {
                    let src_position =
                        (span.origin
                            + k as isize)
                            as usize;

                    let input_base =
                        (batch * input_length
                            + src_position)
                            * in_channels;

                    let weight_base =
                        oc * kernel_width
                            + k * in_channels;

                    for c in 0..in_channels {
                        weight_grad[
                            weight_base + c
                            ] +=
                            dy
                                * input[
                                input_base + c
                                ];

                        input_grad[
                            input_base + c
                            ] +=
                            dy
                                * weights[
                                weight_base + c
                                ];
                    }
                }
            }
        }
    }
}


/* ============================================================
 * Public entry point
 *
 * `grad` is dL/d(output_of_activation).
 *
 * `activation_output` is the output produced by the forward
 * Conv1D activation. It is only used for LeakyReLU.
 * ============================================================
 */

pub fn backward_direct(
    input: &[f32],
    grad: &[f32],
    activation_output: &[f32],

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
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    assert_eq!(
        input.len(),
        batch_size
            * input_length
            * in_channels
    );

    assert_eq!(
        grad.len(),
        batch_size
            * output_length
            * out_channels
    );

    assert_eq!(
        activation_output.len(),
        batch_size
            * output_length
            * out_channels
    );

    assert_eq!(
        weights.len(),
        out_channels
            * kernel_size
            * in_channels
    );

    let mut input_grad =
        vec![0.0f32; input.len()];

    let mut weight_grad =
        vec![0.0f32; weights.len()];

    let mut bias_grad =
        vec![0.0f32; out_channels];

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            match activation {
                /*
                 * IMPORTANT:
                 *
                 * The const generic means the compiler can completely
                 * remove every activation-related operation from the
                 * normal Conv1D path.
                 */
                Activation::None => unsafe {
                    backward_avx2::<false>(
                        input,
                        grad,
                        activation_output,

                        &mut input_grad,
                        &mut weight_grad,
                        &mut bias_grad,

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

                        0.0,
                    );
                },

                Activation::LeakyReLU { slope } => unsafe {
                    backward_avx2::<true>(
                        input,
                        grad,
                        activation_output,

                        &mut input_grad,
                        &mut weight_grad,
                        &mut bias_grad,

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

                        *slope,
                    );
                },
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

        in_channels,
        out_channels,

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
