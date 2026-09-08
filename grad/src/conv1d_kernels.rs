use crate::neuron::{Activation, Conv1DLayer};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_setzero_ps,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps,
    _mm_add_ss,
    _mm_cvtss_f32,
    _mm_movehl_ps,
    _mm_shuffle_ps,
    _mm256_castps256_ps128,
    _mm256_extractf128_ps,
};

const OC_BLOCK: usize = 8;
const SIMD_WIDTH: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(x: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(x);
    let hi = _mm256_extractf128_ps(x, 1);

    let x = _mm_add_ps(lo, hi);

    let x2 = _mm_movehl_ps(x, x);
    let x = _mm_add_ps(x, x2);

    let x2 = _mm_shuffle_ps(x, x, 0x01);
    let x = _mm_add_ss(x, x2);

    _mm_cvtss_f32(x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_block<const OC: usize, const LEAKY: bool>(
    input: *const f32,
    weights: *const f32,
    biases: *const f32,
    output: *mut f32,

    in_channels: usize,
    kernel_size: usize,

    k_start: usize,
    k_end: usize,

    activation_slope: f32,
) { unsafe {
    debug_assert!(OC >= 1);
    debug_assert!(OC <= OC_BLOCK);
    debug_assert!(k_start < k_end);
    debug_assert!(k_end <= kernel_size);

    //
    // Each accumulator contains only the SIMD-vectorized
    // contribution.
    //
    let mut acc =
        [_mm256_setzero_ps(); OC];

    //
    // Scalar remainder contributions must NOT be broadcast into
    // the AVX accumulator, because horizontal reduction would then
    // count them once per lane.
    //
    let mut scalar_tail =
        [0.0f32; OC];

    let kernel_width =
        kernel_size * in_channels;

    //
    // `input` points to the input corresponding to k_start.
    //
    // Thus:
    //
    //     k == k_start
    //         -> input + 0
    //
    //     k == k_start + 1
    //         -> input + in_channels
    //
    // Weights use the absolute kernel position.
    //
    for k in k_start..k_end {
        let input_k =
            input.add(
                (k - k_start) * in_channels
            );

        let weight_k =
            k * in_channels;

        let mut c = 0usize;

        //
        // Main 32-input-channel SIMD body.
        //
        while c + 32 <= in_channels {
            let x0 =
                _mm256_loadu_ps(
                    input_k.add(c)
                );

            let x1 =
                _mm256_loadu_ps(
                    input_k.add(c + 8)
                );

            let x2 =
                _mm256_loadu_ps(
                    input_k.add(c + 16)
                );

            let x3 =
                _mm256_loadu_ps(
                    input_k.add(c + 24)
                );

            for oc in 0..OC {
                let weight_base =
                    oc * kernel_width
                        + weight_k
                        + c;

                let w0 =
                    _mm256_loadu_ps(
                        weights.add(weight_base)
                    );

                let w1 =
                    _mm256_loadu_ps(
                        weights.add(weight_base + 8)
                    );

                let w2 =
                    _mm256_loadu_ps(
                        weights.add(weight_base + 16)
                    );

                let w3 =
                    _mm256_loadu_ps(
                        weights.add(weight_base + 24)
                    );

                acc[oc] =
                    _mm256_fmadd_ps(
                        x0,
                        w0,
                        acc[oc],
                    );

                acc[oc] =
                    _mm256_fmadd_ps(
                        x1,
                        w1,
                        acc[oc],
                    );

                acc[oc] =
                    _mm256_fmadd_ps(
                        x2,
                        w2,
                        acc[oc],
                    );

                acc[oc] =
                    _mm256_fmadd_ps(
                        x3,
                        w3,
                        acc[oc],
                    );
            }

            c += 32;
        }

        //
        // Remaining complete SIMD vectors.
        //
        while c + SIMD_WIDTH <= in_channels {
            let x =
                _mm256_loadu_ps(
                    input_k.add(c)
                );

            for oc in 0..OC {
                let weight_offset =
                    oc * kernel_width
                        + weight_k
                        + c;

                let w =
                    _mm256_loadu_ps(
                        weights.add(weight_offset)
                    );

                acc[oc] =
                    _mm256_fmadd_ps(
                        x,
                        w,
                        acc[oc],
                    );
            }

            c += SIMD_WIDTH;
        }

        //
        // Scalar input-channel remainder.
        //
        // IMPORTANT:
        //
        // Do NOT broadcast these contributions into `acc`.
        // They would otherwise be counted 8× by hsum_avx2().
        //
        while c < in_channels {
            let x =
                *input_k.add(c);

            for oc in 0..OC {
                let weight_offset =
                    oc * kernel_width
                        + weight_k
                        + c;

                let w =
                    *weights.add(weight_offset);

                scalar_tail[oc] +=
                    x * w;
            }

            c += 1;
        }
    }

    //
    // Final reduction.
    //
    // SIMD contribution + scalar remainder + bias.
    //
    for oc in 0..OC {
        let mut y =
            hsum_avx2(acc[oc])
                + scalar_tail[oc]
                + *biases.add(oc);

        if LEAKY && y <= 0.0 {
            y *= activation_slope;
        }

        *output.add(oc) = y;
    }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn forward_avx2<const LEAKY: bool>(
    layer: &Conv1DLayer,

    input: &[f32],
    output: &mut [f32],

    batch_size: usize,
    input_length: usize,
    output_length: usize,

    weights: &[f32],
    biases: &[f32],

    activation_slope: f32,
) {
    let in_channels =
        layer.in_channels;

    let out_channels =
        layer.out_channels;

    let kernel_size =
        layer.kernel_size;

    let stride =
        layer.stride;

    let input_batch_stride =
        input_length * in_channels;

    let output_batch_stride =
        output_length * out_channels;

    let kernel_width =
        kernel_size * in_channels;

    let left_padding =
        if layer.causal {
            kernel_size - 1
        } else {
            layer.padding
        };

    for batch in 0..batch_size {
        let input_batch_base =
            batch * input_batch_stride;

        let output_batch_base =
            batch * output_batch_stride;

        for out_pos in 0..output_length {
            //
            // Input coordinate of kernel element 0.
            //
            let origin =
                (out_pos * stride) as isize
                    - left_padding as isize;

            //
            // Valid kernel positions satisfy:
            //
            //     0 <= origin + k < input_length
            //
            // Therefore:
            //
            //     k >= -origin
            //
            //     k < input_length - origin
            //
            let k_start =
                (-origin)
                    .max(0)
                    .min(kernel_size as isize)
                    as usize;

            let k_end =
                (input_length as isize - origin)
                    .clamp(
                        0,
                        kernel_size as isize,
                    ) as usize;

            let output_row =
                output.as_mut_ptr().add(
                    output_batch_base
                        + out_pos * out_channels
                );

            //
            // Entire kernel is padding.
            //
            // The convolution is therefore only the bias.
            //
            if k_start >= k_end {
                for oc in 0..out_channels {
                    let mut y =
                        biases[oc];

                    if LEAKY && y <= 0.0 {
                        y *= activation_slope;
                    }

                    *output_row.add(oc) = y;
                }

                continue;
            }

            //
            // Since k_start < k_end, this coordinate is guaranteed
            // to be inside the input.
            //
            let input_start_pos =
                (origin + k_start as isize)
                    as usize;

            debug_assert!(
                input_start_pos < input_length
            );

            let input_start =
                input_batch_base
                    + input_start_pos * in_channels;

            let mut out_c = 0usize;

            //
            // Full output-channel blocks.
            //
            while out_c + OC_BLOCK <= out_channels {
                conv_block::<OC_BLOCK, LEAKY>(
                    input.as_ptr().add(
                        input_start
                    ),

                    weights.as_ptr().add(
                        out_c * kernel_width
                    ),

                    biases.as_ptr().add(
                        out_c
                    ),

                    output_row.add(
                        out_c
                    ),

                    in_channels,
                    kernel_size,

                    k_start,
                    k_end,

                    activation_slope,
                );

                out_c += OC_BLOCK;
            }

            //
            // Output-channel tail.
            //
            match out_channels - out_c {
                0 => {}

                1 => {
                    conv_block::<1, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                2 => {
                    conv_block::<2, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                3 => {
                    conv_block::<3, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                4 => {
                    conv_block::<4, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                5 => {
                    conv_block::<5, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                6 => {
                    conv_block::<6, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                7 => {
                    conv_block::<7, LEAKY>(
                        input.as_ptr().add(input_start),
                        weights.as_ptr().add(
                            out_c * kernel_width
                        ),
                        biases.as_ptr().add(out_c),
                        output_row.add(out_c),
                        in_channels,
                        kernel_size,
                        k_start,
                        k_end,
                        activation_slope,
                    );
                }

                _ => unreachable!(),
            }
        }
    }
}

fn forward_scalar(
    layer: &Conv1DLayer,

    input: &[f32],
    output: &mut [f32],

    batch_size: usize,
    input_length: usize,
    output_length: usize,

    weights: &[f32],
    biases: &[f32],
) {
    let in_channels =
        layer.in_channels;

    let out_channels =
        layer.out_channels;

    let kernel_size =
        layer.kernel_size;

    let stride =
        layer.stride;

    let input_batch_stride =
        input_length * in_channels;

    let output_batch_stride =
        output_length * out_channels;

    let kernel_width =
        kernel_size * in_channels;

    let left_padding =
        if layer.causal {
            kernel_size - 1
        } else {
            layer.padding
        };

    for batch in 0..batch_size {
        for out_pos in 0..output_length {
            let origin =
                (out_pos * stride) as isize
                    - left_padding as isize;

            for oc in 0..out_channels {
                let mut sum =
                    biases[oc];

                let weight_base =
                    oc * kernel_width;

                for k in 0..kernel_size {
                    let src_pos =
                        origin + k as isize;

                    if src_pos < 0
                        || src_pos >= input_length as isize
                    {
                        continue;
                    }

                    let input_base =
                        batch * input_batch_stride
                            + src_pos as usize
                            * in_channels;

                    let weight_base_k =
                        weight_base
                            + k * in_channels;

                    for c in 0..in_channels {
                        sum +=
                            input[input_base + c]
                                * weights[weight_base_k + c];
                    }
                }

                let mut y =
                    sum;

                match &layer.activation {
                    Activation::None => {}

                    Activation::LeakyReLU { slope } => {
                        if y <= 0.0 {
                            y *= *slope;
                        }
                    }
                }

                output[
                    batch * output_batch_stride
                        + out_pos * out_channels
                        + oc
                    ] = y;
            }
        }
    }
}

pub fn conv1d_forward(
    layer: &Conv1DLayer,

    input: &[f32],

    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let in_channels =
        layer.in_channels;

    let out_channels =
        layer.out_channels;

    let kernel_size =
        layer.kernel_size;

    let weight_count =
        out_channels
            * kernel_size
            * in_channels;

    debug_assert_eq!(
        input.len(),
        batch_size
            * input_length
            * in_channels
    );

    debug_assert_eq!(
        layer.weight_handles.len(),
        weight_count
    );

    debug_assert_eq!(
        layer.bias_handles.len(),
        out_channels
    );

    let mut weights =
        vec![0.0f32; weight_count];

    let mut biases =
        vec![0.0f32; out_channels];

    crate::handle_data_slice(
        &layer.weight_handles,
        &mut weights,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let output_len =
        batch_size
            * output_length
            * out_channels;

    let mut output =
        vec![0.0f32; output_len];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            match &layer.activation {
                Activation::None => unsafe {
                    forward_avx2::<false>(
                        layer,
                        input,
                        &mut output,
                        batch_size,
                        input_length,
                        output_length,
                        &weights,
                        &biases,
                        0.0,
                    );
                },

                Activation::LeakyReLU { slope } => unsafe {
                    forward_avx2::<true>(
                        layer,
                        input,
                        &mut output,
                        batch_size,
                        input_length,
                        output_length,
                        &weights,
                        &biases,
                        *slope,
                    );
                },
            }

            return output;
        }
    }

    forward_scalar(
        layer,
        input,
        &mut output,
        batch_size,
        input_length,
        output_length,
        &weights,
        &biases,
    );

    output
}
