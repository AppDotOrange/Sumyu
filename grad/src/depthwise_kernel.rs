//! Direct AVX2/FMA Depthwise Conv1D forward kernel.
//!
//! Parameter layout:
//!
//!     weights[c * kernel_size + k]
//!
//! The kernel internally packs this as:
//!
//!     packed_weights[k * in_channels + c]
//!
//! Tensor layout:
//!
//!     [batch][position][channel]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_max_ps,
    _mm256_min_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_setzero_ps,
    _mm256_storeu_ps,
};

use crate::neuron::{
    Activation,
    DepthwiseConv1DLayer,
};

use rayon::prelude::*;
use rayon::ThreadPool;

const SIMD_WIDTH: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_position_avx2<
    const LEAKY: bool,
>(
    layer: &DepthwiseConv1DLayer,
    input: &[f32],
    packed_weights: &[f32],
    biases: &[f32],
    output: &mut [f32],
    out_pos: usize,
    input_length: usize,
    slope: f32,
) {
    let channels =
        layer.in_channels;

    let kernel_size =
        layer.kernel_size;

    let base =
        out_pos * layer.stride;

    let (
        k_start,
        k_end,
    ) = if layer.causal {
        let left =
            kernel_size - 1;

        if base >= left {
            (
                0,
                kernel_size.min(
                    input_length
                        + left
                        - base,
                ),
            )
        } else {
            (
                left - base,
                kernel_size,
            )
        }
    } else {
        let base =
            base as isize;

        let padding =
            layer.padding as isize;

        let first_valid =
            padding - base;

        let last_valid_exclusive =
            input_length as isize
                + padding
                - base;

        let k_start =
            if first_valid <= 0 {
                0
            } else {
                first_valid as usize
            };

        let k_end =
            if last_valid_exclusive
                <= 0
            {
                0
            } else if last_valid_exclusive
                >= kernel_size as isize
            {
                kernel_size
            } else {
                last_valid_exclusive
                    as usize
            };

        (
            k_start.min(kernel_size),
            k_end.min(kernel_size),
        )
    };

    let output_base =
        out_pos * channels;

    let zero =
        _mm256_setzero_ps();

    let slope_vec =
        if LEAKY {
            _mm256_set1_ps(slope)
        } else {
            zero
        };

    let mut c = 0usize;

    while c + SIMD_WIDTH <= channels {
        let mut acc =
            _mm256_loadu_ps(
                biases.as_ptr()
                    .add(c)
            );

        for k in k_start..k_end {
            let src_pos =
                if layer.causal {
                    base + k
                        - (kernel_size - 1)
                } else {
                    base + k
                        - layer.padding
                };

            let input_index =
                src_pos * channels
                    + c;

            let weight_index =
                k * channels
                    + c;

            let x =
                _mm256_loadu_ps(
                    input.as_ptr()
                        .add(input_index)
                );

            let w =
                _mm256_loadu_ps(
                    packed_weights
                        .as_ptr()
                        .add(weight_index)
                );

            acc =
                _mm256_fmadd_ps(
                    x,
                    w,
                    acc,
                );
        }

        if LEAKY {
            let positive =
                _mm256_max_ps(
                    acc,
                    zero,
                );

            let negative =
                _mm256_min_ps(
                    acc,
                    zero,
                );

            acc =
                _mm256_add_ps(
                    positive,
                    _mm256_mul_ps(
                        negative,
                        slope_vec,
                    ),
                );
        }

        _mm256_storeu_ps(
            output
                .as_mut_ptr()
                .add(
                    output_base + c
                ),
            acc,
        );

        c +=
            SIMD_WIDTH;
    }

    while c < channels {
        let mut sum =
            biases[c];

        for k in k_start..k_end {
            let src_pos =
                if layer.causal {
                    base + k
                        - (kernel_size - 1)
                } else {
                    base + k
                        - layer.padding
                };

            let input_index =
                src_pos * channels
                    + c;

            let weight_index =
                k * channels
                    + c;

            sum +=
                input[input_index]
                    * packed_weights[
                    weight_index
                    ];
        }

        let value =
            if LEAKY {
                if sum <= 0.0 {
                    sum * slope
                } else {
                    sum
                }
            } else {
                sum
            };

        output[
            output_base + c
            ] = value;

        c += 1;
    }
}

#[inline]
fn depthwise_position_scalar<
    const LEAKY: bool,
>(
    layer: &DepthwiseConv1DLayer,
    input: &[f32],
    packed_weights: &[f32],
    biases: &[f32],
    output: &mut [f32],
    out_pos: usize,
    input_length: usize,
    slope: f32,
) {
    let channels =
        layer.in_channels;

    let kernel_size =
        layer.kernel_size;

    let base =
        out_pos
            * layer.stride;

    let output_base =
        out_pos * channels;

    for c in 0..channels {
        let mut sum =
            biases[c];

        for k in 0..kernel_size {
            let src_pos =
                if layer.causal {
                    base as isize
                        + k as isize
                        - (
                        kernel_size
                            - 1
                    ) as isize
                } else {
                    base as isize
                        + k as isize
                        - layer.padding
                        as isize
                };

            if src_pos >= 0
                && (src_pos as usize)
                < input_length
            {
                let input_index =
                    src_pos as usize
                        * channels
                        + c;

                let weight_index =
                    k * channels
                        + c;

                sum +=
                    input[
                        input_index
                        ]
                        * packed_weights[
                        weight_index
                        ];
            }
        }

        let value =
            if LEAKY {
                if sum <= 0.0 {
                    sum * slope
                } else {
                    sum
                }
            } else {
                sum
            };

        output[
            output_base + c
            ] = value;
    }
}

fn depthwise_conv1d_forward_impl<
    const LEAKY: bool,
>(
    layer: &DepthwiseConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    weights: &[f32],
    biases: &[f32],
    slope: f32,
    thread_pool: &ThreadPool,
) -> Vec<f32> {
    let channels =
        layer.in_channels;

    let kernel_size =
        layer.kernel_size;

    assert!(channels > 0);
    assert!(kernel_size > 0);
    assert!(batch_size > 0);
    assert!(input_length > 0);

    assert_eq!(
        input.len(),
        batch_size
            * input_length
            * channels,
        "DepthwiseConv1D input length mismatch",
    );

    assert_eq!(
        output_length,
        layer.output_length(
            input_length
        ),
        "DepthwiseConv1D output length mismatch",
    );

    assert_eq!(
        weights.len(),
        channels * kernel_size,
        "DepthwiseConv1D weight count mismatch",
    );

    assert_eq!(
        biases.len(),
        channels,
        "DepthwiseConv1D bias count mismatch",
    );

    let mut packed_weights =
        vec![
            0.0f32;
            channels * kernel_size
        ];

    for c in 0..channels {
        let src_base =
            c * kernel_size;

        for k in 0..kernel_size {
            packed_weights[
                k * channels + c
                ] =
                weights[
                    src_base + k
                    ];
        }
    }

    let mut output =
        vec![
            0.0f32;
            batch_size
                * output_length
                * channels
        ];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!(
            "avx2"
        ) && is_x86_feature_detected!(
            "fma"
        ) {
            if thread_pool
                .current_num_threads()
                == 1
                || batch_size == 1
            {
                for b in 0..batch_size {
                    let input_batch_start =
                        b
                            * input_length
                            * channels;

                    let input_batch_end =
                        input_batch_start
                            + input_length
                            * channels;

                    let output_batch_start =
                        b
                            * output_length
                            * channels;

                    let output_batch_end =
                        output_batch_start
                            + output_length
                            * channels;

                    let input_batch =
                        &input[
                            input_batch_start
                                ..input_batch_end
                            ];

                    let output_batch =
                        &mut output[
                            output_batch_start
                                ..output_batch_end
                            ];

                    for out_pos
                    in 0..output_length
                    {
                        unsafe {
                            depthwise_position_avx2::<LEAKY>(
                                layer,
                                input_batch,
                                &packed_weights,
                                biases,
                                output_batch,
                                out_pos,
                                input_length,
                                slope,
                            );
                        }
                    }
                }
            } else {
                thread_pool.install(
                    || {
                        output
                            .par_chunks_mut(
                                output_length
                                    * channels,
                            )
                            .zip(
                                input.par_chunks(
                                    input_length
                                        * channels,
                                ),
                            )
                            .for_each(
                                |(
                                     output_batch,
                                     input_batch,
                                 )| {
                                    for out_pos
                                    in 0..output_length
                                    {
                                        unsafe {
                                            depthwise_position_avx2::<LEAKY>(
                                                layer,
                                                input_batch,
                                                &packed_weights,
                                                biases,
                                                output_batch,
                                                out_pos,
                                                input_length,
                                                slope,
                                            );
                                        }
                                    }
                                },
                            );
                    },
                );
            }

            return output;
        }
    }

    if thread_pool
        .current_num_threads()
        == 1
        || batch_size == 1
    {
        for b in 0..batch_size {
            let input_batch_start =
                b
                    * input_length
                    * channels;

            let input_batch_end =
                input_batch_start
                    + input_length
                    * channels;

            let output_batch_start =
                b
                    * output_length
                    * channels;

            let output_batch_end =
                output_batch_start
                    + output_length
                    * channels;

            let input_batch =
                &input[
                    input_batch_start
                        ..input_batch_end
                    ];

            let output_batch =
                &mut output[
                    output_batch_start
                        ..output_batch_end
                    ];

            for out_pos
            in 0..output_length
            {
                depthwise_position_scalar::<LEAKY>(
                    layer,
                    input_batch,
                    &packed_weights,
                    biases,
                    output_batch,
                    out_pos,
                    input_length,
                    slope,
                );
            }
        }
    } else {
        thread_pool.install(
            || {
                output
                    .par_chunks_mut(
                        output_length
                            * channels,
                    )
                    .zip(
                        input.par_chunks(
                            input_length
                                * channels,
                        ),
                    )
                    .for_each(
                        |(
                             output_batch,
                             input_batch,
                         )| {
                            for out_pos
                            in 0..output_length
                            {
                                depthwise_position_scalar::<LEAKY>(
                                    layer,
                                    input_batch,
                                    &packed_weights,
                                    biases,
                                    output_batch,
                                    out_pos,
                                    input_length,
                                    slope,
                                );
                            }
                        },
                    );
            },
        );
    }

    output
}

pub fn depthwise_conv1d_forward(
    layer: &DepthwiseConv1DLayer,
    weights: &[f32],
    biases: &[f32],
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    thread_pool: &ThreadPool,
) -> Vec<f32> {
    match &layer.activation {
        Activation::None => {
            depthwise_conv1d_forward_impl::<false>(
                layer,
                input,
                batch_size,
                input_length,
                output_length,
                weights,
                biases,
                0.0,
                thread_pool,
            )
        }

        Activation::LeakyReLU {
            slope,
        } => {
            depthwise_conv1d_forward_impl::<true>(
                layer,
                input,
                batch_size,
                input_length,
                output_length,
                weights,
                biases,
                *slope,
                thread_pool,
            )
        }
    }
}
