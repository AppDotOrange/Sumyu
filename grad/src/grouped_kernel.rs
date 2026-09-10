use std::arch::is_x86_feature_detected;

use crate::backwards::make_conv_positions;
use crate::neuron::{Activation, GroupedConv1DLayer};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps,
    _mm256_blendv_ps,
    _mm256_cmp_ps,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_setr_epi32,
    _mm256_setzero_ps,
    _mm256_storeu_ps,
    _mm256_i32gather_ps,
    _CMP_LE_OQ,
};

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn grouped_conv1d_direct_avx2(
    layer: &GroupedConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    weights: &[f32],
    biases: &[f32],
    positions: &[i32],
    output: &mut [f32],
) {
    let group_in = layer.in_channels / layer.groups;
    let group_out = layer.out_channels / layer.groups;
    let kernel_width = group_in * layer.kernel_size;

    for b in 0..batch_size {
        let input_batch_base = b * input_length * layer.in_channels;
        let output_batch_base = b * output_length * layer.out_channels;

        for group in 0..layer.groups {
            let input_group_base = group * group_in;
            let output_group_base = group * group_out;
            let weight_group_base = group * group_out * kernel_width;
            let bias_group_base = group * group_out;

            for out_pos in 0..output_length {
                let position_base = out_pos * layer.kernel_size;
                let output_base =
                    output_batch_base + out_pos * layer.out_channels + output_group_base;

                // The stored weights are laid out as:
                //
                // [group][output_channel][kernel_element]
                //
                // so eight adjacent output channels are strided by kernel_width.
                // AVX2 gather lets us evaluate those eight channels together without
                // transposing/packing the weights first.
                let mut oc = 0usize;
                if group_out >= 8 {
                    while oc + 8 <= group_out {
                        let mut acc = _mm256_setzero_ps();

                        // Offsets for output channels oc..oc+7 inside one group's
                        // weight matrix, measured in f32 elements.
                        let base =
                            oc * kernel_width;

                        let weight_indices =
                            _mm256_setr_epi32(
                                base as i32,
                                (base + kernel_width) as i32,
                                (base + 2 * kernel_width) as i32,
                                (base + 3 * kernel_width) as i32,
                                (base + 4 * kernel_width) as i32,
                                (base + 5 * kernel_width) as i32,
                                (base + 6 * kernel_width) as i32,
                                (base + 7 * kernel_width) as i32,
                            );

                        for k in 0..layer.kernel_size {
                            let src_pos = positions[position_base + k];
                            if src_pos < 0 || src_pos as usize >= input_length {
                                continue;
                            }

                            let src_base =
                                input_batch_base
                                    + src_pos as usize * layer.in_channels
                                    + input_group_base;

                            // Each input channel contributes to all eight output
                            // channels. Broadcast the scalar input value and gather
                            // the eight corresponding weights.
                            for ic in 0..group_in {
                                let x = _mm256_set1_ps(input[src_base + ic]);
                                let weight_ptr = weights.as_ptr()
                                    .add(weight_group_base + k * group_in + ic);
                                let w = _mm256_i32gather_ps(
                                    weight_ptr,
                                    weight_indices,
                                    4,
                                );
                                acc = _mm256_fmadd_ps(x, w, acc);
                            }
                        }

                        acc = _mm256_add_ps(
                            acc,
                            _mm256_loadu_ps(
                                biases.as_ptr().add(bias_group_base + oc),
                            ),
                        );

                        match &layer.activation {
                            Activation::LeakyReLU { slope } => {
                                let zero = _mm256_setzero_ps();
                                let slope_v = _mm256_set1_ps(*slope);
                                let mask = _mm256_cmp_ps(acc, zero, _CMP_LE_OQ);
                                let negative = _mm256_mul_ps(acc, slope_v);
                                acc = _mm256_blendv_ps(acc, negative, mask);
                            }
                            _ => {
                                let mut tmp = [0.0f32; 8];
                                _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                                for lane in &mut tmp {
                                    *lane = layer.activation.apply(*lane);
                                }
                                acc = _mm256_loadu_ps(tmp.as_ptr());
                            }
                        }

                        _mm256_storeu_ps(
                            output.as_mut_ptr().add(output_base + oc),
                            acc,
                        );

                        oc += 8;
                    }
                }

                while oc < group_out {
                    let mut sum = biases[bias_group_base + oc];

                    for k in 0..layer.kernel_size {
                        let src_pos = positions[position_base + k];
                        if src_pos < 0 || src_pos as usize >= input_length {
                            continue;
                        }

                        let src =
                            input_batch_base
                                + src_pos as usize * layer.in_channels
                                + input_group_base;
                        let weight_base =
                            weight_group_base
                                + oc * kernel_width
                                + k * group_in;

                        for ic in 0..group_in {
                            sum += input[src + ic]
                                * weights[weight_base + ic];
                        }
                    }

                    output[output_base + oc] = layer.activation.apply(sum);
                    oc += 1;
                }
            }
        }
    }
}

#[inline]
fn grouped_conv1d_direct_scalar(
    layer: &GroupedConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    weights: &[f32],
    biases: &[f32],
    positions: &[i32],
    output: &mut [f32],
) {
    let group_in = layer.in_channels / layer.groups;
    let group_out = layer.out_channels / layer.groups;
    let kernel_width = group_in * layer.kernel_size;

    for b in 0..batch_size {
        let input_batch_base = b * input_length * layer.in_channels;
        let output_batch_base = b * output_length * layer.out_channels;

        for group in 0..layer.groups {
            let input_group_base = group * group_in;
            let output_group_base = group * group_out;
            let weight_group_base = group * group_out * kernel_width;
            let bias_group_base = group * group_out;

            for out_pos in 0..output_length {
                let position_base = out_pos * layer.kernel_size;
                let output_base =
                    output_batch_base + out_pos * layer.out_channels + output_group_base;

                for oc in 0..group_out {
                    let mut sum = biases[bias_group_base + oc];

                    for k in 0..layer.kernel_size {
                        let src_pos = positions[position_base + k];
                        if src_pos < 0 || src_pos as usize >= input_length {
                            continue;
                        }

                        let src =
                            input_batch_base
                                + src_pos as usize * layer.in_channels
                                + input_group_base;
                        let weight_base =
                            weight_group_base
                                + oc * kernel_width
                                + k * group_in;

                        for ic in 0..group_in {
                            sum += input[src + ic]
                                * weights[weight_base + ic];
                        }
                    }

                    output[output_base + oc] = layer.activation.apply(sum);
                }
            }
        }
    }
}

/// Direct grouped Conv1D forward pass.
///
/// This is a drop-in replacement for `grouped_conv1d_forward` with the same
/// external behavior, but it avoids the large im2col buffer and GEMM staging.
/// On x86-64 with AVX2+FMA, eight output channels are accumulated at once.
pub fn forward_direct(
    layer: &GroupedConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    debug_assert!(layer.groups > 0);
    debug_assert_eq!(layer.in_channels % layer.groups, 0);
    debug_assert_eq!(layer.out_channels % layer.groups, 0);
    debug_assert_eq!(input.len(), batch_size * input_length * layer.in_channels);
    debug_assert_eq!(
        layer.weight_handles.len(),
        layer.groups
            * (layer.out_channels / layer.groups)
            * (layer.in_channels / layer.groups)
            * layer.kernel_size
    );
    debug_assert_eq!(layer.bias_handles.len(), layer.out_channels);

    let mut weights = vec![0.0f32; layer.weight_handles.len()];
    let mut biases = vec![0.0f32; layer.bias_handles.len()];

    crate::handle_data_slice(&layer.weight_handles, &mut weights);
    crate::handle_data_slice(&layer.bias_handles, &mut biases);

    let mut output = vec![
        0.0f32;
        batch_size * output_length * layer.out_channels
    ];

    let positions = make_conv_positions(
        output_length,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        layer.causal,
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                grouped_conv1d_direct_avx2(
                    layer,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                    &weights,
                    &biases,
                    &positions,
                    &mut output,
                );
            }
            return output;
        }
    }

    grouped_conv1d_direct_scalar(
        layer,
        input,
        batch_size,
        input_length,
        output_length,
        &weights,
        &biases,
        &positions,
        &mut output,
    );

    output
}
