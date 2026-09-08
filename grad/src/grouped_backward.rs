use crate::neuron::Activation;
use crate::TensorHandle;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_set1_ps,
    _mm256_storeu_ps,
};

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_weight_input_avx2(
    input: &[f32],
    grad: f32,
    weights: &[f32],
    input_grads: &mut [f32],
    weight_grads: &mut [f32],
) {
    debug_assert_eq!(input.len(), weights.len());
    debug_assert_eq!(input.len(), input_grads.len());
    debug_assert_eq!(input.len(), weight_grads.len());

    let grad_v = _mm256_set1_ps(grad);

    let mut c = 0usize;

    // Two accumulations in parallel are useful here because this is the
    // hottest inner loop for grouped convolutions with reasonably large
    // group_in.
    while c + 16 <= input.len() {
        let x0 = _mm256_loadu_ps(input.as_ptr().add(c));
        let x1 = _mm256_loadu_ps(input.as_ptr().add(c + 8));

        let w0 = _mm256_loadu_ps(weights.as_ptr().add(c));
        let w1 = _mm256_loadu_ps(weights.as_ptr().add(c + 8));

        let old_dx0 = _mm256_loadu_ps(input_grads.as_ptr().add(c));
        let old_dx1 = _mm256_loadu_ps(input_grads.as_ptr().add(c + 8));

        let old_dw0 = _mm256_loadu_ps(weight_grads.as_ptr().add(c));
        let old_dw1 = _mm256_loadu_ps(weight_grads.as_ptr().add(c + 8));

        // dX += grad * W
        let dx0 = _mm256_fmadd_ps(grad_v, w0, old_dx0);
        let dx1 = _mm256_fmadd_ps(grad_v, w1, old_dx1);

        // dW += grad * X
        let dw0 = _mm256_fmadd_ps(grad_v, x0, old_dw0);
        let dw1 = _mm256_fmadd_ps(grad_v, x1, old_dw1);

        _mm256_storeu_ps(input_grads.as_mut_ptr().add(c), dx0);
        _mm256_storeu_ps(input_grads.as_mut_ptr().add(c + 8), dx1);

        _mm256_storeu_ps(weight_grads.as_mut_ptr().add(c), dw0);
        _mm256_storeu_ps(weight_grads.as_mut_ptr().add(c + 8), dw1);

        c += 16;
    }

    while c + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(c));
        let w = _mm256_loadu_ps(weights.as_ptr().add(c));

        let old_dx = _mm256_loadu_ps(input_grads.as_ptr().add(c));
        let old_dw = _mm256_loadu_ps(weight_grads.as_ptr().add(c));

        let dx = _mm256_fmadd_ps(grad_v, w, old_dx);
        let dw = _mm256_fmadd_ps(grad_v, x, old_dw);

        _mm256_storeu_ps(input_grads.as_mut_ptr().add(c), dx);
        _mm256_storeu_ps(weight_grads.as_mut_ptr().add(c), dw);

        c += 8;
    }

    while c < input.len() {
        input_grads[c] += grad * weights[c];
        weight_grads[c] += grad * input[c];
        c += 1;
    }
}

#[inline]
fn accum_weight_input(
    input: &[f32],
    grad: f32,
    weights: &[f32],
    input_grads: &mut [f32],
    weight_grads: &mut [f32],
) {
    debug_assert_eq!(input.len(), weights.len());
    debug_assert_eq!(input.len(), input_grads.len());
    debug_assert_eq!(input.len(), weight_grads.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                accum_weight_input_avx2(
                    input,
                    grad,
                    weights,
                    input_grads,
                    weight_grads,
                );
            }

            return;
        }
    }

    for c in 0..input.len() {
        input_grads[c] += grad * weights[c];
        weight_grads[c] += grad * input[c];
    }
}

/// Direct grouped Conv1D backward pass.
///
/// Weight layout is identical to the existing grouped_conv1d_forward path:
///
///     [group][out_channel_within_group][kernel_offset]
///
/// where kernel_offset is [k][input_channel_within_group].
///
/// Unlike the old implementation this does not build im2col, group_grad,
/// or col_grad buffers and does not call GEMM. The contiguous input-channel
/// dimension is SIMD-vectorized with AVX2/FMA when available.
pub fn backward_direct(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
) -> Vec<f32> {
    debug_assert!(batch_size > 0);
    debug_assert!(in_channels > 0);
    debug_assert!(out_channels > 0);
    debug_assert!(groups > 0);
    debug_assert_eq!(in_channels % groups, 0);
    debug_assert_eq!(out_channels % groups, 0);

    debug_assert_eq!(
        input.len(),
        batch_size * input_length * in_channels,
    );
    debug_assert_eq!(
        output.len(),
        batch_size * output_length * out_channels,
    );
    debug_assert_eq!(grad.len(), output.len());

    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let kernel_width = group_in * kernel_size;

    debug_assert_eq!(
        weight_handles.len(),
        out_channels * kernel_width,
    );
    debug_assert_eq!(bias_handles.len(), out_channels);

    // Read the current parameters once. Gradients are accumulated into the
    // returned buffers first, then added to TensorHandles by the caller.
    let mut weights = vec![0.0f32; weight_handles.len()];
    let mut biases = vec![0.0f32; bias_handles.len()];

    crate::handle_data_slice(weight_handles, &mut weights);
    crate::handle_data_slice(bias_handles, &mut biases);

    // Convert dL/d(output-after-activation) into dL/d(pre-activation) in
    // place. This preserves the semantics of the existing backward path.
    for i in 0..grad.len() {
        activation.backward(output[i], &mut grad[i]);
    }

    let mut input_grads = vec![0.0f32; input.len()];
    let mut weight_grads = vec![0.0f32; weights.len()];
    let mut bias_grads = vec![0.0f32; out_channels];

    let positions = crate::backwards::make_conv_positions(
        output_length,
        kernel_size,
        stride,
        padding,
        causal,
    );

    for b in 0..batch_size {
        let input_batch_base =
            b * input_length * in_channels;
        let output_batch_base =
            b * output_length * out_channels;

        for out_pos in 0..output_length {
            let output_base =
                output_batch_base + out_pos * out_channels;

            let position_base =
                out_pos * kernel_size;

            for group in 0..groups {
                let input_channel_base = group * group_in;
                let output_channel_base = group * group_out;
                let weight_group_base =
                    group * group_out * kernel_width;

                for oc in 0..group_out {
                    let global_oc = output_channel_base + oc;
                    let g = grad[output_base + global_oc];

                    bias_grads[global_oc] += g;

                    let weight_oc_base =
                        weight_group_base + oc * kernel_width;

                    for k in 0..kernel_size {
                        let src_pos = positions[position_base + k];

                        if src_pos < 0 || src_pos as usize >= input_length {
                            continue;
                        }

                        let src_base = input_batch_base
                            + src_pos as usize * in_channels
                            + input_channel_base;

                        let offset = k * group_in;

                        let input_slice = &input[
                            src_base..src_base + group_in
                            ];

                        let weight_slice = &weights[
                            weight_oc_base + offset
                                ..weight_oc_base + offset + group_in
                            ];

                        let input_grad_slice = &mut input_grads[
                            src_base..src_base + group_in
                            ];

                        let weight_grad_slice = &mut weight_grads[
                            weight_oc_base + offset
                                ..weight_oc_base + offset + group_in
                            ];

                        accum_weight_input(
                            input_slice,
                            g,
                            weight_slice,
                            input_grad_slice,
                            weight_grad_slice,
                        );
                    }
                }
            }
        }
    }

    crate::add_handle_grad_slices_2(
        weight_handles,
        &weight_grads,
        bias_handles,
        &bias_grads,
    );

    input_grads
}
