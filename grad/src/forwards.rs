use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_blendv_ps,
    _mm256_cmp_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_setzero_ps,
    _CMP_LE_OQ,
};
use cblas::{Layout, Transpose};
use crate::backwards::add_f32_slice_simd;
use crate::neuron::{Activation, BatchLayerCache, ChannelScaleLayer, GlobalMixerLayer, Layer, LowRankPointwiseLayer, WeightTyingLayer};
use crate::conv1d_kernels::conv1d_forward;
use crate::depthwise_kernel::depthwise_conv1d_forward;
pub use crate::grouped_kernel::forward_direct as grouped_conv1d_forward;
use crate::TensorHandle;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn leaky_relu_bias_avx2(
    values: &mut [f32],
    biases: &[f32],
    channels: usize,
    slope: f32,
) { unsafe {
    let zero = _mm256_setzero_ps();
    let slope_v = _mm256_set1_ps(slope);

    let rows = values.len() / channels;

    for row in 0..rows {
        let base = row * channels;

        let mut c = 0;

        while c + 8 <= channels {
            let x = _mm256_loadu_ps(
                values.as_ptr().add(base + c)
            );

            let b = _mm256_loadu_ps(
                biases.as_ptr().add(c)
            );

            let x = _mm256_add_ps(x, b);

            let mask =
                _mm256_cmp_ps(x, zero, _CMP_LE_OQ);

            let negative =
                _mm256_mul_ps(x, slope_v);

            let y =
                _mm256_blendv_ps(
                    x,
                    negative,
                    mask,
                );

            _mm256_storeu_ps(
                values.as_mut_ptr().add(base + c),
                y,
            );

            c += 8;
        }

        while c < channels {
            let i = base + c;

            let x = values[i] + biases[c];

            values[i] =
                if x <= 0.0 {
                    x * slope
                } else {
                    x
                };

            c += 1;
        }
    }
}}

fn activation_bias_simd(
    values: &mut [f32],
    biases: &[f32],
    channels: usize,
    activation: &Activation,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            if let Activation::LeakyReLU { slope } =
                activation
            {
                unsafe {
                    leaky_relu_bias_avx2(
                        values,
                        biases,
                        channels,
                        *slope,
                    );
                }

                return;
            }
        }
    }

    let rows = values.len() / channels;

    for row in 0..rows {
        let base = row * channels;

        for c in 0..channels {
            let i = base + c;

            values[i] =
                activation.apply(
                    values[i] + biases[c]
                );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn channel_scale_avx2(
    input: &[f32],
    output: &mut [f32],
    scales: &[f32],
    biases: &[f32],
    channels: usize,
) { unsafe {
    let positions = input.len() / channels;

    for position in 0..positions {
        let offset = position * channels;

        let mut c = 0;

        // 8 f32s = 256 bits
        while c + 8 <= channels {
            let x = _mm256_loadu_ps(
                input.as_ptr().add(offset + c)
            );

            let s = _mm256_loadu_ps(
                scales.as_ptr().add(c)
            );

            let b = _mm256_loadu_ps(
                biases.as_ptr().add(c)
            );

            // x * s + b
            let y = _mm256_fmadd_ps(x, s, b);

            _mm256_storeu_ps(
                output.as_mut_ptr().add(offset + c),
                y,
            );

            c += 8;
        }

        // Scalar fallback for remaining channels.
        while c < channels {
            output[offset + c] =
                input[offset + c] * scales[c] + biases[c];

            c += 1;
        }
    }
}}

pub fn channel_scale_simd(
    input: &[f32],
    output: &mut [f32],
    scales: &[f32],
    biases: &[f32],
    channels: usize,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(channels > 0);
    debug_assert_eq!(input.len() % channels, 0);
    debug_assert_eq!(scales.len(), channels);
    debug_assert_eq!(biases.len(), channels);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                channel_scale_avx2(
                    input,
                    output,
                    scales,
                    biases,
                    channels,
                );
            }

            return;
        }
    }

    // Portable scalar fallback.
    let positions = input.len() / channels;

    for position in 0..positions {
        let offset = position * channels;

        for c in 0..channels {
            output[offset + c] =
                input[offset + c] * scales[c] + biases[c];
        }
    }
}

pub fn channel_scale_forward(
    layer: &ChannelScaleLayer,
    input: &[f32],
    batch_size: usize,
) -> Vec<f32> {
    let mut scales = vec![0.0; layer.channels];
    let mut biases = vec![0.0; layer.channels];

    crate::handle_data_slice(
        &layer.scale_handles,
        &mut scales,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let sequence_size = input.len() / batch_size;

    debug_assert_eq!(
        sequence_size % layer.channels,
        0
    );

    let mut output = vec![0.0; input.len()];

    for b in 0..batch_size {
        let base = b * sequence_size;
        let end = base + sequence_size;

        channel_scale_simd(
            &input[base..end],
            &mut output[base..end],
            &scales,
            &biases,
            layer.channels,
        );
    }

    output
}

pub fn low_rank_pointwise_forward(
    layer: &LowRankPointwiseLayer,
    input: &[f32],
    batch_size: usize,
    positions: usize,
    first_weights: &[f32],
    first_biases: &[f32],
    second_weights: &[f32],
    second_biases: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rows =
        batch_size * positions;

    let mut hidden =
        vec![0.0; rows * layer.rank];

    let mut output =
        vec![0.0; rows * layer.out_channels];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            layer.rank as i32,
            layer.in_channels as i32,
            1.0,
            input,
            layer.in_channels as i32,
            &first_weights,
            layer.in_channels as i32,
            0.0,
            &mut hidden,
            layer.rank as i32,
        );

        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            layer.out_channels as i32,
            layer.rank as i32,
            1.0,
            &hidden,
            layer.rank as i32,
            &second_weights,
            layer.rank as i32,
            0.0,
            &mut output,
            layer.out_channels as i32,
        );
    }

    for row in 0..rows {
        let hidden_start =
            row * layer.rank;

        add_f32_slice_simd(
            &mut hidden[
                hidden_start
                    ..hidden_start + layer.rank
                ],
            first_biases,
        );
    }

    activation_bias_simd(
        &mut output,
        second_biases,
        layer.out_channels,
        &layer.activation,
    );

    (output, hidden)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_group_avx2(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) -> (f32, f32) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len(), gamma.len());
    debug_assert_eq!(input.len(), beta.len());

    let channels = input.len();

    // ------------------------------------------------------------
    // Mean.
    // ------------------------------------------------------------

    let mut sum_vec =
        _mm256_setzero_ps();

    let mut c =
        0usize;

    while c + 8 <= channels {
        let x =
            _mm256_loadu_ps(
                input.as_ptr().add(c)
            );

        sum_vec =
            _mm256_add_ps(
                sum_vec,
                x,
            );

        c += 8;
    }

    // Horizontal reduction.
    let mut tmp =
        [0.0f32; 8];

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        sum_vec,
    );

    let mut sum =
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

    while c < channels {
        sum += input[c];
        c += 1;
    }

    let channels_f32 =
        channels as f32;

    let mean =
        sum / channels_f32;

    // ------------------------------------------------------------
    // Variance.
    // ------------------------------------------------------------

    let mean_vec =
        _mm256_set1_ps(mean);

    let mut variance_vec =
        _mm256_setzero_ps();

    c = 0;

    while c + 8 <= channels {
        let x =
            _mm256_loadu_ps(
                input.as_ptr().add(c)
            );

        let diff =
            _mm256_sub_ps(
                x,
                mean_vec,
            );

        variance_vec =
            _mm256_add_ps(
                variance_vec,
                _mm256_mul_ps(
                    diff,
                    diff,
                ),
            );

        c += 8;
    }

    _mm256_storeu_ps(
        tmp.as_mut_ptr(),
        variance_vec,
    );

    let mut variance =
        tmp[0]
            + tmp[1]
            + tmp[2]
            + tmp[3]
            + tmp[4]
            + tmp[5]
            + tmp[6]
            + tmp[7];

    while c < channels {
        let diff =
            input[c] - mean;

        variance +=
            diff * diff;

        c += 1;
    }

    variance /=
        channels_f32;

    let inv_std =
        1.0f32
            / (variance + epsilon).sqrt();

    // ------------------------------------------------------------
    // Normalize + affine.
    // ------------------------------------------------------------

    let inv_std_vec =
        _mm256_set1_ps(inv_std);

    let mut gamma_ptr =
        gamma.as_ptr();

    let mut beta_ptr =
        beta.as_ptr();

    c = 0;

    while c + 8 <= channels {
        let x =
            _mm256_loadu_ps(
                input.as_ptr().add(c)
            );

        let g =
            _mm256_loadu_ps(
                gamma_ptr
            );

        let b =
            _mm256_loadu_ps(
                beta_ptr
            );

        let normalized =
            _mm256_mul_ps(
                _mm256_sub_ps(
                    x,
                    mean_vec,
                ),
                inv_std_vec,
            );

        let y =
            _mm256_add_ps(
                _mm256_mul_ps(
                    normalized,
                    g,
                ),
                b,
            );

        _mm256_storeu_ps(
            output.as_mut_ptr().add(c),
            y,
        );

        gamma_ptr =
            gamma_ptr.add(8);

        beta_ptr =
            beta_ptr.add(8);

        c += 8;
    }

    while c < channels {
        let normalized =
            (input[c] - mean)
                * inv_std;

        output[c] =
            normalized * gamma[c]
                + beta[c];

        c += 1;
    }

    (mean, inv_std)
}

pub fn layer_norm_forward(
    input: &[f32],
    batch_size: usize,
    channels: usize,
    epsilon: f32,
    gamma_handles: &[TensorHandle],
    beta_handles: &[TensorHandle],
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    debug_assert!(
        batch_size > 0
    );

    debug_assert!(
        channels > 0
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

    let mut gamma =
        vec![0.0f32; channels];

    let mut beta =
        vec![0.0f32; channels];

    crate::handle_data_slice(
        gamma_handles,
        &mut gamma,
    );

    crate::handle_data_slice(
        beta_handles,
        &mut beta,
    );

    let mut output =
        vec![0.0f32; input.len()];

    let mut means =
        vec![0.0f32; group_count];

    let mut inv_stds =
        vec![0.0f32; group_count];

    for group in 0..group_count {
        let base =
            group * channels;

        let input_group =
            &input[
                base..base + channels
                ];

        let output_group =
            &mut output[
                base..base + channels
                ];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let (mean, inv_std) =
                    unsafe {
                        layer_norm_group_avx2(
                            input_group,
                            output_group,
                            &gamma,
                            &beta,
                            epsilon,
                        )
                    };

                means[group] =
                    mean;

                inv_stds[group] =
                    inv_std;

                continue;
            }
        }

        // Scalar fallback.
        let channels_f32 =
            channels as f32;

        let mut sum =
            0.0f32;

        for &x in input_group {
            sum += x;
        }

        let mean =
            sum / channels_f32;

        means[group] =
            mean;

        let mut variance =
            0.0f32;

        for &x in input_group {
            let diff =
                x - mean;

            variance +=
                diff * diff;
        }

        variance /=
            channels_f32;

        let inv_std =
            1.0f32
                / (variance + epsilon).sqrt();

        inv_stds[group] =
            inv_std;

        for c in 0..channels {
            let normalized =
                (
                    input_group[c]
                        - mean
                )
                    * inv_std;

            output_group[c] =
                normalized * gamma[c]
                    + beta[c];
        }
    }

    (
        output,
        means,
        inv_stds,
    )
}

pub fn weight_tying_forward(
    layer: &WeightTyingLayer,
    input: &[f32],
    batch_size: usize,
) -> Vec<f32> {
    let embedding_dim =
        layer.embeddings.embedding_dim();

    let vocab_size =
        layer.embeddings.vocab_size();

    assert_eq!(
        input.len(),
        batch_size * embedding_dim,
        "Invalid WeightTying input size"
    );

    // IMPORTANT:
    // flat_values is already a contiguous copy of the embedding matrix.
    // Do NOT rebuild it here.
    let weights =
        layer.embeddings.flat_values();

    debug_assert_eq!(
        weights.len(),
        vocab_size * embedding_dim
    );

    let mut output =
        vec![0.0; batch_size * vocab_size];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            batch_size as i32,
            vocab_size as i32,
            embedding_dim as i32,
            1.0,
            input,
            embedding_dim as i32,
            &weights,
            embedding_dim as i32,
            0.0,
            &mut output,
            vocab_size as i32,
        );
    }

    output
}

fn global_mixer_aggregate_blas(
    input: &[f32],
    write_probs: &[f32],
    global_vectors: &mut [f32],
    batch_size: usize,
    positions: usize,
    channels: usize,
    global_dim: usize,
) {
    debug_assert_eq!(
        input.len(),
        batch_size * positions * channels
    );

    debug_assert_eq!(
        write_probs.len(),
        batch_size * positions * global_dim
    );

    debug_assert_eq!(
        global_vectors.len(),
        batch_size * global_dim * channels
    );

    for b in 0..batch_size {
        let input_base =
            b * positions * channels;

        let probs_base =
            b * positions * global_dim;

        let global_base =
            b * global_dim * channels;

        let input_batch =
            &input[
                input_base
                    ..input_base + positions * channels
                ];

        let probs_batch =
            &write_probs[
                probs_base
                    ..probs_base + positions * global_dim
                ];

        let global_batch =
            &mut global_vectors[
                global_base
                    ..global_base + global_dim * channels
                ];

        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::Ordinary,
                Transpose::None,
                global_dim as i32,
                channels as i32,
                positions as i32,
                1.0,
                probs_batch,
                global_dim as i32,
                input_batch,
                channels as i32,
                0.0,
                global_batch,
                channels as i32,
            );
        }
    }
}

fn global_mixer_readback_blas(
    input: &[f32],
    output: &mut [f32],
    read_probs: &[f32],
    global_vectors: &[f32],
    batch_size: usize,
    positions: usize,
    channels: usize,
    global_dim: usize,
) {
    debug_assert_eq!(
        input.len(),
        output.len()
    );

    debug_assert_eq!(
        read_probs.len(),
        batch_size * positions * global_dim
    );

    debug_assert_eq!(
        global_vectors.len(),
        batch_size * global_dim * channels
    );

    for b in 0..batch_size {
        let input_base =
            b * positions * channels;

        let probs_base =
            b * positions * global_dim;

        let global_base =
            b * global_dim * channels;

        let probs_batch =
            &read_probs[
                probs_base
                    ..probs_base + positions * global_dim
                ];

        let global_batch =
            &global_vectors[
                global_base
                    ..global_base + global_dim * channels
                ];

        let output_batch =
            &mut output[
                input_base
                    ..input_base + positions * channels
                ];

        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                positions as i32,
                channels as i32,
                global_dim as i32,
                1.0,
                probs_batch,
                global_dim as i32,
                global_batch,
                channels as i32,
                0.0,
                output_batch,
                channels as i32,
            );
        }
    }

    // Residual addition.
    add_f32_slice_simd(
        output,
        input,
    );
}

fn global_mixer_softmax_write(
    values: &mut [f32],
    biases: &[f32],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
) {
    for b in 0..batch_size {
        let base =
            b * positions * global_dim;

        for g in 0..global_dim {
            let mut max_value =
                f32::NEG_INFINITY;

            // Find max, including bias.
            for p in 0..positions {
                let index =
                    base
                        + p * global_dim
                        + g;

                let value =
                    values[index]
                        + biases[g];

                if value > max_value {
                    max_value = value;
                }
            }

            let mut sum =
                0.0f32;

            // exp + store.
            for p in 0..positions {
                let index =
                    base
                        + p * global_dim
                        + g;

                let value =
                    (values[index]
                        + biases[g]
                        - max_value)
                        .exp();

                values[index] =
                    value;

                sum += value;
            }

            let inv_sum =
                1.0f32 / sum;

            for p in 0..positions {
                let index =
                    base
                        + p * global_dim
                        + g;

                values[index] *= inv_sum;
            }
        }
    }
}

fn global_mixer_softmax_read(
    values: &mut [f32],
    biases: &[f32],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
) {
    let rows =
        batch_size * positions;

    for row in 0..rows {
        let base =
            row * global_dim;

        let mut max_value =
            f32::NEG_INFINITY;

        for g in 0..global_dim {
            let value =
                values[base + g]
                    + biases[g];

            if value > max_value {
                max_value = value;
            }
        }

        let mut sum =
            0.0f32;

        for g in 0..global_dim {
            let index =
                base + g;

            let value =
                (values[index]
                    + biases[g]
                    - max_value)
                    .exp();

            values[index] =
                value;

            sum += value;
        }

        let inv_sum =
            1.0f32 / sum;

        for g in 0..global_dim {
            values[base + g] *=
                inv_sum;
        }
    }
}

pub fn global_mixer_forward(
    layer: &GlobalMixerLayer,
    input: &[f32],
    batch_size: usize,
    positions: usize,
    write_weights: &[f32],
    write_biases: &[f32],
    read_weights: &[f32],
    read_biases: &[f32],
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let channels =
        layer.channels;

    let global_dim =
        layer.global_dim;

    debug_assert_eq!(
        input.len(),
        batch_size
            * positions
            * channels
    );

    debug_assert_eq!(
        write_weights.len(),
        global_dim * channels
    );

    debug_assert_eq!(
        write_biases.len(),
        global_dim
    );

    debug_assert_eq!(
        read_weights.len(),
        global_dim * channels
    );

    debug_assert_eq!(
        read_biases.len(),
        global_dim
    );

    let rows =
        batch_size * positions;

    // ------------------------------------------------------------
    // Write projection:
    //
    // X [rows, C] @ W_write^T [C, G]
    // -> [rows, G]
    //
    // W_write is stored [G, C].
    // ------------------------------------------------------------

    let mut write_probs =
        vec![
            0.0f32;
            rows * global_dim
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            global_dim as i32,
            channels as i32,
            1.0,
            input,
            channels as i32,
            write_weights,
            channels as i32,
            0.0,
            &mut write_probs,
            global_dim as i32,
        );
    }

    global_mixer_softmax_write(
        &mut write_probs,
        write_biases,
        batch_size,
        positions,
        global_dim,
    );

    // ------------------------------------------------------------
    // Global vectors:
    //
    // global[g,c] =
    //     Σ_p write_prob[p,g] * X[p,c]
    //
    // [G, C]
    // ------------------------------------------------------------

    let mut global_vectors =
        vec![
            0.0f32;
            batch_size
                * global_dim
                * channels
        ];

    global_mixer_aggregate_blas(
        input,
        &write_probs,
        &mut global_vectors,
        batch_size,
        positions,
        channels,
        global_dim,
    );

    // ------------------------------------------------------------
    // Read projection:
    //
    // X [rows, C] @ W_read^T [C, G]
    // -> [rows, G]
    //
    // Softmax over G.
    // ------------------------------------------------------------

    let mut read_probs =
        vec![
            0.0f32;
            rows * global_dim
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            global_dim as i32,
            channels as i32,
            1.0,
            input,
            channels as i32,
            read_weights,
            channels as i32,
            0.0,
            &mut read_probs,
            global_dim as i32,
        );
    }

    global_mixer_softmax_read(
        &mut read_probs,
        read_biases,
        batch_size,
        positions,
        global_dim,
    );

    // ------------------------------------------------------------
    // Read global vectors back into each position:
    //
    // output[p,c] =
    //     X[p,c]
    //     + Σ_g read_prob[p,g] * global[g,c]
    // ------------------------------------------------------------

    let mut output =
        vec![0.0f32; input.len()];

    global_mixer_readback_blas(
        input,
        &mut output,
        &read_probs,
        &global_vectors,
        batch_size,
        positions,
        channels,
        global_dim,
    );

    (
        output,
        write_probs,
        global_vectors,
        read_probs,
    )
}

pub fn forward_layers_batch(
    layers: &[Layer],
    input: &[f32],
    batch_size: usize,
    input_size: usize,
) -> (Vec<f32>, usize, Vec<BatchLayerCache>) {
    let mut current = input.to_vec();
    let mut current_size = input_size;
    let mut caches = Vec::with_capacity(layers.len());

    for layer in layers {
        let (output, output_size, cache) =
            forward_layer_batch(layer, &current, batch_size, current_size);

        current = output;
        current_size = output_size;
        caches.push(cache);
    }

    (current, current_size, caches)
}

fn forward_layer_batch(
    layer: &Layer,
    input: &[f32],
    batch_size: usize,
    input_size: usize,
) -> (Vec<f32>, usize, BatchLayerCache) {
    match layer {
        Layer::Dense(layer) => {
            let output_size = layer.biases.len();

            assert_eq!(
                input.len(),
                batch_size * input_size
            );

            let mut weights = vec![0.0; layer.fused_weights.len()];
            let mut biases = vec![0.0; layer.fused_biases.len()];

            crate::handle_data_slice(&layer.fused_weights, &mut weights);
            crate::handle_data_slice(&layer.fused_biases, &mut biases);

            let mut output =
                vec![0.0; batch_size * output_size];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    batch_size as i32,
                    output_size as i32,
                    input_size as i32,
                    1.0,
                    input,
                    input_size as i32,
                    &weights,
                    input_size as i32,
                    0.0,
                    &mut output,
                    output_size as i32,
                );
            }

            activation_bias_simd(
                &mut output,
                &biases,
                output_size,
                &layer.activation,
            );

            let activation_output =
                match layer.activation {
                    Activation::None => None,
                    _ => Some(output.clone()),
                };

            let cache = BatchLayerCache::Dense {
                input_size,
                output_size,
                input: input.to_vec(),
                activation_output,
                weights,
                weight_handles: Arc::clone(&layer.fused_weights),
                bias_handles: Arc::clone(&layer.fused_biases),
                activation: layer.activation.clone(),
            };

            (output, output_size, cache)
        }

        Layer::Conv1D(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let input_length =
                input_size / layer.in_channels;

            let output_length =
                layer.output_length(input_length);

            let output = conv1d_forward(
                layer,
                input,
                batch_size,
                input_length,
                output_length,
            );

            let output_size =
                output_length * layer.out_channels;

            let cache = BatchLayerCache::Conv1D {
                input: input.to_vec(),
                output: output.clone(),
                input_length,
                output_length,
                in_channels: layer.in_channels,
                out_channels: layer.out_channels,
                kernel_size: layer.kernel_size,
                stride: layer.stride,
                padding: layer.padding,
                causal: layer.causal,
                weight_handles: Arc::clone(
                    &layer.weight_handles
                ),
                bias_handles: Arc::clone(
                    &layer.bias_handles
                ),
                activation: layer.activation.clone(),
            };

            (output, output_size, cache)
        }

        Layer::Residual(layer) => {
            assert_eq!(
                input_size,
                layer.input_size
            );

            let (
                inner_output,
                inner_output_size,
                inner_caches,
            ) = forward_layers_batch(
                &layer.layers,
                input,
                batch_size,
                input_size,
            );

            assert_eq!(
                inner_output_size,
                input_size,
                "Residual block changed tensor size"
            );

            let mut output = inner_output;

            add_f32_slice_simd(
                &mut output,
                input,
            );

            let cache = BatchLayerCache::Residual {
                inner: inner_caches,
            };

            (
                output,
                input_size,
                cache,
            )
        }

        Layer::DepthwiseConv1D(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let input_length =
                input_size / layer.in_channels;

            let output_length =
                layer.output_length(input_length);

            let output =
                depthwise_conv1d_forward(
                    layer,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                );

            let output_size =
                output_length * layer.in_channels;

            let cache =
                BatchLayerCache::DepthwiseConv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels: layer.in_channels,
                    kernel_size: layer.kernel_size,
                    stride: layer.stride,
                    padding: layer.padding,
                    causal: layer.causal,
                    weight_handles: Arc::clone(
                        &layer.weight_handles
                    ),
                    bias_handles: Arc::clone(
                        &layer.bias_handles
                    ),
                    activation: layer.activation.clone(),
                };

            (
                output,
                output_size,
                cache,
            )
        }

        Layer::GroupedConv1D(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let input_length =
                input_size / layer.in_channels;

            let output_length =
                layer.output_length(input_length);

            let output =
                grouped_conv1d_forward(
                    layer,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                );

            let output_size =
                output_length * layer.out_channels;

            let cache =
                BatchLayerCache::GroupedConv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels: layer.in_channels,
                    out_channels: layer.out_channels,
                    groups: layer.groups,
                    kernel_size: layer.kernel_size,
                    stride: layer.stride,
                    padding: layer.padding,
                    causal: layer.causal,
                    weight_handles: Arc::clone(
                        &layer.weight_handles
                    ),
                    bias_handles: Arc::clone(
                        &layer.bias_handles
                    ),
                    activation: layer.activation.clone(),
                };

            (
                output,
                output_size,
                cache,
            )
        }

        Layer::LowRankPointwise(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let positions =
                input_size / layer.in_channels;

            let mut first_weights =
                vec![0.0; layer.first_weight_handles.len()];

            let mut first_biases =
                vec![0.0; layer.first_bias_handles.len()];

            let mut second_weights =
                vec![0.0; layer.second_weight_handles.len()];

            let mut second_biases =
                vec![0.0; layer.second_bias_handles.len()];

            crate::handle_data_slice(
                &layer.first_weight_handles,
                &mut first_weights,
            );

            crate::handle_data_slice(
                &layer.first_bias_handles,
                &mut first_biases,
            );

            crate::handle_data_slice(
                &layer.second_weight_handles,
                &mut second_weights,
            );

            crate::handle_data_slice(
                &layer.second_bias_handles,
                &mut second_biases,
            );

            let (
                output,
                hidden,
            ) = low_rank_pointwise_forward(
                layer,
                input,
                batch_size,
                positions,
                &first_weights,
                &first_biases,
                &second_weights,
                &second_biases,
            );

            let output_size =
                positions * layer.out_channels;

            let rows =
                batch_size * positions;

            let cache =
                BatchLayerCache::LowRankPointwise {
                    input: input.to_vec(),
                    hidden,
                    output: output.clone(),
                    rows,
                    in_channels: layer.in_channels,
                    rank: layer.rank,
                    out_channels: layer.out_channels,
                    first_weights,
                    second_weights,
                    first_weight_handles: Arc::clone(
                        &layer.first_weight_handles
                    ),
                    first_bias_handles: Arc::clone(
                        &layer.first_bias_handles
                    ),
                    second_weight_handles: Arc::clone(
                        &layer.second_weight_handles
                    ),
                    second_bias_handles: Arc::clone(
                        &layer.second_bias_handles
                    ),
                    activation: layer.activation.clone(),
                };

            (
                output,
                output_size,
                cache,
            )
        }

        Layer::ChannelScale(layer) => {
            assert_eq!(
                input_size % layer.channels,
                0
            );

            let output =
                channel_scale_forward(
                    layer,
                    input,
                    batch_size,
                );

            let cache =
                BatchLayerCache::ChannelScale {
                    input: input.to_vec(),
                    channels: layer.channels,
                    scale_handles: Arc::clone(
                        &layer.scale_handles
                    ),
                    bias_handles: Arc::clone(
                        &layer.bias_handles
                    ),
                };

            (
                output,
                input_size,
                cache,
            )
        }

        Layer::LayerNorm(layer) => {
            assert_eq!(
                input_size % layer.channels,
                0,
                "LayerNorm channels must divide input size"
            );

            let (
                output,
                means,
                inv_stds,
            ) =
                layer_norm_forward(
                    input,
                    batch_size,
                    layer.channels,
                    layer.epsilon,
                    &layer.gamma_handles,
                    &layer.beta_handles,
                );

            let cache =
                BatchLayerCache::LayerNorm {
                    input: input.to_vec(),
                    means,
                    inv_stds,
                    channels: layer.channels,
                    gamma_handles: Arc::clone(
                        &layer.gamma_handles
                    ),
                    beta_handles: Arc::clone(
                        &layer.beta_handles
                    ),
                };

            (
                output,
                input_size,
                cache,
            )
        }

        Layer::WeightTying(layer) => {
            let embedding_dim =
                layer.embeddings.embedding_dim();

            let vocab_size =
                layer.embeddings.vocab_size();

            assert_eq!(
                input_size,
                embedding_dim,
                "WeightTying input size must equal embedding dimension"
            );

            let output =
                weight_tying_forward(
                    layer,
                    input,
                    batch_size,
                );

            let cache =
                BatchLayerCache::WeightTying {
                    input: input.to_vec(),
                    embeddings: Arc::clone(
                        &layer.embeddings
                    ),
                    batch_size,
                    embedding_dim,
                    vocab_size,
                };

            (
                output,
                vocab_size,
                cache,
            )
        }

        Layer::GlobalMixer(layer) => {
            assert!(
                layer.channels > 0,
                "GlobalMixer channels must be > 0"
            );

            assert!(
                layer.global_dim > 0,
                "GlobalMixer global_dim must be > 0"
            );

            assert_eq!(
                input_size % layer.channels,
                0,
                "GlobalMixer channels must divide input size"
            );

            let positions =
                input_size / layer.channels;

            let mut write_weights =
                vec![
                    0.0f32;
                    layer.write_weight_handles.len()
                ];

            let mut write_biases =
                vec![
                    0.0f32;
                    layer.write_bias_handles.len()
                ];

            let mut read_weights =
                vec![
                    0.0f32;
                    layer.read_weight_handles.len()
                ];

            let mut read_biases =
                vec![
                    0.0f32;
                    layer.read_bias_handles.len()
                ];

            crate::handle_data_slice(
                &layer.write_weight_handles,
                &mut write_weights,
            );

            crate::handle_data_slice(
                &layer.write_bias_handles,
                &mut write_biases,
            );

            crate::handle_data_slice(
                &layer.read_weight_handles,
                &mut read_weights,
            );

            crate::handle_data_slice(
                &layer.read_bias_handles,
                &mut read_biases,
            );

            debug_assert_eq!(
                write_weights.len(),
                layer.global_dim
                    * layer.channels
            );

            debug_assert_eq!(
                read_weights.len(),
                layer.global_dim
                    * layer.channels
            );

            let (
                output,
                write_probs,
                global_vectors,
                read_probs,
            ) =
                global_mixer_forward(
                    layer,
                    input,
                    batch_size,
                    positions,
                    &write_weights,
                    &write_biases,
                    &read_weights,
                    &read_biases,
                );

            let cache =
                BatchLayerCache::GlobalMixer {
                    input: input.to_vec(),
                    write_probs,
                    global_vectors,
                    read_probs,
                    positions,
                    channels: layer.channels,
                    global_dim: layer.global_dim,
                    write_weight_handles: Arc::clone(
                        &layer.write_weight_handles
                    ),
                    write_bias_handles: Arc::clone(
                        &layer.write_bias_handles
                    ),
                    read_weight_handles: Arc::clone(
                        &layer.read_weight_handles
                    ),
                    read_bias_handles: Arc::clone(
                        &layer.read_bias_handles
                    ),
                };

            (
                output,
                input_size,
                cache,
            )
        }
    }
}
