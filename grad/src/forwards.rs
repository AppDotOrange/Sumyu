use std::sync::Arc;
use cblas::{Layout, Transpose};
use crate::neuron::{Activation, BatchLayerCache, ChannelScaleLayer, Conv1DLayer, DepthwiseConv1DLayer, GroupedConv1DLayer, Layer, LowRankPointwiseLayer, WeightTyingLayer, CONV1D_TILE};

pub fn conv1d_forward(
    layer: &Conv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let kernel_width = layer.kernel_size * layer.in_channels;
    let mut output = vec![0.0; batch_size * output_length * layer.out_channels];
    let mut weights = vec![0.0; layer.weight_handles.len()];
    let mut biases = vec![0.0; layer.bias_handles.len()];
    crate::handle_data_slice(&layer.weight_handles, &mut weights);
    crate::handle_data_slice(&layer.bias_handles, &mut biases);
    let mut col = vec![0.0; CONV1D_TILE * output_length * kernel_width];
    for batch_start in (0..batch_size).step_by(CONV1D_TILE) {
        let tile_batch = (batch_size - batch_start).min(CONV1D_TILE);
        let rows = tile_batch * output_length;
        for b in 0..tile_batch {
            for out_pos in 0..output_length {
                let row = b * output_length + out_pos;
                let row_start = row * kernel_width;

                for k in 0..layer.kernel_size {
                    let pos = out_pos * layer.stride + k;

                    let src_pos = if layer.causal {
                        // Causal convolution:
                        // output[t] may only read input positions <= t.
                        pos as isize - (layer.kernel_size - 1) as isize
                    } else {
                        // Existing symmetric-padding behavior.
                        pos as isize - layer.padding as isize
                    };

                    let dst = row_start + k * layer.in_channels;

                    if src_pos >= 0 && (src_pos as usize) < input_length {
                        let src = ((batch_start + b) * input_length + src_pos as usize)
                            * layer.in_channels;

                        col[dst..dst + layer.in_channels]
                            .copy_from_slice(&input[src..src + layer.in_channels]);
                    } else {
                        col[dst..dst + layer.in_channels].fill(0.0);
                    }
                }
            }
        }

        let output_offset = batch_start * output_length * layer.out_channels;
        let output_tile = &mut output[
            output_offset..output_offset + rows * layer.out_channels
            ];

        crate::batched::gemm(
            &col[..rows * kernel_width],
            &weights,
            output_tile,
            rows,
            layer.out_channels,
            kernel_width,
            Transpose::None,
            Transpose::Ordinary,
            kernel_width,
            kernel_width,
            layer.out_channels,
        );

        for row in 0..rows {
            for oc in 0..layer.out_channels {
                let i = row * layer.out_channels + oc;
                output_tile[i] = layer.activation.apply(output_tile[i] + biases[oc]);
            }
        }
    }

    output
}

pub fn depthwise_conv1d_forward(
    layer: &DepthwiseConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let mut weights = vec![0.0; layer.weight_handles.len()];

    let mut biases = vec![0.0; layer.bias_handles.len()];

    crate::handle_data_slice(
        &layer.weight_handles,
        &mut weights,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let mut output =
        vec![
            0.0;
            batch_size
                * output_length
                * layer.in_channels
        ];

    for b in 0..batch_size {
        for out_pos in 0..output_length {
            for c in 0..layer.in_channels {
                let mut sum = biases[c];

                for k in 0..layer.kernel_size {
                    let pos = out_pos * layer.stride + k;

                    let src_pos =
                        if layer.causal {
                            pos as isize
                                - (layer.kernel_size - 1)
                                as isize
                        } else {
                            pos as isize
                                - layer.padding as isize
                        };

                    if src_pos >= 0
                        && (src_pos as usize) < input_length
                    {
                        let src =
                            ((b * input_length
                                + src_pos as usize)
                                * layer.in_channels)
                                + c;

                        let weight =
                            weights[
                                c * layer.kernel_size + k
                                ];

                        sum += input[src] * weight;
                    }
                }

                let dst =
                    ((b * output_length + out_pos)
                        * layer.in_channels)
                        + c;

                output[dst] = layer.activation.apply(sum);
            }
        }
    }

    output
}

pub fn channel_scale_forward(
    layer: &ChannelScaleLayer,
    input: &[f32],
    batch_size: usize,
) -> Vec<f32> {
    let mut scales = vec![0.0; layer.channels];
    let mut biases = vec![0.0; layer.channels];

    crate::handle_data_slice(&layer.scale_handles, &mut scales);
    crate::handle_data_slice(&layer.bias_handles, &mut biases);

    let sequence_size = input.len() / batch_size;

    debug_assert_eq!(sequence_size % layer.channels, 0);

    let mut output = vec![0.0; input.len()];

    for b in 0..batch_size {
        let base = b * sequence_size;
        let end = base + sequence_size;

        let input_b = &input[base..end];
        let output_b = &mut output[base..end];

        for (src, dst) in input_b
            .chunks_exact(layer.channels)
            .zip(output_b.chunks_exact_mut(layer.channels))
        {
            for c in 0..layer.channels {
                dst[c] = src[c] * scales[c] + biases[c];
            }
        }
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
        for r in 0..layer.rank {
            hidden[row * layer.rank + r] +=
                first_biases[r];
        }

        for oc in 0..layer.out_channels {
            let i =
                row * layer.out_channels + oc;

            output[i] =
                layer.activation.apply(
                    output[i] + second_biases[oc]
                );
        }
    }

    (output, hidden)
}

pub fn grouped_conv1d_forward(
    layer: &GroupedConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let group_in =
        layer.in_channels / layer.groups;

    let group_out =
        layer.out_channels / layer.groups;

    let kernel_width =
        group_in * layer.kernel_size;

    let mut weights =
        vec![0.0; layer.weight_handles.len()];

    let mut biases =
        vec![0.0; layer.bias_handles.len()];

    crate::handle_data_slice(
        &layer.weight_handles,
        &mut weights,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let mut output =
        vec![
            0.0;
            batch_size
                * output_length
                * layer.out_channels
        ];

    let mut col =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * kernel_width
        ];

    let mut group_output =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * group_out
        ];

    for batch_start
    in (0..batch_size).step_by(CONV1D_TILE)
    {
        let tile_batch =
            (batch_size - batch_start)
                .min(CONV1D_TILE);

        let rows =
            tile_batch * output_length;

        for group in 0..layer.groups {
            for b in 0..tile_batch {
                for out_pos in 0..output_length {
                    let row =
                        b * output_length + out_pos;

                    let row_start =
                        row * kernel_width;

                    for k in 0..layer.kernel_size {
                        let pos =
                            out_pos * layer.stride + k;

                        let src_pos =
                            if layer.causal {
                                pos as isize
                                    - (layer.kernel_size - 1)
                                    as isize
                            } else {
                                pos as isize
                                    - layer.padding as isize
                            };

                        let dst =
                            row_start
                                + k * group_in;

                        if src_pos >= 0
                            && (src_pos as usize)
                            < input_length
                        {
                            let src_channel =
                                group * group_in;

                            let src =
                                ((batch_start + b)
                                    * input_length
                                    + src_pos as usize)
                                    * layer.in_channels
                                    + src_channel;

                            col[
                                dst..dst + group_in
                                ].copy_from_slice(
                                &input[
                                    src..src + group_in
                                    ],
                            );
                        } else {
                            col[
                                dst..dst + group_in
                                ].fill(0.0);
                        }
                    }
                }
            }

            let weight_start =
                group
                    * group_out
                    * kernel_width;

            let weight_end =
                weight_start
                    + group_out * kernel_width;

            crate::batched::gemm(
                &col[..rows * kernel_width],
                &weights[
                    weight_start..weight_end
                    ],
                &mut group_output[
                    ..rows * group_out
                    ],
                rows,
                group_out,
                kernel_width,
                Transpose::None,
                Transpose::Ordinary,
                kernel_width,
                kernel_width,
                group_out,
            );

            let output_channel =
                group * group_out;

            for row in 0..rows {
                for oc in 0..group_out {
                    let src =
                        row * group_out + oc;

                    group_output[src] =
                        layer.activation.apply(
                            group_output[src]
                                + biases[
                                output_channel + oc
                                ],
                        );

                    let batch_row =
                        batch_start * output_length
                            * layer.out_channels;

                    output[
                        batch_row
                            + row * layer.out_channels
                            + output_channel
                            + oc
                        ] = group_output[src];
                }
            }
        }
    }

    output
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
            let output_size = layer.neurons.len();

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

            for b in 0..batch_size {
                let base = b * output_size;

                for o in 0..output_size {
                    let i = base + o;

                    output[i] =
                        layer.activation.apply(
                            output[i] + biases[o]
                        );
                }
            }

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

            let mut output =
                inner_output.clone();

            for i in 0..output.len() {
                output[i] += input[i];
            }

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
    }
}
