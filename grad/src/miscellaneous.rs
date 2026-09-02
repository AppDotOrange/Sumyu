use crate::neuron::LayerSpec;

pub fn stream_token(
    token: &str,
    byte_buffer: &mut Vec<u8>,
) {
    let bytes = token.as_bytes();
    let mut i = 0;
    let mut normal_start = 0;

    while i < bytes.len() {
        if i + 5 < bytes.len()
            && bytes[i] == b'<'
            && bytes[i + 1] == b'0'
            && bytes[i + 2] == b'x'
            && bytes[i + 5] == b'>'
        {
            if let Ok(byte) =
                u8::from_str_radix(&token[i + 3..i + 5], 16)
            {
                // Print ordinary text before the byte token.
                if normal_start < i {
                    if !byte_buffer.is_empty() {
                        print!("{}", String::from_utf8_lossy(byte_buffer));
                        byte_buffer.clear();
                    }

                    print!("{}", &token[normal_start..i]);
                }

                byte_buffer.push(byte);

                // If this is now a complete UTF-8 sequence,
                // print it immediately.
                if let Ok(text) = std::str::from_utf8(byte_buffer) {
                    print!("{}", text);
                    byte_buffer.clear();
                }

                i += 6;
                normal_start = i;
                continue;
            }
        }

        i += 1;
    }

    // Print ordinary text remaining after the final byte token.
    if normal_start < bytes.len() {
        if !byte_buffer.is_empty() {
            print!("{}", String::from_utf8_lossy(byte_buffer));
            byte_buffer.clear();
        }

        print!("{}", &token[normal_start..]);
    }
}

pub fn decode_token_stream(tokens: &[String]) -> String {
    let mut output = String::new();
    let mut byte_buffer = Vec::<u8>::new();

    for token in tokens {
        let bytes = token.as_bytes();
        let mut i = 0;
        let mut text_start = 0;

        while i < bytes.len() {
            if i + 5 < bytes.len()
                && bytes[i] == b'<'
                && bytes[i + 1] == b'0'
                && bytes[i + 2] == b'x'
                && bytes[i + 5] == b'>'
            {
                if let Ok(byte) =
                    u8::from_str_radix(&token[i + 3..i + 5], 16)
                {
                    // Flush normal text before the byte token.
                    if text_start < i {
                        if !byte_buffer.is_empty() {
                            output.push_str(
                                &String::from_utf8_lossy(&byte_buffer)
                            );
                            byte_buffer.clear();
                        }

                        output.push_str(&token[text_start..i]);
                    }

                    byte_buffer.push(byte);

                    // If we now have valid UTF-8, emit it.
                    if let Ok(text) =
                        std::str::from_utf8(&byte_buffer)
                    {
                        output.push_str(text);
                        byte_buffer.clear();
                    }

                    i += 6;
                    text_start = i;
                    continue;
                }
            }

            i += 1;
        }

        // Flush normal text after the final byte token.
        if text_start < bytes.len() {
            if !byte_buffer.is_empty() {
                output.push_str(
                    &String::from_utf8_lossy(&byte_buffer)
                );
                byte_buffer.clear();
            }

            output.push_str(&token[text_start..]);
        }
    }

    // Handle an incomplete/invalid sequence at the very end.
    if !byte_buffer.is_empty() {
        output.push_str(
            &String::from_utf8_lossy(&byte_buffer)
        );
    }

    output
}

pub fn print_layer_specs(
    specs: &[LayerSpec],
    mut sequence_length: usize,
    mut channels: usize,
    vocab_size: usize,
    indent: usize,
) -> (usize, usize) {
    let prefix = "  ".repeat(indent);

    for (idx, layer) in specs.iter().enumerate() {
        match layer {
            LayerSpec::Dense {
                output_size,
                ..
            } => {
                let input_size =
                    sequence_length * channels;

                let weights =
                    input_size * output_size;

                let biases =
                    *output_size;

                println!(
                    "{}ЖХЖХЖХЖХЖХЖХЖХЖХЖХЖХЖХЖХЖХЖХ   {} weights + {} biases",
                    prefix,
                    weights,
                    biases
                );

                println!(
                    "{}O O O O O O O O O O O O O O O O   Dense {}: {} neurons",
                    prefix,
                    idx + 1,
                    output_size
                );

                sequence_length = 1;
                channels = *output_size;
            }

            LayerSpec::Conv1D {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                causal,
                ..
            } => {
                debug_assert_eq!(
                    channels,
                    *in_channels,
                    "Conv1D channel mismatch"
                );

                let output_length =
                    if *causal {
                        (sequence_length - 1) / stride + 1
                    } else {
                        (sequence_length + 2 * padding - kernel_size)
                            / stride
                            + 1
                    };

                let weights =
                    out_channels
                        * in_channels
                        * kernel_size;

                let biases =
                    *out_channels;

                println!(
                    "{}████████████████████████████████   Conv1D {}: [{} × {}] → [{} × {}]",
                    prefix,
                    idx + 1,
                    sequence_length,
                    in_channels,
                    output_length,
                    out_channels,
                );

                println!(
                    "{}                                  kernel {} stride {} padding {} causal {} | {} weights + {} biases",
                    prefix,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                    weights,
                    biases
                );

                sequence_length = output_length;
                channels = *out_channels;
            }

            LayerSpec::DepthwiseConv1D {
                in_channels,
                kernel_size,
                stride,
                padding,
                causal,
                ..
            } => {
                debug_assert_eq!(
                    channels,
                    *in_channels,
                    "DepthwiseConv1D channel mismatch"
                );

                let output_length =
                    if *causal {
                        (sequence_length - 1) / stride + 1
                    } else {
                        (sequence_length + 2 * padding - kernel_size)
                            / stride
                            + 1
                    };

                let weights =
                    in_channels * kernel_size;

                let biases =
                    *in_channels;

                println!(
                    "{}████████████████████████████████   DepthwiseConv1D {}: [{} × {}] → [{} × {}]",
                    prefix,
                    idx + 1,
                    sequence_length,
                    in_channels,
                    output_length,
                    in_channels,
                );

                println!(
                    "{}                                  kernel {} stride {} padding {} causal {} | {} weights + {} biases",
                    prefix,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                    weights,
                    biases
                );

                sequence_length = output_length;
                channels = *in_channels;
            }

            LayerSpec::GroupedConv1D {
                in_channels,
                out_channels,
                groups,
                kernel_size,
                stride,
                padding,
                causal,
                ..
            } => {
                debug_assert_eq!(
                    channels,
                    *in_channels,
                    "GroupedConv1D channel mismatch"
                );

                let output_length =
                    if *causal {
                        (sequence_length - 1) / stride + 1
                    } else {
                        (sequence_length + 2 * padding - kernel_size)
                            / stride
                            + 1
                    };

                let group_in =
                    in_channels / groups;

                let weights =
                    out_channels
                        * group_in
                        * kernel_size;

                let biases =
                    *out_channels;

                println!(
                    "{}████████████████████████████████   GroupedConv1D {}: [{} × {}] → [{} × {}]",
                    prefix,
                    idx + 1,
                    sequence_length,
                    in_channels,
                    output_length,
                    out_channels,
                );

                println!(
                    "{}                                  groups {} kernel {} stride {} padding {} causal {} | {} weights + {} biases",
                    prefix,
                    groups,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                    weights,
                    biases
                );

                sequence_length = output_length;
                channels = *out_channels;
            }

            LayerSpec::LowRankPointwise {
                in_channels,
                rank,
                out_channels,
                ..
            } => {
                debug_assert_eq!(
                    channels,
                    *in_channels,
                    "LowRankPointwise channel mismatch"
                );

                let first_weights =
                    in_channels * rank;

                let first_biases =
                    *rank;

                let second_weights =
                    rank * out_channels;

                let second_biases =
                    *out_channels;

                let total =
                    first_weights
                        + first_biases
                        + second_weights
                        + second_biases;

                println!(
                    "{}████████████████████████████████   LowRankPointwise {}: [{} × {}] → [{} × {}]",
                    prefix,
                    idx + 1,
                    sequence_length,
                    in_channels,
                    sequence_length,
                    out_channels,
                );

                println!(
                    "{}                                  rank {} | {} + {} + {} + {} = {} params",
                    prefix,
                    rank,
                    first_weights,
                    first_biases,
                    second_weights,
                    second_biases,
                    total
                );

                sequence_length = sequence_length;
                channels = *out_channels;
            }

            LayerSpec::ChannelScale {
                channels: scale_channels,
            } => {
                debug_assert_eq!(
                    channels,
                    *scale_channels,
                    "ChannelScale channel mismatch"
                );

                let scales =
                    *scale_channels;

                let biases =
                    *scale_channels;

                println!(
                    "{}████████████████████████████████   ChannelScale {}: [{} × {}] → [{} × {}]",
                    prefix,
                    idx + 1,
                    sequence_length,
                    scale_channels,
                    sequence_length,
                    scale_channels,
                );

                println!(
                    "{}                                  {} scales + {} biases",
                    prefix,
                    scales,
                    biases
                );

                channels = *scale_channels;
            }

            LayerSpec::Residual {
                layers: inner_layers,
            } => {
                println!(
                    "{}╔══════════════════════════════════   Residual {}",
                    prefix,
                    idx + 1
                );

                let (inner_sequence_length, inner_channels) =
                    print_layer_specs(
                        inner_layers,
                        sequence_length,
                        channels,
                        vocab_size,
                        indent + 1,
                    );

                debug_assert_eq!(
                    inner_sequence_length,
                    sequence_length,
                    "Residual block changed sequence length"
                );

                debug_assert_eq!(
                    inner_channels,
                    channels,
                    "Residual block changed channel count"
                );

                println!(
                    "{}╚══════════════════════════════════   Residual {}",
                    prefix,
                    idx + 1
                );
            }

            LayerSpec::WeightTying => {
                println!(
                    "{}████████████████████████████████   WeightTying {}: [{} × {}] → [{} logits]",
                    prefix,
                    idx + 1,
                    sequence_length,
                    channels,
                    vocab_size,
                );

                println!(
                    "{}                                  shared embedding matrix | 0 new params",
                    prefix
                );

                sequence_length = 1;
                channels = vocab_size;
            }
        }
    }

    (sequence_length, channels)
}
