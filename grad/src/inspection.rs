use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::lm::LM;
use crate::neuron::BatchLayerCache;

#[derive(Debug, Clone)]
pub struct InspectedToken {
    pub id: u16,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct GlobalMixerInspection {
    /// One-based layer path. Example: "2.1" = layer 2, residual child 1.
    pub layer_path: String,

    pub positions: usize,
    pub channels: usize,
    pub global_dim: usize,

    /// Flattened [positions, global_dim].
    ///
    /// These are causal prefix-normalized write weights. For each global slot,
    /// the weights are normalized over the token prefix ending at that position.
    /// They are recurrence coefficients, not a distribution over slots.
    pub write_probs: Vec<f32>,

    /// Flattened [positions, global_dim].
    /// Read probabilities are normalized over global slots independently
    /// at each token position.
    pub read_probs: Vec<f32>,

    /// Flattened [positions, global_dim].
    /// Each value is the L2 norm of one slot's running weighted-mean state
    /// after incorporating that token.
    pub global_magnitudes: Vec<f32>,

    /// Flattened [positions, positions].
    /// Cosine similarity between input representations at each pair of positions.
    pub input_cosine_similarity: Vec<f32>,

    /// Maximum absolute difference between the reconstructed final state and
    /// the final state stored by the forward pass.
    pub final_state_max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct ModelInspection {
    pub tokens: Vec<InspectedToken>,
    pub global_mixers: Vec<GlobalMixerInspection>,
}

impl ModelInspection {
    /// Export token labels, a reconstruction summary, and one CSV per signal
    /// for every GlobalMixer.
    pub fn write_csv(&self, directory: impl AsRef<Path>) -> io::Result<()> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;

        self.write_tokens_csv(directory.join("tokens.csv"))?;
        self.write_global_mixer_summary_csv(
            directory.join("global_mixer_summary.csv"),
        )?;
        self.write_combined_csv(directory.join("all_global_mixer_data.csv"))?;

        for mixer in &self.global_mixers {
            let prefix = format!(
                "global_mixer_layer_{}",
                mixer.layer_path.replace('.', "_")
            );

            write_matrix_csv(
                directory.join(format!("{prefix}_write.csv")),
                &self.tokens,
                mixer.global_dim,
                &mixer.write_probs,
            )?;

            write_matrix_csv(
                directory.join(format!("{prefix}_read.csv")),
                &self.tokens,
                mixer.global_dim,
                &mixer.read_probs,
            )?;

            write_matrix_csv(
                directory.join(format!("{prefix}_magnitudes.csv")),
                &self.tokens,
                mixer.global_dim,
                &mixer.global_magnitudes,
            )?;

            write_similarity_csv(
                directory.join(format!("{prefix}_similarity.csv")),
                &self.tokens,
                &mixer.input_cosine_similarity,
            )?;
        }

        Ok(())
    }

    fn write_tokens_csv(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writeln!(writer, "position,token_id,token")?;

        for (position, token) in self.tokens.iter().enumerate() {
            writeln!(
                writer,
                "{},{},{}",
                position,
                token.id,
                csv_escape(&token.text),
            )?;
        }

        writer.flush()
    }

    fn write_global_mixer_summary_csv(
        &self,
        path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writeln!(
            writer,
            "layer_path,positions,channels,global_dim,final_state_max_abs_error"
        )?;

        for mixer in &self.global_mixers {
            writeln!(
                writer,
                "{},{},{},{},{:.10}",
                csv_escape(&mixer.layer_path),
                mixer.positions,
                mixer.channels,
                mixer.global_dim,
                mixer.final_state_max_abs_error,
            )?;
        }

        writer.flush()
    }

    /// Export every inspection result into one long-format CSV.
    fn write_combined_csv(
        &self,
        path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        write_combined_csv_row(
            &mut writer,
            &[
                "record_type", "layer_path", "metric", "position",
                "token_id", "token", "slot", "value",
                "other_position", "other_token_id", "other_token",
                "positions", "channels", "global_dim",
                "final_state_max_abs_error",
            ],
        )?;

        // Token inventory.
        for (position, token) in self.tokens.iter().enumerate() {
            let position_text = position.to_string();
            let token_id_text = token.id.to_string();

            write_combined_csv_row(
                &mut writer,
                &[
                    "token", "", "token",
                    &position_text,
                    &token_id_text,
                    &token.text,
                    "", "", "", "", "", "", "", "", "",
                ],
            )?;
        }

        for mixer in &self.global_mixers {
            if mixer.positions != self.tokens.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GlobalMixer {} has {} positions, but the inspection has {} tokens",
                        mixer.layer_path,
                        mixer.positions,
                        self.tokens.len(),
                    ),
                ));
            }

            let expected_signals = mixer.positions * mixer.global_dim;
            let expected_similarities = mixer.positions * mixer.positions;

            if mixer.write_probs.len() != expected_signals
                || mixer.read_probs.len() != expected_signals
                || mixer.global_magnitudes.len() != expected_signals
                || mixer.input_cosine_similarity.len() != expected_similarities
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GlobalMixer {} has inconsistent inspection data lengths",
                        mixer.layer_path,
                    ),
                ));
            }

            let positions_text = mixer.positions.to_string();
            let channels_text = mixer.channels.to_string();
            let global_dim_text = mixer.global_dim.to_string();
            let final_error_text = format!("{:.10}", mixer.final_state_max_abs_error);

            // One metadata row per GlobalMixer.
            write_combined_csv_row(
                &mut writer,
                &[
                    "summary", &mixer.layer_path, "summary",
                    "", "", "", "", "", "", "", "",
                    &positions_text,
                    &channels_text,
                    &global_dim_text,
                    &final_error_text,
                ],
            )?;

            // All per-position, per-slot signals.
            let signals = [
                ("write_weight", mixer.write_probs.as_slice()),
                ("read_probability", mixer.read_probs.as_slice()),
                ("global_state_l2_norm", mixer.global_magnitudes.as_slice()),
            ];

            for (metric, values) in signals {
                for position in 0..mixer.positions {
                    let token = &self.tokens[position];
                    let position_text = position.to_string();
                    let token_id_text = token.id.to_string();
                    let row_start = position * mixer.global_dim;

                    for slot in 0..mixer.global_dim {
                        let slot_text = slot.to_string();
                        let value_text = format!("{:.8}", values[row_start + slot]);

                        write_combined_csv_row(
                            &mut writer,
                            &[
                                "signal",
                                &mixer.layer_path,
                                metric,
                                &position_text,
                                &token_id_text,
                                &token.text,
                                &slot_text,
                                &value_text,
                                "", "", "", "", "", "", "",
                            ],
                        )?;
                    }
                }
            }

            // Full position-by-position cosine similarity matrix.
            // Each row represents one ordered pair of token positions.
            for position in 0..mixer.positions {
                let token = &self.tokens[position];
                let position_text = position.to_string();
                let token_id_text = token.id.to_string();
                let row_start = position * mixer.positions;

                for other_position in 0..mixer.positions {
                    let other_token = &self.tokens[other_position];
                    let other_position_text = other_position.to_string();
                    let other_token_id_text = other_token.id.to_string();

                    let value_text = format!(
                        "{:.8}",
                        mixer.input_cosine_similarity[row_start + other_position],
                    );

                    write_combined_csv_row(
                        &mut writer,
                        &[
                            "similarity",
                            &mixer.layer_path,
                            "input_cosine_similarity",
                            &position_text,
                            &token_id_text,
                            &token.text,
                            "",
                            &value_text,
                            &other_position_text,
                            &other_token_id_text,
                            &other_token.text,
                            "", "", "", "",
                        ],
                    )?;
                }
            }
        }

        writer.flush()
    }
}

/// Run a full-sequence inspection pass using the model's own tokenizer.
pub fn inspect_text(lm: &LM, text: &str) -> Result<ModelInspection, String> {
    let ids = lm.tokenize_text(text);
    inspect_ids(lm, &ids)
}

/// Run a full-sequence inspection pass using already-tokenized input.
pub fn inspect_ids(lm: &LM, ids: &[u16]) -> Result<ModelInspection, String> {
    if ids.is_empty() {
        return Err("Cannot inspect an empty token sequence".to_string());
    }

    let tokens = ids
        .iter()
        .map(|&id| InspectedToken {
            id,
            text: lm.token_text(id).to_string(),
        })
        .collect();

    // Ordinary forward pass. No training or parameter updates.
    // The inspector expects one sequence (batch size = 1).
    let caches = lm.inspection_forward(ids);

    let mut global_mixers = Vec::new();
    collect_global_mixers(&caches, "", &mut global_mixers)?;

    Ok(ModelInspection {
        tokens,
        global_mixers,
    })
}

fn collect_global_mixers(
    caches: &[BatchLayerCache],
    parent_path: &str,
    output: &mut Vec<GlobalMixerInspection>,
) -> Result<(), String> {
    for (index, cache) in caches.iter().enumerate() {
        let path = if parent_path.is_empty() {
            format!("{}", index + 1)
        } else {
            format!("{}.{}", parent_path, index + 1)
        };

        match cache {
            BatchLayerCache::GlobalMixer {
                input,
                write_probs,
                global_vectors,
                read_probs,
                positions,
                channels,
                global_dim,
                ..
            } => {
                output.push(inspect_global_mixer(
                    &path,
                    input,
                    write_probs,
                    global_vectors,
                    read_probs,
                    *positions,
                    *channels,
                    *global_dim,
                )?);
            }

            BatchLayerCache::Residual { inner } => {
                collect_global_mixers(inner, &path, output)?;
            }

            _ => {}
        }
    }

    Ok(())
}

fn inspect_global_mixer(
    layer_path: &str,
    input: &[f32],
    write_probs: &[f32],
    final_global_vectors: &[f32],
    read_probs: &[f32],
    positions: usize,
    channels: usize,
    global_dim: usize,
) -> Result<GlobalMixerInspection, String> {
    let expected_input = positions * channels;
    let expected_probs = positions * global_dim;
    let expected_state = global_dim * channels;

    if input.len() != expected_input {
        return Err(format!(
            "GlobalMixer {layer_path}: expected {expected_input} input values for one inspected sequence, got {}",
            input.len(),
        ));
    }

    if write_probs.len() != expected_probs || read_probs.len() != expected_probs {
        return Err(format!(
            "GlobalMixer {layer_path}: probability cache has an unexpected length"
        ));
    }

    if final_global_vectors.len() != expected_state {
        return Err(format!(
            "GlobalMixer {layer_path}: final state has an unexpected length"
        ));
    }

    let input_cosine_similarity =
        calculate_cosine_similarity_matrix(input, positions, channels);

    // Reconstruct the exact recurrence used by global_mixer_forward:
    //
    //     G[p, g, c] = (1 - a[p, g]) * G[p-1, g, c]
    //                  + a[p, g] * X[p, c]
    //
    // where a[p, g] is the causal prefix-softmax write weight. It is
    // normalized across positions 0..=p for each global slot g.
    //
    // Layouts (inspection is for batch_size = 1):
    // input:         [positions, channels]
    // write/read:    [positions, global_dim]
    // final state:   [global_dim, channels]
    //
    // The forward stores only the final state, so we replay the recurrence
    // and record each prefix state's L2 norm as we go.
    let mut state = vec![0.0f32; expected_state];
    let mut magnitudes = vec![0.0f32; expected_probs];

    for position in 0..positions {
        let input_row = position * channels;
        let probs_row = position * global_dim;

        for global in 0..global_dim {
            let state_row = global * channels;
            let write_weight = write_probs[probs_row + global];
            let old_weight = 1.0f32 - write_weight;

            for channel in 0..channels {
                let state_index = state_row + channel;
                state[state_index] =
                    old_weight * state[state_index]
                        + write_weight * input[input_row + channel];
            }

            let squared_norm: f32 = state[state_row..state_row + channels]
                .iter()
                .map(|value| value * value)
                .sum();

            magnitudes[probs_row + global] = squared_norm.sqrt();
        }
    }

    let final_state_max_abs_error = state
        .iter()
        .zip(final_global_vectors.iter())
        .map(|(reconstructed, cached)| (reconstructed - cached).abs())
        .fold(0.0f32, f32::max);

    Ok(GlobalMixerInspection {
        layer_path: layer_path.to_string(),
        positions,
        channels,
        global_dim,
        write_probs: write_probs.to_vec(),
        read_probs: read_probs.to_vec(),
        global_magnitudes: magnitudes,
        input_cosine_similarity,
        final_state_max_abs_error,
    })
}

/// Calculate a symmetric position-by-position cosine similarity matrix.
///
/// Input layout: [positions, channels]
/// Output layout: [positions, positions]
fn calculate_cosine_similarity_matrix(
    input: &[f32],
    positions: usize,
    channels: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; positions * positions];
    let mut norms = vec![0.0f32; positions];

    for position in 0..positions {
        let start = position * channels;

        let squared_norm: f32 = input[start..start + channels]
            .iter()
            .map(|x| x * x)
            .sum();

        norms[position] = squared_norm.sqrt();
    }

    // Compute only the upper triangle, then mirror it.
    for i in 0..positions {
        let a_start = i * channels;

        for j in i..positions {
            let b_start = j * channels;

            let dot: f32 = (0..channels)
                .map(|c| input[a_start + c] * input[b_start + c])
                .sum();

            let denominator = norms[i] * norms[j];

            let similarity = if denominator > 1e-12 {
                (dot / denominator).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            output[i * positions + j] = similarity;
            output[j * positions + i] = similarity;
        }
    }

    output
}

fn write_matrix_csv(
    path: impl AsRef<Path>,
    tokens: &[InspectedToken],
    columns: usize,
    values: &[f32],
) -> io::Result<()> {
    let expected = tokens.len() * columns;

    if values.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Expected {expected} matrix values, got {}", values.len()),
        ));
    }

    let mut writer = BufWriter::new(File::create(path)?);

    write!(writer, "position,token_id,token")?;

    for column in 0..columns {
        write!(writer, ",slot_{column}")?;
    }

    writeln!(writer)?;

    for (position, token) in tokens.iter().enumerate() {
        write!(
            writer,
            "{},{},{}",
            position,
            token.id,
            csv_escape(&token.text),
        )?;

        let row_start = position * columns;

        for column in 0..columns {
            write!(writer, ",{:.8}", values[row_start + column])?;
        }

        writeln!(writer)?;
    }

    writer.flush()
}

fn write_similarity_csv(
    path: impl AsRef<Path>,
    tokens: &[InspectedToken],
    values: &[f32],
) -> io::Result<()> {
    let positions = tokens.len();
    let expected = positions * positions;

    if values.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Expected {expected} similarity values, got {}",
                values.len()
            ),
        ));
    }

    let mut writer = BufWriter::new(File::create(path)?);

    write!(writer, "position,token_id,token")?;

    for column in 0..positions {
        write!(writer, ",pos_{column}")?;
    }

    writeln!(writer)?;

    for (position, token) in tokens.iter().enumerate() {
        write!(
            writer,
            "{},{},{}",
            position,
            token.id,
            csv_escape(&token.text),
        )?;

        let row_start = position * positions;

        for column in 0..positions {
            write!(writer, ",{:.6}", values[row_start + column])?;
        }

        writeln!(writer)?;
    }

    writer.flush()
}

/// Write one CSV record, escaping every field as needed.
fn write_combined_csv_row(
    writer: &mut impl Write,
    fields: &[&str],
) -> io::Result<()> {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }

        write!(writer, "{}", csv_escape(field))?;
    }

    writeln!(writer)
}

fn csv_escape(value: &str) -> String {
    if value.contains(',')
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
