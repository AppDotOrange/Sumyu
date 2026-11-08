#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps,
    _mm256_blendv_ps,
    _mm256_cmp_ps,
    _mm256_fmadd_ps,
    _mm256_loadu_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_setzero_ps,
    _mm256_storeu_ps,
    _mm256_sub_ps,
    _CMP_LE_OQ,
};

use std::sync::Arc;

use cblas::{
    Layout,
    Transpose,
};
use rayon::ThreadPool;
use crate::backwards::add_f32_slice_simd;
use crate::conv1d_kernels::conv1d_forward;
use crate::depthwise_kernel::depthwise_conv1d_forward;
pub use crate::grouped_kernel::forward_direct as grouped_conv1d_forward;
use crate::neuron::{
    Activation,
    BatchLayerCache,
    ChannelScaleLayer,
    GlobalMixerLayer,
    Layer,
    LayerNormLayer,
    LowRankPointwiseLayer,
    WeightTyingLayer,
    GLOBAL_MIXER_POS_FEATURES,
};
use crate::parameters::ParameterStore;

// ============================================================================
// Activation + bias
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn leaky_relu_bias_avx2(
    values: &mut [f32],
    biases: &[f32],
    channels: usize,
    slope: f32,
) {
    unsafe {
        let zero =
            _mm256_setzero_ps();

        let slope_v =
            _mm256_set1_ps(
                slope
            );

        let rows =
            values.len() / channels;

        for row in 0..rows {
            let base =
                row * channels;

            let mut c =
                0usize;

            while c + 8 <= channels {
                let x =
                    _mm256_loadu_ps(
                        values.as_ptr()
                            .add(base + c)
                    );

                let b =
                    _mm256_loadu_ps(
                        biases.as_ptr()
                            .add(c)
                    );

                let x =
                    _mm256_add_ps(
                        x,
                        b,
                    );

                let mask =
                    _mm256_cmp_ps(
                        x,
                        zero,
                        _CMP_LE_OQ,
                    );

                let negative =
                    _mm256_mul_ps(
                        x,
                        slope_v,
                    );

                let y =
                    _mm256_blendv_ps(
                        x,
                        negative,
                        mask,
                    );

                _mm256_storeu_ps(
                    values
                        .as_mut_ptr()
                        .add(base + c),
                    y,
                );

                c += 8;
            }

            while c < channels {
                let i =
                    base + c;

                let x =
                    values[i]
                        + biases[c];

                values[i] =
                    if x <= 0.0 {
                        x * slope
                    } else {
                        x
                    };

                c += 1;
            }
        }
    }
}

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

    let rows =
        values.len() / channels;

    for row in 0..rows {
        let base =
            row * channels;

        for c in 0..channels {
            let i =
                base + c;

            values[i] =
                activation.apply(
                    values[i]
                        + biases[c]
                );
        }
    }
}

// ============================================================================
// Global Mixer
// ============================================================================
fn global_mixer_softmax_write_causal(
    values: &mut [f32],
    scratch_log_norm: &mut [f32],
    biases: &[f32],
    positional_biases: &[f32],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(
        values.len(),
        batch_size * positions * global_dim
    );

    debug_assert_eq!(
        scratch_log_norm.len(),
        values.len()
    );

    // Final prefix log-normalizer for each batch item and global channel.
    // This is retained for continuation by incremental inference.
    let mut final_log_norm = vec![
        f32::NEG_INFINITY;
        batch_size * global_dim
    ];

    if positions == 0 {
        return final_log_norm;
    }

    for b in 0..batch_size {
        let base =
            b * positions * global_dim;

        for g in 0..global_dim {
            // ------------------------------------------------------------
            // Pass 1:
            // Compute log(sum(exp(score[0..=p]))) for every prefix.
            //
            // This uses an online log-sum-exp recurrence, so we do not
            // need O(P^2) work and don't need another large allocation.
            // ------------------------------------------------------------

            let mut max_value =
                f32::NEG_INFINITY;

            let mut exp_sum =
                0.0f32;

            for p in 0..positions {
                let index =
                    base
                        + p * global_dim
                        + g;

                let score =
                    values[index]
                        + biases[g]
                        + positional_biases[
                        p * global_dim + g
                        ];

                if score > max_value {
                    if max_value.is_finite() {
                        exp_sum *=
                            (max_value - score).exp();
                    } else {
                        exp_sum = 0.0;
                    }

                    max_value =
                        score;

                    exp_sum += 1.0;
                } else {
                    exp_sum +=
                        (score - max_value).exp();
                }

                scratch_log_norm[index] =
                    max_value
                        + exp_sum.ln();
            }

            final_log_norm[b * global_dim + g] =
                scratch_log_norm[
                    base + (positions - 1) * global_dim + g
                    ];

            // ------------------------------------------------------------
            // Pass 2:
            // Convert each score into its prefix-normalized probability.
            // ------------------------------------------------------------

            for p in 0..positions {
                let index =
                    base
                        + p * global_dim
                        + g;

                let score =
                    values[index]
                        + biases[g]
                        + positional_biases[
                        p * global_dim + g
                        ];

                values[index] =
                    (
                        score
                            - scratch_log_norm[index]
                    )
                        .exp();
            }
        }
    }

    final_log_norm
}

fn global_mixer_softmax_read(
    values: &mut [f32],
    biases: &[f32],
    positional_biases: &[f32],
    batch_size: usize,
    positions: usize,
    global_dim: usize,
) {
    let rows =
        batch_size
            * positions;

    for row in 0..rows {
        let base =
            row * global_dim;

        let position =
            row % positions;

        let positional_base =
            position
                * global_dim;

        let mut max_value =
            f32::NEG_INFINITY;

        for g in 0..global_dim {
            let value =
                values[base + g]
                    + biases[g]
                    + positional_biases[
                    positional_base + g
                    ];

            if value > max_value {
                max_value =
                    value;
            }
        }

        let mut sum =
            0.0f32;

        for g in 0..global_dim {
            let index =
                base + g;

            let value =
                (
                    values[index]
                        + biases[g]
                        + positional_biases[
                        positional_base + g
                        ]
                        - max_value
                )
                    .exp();

            values[index] =
                value;

            sum +=
                value;
        }

        let inv_sum =
            1.0f32 / sum;

        for g in 0..global_dim {
            values[base + g] *=
                inv_sum;
        }
    }
}

const GLOBAL_MIXER_POSITION_MIN_PERIOD: f64 = 3.0;
const GLOBAL_MIXER_POSITION_MAX_PERIOD: f64 = 1_048_576.0;
const GLOBAL_MIXER_POSITION_REFERENCE: f64 = 128.0;

#[inline]
pub fn global_mixer_position_features(
    positions: usize,
    position_offset: usize,
) -> Vec<[f32; GLOBAL_MIXER_POS_FEATURES]> {
    const FREQUENCY_BANDS: usize =
        GLOBAL_MIXER_POS_FEATURES / 2;

    let mut features = vec![
        [0.0f32; GLOBAL_MIXER_POS_FEATURES];
        positions
    ];

    if positions == 0 {
        return features;
    }

    // Validate the last absolute position once. This also guarantees that
    // position_offset + local_position cannot overflow inside the loop.
    position_offset
        .checked_add(positions - 1)
        .expect("GlobalMixer position overflow");

    let period_ratio =
        GLOBAL_MIXER_POSITION_MAX_PERIOD
            / GLOBAL_MIXER_POSITION_MIN_PERIOD;

    // Wavelengths are logarithmically spaced from 3 to 1,048,576 tokens.
    // Compute them once per call rather than once per token.
    let mut periods = [0.0f64; FREQUENCY_BANDS];
    for band in 0..FREQUENCY_BANDS {
        let scale =
            band as f64 / (FREQUENCY_BANDS - 1) as f64;

        periods[band] =
            GLOBAL_MIXER_POSITION_MIN_PERIOD
                * period_ratio.powf(scale);
    }

    for local_position in 0..positions {
        let absolute_position =
            position_offset + local_position;
        let p = absolute_position as f64;

        for band in 0..FREQUENCY_BANDS {
            let angle =
                std::f64::consts::TAU * p / periods[band];

            features[local_position][band * 2] =
                angle.sin() as f32;
            features[local_position][band * 2 + 1] =
                angle.cos() as f32;
        }

        // One non-periodic, slowly varying absolute-position feature.
        features[local_position][GLOBAL_MIXER_POS_FEATURES - 1] =
            (1.0 + p / GLOBAL_MIXER_POSITION_REFERENCE).ln() as f32;
    }

    features
}

fn global_mixer_build_positional_biases(
    features: &[
        [f32; GLOBAL_MIXER_POS_FEATURES]
    ],
    positional_weights: &[f32],
    global_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(
        positional_weights.len(),
        global_dim
            * GLOBAL_MIXER_POS_FEATURES
    );

    let positions =
        features.len();

    let mut biases =
        vec![
            0.0f32;
            positions
                * global_dim
        ];

    for p in 0..positions {
        let f =
            features[p];

        let row =
            p * global_dim;

        for g in 0..global_dim {
            let w =
                g
                    * GLOBAL_MIXER_POS_FEATURES;

            let mut bias = 0.0f32;
            for feature in 0..GLOBAL_MIXER_POS_FEATURES {
                bias +=
                    f[feature]
                        * positional_weights[w + feature];
            }
            biases[row + g] = bias;
        }
    }

    biases
}

pub fn global_mixer_forward(
    layer: &GlobalMixerLayer,
    input: &[f32],
    batch_size: usize,
    positions: usize,
    position_offset: usize,
    write_weights: &[f32],
    write_biases: &[f32],
    read_weights: &[f32],
    read_biases: &[f32],
    write_positional_weights: &[f32],
    read_positional_weights: &[f32],
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<[f32; GLOBAL_MIXER_POS_FEATURES]>,
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
        global_dim
            * channels
    );

    debug_assert_eq!(
        write_biases.len(),
        global_dim
    );

    debug_assert_eq!(
        read_weights.len(),
        global_dim
            * channels
    );

    debug_assert_eq!(
        read_biases.len(),
        global_dim
    );

    debug_assert_eq!(
        write_positional_weights.len(),
        global_dim
            * GLOBAL_MIXER_POS_FEATURES
    );

    debug_assert_eq!(
        read_positional_weights.len(),
        global_dim
            * GLOBAL_MIXER_POS_FEATURES
    );

    let rows =
        batch_size
            * positions;

    let position_features =
        global_mixer_position_features(
            positions,
            position_offset,
        );

    let write_positional_biases =
        global_mixer_build_positional_biases(
            &position_features,
            write_positional_weights,
            global_dim,
        );

    let read_positional_biases =
        global_mixer_build_positional_biases(
            &position_features,
            read_positional_weights,
            global_dim,
        );

    // ------------------------------------------------------------------------
    // Write projection
    // ------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------
    // Allocate read_probs early.
    //
    // We temporarily use it as scratch space for prefix log-normalizers.
    // It gets overwritten by the read projection immediately afterward.
    // ------------------------------------------------------------------------

    let mut read_probs =
        vec![
            0.0f32;
            rows * global_dim
        ];

    let write_log_norm =
        global_mixer_softmax_write_causal(
            &mut write_probs,
            &mut read_probs,
            write_biases,
            &write_positional_biases,
            batch_size,
            positions,
            global_dim,
        );

    // ------------------------------------------------------------------------
    // Read projection
    // ------------------------------------------------------------------------

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
        &read_positional_biases,
        batch_size,
        positions,
        global_dim,
    );

    // ------------------------------------------------------------------------
    // Causal global accumulation + readback.
    //
    // G_p = G_{p-1} + write_probs[p] * X_p
    //
    // Y_p = X_p + read_probs[p] * G_p
    //
    // global_vectors stores only the FINAL prefix state for each batch item.
    // That is enough for the causal backward pass to reconstruct G_p by
    // walking backwards.
    // ------------------------------------------------------------------------

    let mut global_vectors =
        vec![
            0.0f32;
            batch_size
                * global_dim
                * channels
        ];

    let mut output =
        input.to_vec();

    for b in 0..batch_size {
        let input_base =
            b
                * positions
                * channels;

        let probs_base =
            b
                * positions
                * global_dim;

        let global_base =
            b
                * global_dim
                * channels;

        for p in 0..positions {
            let input_row =
                input_base
                    + p * channels;

            let probs_row =
                probs_base
                    + p * global_dim;

            for g in 0..global_dim {
                let write_prob =
                    write_probs[
                        probs_row + g
                        ];

                let global_row =
                    global_base
                        + g * channels;

                // Update G_p first, so readback sees the current token.
                for c in 0..channels {
                    let old_weight = 1.0f32 - write_prob;

                    global_vectors[global_row + c] =
                        old_weight * global_vectors[global_row + c]
                            + write_prob * input[input_row + c];
                }

                let read_prob =
                    read_probs[
                        probs_row + g
                        ];

                // Read G_p into Y_p.
                for c in 0..channels {
                    output[
                        input_row + c
                        ] +=
                        read_prob
                            * global_vectors[
                            global_row + c
                            ];
                }
            }
        }
    }

    (
        output,
        write_probs,
        global_vectors,
        read_probs,
        write_log_norm,
        position_features,
    )
}


// ============================================================================
// Incremental inference state
// ============================================================================

/// Per-layer state required to process one token at a time.
///
/// Causal convolutions retain only their last K-1 input vectors. GlobalMixer
/// retains its cumulative global vectors and prefix write log-normalizers.
/// All other supported layers are stateless between tokens.
#[derive(Clone)]
pub enum LayerInferenceState {
    Stateless,
    CausalConvHistory {
        history: Vec<f32>,
    },
    Residual {
        inner: Vec<LayerInferenceState>,
    },
    GlobalMixer {
        global_vectors: Vec<f32>,
        write_log_norm: Vec<f32>,
    },
}

/// Stateful inference cache for an MLP whose final layer is WeightTying.
/// The cache is tied to the architecture and parameter layout used to create it.
#[derive(Clone)]
pub struct MLPInferenceState {
    pub(crate) layer_states: Vec<LayerInferenceState>,
    pub(crate) next_position: usize,
}

impl MLPInferenceState {
    pub(crate) fn new(backbone_layers: &[Layer]) -> Result<Self, String> {
        let mut layer_states = Vec::with_capacity(backbone_layers.len());

        for (index, layer) in backbone_layers.iter().enumerate() {
            layer_states.push(new_layer_inference_state(layer).map_err(|error| {
                format!("Cannot initialize incremental state for backbone layer {}: {}", index + 1, error)
            })?);
        }

        Ok(Self {
            layer_states,
            next_position: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.next_position
    }
}

fn new_layer_inference_state(
    layer: &Layer,
) -> Result<LayerInferenceState, String> {
    match layer {
        Layer::Dense(_) => Err(
            "Dense is not supported by variable-context incremental inference".to_string()
        ),

        Layer::Conv1D(layer) => {
            if !layer.causal || layer.stride != 1 {
                return Err(format!(
                    "Conv1D requires causal=true and stride=1 (got causal={}, stride={})",
                    layer.causal,
                    layer.stride,
                ));
            }
            Ok(LayerInferenceState::CausalConvHistory {
                history: Vec::with_capacity(
                    layer.kernel_size.saturating_sub(1) * layer.in_channels
                ),
            })
        }

        Layer::DepthwiseConv1D(layer) => {
            if !layer.causal || layer.stride != 1 {
                return Err(format!(
                    "DepthwiseConv1D requires causal=true and stride=1 (got causal={}, stride={})",
                    layer.causal,
                    layer.stride,
                ));
            }
            Ok(LayerInferenceState::CausalConvHistory {
                history: Vec::with_capacity(
                    layer.kernel_size.saturating_sub(1) * layer.in_channels
                ),
            })
        }

        Layer::GroupedConv1D(layer) => {
            if !layer.causal || layer.stride != 1 {
                return Err(format!(
                    "GroupedConv1D requires causal=true and stride=1 (got causal={}, stride={})",
                    layer.causal,
                    layer.stride,
                ));
            }
            Ok(LayerInferenceState::CausalConvHistory {
                history: Vec::with_capacity(
                    layer.kernel_size.saturating_sub(1) * layer.in_channels
                ),
            })
        }

        Layer::Residual(layer) => {
            let mut inner = Vec::with_capacity(layer.layers.len());
            for (index, inner_layer) in layer.layers.iter().enumerate() {
                inner.push(new_layer_inference_state(inner_layer).map_err(|error| {
                    format!("Residual inner layer {}: {}", index + 1, error)
                })?);
            }
            Ok(LayerInferenceState::Residual { inner })
        }

        Layer::LowRankPointwise(_)
        | Layer::ChannelScale(_)
        | Layer::LayerNorm(_) => Ok(LayerInferenceState::Stateless),

        Layer::GlobalMixer(layer) => Ok(LayerInferenceState::GlobalMixer {
            global_vectors: vec![
                0.0f32;
                layer.global_dim * layer.channels
            ],
            write_log_norm: vec![
                f32::NEG_INFINITY;
                layer.global_dim
            ],
        }),

        Layer::WeightTying(_) => Err(
            "WeightTying must be the final LM layer, not part of the incremental backbone".to_string()
        ),
    }
}

#[inline]
fn append_causal_history(
    history: &mut Vec<f32>,
    current: &[f32],
    channels: usize,
    kernel_size: usize,
) {
    let keep = kernel_size.saturating_sub(1) * channels;
    if keep == 0 {
        history.clear();
        return;
    }

    history.extend_from_slice(current);
    if history.len() > keep {
        let remove = history.len() - keep;
        history.drain(0..remove);
    }
}

#[inline]
fn causal_window_source<'a>(
    history: &'a [f32],
    current: &'a [f32],
    channels: usize,
    leading_padding: usize,
    kernel_offset: usize,
) -> Option<&'a [f32]> {
    if kernel_offset < leading_padding {
        return None;
    }

    let window_position = kernel_offset - leading_padding;
    let history_positions = history.len() / channels;

    if window_position < history_positions {
        let start = window_position * channels;
        Some(&history[start..start + channels])
    } else if window_position == history_positions {
        Some(current)
    } else {
        None
    }
}

#[inline]
fn global_mixer_logaddexp(a: f32, b: f32) -> f32 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }

    let maximum = a.max(b);
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

/// Process one token through a GlobalMixer while preserving its prefix state.
/// This is algebraically the one-token continuation of global_mixer_forward:
/// logZ_new = logaddexp(logZ_old, score), a = exp(score - logZ_new).
fn global_mixer_forward_incremental(
    layer: &GlobalMixerLayer,
    input: &[f32],
    absolute_position: usize,
    global_vectors: &mut [f32],
    write_log_norm: &mut [f32],
    params: &ParameterStore,
) -> Vec<f32> {
    let channels = layer.channels;
    let global_dim = layer.global_dim;

    assert_eq!(input.len(), channels);
    assert_eq!(global_vectors.len(), global_dim * channels);
    assert_eq!(write_log_norm.len(), global_dim);

    let write_weights = params.values(layer.write_weights);
    let write_biases = params.values(layer.write_biases);
    let read_weights = params.values(layer.read_weights);
    let read_biases = params.values(layer.read_biases);
    let write_positional_weights = params.values(layer.write_positional_weights);
    let read_positional_weights = params.values(layer.read_positional_weights);

    debug_assert_eq!(write_weights.len(), global_dim * channels);
    debug_assert_eq!(read_weights.len(), global_dim * channels);
    debug_assert_eq!(write_biases.len(), global_dim);
    debug_assert_eq!(read_biases.len(), global_dim);
    debug_assert_eq!(
        write_positional_weights.len(),
        global_dim * GLOBAL_MIXER_POS_FEATURES
    );
    debug_assert_eq!(
        read_positional_weights.len(),
        global_dim * GLOBAL_MIXER_POS_FEATURES
    );

    let position_features =
        global_mixer_position_features(1, absolute_position);

    let write_positional_biases =
        global_mixer_build_positional_biases(
            &position_features,
            write_positional_weights,
            global_dim,
        );

    let read_positional_biases =
        global_mixer_build_positional_biases(
            &position_features,
            read_positional_weights,
            global_dim,
        );

    // Update the persistent prefix-normalized write state.
    for g in 0..global_dim {
        let weight_base = g * channels;
        let mut score = write_biases[g] + write_positional_biases[g];

        for c in 0..channels {
            score += input[c] * write_weights[weight_base + c];
        }

        let new_log_norm =
            global_mixer_logaddexp(write_log_norm[g], score);

        let write_probability =
            (score - new_log_norm).exp();

        write_log_norm[g] = new_log_norm;

        let global_base = g * channels;
        for c in 0..channels {
            let old_weight = 1.0f32 - write_probability;

            global_vectors[global_base + c] =
                old_weight * global_vectors[global_base + c]
                    + write_probability * input[c];
        }
    }

    // Read from the updated state, matching the batched order G_p then Y_p.
    let mut read_scores = vec![0.0f32; global_dim];
    let mut maximum = f32::NEG_INFINITY;

    for g in 0..global_dim {
        let weight_base = g * channels;
        let mut score = read_biases[g] + read_positional_biases[g];

        for c in 0..channels {
            score += input[c] * read_weights[weight_base + c];
        }

        read_scores[g] = score;
        maximum = maximum.max(score);
    }

    let mut denominator = 0.0f32;
    for score in &mut read_scores {
        *score = (*score - maximum).exp();
        denominator += *score;
    }

    let inverse_denominator = 1.0f32 / denominator;
    let mut output = input.to_vec();

    for g in 0..global_dim {
        let read_probability = read_scores[g] * inverse_denominator;
        let global_base = g * channels;

        for c in 0..channels {
            output[c] +=
                read_probability * global_vectors[global_base + c];
        }
    }

    output
}

/// Run the variable-context LM backbone for exactly one new token.
/// `position` is absolute and must advance by one for each consumed token.
pub fn forward_layers_incremental(
    layers: &[Layer],
    params: &ParameterStore,
    input: &[f32],
    position: usize,
    states: &mut [LayerInferenceState],
) -> Vec<f32> {
    assert_eq!(
        layers.len(),
        states.len(),
        "Incremental state does not match the backbone layer count"
    );

    let mut current = input.to_vec();

    for (layer, state) in layers.iter().zip(states.iter_mut()) {
        current = forward_layer_incremental(
            layer,
            params,
            &current,
            position,
            state,
        );
    }

    current
}

fn forward_layer_incremental(
    layer: &Layer,
    params: &ParameterStore,
    input: &[f32],
    position: usize,
    state: &mut LayerInferenceState,
) -> Vec<f32> {
    match layer {
        Layer::Conv1D(layer) => {
            let history = match state {
                LayerInferenceState::CausalConvHistory { history } => history,
                _ => panic!("Conv1D incremental state mismatch"),
            };

            let channels = layer.in_channels;
            let kernel_size = layer.kernel_size;
            let kernel_width = kernel_size * channels;
            let weights = params.values(layer.weights);
            let biases = params.values(layer.biases);

            debug_assert_eq!(input.len(), channels);
            debug_assert_eq!(weights.len(), layer.out_channels * kernel_width);
            debug_assert_eq!(biases.len(), layer.out_channels);

            let history_positions = history.len() / channels;
            let leading_padding = kernel_size.saturating_sub(history_positions + 1);
            let mut output = vec![0.0f32; layer.out_channels];

            for oc in 0..layer.out_channels {
                let mut sum = biases[oc];
                let output_weight_base = oc * kernel_width;

                for k in leading_padding..kernel_size {
                    let Some(source) = causal_window_source(
                        history,
                        input,
                        channels,
                        leading_padding,
                        k,
                    ) else {
                        continue;
                    };

                    let weight_base = output_weight_base + k * channels;
                    for ic in 0..channels {
                        sum += source[ic] * weights[weight_base + ic];
                    }
                }

                output[oc] = layer.activation.apply(sum);
            }

            append_causal_history(history, input, channels, kernel_size);
            output
        }

        Layer::DepthwiseConv1D(layer) => {
            let history = match state {
                LayerInferenceState::CausalConvHistory { history } => history,
                _ => panic!("DepthwiseConv1D incremental state mismatch"),
            };

            let channels = layer.in_channels;
            let kernel_size = layer.kernel_size;
            let weights = params.values(layer.weights);
            let biases = params.values(layer.biases);

            debug_assert_eq!(input.len(), channels);
            debug_assert_eq!(weights.len(), channels * kernel_size);
            debug_assert_eq!(biases.len(), channels);

            let history_positions = history.len() / channels;
            let leading_padding = kernel_size.saturating_sub(history_positions + 1);
            let mut output = vec![0.0f32; channels];

            for c in 0..channels {
                let mut sum = biases[c];

                for k in leading_padding..kernel_size {
                    let Some(source) = causal_window_source(
                        history,
                        input,
                        channels,
                        leading_padding,
                        k,
                    ) else {
                        continue;
                    };

                    sum += source[c] * weights[c * kernel_size + k];
                }

                output[c] = layer.activation.apply(sum);
            }

            append_causal_history(history, input, channels, kernel_size);
            output
        }

        Layer::GroupedConv1D(layer) => {
            let history = match state {
                LayerInferenceState::CausalConvHistory { history } => history,
                _ => panic!("GroupedConv1D incremental state mismatch"),
            };

            let channels = layer.in_channels;
            let kernel_size = layer.kernel_size;
            let group_in = channels / layer.groups;
            let group_out = layer.out_channels / layer.groups;
            let kernel_width = group_in * kernel_size;
            let weights = params.values(layer.weights);
            let biases = params.values(layer.biases);

            debug_assert_eq!(input.len(), channels);
            debug_assert_eq!(
                weights.len(),
                layer.out_channels * kernel_width
            );
            debug_assert_eq!(biases.len(), layer.out_channels);

            let history_positions = history.len() / channels;
            let leading_padding = kernel_size.saturating_sub(history_positions + 1);
            let mut output = vec![0.0f32; layer.out_channels];

            for group in 0..layer.groups {
                let input_group_base = group * group_in;
                let output_group_base = group * group_out;
                let weight_group_base = group * group_out * kernel_width;

                for oc in 0..group_out {
                    let output_channel = output_group_base + oc;
                    let weight_output_base =
                        weight_group_base + oc * kernel_width;
                    let mut sum = biases[output_channel];

                    for k in leading_padding..kernel_size {
                        let Some(source) = causal_window_source(
                            history,
                            input,
                            channels,
                            leading_padding,
                            k,
                        ) else {
                            continue;
                        };

                        let weight_base = weight_output_base + k * group_in;
                        for ic in 0..group_in {
                            sum += source[input_group_base + ic]
                                * weights[weight_base + ic];
                        }
                    }

                    output[output_channel] = layer.activation.apply(sum);
                }
            }

            append_causal_history(history, input, channels, kernel_size);
            output
        }

        Layer::Residual(layer) => {
            let inner_states = match state {
                LayerInferenceState::Residual { inner } => inner,
                _ => panic!("Residual incremental state mismatch"),
            };

            let mut output = forward_layers_incremental(
                &layer.layers,
                params,
                input,
                position,
                inner_states,
            );

            assert_eq!(output.len(), input.len(), "Residual block changed token width");
            for (value, residual) in output.iter_mut().zip(input.iter()) {
                *value += *residual;
            }
            output
        }

        Layer::LowRankPointwise(layer) => {
            let first_weights = params.values(layer.first_weights);
            let first_biases = params.values(layer.first_biases);
            let second_weights = params.values(layer.second_weights);
            let second_biases = params.values(layer.second_biases);

            let (output, _) = low_rank_pointwise_forward(
                layer,
                input,
                1,
                1,
                first_weights,
                first_biases,
                second_weights,
                second_biases,
            );
            output
        }

        Layer::ChannelScale(layer) =>
            channel_scale_forward(layer, input, 1, params),

        Layer::LayerNorm(layer) => {
            let (output, _, _) = layer_norm_forward(layer, input, 1, params);
            output
        }

        Layer::GlobalMixer(layer) => {
            let (global_vectors, write_log_norm) = match state {
                LayerInferenceState::GlobalMixer {
                    global_vectors,
                    write_log_norm,
                } => (global_vectors, write_log_norm),
                _ => panic!("GlobalMixer incremental state mismatch"),
            };

            global_mixer_forward_incremental(
                layer,
                input,
                position,
                global_vectors,
                write_log_norm,
                params,
            )
        }

        Layer::Dense(_) => panic!(
            "Dense is not supported by variable-context incremental inference"
        ),

        Layer::WeightTying(_) => panic!(
            "WeightTying is the final LM projection and must not be in the incremental backbone"
        ),
    }
}

// ============================================================================
// Channel Scale
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn channel_scale_avx2(
    input: &[f32],
    output: &mut [f32],
    scales: &[f32],
    biases: &[f32],
    channels: usize,
) {
    unsafe {
        let positions =
            input.len() / channels;

        for position in 0..positions {
            let offset =
                position * channels;

            let mut c =
                0usize;

            while c + 8 <= channels {
                let x =
                    _mm256_loadu_ps(
                        input.as_ptr()
                            .add(offset + c)
                    );

                let s =
                    _mm256_loadu_ps(
                        scales.as_ptr()
                            .add(c)
                    );

                let b =
                    _mm256_loadu_ps(
                        biases.as_ptr()
                            .add(c)
                    );

                let y =
                    _mm256_fmadd_ps(
                        x,
                        s,
                        b,
                    );

                _mm256_storeu_ps(
                    output
                        .as_mut_ptr()
                        .add(offset + c),
                    y,
                );

                c += 8;
            }

            while c < channels {
                output[offset + c] =
                    input[offset + c]
                        * scales[c]
                        + biases[c];

                c += 1;
            }
        }
    }
}

pub fn channel_scale_simd(
    input: &[f32],
    output: &mut [f32],
    scales: &[f32],
    biases: &[f32],
    channels: usize,
) {
    debug_assert_eq!(
        input.len(),
        output.len()
    );

    debug_assert!(
        channels > 0
    );

    debug_assert_eq!(
        input.len() % channels,
        0
    );

    debug_assert_eq!(
        scales.len(),
        channels
    );

    debug_assert_eq!(
        biases.len(),
        channels
    );

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

    let positions =
        input.len() / channels;

    for position in 0..positions {
        let offset =
            position * channels;

        for c in 0..channels {
            output[offset + c] =
                input[offset + c]
                    * scales[c]
                    + biases[c];
        }
    }
}

pub fn channel_scale_forward(
    layer: &ChannelScaleLayer,
    input: &[f32],
    batch_size: usize,
    params: &ParameterStore,
) -> Vec<f32> {
    let scales =
        params.values(
            layer.scales
        );

    let biases =
        params.values(
            layer.biases
        );

    let sequence_size =
        input.len() / batch_size;

    debug_assert_eq!(
        sequence_size % layer.channels,
        0
    );

    let mut output =
        vec![0.0; input.len()];

    for b in 0..batch_size {
        let base =
            b * sequence_size;

        let end =
            base + sequence_size;

        channel_scale_simd(
            &input[base..end],
            &mut output[base..end],
            scales,
            biases,
            layer.channels,
        );
    }

    output
}

// ============================================================================
// Low-rank pointwise
// ============================================================================

pub fn low_rank_pointwise_forward(
    layer: &LowRankPointwiseLayer,
    input: &[f32],
    batch_size: usize,
    positions: usize,
    first_weights: &[f32],
    first_biases: &[f32],
    second_weights: &[f32],
    second_biases: &[f32],
) -> (
    Vec<f32>,
    Vec<f32>,
) {
    let rows =
        batch_size
            * positions;

    let mut hidden =
        vec![
            0.0;
            rows * layer.rank
        ];

    let mut output =
        vec![
            0.0;
            rows * layer.out_channels
        ];

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
            first_weights,
            layer.in_channels as i32,
            0.0,
            &mut hidden,
            layer.rank as i32,
        );
    }

    activation_bias_simd(
        &mut hidden,
        first_biases,
        layer.rank,
        &layer.activation,
    );

    unsafe {
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
            second_weights,
            layer.rank as i32,
            0.0,
            &mut output,
            layer.out_channels as i32,
        );
    }

    activation_bias_simd(
        &mut output,
        second_biases,
        layer.out_channels,
        &layer.activation,
    );

    (
        output,
        hidden,
    )
}

// ============================================================================
// LayerNorm
// ============================================================================

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
    unsafe {
        debug_assert_eq!(
            input.len(),
            output.len()
        );

        debug_assert_eq!(
            input.len(),
            gamma.len()
        );

        debug_assert_eq!(
            input.len(),
            beta.len()
        );

        let channels =
            input.len();

        let mut sum_vec =
            _mm256_setzero_ps();

        let mut c =
            0usize;

        while c + 8 <= channels {
            let x =
                _mm256_loadu_ps(
                    input.as_ptr()
                        .add(c)
                );

            sum_vec =
                _mm256_add_ps(
                    sum_vec,
                    x,
                );

            c += 8;
        }

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
            sum +=
                input[c];

            c += 1;
        }

        let channels_f32 =
            channels as f32;

        let mean =
            sum / channels_f32;

        let mean_vec =
            _mm256_set1_ps(
                mean
            );

        let mut variance_vec =
            _mm256_setzero_ps();

        c = 0;

        while c + 8 <= channels {
            let x =
                _mm256_loadu_ps(
                    input.as_ptr()
                        .add(c)
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
                input[c]
                    - mean;

            variance +=
                diff * diff;

            c += 1;
        }

        variance /=
            channels_f32;

        let inv_std =
            1.0f32
                / (variance + epsilon).sqrt();

        let inv_std_vec =
            _mm256_set1_ps(
                inv_std
            );

        let mut c =
            0usize;

        while c + 8 <= channels {
            let x =
                _mm256_loadu_ps(
                    input.as_ptr()
                        .add(c)
                );

            let g =
                _mm256_loadu_ps(
                    gamma.as_ptr()
                        .add(c)
                );

            let b =
                _mm256_loadu_ps(
                    beta.as_ptr()
                        .add(c)
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
                output.as_mut_ptr()
                    .add(c),
                y,
            );

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

        (
            mean,
            inv_std,
        )
    }
}

pub fn layer_norm_forward(
    layer: &LayerNormLayer,
    input: &[f32],
    batch_size: usize,
    params: &ParameterStore,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    debug_assert!(
        batch_size > 0
    );

    debug_assert!(
        layer.channels > 0
    );

    debug_assert_eq!(
        input.len() % batch_size,
        0
    );

    let channels =
        layer.channels;

    let epsilon =
        layer.epsilon;

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

    let gamma =
        params.values(
            layer.gamma
        );

    let beta =
        params.values(
            layer.beta
        );

    debug_assert_eq!(
        gamma.len(),
        channels
    );

    debug_assert_eq!(
        beta.len(),
        channels
    );

    let mut output =
        vec![
            0.0f32;
            input.len()
        ];

    let mut means =
        vec![
            0.0f32;
            group_count
        ];

    let mut inv_stds =
        vec![
            0.0f32;
            group_count
        ];

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
            if is_x86_feature_detected!(
                "avx2"
            ) {
                let (
                    mean,
                    inv_std,
                ) =
                    unsafe {
                        layer_norm_group_avx2(
                            input_group,
                            output_group,
                            gamma,
                            beta,
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
                / (variance + epsilon)
                .sqrt();

        inv_stds[group] =
            inv_std;

        for c in 0..channels {
            let normalized =
                (input_group[c] - mean)
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

// ============================================================================
// Weight tying
// ============================================================================

pub fn weight_tying_forward(
    layer: &WeightTyingLayer,
    params: &ParameterStore,
    input: &[f32],
    batch_size: usize,
    positions: usize,
) -> Vec<f32> {
    let embedding_dim =
        layer.embeddings.embedding_dim();

    let vocab_size =
        layer.embeddings.vocab_size();

    let rows =
        batch_size * positions;

    assert_eq!(
        input.len(),
        rows * embedding_dim,
        "Invalid WeightTying input size"
    );

    let weights =
        params.values(
            layer.embeddings.parameter_range()
        );

    debug_assert_eq!(
        weights.len(),
        vocab_size * embedding_dim
    );

    // [B * P, E] × [V, E]^T
    // → [B * P, V]
    let mut output =
        vec![
            0.0;
            rows * vocab_size
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            vocab_size as i32,
            embedding_dim as i32,
            1.0,
            input,
            embedding_dim as i32,
            weights,
            embedding_dim as i32,
            0.0,
            &mut output,
            vocab_size as i32,
        );
    }

    output
}

// ============================================================================
// Tiled tied-vocabulary softmax cross entropy
// ============================================================================
//
// Training-only LM head.
//
// Instead of materializing:
//
//     [rows, vocab]
//
// this computes the vocabulary projection in tiles and uses an online
// log-sum-exp recurrence:
//
//     logsumexp(x) = max + log(sum(exp(x - max)))
//
// The same vocabulary tiles are recomputed for the backward pass.
//
// This is analogous to FlashAttention's basic strategy:
// don't materialize the enormous intermediate matrix.
//
// `grad_hidden` receives dLoss/dHidden.
// `embedding_grads` receives the classifier-side gradient for the tied
// embedding matrix.
//
// Neither output is normalized by valid_count. The trainer performs the
// same normalization it already uses for the old implementation.
// ============================================================================

pub const TIED_VOCAB_TILE: usize = 256;

#[inline]
fn online_softmax_row_update(
    logits: &[f32],
    row_max: &mut f32,
    row_sum: &mut f32,
) {
    debug_assert!(!logits.is_empty());

    let mut tile_max =
        f32::NEG_INFINITY;

    for &x in logits {
        tile_max =
            tile_max.max(x);
    }

    let old_max =
        *row_max;

    let new_max =
        old_max.max(tile_max);

    let old_scale =
        if old_max.is_finite() {
            (old_max - new_max).exp()
        } else {
            0.0
        };

    let mut tile_sum =
        0.0f32;

    for &x in logits {
        tile_sum +=
            (x - new_max).exp();
    }

    *row_sum =
        *row_sum * old_scale
            + tile_sum;

    *row_max =
        new_max;
}

pub fn weight_tying_softmax_cross_entropy_tiled(
    hidden: &[f32],
    targets: &[u16],

    grad_hidden: &mut Vec<f32>,
    embedding_grads: &mut Vec<f32>,

    row_max: &mut Vec<f32>,
    row_sum: &mut Vec<f32>,
    target_logits: &mut Vec<f32>,

    tile_logits: &mut Vec<f32>,
    tile_embedding_grads: &mut Vec<f32>,

    batch_size: usize,
    positions: usize,
    embedding_dim: usize,
    vocab_size: usize,
    weights: &[f32],

    ignore_index: u16,
) -> (f32, usize) {
    debug_assert_eq!(
        hidden.len(),
        batch_size
            * positions
            * embedding_dim
    );

    debug_assert_eq!(
        targets.len(),
        batch_size * positions
    );

    debug_assert_eq!(
        weights.len(),
        vocab_size * embedding_dim
    );

    let rows =
        batch_size * positions;

    // ------------------------------------------------------------------------
    // Resize reusable buffers.
    // ------------------------------------------------------------------------

    grad_hidden.resize(
        rows * embedding_dim,
        0.0,
    );

    grad_hidden.fill(0.0);

    embedding_grads.resize(
        vocab_size * embedding_dim,
        0.0,
    );

    embedding_grads.fill(0.0);

    row_max.resize(
        rows,
        f32::NEG_INFINITY,
    );

    row_sum.resize(
        rows,
        0.0,
    );

    target_logits.resize(
        rows,
        0.0,
    );

    row_max.fill(
        f32::NEG_INFINITY
    );

    row_sum.fill(0.0);

    target_logits.fill(0.0);

    tile_logits.resize(
        rows * TIED_VOCAB_TILE,
        0.0,
    );

    tile_embedding_grads.resize(
        TIED_VOCAB_TILE * embedding_dim,
        0.0,
    );

    let mut valid_count =
        0usize;

    for &target in targets {
        if target != ignore_index {
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return (
            0.0,
            0,
        );
    }

    // ------------------------------------------------------------------------
    // PASS 1
    //
    // Compute the exact global softmax normalization without storing all
    // vocabulary logits.
    // ------------------------------------------------------------------------

    for vocab_start
    in (0..vocab_size).step_by(
        TIED_VOCAB_TILE
    )
    {
        let tile_vocab =
            (vocab_size - vocab_start)
                .min(TIED_VOCAB_TILE);

        let tile_logits_len =
            rows * tile_vocab;

        debug_assert!(
            tile_logits_len
                <= tile_logits.len()
        );

        // [rows, D] × [tile_vocab, D]^T
        //
        // -> [rows, tile_vocab]
        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                rows as i32,
                tile_vocab as i32,
                embedding_dim as i32,
                1.0,
                hidden,
                embedding_dim as i32,
                &weights[
                    vocab_start * embedding_dim
                        ..
                        (vocab_start + tile_vocab)
                            * embedding_dim
                    ],
                embedding_dim as i32,
                0.0,
                &mut tile_logits[
                    ..tile_logits_len
                    ],
                tile_vocab as i32,
            );
        }

        for row in 0..rows {
            let target =
                targets[row];

            if target == ignore_index {
                continue;
            }

            let base =
                row * tile_vocab;

            online_softmax_row_update(
                &tile_logits[
                    base..base + tile_vocab
                    ],
                &mut row_max[row],
                &mut row_sum[row],
            );

            let target =
                target as usize;

            if target >= vocab_start
                && target < vocab_start + tile_vocab
            {
                target_logits[row] =
                    tile_logits[
                        base
                            + target
                            - vocab_start
                        ];
            }
        }
    }

    // ------------------------------------------------------------------------
    // Loss.
    //
    // CE = logsumexp(logits) - target_logit
    // ------------------------------------------------------------------------

    let mut total_loss =
        0.0f32;

    for row in 0..rows {
        if targets[row] == ignore_index {
            continue;
        }

        let logsumexp =
            row_max[row]
                + row_sum[row].ln();

        total_loss +=
            logsumexp
                - target_logits[row];
    }

    // ------------------------------------------------------------------------
    // PASS 2
    //
    // Recompute each vocabulary tile.
    //
    // tile_logits is immediately overwritten with:
    //
    //     softmax(logits) - one_hot(target)
    //
    // so no separate dLogits buffer is necessary.
    // ------------------------------------------------------------------------

    for vocab_start
    in (0..vocab_size).step_by(
        TIED_VOCAB_TILE
    )
    {
        let tile_vocab =
            (vocab_size - vocab_start)
                .min(TIED_VOCAB_TILE);

        let tile_logits_len =
            rows * tile_vocab;

        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                rows as i32,
                tile_vocab as i32,
                embedding_dim as i32,
                1.0,
                hidden,
                embedding_dim as i32,
                &weights[
                    vocab_start * embedding_dim
                        ..
                        (vocab_start + tile_vocab)
                            * embedding_dim
                    ],
                embedding_dim as i32,
                0.0,
                &mut tile_logits[
                    ..tile_logits_len
                    ],
                tile_vocab as i32,
            );
        }

        // Convert logits into CE gradients in-place.
        for row in 0..rows {
            let target =
                targets[row];

            let base =
                row * tile_vocab;

            if target == ignore_index {
                tile_logits[
                    base..base + tile_vocab
                    ]
                    .fill(0.0);

                continue;
            }

            let inv_sum =
                1.0f32
                    / row_sum[row];

            let max_value =
                row_max[row];

            for c in 0..tile_vocab {
                tile_logits[
                    base + c
                    ] =
                    (
                        tile_logits[
                            base + c
                            ]
                            - max_value
                    )
                        .exp()
                        * inv_sum;
            }

            let target =
                target as usize;

            if target >= vocab_start
                && target < vocab_start + tile_vocab
            {
                tile_logits[
                    base
                        + target
                        - vocab_start
                    ] -= 1.0;
            }
        }

        // --------------------------------------------------------------------
        // dHidden += dLogits × E_tile
        //
        // [rows, tile_vocab] × [tile_vocab, D]
        //
        // -> [rows, D]
        // --------------------------------------------------------------------

        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                rows as i32,
                embedding_dim as i32,
                tile_vocab as i32,
                1.0,
                &tile_logits[
                    ..tile_logits_len
                    ],
                tile_vocab as i32,
                &weights[
                    vocab_start * embedding_dim
                        ..
                        (vocab_start + tile_vocab)
                            * embedding_dim
                    ],
                embedding_dim as i32,
                1.0,
                grad_hidden,
                embedding_dim as i32,
            );
        }

        // --------------------------------------------------------------------
        // dEmbedding_tile = dLogitsᵀ × Hidden
        //
        // [tile_vocab, rows] × [rows, D]
        //
        // -> [tile_vocab, D]
        // --------------------------------------------------------------------

        let tile_grad_len =
            tile_vocab
                * embedding_dim;

        tile_embedding_grads[
            ..tile_grad_len
            ]
            .fill(0.0);

        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::Ordinary,
                Transpose::None,
                tile_vocab as i32,
                embedding_dim as i32,
                rows as i32,
                1.0,
                &tile_logits[
                    ..tile_logits_len
                    ],
                tile_vocab as i32,
                hidden,
                embedding_dim as i32,
                0.0,
                &mut tile_embedding_grads[
                    ..tile_grad_len
                    ],
                embedding_dim as i32,
            );
        }

        let destination =
            &mut embedding_grads[
                vocab_start * embedding_dim
                    ..
                    (vocab_start + tile_vocab)
                        * embedding_dim
                ];

        destination.copy_from_slice(
            &tile_embedding_grads[
                ..tile_grad_len
                ]
        );
    }

    (
        total_loss,
        valid_count,
    )
}

pub fn weight_tying_forward_last(
    layer: &WeightTyingLayer,
    params: &ParameterStore,
    input: &[f32],
    batch_size: usize,
    positions: usize,
) -> Vec<f32> {
    let embedding_dim =
        layer.embeddings.embedding_dim();

    let vocab_size =
        layer.embeddings.vocab_size();

    assert_eq!(
        batch_size,
        1,
        "weight_tying_forward_last currently requires batch_size = 1"
    );

    assert_eq!(
        input.len(),
        positions * embedding_dim,
        "Invalid WeightTying input size"
    );

    assert!(
        positions > 0,
        "WeightTying requires at least one position"
    );

    let weights =
        params.values(
            layer.embeddings.parameter_range()
        );

    debug_assert_eq!(
        weights.len(),
        vocab_size * embedding_dim
    );

    // Only project the final hidden state:
    //
    // [1, E] × [V, E]^T
    // → [1, V]
    //
    // instead of:
    //
    // [P, E] × [V, E]^T
    // → [P, V]

    let last_start =
        (positions - 1)
            * embedding_dim;

    let last_hidden =
        &input[
            last_start
                ..
                last_start + embedding_dim
            ];

    let mut output =
        vec![
            0.0f32;
            vocab_size
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            1,
            vocab_size as i32,
            embedding_dim as i32,
            1.0,
            last_hidden,
            embedding_dim as i32,
            weights,
            embedding_dim as i32,
            0.0,
            &mut output,
            vocab_size as i32,
        );
    }

    output
}

// ============================================================================
// Layer stack
// ============================================================================

pub fn forward_layers_batch(
    layers: &[Layer],
    params: &ParameterStore,
    input: &[f32],
    batch_size: usize,
    input_size: usize,
    thread_pool: &ThreadPool,
) -> (
    Vec<f32>,
    usize,
    Vec<BatchLayerCache>,
) {
    let mut current =
        input.to_vec();

    let mut current_size =
        input_size;

    let mut caches =
        Vec::with_capacity(
            layers.len()
        );

    for layer in layers {
        let (
            output,
            output_size,
            cache,
        ) =
            forward_layer_batch(
                layer,
                params,
                &current,
                batch_size,
                current_size,
                thread_pool,
            );

        current =
            output;

        current_size =
            output_size;

        caches.push(cache);
    }

    (
        current,
        current_size,
        caches,
    )
}

fn forward_layer_batch(
    layer: &Layer,
    params: &ParameterStore,
    input: &[f32],
    batch_size: usize,
    input_size: usize,
    thread_pool: &ThreadPool,
) -> (
    Vec<f32>,
    usize,
    BatchLayerCache,
) {
    match layer {
        // ====================================================================
        // Dense
        // ====================================================================

        Layer::Dense(layer) => {
            let output_size =
                layer.output_size;

            assert_eq!(
                input.len(),
                batch_size
                    * input_size
            );

            let weights =
                params.values(
                    layer.weights
                );

            let biases =
                params.values(
                    layer.biases
                );

            debug_assert_eq!(
                weights.len(),
                input_size
                    * output_size
            );

            debug_assert_eq!(
                biases.len(),
                output_size
            );

            let mut output =
                vec![
                    0.0;
                    batch_size
                        * output_size
                ];

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
                    weights,
                    input_size as i32,
                    0.0,
                    &mut output,
                    output_size as i32,
                );
            }

            activation_bias_simd(
                &mut output,
                biases,
                output_size,
                &layer.activation,
            );

            let activation_output =
                match layer.activation {
                    Activation::None =>
                        None,

                    _ =>
                        Some(
                            output.clone()
                        ),
                };

            let cache =
                BatchLayerCache::Dense {
                    input_size,
                    output_size,
                    input: input.to_vec(),
                    activation_output,
                    activation:
                    layer.activation,
                };

            (
                output,
                output_size,
                cache,
            )
        }

        // ====================================================================
        // Conv1D
        // ====================================================================

        Layer::Conv1D(layer) => {
            assert_eq!(
                input_size
                    % layer.in_channels,
                0
            );

            let input_length =
                input_size
                    / layer.in_channels;

            let output_length =
                layer.output_length(
                    input_length
                );

            let weights =
                params.values(
                    layer.weights
                );

            let biases =
                params.values(
                    layer.biases
                );

            let output =
                conv1d_forward(
                    layer,
                    weights,
                    biases,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                );

            let output_size =
                output_length
                    * layer.out_channels;

            let cache =
                BatchLayerCache::Conv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels:
                    layer.in_channels,
                    out_channels:
                    layer.out_channels,
                    kernel_size:
                    layer.kernel_size,
                    stride:
                    layer.stride,
                    padding:
                    layer.padding,
                    causal:
                    layer.causal,
                    activation:
                    layer.activation,
                };

            (
                output,
                output_size,
                cache,
            )
        }

        // ====================================================================
        // Residual
        // ====================================================================

        Layer::Residual(layer) => {
            let channels =
                layer.input_size;

            assert!(
                channels > 0,
                "Residual channel count must be > 0"
            );

            assert_eq!(
                input_size % channels,
                0,
                "Residual channels must divide input size"
            );

            let (
                inner_output,
                inner_output_size,
                inner_caches,
            ) =
                forward_layers_batch(
                    &layer.layers,
                    params,
                    input,
                    batch_size,
                    input_size,
                    thread_pool,
                );

            assert_eq!(
                inner_output_size,
                input_size,
                "Residual block changed tensor size"
            );

            let mut output =
                inner_output;

            add_f32_slice_simd(
                &mut output,
                input,
            );

            let cache =
                BatchLayerCache::Residual {
                    inner:
                    inner_caches,
                };

            (
                output,
                input_size,
                cache,
            )
        }

        // ====================================================================
        // Depthwise Conv1D
        // ====================================================================

        Layer::DepthwiseConv1D(layer) => {
            assert_eq!(
                input_size
                    % layer.in_channels,
                0
            );

            let input_length =
                input_size
                    / layer.in_channels;

            let output_length =
                layer.output_length(
                    input_length
                );

            let weights =
                params.values(
                    layer.weights
                );

            let biases =
                params.values(
                    layer.biases
                );

            let output =
                depthwise_conv1d_forward(
                    layer,
                    weights,
                    biases,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                    thread_pool,
                );

            let output_size =
                output_length
                    * layer.in_channels;

            let cache =
                BatchLayerCache::DepthwiseConv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels:
                    layer.in_channels,
                    kernel_size:
                    layer.kernel_size,
                    stride:
                    layer.stride,
                    padding:
                    layer.padding,
                    causal:
                    layer.causal,
                    activation:
                    layer.activation,
                };

            (
                output,
                output_size,
                cache,
            )
        }

        // ====================================================================
        // Grouped Conv1D
        // ====================================================================

        Layer::GroupedConv1D(layer) => {
            assert_eq!(
                input_size
                    % layer.in_channels,
                0
            );

            let input_length =
                input_size
                    / layer.in_channels;

            let output_length =
                layer.output_length(
                    input_length
                );

            let weights =
                params.values(
                    layer.weights
                );

            let biases =
                params.values(
                    layer.biases
                );

            let output =
                grouped_conv1d_forward(
                    layer,
                    weights,
                    biases,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                );

            let output_size =
                output_length
                    * layer.out_channels;

            let cache =
                BatchLayerCache::GroupedConv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels:
                    layer.in_channels,
                    out_channels:
                    layer.out_channels,
                    groups:
                    layer.groups,
                    kernel_size:
                    layer.kernel_size,
                    stride:
                    layer.stride,
                    padding:
                    layer.padding,
                    causal:
                    layer.causal,
                    activation:
                    layer.activation,
                };

            (
                output,
                output_size,
                cache,
            )
        }

        // ====================================================================
        // Low-Rank Pointwise
        // ====================================================================

        Layer::LowRankPointwise(layer) => {
            assert_eq!(
                input_size
                    % layer.in_channels,
                0
            );

            let positions =
                input_size
                    / layer.in_channels;

            let first_weights =
                params.values(
                    layer.first_weights
                );

            let first_biases =
                params.values(
                    layer.first_biases
                );

            let second_weights =
                params.values(
                    layer.second_weights
                );

            let second_biases =
                params.values(
                    layer.second_biases
                );

            let (
                output,
                hidden,
            ) =
                low_rank_pointwise_forward(
                    layer,
                    input,
                    batch_size,
                    positions,
                    first_weights,
                    first_biases,
                    second_weights,
                    second_biases,
                );

            let output_size =
                positions
                    * layer.out_channels;

            let rows =
                batch_size
                    * positions;

            let cache =
                BatchLayerCache::LowRankPointwise {
                    input: input.to_vec(),
                    hidden,
                    output: output.clone(),
                    rows,
                    in_channels:
                    layer.in_channels,
                    rank:
                    layer.rank,
                    out_channels:
                    layer.out_channels,
                    first_weights:
                    first_weights.to_vec(),
                    second_weights:
                    second_weights.to_vec(),
                    activation:
                    layer.activation,
                };

            (
                output,
                output_size,
                cache,
            )
        }

        // ====================================================================
        // Channel Scale
        // ====================================================================

        Layer::ChannelScale(layer) => {
            assert_eq!(
                input_size
                    % layer.channels,
                0
            );

            let output =
                channel_scale_forward(
                    layer,
                    input,
                    batch_size,
                    params,
                );

            let cache =
                BatchLayerCache::ChannelScale {
                    input: input.to_vec(),
                    channels:
                    layer.channels,
                };

            (
                output,
                input_size,
                cache,
            )
        }

        // ====================================================================
        // LayerNorm
        // ====================================================================

        Layer::LayerNorm(layer) => {
            assert_eq!(
                input_size
                    % layer.channels,
                0,
                "LayerNorm channels must divide input size"
            );

            let (
                output,
                means,
                inv_stds,
            ) =
                layer_norm_forward(
                    layer,
                    input,
                    batch_size,
                    params,
                );

            let cache =
                BatchLayerCache::LayerNorm {
                    input: input.to_vec(),
                    means,
                    inv_stds,
                    channels:
                    layer.channels,
                };

            (
                output,
                input_size,
                cache,
            )
        }

        // ====================================================================
        // Weight Tying
        // ====================================================================

        Layer::WeightTying(layer) => {
            let embedding_dim =
                layer.embeddings
                    .embedding_dim();

            let vocab_size =
                layer.embeddings
                    .vocab_size();

            assert_eq!(
                input_size % embedding_dim,
                0,
                "WeightTying input size must be divisible by embedding dimension"
            );

            let positions =
                input_size / embedding_dim;

            let output =
                weight_tying_forward(
                    layer,
                    params,
                    input,
                    batch_size,
                    positions,
                );

            let cache =
                BatchLayerCache::WeightTying {
                    input: input.to_vec(),
                    embeddings:
                    Arc::clone(
                        &layer.embeddings
                    ),
                    batch_size,
                    positions,
                    embedding_dim,
                    vocab_size,
                };

            (
                output,
                positions * vocab_size,
                cache,
            )
        }

        // ====================================================================
        // Global Mixer
        // ====================================================================

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
                input_size
                    % layer.channels,
                0,
                "GlobalMixer channels must divide input size"
            );

            let positions =
                input_size
                    / layer.channels;

            let write_weights =
                params.values(
                    layer.write_weights
                );

            let write_biases =
                params.values(
                    layer.write_biases
                );

            let read_weights =
                params.values(
                    layer.read_weights
                );

            let read_biases =
                params.values(
                    layer.read_biases
                );

            let write_positional_weights =
                params.values(
                    layer.write_positional_weights
                );

            let read_positional_weights =
                params.values(
                    layer.read_positional_weights
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

            debug_assert_eq!(
                write_biases.len(),
                layer.global_dim
            );

            debug_assert_eq!(
                read_biases.len(),
                layer.global_dim
            );

            let positional_count =
                layer.global_dim
                    * GLOBAL_MIXER_POS_FEATURES;

            debug_assert_eq!(
                write_positional_weights.len(),
                positional_count
            );

            debug_assert_eq!(
                read_positional_weights.len(),
                positional_count
            );

            let (
                output,
                write_probs,
                global_vectors,
                read_probs,
                write_log_norm,
                position_features,
            ) =
                global_mixer_forward(
                    layer,
                    input,
                    batch_size,
                    positions,
                    0, // Training sequences begin at absolute position zero.
                    write_weights,
                    write_biases,
                    read_weights,
                    read_biases,
                    write_positional_weights,
                    read_positional_weights,
                );

            let cache =
                BatchLayerCache::GlobalMixer {
                    input: input.to_vec(),
                    write_probs,
                    global_vectors,
                    read_probs,
                    write_log_norm,
                    position_features,
                    positions,
                    channels: layer.channels,
                    global_dim: layer.global_dim,
                };

            (
                output,
                input_size,
                cache,
            )
        }
    }
}


#[cfg(test)]
mod global_mixer_position_tests {
    use super::global_mixer_position_features;

    #[test]
    fn absolute_position_features_are_chunk_invariant() {
        let full = global_mixer_position_features(512, 0);
        let chunk = global_mixer_position_features(128, 128);

        assert_eq!(&full[128..256], &chunk[..]);
    }

    #[test]
    fn position_features_are_finite_at_long_offsets() {
        let features = global_mixer_position_features(4, 1_000_000);

        assert!(
            features.iter().flatten().all(|value| value.is_finite()),
            "GlobalMixer positional features must remain finite"
        );
    }
}
