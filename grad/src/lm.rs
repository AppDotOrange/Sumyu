use crate::embeddings::{
    Embeddings,
    SavedEmbeddings,
};
use crate::helper::{
    Config,
    HybridConfig,
};
use crate::neuron::{
    Activation,
    LayerSpec,
    MLP,
    MLPInferenceState,
    SavedMLP,
    BatchLayerCache
};
use crate::parameters::ParameterStore;
use crate::trainer::{
    CheckpointFrequency,
    CheckpointKind,
    CheckpointState,
    LmLossMask,
    LrSchedule,
    PermutationSampler,
    ResumeState,
    Trainer,
};
pub use crate::miscellaneous::{
    append_decoded_token_bytes,
    decode_token_stream,
    print_layer_specs,
    stream_token,
};

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rng;
use serde::{
    Deserialize,
    Serialize,
};
use std::fs;
use std::io::Write;
use std::ops::Range;
use std::sync::Arc;
use std::path::Path;

// ============================================================================
// Saved model
// ============================================================================

///
/// A struct that contains everything needed for LM inference, but not training.
///
/// # Contents:
/// * description: String
/// * mlp: SavedMLP
/// * vocab: Vec<String>
/// * context_len: u32
/// * hidden_layers: Vec<usize>
/// * embeddings: SavedEmbeddings
///
/// This struct is serialized into a .sumyu file using serde and bincode.
#[derive(Serialize, Deserialize)]
pub struct SavedLM {
    description: String,
    mlp: SavedMLP,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: SavedEmbeddings,
}

#[derive(Serialize, Deserialize)]
pub struct SavedCheckpoint {
    pub model: SavedLM,

    pub epoch: usize,
    pub batch: usize,
    pub sample: usize,

    pub sampler_seed: u64,
    pub sampler_version: u32,
    pub sampler_data_len: usize,

    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,

    #[serde(default)]
    pub adam_first_moments: Vec<f32>,

    #[serde(default)]
    pub adam_second_moments: Vec<f32>,

    #[serde(default)]
    pub adam_step: u64,
}

#[derive(Deserialize)]
pub struct SavedCheckpointV1 {
    pub model: SavedLM,

    pub epoch: usize,
    pub batch: usize,
    pub sample: usize,

    pub sampler_seed: u64,
    pub sampler_version: u32,
    pub sampler_data_len: usize,

    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,

    #[serde(default)]
    pub layer_second_moments: Vec<f32>,

    #[serde(default)]
    pub layer_first_moments: Vec<f32>,

    #[serde(default)]
    pub layer_lr_scales: Vec<f32>,

    #[serde(default)]
    pub layer_search_direction: Vec<f32>,

    #[serde(default)]
    pub layer_search_factor: Vec<f32>,

    #[serde(default)]
    pub layer_adaptive_step: u64,

    #[serde(default)]
    pub embedding_first_moments: Vec<f32>,

    #[serde(default)]
    pub embedding_second_moments: Vec<f32>,

    #[serde(default)]
    pub embedding_lr_scale: f32,

    #[serde(default)]
    pub embedding_search_direction: f32,

    #[serde(default)]
    pub embedding_search_factor: f32,
}

// ============================================================================
// Tokenization
// ============================================================================

pub fn tokenize(
    text: &str,
    vocab: &[String],
) -> Vec<usize> {
    let trie =
        crate::helper::Trie::from_vocab(
            vocab,
            byte_fallback_base(vocab),
        );

    trie.tokenize_u32(text)
        .into_iter()
        .map(|x| x as usize)
        .collect()
}

pub fn tokenize_u16(
    text: &str,
    vocab: &[String],
) -> Vec<u16> {
    let trie =
        crate::helper::Trie::from_vocab(
            vocab,
            byte_fallback_base(vocab),
        );

    trie.tokenize_bytes_u16(
        text.as_bytes()
    )
}

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,

    /// 1.0 disables repetition penalty.
    /// Values above 1 discourage repetition; values below 1 encourage it.
    pub repetition_penalty: f32,

    /// Number of recent tokens considered.
    /// 0 = entire available history.
    pub repetition_window: usize,

    /// 0 = disabled.
    pub top_k: usize,

    /// Probability mass to retain. Must be in (0, 1].
    pub top_p: f32,

    /// Minimum probability relative to the most likely token.
    /// 0 = disabled.
    pub min_p: f32,
}

impl SamplingConfig {
    /// A sensible starting preset for general generation.
    pub const fn recommended(temperature: f32) -> Self {
        Self {
            temperature,
            repetition_penalty: 1.08,
            repetition_window: 64,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
        }
    }

    /// Reproduces temperature-only sampling.
    pub const fn temperature_only(temperature: f32) -> Self {
        Self {
            temperature,
            repetition_penalty: 1.0,
            repetition_window: 0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }

    fn validate(self) {
        assert!(
            self.temperature.is_finite(),
            "Temperature must be finite"
        );

        assert!(
            self.repetition_penalty.is_finite()
                && self.repetition_penalty > 0.0,
            "Repetition penalty must be finite and positive"
        );

        assert!(
            self.top_p.is_finite()
                && self.top_p > 0.0
                && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );

        assert!(
            self.min_p.is_finite()
                && self.min_p >= 0.0
                && self.min_p <= 1.0,
            "min_p must be in [0, 1]"
        );
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::recommended(0.7)
    }
}

fn sample_logits_with_config(
    logits: &[f32],
    history: &[u16],
    config: SamplingConfig,
) -> usize {
    assert!(
        !logits.is_empty(),
        "Cannot sample from empty logits"
    );

    config.validate();

    assert!(
        logits.iter().all(|x| x.is_finite()),
        "Cannot sample from non-finite logits"
    );

    // ---------------------------------------------------------
    // 1. Copy logits into f64.
    // ---------------------------------------------------------

    let mut adjusted: Vec<f64> =
        logits.iter()
            .map(|&x| x as f64)
            .collect();

    // ---------------------------------------------------------
    // 2. Repetition penalty.
    //
    // Apply once per unique token in the recent history.
    //
    // Positive logits are divided by the penalty.
    // Negative logits are multiplied by the penalty.
    // ---------------------------------------------------------

    if config.repetition_penalty != 1.0 {
        let mut seen = vec![false; adjusted.len()];

        let history_slice =
            if config.repetition_window == 0 {
                history
            } else {
                let start =
                    history.len().saturating_sub(
                        config.repetition_window
                    );

                &history[start..]
            };

        let penalty =
            config.repetition_penalty as f64;

        for &token_id in history_slice {
            let id = token_id as usize;

            if id >= adjusted.len() || seen[id] {
                continue;
            }

            seen[id] = true;

            let logit = &mut adjusted[id];

            if *logit < 0.0 {
                *logit *= penalty;
            } else {
                *logit /= penalty;
            }
        }
    }

    // ---------------------------------------------------------
    // 3. Greedy decoding.
    // ---------------------------------------------------------

    if config.temperature <= 0.0 {
        return adjusted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
    }

    // ---------------------------------------------------------
    // 4. Temperature.
    // ---------------------------------------------------------

    let temperature =
        config.temperature as f64;

    let scaled: Vec<f64> =
        adjusted.iter()
            .map(|&x| x / temperature)
            .collect();

    // ---------------------------------------------------------
    // 5. Candidate selection.
    // ---------------------------------------------------------

    let mut candidates: Vec<usize> =
        (0..scaled.len()).collect();

    // Top-k uses partial selection rather than sorting
    // the entire vocabulary.
    if config.top_k > 0
        && config.top_k < candidates.len()
    {
        candidates.select_nth_unstable_by(
            config.top_k,
            |a, b| {
                scaled[*b].total_cmp(
                    &scaled[*a]
                )
            }
        );

        candidates.truncate(config.top_k);
    }

    // Nucleus sampling requires sorted candidates.
    // When top-k is enabled, this only sorts the top-k set.
    if config.top_p < 1.0 {
        candidates.sort_unstable_by(
            |a, b| {
                scaled[*b].total_cmp(
                    &scaled[*a]
                )
            }
        );
    }

    // ---------------------------------------------------------
    // 6. Min-p filtering.
    //
    // Retain tokens satisfying:
    //
    // P(token) >= min_p * P(best_token)
    //
    // Comparing scaled logits avoids computing softmax.
    // ---------------------------------------------------------

    if config.min_p > 0.0 {
        let max_logit =
            candidates.iter()
                .map(|&i| scaled[i])
                .fold(f64::NEG_INFINITY, f64::max);

        let threshold =
            max_logit + (config.min_p as f64).ln();

        candidates.retain(
            |&i| scaled[i] >= threshold
        );
    }

    // ---------------------------------------------------------
    // 7. Stable softmax over remaining candidates.
    // ---------------------------------------------------------

    let max_logit =
        candidates.iter()
            .map(|&i| scaled[i])
            .fold(f64::NEG_INFINITY, f64::max);

    let mut weights: Vec<f64> =
        candidates.iter()
            .map(|&i| {
                (scaled[i] - max_logit).exp()
            })
            .collect();

    // ---------------------------------------------------------
    // 8. Top-p filtering.
    //
    // Keep the smallest descending-probability prefix
    // whose cumulative probability reaches top_p.
    // ---------------------------------------------------------

    if config.top_p < 1.0 {
        let total_weight: f64 =
            weights.iter().sum();

        let target =
            total_weight * config.top_p as f64;

        let mut cumulative = 0.0;
        let mut keep_count = weights.len();

        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;

            if cumulative >= target {
                keep_count = i + 1;
                break;
            }
        }

        candidates.truncate(keep_count);
        weights.truncate(keep_count);
    }

    // ---------------------------------------------------------
    // 9. Weighted random sampling.
    // ---------------------------------------------------------

    let distribution =
        WeightedIndex::new(&weights)
            .expect("Invalid sampling weights");

    let mut random = rng();

    let sampled_position =
        distribution.sample(&mut random);

    candidates[sampled_position]
}

fn byte_fallback_base(vocab: &[String]) -> u32 {
    let base =
        vocab
            .iter()
            .position(
                |t| t == "<0x00>"
            )
            .expect(
                "Vocabulary missing <0x00>"
            ) as u32;

    debug_assert!(
        base + 256
            <= vocab.len() as u32,
        "Byte fallback block incomplete"
    );

    base
}

// ============================================================================
// LM
// ============================================================================

/// Persistent state for token-by-token LM inference.
///
/// The state is model-specific: create it from an LM and keep using it with
/// that same LM instance until the sequence is reset.
#[derive(Clone)]
pub struct LMInferenceState {
    mlp_state: MLPInferenceState,
}

impl LMInferenceState {
    pub fn position(&self) -> usize {
        self.mlp_state.position()
    }
}

fn build_tokenizer_data(
    vocab: &[String],
) -> (Arc<crate::helper::Trie>, usize) {
    let byte_fallback = byte_fallback_base(vocab);

    let trie = Arc::new(
        crate::helper::Trie::from_vocab(
            vocab,
            byte_fallback,
        )
    );

    let max_token_bytes = trie.max_token_bytes();

    (trie, max_token_bytes)
}

#[derive(Clone)]
pub struct LM {
    trainer: Trainer,
    mlp: MLP,

    // Tokenized training dataset.
    dataset: Vec<u16>,

    // Optional loss mask for JSON datasets.
    //
    // None = ordinary LM training.
    // Some(...) = only selected target ranges contribute to loss.
    dataset_loss_mask: Option<LmLossMask>,

    // Optional hard boundaries between independently sampled JSON chats.
    // None = one ordinary continuous token stream.
    dataset_chat_ranges: Option<Vec<Range<usize>>>,

    vocab: Vec<String>,
    // Shared tokenizer trie used by text encoding and generated-token merging.
    // Keeping it here avoids rebuilding a vocabulary-sized trie during generation.
    tokenizer_trie: Arc<crate::helper::Trie>,
    max_token_bytes: usize,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: Arc<Embeddings>,
}

impl LM {
    pub fn new(
        context_len: u32,
        vocab: Vec<String>,
        hidden_dim: &[usize],
        embedding_dim: usize,
    ) -> Self {
        let input_size =
            context_len as usize
                * embedding_dim;

        let mut specs = Vec::with_capacity(hidden_dim.len() + 1);

        for &size in hidden_dim {
            specs.push(
                LayerSpec::Dense {
                    output_size: size,
                    activation:
                    Activation::LeakyReLU {
                        slope: 0.01,
                    },
                }
            );
        }

        specs.push(
            LayerSpec::Dense {
                output_size: vocab.len(),
                activation:
                Activation::None,
            }
        );

        let mut params =
            ParameterStore::new();

        let embeddings =
            Arc::new(
                Embeddings::new(
                    &mut params,
                    vocab.len(),
                    embedding_dim,
                )
            );

        let mlp =
            MLP::from_layers_with_embeddings(
                input_size,
                &specs,
                Arc::clone(&embeddings),
                params,
            );

        let trainer =
            Trainer::new(
                0.0,
                0,
                0,
                0,
            );

        let (tokenizer_trie, max_token_bytes) = build_tokenizer_data(&vocab);

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            dataset_loss_mask: None,
            dataset_chat_ranges: None,
            vocab,
            tokenizer_trie,
            max_token_bytes,
            context_len,
            hidden_layers:
            hidden_dim.to_vec(),
            embeddings,
        }
    }

    // ------------------------------------------------------------------------
    // Layer-based LM
    // ------------------------------------------------------------------------

    pub fn from_layers(
        context_len: u32,
        vocab: Vec<String>,
        layers: &[LayerSpec],
        embedding_dim: usize,
    ) -> Self {
        let mut params =
            ParameterStore::new();

        let embeddings =
            Arc::new(
                Embeddings::new(
                    &mut params,
                    vocab.len(),
                    embedding_dim,
                )
            );

        let mlp =
            MLP::from_layers_with_embeddings(
                embedding_dim,
                layers,
                Arc::clone(&embeddings),
                params,
            );

        let trainer =
            Trainer::new(
                0.0,
                0,
                0,
                0,
            );

        let (tokenizer_trie, max_token_bytes) = build_tokenizer_data(&vocab);

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            dataset_loss_mask: None,
            dataset_chat_ranges: None,
            vocab,
            tokenizer_trie,
            max_token_bytes,
            context_len,
            hidden_layers: Vec::new(),
            embeddings,
        }
    }

    pub fn from_config(
        config: Config,
    ) -> Self {
        let mut specs =
            Vec::with_capacity(
                config.hidden_dim.len() + 1
            );

        for &size in config.hidden_dim {
            specs.push(
                LayerSpec::Dense {
                    output_size: size,
                    activation:
                    Activation::LeakyReLU {
                        slope: 0.01,
                    },
                }
            );
        }

        specs.push(
            LayerSpec::Dense {
                output_size: config.vocab.len(),
                activation:
                Activation::None,
            }
        );

        let mut params =
            ParameterStore::new();

        let embeddings =
            Arc::new(
                Embeddings::new(
                    &mut params,
                    config.vocab.len(),
                    config.emb_dim,
                )
            );

        let mlp =
            MLP::from_layers_with_embeddings(
                config.context_len
                    * config.emb_dim,
                &specs,
                Arc::clone(&embeddings),
                params,
            );

        let trainer =
            Trainer::new(
                config.lr,
                config.epochs,
                config.batch_size,
                config.max_batches_per_epoch,
            );

        let (tokenizer_trie, max_token_bytes) = build_tokenizer_data(&config.vocab);

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            dataset_loss_mask: None,
            dataset_chat_ranges: None,
            vocab: config.vocab,
            tokenizer_trie,
            max_token_bytes,
            context_len:
            config.context_len as u32,
            hidden_layers:
            config.hidden_dim.to_vec(),
            embeddings,
        }
    }

    pub fn from_hybrid_config(
        config: HybridConfig,
    ) -> Self {
        let mut params =
            ParameterStore::new();

        let embeddings =
            Arc::new(
                Embeddings::new(
                    &mut params,
                    config.vocab.len(),
                    config.emb_dim,
                )
            );

        let trainer =
            Trainer::new(
                config.lr,
                config.epochs,
                config.batch_size,
                config.max_batches_per_epoch,
            );

        let mlp =
            MLP::from_layers_with_embeddings(
                config.emb_dim,
                &config.layer_specs,
                Arc::clone(&embeddings),
                params,
            );

        let (tokenizer_trie, max_token_bytes) = build_tokenizer_data(&config.vocab);

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            dataset_loss_mask: None,
            dataset_chat_ranges: None,
            vocab: config.vocab,
            tokenizer_trie,
            max_token_bytes,
            context_len:
            config.context_len as u32,
            hidden_layers: Vec::new(),
            embeddings,
        }
    }

    pub fn reconfigure_context_length(&mut self, new_context_len: usize) {
        println!("Reconfiguring context length. This can be dangerous for a model's performance!");
        println!("OLD CONTEXT: {}", self.context_len);
        self.context_len = new_context_len as u32;
        println!("NEW CONTEXT: {}", self.context_len);
    }

    fn is_repetition_exempt(&self, token_id: u16) -> bool {
        matches!(
        self.vocab.get(token_id as usize).map(String::as_str),
        Some("<USER>") | Some("<BOT>") | Some("<EOT>")
    )
    }

    fn initial_repetition_history(
        &self,
        prompt_ids: &[u16],
        config: SamplingConfig,
    ) -> Vec<u16> {
        if config.repetition_penalty == 1.0 {
            return Vec::new();
        }

        let mut history: Vec<u16> =
            prompt_ids.iter()
                .copied()
                .filter(|&id| !self.is_repetition_exempt(id))
                .collect();

        if config.repetition_window > 0
            && history.len() > config.repetition_window
        {
            let excess =
                history.len() - config.repetition_window;

            history.drain(..excess);
        }

        history
    }

    fn remember_repetition_token(
        &self,
        history: &mut Vec<u16>,
        token_id: u16,
        config: SamplingConfig,
    ) {
        if config.repetition_penalty == 1.0
            || self.is_repetition_exempt(token_id)
        {
            return;
        }

        history.push(token_id);

        if config.repetition_window > 0
            && history.len() > config.repetition_window
        {
            let excess =
                history.len() - config.repetition_window;

            history.drain(..excess);
        }
    }

    // ------------------------------------------------------------------------
    // Checkpoint loading
    // ------------------------------------------------------------------------

    pub fn from_checkpoint(
        path: &str,
    ) -> Self {
        let bytes =
            fs::read(path)
                .unwrap();

        let (
            checkpoint,
            _,
        ):
            (SavedCheckpoint, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            )
                .unwrap();

        let vocab =
            checkpoint.model.vocab;

        let context_len =
            checkpoint.model.context_len;

        let hidden_layers =
            checkpoint
                .model
                .hidden_layers;

        let mut params =
            ParameterStore::new();

        let embeddings =
            Arc::new(
                Embeddings::load_into(
                    &mut params,
                    checkpoint
                        .model
                        .embeddings,
                )
            );

        let mlp =
            MLP::load_with_embeddings(
                &checkpoint.model.mlp,
                Arc::clone(
                    &embeddings
                ),
                params,
            );

        println!(
            "Loaded checkpoint: epoch {}, batch {}, sample {}.",
            checkpoint.epoch,
            checkpoint.batch,
            checkpoint.sample,
        );

        let trainer =
            Trainer::new(
                0.0,
                0,
                0,
                0,
            );

        let (tokenizer_trie, max_token_bytes) = build_tokenizer_data(&vocab);

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            dataset_loss_mask: None,
            dataset_chat_ranges: None,
            vocab,
            tokenizer_trie,
            max_token_bytes,
            context_len,
            hidden_layers,
            embeddings,
        }
    }

    // ------------------------------------------------------------------------
    // Basic metadata
    // ------------------------------------------------------------------------

    pub fn dataset_size(
        &self,
    ) -> usize {
        self.dataset.len()
    }

    pub fn encode_embeddings(
        &self,
        ids: &[u16],
    ) -> Vec<f32> {
        self.embeddings.encode(
            &self.mlp.params,
            ids,
        )
    }

    pub fn train_options(
        &mut self,
        lr: f32,
        epochs: usize,
        batch_size: usize,
        max_batches_per_epoch: usize,
    ) {
        self.trainer
            .reinit_batch(
                batch_size
            );

        self.trainer
            .reinit_lr(
                lr
            );

        self.trainer
            .reinit_epochs(
                epochs
            );

        self.trainer
            .reinit_batch_per_epoch(
                max_batches_per_epoch
            );
    }

    pub fn set_lr_schedule(
        &mut self,
        schedule: LrSchedule,
    ) {
        self.trainer
            .reinit_lr_schedule(
                schedule
            );
    }

    pub fn encode_nums(
        &self,
        string: String,
    ) -> Vec<u16> {
        self.tokenize_text(&string)
    }

    pub fn max(
        &self,
        vec: Vec<f32>,
    ) -> usize {
        let mut best = 0usize;
        let mut best_val = vec[0];
        for (
            i,
            &value
        ) in vec.iter()
            .enumerate()
            .skip(1)
        {
            if value > best_val {
                best = i;
                best_val = value;
            }
        }
        best
    }

    // ------------------------------------------------------------------------
    // Incremental inference
    // ------------------------------------------------------------------------

    /// Create fresh, bounded-memory inference state for this LM.
    ///
    /// This requires a variable-context model whose causal convolutions use
    /// stride 1. Dense, non-causal and strided convolution layers are rejected.
    pub fn new_inference_state(&self) -> Result<LMInferenceState, String> {
        Ok(LMInferenceState {
            mlp_state: self.mlp.new_lm_inference_state()?,
        })
    }

    /// Number of entries in this model's vocabulary.
    ///
    /// Special tokens represented by empty strings still occupy ordinary
    /// vocabulary IDs and must be supplied to chat/agent configs by ID.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Re-tokenize a sequence of generated token IDs using the same greedy
    /// byte-aware trie used for training text. This merges adjacent generated
    /// pieces whenever their concatenated bytes form a larger vocabulary token.
    ///
    /// Empty-string special tokens are hard boundaries: pending text bytes are
    /// flushed before them, and the special token ID is preserved unchanged.
    pub fn merge_token_ids(&self, ids: &[u16]) -> Result<Vec<u16>, String> {
        let mut merged = Vec::<u16>::with_capacity(ids.len());
        let mut pending_bytes = Vec::<u8>::new();

        for &id in ids {
            let token = self.vocab.get(id as usize).ok_or_else(|| {
                format!("Token ID {} is outside the vocabulary", id)
            })?;

            if token.is_empty() {
                self.tokenizer_trie
                    .tokenize_bytes_u16_into(&pending_bytes, &mut merged);
                pending_bytes.clear();
                merged.push(id);
            } else {
                append_decoded_token_bytes(
                    token,
                    &mut pending_bytes,
                );
            }
        }

        self.tokenizer_trie
            .tokenize_bytes_u16_into(&pending_bytes, &mut merged);
        Ok(merged)
    }

    /// Number of trailing token IDs whose segmentation may still change when
    /// more bytes are appended. The prefix before this boundary is guaranteed
    /// stable for the current vocabulary's maximum token byte length.
    /// Empty-string special tokens split text into independent tokenization runs.
    pub fn stable_token_prefix_len(&self, ids: &[u16]) -> usize {
        let run_start = ids
            .iter()
            .rposition(|&id| self.vocab[id as usize].is_empty())
            .map_or(0, |index| index + 1);

        if run_start == ids.len() {
            return ids.len();
        }

        let mut byte_buffer = Vec::<u8>::new();
        let mut run_bytes = 0usize;
        for &id in &ids[run_start..] {
            byte_buffer.clear();
            append_decoded_token_bytes(&self.vocab[id as usize], &mut byte_buffer);
            run_bytes = run_bytes.saturating_add(byte_buffer.len());
        }

        let stable_byte_end = run_bytes.saturating_sub(self.max_token_bytes);
        let mut stable_bytes = 0usize;
        let mut stable_count = run_start;

        for &id in &ids[run_start..] {
            byte_buffer.clear();
            append_decoded_token_bytes(&self.vocab[id as usize], &mut byte_buffer);
            stable_bytes = stable_bytes.saturating_add(byte_buffer.len());
            if stable_bytes > stable_byte_end {
                break;
            }
            stable_count += 1;
        }

        stable_count
    }

    /// Tokenize text using this model's vocabulary and byte fallback.
    /// Empty-string special tokens are intentionally not produced by text
    /// tokenization; inject their IDs directly into the token stream.
    pub fn tokenize_text(&self, text: &str) -> Vec<u16> {
        let mut ids = Vec::with_capacity(text.len().saturating_div(2));
        self.tokenizer_trie.tokenize_bytes_u16_into(text.as_bytes(), &mut ids);
        ids
    }

    /// Sample a token ID from model logits.
    pub fn sample_next_token(
        &self,
        logits: &[f32],
        temperature: f32,
    ) -> u16 {
        sample_logits_with_config(
            logits,
            &[],
            SamplingConfig::recommended(temperature),
        ) as u16
    }

    pub fn sample_next_token_with_history(
        &self,
        logits: &[f32],
        history: &[u16],
        config: SamplingConfig,
    ) -> u16 {
        sample_logits_with_config(
            logits,
            history,
            config,
        ) as u16
    }

    pub fn token_text(&self, token_id: u16) -> &str {
        self.vocab
            .get(token_id as usize)
            .expect("Token ID is outside the vocabulary")
    }

    /// Consume one token ID and return logits predicting the following token.
    pub fn forward_token_id(
        &self,
        token_id: u16,
        state: &mut LMInferenceState,
    ) -> Vec<f32> {
        assert!(
            (token_id as usize) < self.vocab.len(),
            "Token ID is outside the vocabulary"
        );

        let input = self.embeddings.encode_batch(
            &self.mlp.params,
            &[token_id],
            1,
            1,
        );

        self.mlp.forward_lm_incremental(
            &input,
            &mut state.mlp_state,
        )
    }

    /// Feed a prompt through a fresh or existing state and return the final
    /// next-token logits. The prompt itself is not retained in memory.
    pub fn prefill_ids(
        &self,
        ids: &[u16],
        state: &mut LMInferenceState,
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("Cannot prefill incremental inference with an empty prompt".to_string());
        }

        let mut logits = Vec::new();
        for &token_id in ids {
            logits = self.forward_token_id(token_id, state);
        }
        Ok(logits)
    }

    // ------------------------------------------------------------------------
    // Single-token generation
    // ------------------------------------------------------------------------



    // ------------------------------------------------------------------------
    // Generation
    // ------------------------------------------------------------------------

    pub fn generate(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
        threads: usize,
    ) -> String {
        self.generate_with_sampling(
            context,
            gen_length,
            SamplingConfig::recommended(temp),
            threads,
        )
    }

    pub fn generate_with_sampling(
        &self,
        context: String,
        gen_length: usize,
        sampling: SamplingConfig,
        threads: usize,
    ) -> String {
        let _ = threads;

        if gen_length == 0 {
            return String::new();
        }

        let prompt_ids =
            tokenize_u16(&context, &self.vocab);

        assert!(
            !prompt_ids.is_empty(),
            "Generation input cannot be empty"
        );

        let mut history =
            self.initial_repetition_history(
                &prompt_ids,
                sampling,
            );

        let mut state =
            self.new_inference_state()
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot start incremental inference: {}",
                        error
                    )
                });

        let mut logits =
            self.prefill_ids(&prompt_ids, &mut state)
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot prefill incremental inference: {}",
                        error
                    )
                });

        let mut tokens =
            Vec::<String>::with_capacity(gen_length);

        for step in 0..gen_length {
            let idx =
                sample_logits_with_config(
                    &logits,
                    &history,
                    sampling,
                );

            let token_id = idx as u16;

            tokens.push(
                self.vocab[idx].clone()
            );

            self.remember_repetition_token(
                &mut history,
                token_id,
                sampling,
            );

            if step + 1 < gen_length {
                logits = self.forward_token_id(
                    token_id,
                    &mut state,
                );
            }
        }

        decode_token_stream(&tokens)
    }

    pub fn generate_one_ids(
        &self,
        ids: &[u16],
        temp: f32,
        threads: usize,
    ) -> usize {
        self.generate_one_ids_with_sampling(
            ids,
            SamplingConfig::recommended(temp),
            threads,
        )
    }

    pub fn generate_one_ids_with_sampling(
        &self,
        ids: &[u16],
        sampling: SamplingConfig,
        _threads: usize,
    ) -> usize {
        assert!(
            !ids.is_empty(),
            "Generation input cannot be empty"
        );

        let positions = ids.len();

        let max_context_len =
            self.context_len as usize;

        assert!(
            positions <= max_context_len,
            "Generation input exceeds maximum context length"
        );

        let embedding_dim =
            self.embeddings.embedding_dim();

        let input =
            self.embeddings.encode_batch(
                &self.mlp.params,
                ids,
                1,
                positions,
            );

        let logits =
            self.mlp.forward_lm_last(
                &input,
                positions * embedding_dim,
            );

        let history =
            self.initial_repetition_history(
                ids,
                sampling,
            );

        sample_logits_with_config(
            &logits,
            &history,
            sampling,
        )
    }

    pub fn generate_one(
        &self,
        context: String,
        temp: f32,
        threads: usize,
    ) -> String {
        self.generate_one_with_sampling(
            context,
            SamplingConfig::recommended(temp),
            threads,
        )
    }

    pub fn generate_one_with_sampling(
        &self,
        context: String,
        sampling: SamplingConfig,
        threads: usize,
    ) -> String {
        let mut ids =
            self.encode_nums(context);

        let max_context_len =
            self.context_len as usize;

        if ids.len() > max_context_len {
            ids = ids[
                ids.len() - max_context_len..
                ].to_vec();
        }

        let idx =
            self.generate_one_ids_with_sampling(
                &ids,
                sampling,
                threads,
            );

        self.vocab[idx].clone()
    }

    pub fn generate_one_distribution(
        &self,
        context: String,
        top_k: usize,
        threads: usize,
    ) {
        let mut ids =
            self.encode_nums(
                context
            );

        let max_context_len =
            self.context_len
                as usize;

        if ids.len()
            > max_context_len
        {
            ids =
                ids[
                    ids.len()
                        - max_context_len
                        ..
                    ]
                    .to_owned();
        }

        assert!(
            !ids.is_empty(),
            "Generation input cannot be empty"
        );

        let positions = ids.len();

        let embedding_dim = self.embeddings.embedding_dim();

        println!(
            "IDs: {:?}",
            ids
        );

        println!(
            "Split text: {:?}",
            ids.iter()
                .map(
                    |x| {
                        self.vocab[
                            *x as usize
                            ]
                            .clone()
                    }
                )
                .collect::<Vec<_>>()
        );

        let input =
            self.embeddings.encode_batch(
                &self.mlp.params,
                &ids,
                1,
                positions,
            );

        let logits =
            self.mlp.forward_lm_last(
                &input,
                positions * embedding_dim,
            );

        let max_logit =
            logits
                .iter()
                .copied()
                .fold(
                    f32::NEG_INFINITY,
                    f32::max,
                );

        let exp_logits:
            Vec<f32> =
            logits
                .iter()
                .map(
                    |&x| {
                        (
                            x - max_logit
                        )
                            .exp()
                    }
                )
                .collect();

        let sum_exp:
            f32 =
            exp_logits
                .iter()
                .sum();

        let probs:
            Vec<f32> =
            exp_logits
                .iter()
                .map(
                    |&x| {
                        x / sum_exp
                    }
                )
                .collect();

        let mut newest:
            Vec<(usize, f32)> =
            probs
                .into_iter()
                .enumerate()
                .collect();

        newest.sort_by(
            |a, b| {
                b.1.total_cmp(
                    &a.1
                )
            }
        );

        for (
            idx,
            prob,
        ) in newest
            .iter()
            .take(
                top_k.min(
                    newest.len()
                )
            )
        {
            println!(
                "{}- {:.2}%",
                self.vocab[*idx],
                prob * 100.0
            );
        }

        let _ =
            threads;
    }

    pub fn generate_gpt(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
        threads: usize,
    ) -> String {
        self.generate_gpt_with_sampling(
            context,
            gen_length,
            SamplingConfig::recommended(temp),
            threads,
        )
    }

    pub fn generate_gpt_with_sampling(
        &self,
        context: String,
        gen_length: usize,
        sampling: SamplingConfig,
        threads: usize,
    ) -> String {
        let _ = threads;

        if gen_length == 0 {
            return String::new();
        }

        let prompt_ids =
            tokenize_u16(&context, &self.vocab);

        assert!(
            !prompt_ids.is_empty(),
            "Generation input cannot be empty"
        );

        let mut history =
            self.initial_repetition_history(
                &prompt_ids,
                sampling,
            );

        let mut state =
            self.new_inference_state()
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot start incremental inference: {}",
                        error
                    )
                });

        let mut logits =
            self.prefill_ids(&prompt_ids, &mut state)
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot prefill incremental inference: {}",
                        error
                    )
                });

        let mut output = String::new();
        let mut byte_buffer = Vec::<u8>::new();

        for step in 0..gen_length {
            let idx =
                sample_logits_with_config(
                    &logits,
                    &history,
                    sampling,
                );

            let token_id = idx as u16;
            let generation = &self.vocab[idx];

            output.push_str(generation);

            stream_token(
                generation,
                &mut byte_buffer,
            );

            let _ = std::io::stdout().flush();

            self.remember_repetition_token(
                &mut history,
                token_id,
                sampling,
            );

            if step + 1 < gen_length {
                logits = self.forward_token_id(
                    token_id,
                    &mut state,
                );
            }
        }

        if !byte_buffer.is_empty() {
            print!(
                "{}",
                String::from_utf8_lossy(&byte_buffer)
            );
        }

        let _ = std::io::stdout().flush();

        output
    }

    pub fn generate_gpt_chatter(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
        threads: usize,
    ) -> String {
        self.generate_gpt_chatter_with_sampling(
            context,
            gen_length,
            SamplingConfig::recommended(temp),
            threads,
        )
    }

    pub fn generate_gpt_chatter_with_sampling(
        &self,
        context: String,
        gen_length: usize,
        sampling: SamplingConfig,
        threads: usize,
    ) -> String {
        let _ = threads;

        if gen_length == 0 {
            return String::new();
        }

        let prompt_ids =
            tokenize_u16(&context, &self.vocab);

        assert!(
            !prompt_ids.is_empty(),
            "Generation input cannot be empty"
        );

        let mut history =
            self.initial_repetition_history(
                &prompt_ids,
                sampling,
            );

        let mut state =
            self.new_inference_state()
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot start incremental inference: {}",
                        error
                    )
                });

        let mut logits =
            self.prefill_ids(&prompt_ids, &mut state)
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot prefill incremental inference: {}",
                        error
                    )
                });

        let mut output = String::new();
        let mut byte_buffer = Vec::<u8>::new();

        for step in 0..gen_length {
            let idx =
                sample_logits_with_config(
                    &logits,
                    &history,
                    sampling,
                );

            let token_id = idx as u16;
            let generation = &self.vocab[idx];

            // Stop at the end-of-turn token.
            if generation.contains("<EOT>") {
                let before_eot =
                    generation.split("<EOT>")
                        .next()
                        .unwrap();

                stream_token(
                    before_eot,
                    &mut byte_buffer,
                );

                if !byte_buffer.is_empty() {
                    print!(
                        "{}",
                        String::from_utf8_lossy(&byte_buffer)
                    );
                    byte_buffer.clear();
                }

                let _ = std::io::stdout().flush();

                output.push_str(before_eot);

                return output;
            }

            output.push_str(generation);

            stream_token(
                generation,
                &mut byte_buffer,
            );

            let _ = std::io::stdout().flush();

            self.remember_repetition_token(
                &mut history,
                token_id,
                sampling,
            );

            if step + 1 < gen_length {
                logits = self.forward_token_id(
                    token_id,
                    &mut state,
                );
            }
        }

        if !byte_buffer.is_empty() {
            print!(
                "{}",
                String::from_utf8_lossy(&byte_buffer)
            );
        }

        let _ = std::io::stdout().flush();

        println!();

        output
    }

    // ------------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------------

    pub fn poison_check(
        &self,
    ) -> usize {
        let mut count =
            0usize;

        for &x in
            &self.mlp.params.values
        {
            if !x.is_finite() {
                println!(
                    "Non-finite: {}",
                    x
                );

                count += 1;
            }
        }

        count
    }

    pub fn poison_check_silent(
        &self,
    ) -> usize {
        self.mlp.params.values
            .iter()
            .filter(
                |&&x| !x.is_finite()
            )
            .count()
    }

    // ------------------------------------------------------------------------
    // Dataset
    // ------------------------------------------------------------------------

    pub fn set_dataset(
        &mut self,
        dataset: Vec<u16>,
    ) {
        self.dataset =
            dataset;

        // Explicitly supplied raw-token datasets are unmasked.
        self.dataset_loss_mask =
            None;

        self.dataset_chat_ranges = None;
    }

    pub fn set_dataset_with_mask(
        &mut self,
        dataset: Vec<u16>,
        loss_mask: LmLossMask,
    ) {
        self.dataset =
            dataset;

        self.dataset_loss_mask =
            Some(
                loss_mask
            );

        self.dataset_chat_ranges = None;
    }

    pub fn load_corpus(
        &mut self,
        corpus: &str,
    ) {
        println!("Tokenizing corpus...");

        self.dataset =
            tokenize_u16(
                corpus,
                &self.vocab,
            );

        // Raw text corpus = ordinary next-token training.
        self.dataset_loss_mask = None;
        self.dataset_chat_ranges = None;

        println!(
            "Done! Loaded {} tokens ({} training samples).",
            self.dataset.len(),
            self.dataset
                .len()
                .saturating_sub(self.context_len as usize)
        );
    }

    pub fn load_corpus_silent(
        &mut self,
        corpus: &str,
    ) {
        self.dataset =
            tokenize_u16(
                corpus,
                &self.vocab,
            );

        self.dataset_loss_mask =
            None;

        self.dataset_chat_ranges = None;
    }

    pub fn load_dataset_file(
        &mut self,
        path: &str,
    ) {
        let path_ref =
            Path::new(path);

        let is_json =
            path_ref
                .extension()
                .and_then(
                    |extension| {
                        extension.to_str()
                    }
                )
                .map(
                    |extension| {
                        extension.eq_ignore_ascii_case(
                            "json"
                        )
                    }
                )
                .unwrap_or(false);

        if is_json {
            // =========================================================
            // JSON MASKED DATASET PATH
            //
            // This is the ONLY case that loads the JSON-aware format.
            // The loader tokenizes one turn at a time and produces:
            //
            //     tokens
            //     loss ranges
            //
            // without keeping all original strings in RAM.
            // =========================================================

            println!("Loading masked JSON dataset: {}", path);

            let byte_fallback = byte_fallback_base(&self.vocab);

            let trie =
                crate::helper::Trie::from_vocab(
                    &self.vocab,
                    byte_fallback,
                );

            let loaded =
                crate::trainer::load_masked_json_lm_dataset_with_vocab_size(
                    path_ref,
                    1024 * 1024,
                    self.vocab.len(),
                    move |text, out| {
                        trie.tokenize_bytes_u16_into(
                            text.as_bytes(),
                            out,
                        );
                    },
                )
                    .unwrap_or_else(
                        |error| {
                            panic!(
                                "Failed to load JSON LM dataset '{}': {}",
                                path,
                                error,
                            );
                        }
                    );

            self.dataset = loaded.tokens;
            self.dataset_loss_mask = Some(loaded.loss_mask);

            let context_len = self.context_len as usize;
            let training_samples: usize = loaded.chat_ranges
                .iter()
                .map(|range| {
                    (range.end - range.start).saturating_sub(context_len)
                })
                .sum();
            let chat_count = loaded.chat_ranges.len();
            let too_short_chats = loaded.chat_ranges
                .iter()
                .filter(|range| range.end - range.start <= context_len)
                .count();

            self.dataset_chat_ranges = Some(loaded.chat_ranges);

            let masked_ranges =
                self.dataset_loss_mask
                    .as_ref()
                    .map(
                        |mask| {
                            mask.ranges().len()
                        }
                    )
                    .unwrap_or(0);

            println!(
                "Done! Loaded {} tokens across {} chats ({} training samples, {} chats too short for context length {}, {} selected loss ranges).",
                self.dataset.len(),
                chat_count,
                training_samples,
                too_short_chats,
                context_len,
                masked_ranges,
            );

            return;
        }

        // =============================================================
        // NORMAL DATASET PATH
        //
        // .txt, .bin, .whatever-you-currently-use:
        // read it exactly like before and do ordinary LM training.
        // =============================================================

        println!(
            "Loading text dataset: {}",
            path
        );

        let corpus =
            fs::read_to_string(
                path
            )
                .unwrap_or_else(
                    |error| {
                        panic!(
                            "Failed to read dataset '{}': {}",
                            path,
                            error,
                        );
                    }
                );

        self.load_corpus(
            &corpus
        );
    }

    // ------------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------------

    pub fn param(
        &self,
    ) -> Vec<f32> {
        self.mlp.params.values.clone()
    }

    pub fn param_count(
        &self,
    ) -> usize {
        self.mlp.parameter_count()
    }

    pub fn params(
        &self,
    ) {
        println!(
            "Parameter count: {}.",
            self.param_count()
        );

        println!(
            "Embedding table: {} params",
            self.embeddings
                .parameter_count()
        );

        let sequence_length =
            self.context_len as usize;

        let channels =
            self.embeddings
                .embedding_dim();

        println!(
            "O O O O O O O O O O O O O O O O   [{} × {}]",
            sequence_length,
            channels
        );

        print_layer_specs(
            &self.mlp.layer_specs(),
            sequence_length,
            channels,
            self.vocab.len(),
            0,
        );
    }

    // ------------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------------

    pub fn train(
        &mut self,
        batch_update_frequency: Option<usize>,
        checkpoint: Option<&str>,
        checkpoint_path: Option<&str>,
        checkpoint_frequency: CheckpointFrequency,
        lr: Option<f32>,
        seed: Option<u64>,
        variable_context: bool,
        threads: i32,
    ) {
        assert!(
            threads > 0,
            "Must use at least one thread!"
        );

        unsafe extern "C" {
            fn openblas_set_num_threads(
                num_threads: i32
            );

            fn openblas_get_num_threads()
                -> i32;
        }

        unsafe {
            openblas_set_num_threads(
                threads
            );

            println!(
                "OpenBLAS threads: {}",
                openblas_get_num_threads()
            );
        }

        // ---------------------------------------------------------
        // Load checkpoint
        // ---------------------------------------------------------

        let resume_state =
            checkpoint
                .as_deref()
                .filter(
                    |path| {
                        Path::new(path).exists()
                    }
                )
                .map(
                    |path| {
                        println!(
                            "Loading checkpoint: {}",
                            path
                        );

                        self.load_checkpoint(
                            path,
                            lr,
                        )
                    }
                );

        // ---------------------------------------------------------
        // Metadata for checkpoint saving
        // ---------------------------------------------------------

        let vocab =
            self.vocab.clone();

        let context_len =
            self.context_len;

        let hidden_layers =
            self.hidden_layers.clone();

        let byte_fallback =
            byte_fallback_base(
                &self.vocab
            ) as usize;

        let token_byte_lens:
            Vec<u32> =
            self.vocab
                .iter()
                .enumerate()
                .map(
                    |(i, token)| {
                        if i >= byte_fallback
                            && i < byte_fallback + 256
                        {
                            1
                        } else {
                            token
                                .as_bytes()
                                .len() as u32
                        }
                    }
                )
                .collect();

        let should_save_checkpoints =
            checkpoint_path.is_some();

        // ---------------------------------------------------------
        // Save callback
        // ---------------------------------------------------------

        let mut savefn =
            move |
                state: CheckpointState,
                mlp: &MLP,
                embeddings: &Embeddings,
            | {
                let Some(base_path) =
                    checkpoint_path
                        .as_deref()
                else {
                    return;
                };

                let suffix =
                    match state.kind {
                        CheckpointKind::Batch => {
                            format!(
                                "_batch_{}_epoch_{}",
                                state.batch,
                                state.epoch,
                            )
                        }

                        CheckpointKind::Epoch => {
                            format!(
                                "_epoch_{}",
                                state.epoch,
                            )
                        }
                    };

                let path =
                    format!(
                        "{}{}.check",
                        base_path,
                        suffix,
                    );

                let model =
                    SavedLM {
                        description:
                        format!(
                            "Training checkpoint | Epoch {} | Batch {} | Sample {}",
                            state.epoch,
                            state.batch,
                            state.sample,
                        ),

                        mlp:
                        mlp.save(),

                        vocab:
                        vocab.clone(),

                        context_len,

                        hidden_layers:
                        hidden_layers.clone(),

                        embeddings:
                        embeddings
                            .save(
                                &mlp.params
                            ),
                    };

                let saved =
                    SavedCheckpoint {
                        model,

                        epoch:
                        state.epoch,

                        batch:
                        state.batch,

                        sample:
                        state.sample,

                        sampler_seed:
                        state.sampler_seed,

                        sampler_version:
                        state.sampler_version,

                        sampler_data_len:
                        state.sampler_data_len,

                        lr:
                        state.lr,

                        best_loss:
                        state.best_loss,

                        plateau_count:
                        state.plateau_count,

                        adam_first_moments:
                        state
                            .adam_first_moments,

                        adam_second_moments:
                        state
                            .adam_second_moments,

                        adam_step:
                        state.adam_step,
                    };

                let bytes =
                    bincode::serde::encode_to_vec(
                        &saved,
                        bincode::config::standard(),
                    )
                        .unwrap();

                if let Some(parent) =
                    Path::new(
                        &path
                    )
                        .parent()
                {
                    fs::create_dir_all(
                        parent
                    )
                        .unwrap();
                }

                fs::write(
                    &path,
                    bytes,
                )
                    .unwrap();

                println!(
                    "Checkpoint saved: {}",
                    path
                );
            };

        let savefn:
            Option<
                &mut dyn FnMut(
                    CheckpointState,
                    &MLP,
                    &Embeddings,
                ),
            > =
            if should_save_checkpoints {
                Some(&mut savefn)
            } else {
                None
            };

        self.mlp.set_num_threads(threads as usize);
        self.trainer.train_lm_with_chat_ranges(
            1,
            batch_update_frequency,
            resume_state,
            variable_context,
            checkpoint_frequency,
            savefn,
            &mut self.mlp,
            &self.dataset,
            self.context_len as usize,
            &self.embeddings,
            &token_byte_lens,
            self.dataset_loss_mask.as_ref(),
            self.dataset_chat_ranges.as_deref(),
            seed.unwrap_or(
                PermutationSampler::DEFAULT_SEED
            ),
            threads as usize,
        );
    }

    pub fn dataset_is_masked(
        &self,
    ) -> bool {
        self.dataset_loss_mask
            .is_some()
    }

    /// Number of independent chat ranges in the currently loaded dataset.
    /// Raw text datasets are represented as one continuous stream and return 0.
    pub fn dataset_chat_count(
        &self,
    ) -> usize {
        self.dataset_chat_ranges
            .as_ref()
            .map(Vec::len)
            .unwrap_or(0)
    }

    pub fn dataset_loss_mask(
        &self,
    ) -> Option<&LmLossMask> {
        self.dataset_loss_mask
            .as_ref()
    }

    pub fn set_cosine_lr(
        &mut self,
        warmup_steps: u64,
        total_steps: u64,
        min_lr_ratio: f32,
    ) {
        self.trainer
            .reinit_lr_schedule(
                LrSchedule::Cosine {
                    warmup_steps,
                    total_steps,
                    min_lr_ratio,
                }
            );
    }

    // ------------------------------------------------------------------------
    // Checkpoint loading
    // ------------------------------------------------------------------------

    pub fn load_checkpoint(
        &mut self,
        path: &str,
        lr: Option<f32>,
    ) -> ResumeState {
        let bytes =
            fs::read(path)
                .unwrap();

        let checkpoint:
            SavedCheckpoint =
            match bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ) {
                Ok((
                       checkpoint,
                       _,
                   )) =>
                    checkpoint,

                Err(_) => {
                    println!(
                        "Checkpoint uses legacy format; \
                         loading model/position and \
                         initializing fresh Adam state."
                    );

                    let (
                        old,
                        _,
                    ):
                        (
                            SavedCheckpointV1,
                            usize,
                        ) =
                        bincode::serde::decode_from_slice(
                            &bytes,
                            bincode::config::standard(),
                        )
                            .expect(
                                "Failed to load checkpoint as either \
                             current or legacy format."
                            );

                    SavedCheckpoint {
                        model:
                        old.model,

                        epoch:
                        old.epoch,

                        batch:
                        old.batch,

                        sample:
                        old.sample,

                        sampler_seed:
                        old.sampler_seed,

                        sampler_version:
                        old.sampler_version,

                        sampler_data_len:
                        old.sampler_data_len,

                        lr:
                        old.lr,

                        best_loss:
                        old.best_loss,

                        plateau_count:
                        old.plateau_count,

                        adam_first_moments:
                        Vec::new(),

                        adam_second_moments:
                        Vec::new(),

                        adam_step:
                        0,
                    }
                }
            };

        // ---------------------------------------------------------
        // Restore model into ONE ParameterStore
        // ---------------------------------------------------------

        self.vocab =
            checkpoint.model.vocab;

        // The checkpoint may contain a different vocabulary, so rebuild the
        // tokenizer cache before any subsequent text encoding or ID merging.
        let (tokenizer_trie, max_token_bytes) =
            build_tokenizer_data(&self.vocab);
        self.tokenizer_trie = tokenizer_trie;
        self.max_token_bytes = max_token_bytes;

        self.context_len =
            checkpoint.model.context_len;

        self.hidden_layers =
            checkpoint.model.hidden_layers;

        let mut params =
            ParameterStore::new();

        self.embeddings =
            Arc::new(
                Embeddings::load_into(
                    &mut params,
                    checkpoint
                        .model
                        .embeddings,
                )
            );

        self.mlp =
            MLP::load_with_embeddings(
                &checkpoint.model.mlp,
                Arc::clone(
                    &self.embeddings
                ),
                params,
            );

        println!(
            "Loaded checkpoint: epoch {}, batch {}, sample {}.",
            checkpoint.epoch,
            checkpoint.batch,
            checkpoint.sample,
        );

        let lr_ =
            match lr {
                Some(value) =>
                    value,

                None =>
                    checkpoint.lr,
            };

        ResumeState {
            epoch:
            checkpoint.epoch,

            batch:
            checkpoint.batch,

            sample:
            checkpoint.sample,

            sampler_seed:
            checkpoint.sampler_seed,

            sampler_version:
            checkpoint.sampler_version,

            sampler_data_len:
            checkpoint.sampler_data_len,

            lr:
            lr_,

            best_loss:
            checkpoint.best_loss,

            plateau_count:
            checkpoint.plateau_count,

            adam_first_moments:
            checkpoint
                .adam_first_moments,

            adam_second_moments:
            checkpoint
                .adam_second_moments,

            adam_step:
            checkpoint.adam_step,
        }
    }

    // ------------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------------

    pub fn to_saved(
        &self,
        description: &str,
    ) -> SavedLM {
        SavedLM {
            description:
            description.to_string(),

            mlp:
            self.mlp.save(),

            vocab:
            self.vocab.clone(),

            context_len:
            self.context_len,

            hidden_layers:
            self.hidden_layers.clone(),

            embeddings:
            self.embeddings
                .save(
                    &self.mlp.params
                ),
        }
    }

    pub fn from_saved(
        saved: SavedLM,
    ) -> Self {
        println!(
            "Description:\n{}",
            saved.description
        );

        println!(
            "Loading..."
        );

        Self::from_saved_silent(
            saved
        )
    }

    pub fn from_saved_silent(
        saved: SavedLM,
    ) -> Self {
        let mut params =
            ParameterStore::new();

        let embeddings =
            Arc::new(
                Embeddings::load_into(
                    &mut params,
                    saved.embeddings,
                )
            );

        let mlp =
            MLP::load_with_embeddings(
                &saved.mlp,
                Arc::clone(
                    &embeddings
                ),
                params,
            );

        let (tokenizer_trie, max_token_bytes) =
            build_tokenizer_data(&saved.vocab);

        Self {
            trainer: Trainer::new(0.0, 0, 0, 0),
            mlp,
            dataset: Vec::new(),
            dataset_loss_mask: None,
            dataset_chat_ranges: None,
            vocab: saved.vocab,
            tokenizer_trie,
            max_token_bytes,
            context_len: saved.context_len,
            hidden_layers: saved.hidden_layers,
            embeddings,
        }
    }

    pub fn embeds(
        &self,
    ) -> Embeddings {
        (*self.embeddings).clone()
    }

    pub fn save(
        &self,
        path: &str,
        description: &str,
    ) {
        let saved =
            self.to_saved(
                description
            );

        let bytes =
            bincode::serde::encode_to_vec(
                &saved,
                bincode::config::standard(),
            )
                .unwrap();

        fs::write(
            path,
            bytes,
        )
            .unwrap();
    }

    pub fn load(
        path: &str,
    ) -> (String, Self) {
        let bytes =
            fs::read(path)
                .unwrap();

        let (
            model,
            _,
        ):
            (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            )
                .unwrap();

        (
            model.description.clone(),
            Self::from_saved(
                model
            ),
        )
    }

    pub fn load_silent(
        path: &str,
    ) -> (String, Self) {
        let bytes =
            fs::read(path)
                .unwrap();

        let (
            model,
            _):
            (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            )
                .unwrap();

        (
            model.description.clone(),
            Self::from_saved_silent(
                model
            ),
        )
    }

    pub fn find_clusters(&self, threshold: f32) {
        self.embeddings.find_clusters(&self.mlp.parameters(), threshold, self.vocab.clone())
    }

    /// Inspect GlobalMixer signals for a text prompt.
    pub fn inspect_text(
        &self,
        text: &str,
    ) -> Result<crate::inspection::ModelInspection, String> {
        crate::inspection::inspect_text(self, text)
    }

    /// Inspect GlobalMixer signals for token IDs.
    pub fn inspect_ids(
        &self,
        ids: &[u16],
    ) -> Result<crate::inspection::ModelInspection, String> {
        crate::inspection::inspect_ids(self, ids)
    }

    /// Internal full-sequence forward pass used by inspection.rs.
    pub(crate) fn inspection_forward(
        &self,
        ids: &[u16],
    ) -> Vec<BatchLayerCache> {
        assert!(!ids.is_empty(), "Cannot inspect an empty token sequence");

        let embeddings = self.encode_embeddings(ids);
        let input_size = embeddings.len();

        let forward = self.mlp.forward_lm_batch(
            &embeddings,
            1,
            input_size,
        );

        forward.layers
    }

    pub fn user_bot_eot_to_blank_in_vocab(&mut self) {
        for token in &mut self.vocab {
            if matches!(token.as_str(), "<USER>" | "<BOT>" | "<EOT>") {
                token.clear();
            }
        }

        let (trie, max_token_bytes) =
            build_tokenizer_data(&self.vocab);

        self.tokenizer_trie = trie;
        self.max_token_bytes = max_token_bytes;
    }
}
