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
    SavedMLP,
};
use crate::parameters::ParameterStore;
use crate::trainer::{
    CheckpointFrequency,
    CheckpointKind,
    CheckpointState,
    PermutationSampler,
    ResumeState,
    Trainer,
};
pub use crate::miscellaneous::{
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
use std::sync::Arc;

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
/// This struct is serialized into a .sumyu file using serde.
#[derive(
    Serialize,
    Deserialize,
)]
pub struct SavedLM {
    description: String,
    mlp: SavedMLP,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: SavedEmbeddings,
}

#[derive(
    Serialize,
    Deserialize,
)]
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

#[derive(
    Deserialize,
)]
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
            byte_fallback_base(
                vocab.to_vec()
            ),
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
            byte_fallback_base(
                vocab.to_vec()
            ),
        );

    trie.tokenize_bytes_u16(
        text.as_bytes()
    )
}

#[inline]
fn byte_fallback_base(
    vocab: Vec<String>,
) -> u32 {
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

pub struct LM {
    trainer: Trainer,
    mlp: MLP,
    dataset: Vec<u16>,
    vocab: Vec<String>,
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
                crate::neuron::Activation::None,
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

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            vocab,
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
                context_len as usize
                    * embedding_dim,
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

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            vocab,
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

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            vocab: config.vocab,
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
                config.context_len
                    * config.emb_dim,
                &config.layer_specs,
                Arc::clone(&embeddings),
                params,
            );

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            vocab: config.vocab,
            context_len:
            config.context_len as u32,
            hidden_layers: Vec::new(),
            embeddings,
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

        Self {
            trainer,
            mlp,
            dataset: Vec::new(),
            vocab,
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

    pub fn encode_nums(
        &self,
        string: String,
    ) -> Vec<u16> {
        tokenize_u16(
            &string,
            &self.vocab,
        )
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
    // Single-token generation
    // ------------------------------------------------------------------------

    pub fn generate_one_ids(
        &self,
        ids: &[u16],
        temp: f32,
        _threads: usize,
    ) -> usize {
        let input =
            self.embeddings.encode_batch(
                &self.mlp.params,
                ids,
                1,
                self.context_len as usize,
            );

        let forward =
            self.mlp.forward_batch(
                &input,
                1,
                self.context_len as usize
                    * self.embeddings.embedding_dim(),
            );

        let logits =
            forward.output;

        if temp <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(
                    |a, b| {
                        a.1.total_cmp(
                            b.1
                        )
                    }
                )
                .unwrap()
                .0;
        }

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
                            (x / temp)
                                - max_logit
                        )
                            .exp()
                    }
                )
                .collect();

        let sum_exp:
            f32 =
            exp_logits.iter().sum();

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

        let dist =
            WeightedIndex::new(
                &probs
            )
                .unwrap();

        dist.sample(
            &mut rng()
        )
    }

    pub fn generate_one(
        &self,
        context: String,
        temp: f32,
        threads: usize,
    ) -> String {
        let mut ids =
            self.encode_nums(
                context
            );

        if ids.len()
            > self.context_len
            as usize
        {
            ids =
                ids[
                    ids.len()
                        - self.context_len
                        as usize
                        ..
                    ]
                    .to_vec();
        } else if ids.len()
            < self.context_len
            as usize
        {
            let num_to_add =
                self.context_len
                    as usize
                    - ids.len();

            let mut new_ids =
                vec![
                    0u16;
                    num_to_add
                ];

            new_ids.extend(
                ids
            );

            ids =
                new_ids;
        }

        let idx =
            self.generate_one_ids(
                &ids,
                temp,
                threads,
            );

        self.vocab[idx]
            .clone()
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

        if ids.len()
            > self.context_len
            as usize
        {
            ids =
                ids[
                    ids.len()
                        - self.context_len
                        as usize
                        ..
                    ]
                    .to_owned();
        } else if ids.len()
            < self.context_len
            as usize
        {
            let num_to_add =
                self.context_len
                    as usize
                    - ids.len();

            let mut new_ids =
                vec![
                    0u16;
                    num_to_add
                ];

            new_ids.extend(
                ids
            );

            ids =
                new_ids;
        }

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
                self.context_len as usize,
            );

        let forward =
            self.mlp.forward_batch(
                &input,
                1,
                self.context_len as usize
                    * self.embeddings.embedding_dim(),
            );

        let logits =
            forward.output;

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
                        (x - max_logit).exp()
                    }
                )
                .collect();

        let sum_exp:
            f32 =
            exp_logits.iter().sum();

        let probs:
            Vec<f32> =
            exp_logits
                .iter()
                .map(
                    |&x| x / sum_exp
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

        // Keep the public argument used.
        let _ =
            threads;
    }

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
        let mut context_ =
            context;

        let mut tokens =
            Vec::<String>::new();

        for _ in 0..gen_length {
            let token =
                self.generate_one(
                    context_.clone(),
                    temp,
                    threads,
                );

            context_.push_str(
                &token
            );

            tokens.push(
                token
            );
        }

        decode_token_stream(
            &tokens
        )
    }

    pub fn generate_gpt(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
        threads: usize,
    ) -> String {
        let trie =
            crate::helper::Trie::from_vocab(
                &self.vocab,
                byte_fallback_base(
                    self.vocab.clone()
                ),
            );

        let context_len =
            self.context_len
                as usize;

        let mut tokenizer =
            crate::helper::IncrementalTokenizer::new(
                &trie,
                context_len,
            );

        let initial_ids =
            trie.tokenize_u32(
                &context
            );

        tokenizer.push_raw_bytes(
            context.as_bytes()
        );

        debug_assert_eq!(
            tokenizer.current_ids(
                context_len,
                1,
            ),
            {
                let mut expected =
                    initial_ids
                        .iter()
                        .map(
                            |&x|
                                x as usize
                        )
                        .collect::<Vec<_>>();

                if expected.len()
                    > context_len
                {
                    expected =
                        expected[
                            expected.len()
                                - context_len
                                ..
                            ]
                            .to_vec();
                } else if expected.len()
                    < context_len
                {
                    let mut padded =
                        vec![
                            1usize;
                            context_len
                                - expected.len()
                        ];

                    padded.extend(
                        expected
                    );

                    expected =
                        padded;
                }

                expected
            }
        );

        let mut output =
            String::new();

        let mut byte_buffer =
            Vec::<u8>::new();

        for _ in 0..gen_length {
            let ids =
                tokenizer
                    .current_ids_u16(
                        context_len,
                        1,
                    );

            let idx =
                self.generate_one_ids(
                    &ids,
                    temp,
                    threads,
                );

            let generation =
                &self.vocab[idx];

            tokenizer.push_token(
                generation
            );

            output.push_str(
                generation
            );

            stream_token(
                generation,
                &mut byte_buffer,
            );

            let _ =
                std::io::stdout()
                    .flush();
        }

        if !byte_buffer.is_empty() {
            print!(
                "{}",
                String::from_utf8_lossy(
                    &byte_buffer
                )
            );
        }

        let _ =
            std::io::stdout()
                .flush();

        output
    }

    pub fn generate_gpt_chatter(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
        threads: usize,
    ) -> String {
        let trie =
            crate::helper::Trie::from_vocab(
                &self.vocab,
                byte_fallback_base(
                    self.vocab.clone()
                ),
            );

        let context_len =
            self.context_len
                as usize;

        let mut tokenizer =
            crate::helper::IncrementalTokenizer::new(
                &trie,
                context_len,
            );

        tokenizer.push_raw_bytes(
            context.as_bytes()
        );

        let mut output =
            String::new();

        let mut byte_buffer =
            Vec::<u8>::new();

        for _ in 0..gen_length {
            let ids =
                tokenizer
                    .current_ids_u16(
                        context_len,
                        1,
                    );

            let idx =
                self.generate_one_ids(
                    &ids,
                    temp,
                    threads,
                );

            let generation =
                &self.vocab[idx];

            if generation.contains(
                "<EOT>"
            ) {
                let before_eot =
                    generation
                        .split("<EOT>")
                        .next()
                        .unwrap();

                stream_token(
                    before_eot,
                    &mut byte_buffer,
                );

                if !byte_buffer.is_empty() {
                    print!(
                        "{}",
                        String::from_utf8_lossy(
                            &byte_buffer
                        )
                    );

                    byte_buffer.clear();
                }

                let _ =
                    std::io::stdout()
                        .flush();

                output.push_str(
                    generation
                );

                return output;
            }

            tokenizer.push_token(
                generation
            );

            output.push_str(
                generation
            );

            stream_token(
                generation,
                &mut byte_buffer,
            );

            let _ =
                std::io::stdout()
                    .flush();
        }

        if !byte_buffer.is_empty() {
            print!(
                "{}",
                String::from_utf8_lossy(
                    &byte_buffer
                )
            );
        }

        let _ =
            std::io::stdout()
                .flush();

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
    }

    pub fn load_corpus(
        &mut self,
        corpus: &str,
    ) {
        println!(
            "Tokenizing corpus..."
        );

        self.dataset =
            tokenize_u16(
                corpus,
                &self.vocab,
            );

        println!(
            "Done! Loaded {} tokens ({} training samples).",
            self.dataset.len(),
            self.dataset
                .len()
                .saturating_sub(
                    self.context_len as usize
                )
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
                        std::path::Path::new(
                            path
                        )
                            .exists()
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
                    std::path::Path::new(
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
        self.trainer.train_lm(
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
            seed.unwrap_or(
                PermutationSampler::DEFAULT_SEED
            ),
            threads as usize,
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

        Self {
            trainer:
            Trainer::new(
                0.0,
                0,
                0,
                0,
            ),

            mlp,

            dataset:
            Vec::new(),

            vocab:
            saved.vocab,

            context_len:
            saved.context_len,

            hidden_layers:
            saved.hidden_layers,

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
}
