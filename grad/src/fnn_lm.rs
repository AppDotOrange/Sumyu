use crate::neuron::{LayerSpec, SavedMLP, MLP};
use crate::Tensor;
use crate::trainer::{CheckpointFrequency, ResumeState, Trainer, CheckpointKind, CheckpointState, PermutationSampler};
use crate::embeddings::{Embeddings, SavedEmbeddings};
use crate::helper::{Config, HybridConfig};
pub use crate::miscellaneous::{decode_token_stream, print_layer_specs, stream_token};
use std::fs;
use std::io::Write;
use std::sync::Arc;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;
use serde::{Serialize, Deserialize};

///
/// A struct that contains everything needed for LM inference, but not training.
///
/// # Contents:
/// * description: String
/// * mlp: SavedMLP
/// * vocab: Vec\<String\>
/// * context_len: u32
/// * hidden_layers: Vec<usize>
/// * embeddings: SavedEmbeddings
///
/// This struct is serialized into a .sumyu file using serde.
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


pub fn tokenize(text: &str, vocab: &[String]) -> Vec<usize> {
    let trie = crate::helper::Trie::from_vocab(vocab, byte_fallback_base(Vec::from(vocab)));

    trie.tokenize_u32(text)
        .into_iter()
        .map(|x| x as usize)
        .collect()
}

pub fn tokenize_u16(text: &str, vocab: &[String]) -> Vec<u16> {
    let trie = crate::helper::Trie::from_vocab(
        vocab,
        byte_fallback_base(vocab.to_vec()),
    );

    trie.tokenize_bytes_u16(text.as_bytes())
}

#[inline]
fn byte_fallback_base(vocab: Vec<String>) -> u32 {
    let base = vocab
        .iter()
        .position(|t| t == "<0x00>")
        .expect("Vocabulary missing <0x00>") as u32;

    debug_assert!(
        base + 256 <= vocab.len() as u32,
        "Byte fallback block incomplete"
    );

    base
}

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
    ///
    /// LM::new() is used to create a new LM.
    ///
    /// # Arguments
    ///
    /// * `context_len`: u32
    /// * `vocab`: Vec<String>
    /// * `hidden_dim`: &\[usize\]
    /// * `embedding_dim`: usize
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grad::fnn_lm::LM;
    /// use grad::vocabs;
    ///
    /// let lm = LM::new(
    ///     16,  // context length
    ///     vocabs::fineweb_v2(),  // vocabulary trained on the fineweb dataset family
    ///     &[300, 300],  // MLP hidden layer dimensions
    ///     64,  // embedding dimension
    /// )
    /// ```
    pub fn new(context_len: u32, vocab: Vec<String>, hidden_dim: &[usize], embedding_dim: usize) -> Self {
        let mut dim = vec![context_len as usize * embedding_dim];
        dim.extend(hidden_dim.to_vec());
        dim.push(vocab.len());
        let embeddings = Arc::new(
            Embeddings::new(
                vocab.len(),
                embedding_dim,
            )
        );

        let trainer = Trainer::new(0.0, 0, 0, 0);

        Self {
            trainer,
            mlp: MLP::new(dim[0], &dim[1..]),
            dataset: vec![],
            vocab,
            context_len,
            hidden_layers: hidden_dim.to_vec(),
            embeddings,
        }
    }

    pub fn from_layers(
        context_len: u32,
        vocab: Vec<String>,
        layers: &[LayerSpec],
        embedding_dim: usize,
    ) -> Self {
        let embeddings = Arc::new(
            Embeddings::new(
                vocab.len(),
                embedding_dim,
            )
        );

        let trainer = Trainer::new(0.0, 0, 0, 0);

        Self {
            trainer,
            mlp: MLP::from_layers_with_embeddings(
                context_len as usize * embedding_dim,
                layers,
                Arc::clone(&embeddings),
            ),
            dataset: vec![],
            vocab,
            context_len,
            hidden_layers: vec![],
            embeddings,
        }
    }

    pub fn from_config(config: Config) -> Self {
        let mut dim = vec![config.context_len * config.emb_dim];
        dim.extend(config.hidden_dim.to_vec());
        dim.push(config.vocab.len());
        let embeddings = Arc::new(
            Embeddings::new(
                config.vocab.len(),
                config.emb_dim,
            )
        );

        let trainer = Trainer::new(config.lr, config.epochs, config.batch_size, config.max_batches_per_epoch);
        Self {
            trainer,
            mlp: MLP::new(dim[0], &dim[1..]),
            dataset: vec![],
            vocab: config.vocab,
            context_len: config.context_len as u32,
            hidden_layers: config.hidden_dim.to_vec(),
            embeddings,
        }
    }

    pub fn from_hybrid_config(config: HybridConfig) -> Self {
        let embeddings = Arc::new(
            Embeddings::new(
                config.vocab.len(),
                config.emb_dim,
            )
        );

        let trainer = Trainer::new(
            config.lr,
            config.epochs,
            config.batch_size,
            config.max_batches_per_epoch,
        );

        Self {
            trainer,
            mlp: MLP::from_layers_with_embeddings(
                config.context_len * config.emb_dim,
                &config.layer_specs,
                Arc::clone(&embeddings),
            ),
            dataset: vec![],
            vocab: config.vocab,
            context_len: config.context_len as u32,
            hidden_layers: vec![],
            embeddings,
        }
    }

    pub fn from_checkpoint(
        path: &str,
    ) -> Self {
        let bytes = fs::read(path).unwrap();

        let (checkpoint, _): (SavedCheckpoint, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();

        let vocab = checkpoint.model.vocab;
        let context_len = checkpoint.model.context_len;
        let hidden_layers = checkpoint.model.hidden_layers;
        let embeddings = Arc::new(
            Embeddings::load(
                checkpoint.model.embeddings
            )
        );

        let mlp = MLP::load_with_embeddings(
            &checkpoint.model.mlp,
            Arc::clone(&embeddings),
        );
        let dataset = vec![];

        println!(
            "Loaded checkpoint: epoch {}, batch {}, sample {}.",
            checkpoint.epoch,
            checkpoint.batch,
            checkpoint.sample,
        );

        let trainer = Trainer::new(
            0f32,
            0,
            0,
            0,
        );

        Self {
            trainer,
            mlp,
            dataset,
            vocab,
            context_len,
            hidden_layers,
            embeddings,
        }
    }

    pub fn dataset_size(&self) -> usize {
        self.dataset.len()
    }

    pub fn encode_embeddings(&self, ids: &[u16]) -> Vec<Tensor> {
        self.embeddings.encode(ids)
    }

    pub fn train_options(
        &mut self,
        lr: f32,
        epochs: usize,
        batch_size: usize,
        max_batches_per_epoch: usize,
    ) {
        self.trainer.reinit_batch(batch_size);
        self.trainer.reinit_lr(lr);
        self.trainer.reinit_epochs(epochs);
        self.trainer.reinit_batch_per_epoch(max_batches_per_epoch);
    }

    pub fn encode_nums(&self, string: String) -> Vec<u16> {
        tokenize_u16(&string, &self.vocab)
    }

    pub fn max(&self, vec: Vec<Tensor>) -> usize {
        let mut best = 0;
        let mut best_val = vec[0].data();

        for (i, t) in vec.iter().enumerate().skip(1) {
            let v = t.data();
            if v > best_val {
                best = i;
                best_val = v;
            }
        }

        best
    }

    pub fn generate_one_ids(
        &self,
        ids: &[u16],
        temp: f32,
    ) -> usize {
        let input = self.embeddings.encode_batch(
            ids,
            1,
            self.context_len as usize,
        );

        let forward = self.mlp.forward_batch(
            &input,
            1,
            self.context_len as usize * self.embeddings.embedding_dim(),
        );

        let logits = forward.output;

        if temp <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
        }

        let scaled_logits: Vec<f32> =
            logits
                .iter()
                .map(|&x| x / temp)
                .collect();

        let max_logit =
            scaled_logits
                .iter()
                .copied()
                .fold(
                    f32::NEG_INFINITY,
                    f32::max,
                );

        let exp_logits: Vec<f32> =
            scaled_logits
                .iter()
                .map(|&x| {
                    (x - max_logit).exp()
                })
                .collect();

        let sum_exp: f32 =
            exp_logits.iter().sum();

        let probs: Vec<f32> =
            exp_logits
                .iter()
                .map(|&x| x / sum_exp)
                .collect();

        let dist =
            WeightedIndex::new(&probs)
                .unwrap();

        dist.sample(&mut rng())
    }

    pub fn generate_one(
        &self,
        context: String,
        temp: f32,
    ) -> String {
        let mut ids =
            self.encode_nums(context);

        if ids.len() > self.context_len as usize {
            ids =
                ids[ids.len() -
                    self.context_len as usize..]
                    .to_vec();
        } else if ids.len() < self.context_len as usize {
            let num_to_add =
                self.context_len as usize -
                    ids.len();

            let mut new_ids =
                vec![0u16; num_to_add];

            new_ids.extend(ids);
            ids = new_ids;
        }

        let idx =
            self.generate_one_ids(
                &ids,
                temp,
            );

        self.vocab[idx].clone()
    }

    pub fn generate_one_distribution(&self, context: String, top_k: usize) {
        let mut ids = self.encode_nums(context);
        if ids.len() > self.context_len as usize {
            ids = ids[ids.len() - self.context_len as usize..ids.len()].to_owned();
        } else if ids.len() < self.context_len as usize {
            let num_to_add = self.context_len as usize-ids.len();
            let mut new_ids = vec![0u16; num_to_add];
            new_ids.extend(ids);
            ids = new_ids
        }
        println!("IDs: {:?}", ids);
        println!("Split text: {:?}", ids.iter().map(|x1| {self.vocab[*x1 as usize].clone()}).collect::<Vec<_>>());
        let input = self.embeddings.encode_batch(
            &*ids,
            1,
            self.context_len as usize,
        );

        let forward = self.mlp.forward_batch(
            &input,
            1,
            self.context_len as usize * self.embeddings.embedding_dim(),
        );

        let logits = forward.output;
        let max_logit = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits
            .iter()
            .map(|&x| x / sum_exp)
            .collect();
        let mut newest: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        newest.sort_by(|a, b| b.1.total_cmp(&a.1));

        for i in newest.iter().take(top_k.min(newest.len())) {
            let (idx, prob) = i;
            println!("{}- {:.2}%", self.vocab[*idx], prob * 100.0);
        }
    }

    pub fn generate(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
    ) -> String {
        let mut context_ = context;
        let mut tokens = Vec::<String>::new();

        for _ in 0..gen_length {
            let token = self.generate_one(context_.clone(), temp);

            context_.push_str(&token);
            tokens.push(token);
        }

        decode_token_stream(&tokens)
    }

    pub fn generate_gpt(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
    ) -> String {
        let trie =
            crate::helper::Trie::from_vocab(
                &self.vocab,
                byte_fallback_base(self.vocab.clone())
            );

        let context_len =
            self.context_len as usize;

        let mut tokenizer =
            crate::helper::IncrementalTokenizer::new(
                &trie,
                context_len,
            );

        // Tokenize the initial prompt once.
        let initial_ids =
            trie.tokenize_u32(&context);

        // Feed the prompt through the same incremental
        // machinery. Reconstructing these bytes is unnecessary
        // if the prompt is huge, so we simply initialize the
        // stable state from its exact tokenization.
        //
        // Since this is only the initial prompt, this one-time
        // allocation is fine.
        tokenizer.push_raw_bytes(
            context.as_bytes()
        );

        // The model should start from the exact tokenization
        // of the prompt, including its EOF boundary.
        debug_assert_eq!(
            tokenizer.current_ids(
                context_len,
                1,
            ),
            {
                let mut expected = initial_ids
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>();

                if expected.len() > context_len {
                    expected =
                        expected[
                            expected.len() -
                                context_len..
                            ]
                            .to_vec();
                } else if expected.len() < context_len {
                    let mut padded =
                        vec![1usize;
                             context_len -
                                 expected.len()];

                    padded.extend(expected);
                    expected = padded;
                }

                expected
            }
        );

        let mut output =
            String::new();

        let mut byte_buffer =
            Vec::<u8>::new();

        for _ in 0..gen_length {
            // IMPORTANT:
            // current_ids() tokenizes the unresolved suffix as EOF,
            // so this exactly matches tokenize_u32() on the current
            // complete context.
            let ids =
                tokenizer.current_ids_u16(
                    context_len,
                    1,
                );

            let idx =
                self.generate_one_ids(
                    &*ids,
                    temp,
                );

            let generation =
                &self.vocab[idx];

            // Add the model's actual vocabulary token to the
            // incremental tokenizer. This may cause a previous
            // "pending" token such as "th" to become "the".
            tokenizer.push_token(generation);

            output.push_str(generation);

            stream_token(
                generation,
                &mut byte_buffer,
            );

            let _ =
                std::io::stdout().flush();
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
            std::io::stdout().flush();

        output
    }

    pub fn generate_gpt_chatter(
        &self,
        context: String,
        gen_length: usize,
        temp: f32,
    ) -> String {
        let trie =
            crate::helper::Trie::from_vocab(
                &self.vocab,
                byte_fallback_base(self.vocab.clone()),
            );

        let context_len =
            self.context_len as usize;

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
                tokenizer.current_ids_u16(
                    context_len,
                    1,
                );

            let idx =
                self.generate_one_ids(
                    &*ids,
                    temp,
                );

            let generation =
                &self.vocab[idx];

            if generation.contains("<EOT>") {
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

                let _ = std::io::stdout().flush();

                output.push_str(generation);

                return output;
            }

            tokenizer.push_token(generation);

            output.push_str(generation);

            stream_token(
                generation,
                &mut byte_buffer,
            );

            let _ = std::io::stdout().flush();
        }

        if !byte_buffer.is_empty() {
            print!(
                "{}",
                String::from_utf8_lossy(
                    &byte_buffer
                )
            );
        }

        let _ = std::io::stdout().flush();
        println!();

        output
    }

    pub fn poison_check(&self) -> usize {
        let mut count = 0;
        for x in self.param() {
            if !x.is_finite() {
                println!("Non-finite: {}", x);
                count += 1;
            }
        }
        count
    }

    pub fn poison_check_silent(&self) -> usize {
        let mut count = 0;

        for x in self.param() {
            if !x.is_finite() {
                count += 1;
            }
        }

        count
    }

    pub fn set_dataset(&mut self, dataset: Vec<u16>) {
        self.dataset = dataset;
    }

    pub fn load_corpus(&mut self, corpus: &str) {
        println!("Tokenizing corpus...");

        self.dataset = tokenize_u16(corpus, &self.vocab);

        println!(
            "Done! Loaded {} tokens ({} training samples).",
            self.dataset.len(),
            self.dataset
                .len()
                .saturating_sub(self.context_len as usize)
        );
    }

    pub fn load_corpus_silent(&mut self, corpus: &str) {
        self.dataset = tokenize_u16(corpus, &self.vocab);
    }

    pub fn param(&self) -> Vec<f32> {
        let mut params = self.embeddings.parameters();
        params.extend(self.mlp.parameters());

        params.iter()
            .map(|x| x.data())
            .collect()
    }

    pub fn train(
        &mut self,
        batch_update_frequency: Option<usize>,
        checkpoint: Option<&str>,
        checkpoint_path: Option<&str>,
        checkpoint_frequency: CheckpointFrequency,
        lr: Option<f32>,
        seed: Option<u64>,
    ) {
        // ---------------------------------------------------------
        // Load an existing checkpoint, if supplied.
        // ---------------------------------------------------------

        let resume_state = checkpoint
            .as_deref()
            .filter(|path| std::path::Path::new(path).exists())
            .map(|path| {
                println!("Loading checkpoint: {}", path);
                self.load_checkpoint(path, lr)
            });

        // ---------------------------------------------------------
        // Parameters
        // ---------------------------------------------------------

        let mut params = self.embeddings.parameters();
        params.extend(self.mlp.parameters());

        // ---------------------------------------------------------
        // Metadata for checkpoint saving.
        // ---------------------------------------------------------

        let vocab = self.vocab.clone();
        let context_len = self.context_len;
        let hidden_layers = self.hidden_layers.clone();

        // IMPORTANT:
        // Save this before checkpoint_path is moved into the closure.
        let should_save_checkpoints = checkpoint_path.is_some();

        // ---------------------------------------------------------
        // Checkpoint save callback
        // ---------------------------------------------------------

        let mut savefn = move |
            state: CheckpointState,
            mlp: &MLP,
            embeddings: &Embeddings,
        | {
            let Some(base_path) = checkpoint_path.as_deref() else {
                return;
            };

            let suffix = match state.kind {
                CheckpointKind::Batch => {
                    format!("_batch_{}_epoch_{}", state.batch, state.epoch)
                }

                CheckpointKind::Epoch => {
                    format!("_epoch_{}", state.epoch)
                }
            };

            let path = format!("{}{}.check", base_path, suffix);

            let model = SavedLM {
                description: format!(
                    "Training checkpoint | Epoch {} | Batch {} | Sample {}",
                    state.epoch,
                    state.batch,
                    state.sample,
                ),
                mlp: mlp.save(),
                vocab: vocab.clone(),
                context_len,
                hidden_layers: hidden_layers.clone(),
                embeddings: embeddings.save().clone(),
            };

            let saved = SavedCheckpoint {
                model,

                epoch: state.epoch,
                batch: state.batch,
                sample: state.sample,

                sampler_seed: state.sampler_seed,
                sampler_version: state.sampler_version,
                sampler_data_len: state.sampler_data_len,

                lr: state.lr,
                best_loss: state.best_loss,
                plateau_count: state.plateau_count,

                adam_first_moments:
                state.adam_first_moments,

                adam_second_moments:
                state.adam_second_moments,

                adam_step:
                state.adam_step,
            };

            let bytes = bincode::serde::encode_to_vec(
                &saved,
                bincode::config::standard(),
            )
                .unwrap();

            if let Some(parent) = std::path::Path::new(&path).parent() {
                fs::create_dir_all(parent).unwrap();
            }

            fs::write(&path, bytes).unwrap();

            println!("Checkpoint saved: {}", path);
        };

        let savefn: Option<
            &mut dyn FnMut(
                CheckpointState,
                &MLP,
                &Embeddings,
            ),
        > = if should_save_checkpoints {
            Some(&mut savefn)
        } else {
            None
        };

        // ---------------------------------------------------------
        // Training
        // ---------------------------------------------------------

        self.trainer.train_lm(
            1,
            batch_update_frequency,
            resume_state,
            checkpoint_frequency,
            savefn,
            &mut self.mlp,
            &self.dataset,
            self.context_len as usize,
            &self.embeddings,
            params,
            seed.unwrap_or(PermutationSampler::DEFAULT_SEED),
        );
    }

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
                Ok((checkpoint, _)) => checkpoint,

                Err(_) => {
                    println!(
                        "Checkpoint uses legacy format; \
                     loading model/position and \
                     initializing fresh Adam state."
                    );

                    let (
                        old,
                        _,
                    ): (
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

                        // Old optimizer state deliberately discarded.
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
        // Restore model
        // ---------------------------------------------------------

        self.vocab =
            checkpoint.model.vocab;

        self.context_len =
            checkpoint.model.context_len;

        self.hidden_layers =
            checkpoint.model.hidden_layers;

        self.embeddings =
            Arc::new(
                Embeddings::load(
                    checkpoint.model.embeddings
                )
            );

        self.mlp =
            MLP::load_with_embeddings(
                &checkpoint.model.mlp,
                Arc::clone(
                    &self.embeddings
                ),
            );

        println!(
            "Loaded checkpoint: epoch {}, batch {}, sample {}.",
            checkpoint.epoch,
            checkpoint.batch,
            checkpoint.sample,
        );

        // ---------------------------------------------------------
        // Learning rate
        //
        // Explicit LR overrides checkpoint LR.
        // Otherwise resume using checkpoint LR.
        // ---------------------------------------------------------

        let lr_ =
            match lr {
                Some(value) => value,
                None => checkpoint.lr,
            };

        // ---------------------------------------------------------
        // Resume state
        // ---------------------------------------------------------

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
            checkpoint.adam_first_moments,

            adam_second_moments:
            checkpoint.adam_second_moments,

            adam_step:
            checkpoint.adam_step,
        }
    }

    pub fn param_count(&self) -> usize {
        self.mlp.parameter_count()
            + self.embeddings.parameter_count()
    }

    pub fn params(&self) {
        println!("Parameter count: {}.", self.param_count());
        println!(
            "Embedding table: {} params",
            self.embeddings.parameter_count()
        );

        let sequence_length = self.context_len as usize;
        let channels = self.embeddings.embedding_dim();

        println!(
            "O O O O O O O O O O O O O O O O   [{} × {}]",
            sequence_length, channels
        );

        print_layer_specs(
            &self.mlp.layer_specs(),
            sequence_length,
            channels,
            self.vocab.len(),
            0,
        );
    }

    pub fn params_sumyu(&self) {
        println!("Parameter count: {}.", self.param_count());

        let mut prev =
            self.context_len as usize
                * self.embeddings.embedding_dim();
        println!(
            "    Embedding table: {} params",
            self.embeddings.parameter_count()
        );
        println!("    O O O        {} neurons   -   input layer", prev);
        let mut layers = self.hidden_layers.clone();
        layers.push(self.vocab.len());
        for (idx, x) in layers.clone().iter().enumerate() {
            println!(r"    ЖХЖХЖ           {} weights   -   layer {} weights", prev*x, idx+1);
            println!(r"    O O O        {} neurons   -   layer {}", x, idx+1);
            prev = layers[idx]
        }
    }

    pub fn to_saved(&self, description: &str) -> SavedLM {
        SavedLM {
            description: description.to_string(),
            mlp: self.mlp.save(),
            vocab: self.vocab.clone(),
            context_len: self.context_len,
            hidden_layers: self.hidden_layers.clone(),
            embeddings: self.embeddings.save().clone(),
        }
    }

    pub fn from_saved(saved: SavedLM) -> Self {
        println!("Description:\n{}", saved.description);
        println!("Loading...");
        LM::from_saved_silent(saved)
    }

    pub fn from_saved_silent(saved: SavedLM) -> Self {
        let embeddings = Arc::new(
            Embeddings::load(
                saved.embeddings
            )
        );

        Self {
            trainer: Trainer::new(0.0, 0, 0, 0), // defaults; configure later
            mlp: MLP::load_with_embeddings(
                &saved.mlp,
                Arc::clone(&embeddings),
            ),
            dataset: Vec::new(),
            vocab: saved.vocab,
            context_len: saved.context_len,
            hidden_layers: saved.hidden_layers,
            embeddings,
        }
    }

    pub fn embeds(&self) -> Embeddings {
        (*self.embeddings).clone()
    }

    pub fn save(&self, path: &str, description: &str) {
        let saved = self.to_saved(description);
        let bytes = bincode::serde::encode_to_vec(
            &saved,
            bincode::config::standard(),
        ).unwrap();
        fs::write(path, bytes).unwrap();
    }
    
    pub fn load(path: &str) -> (String, Self) {
        let bytes = fs::read(path).unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        (model.description.clone(), LM::from_saved(model))
    }


    pub fn load_silent(path: &str) -> (String, Self) {
        let bytes = fs::read(path).unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        (model.description.clone(), LM::from_saved_silent(model))
    }
}
