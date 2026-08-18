use std::fs;
use crate::neuron::{SavedMLP, MLP, OldSavedMLP};
use crate::Tensor;
use crate::trainer::{CheckpointFrequency, ResumeState, Trainer, TrainInfo, TrainResult, CheckpointKind, CheckpointState};
use crate::embeddings::{Embeddings, OldSavedEmbeddings, SavedEmbeddings};
use serde::{Serialize, Deserialize};
use crate::helper::Config;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;
use std::sync::mpsc::Sender;

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

    pub indices: Vec<usize>,

    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,
}

#[derive(Deserialize)]
pub struct SavedLMf64 {
    mlp: OldSavedMLP,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: OldSavedEmbeddings,
}

#[derive(Deserialize)]
pub struct SavedLMf64desc {
    description: String,
    mlp: OldSavedMLP,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: OldSavedEmbeddings,
}

pub enum LegacyLM {
    NoDesc(SavedLMf64),
    Desc(SavedLMf64desc),
}

pub fn tokenize(text: &str, vocab: &[String]) -> Vec<usize> {
    let trie = crate::helper::Trie::from_vocab(vocab);

    trie.tokenize_u32(text)
        .into_iter()
        .map(|x| x as usize)
        .collect()
}

pub struct LM {
    trainer: Trainer,
    mlp: MLP,
    dataset: Vec<usize>,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: Embeddings,
}

impl LM {
    pub fn new(context_len: u32, vocab: Vec<String>, hidden_dim: &[usize], embedding_dim: usize) -> Self {
        let mut dim = vec![context_len as usize * embedding_dim];
        dim.extend(hidden_dim.to_vec());
        dim.push(vocab.len());
        let embeddings =
            Embeddings::new(
                vocab.len(),
                embedding_dim,
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

    pub fn from_config(config: Config) -> Self {
        let mut dim = vec![config.context_len * config.emb_dim];
        dim.extend(config.hidden_dim.to_vec());
        dim.push(config.vocab.len());
        let embeddings =
            Embeddings::new(
                config.vocab.len(),
                config.emb_dim,
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

    pub fn encode_embeddings(&self, ids: &[usize]) -> Vec<Tensor> {
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

    pub fn encode_nums(&self, string: String) -> Vec<usize> {
        tokenize(&string, &self.vocab)
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

    pub fn generate_one(&self, context: String, temp: f32) -> String {
        let mut ids = self.encode_nums(context);

        if ids.len() > self.context_len as usize {
            ids = ids[ids.len() - self.context_len as usize..ids.len()].to_owned();
        } else if ids.len() < self.context_len as usize {
            let num_to_add = self.context_len as usize - ids.len();
            let mut new_ids = vec![1usize; num_to_add];
            new_ids.extend(ids);
            ids = new_ids;
        }
        let input = self.encode_embeddings(&ids);
        let out: Vec<Tensor> = self.mlp.forward(&input);

        let logits: Vec<f32> = out.iter()
            .map(|x| x.data())
            .collect();

        if temp <= 0.0 {
            let idx = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;

            return self.vocab[idx].clone();
        }

        // Apply temperature
        let scaled_logits: Vec<f32> = logits
            .iter()
            .map(|&x| x / temp)
            .collect();

        // Stable softmax
        let max_logit = scaled_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_logits: Vec<f32> = scaled_logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        let sum_exp: f32 = exp_logits.iter().sum();

        let probs: Vec<f32> = exp_logits
            .iter()
            .map(|&x| x / sum_exp)
            .collect();

        // Sample
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = rng();

        let idx = dist.sample(&mut rng);

        self.vocab[idx].clone()
    }

    pub fn generate_one_distribution(&self, context: String, top_k: usize) {
        let mut ids = self.encode_nums(context);
        if ids.len() > self.context_len as usize {
            ids = ids[ids.len() - self.context_len as usize..ids.len()].to_owned();
        } else if ids.len() < self.context_len as usize {
            let num_to_add = self.context_len as usize-ids.len();
            let mut new_ids = vec![1usize; num_to_add];
            new_ids.extend(ids);
            ids = new_ids
        }
        println!("IDs: {:?}", ids);
        println!("Split text: {:?}", ids.iter().map(|x1| {self.vocab[*x1].clone()}).collect::<Vec<_>>());
        let input = self.encode_embeddings(&ids);
        let out: Vec<Tensor> = self.mlp.forward(&input);
        let logits: Vec<f32> = out.iter().map(|x| x.data()).collect();
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

    pub fn generate(&self, context: String, gen_length: usize, temp: f32) -> String {
        let mut context_ = context.clone();
        let mut output = "".to_string();
        for _ in 0..gen_length {
            let generation = self.generate_one(context_.clone(), temp);
            context_.push_str(&generation);
            output.push_str(&generation);
        }
        output
    }

    pub fn set_dataset(&mut self, dataset: Vec<usize>) {
        self.dataset = dataset;
    }

    pub fn load_corpus(&mut self, corpus: &str) {
        println!("Tokenizing corpus...");

        self.dataset = tokenize(corpus, &self.vocab);

        println!(
            "Done! Loaded {} tokens ({} training samples).",
            self.dataset.len(),
            self.dataset.len().saturating_sub(self.context_len as usize)
        );
    }

    pub fn load_corpus_silent(&mut self, corpus: &str) {
        self.dataset = tokenize(corpus, &self.vocab);
    }

    pub fn param(&self) -> Vec<f32> {
        let mut params = self.mlp.parameters();
        params.extend(self.embeddings.parameters());
        params.iter().map(|x| {x.data()}).collect()
    }

    pub fn train(
        &mut self,
        batch_update_frequency: Option<usize>,
        checkpoint: Option<String>,
        checkpoint_path: Option<String>,
        checkpoint_frequency: CheckpointFrequency,
        lr: Option<f32>,
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

        let mut params = self.mlp.parameters();
        params.extend(self.embeddings.parameters());

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
                    format!("_batch_{}", state.batch)
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
                indices: state.indices,
                lr: state.lr,
                best_loss: state.best_loss,
                plateau_count: state.plateau_count,
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
        );
    }

    pub fn load_checkpoint(&mut self, path: &str, lr: Option<f32>) -> ResumeState {
        let bytes = fs::read(path).unwrap();

        let (checkpoint, _): (SavedCheckpoint, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();

        self.mlp = MLP::load(&checkpoint.model.mlp);
        self.vocab = checkpoint.model.vocab;
        self.context_len = checkpoint.model.context_len;
        self.hidden_layers = checkpoint.model.hidden_layers;
        self.embeddings =
            Embeddings::load(checkpoint.model.embeddings);

        println!(
            "Loaded checkpoint: epoch {}, batch {}, sample {}.",
            checkpoint.epoch,
            checkpoint.batch,
            checkpoint.sample,
        );

        let lr_ = match lr {
            Some(t) => t,
            None => checkpoint.lr
        };

        ResumeState {
            epoch: checkpoint.epoch,
            batch: checkpoint.batch,
            sample: checkpoint.sample,
            indices: checkpoint.indices,
            lr: lr_,
            best_loss: checkpoint.best_loss,
            plateau_count: checkpoint.plateau_count,
        }
    }

    pub fn train_sumyu(
        &mut self,
        tx: Sender<TrainInfo>,
    ) -> TrainResult {
        let mut params = self.mlp.parameters();
        params.extend(self.embeddings.parameters());

        self.trainer.train_lm_sumyu(
            &mut self.mlp,
            &self.dataset,
            self.context_len as usize,
            &self.embeddings,
            params,
            tx,
        )
    }

    pub fn param_count(&self) -> usize {
        let input =
            self.context_len as usize
                * self.embeddings.embedding_dim();
        let mut dims = self.hidden_layers.clone();
        dims.push(self.vocab.len());
        let mut params = 0;
        params += dims.iter().sum::<usize>();
        let mut full_dims = vec![input];
        full_dims.extend(dims);
        for x in 1..full_dims.len() {
            params += full_dims[x] * full_dims[x-1]
        }
        params += self.embeddings.parameter_count();

        params
    }

    pub fn params(&self) {
        println!("Parameter count: {}.", self.param_count());

        let mut prev =
            self.context_len as usize
                * self.embeddings.embedding_dim();
        println!(
            "Embedding table: {} params",
            self.embeddings.parameter_count()
        );
        println!("O O O        {} neurons   -   input layer", prev);
        let mut layers = self.hidden_layers.clone();
        layers.push(self.vocab.len());
        for (idx, x) in layers.clone().iter().enumerate() {
            println!(r"ЖХЖХЖ           {} weights   -   layer {} weights", prev*x, idx+1);
            println!(r"O O O        {} neurons   -   layer {}", x, idx+1);
            prev = layers[idx]
        }
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
        Self {
            trainer: Trainer::new(0.0, 0, 0, 0), // defaults; configure later
            mlp: MLP::load(&saved.mlp),
            dataset: Vec::new(),
            vocab: saved.vocab,
            context_len: saved.context_len,
            hidden_layers: saved.hidden_layers,
            embeddings: Embeddings::load(saved.embeddings),
        }
    }

    pub fn from_saved_silent(saved: SavedLM) -> Self {
        Self {
            trainer: Trainer::new(0.0, 0, 0, 0), // defaults; configure later
            mlp: MLP::load(&saved.mlp),
            dataset: Vec::new(),
            vocab: saved.vocab,
            context_len: saved.context_len,
            hidden_layers: saved.hidden_layers,
            embeddings: Embeddings::load(saved.embeddings),
        }
    }

    pub fn from_saved_legacy(saved: LegacyLM) -> Self {
        match saved {
            LegacyLM::NoDesc(x) => {
                println!("Loading...");
                Self {
                    trainer: Trainer::new(0.0, 0, 0, 0), // defaults; configure later
                    mlp: MLP::load(&x.mlp.into()),
                    dataset: Vec::new(),
                    vocab: x.vocab,
                    context_len: x.context_len,
                    hidden_layers: x.hidden_layers,
                    embeddings: Embeddings::load(x.embeddings.into()),
                }
            }
            LegacyLM::Desc(y) => {
                println!("Description:\n{}", y.description);
                println!("Loading...");
                Self {
                    trainer: Trainer::new(0.0, 0, 0, 0), // defaults; configure later
                    mlp: MLP::load(&y.mlp.into()),
                    dataset: Vec::new(),
                    vocab: y.vocab,
                    context_len: y.context_len,
                    hidden_layers: y.hidden_layers,
                    embeddings: Embeddings::load(y.embeddings.into()),
                }
            }
        }
    }

    pub fn embeds(&self) -> Embeddings {
        self.embeddings.clone()
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

    pub fn load_legacy(path: &str) -> Self {
        let bytes = fs::read(path).unwrap();

        let model = match bincode::serde::decode_from_slice::<SavedLMf64desc, _>(
            &bytes,
            bincode::config::standard(),
        ) {
            Ok((model, _)) => LegacyLM::Desc(model),

            Err(_) => {
                let (model, _) =
                    bincode::serde::decode_from_slice::<SavedLMf64, _>(
                        &bytes,
                        bincode::config::standard(),
                    )
                        .unwrap();

                LegacyLM::NoDesc(model)
            }
        };

        LM::from_saved_legacy(model)
    }
}

impl From<SavedLMf64> for SavedLM {
    fn from(old: SavedLMf64) -> Self {
        SavedLM {
            description: String::new(),
            mlp: old.mlp.into(),
            vocab: old.vocab,
            context_len: old.context_len,
            hidden_layers: old.hidden_layers,
            embeddings: old.embeddings.into(),
        }
    }
}
