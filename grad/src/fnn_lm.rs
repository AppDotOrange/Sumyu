use std::collections::HashMap;
use std::fs;
use crate::neuron::{SavedMLP, MLP, OldSavedMLP};
use crate::Tensor;
use crate::trainer::Trainer;
use crate::embeddings::{Embeddings, OldSavedEmbeddings, SavedEmbeddings};
use serde::{Serialize, Deserialize};
use crate::helper::Config;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;

#[derive(Serialize, Deserialize)]
pub struct SavedLM {
    description: String,
    mlp: SavedMLP,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: SavedEmbeddings,
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

pub fn tokenize(text: &str, vocab: &Vec<String>) -> Vec<usize> {
    struct Node {
        children: HashMap<char, usize>,
        id: Option<usize>,
    }
    let total_chars: usize = vocab.iter().map(|s| s.len()).sum();
    let mut nodes: Vec<Node> = Vec::with_capacity(total_chars + 1);

    // Create root node (index 0)
    nodes.push(Node { children: HashMap::new(), id: None });

    // 1. Build Trie using indices (avoids borrow checker conflicts)
    for (id, token) in vocab.iter().enumerate() {
        let mut current_idx = 0; // Start at root

        for ch in token.chars() {
            // Check if child exists
            if let Some(&next_idx) = nodes[current_idx].children.get(&ch) {
                current_idx = next_idx;
            } else {
                // Create new node
                let new_idx = nodes.len();
                nodes.push(Node { children: HashMap::new(), id: None });

                nodes[current_idx].children.insert(ch, new_idx);
                current_idx = new_idx;
            }
        }
        // Mark the end of the token
        nodes[current_idx].id = Some(id);
    }

    // 2. Encode text (Greedy Longest Match)
    let chars: Vec<char> = text.chars().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let mut current_idx = 0;
        let mut best_match_id: Option<usize> = None;
        let mut match_len = 0;
        let mut j = i;

        // Traverse Trie to find longest match starting at i
        while j < chars.len() {
            let ch = chars[j];
            // Use immutable borrow to check children
            if let Some(&next_idx) = nodes[current_idx].children.get(&ch) {
                current_idx = next_idx;
                j += 1;

                // If this node is a valid token end, remember it
                if let Some(id) = nodes[current_idx].id {
                    best_match_id = Some(id);
                    match_len = j - i;
                }
            } else {
                break;
            }
        }

        if let Some(id) = best_match_id {
            result.push(id);
            i += match_len;
        } else {
            result.push(0);  // unknown char
            i += 1;
        }
    }
    result
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

        let trainer = Trainer::new(config.lr / config.batch_size as f32, config.epochs, config.batch_size, config.max_batches_per_epoch);
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

    pub fn train_options(&mut self, lr: f32, epochs: usize, batch_size: usize, max_batches_per_epoch: usize) {
        self.trainer.reinit_lr(lr/batch_size as f32);
        self.trainer.reinit_epochs(epochs);
        self.trainer.reinit_batch(batch_size);
        self.trainer.reinit_batch_per_epoch(max_batches_per_epoch);
    }

    pub fn encode_nums(&self, string: String) -> Vec<usize> {
        tokenize(&*string, &self.vocab)
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
        let out: Vec<Tensor> = self.mlp.forward(&*input);

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
        let out: Vec<Tensor> = self.mlp.forward(&*input);
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

        for i in 0..top_k.min(newest.len()) {
            let (idx, prob) = newest[i];
            println!("{}- {:.2}%", self.vocab[idx], prob * 100.0);
        }
    }

    pub fn generate(&self, context: String, gen_length: usize, temp: f32) -> String {
        let mut context_ = context.clone();
        let mut output = "".to_string();
        for _ in 0..gen_length {
            let generation = self.generate_one(context_.clone(), temp);
            context_.push_str(&*generation);
            output.push_str(&*generation);
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

    pub fn train(&mut self) {
        let mut params = self.mlp.parameters();
        params.extend(self.embeddings.parameters());

        self.trainer.train_lm(
            1,
            &mut self.mlp,
            &self.dataset,
            self.context_len as usize,
            &self.embeddings,
            params,
        );
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
