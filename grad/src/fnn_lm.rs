use std::collections::HashMap;
use crate::neuron::{SavedMLP, MLP};
use crate::Tensor;
use crate::trainer::Trainer;
use crate::embeddings::{Embeddings, SavedEmbeddings};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct SavedLM {
    mlp: SavedMLP,
    vocab: Vec<String>,
    context_len: u32,
    hidden_layers: Vec<usize>,
    embeddings: SavedEmbeddings,
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

    pub fn encode_embeddings(&self, ids: &[usize]) -> Vec<Tensor> {
        self.embeddings.encode(ids)
    }

    pub fn train_options(&mut self, lr: f64, epochs: usize, batch_size: usize, max_batches_per_epoch: usize) {
        self.trainer.reinit_lr(lr);
        self.trainer.reinit_epochs(epochs);
        self.trainer.reinit_batch(batch_size);
        self.trainer.reinit_batch_per_epoch(max_batches_per_epoch);
    }

    pub fn encode_nums(&self, string: String) -> Vec<usize> {
        tokenize(&*string, &self.vocab)
    }

    pub fn encode_one_hot_from_nums(&self, nums: Vec<usize>) -> Vec<Tensor> {
        let mut one_hot: Vec<Tensor> = vec![Tensor::new(0f64); nums.len()*self.vocab.len()];
        for (idx, x) in nums.iter().enumerate() {
            one_hot[x+(idx*self.vocab.len())] = Tensor::new(1f64);
        }
        one_hot
    }

    pub fn decode_one_hot_char(&self, one_hot: Vec<Tensor>) -> &String {
        let num = one_hot.iter().position(|x| {x.data() == 1.0}).unwrap();
        &self.vocab[num]
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

    pub fn generate_one(&self, context: String) -> &String {
        let mut new_context: String = context.clone();
        if context.len() > self.context_len as usize {
            new_context = context[context.len() - self.context_len as usize..context.len()].parse().unwrap();
        }
        let ids = self.encode_nums(new_context);

        let input = self.encode_embeddings(&ids);
        let out: Vec<Tensor> = self.mlp.forward(&*input);
        &self.vocab[self.max(out)]
    }

    pub fn generate_one_distribution(&self, context: String, top_k: usize) {
        let mut new_context = context.clone();
        if context.len() > self.context_len as usize {
            new_context = context[context.len() - self.context_len as usize..]
                .parse()
                .unwrap();
        }
        let ids = self.encode_nums(new_context);
        let input = self.encode_embeddings(&ids);
        let out: Vec<Tensor> = self.mlp.forward(&*input);
        let logits: Vec<f64> = out.iter().map(|x| x.data()).collect();
        let max_logit = logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        let probs: Vec<f64> = exp_logits
            .iter()
            .map(|&x| x / sum_exp)
            .collect();
        let mut newest: Vec<(usize, f64)> = probs.into_iter().enumerate().collect();
        newest.sort_by(|a, b| b.1.total_cmp(&a.1));

        for i in 0..top_k.min(newest.len()) {
            let (idx, prob) = newest[i];
            println!("{}- {:.2}%", self.vocab[idx], prob * 100.0);
        }
    }

    pub fn generate(&self, context: String, gen_length: usize) -> String {
        let mut context_ = context.clone();
        let mut output = "".to_string();
        for _ in 0..gen_length {
            let generation = self.generate_one(context_.clone());
            context_.push_str(generation);
            output.push_str(generation);
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

    pub fn to_saved(&self) -> SavedLM {
        SavedLM {
            mlp: self.mlp.save(),
            vocab: self.vocab.clone(),
            context_len: self.context_len,
            hidden_layers: self.hidden_layers.clone(),
            embeddings: self.embeddings.save().clone(),
        }
    }

    pub fn from_saved(saved: SavedLM) -> Self {
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
}
