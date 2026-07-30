use std::time::Instant;
use rand::prelude::SliceRandom;
use crate::neuron::MLP;
use crate::{Op, Tensor};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub enum TrainResult {
    Finished,
    Interrupted,
}

pub struct Trainer {
    lr: f64,
    epochs: usize,
    batch_size: usize,
    max_batches_per_epoch: usize,
}

impl Trainer {
    pub fn new(lr: f64, epochs: usize, batch_size: usize, max_batches_per_epoch: usize) -> Self {
        Trainer { lr, epochs, batch_size, max_batches_per_epoch }
    }
    
    pub fn reinit_lr(&mut self, lr: f64) {
        self.lr = lr
    }

    pub fn reinit_epochs(&mut self, epochs: usize) {
        self.epochs = epochs
    }

    pub fn reinit_batch(&mut self, batch_size: usize) {
        self.batch_size = batch_size
    }

    pub fn reinit_batch_per_epoch(&mut self, max_batches_per_epoch: usize) {
        self.max_batches_per_epoch = max_batches_per_epoch
    }

    pub fn softmax_cross_entropy(
        logits: &[Tensor],
        target: usize,
    ) -> Tensor {
        let n = logits.len();

        let max_val = logits
            .iter()
            .map(|t| t.data())
            .fold(f64::NEG_INFINITY, f64::max);

        let mut exps = Vec::with_capacity(n);
        let mut sum_exp = 0.0;

        for t in logits {
            let val = t.data();
            let e = (val - max_val).exp();
            exps.push(e);
            sum_exp += e;
        }

        let probs: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

        let loss = -probs[target].max(1e-7).ln();

        Tensor::from_op(
            loss,
            logits.iter().map(|x| x.handle.clone()).collect(),
            Op::SoftmaxCrossEntropy {
                probs,
                target,
            },
        )
    }

    pub fn softmax_cross_entropy_old(
        logits: &[Tensor],
        targets: &[f64],
    ) -> Tensor {
        let n = logits.len();

        let max_val = logits
            .iter()
            .map(|t| t.data())
            .fold(f64::NEG_INFINITY, f64::max);

        let mut exps = Vec::with_capacity(n);
        let mut sum_exp = 0.0;

        for t in logits {
            let val = t.data();
            let e = (val - max_val).exp();
            exps.push(e);
            sum_exp += e;
        }

        let probs: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

        let mut loss = 0.0;
        for (i, &target) in targets.iter().enumerate() {
            if target > 0.5 {
                loss -= probs[i].max(1e-7).ln();
            }
        }

        Tensor::from_op(
            loss,
            logits.iter().map(|x| x.handle.clone()).collect(),
            Op::SoftmaxCrossEntropyOld {
                probs,
                targets: targets.to_vec(),
            },
        )
    }

    pub fn train(
        &mut self,
        update_frequency: usize,
        mlp: &mut MLP,
        dataset: &mut [(Vec<Tensor>, Vec<f64>)],
        params: Vec<Tensor>,
    ) -> TrainResult {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        ctrlc::set_handler(move || {
            println!("\nStopping after current batch...");
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");
        let mut rng = rand::rng();
        let data_len = dataset.len() as f64;
        if self.batch_size > data_len as usize {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }
        if self.batch_size == 0 {
            self.batch_size = data_len as usize;
        }
        let data: &mut [(Vec<Tensor>, Vec<f64>)] = dataset;
        let parameter_boundary = crate::tape_len();
        for epoch in 1..self.epochs+1 {
            let now = Instant::now();
            data.shuffle(&mut rng);
            // zero out gradients in between epoch runs
            for j in params.iter() {
                j.update(self.lr);
                j.zero_grad();
            }
            let mut grad_sum = 0.0;

            // initialize loss
            let mut total_loss = 0.0;
            let mut count = 0;

            for (input, target) in data.iter() {
                count += 1;
                // call forward pass
                let prediction = mlp.forward(input);

                // calc loss
                let loss: Tensor = Trainer::softmax_cross_entropy_old(&prediction, &target);
                total_loss += loss.data();
                loss.backward();
                if count % self.batch_size == 0 {
                    for p in &params {
                        grad_sum += p.grad().abs();
                        p.update(self.lr);
                        p.zero_grad();
                    }
                    crate::clear_tape_after(parameter_boundary);
                    if !running.load(Ordering::SeqCst) {
                        println!("Interrupted");
                        return TrainResult::Interrupted;
                    }
                    if count/self.batch_size >= self.max_batches_per_epoch && self.max_batches_per_epoch != 0 {
                        break;
                    }
                }
            }
            grad_sum += params.iter().map(|p| p.grad().abs()).sum::<f64>();

            if epoch % update_frequency == 0 {
                let elapsed = now.elapsed();
                println!(
                    "Epoch {} | Loss (CE) = {:.6} | Grad sum (avg over samples) = {:.6} | Time elapsed: {:.2?} sec.",
                    epoch, total_loss/data_len, grad_sum/data_len, elapsed
                );
                if grad_sum/data_len <= 1e-9 {
                    println!("Early stopping, network will not learn anymore!");
                    return TrainResult::Finished;
                }
            }
        }

        if data.len() <= 50 {
            println!("\nTesting trained model:");
            for (inputs, target) in data {
                let prediction = mlp.forward(inputs);
                println!(
                    "Input: {:.1?} -> Prediction: {:.4?}, Target: {:.1?}",
                    &inputs.iter().map(|x| { x.data() }).collect::<Vec<_>>(),
                    &prediction.iter().map(|x| { x.data() }).collect::<Vec<_>>(),
                    &target
                );
            }
        }
        TrainResult::Finished
    }

    pub fn train_lm(
        &mut self,
        update_frequency: usize,
        mlp: &mut MLP,
        tokens: &[usize],
        context_len: usize,
        embeddings: &crate::embeddings::Embeddings,
        params: Vec<Tensor>,
    ) -> TrainResult {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        ctrlc::set_handler(move || {
            println!("\nStopping after current batch...");
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");
        let mut rng = rand::rng();
        let data_len = (tokens.len() - context_len) as f64;
        if self.batch_size > data_len as usize {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }
        if self.batch_size == 0 {
            self.batch_size = data_len as usize;
        }
        let parameter_boundary = crate::tape_len();
        for epoch in 1..self.epochs+1 {
            let now = Instant::now();
            let mut indices: Vec<usize> =
                (0..tokens.len() - context_len).collect();

            indices.shuffle(&mut rng);
            // zero out gradients in between epoch runs
            for j in params.iter() {
                j.update(self.lr);
                j.zero_grad();
            }
            let mut grad_sum = 0.0;

            // initialize loss
            let mut total_loss = 0.0;
            let mut count = 0;

            for &sample in &indices {

                let ids =
                    &tokens[sample..sample + context_len];

                let target =
                    tokens[sample + context_len];

                let input =
                    embeddings.encode(ids);
                count += 1;
                // call forward pass
                let prediction = mlp.forward(&input);

                // calc loss
                let loss: Tensor = Trainer::softmax_cross_entropy(&prediction, target);
                total_loss += loss.data();
                loss.backward();
                if count % self.batch_size == 0 {
                    for p in &params {
                        grad_sum += p.grad().abs();
                        p.update(self.lr);
                        p.zero_grad();
                    }
                    crate::clear_tape_after(parameter_boundary);
                    if !running.load(Ordering::SeqCst) {
                        println!("Interrupted");
                        return TrainResult::Interrupted;
                    }
                    if count/self.batch_size >= self.max_batches_per_epoch && self.max_batches_per_epoch != 0 {
                        break;
                    }
                }
            }
            grad_sum += params.iter().map(|p| p.grad().abs()).sum::<f64>();

            if epoch % update_frequency == 0 {
                let elapsed = now.elapsed();
                println!(
                    "Epoch {} | Loss (CE) = {:.6} | Grad sum (avg over samples) = {:.6} | Time elapsed: {:.2?} sec.",
                    epoch, total_loss/data_len, grad_sum/data_len, elapsed
                );
                if grad_sum/data_len <= 1e-9 {
                    println!("Early stopping, network will not learn anymore!");
                    return TrainResult::Finished;
                }
            }
        }
        TrainResult::Finished
    }
}
