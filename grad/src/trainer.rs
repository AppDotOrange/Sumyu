use std::time::{Duration, Instant};
use rand::prelude::SliceRandom;
use crate::neuron::MLP;
use crate::{Op, Tensor};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::sync::mpsc::Sender;
use crate::batched::softmax_cross_entropy_batch;

pub enum TrainResult {
    Finished,
    Interrupted,
}

pub struct TrainInfo {
    pub epoch: usize,
    pub loss: f32,
    pub perplexity: f32,
    pub time: Duration,
    pub done: bool,
}

pub struct Trainer {
    lr: f32,
    epochs: usize,
    batch_size: usize,
    max_batches_per_epoch: usize,
}

impl Trainer {
    pub fn new(lr: f32, epochs: usize, batch_size: usize, max_batches_per_epoch: usize) -> Self {
        Trainer { lr: lr/batch_size as f32, epochs, batch_size, max_batches_per_epoch }
    }
    
    pub fn reinit_lr(&mut self, lr: f32) {
        self.lr = lr/self.batch_size as f32
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
            .fold(f32::NEG_INFINITY, f32::max);

        let mut exps = Vec::with_capacity(n);
        let mut sum_exp = 0.0;

        for t in logits {
            let val = t.data();
            let e = (val - max_val).exp();
            exps.push(e);
            sum_exp += e;
        }

        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        let loss = -probs[target].max(1e-7).ln();

        Tensor::from_op(
            loss,
            logits.iter().map(|x| x.handle).collect(),
            Op::SoftmaxCrossEntropy {
                probs,
                target,
            },
        )
    }

    pub fn softmax_cross_entropy_old(
        logits: &[Tensor],
        targets: &[f32],
    ) -> Tensor {
        let n = logits.len();

        let max_val = logits
            .iter()
            .map(|t| t.data())
            .fold(f32::NEG_INFINITY, f32::max);

        let mut exps = Vec::with_capacity(n);
        let mut sum_exp = 0.0;

        for t in logits {
            let val = t.data();
            let e = (val - max_val).exp();
            exps.push(e);
            sum_exp += e;
        }

        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        let mut loss = 0.0;
        for (i, &target) in targets.iter().enumerate() {
            if target > 0.5 {
                loss -= probs[i].max(1e-7).ln();
            }
        }

        Tensor::from_op(
            loss,
            logits.iter().map(|x| x.handle).collect(),
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
        dataset: &mut [(Vec<Tensor>, Vec<f32>)],
        params: Vec<Tensor>,
    ) -> TrainResult {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        ctrlc::set_handler(move || {
            println!("\nStopping after current batch...");
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");
        let mut rng = rand::rng();
        let data_len = dataset.len() as f32;
        if self.batch_size > data_len as usize {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }
        if self.batch_size == 0 {
            self.batch_size = data_len as usize;
        }
        let data: &mut [(Vec<Tensor>, Vec<f32>)] = dataset;
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
                let loss: Tensor = Trainer::softmax_cross_entropy_old(&prediction, target);
                total_loss += loss.data();
                loss.backward(parameter_boundary);
                crate::clear_tape_after(parameter_boundary);
                if count % self.batch_size == 0 {
                    for p in &params {
                        grad_sum += p.grad().abs();
                        p.update(self.lr);
                        p.zero_grad();
                    }
                    if !running.load(Ordering::SeqCst) {
                        println!("Interrupted");
                        return TrainResult::Interrupted;
                    }
                    if self.max_batches_per_epoch != 0
                        && count / self.batch_size >= self.max_batches_per_epoch
                    {
                        break;
                    }
                }
            }
            grad_sum += params.iter().map(|p| p.grad().abs()).sum::<f32>();

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
        })
            .expect("Error setting Ctrl+C handler");

        let mut rng = rand::rng();

        let data_len = tokens.len().saturating_sub(context_len);

        if data_len == 0 {
            println!("Dataset is too small for the selected context length!");
            return TrainResult::Finished;
        }

        if self.batch_size > data_len {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }

        if self.batch_size == 0 {
            self.batch_size = data_len;
        }

        let parameter_boundary = crate::tape_len();

        let mut lr = self.lr;
        let min_lr = 0.001 / self.batch_size as f32;

        let mut best_loss = f32::MAX;
        let mut plateau_count = 0;

        // MLP input size = context_len * embedding_dim.
        let input_size =
            context_len * embeddings.embedding_dim();

        for epoch in 1..self.epochs + 1 {
            let now = Instant::now();

            // Shuffle sample positions rather than copying samples.
            let mut indices: Vec<usize> =
                (0..data_len).collect();

            indices.shuffle(&mut rng);

            let mut grad_sum = 0.0f32;
            let mut total_loss = 0.0f32;
            let mut count = 0usize;
            let mut batches_done = 0usize;

            // ---------------------------------------------------------
            // Optional timing
            // ---------------------------------------------------------

            #[cfg(feature = "timing")]
            let mut encode_time = Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut forward_time = Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut loss_time = Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut backward_time = Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut embedding_grad_time = Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut clear_tape_time = Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut update_time = Duration::ZERO;

            // ---------------------------------------------------------
            // Process batches
            // ---------------------------------------------------------

            while count < data_len {
                if self.max_batches_per_epoch != 0
                    && batches_done >= self.max_batches_per_epoch
                {
                    break;
                }

                let remaining = data_len - count;

                let current_batch =
                    self.batch_size.min(remaining);

                // -----------------------------------------------------
                // Build the batch's token IDs.
                //
                // Each sample needs:
                //
                //   context_len input tokens
                //   + 1 target token
                //
                // We keep the IDs contiguous so encode_batch() can
                // directly turn them into [batch × input] floats.
                // -----------------------------------------------------

                let mut batch_ids =
                    Vec::with_capacity(
                        current_batch * context_len
                    );

                let mut targets =
                    Vec::with_capacity(current_batch);

                for batch_index in 0..current_batch {
                    let sample =
                        indices[count + batch_index];

                    batch_ids.extend_from_slice(
                        &tokens[
                            sample
                                ..sample + context_len
                            ],
                    );

                    targets.push(
                        tokens[sample + context_len]
                    );
                }

                // -----------------------------------------------------
                // Embedding lookup
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let input =
                    embeddings.encode_batch(
                        &batch_ids,
                        current_batch,
                        context_len,
                    );

                #[cfg(feature = "timing")]
                {
                    encode_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Batched MLP forward
                //
                // Everything here is SGEMM.
                //
                // input:
                //     batch × input_size
                //
                // Each dense layer:
                //     X × Wᵀ
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let forward =
                    mlp.forward_batch(
                        &input,
                        current_batch,
                        input_size,
                    );

                #[cfg(feature = "timing")]
                {
                    forward_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Batched softmax cross entropy
                // -----------------------------------------------------

                let output_size =
                    forward.output_size;

                let mut output_grads =
                    vec![
                        0.0f32;
                        current_batch * output_size
                    ];

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let batch_loss =
                    softmax_cross_entropy_batch(
                        &forward.output,
                        &targets,
                        &mut output_grads,
                        current_batch,
                        output_size,
                    );

                total_loss += batch_loss;

                #[cfg(feature = "timing")]
                {
                    loss_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Batched MLP backward
                //
                // Each dense layer uses:
                //
                // dW = dYᵀ X       SGEMM
                // dX = dY W        SGEMM
                //
                // Parameter gradients are accumulated directly into
                // the Tensor tape.
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let input_grads =
                    mlp.backward_batch(
                        &forward,
                        &output_grads,
                    );

                #[cfg(feature = "timing")]
                {
                    backward_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Embedding gradients
                //
                // input_grads is:
                //
                //     batch × (context_len × embedding_dim)
                //
                // Accumulate each position back into its token's
                // embedding vector.
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                embeddings.accumulate_batch_grads(
                    &batch_ids,
                    &input_grads,
                    current_batch,
                    context_len,
                );

                #[cfg(feature = "timing")]
                {
                    embedding_grad_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Clear temporary autograd tape.
                //
                // The batched path doesn't create the normal per-sample
                // forward graph, but keep the same tape boundary
                // contract as the rest of the trainer.
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                crate::clear_tape_after(
                    parameter_boundary
                );

                #[cfg(feature = "timing")]
                {
                    clear_tape_time += timer.elapsed();
                }

                count += current_batch;
                batches_done += 1;

                // -----------------------------------------------------
                // Update after every batch.
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                grad_sum +=
                    crate::zero_grad_and_update(
                        &params,
                        lr,
                    );

                #[cfg(feature = "timing")]
                {
                    update_time += timer.elapsed();
                }

                if !running.load(Ordering::SeqCst) {
                    println!("Interrupted");
                    return TrainResult::Interrupted;
                }
            }

            let samples = count as f32;

            if samples == 0.0 {
                continue;
            }

            let avg_loss =
                total_loss / samples;

            let perplexity =
                avg_loss.exp();

            // ---------------------------------------------------------
            // Learning-rate plateau detection
            // ---------------------------------------------------------

            if avg_loss < best_loss * 0.998 {
                best_loss = avg_loss;
                plateau_count = 0;
            } else {
                plateau_count += 1;
            }

            if plateau_count >= 10 {
                lr = (lr * 0.8).max(min_lr);
                plateau_count = 0;

                println!("LR reduced to {}", lr);
            }

            // ---------------------------------------------------------
            // Epoch statistics
            // ---------------------------------------------------------

            if epoch % update_frequency == 0 {
                let elapsed = now.elapsed();

                println!(
                    "Epoch {} | Loss (CE) = {:.6} | Grad sum (avg over samples) = {:.6} | Perplexity = {:.6} | Time elapsed: {:.2?} sec.",
                    epoch,
                    avg_loss,
                    grad_sum / samples,
                    perplexity,
                    elapsed,
                );

                // -----------------------------------------------------
                // Optional timing breakdown
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                {
                    let elapsed_secs =
                        elapsed.as_secs_f64();

                    let pct = |duration: Duration| {
                        duration.as_secs_f64()
                            / elapsed_secs
                            * 100.0
                    };

                    println!(
                        "  Encode:          {:>10.3?} ({:>6.2}%)",
                        encode_time,
                        pct(encode_time)
                    );

                    println!(
                        "  Forward SGEMM:   {:>10.3?} ({:>6.2}%)",
                        forward_time,
                        pct(forward_time)
                    );

                    println!(
                        "  Loss:            {:>10.3?} ({:>6.2}%)",
                        loss_time,
                        pct(loss_time)
                    );

                    println!(
                        "  Backward SGEMM:  {:>10.3?} ({:>6.2}%)",
                        backward_time,
                        pct(backward_time)
                    );

                    println!(
                        "  Embedding grad:  {:>10.3?} ({:>6.2}%)",
                        embedding_grad_time,
                        pct(embedding_grad_time)
                    );

                    println!(
                        "  Clear tape:      {:>10.3?} ({:>6.2}%)",
                        clear_tape_time,
                        pct(clear_tape_time)
                    );

                    println!(
                        "  Update:          {:>10.3?} ({:>6.2}%)",
                        update_time,
                        pct(update_time)
                    );
                }

                if grad_sum / samples <= 1e-9 {
                    println!(
                        "Early stopping, network will not learn anymore!"
                    );

                    return TrainResult::Finished;
                }
            }
        }

        TrainResult::Finished
    }


    pub fn train_lm_sumyu(
        &mut self,
        update_frequency: usize,
        mlp: &mut MLP,
        tokens: &[usize],
        context_len: usize,
        embeddings: &crate::embeddings::Embeddings,
        params: Vec<Tensor>,
        tx: Sender<TrainInfo>,
    ) -> TrainResult {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        ctrlc::set_handler(move || {
            println!("\nStopping after current batch...");
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");
        let mut rng = rand::rng();
        let data_len = (tokens.len() - context_len) as f32;
        if self.batch_size > data_len as usize {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }
        if self.batch_size == 0 {
            self.batch_size = data_len as usize;
        }
        let parameter_boundary = crate::tape_len();
        let mut lr = self.lr;
        let min_lr = 0.001 / self.batch_size as f32;
        let mut best_loss = f32::MAX;
        let mut plateau_count = 0;
        for epoch in 1..self.epochs+1 {
            let now = Instant::now();
            let mut indices: Vec<usize> =
                (0..tokens.len() - context_len).collect();

            indices.shuffle(&mut rng);
            // zero out gradients in between epoch runs
            crate::zero_grad_and_update(&params, lr);
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
                loss.backward(parameter_boundary);
                crate::clear_tape_after(parameter_boundary);
                if count % self.batch_size == 0 {
                    grad_sum += params.iter().map(|x| {x.grad().abs()}).sum::<f32>();
                    crate::zero_grad_and_update(&params, lr);
                    if !running.load(Ordering::SeqCst) {
                        println!("Interrupted");
                        return TrainResult::Interrupted;
                    }
                    if count/self.batch_size >= self.max_batches_per_epoch && self.max_batches_per_epoch != 0 {
                        break;
                    }
                }
            }
            grad_sum += params.iter().map(|p| p.grad().abs()).sum::<f32>();

            let samples = count as f32;
            let avg_loss = total_loss / samples;

            if avg_loss < best_loss * 0.998 {
                best_loss = avg_loss;
                plateau_count = 0;
            } else {
                plateau_count += 1;
            }

            if plateau_count >= 10 {
                lr = (lr * 0.8).max(min_lr);
                plateau_count = 0;
                println!("LR reduced to {}", lr);
            }
            let elapsed = now.elapsed();
            if epoch % update_frequency == 0 {
                println!(
                    "Epoch {} | Loss (CE) = {:.6} | Grad sum (avg over samples) = {:.6} | Perplexity = {:.6} | Time elapsed: {:.2?} sec.",
                    epoch, total_loss/samples, grad_sum/samples, (total_loss/samples).exp(), elapsed,
                );
            }
            let info = TrainInfo {
                epoch,
                loss: total_loss/samples,
                perplexity: (total_loss/samples).exp(),
                time: elapsed,
                done: false,
            };
            if grad_sum/samples <= 1e-9 {
                println!("Early stopping, network will not learn anymore!");
                let info = TrainInfo {
                    epoch,
                    loss: total_loss/samples,
                    perplexity: (total_loss/samples).exp(),
                    time: elapsed,
                    done: true,
                };
                tx.send(info).unwrap();
                return TrainResult::Finished;
            }
            tx.send(info).unwrap();
        }
        TrainResult::Finished
    }
}
