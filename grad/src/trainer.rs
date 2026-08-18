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
use crate::embeddings::Embeddings;

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

#[derive(Clone, Copy, Debug)]
pub enum CheckpointKind {
    Batch,
    Epoch,
}

#[derive(Clone, Copy, Debug)]
pub enum CheckpointFrequency {
    Disabled,
    EveryEpoch(usize),
    EveryBatch(usize),
}

#[derive(Debug)]
pub struct ResumeState {
    pub epoch: usize,
    pub batch: usize,
    pub sample: usize,
    pub indices: Vec<usize>,
    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,
}

pub struct CheckpointState {
    pub kind: CheckpointKind,
    pub epoch: usize,
    pub batch: usize,
    pub sample: usize,
    pub indices: Vec<usize>,
    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,
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
        let param_count = params.len() as f32;

        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        ctrlc::set_handler(move || {
            println!("\nStopping after current batch...");
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");

        let mut rng = rand::rng();

        let data_len = dataset.len();

        if data_len == 0 {
            println!("Dataset is empty!");
            return TrainResult::Finished;
        }

        if self.batch_size > data_len {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }

        if self.batch_size == 0 {
            self.batch_size = data_len;
        }

        let input_size = dataset[0].0.len();
        let output_size = dataset[0].1.len();

        if input_size == 0 || output_size == 0 {
            println!("Input/output vectors cannot be empty!");
            return TrainResult::Finished;
        }

        // All samples must have the same input/output dimensions.
        for (input, target) in dataset.iter() {
            assert_eq!(
                input.len(),
                input_size,
                "All inputs must have the same size"
            );

            assert_eq!(
                target.len(),
                output_size,
                "All targets must have the same size"
            );
        }

        let mut indices: Vec<usize> = (0..data_len).collect();

        let mut lr = self.lr;
        let min_lr = 0.001 / self.batch_size as f32;

        let mut best_loss = f32::MAX;
        let mut plateau_count = 0;

        for epoch in 1..self.epochs + 1 {
            let now = Instant::now();

            indices.shuffle(&mut rng);

            let mut total_loss = 0.0f32;
            let mut grad_sum = 0.0f32;

            let mut count = 0usize;
            let mut batches_done = 0usize;

            while count < data_len {
                if self.max_batches_per_epoch != 0
                    && batches_done >= self.max_batches_per_epoch
                {
                    break;
                }

                let remaining = data_len - count;
                let current_batch = self.batch_size.min(remaining);

                // ---------------------------------------------------------
                // Pack inputs and targets into contiguous batch arrays.
                // ---------------------------------------------------------

                let mut batch_input =
                    Vec::with_capacity(current_batch * input_size);

                let mut targets =
                    Vec::with_capacity(current_batch * output_size);

                for batch_index in 0..current_batch {
                    let sample = indices[count + batch_index];

                    let (input, target) = &dataset[sample];

                    for x in input {
                        batch_input.push(x.data());
                    }

                    targets.extend_from_slice(target);
                }

                // ---------------------------------------------------------
                // Batched forward — SGEMM
                // ---------------------------------------------------------

                let forward = mlp.forward_batch(
                    &batch_input,
                    current_batch,
                    input_size,
                );

                debug_assert_eq!(
                    forward.output_size,
                    output_size
                );

                // ---------------------------------------------------------
                // MSE + output gradients
                // ---------------------------------------------------------

                let mut output_grads =
                    vec![0.0f32; current_batch * output_size];

                let batch_loss = crate::batched::mse_batch(
                    &forward.output,
                    &targets,
                    &mut output_grads,
                    current_batch,
                    output_size,
                );

                // mse_batch() returns the mean for this batch.
                // Weight it by the number of samples so a smaller final
                // batch doesn't get the same influence as a full batch.
                total_loss += batch_loss * current_batch as f32;

                // ---------------------------------------------------------
                // Batched backward — SGEMM
                // ---------------------------------------------------------

                mlp.backward_batch(
                    &forward,
                    &output_grads,
                );

                // ---------------------------------------------------------
                // Update parameters
                // ---------------------------------------------------------

                grad_sum += crate::zero_grad_and_update(
                    &params,
                    lr,
                );

                count += current_batch;
                batches_done += 1;

                if !running.load(Ordering::SeqCst) {
                    println!("Interrupted");
                    return TrainResult::Interrupted;
                }
            }

            let samples = count as f32;

            if samples == 0.0 {
                continue;
            }

            // Since each batch MSE was weighted by its sample count,
            // this is the epoch-wide MSE.
            let avg_loss = total_loss / samples;

            let grad_avg = grad_sum / samples;

            // -------------------------------------------------------------
            // Learning-rate plateau detection
            // -------------------------------------------------------------

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

            // -------------------------------------------------------------
            // Statistics
            // -------------------------------------------------------------

            if epoch % update_frequency == 0 {
                println!(
                    "Epoch {} | Loss (MSE) = {:.6} | Grad sum (avg per param) = {:.8} | Time elapsed: {:.2?} sec.",
                    epoch,
                    avg_loss,
                    grad_avg / param_count,
                    elapsed,
                );
            }

            if grad_avg <= 1e-9 {
                println!("Early stopping, network will not learn anymore!");
                return TrainResult::Finished;
            }
        }

        // Keep the old small-dataset inspection.
        if data_len <= 50 {
            println!("\nTesting trained model:");

            for (inputs, target) in dataset.iter() {
                let prediction = mlp.forward(inputs);

                println!(
                    "Input: {:.4?} -> Prediction: {:.4?}, Target: {:.4?}",
                    inputs
                        .iter()
                        .map(|x| x.data())
                        .collect::<Vec<_>>(),
                    prediction
                        .iter()
                        .map(|x| x.data())
                        .collect::<Vec<_>>(),
                    target
                );
            }
        }

        TrainResult::Finished
    }

    pub fn train_lm(
        &mut self,
        update_frequency: usize,
        batch_update_frequency: Option<usize>,
        mut resume: Option<ResumeState>,
        checkpoint_frequency: CheckpointFrequency,
        mut savefn: Option<
            &mut dyn FnMut(
                CheckpointState,
                &MLP,
                &Embeddings,
            ),
        >,
        mlp: &mut MLP,
        tokens: &[usize],
        context_len: usize,
        embeddings: &Embeddings,
        params: Vec<Tensor>,
    ) -> TrainResult {
        use std::io::{self, Write};
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        let param_count = params.len() as f32;

        // ---------------------------------------------------------
        // Ctrl+C handling
        //
        // The signal handler only sets a flag.
        // It never prints or performs I/O.
        //
        // compare_exchange means repeated Ctrl+C presses are ignored
        // while an interrupt is already pending.
        // ---------------------------------------------------------

        let interrupt_requested = Arc::new(AtomicBool::new(false));
        let interrupt_flag = interrupt_requested.clone();

        ctrlc::set_handler(move || {
            let _ = interrupt_flag.compare_exchange(
                false,
                true,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
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
        let mut plateau_count = 0usize;

        // ---------------------------------------------------------
        // Resume state
        //
        // Load this ONCE, before entering the epoch loop.
        // ---------------------------------------------------------

        let (start_epoch, mut resume_state) = match resume.take() {
            Some(state) => {
                println!(
                    "Resuming from epoch {}, batch {} (sample {}/{}).",
                    state.epoch,
                    state.batch,
                    state.sample,
                    data_len,
                );

                lr = state.lr;
                best_loss = state.best_loss;
                plateau_count = state.plateau_count;

                (state.epoch, Some(state))
            }

            None => (1, None),
        };

        // MLP input size = context_len * embedding_dim.
        let input_size = context_len * embeddings.embedding_dim();

        // ---------------------------------------------------------
        // Checkpoint helper.
        // ---------------------------------------------------------

        let mut make_checkpoint =
            |kind: CheckpointKind,
             epoch: usize,
             batch: usize,
             sample: usize,
             indices: &[usize],
             lr: f32,
             best_loss: f32,
             plateau_count: usize| {
                if let Some(savefn) = savefn.as_mut() {
                    savefn(
                        CheckpointState {
                            kind,
                            epoch,
                            batch,
                            sample,
                            indices: indices.to_vec(),
                            lr,
                            best_loss,
                            plateau_count,
                        },
                        mlp,
                        embeddings,
                    );
                }
            };

        // ---------------------------------------------------------
        // Process epochs
        // ---------------------------------------------------------

        for epoch in start_epoch..self.epochs + 1 {
            let now = Instant::now();

            let mut indices: Vec<usize>;
            let mut count: usize;
            let mut batches_done: usize;

            let mut grad_sum = 0.0f32;
            let mut total_loss = 0.0f32;

            // ---------------------------------------------------------
            // Restore the checkpoint ONLY for its exact epoch.
            //
            // Since resume_state is consumed with take(), this can
            // happen only once: on the first epoch after loading.
            // ---------------------------------------------------------

            if let Some(state) = resume_state.take() {
                debug_assert_eq!(state.epoch, epoch);

                indices = state.indices;
                count = state.sample;
                batches_done = state.batch;

                println!(
                    "Continuing epoch {} from batch {}.",
                    epoch,
                    batches_done,
                );
            } else {
                indices = (0..data_len).collect();
                indices.shuffle(&mut rng);

                count = 0;
                batches_done = 0;
            }
            let epoch_start_count = count;

            let total_batches =
                (data_len + self.batch_size - 1) / self.batch_size;

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
                let current_batch = self.batch_size.min(remaining);

                // -----------------------------------------------------
                // Build the batch's token IDs.
                //
                // Each sample:
                //   context_len input tokens
                //   + 1 target token
                // -----------------------------------------------------

                let mut batch_ids =
                    Vec::with_capacity(current_batch * context_len);

                let mut targets =
                    Vec::with_capacity(current_batch);

                for batch_index in 0..current_batch {
                    let sample = indices[count + batch_index];

                    batch_ids.extend_from_slice(
                        &tokens[sample..sample + context_len]
                    );

                    targets.push(tokens[sample + context_len]);
                }

                // -----------------------------------------------------
                // Embedding lookup
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let input = embeddings.encode_batch(
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
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let forward = mlp.forward_batch(
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

                let output_size = forward.output_size;

                let mut output_grads =
                    vec![0.0f32; current_batch * output_size];

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let batch_loss = softmax_cross_entropy_batch(
                    &forward.output,
                    &targets,
                    &mut output_grads,
                    current_batch,
                    output_size,
                );

                // batch_loss is the sum over this batch.
                total_loss += batch_loss;

                #[cfg(feature = "timing")]
                {
                    loss_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Batched MLP backward
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                let input_grads = mlp.backward_batch(
                    &forward,
                    &output_grads,
                );

                #[cfg(feature = "timing")]
                {
                    backward_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Embedding gradients
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
                // Clear temporary autograd tape
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                crate::clear_tape_after(parameter_boundary);

                #[cfg(feature = "timing")]
                {
                    clear_tape_time += timer.elapsed();
                }

                count += current_batch;
                batches_done += 1;

                // -----------------------------------------------------
                // Update parameters
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer = Instant::now();

                grad_sum += crate::zero_grad_and_update(
                    &params,
                    lr,
                );

                #[cfg(feature = "timing")]
                {
                    update_time += timer.elapsed();
                }

                // -----------------------------------------------------
                // Periodic batch checkpoint
                //
                // IMPORTANT:
                // This happens AFTER the parameter update.
                // -----------------------------------------------------

                let should_checkpoint = match checkpoint_frequency {
                    CheckpointFrequency::EveryBatch(n) => {
                        n > 0 && batches_done % n == 0
                    }

                    _ => false,
                };

                if should_checkpoint {
                    make_checkpoint(
                        CheckpointKind::Batch,
                        epoch,
                        batches_done,
                        count,
                        &indices,
                        lr,
                        best_loss,
                        plateau_count,
                    );
                }

                // -----------------------------------------------------
                // Intra-epoch progress update
                // -----------------------------------------------------

                if let Some(frequency) = batch_update_frequency {
                    if frequency > 0 && batches_done % frequency == 0 {
                        let elapsed = now.elapsed();

                        // Samples processed since this training invocation
                        // (not the absolute checkpoint position).
                        let processed_samples =
                            count.saturating_sub(epoch_start_count);

                        // Progress through the REMAINING portion of the epoch.
                        let remaining_at_start =
                            data_len.saturating_sub(epoch_start_count);

                        let progress =
                            if remaining_at_start > 0 {
                                processed_samples as f64
                                    / remaining_at_start as f64
                            } else {
                                1.0
                            };

                        let running_loss =
                            if processed_samples > 0 {
                                total_loss / processed_samples as f32
                            } else {
                                0.0
                            };

                        let running_ppl =
                            running_loss.exp();

                        let elapsed_secs =
                            elapsed.as_secs_f64();

                        let samples_per_sec =
                            if elapsed_secs > 0.0 {
                                processed_samples as f64
                                    / elapsed_secs
                            } else {
                                0.0
                            };

                        let eta =
                            if samples_per_sec > 0.0 {
                                let remaining_samples =
                                    data_len.saturating_sub(count);

                                Duration::from_secs_f64(
                                    remaining_samples as f64
                                        / samples_per_sec
                                )
                            } else {
                                Duration::ZERO
                            };

                        let samples_per_sec =
                            if elapsed_secs > 0.0 {
                                (count-epoch_start_count) as f64 / elapsed_secs
                            } else {
                                0.0
                            };
                        let avgbatchloss = batch_loss/frequency as f32;
                        println!(
                            "Epoch {} | Batch {}/{} | {:>6.2}% | \
                         Samples {}/{}\nAvgLoss = {:.6} | AvgPPL = {:.6} | Loss={:.6} | PPL= {:.6}\n\
                         {:.1} samples/s | Elapsed: {:.2?} | ETA: {:.2?}",
                            epoch,
                            batches_done,
                            total_batches,
                            progress * 100.0,
                            count,
                            data_len,
                            running_loss,
                            running_ppl,
                            avgbatchloss,
                            avgbatchloss.exp(),
                            samples_per_sec,
                            elapsed,
                            eta,
                        );
                    }
                }

                // -----------------------------------------------------
                // Ctrl+C
                //
                // The current batch has already:
                //   1. finished forward
                //   2. finished backward
                //   3. accumulated embedding gradients
                //   4. cleared the tape
                //   5. updated parameters
                //
                // So the checkpoint is safe to save here.
                // -----------------------------------------------------

                if interrupt_requested.load(Ordering::SeqCst) {
                    println!();
                    println!("Ctrl+C received. Current batch has finished.");

                    // ALWAYS checkpoint on Ctrl+C if a save function
                    // was supplied, regardless of checkpoint frequency.
                    make_checkpoint(
                        CheckpointKind::Batch,
                        epoch,
                        batches_done,
                        count,
                        &indices,
                        lr,
                        best_loss,
                        plateau_count,
                    );

                    loop {
                        print!("Exit training? [y/N]: ");
                        io::stdout().flush().ok();

                        let mut input = String::new();

                        match io::stdin().read_line(&mut input) {
                            Ok(_) => {
                                match input.trim().to_ascii_lowercase().as_str() {
                                    "y" | "yes" => {
                                        println!(
                                            "Training interrupted."
                                        );

                                        return TrainResult::Interrupted;
                                    }

                                    "" | "n" | "no" => {
                                        interrupt_requested.store(
                                            false,
                                            Ordering::SeqCst,
                                        );

                                        println!(
                                            "Resuming training..."
                                        );

                                        break;
                                    }

                                    _ => {
                                        println!(
                                            "Please enter Y or N."
                                        );
                                    }
                                }
                            }

                            Err(_) => {
                                println!(
                                    "Could not read input. \
                                 Exiting training."
                                );

                                return TrainResult::Interrupted;
                            }
                        }
                    }
                }
            }

            let samples = count.saturating_sub(epoch_start_count) as f32;

            if samples == 0.0 {
                continue;
            }

            let avg_loss = total_loss / samples;
            let perplexity = avg_loss.exp();

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

            let elapsed = now.elapsed();
            let grad_avg = grad_sum / samples;

            // ---------------------------------------------------------
            // Periodic epoch checkpoint
            //
            // Save AFTER LR/plateau state has been updated.
            // ---------------------------------------------------------

            let should_checkpoint = match checkpoint_frequency {
                CheckpointFrequency::EveryEpoch(n) => {
                    n > 0 && epoch % n == 0
                }

                _ => false,
            };

            if should_checkpoint {
                make_checkpoint(
                    CheckpointKind::Epoch,
                    epoch,
                    batches_done,
                    count,
                    &indices,
                    lr,
                    best_loss,
                    plateau_count,
                );
            }

            // ---------------------------------------------------------
            // Epoch statistics
            // ---------------------------------------------------------

            if update_frequency > 0 && epoch % update_frequency == 0 {
                println!(
                    "Epoch {} | Loss (CE) = {:.6} | \
                 Grad sum (avg per param) = {:.8} | \
                 PPL = {:.6} | Time elapsed: {:.2?}.",
                    epoch,
                    avg_loss,
                    grad_avg / param_count,
                    perplexity,
                    elapsed,
                );

                // -----------------------------------------------------
                // Optional timing breakdown
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                {
                    let elapsed_secs = elapsed.as_secs_f64();

                    let pct = |duration: Duration| {
                        if elapsed_secs > 0.0 {
                            duration.as_secs_f64()
                                / elapsed_secs
                                * 100.0
                        } else {
                            0.0
                        }
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
            }
            if grad_avg <= 1e-9 {
                println!("Early stopping, network will not learn anymore!");
                return TrainResult::Finished;
            }
        }
        TrainResult::Finished
    }


    pub fn train_lm_sumyu(
        &mut self,
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
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");

        let mut rng = rand::rng();

        let data_len = tokens.len().saturating_sub(context_len);

        if data_len == 0 {
            return TrainResult::Finished;
        }

        if self.batch_size > data_len {
            return TrainResult::Finished;
        }

        if self.batch_size == 0 {
            self.batch_size = data_len;
        }

        let mut lr = self.lr;
        let min_lr = 0.001 / self.batch_size as f32;

        let mut best_loss = f32::MAX;
        let mut plateau_count = 0;

        let input_size = context_len * embeddings.embedding_dim();

        let parameter_boundary = crate::tape_len();
        for epoch in 1..self.epochs + 1 {
            let now = Instant::now();

            let mut indices: Vec<usize> = (0..data_len).collect();
            indices.shuffle(&mut rng);

            let mut grad_sum = 0.0f32;
            let mut total_loss = 0.0f32;
            let mut count = 0usize;
            let mut batches_done = 0usize;

            while count < data_len {
                if self.max_batches_per_epoch != 0
                    && batches_done >= self.max_batches_per_epoch
                {
                    break;
                }

                let remaining = data_len - count;
                let current_batch = self.batch_size.min(remaining);

                // ---------------------------------------------------------
                // Build batch
                // ---------------------------------------------------------

                let mut batch_ids =
                    Vec::with_capacity(current_batch * context_len);

                let mut targets =
                    Vec::with_capacity(current_batch);

                for batch_index in 0..current_batch {
                    let sample = indices[count + batch_index];

                    batch_ids.extend_from_slice(
                        &tokens[sample..sample + context_len]
                    );

                    targets.push(tokens[sample + context_len]);
                }

                // ---------------------------------------------------------
                // Batched embedding lookup
                // ---------------------------------------------------------

                let input = embeddings.encode_batch(
                    &batch_ids,
                    current_batch,
                    context_len,
                );

                // ---------------------------------------------------------
                // Batched forward
                // ---------------------------------------------------------

                let forward = mlp.forward_batch(
                    &input,
                    current_batch,
                    input_size,
                );

                let output_size = forward.output_size;

                let mut output_grads =
                    vec![0.0f32; current_batch * output_size];

                // ---------------------------------------------------------
                // Batched loss
                // ---------------------------------------------------------

                let batch_loss = softmax_cross_entropy_batch(
                    &forward.output,
                    &targets,
                    &mut output_grads,
                    current_batch,
                    output_size,
                );

                total_loss += batch_loss;

                // ---------------------------------------------------------
                // Batched backward
                // ---------------------------------------------------------

                let input_grads = mlp.backward_batch(
                    &forward,
                    &output_grads,
                );

                // ---------------------------------------------------------
                // Embedding gradients
                // ---------------------------------------------------------

                embeddings.accumulate_batch_grads(
                    &batch_ids,
                    &input_grads,
                    current_batch,
                    context_len,
                );

                // This path doesn't build the normal autograd graph,
                // but preserve the tape boundary contract.
                crate::clear_tape_after(parameter_boundary);

                count += current_batch;
                batches_done += 1;

                // ---------------------------------------------------------
                // Update
                // ---------------------------------------------------------

                grad_sum += crate::zero_grad_and_update(
                    &params,
                    lr,
                );

                if !running.load(Ordering::SeqCst) {
                    return TrainResult::Interrupted;
                }
            }

            let samples = count as f32;

            if samples == 0.0 {
                continue;
            }

            let avg_loss = total_loss / samples;
            let perplexity = avg_loss.exp();
            let grad_avg = grad_sum / samples;

            // -------------------------------------------------------------
            // Learning-rate plateau detection
            // -------------------------------------------------------------

            if avg_loss < best_loss * 0.998 {
                best_loss = avg_loss;
                plateau_count = 0;
            } else {
                plateau_count += 1;
            }

            if plateau_count >= 10 {
                lr = (lr * 0.8).max(min_lr);
                plateau_count = 0;
            }

            let elapsed = now.elapsed();

            // -------------------------------------------------------------
            // Send UI/training information
            // -------------------------------------------------------------

            if grad_avg <= 1e-9 {
                tx.send(TrainInfo {
                    epoch,
                    loss: avg_loss,
                    perplexity,
                    time: elapsed,
                    done: true,
                }).unwrap();

                return TrainResult::Finished;
            }

            if tx.send(TrainInfo {
                epoch,
                loss: avg_loss,
                perplexity,
                time: elapsed,
                done: false,
            }).is_err() {
                return TrainResult::Interrupted;
            }
        }

        TrainResult::Finished
    }
}
