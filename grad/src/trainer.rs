use std::time::{Duration, Instant};
use rand::prelude::SliceRandom;
use crate::neuron::{MLP};
use crate::Tensor;
use crate::batched::softmax_cross_entropy_batch;
use crate::embeddings::Embeddings;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::io::{self, Write};

const PERMUTATION_SAMPLER_VERSION: u32 = 1;
const DEFAULT_SAMPLER_SEED: u64 = 1;

#[derive(Clone, Copy, Debug)]
pub struct PermutationSampler {
    seed: u64,
    epoch: u64,
    len: usize,
    bits: u32,
    mask: u64,
    key: u64,
}

impl PermutationSampler {
    pub const VERSION: u32 = PERMUTATION_SAMPLER_VERSION;
    pub const DEFAULT_SEED: u64 = DEFAULT_SAMPLER_SEED;

    pub fn new(seed: u64, epoch: usize, len: usize) -> Self {
        assert!(len > 0, "PermutationSampler length must be > 0");

        let bits = if len <= 1 {
            0
        } else {
            usize::BITS - (len - 1).leading_zeros()
        };

        let mask = Self::mask_for_bits(bits);

        let key = Self::derive_epoch_key(
            seed,
            epoch as u64,
        );

        Self {
            seed,
            epoch: epoch as u64,
            len,
            bits,
            mask,
            key,
        }
    }

    #[inline]
    fn mask_for_bits(bits: u32) -> u64 {
        match bits {
            0 => 0,
            64 => u64::MAX,
            _ => (1u64 << bits) - 1,
        }
    }

    #[inline]
    fn derive_epoch_key(seed: u64, epoch: u64) -> u64 {
        // Cheap SplitMix-style key derivation.
        // This does not need to be reversible; it only needs to
        // deterministically derive a different key for each epoch.
        let mut x =
            seed ^ epoch.wrapping_mul(0x9E3779B97F4A7C15);

        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30))
            .wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27))
            .wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    #[inline]
    fn permute_power_of_two(
        mut x: u64,
        bits: u32,
        mask: u64,
        key: u64,
    ) -> u64 {
        if bits == 0 {
            return 0;
        }

        // Every operation here is bijective modulo 2^bits:
        //
        //   x + c                  => bijective
        //   x ^= x >> n            => bijective
        //   x *= odd_constant      => bijective
        //   x ^= x << n            => bijective
        //
        // Mask after operations which can overflow the k-bit domain.

        x = x.wrapping_add(key) & mask;

        x ^= x >> 17;

        x = x
            .wrapping_mul(0xD6E8FEB86659FD93)
            & mask;

        x ^= (x << 13) & mask;

        x = x
            .wrapping_mul(0xA5A3564E27F8863D)
            & mask;

        x ^= x >> 11;

        x = x
            .wrapping_add(key.rotate_left(29))
            & mask;

        x ^= (x << 7) & mask;

        x
    }

    /// Returns the dataset index corresponding to `position`
    /// in this epoch's deterministic permutation.
    ///
    /// `position` must be in `0..self.len`.
    #[inline]
    pub fn index(&self, position: usize) -> usize {
        debug_assert!(
            position < self.len,
            "PermutationSampler position out of bounds"
        );

        if self.len == 1 {
            return 0;
        }

        let mut x = Self::permute_power_of_two(
            position as u64,
            self.bits,
            self.mask,
            self.key,
        );

        // Cycle walking converts the permutation of [0, 2^k)
        // into a permutation of [0, len).
        //
        // Starting from a valid position means we remain inside
        // that position's permutation cycle until we hit another
        // valid value.
        while x >= self.len as u64 {
            x = Self::permute_power_of_two(
                x,
                self.bits,
                self.mask,
                self.key,
            );
        }

        x as usize
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

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
    pub(crate) epoch: usize,
    pub(crate) batch: usize,
    pub(crate) sample: usize,

    pub(crate) sampler_seed: u64,
    pub(crate) sampler_version: u32,
    pub(crate) sampler_data_len: usize,

    pub(crate) lr: f32,
    pub(crate) best_loss: f32,
    pub(crate) plateau_count: usize,

    pub(crate) adam_first_moments: Vec<f32>,
    pub(crate) adam_second_moments: Vec<f32>,
    pub(crate) adam_step: u64,
}

pub struct CheckpointState {
    pub(crate) kind: CheckpointKind,

    pub(crate) epoch: usize,
    pub(crate) batch: usize,
    pub(crate) sample: usize,

    pub(crate) sampler_seed: u64,
    pub(crate) sampler_version: u32,
    pub(crate) sampler_data_len: usize,

    pub(crate) lr: f32,
    pub(crate) best_loss: f32,
    pub(crate) plateau_count: usize,

    pub(crate) adam_first_moments: Vec<f32>,
    pub(crate) adam_second_moments: Vec<f32>,
    pub(crate) adam_step: u64,
}

pub struct Trainer {
    lr: f32,
    epochs: usize,
    batch_size: usize,
    max_batches_per_epoch: usize,
}

impl Trainer {
    pub fn new(
        lr: f32,
        epochs: usize,
        batch_size: usize,
        max_batches_per_epoch: usize,
    ) -> Self {
        let batch_size =
            batch_size.max(1);

        Trainer {
            lr: lr / batch_size as f32,
            epochs,
            batch_size,
            max_batches_per_epoch,
        }
    }

    pub fn reinit_lr(
        &mut self,
        lr: f32,
    ) {
        self.lr =
            lr / self.batch_size.max(1) as f32;
    }

    pub fn reinit_epochs(&mut self, epochs: usize) {
        self.epochs = epochs
    }

    pub fn reinit_batch(
        &mut self,
        batch_size: usize,
    ) {
        let user_lr =
            self.lr
                * self.batch_size.max(1) as f32;

        self.batch_size =
            batch_size.max(1);

        self.lr =
            user_lr
                / self.batch_size as f32;
    }

    pub fn reinit_batch_per_epoch(&mut self, max_batches_per_epoch: usize) {
        self.max_batches_per_epoch = max_batches_per_epoch
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
        tokens: &[u16],
        context_len: usize,
        embeddings: &Embeddings,
        params: Vec<Tensor>,
        sampler_seed: u64,
    ) -> TrainResult {
        let param_count =
            params.len() as f32;

        // =============================================================
        // Textbook Adam
        //
        //     m_t = beta1 * m + (1-beta1) * g
        //     v_t = beta2 * v + (1-beta2) * g²
        //
        //     m_hat = m / (1-beta1^t)
        //     v_hat = v / (1-beta2^t)
        //
        //     param -= lr * m_hat / (sqrt(v_hat) + eps)
        //
        // =============================================================

        const ADAM_BETA1: f32 = 0.9;
        const ADAM_BETA2: f32 = 0.999;

        // =============================================================
        // Ctrl+C handling
        // =============================================================

        let interrupt_requested =
            Arc::new(
                AtomicBool::new(false)
            );

        let interrupt_flag =
            interrupt_requested.clone();

        ctrlc::set_handler(move || {
            let _ =
                interrupt_flag.compare_exchange(
                    false,
                    true,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
        })
            .expect("Error setting Ctrl+C handler");

        // =============================================================
        // Dataset validation
        // =============================================================

        let data_len =
            tokens
                .len()
                .saturating_sub(context_len);

        if data_len == 0 {
            println!(
                "Dataset is too small for the selected context length!"
            );

            return TrainResult::Finished;
        }

        if self.batch_size > data_len {
            println!(
                "Batch size too high! Quitting..."
            );

            return TrainResult::Finished;
        }

        if self.batch_size == 0 {
            self.batch_size = data_len;
        }

        // =============================================================
        // Basic training state
        // =============================================================

        let parameter_boundary =
            crate::tape_len();

        // IMPORTANT:
        //
        // Adam uses the real user-supplied LR directly.
        //
        // Trainer::new() should therefore store `lr` directly rather
        // than dividing it by batch size.
        let mut lr =
            self.lr * self.batch_size as f32;

        let mut best_loss =
            f32::MAX;

        let mut plateau_count =
            0usize;

        // =============================================================
        // MLP input size
        // =============================================================

        let input_size =
            context_len
                * embeddings.embedding_dim();

        // =============================================================
        // Parameter boundary
        // =============================================================

        let mlp_parameter_count =
            mlp.parameter_count();

        assert!(
            params.len() >= mlp_parameter_count,
            "Parameter vector is smaller than MLP parameter count"
        );

        let mlp_parameter_start =
            params.len()
                - mlp_parameter_count;

        // `mlp_parameter_start` is retained because it identifies the
        // embedding/MLP boundary for diagnostics and gradient handling.
        //
        // Adam itself does NOT use separate optimizer logic for them.

        // =============================================================
        // Adam state
        // =============================================================

        let mut adam_first_moments =
            vec![0.0f32; params.len()];

        let mut adam_second_moments =
            vec![0.0f32; params.len()];

        let mut adam_step =
            0u64;

        // Keep beta powers incrementally so we do NOT calculate powi()
        // every training batch.
        //
        // At step t:
        //
        //     beta1_power = beta1^t
        //     beta2_power = beta2^t
        //
        let mut beta1_power =
            1.0f32;

        let mut beta2_power =
            1.0f32;

        // =============================================================
        // Resume state
        // =============================================================

        let (
            start_epoch,
            mut resume_state,
        ) =
            match resume.take() {
                Some(state) => {
                    println!(
                        "Resuming from epoch {}, batch {} (sample {}/{}).",
                        state.epoch,
                        state.batch,
                        state.sample,
                        data_len,
                    );

                    lr =
                        state.lr;

                    best_loss =
                        state.best_loss;

                    plateau_count =
                        state.plateau_count;

                    // -------------------------------------------------
                    // Restore Adam state
                    // -------------------------------------------------

                    if state.adam_first_moments.len()
                        == params.len()
                        && state.adam_second_moments.len()
                        == params.len()
                    {
                        adam_first_moments =
                            state.adam_first_moments.clone();

                        adam_second_moments =
                            state.adam_second_moments.clone();

                        adam_step =
                            state.adam_step;

                        println!(
                            "Restored Adam state: {} parameters, step {}.",
                            adam_first_moments.len(),
                            adam_step,
                        );

                        // Reconstruct beta powers once at resume.
                        //
                        // They quickly underflow to zero at large t,
                        // which is harmless because the bias correction
                        // approaches 1.
                        beta1_power =
                            ADAM_BETA1.powi(
                                adam_step as i32
                            );

                        beta2_power =
                            ADAM_BETA2.powi(
                                adam_step as i32
                            );
                    } else {
                        println!(
                            "Checkpoint has incompatible Adam state; \
         starting Adam moments from zero."
                        );

                        adam_first_moments.fill(0.0);
                        adam_second_moments.fill(0.0);

                        adam_step =
                            0;

                        beta1_power =
                            1.0;

                        beta2_power =
                            1.0;
                    }

                    if state.sample >= data_len {
                        (
                            state.epoch + 1,
                            None,
                        )
                    } else {
                        (
                            state.epoch,
                            Some(state),
                        )
                    }
                }

                None => {
                    (
                        1,
                        None,
                    )
                }
            };

        // =============================================================
        // Sampler state
        // =============================================================

        let sampler_seed =
            match resume_state.as_ref() {
                Some(state) => {
                    assert_eq!(
                        state.sampler_version,
                        PermutationSampler::VERSION,
                        "Unsupported permutation sampler version in checkpoint"
                    );

                    assert_eq!(
                        state.sampler_data_len,
                        data_len,
                        "Dataset length differs from checkpoint. \
                 Exact deterministic resume is impossible."
                    );

                    state.sampler_seed
                }

                None => sampler_seed,
            };

        // =============================================================
        // Checkpoint helper
        // =============================================================

        let mut make_checkpoint =
            |kind: CheckpointKind,
             epoch: usize,
             batch: usize,
             sample: usize,
             sampler_seed: u64,
             sampler_data_len: usize,
             lr: f32,
             best_loss: f32,
             plateau_count: usize,
             adam_first_moments: &[f32],
             adam_second_moments: &[f32],
             adam_step: u64| {
                if let Some(savefn) =
                    savefn.as_mut()
                {
                    savefn(
                        CheckpointState {
                            kind,
                            epoch,
                            batch,
                            sample,

                            sampler_seed,
                            sampler_version:
                            PermutationSampler::VERSION,
                            sampler_data_len,

                            lr,
                            best_loss,
                            plateau_count,

                            adam_first_moments:
                            adam_first_moments.to_vec(),

                            adam_second_moments:
                            adam_second_moments.to_vec(),

                            adam_step,
                        },
                        mlp,
                        embeddings,
                    );
                }
            };

        // =============================================================
        // Reusable batch buffers
        // =============================================================

        let mut batch_ids =
            Vec::with_capacity(
                self.batch_size
                    * context_len
            );

        let mut targets =
            Vec::with_capacity(
                self.batch_size
            );

        let mut batch_input =
            Vec::with_capacity(
                self.batch_size
                    * context_len
                    * embeddings.embedding_dim()
            );

        let mut output_grads =
            Vec::<f32>::new();

        // =============================================================
        // Process epochs
        // =============================================================

        for epoch in
            start_epoch..self.epochs + 1
        {
            let now =
                Instant::now();

            let mut count: usize;
            let mut batches_done: usize;

            let mut grad_sum =
                0.0f32;

            let mut total_loss =
                0.0f32;

            let sampler =
                PermutationSampler::new(
                    sampler_seed,
                    epoch,
                    data_len,
                );

            // ---------------------------------------------------------
            // Restore position
            // ---------------------------------------------------------

            if let Some(state) =
                resume_state.take()
            {
                debug_assert_eq!(
                    state.epoch,
                    epoch
                );

                count =
                    state.sample;

                batches_done =
                    state.batch;

                println!(
                    "Continuing epoch {} from batch {}.",
                    epoch,
                    batches_done,
                );
            } else {
                count =
                    0;

                batches_done =
                    0;
            }

            let epoch_start_count =
                count;

            let total_batches =
                (
                    data_len
                        + self.batch_size
                        - 1
                )
                    / self.batch_size;

            // ---------------------------------------------------------
            // Timing
            // ---------------------------------------------------------

            #[cfg(feature = "timing")]
            let mut encode_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut forward_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut loss_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut backward_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut embedding_grad_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut clear_tape_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut update_time =
                Duration::ZERO;

            // =========================================================
            // Batches
            // =========================================================

            while count < data_len {
                if self.max_batches_per_epoch != 0
                    && batches_done
                    >= self.max_batches_per_epoch
                {
                    break;
                }

                let remaining =
                    data_len - count;

                let current_batch =
                    self.batch_size
                        .min(remaining);

                // -----------------------------------------------------
                // Build token batch
                // -----------------------------------------------------

                batch_ids.clear();
                targets.clear();

                for batch_index in
                    0..current_batch
                {
                    let sample =
                        sampler.index(
                            count
                                + batch_index
                        );

                    batch_ids.extend_from_slice(
                        &tokens[
                            sample
                                ..sample + context_len
                            ]
                    );

                    targets.push(
                        tokens[
                            sample
                                + context_len
                            ]
                    );
                }

                // -----------------------------------------------------
                // Embedding lookup
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                embeddings.encode_batch_into(
                    &batch_ids,
                    current_batch,
                    context_len,
                    &mut batch_input,
                );

                #[cfg(feature = "timing")]
                {
                    encode_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Forward
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                let forward =
                    mlp.forward_batch(
                        &batch_input,
                        current_batch,
                        input_size,
                    );

                #[cfg(feature = "timing")]
                {
                    forward_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Softmax cross entropy
                //
                // softmax_cross_entropy_batch() returns SUM CE.
                // -----------------------------------------------------

                let output_size =
                    forward.output_size;

                let grad_len =
                    current_batch
                        * output_size;

                if output_grads.len()
                    != grad_len
                {
                    output_grads.resize(
                        grad_len,
                        0.0,
                    );
                } else {
                    output_grads.fill(
                        0.0
                    );
                }

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                let batch_loss =
                    softmax_cross_entropy_batch(
                        &forward.output,
                        &*targets,
                        &mut output_grads,
                        current_batch,
                        output_size,
                    );

                if !batch_loss.is_finite() {
                    println!(
                        "Non-finite batch loss detected. \
     Stopping training before optimizer update."
                    );

                    crate::clear_tape_after(
                        parameter_boundary
                    );

                    return TrainResult::Finished;
                }

                total_loss +=
                    batch_loss;

                #[cfg(feature = "timing")]
                {
                    loss_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Backward
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                let input_grads =
                    mlp.backward_batch(
                        &forward,
                        &output_grads,
                    );

                #[cfg(feature = "timing")]
                {
                    backward_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Embedding gradients
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                embeddings.accumulate_batch_grads(
                    &batch_ids,
                    &input_grads,
                    current_batch,
                    context_len,
                );

                #[cfg(feature = "timing")]
                {
                    embedding_grad_time +=
                        timer.elapsed();
                }

                // =====================================================
                // GLOBAL GRADIENT NORM
                //
                // Gradients are NOT clipped element-by-element.
                //
                // If the total norm is <= MAX_GRAD_NORM:
                //     gradient is completely unchanged.
                //
                // If the norm exceeds the limit:
                //     the ENTIRE gradient vector is uniformly scaled.
                //
                // This preserves the gradient direction.
                // =====================================================

                const MAX_GRAD_NORM: f64 = 1.0;

                // The backward pass accumulates SUM gradients over the batch.
                // Adam should operate on the MEAN batch gradient.
                let batch_normalization =
                    1.0f32 / current_batch.max(1) as f32;

                // -----------------------------------------------------
                // MLP gradient norm
                // -----------------------------------------------------

                let mut mlp_grad_sq_sum =
                    0.0f64;

                for param_index
                in mlp_parameter_start..params.len()
                {
                    let raw_g =
                        params[param_index].grad();

                    if !raw_g.is_finite() {
                        println!(
                            "Non-finite MLP gradient detected before optimizer update. \
Stopping training."
                        );

                        crate::clear_tape_after(
                            parameter_boundary
                        );

                        return TrainResult::Finished;
                    }

                    // Measure the norm of the MEAN gradient,
                    // not the summed batch gradient.
                    let g =
                        raw_g * batch_normalization;

                    let gf =
                        g as f64;

                    mlp_grad_sq_sum +=
                        gf * gf;
                }

                let mlp_grad_norm =
                    mlp_grad_sq_sum.sqrt();

                let mlp_grad_scale =
                    if mlp_grad_norm > MAX_GRAD_NORM {
                        (
                            MAX_GRAD_NORM
                                / mlp_grad_norm
                        ) as f32
                    } else {
                        1.0
                    };

                // -----------------------------------------------------
                // Embedding gradient norm
                // -----------------------------------------------------

                let mut embedding_grad_sq_sum =
                    0.0f64;

                for param_index
                in 0..mlp_parameter_start
                {
                    let raw_g =
                        params[param_index].grad();

                    if !raw_g.is_finite() {
                        println!(
                            "Non-finite embedding gradient detected before optimizer update. \
Stopping training."
                        );

                        crate::clear_tape_after(
                            parameter_boundary
                        );

                        return TrainResult::Finished;
                    }

                    // Measure the norm of the MEAN gradient,
                    // not the summed batch gradient.
                    let g =
                        raw_g * batch_normalization;

                    let gf =
                        g as f64;

                    embedding_grad_sq_sum +=
                        gf * gf;
                }

                let embedding_grad_norm =
                    embedding_grad_sq_sum.sqrt();

                let embedding_grad_scale =
                    if embedding_grad_norm > MAX_GRAD_NORM {
                        (
                            MAX_GRAD_NORM
                                / embedding_grad_norm
                        ) as f32
                    } else {
                        1.0
                    };

                // =====================================================
                // Adam step
                // =====================================================

                adam_step +=
                    1;

                beta1_power *=
                    ADAM_BETA1;

                beta2_power *=
                    ADAM_BETA2;

                let beta1_correction_inv =
                    1.0f32
                        / (1.0f32
                        - beta1_power);

                let beta2_correction_inv =
                    1.0f32
                        / (1.0f32
                        - beta2_power);

                // -----------------------------------------------------
                // Clear backward tape before parameter update.
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                crate::clear_tape_after(
                    parameter_boundary
                );

                #[cfg(feature = "timing")]
                {
                    clear_tape_time +=
                        timer.elapsed();
                }

                count +=
                    current_batch;

                batches_done +=
                    1;

                // -----------------------------------------------------
                // Parameter update
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                grad_sum +=
                    crate::zero_grad_and_update_adam(
                        &params,
                        lr,
                        embeddings,
                        &mut adam_first_moments,
                        &mut adam_second_moments,
                        beta1_correction_inv,
                        beta2_correction_inv,
                        mlp_parameter_start,
                        mlp_grad_scale,
                        embedding_grad_scale,
                        batch_normalization,
                    );

                #[cfg(feature = "timing")]
                {
                    update_time +=
                        timer.elapsed();
                }

                // =====================================================
                // Batch checkpoint
                // =====================================================

                let should_checkpoint =
                    match checkpoint_frequency {
                        CheckpointFrequency::EveryBatch(n) => {
                            n > 0
                                && batches_done
                                % n
                                == 0
                        }

                        _ => false,
                    };

                if should_checkpoint {
                    make_checkpoint(
                        CheckpointKind::Batch,
                        epoch,
                        batches_done,
                        count,
                        sampler_seed,
                        data_len,
                        lr,
                        best_loss,
                        plateau_count,
                        &adam_first_moments,
                        &adam_second_moments,
                        adam_step,
                    );
                }

                // =====================================================
                // Progress
                // =====================================================

                if let Some(
                    frequency
                ) =
                    batch_update_frequency
                {
                    if frequency > 0
                        && batches_done
                        % frequency
                        == 0
                    {
                        let elapsed =
                            now.elapsed();

                        let processed_samples =
                            count.saturating_sub(
                                epoch_start_count
                            );

                        let remaining_at_start =
                            data_len.saturating_sub(
                                epoch_start_count
                            );

                        let progress =
                            if remaining_at_start
                                > 0
                            {
                                processed_samples
                                    as f64
                                    / remaining_at_start
                                    as f64
                            } else {
                                1.0
                            };

                        let running_loss =
                            if processed_samples
                                > 0
                            {
                                total_loss
                                    / processed_samples
                                    as f32
                            } else {
                                0.0
                            };

                        let running_ppl =
                            running_loss.exp();

                        let elapsed_secs =
                            elapsed.as_secs_f64();

                        let samples_per_sec =
                            if elapsed_secs
                                > 0.0
                            {
                                (
                                    count
                                        - epoch_start_count
                                ) as f64
                                    / elapsed_secs
                            } else {
                                0.0
                            };

                        let eta =
                            if samples_per_sec
                                > 0.0
                            {
                                let remaining_samples =
                                    data_len
                                        .saturating_sub(
                                            count
                                        );

                                Duration::from_secs_f64(
                                    remaining_samples
                                        as f64
                                        / samples_per_sec
                                )
                            } else {
                                Duration::ZERO
                            };

                        // batch_loss is SUM CE, so divide by batch size
                        // for the per-sample batch loss.
                        let avgbatchloss =
                            batch_loss
                                / current_batch
                                as f32;

                        println!(
                            "Epoch {} | Batch {}/{} | {:>6.2}% | \
                     Samples {}/{} | AvgLoss = {:.6} | \
                     AvgPPL = {:.6}\n\
                     Loss={:.6} | PPL= {:.6} | \
                     {:.1} samples/s | Elapsed: {:.2?} | \
                     ETA: {:.2?}",
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

                // =====================================================
                // Ctrl+C
                // =====================================================

                if interrupt_requested
                    .load(Ordering::SeqCst)
                {
                    println!();
                    println!(
                        "Ctrl+C received. Current batch has finished."
                    );

                    make_checkpoint(
                        CheckpointKind::Batch,
                        epoch,
                        batches_done,
                        count,
                        sampler_seed,
                        data_len,
                        lr,
                        best_loss,
                        plateau_count,
                        &adam_first_moments,
                        &adam_second_moments,
                        adam_step,
                    );

                    loop {
                        print!(
                            "Exit training? [y/N]: "
                        );

                        io::stdout()
                            .flush()
                            .ok();

                        let mut input =
                            String::new();

                        match io::stdin()
                            .read_line(
                                &mut input
                            )
                        {
                            Ok(_) => {
                                match input
                                    .trim()
                                    .to_ascii_lowercase()
                                    .as_str()
                                {
                                    "y" | "yes" => {
                                        println!(
                                            "Training interrupted."
                                        );

                                        return TrainResult::Interrupted;
                                    }

                                    "" | "n" | "no" => {
                                        interrupt_requested
                                            .store(
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

            // =========================================================
            // Epoch statistics
            // =========================================================

            let samples =
                count.saturating_sub(
                    epoch_start_count
                ) as f32;

            if samples == 0.0 {
                continue;
            }

            // softmax_cross_entropy_batch returns SUM CE, therefore this
            // is the correct per-sample CE.
            let avg_loss =
                total_loss
                    / samples;

            let perplexity =
                avg_loss.exp();

            // =========================================================
            // Track best loss
            //
            // Kept for checkpoint/output compatibility.
            //
            // Adam itself does not perform the old custom layerwise
            // LR adaptation.
            // =========================================================

            if avg_loss
                < best_loss
            {
                best_loss =
                    avg_loss;

                plateau_count =
                    0;
            } else {
                plateau_count +=
                    1;
            }

            // =========================================================
            // Epoch checkpoint
            // =========================================================

            let should_checkpoint =
                match checkpoint_frequency {
                    CheckpointFrequency::EveryEpoch(n) => {
                        n > 0
                            && epoch % n
                            == 0
                    }

                    _ => false,
                };

            if should_checkpoint {
                make_checkpoint(
                    CheckpointKind::Epoch,
                    epoch,
                    batches_done,
                    count,
                    sampler_seed,
                    data_len,
                    lr,
                    best_loss,
                    plateau_count,
                    &adam_first_moments,
                    &adam_second_moments,
                    adam_step,
                );
            }

            // =========================================================
            // Epoch statistics
            // =========================================================

            let elapsed =
                now.elapsed();

            let grad_avg =
                grad_sum
                    / samples;

            if update_frequency > 0
                && epoch % update_frequency
                == 0
            {
                println!(
                    "Epoch {} | Loss (CE) = {:.6} | \
             Grad sum (avg per param) = {:.8} | \
             PPL = {:.6} | Time elapsed: {:.2?}.",
                    epoch,
                    avg_loss,
                    grad_avg
                        / param_count,
                    perplexity,
                    elapsed,
                );

                #[cfg(feature = "timing")]
                {
                    let elapsed_secs =
                        elapsed.as_secs_f64();

                    let pct =
                        |duration: Duration| {
                            if elapsed_secs
                                > 0.0
                            {
                                duration
                                    .as_secs_f64()
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
                        "  Update:           {:>10.3?} ({:>6.2}%)",
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
}
