use std::time::{Duration, Instant};
use rand::prelude::SliceRandom;
use crate::neuron::{LayerSpec, MLP};
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
    pub epoch: usize,
    pub batch: usize,
    pub sample: usize,

    pub sampler_seed: u64,
    pub sampler_version: u32,
    pub sampler_data_len: usize,

    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,

    // Layerwise adaptive LR optimizer state.
    pub layer_second_moments: Vec<f32>,
    pub layer_adaptive_step: u64,
}

pub struct CheckpointState {
    pub kind: CheckpointKind,
    pub epoch: usize,
    pub batch: usize,
    pub sample: usize,

    pub sampler_seed: u64,
    pub sampler_version: u32,
    pub sampler_data_len: usize,

    pub lr: f32,
    pub best_loss: f32,
    pub plateau_count: usize,

    // Layerwise adaptive LR optimizer state.
    pub layer_second_moments: Vec<f32>,
    pub layer_adaptive_step: u64,
}

const LAYER_ADAPT_MIN_SCALE: f32 = 0.25;
const LAYER_ADAPT_MAX_SCALE: f32 = 4.0;
const LAYER_ADAPT_BETA: f32 = 0.99;
const LAYER_ADAPT_EPS: f32 = 1e-8;

/// Returns the number of scalar parameters owned by this layer.
///
/// The parameter ordering exactly matches Layer::parameters():
/// weights first, then biases, recursively for Residual layers.
fn layer_parameter_count(
    spec: &LayerSpec,
    current_size: usize,
    embeddings_vocab_size: usize,
) -> (usize, usize) {
    match spec {
        LayerSpec::Dense { output_size, .. } => {
            let count =
                current_size * output_size
                    + output_size;

            (count, *output_size)
        }

        LayerSpec::Conv1D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            causal,
            ..
        } => {
            assert_eq!(
                current_size % in_channels,
                0
            );

            let input_length =
                current_size / in_channels;

            let output_length =
                if *causal {
                    (input_length - 1) / stride + 1
                } else {
                    assert!(
                        input_length + 2 * padding
                            >= *kernel_size
                    );

                    (
                        input_length
                            + 2 * padding
                            - kernel_size
                    ) / stride
                        + 1
                };

            let count =
                out_channels
                    * in_channels
                    * kernel_size
                    + out_channels;

            let output_size =
                output_length * out_channels;

            (count, output_size)
        }

        LayerSpec::Residual { layers } => {
            let mut size = current_size;
            let mut count = 0;

            for inner in layers {
                let (inner_count, inner_size) =
                    layer_parameter_count(
                        inner,
                        size,
                        embeddings_vocab_size,
                    );

                count += inner_count;
                size = inner_size;
            }

            assert_eq!(
                size,
                current_size,
                "Residual block changed tensor size"
            );

            (count, size)
        }

        LayerSpec::DepthwiseConv1D {
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            ..
        } => {
            assert_eq!(
                current_size % in_channels,
                0
            );

            let input_length =
                current_size / in_channels;

            let output_length =
                if *causal {
                    (input_length - 1) / stride + 1
                } else {
                    assert!(
                        input_length + 2 * padding
                            >= *kernel_size
                    );

                    (
                        input_length
                            + 2 * padding
                            - kernel_size
                    ) / stride
                        + 1
                };

            let count =
                in_channels * kernel_size
                    + in_channels;

            (
                count,
                output_length * in_channels,
            )
        }

        LayerSpec::GroupedConv1D {
            in_channels,
            out_channels,
            groups,
            kernel_size,
            stride,
            padding,
            causal,
            ..
        } => {
            assert_eq!(
                in_channels % groups,
                0
            );

            assert_eq!(
                out_channels % groups,
                0
            );

            assert_eq!(
                current_size % in_channels,
                0
            );

            let input_length =
                current_size / in_channels;

            let output_length =
                if *causal {
                    (input_length - 1) / stride + 1
                } else {
                    assert!(
                        input_length + 2 * padding
                            >= *kernel_size
                    );

                    (
                        input_length
                            + 2 * padding
                            - kernel_size
                    ) / stride
                        + 1
                };

            let group_in =
                in_channels / groups;

            let count =
                out_channels
                    * group_in
                    * kernel_size
                    + out_channels;

            (
                count,
                output_length * out_channels,
            )
        }

        LayerSpec::LowRankPointwise {
            in_channels,
            rank,
            out_channels,
            ..
        } => {
            assert_eq!(
                current_size % in_channels,
                0
            );

            let positions =
                current_size / in_channels;

            let count =
                rank * in_channels
                    + rank
                    + out_channels * rank
                    + out_channels;

            (
                count,
                positions * out_channels,
            )
        }

        LayerSpec::ChannelScale { channels } => {
            assert_eq!(
                current_size % channels,
                0
            );

            (
                channels * 2,
                current_size,
            )
        }

        LayerSpec::WeightTying => {
            (
                0,
                embeddings_vocab_size,
            )
        }

        LayerSpec::LayerNorm { channels, .. } => {
            assert_eq!(
                current_size % channels,
                0
            );

            (
                channels * 2,
                current_size,
            )
        }

        LayerSpec::GlobalMixer {
            channels,
            global_dim,
        } => {
            assert_eq!(
                current_size % channels,
                0
            );

            let count =
                2 * (
                    channels * global_dim
                        + global_dim
                );

            (
                count,
                current_size,
            )
        }
    }
}

/// Builds parameter ranges for actual parameter-bearing layers.
///
/// Residual itself is not an optimizer group; its inner layers are.
/// This gives CNN / mixer / norm / pointwise components independent
/// adaptive learning rates.
fn build_layer_parameter_ranges(
    specs: &[LayerSpec],
    mut current_size: usize,
    embeddings_vocab_size: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut offset = 0usize;

    fn recurse(
        specs: &[LayerSpec],
        current_size: &mut usize,
        offset: &mut usize,
        embeddings_vocab_size: usize,
        ranges: &mut Vec<(usize, usize)>,
    ) {
        for spec in specs {
            match spec {
                LayerSpec::Residual { layers } => {
                    recurse(
                        layers,
                        current_size,
                        offset,
                        embeddings_vocab_size,
                        ranges,
                    );
                }

                _ => {
                    let (count, output_size) =
                        layer_parameter_count(
                            spec,
                            *current_size,
                            embeddings_vocab_size,
                        );

                    if count > 0 {
                        ranges.push((
                            *offset,
                            *offset + count,
                        ));
                    }

                    *offset += count;
                    *current_size = output_size;
                }
            }
        }
    }

    recurse(
        specs,
        &mut current_size,
        &mut offset,
        embeddings_vocab_size,
        &mut ranges,
    );

    ranges
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

        // ---------------------------------------------------------
        // Ctrl+C handling
        // ---------------------------------------------------------

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

        // ---------------------------------------------------------
        // Dataset validation
        // ---------------------------------------------------------

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

        // ---------------------------------------------------------
        // Basic training state
        // ---------------------------------------------------------

        let parameter_boundary =
            crate::tape_len();

        let mut lr =
            self.lr;

        let min_lr =
            0.001
                / self.batch_size as f32;

        let mut best_loss =
            f32::MAX;

        let mut plateau_count =
            0usize;

        // ---------------------------------------------------------
        // MLP input size
        // ---------------------------------------------------------

        let input_size =
            context_len
                * embeddings.embedding_dim();

        // ---------------------------------------------------------
        // Layerwise adaptive LR
        //
        // params is expected to contain:
        //
        //   [embedding parameters][MLP parameters]
        //
        // which matches the existing embedding updater convention.
        // ---------------------------------------------------------

        let mlp_parameter_count =
            mlp.parameter_count();

        assert!(
            params.len() >= mlp_parameter_count,
            "Parameter vector is smaller than MLP parameter count"
        );

        let mlp_parameter_start =
            params.len()
                - mlp_parameter_count;

        let layer_ranges =
            build_layer_parameter_ranges(
                &mlp.layer_specs(),
                input_size,
                embeddings.vocab_size(),
            );

        let expected_mlp_parameter_count =
            layer_ranges
                .iter()
                .map(|(start, end)| end - start)
                .sum::<usize>();

        assert_eq!(
            expected_mlp_parameter_count,
            mlp_parameter_count,
            "Layer parameter accounting does not match MLP::parameters()"
        );

        // One second-moment value per actual parameter-bearing layer.
        let mut layer_second_moments =
            vec![
                0.0f32;
                layer_ranges.len()
            ];

        // Number of adaptive updates performed.
        let mut layer_adaptive_step =
            0u64;

        // ---------------------------------------------------------
        // Resume state
        //
        // Restore optimizer state BEFORE entering the epoch loop.
        // ---------------------------------------------------------

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
                    // Restore layerwise optimizer state.
                    // -------------------------------------------------

                    if state.layer_second_moments.is_empty() {
                        println!(
                            "Checkpoint has no layerwise optimizer state; \
                     starting adaptive LR state from zero."
                        );

                        layer_second_moments.fill(0.0);

                        layer_adaptive_step = 0;
                    } else {
                        assert_eq!(
                            state.layer_second_moments.len(),
                            layer_ranges.len(),
                            "Checkpoint layer optimizer state \
                     does not match current MLP architecture"
                        );

                        layer_second_moments =
                            state.layer_second_moments.clone();

                        layer_adaptive_step =
                            state.layer_adaptive_step;

                        println!(
                            "Restored layerwise optimizer state: \
                     {} layers, step {}.",
                            layer_second_moments.len(),
                            layer_adaptive_step,
                        );
                    }

                    // -------------------------------------------------
                    // If sample == data_len, the checkpoint represents
                    // a completed epoch. Continue with the next epoch.
                    // Otherwise resume inside the same epoch.
                    // -------------------------------------------------

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

        // ---------------------------------------------------------
        // Sampler state
        // ---------------------------------------------------------

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

        // ---------------------------------------------------------
        // Checkpoint helper
        // ---------------------------------------------------------

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
             layer_second_moments: &[f32],
             layer_adaptive_step: u64| {
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

                            layer_second_moments:
                            layer_second_moments
                                .to_vec(),

                            layer_adaptive_step,
                        },
                        mlp,
                        embeddings,
                    );
                }
            };

        // ---------------------------------------------------------
        // Reusable batch buffers
        // ---------------------------------------------------------

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

        // ---------------------------------------------------------
        // Process epochs
        // ---------------------------------------------------------

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

            // -----------------------------------------------------
            // Restore position within resumed epoch.
            // -----------------------------------------------------

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
            // Optional timing
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

            // ---------------------------------------------------------
            // Process batches
            // ---------------------------------------------------------

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
                                ..sample
                                + context_len
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
                // -----------------------------------------------------

                let output_size =
                    forward.output_size;

                let grad_len =
                    current_batch
                        * output_size;

                if output_grads.len()
                    != grad_len
                {
                    output_grads
                        .resize(
                            grad_len,
                            0.0,
                        );
                } else {
                    output_grads.fill(0.0);
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

                // -----------------------------------------------------
                // Layerwise adaptive LR
                //
                // Update one EMA value per layer.
                //
                // IMPORTANT:
                // This happens before zero_grad_and_update_embeddings...
                // clears the gradients.
                // -----------------------------------------------------

                layer_adaptive_step +=
                    1;

                let beta =
                    LAYER_ADAPT_BETA;

                let correction =
                    1.0f32
                        - beta.powi(
                        layer_adaptive_step
                            as i32
                    );

                let mut layer_rms =
                    vec![
                        0.0f32;
                        layer_ranges.len()
                    ];

                for (
                    layer_index,
                    &(start, end),
                ) in layer_ranges
                    .iter()
                    .enumerate()
                {
                    let mut sum_sq =
                        0.0f64;

                    let count_params =
                        end - start;

                    for local_index in
                        0..count_params
                    {
                        let param_index =
                            mlp_parameter_start
                                + start
                                + local_index;

                        let g =
                            params[
                                param_index
                                ]
                                .grad()
                                as f64;

                        sum_sq +=
                            g * g;
                    }

                    let mean_sq =
                        if count_params > 0 {
                            (
                                sum_sq
                                    / count_params
                                    as f64
                            ) as f32
                        } else {
                            0.0
                        };

                    layer_second_moments[
                        layer_index
                        ] =
                        beta
                            * layer_second_moments[
                            layer_index
                            ]
                            + (
                            1.0
                                - beta
                        )
                            * mean_sq;

                    let bias_corrected =
                        if correction
                            > 1e-12
                        {
                            layer_second_moments[
                                layer_index
                                ]
                                / correction
                        } else {
                            layer_second_moments[
                                layer_index
                                ]
                        };

                    layer_rms[
                        layer_index
                        ] =
                        bias_corrected
                            .max(0.0)
                            .sqrt();
                }

                // -----------------------------------------------------
                // Reference RMS
                //
                // Only active layers participate. A layer with exactly
                // zero gradient should not drag every other layer's LR.
                // -----------------------------------------------------

                let mut rms_sum =
                    0.0f32;

                let mut active_layers =
                    0usize;

                for &rms in
                    &layer_rms
                {
                    if rms
                        > LAYER_ADAPT_EPS
                    {
                        rms_sum +=
                            rms;

                        active_layers +=
                            1;
                    }
                }

                let reference_rms =
                    if active_layers > 0 {
                        rms_sum
                            / active_layers
                            as f32
                    } else {
                        1.0
                    };

                // -----------------------------------------------------
                // Per-parameter LR scales
                //
                // Embeddings remain at 1.0x.
                // MLP parameter groups get their layer scale.
                // -----------------------------------------------------

                let mut lr_scales =
                    vec![
                        1.0f32;
                        params.len()
                    ];

                for (
                    layer_index,
                    &(start, end),
                ) in layer_ranges
                    .iter()
                    .enumerate()
                {
                    let rms =
                        layer_rms[
                            layer_index
                            ];

                    let scale =
                        if rms
                            > LAYER_ADAPT_EPS
                        {
                            (
                                reference_rms
                                    / (
                                    rms
                                        + LAYER_ADAPT_EPS
                                )
                            )
                                .clamp(
                                    LAYER_ADAPT_MIN_SCALE,
                                    LAYER_ADAPT_MAX_SCALE,
                                )
                        } else {
                            1.0
                        };

                    let absolute_start =
                        mlp_parameter_start
                            + start;

                    let absolute_end =
                        mlp_parameter_start
                            + end;

                    for index in
                        absolute_start
                            ..absolute_end
                    {
                        lr_scales[index] =
                            scale;
                    }
                }

                // -----------------------------------------------------
                // Clear temporary tape
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
                // Update parameters
                //
                // Existing global gradient clipping is still performed
                // inside this function.
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                grad_sum +=
                    crate::zero_grad_and_update_embeddings_layerwise(
                        &params,
                        lr,
                        embeddings,
                        &lr_scales,
                    );

                #[cfg(feature = "timing")]
                {
                    update_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Periodic batch checkpoint
                //
                // Save AFTER:
                //   - optimizer state update
                //   - parameter update
                // -----------------------------------------------------

                let should_checkpoint =
                    match checkpoint_frequency {
                        CheckpointFrequency::EveryBatch(
                            n
                        ) => {
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
                        &layer_second_moments,
                        layer_adaptive_step,
                    );
                }

                // -----------------------------------------------------
                // Intra-epoch progress
                // -----------------------------------------------------

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
                            count
                                .saturating_sub(
                                    epoch_start_count
                                );

                        let remaining_at_start =
                            data_len
                                .saturating_sub(
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
                            elapsed
                                .as_secs_f64();

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

                // -----------------------------------------------------
                // Ctrl+C
                //
                // The current batch has already:
                //   1. finished forward
                //   2. finished backward
                //   3. accumulated embedding gradients
                //   4. updated adaptive optimizer state
                //   5. cleared the tape
                //   6. updated parameters
                //
                // Therefore the checkpoint is consistent.
                // -----------------------------------------------------

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
                        &layer_second_moments,
                        layer_adaptive_step,
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

            // ---------------------------------------------------------
            // Epoch statistics
            // ---------------------------------------------------------

            let samples =
                count
                    .saturating_sub(
                        epoch_start_count
                    ) as f32;

            if samples == 0.0 {
                continue;
            }

            let avg_loss =
                total_loss
                    / samples;

            let perplexity =
                avg_loss.exp();

            // ---------------------------------------------------------
            // Learning-rate plateau detection
            // ---------------------------------------------------------

            if avg_loss
                < best_loss * 0.998
            {
                best_loss =
                    avg_loss;

                plateau_count =
                    0;
            } else {
                plateau_count +=
                    1;
            }

            if plateau_count >= 10 {
                lr =
                    (lr * 0.8)
                        .max(min_lr);

                plateau_count =
                    0;

                println!(
                    "LR reduced to {}",
                    lr
                );
            }

            let elapsed =
                now.elapsed();

            let grad_avg =
                grad_sum
                    / samples;

            // ---------------------------------------------------------
            // Epoch checkpoint
            //
            // SAVE AFTER the plateau LR state has been updated.
            // ---------------------------------------------------------

            let should_checkpoint =
                match checkpoint_frequency {
                    CheckpointFrequency::EveryEpoch(
                        n
                    ) => {
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
                    &layer_second_moments,
                    layer_adaptive_step,
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
                    grad_avg
                        / param_count,
                    perplexity,
                    elapsed,
                );

                // -----------------------------------------------------
                // Optional timing breakdown
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                {
                    let elapsed_secs =
                        elapsed
                            .as_secs_f64();

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
}
