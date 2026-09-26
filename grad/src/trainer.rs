use std::borrow::Cow;
use std::fmt;
use std::io::{self, Write};
use std::ops::Range;
use std::path::Path;
use serde_json;
use serde::de::{
    DeserializeSeed,
    IgnoredAny,
    MapAccess,
    SeqAccess,
    Visitor,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{
    Duration,
    Instant,
};
use rand::prelude::SliceRandom;
use serde::Deserializer;
use crate::embeddings::Embeddings;
use crate::forwards::weight_tying_softmax_cross_entropy_tiled;
use crate::neuron::MLP;

// ============================================================================
// Deterministic permutation sampler
// ============================================================================

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
    pub const VERSION: u32 =
        PERMUTATION_SAMPLER_VERSION;

    pub const DEFAULT_SEED: u64 =
        DEFAULT_SAMPLER_SEED;

    pub fn new(
        seed: u64,
        epoch: usize,
        len: usize,
    ) -> Self {
        assert!(
            len > 0,
            "PermutationSampler length must be > 0"
        );

        let bits =
            if len <= 1 {
                0
            } else {
                usize::BITS
                    - (len - 1).leading_zeros()
            };

        let mask =
            Self::mask_for_bits(bits);

        let key =
            Self::derive_epoch_key(
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
    fn mask_for_bits(
        bits: u32,
    ) -> u64 {
        match bits {
            0 => 0,
            64 => u64::MAX,
            _ =>
                (1u64 << bits) - 1,
        }
    }

    #[inline]
    fn derive_epoch_key(
        seed: u64,
        epoch: u64,
    ) -> u64 {
        let mut x =
            seed
                ^ epoch.wrapping_mul(
                0x9E3779B97F4A7C15
            );

        x =
            x.wrapping_add(
                0x9E3779B97F4A7C15
            );

        x =
            (x ^ (x >> 30))
                .wrapping_mul(
                    0xBF58476D1CE4E5B9
                );

        x =
            (x ^ (x >> 27))
                .wrapping_mul(
                    0x94D049BB133111EB
                );

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

        x =
            x.wrapping_add(key)
                & mask;

        x ^=
            x >> 17;

        x =
            x.wrapping_mul(
                0xD6E8FEB86659FD93
            ) & mask;

        x ^=
            (x << 13) & mask;

        x =
            x.wrapping_mul(
                0xA5A3564E27F8863D
            ) & mask;

        x ^=
            x >> 11;

        x =
            x.wrapping_add(
                key.rotate_left(29)
            ) & mask;

        x ^=
            (x << 7) & mask;

        x
    }

    /// Returns the dataset index corresponding to `position`
    /// in this epoch's deterministic permutation.
    #[inline]
    pub fn index(
        &self,
        position: usize,
    ) -> usize {
        debug_assert!(
            position < self.len,
            "PermutationSampler position out of bounds"
        );

        if self.len == 1 {
            return 0;
        }

        let mut x =
            Self::permute_power_of_two(
                position as u64,
                self.bits,
                self.mask,
                self.key,
            );

        while x >= self.len as u64 {
            x =
                Self::permute_power_of_two(
                    x,
                    self.bits,
                    self.mask,
                    self.key,
                );
        }

        x as usize
    }

    pub fn seed(
        &self,
    ) -> u64 {
        self.seed
    }

    pub fn epoch(
        &self,
    ) -> u64 {
        self.epoch
    }

    pub fn len(
        &self,
    ) -> usize {
        self.len
    }
}

// ============================================================================
// Variable context
// ============================================================================

#[inline]
fn derive_context_len(
    sample: usize,
    epoch: usize,
    max_context_len: usize,
) -> usize {
    debug_assert!(
        max_context_len > 0
    );

    let mut x =
        (sample as u64)
            .wrapping_add(
                (epoch as u64)
                    .wrapping_mul(
                        0x9E3779B97F4A7C15
                    )
            );

    x ^=
        x >> 30;

    x =
        x.wrapping_mul(
            0xBF58476D1CE4E5B9
        );

    x ^=
        x >> 27;

    x =
        x.wrapping_mul(
            0x94D049BB133111EB
        );

    x ^=
        x >> 31;

    let min_context_len =
        max_context_len.min(8);

    min_context_len
        + (x as usize
        % (
        max_context_len
            - min_context_len
            + 1
    ))
}

// ============================================================================
// Training result / checkpoint types
// ============================================================================

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

#[derive(Clone, Copy, Debug)]
pub enum LrSchedule {
    /// No scheduling. LR stays at Trainer::lr.
    Constant,

    /// Linear warmup followed by cosine decay.
    ///
    /// `total_steps` includes warmup.
    ///
    /// Example:
    ///     warmup_steps = 1000
    ///     total_steps  = 340_788
    ///     min_lr_ratio = 0.1
    ///
    /// means:
    ///   step 1      -> ~0.1% of base LR
    ///   step 1000   -> 100% of base LR
    ///   step 340788 -> 10% of base LR
    Cosine {
        warmup_steps: u64,
        total_steps: u64,
        min_lr_ratio: f32,
    },
}

// ============================================================================
// Optional LM loss masking
// ============================================================================

/// Sorted, non-overlapping global token ranges whose target tokens
/// are allowed to contribute to the LM loss.
///
/// Ranges are half-open: [start, end).
///
/// IMPORTANT:
/// These ranges refer to TARGET token positions in `tokens`, not input
/// positions. If tokens[100] is predicted from tokens[99], then selecting
/// target position 100 means the prediction at input position 99 is trained.
#[derive(Debug, Clone)]
pub struct LmLossMask {
    ranges: Vec<Range<usize>>,
}

impl LmLossMask {
    pub fn new(
        mut ranges: Vec<Range<usize>>,
    ) -> Self {
        ranges.retain(|range| {
            range.start < range.end
        });

        ranges.sort_unstable_by_key(
            |range| range.start
        );

        // Merge overlapping / adjacent ranges.
        let mut merged =
            Vec::<Range<usize>>::with_capacity(
                ranges.len()
            );

        for range in ranges {
            if let Some(last) =
                merged.last_mut()
            {
                if range.start <= last.end {
                    last.end =
                        last.end.max(
                            range.end
                        );

                    continue;
                }
            }

            merged.push(range);
        }

        Self {
            ranges: merged,
        }
    }

    #[inline]
    pub fn ranges(
        &self,
    ) -> &[Range<usize>] {
        &self.ranges
    }

    /// Writes masked targets for one training sample.
    ///
    /// `sample` is the input-token starting position.
    ///
    /// The real target range is:
    ///
    ///     [sample + 1, sample + 1 + actual_context_len)
    ///
    /// Everything is initially IGNORE_TARGET, then only selected target
    /// positions are copied from `tokens`.
    #[inline]
    fn write_targets(
        &self,
        targets: &mut Vec<u16>,
        tokens: &[u16],
        sample: usize,
        actual_context_len: usize,
        ignore_target: u16,
    ) {
        let target_start =
            sample + 1;

        let target_end =
            target_start
                + actual_context_len;

        let output_start =
            targets.len();

        targets.resize(
            output_start
                + actual_context_len,
            ignore_target,
        );

        // Find the first range that could overlap this target window.
        //
        // Because `ranges` is sorted and merged, every range before this
        // point is guaranteed to end before the current window.
        let mut left = 0usize;
        let mut right =
            self.ranges.len();

        while left < right {
            let mid =
                left + (right - left) / 2;

            if self.ranges[mid].end
                <= target_start
            {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Copy only selected target spans.
        for range
        in &self.ranges[left..]
        {
            if range.start
                >= target_end
            {
                break;
            }

            let copy_start =
                range.start
                    .max(target_start);

            let copy_end =
                range.end
                    .min(target_end);

            if copy_start >= copy_end {
                continue;
            }

            let source =
                copy_start
                    ..copy_end;

            let destination_start =
                output_start
                    + (
                    copy_start
                        - target_start
                );

            let destination_end =
                destination_start
                    + source.len();

            targets[
                destination_start
                    ..destination_end
                ]
                .copy_from_slice(
                    &tokens[source]
                );
        }
    }
}

// ============================================================================
// Streaming JSON LM dataset
// ============================================================================

#[derive(Debug)]
pub struct JsonLmDataset {
    pub tokens: Vec<u16>,
    pub loss_mask: LmLossMask,
}

struct JsonBuildState<F> {
    tokens: Vec<u16>,
    ranges: Vec<Range<usize>>,
    train_on: Vec<String>,
    tokenizer: F,
}

struct TurnsSeed<'a, F> {
    state: &'a mut JsonBuildState<F>,
}

struct TurnSeed<'a, F> {
    state: &'a mut JsonBuildState<F>,
}

impl<'de, F> DeserializeSeed<'de>
for TurnsSeed<'_, F>
where
    F: FnMut(&str, &mut Vec<u16>),
{
    type Value = ();

    fn deserialize<D>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TurnsVisitor<'a, F> {
            state: &'a mut JsonBuildState<F>,
        }

        impl<'de, 'a, F> Visitor<'de>
        for TurnsVisitor<'a, F>
        where
            F: FnMut(&str, &mut Vec<u16>),
        {
            type Value = ();

            fn expecting(
                &self,
                formatter: &mut fmt::Formatter,
            ) -> fmt::Result {
                formatter.write_str(
                    "an array of turn objects"
                )
            }

            fn visit_seq<A>(
                self,
                mut seq: A,
            ) -> Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                while seq
                    .next_element_seed(
                        TurnSeed {
                            state: self.state,
                        }
                    )?
                    .is_some()
                {}

                Ok(())
            }
        }

        deserializer.deserialize_seq(
            TurnsVisitor {
                state: self.state,
            }
        )
    }
}

impl<'de, F> DeserializeSeed<'de>
for TurnSeed<'_, F>
where
    F: FnMut(&str, &mut Vec<u16>),
{
    type Value = ();

    fn deserialize<D>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TurnVisitor<'a, F> {
            state: &'a mut JsonBuildState<F>,
        }

        impl<'de, 'a, F> Visitor<'de>
        for TurnVisitor<'a, F>
        where
            F: FnMut(&str, &mut Vec<u16>),
        {
            type Value = ();

            fn expecting(
                &self,
                formatter: &mut fmt::Formatter,
            ) -> fmt::Result {
                formatter.write_str(
                    "a turn object"
                )
            }

            fn visit_map<M>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut role:
                    Option<Cow<'de, str>> = None;

                let mut text:
                    Option<Cow<'de, str>> = None;

                while let Some(key) =
                    map.next_key::<Cow<'de, str>>()?
                {
                    match key.as_ref() {
                        "role" => {
                            role =
                                Some(
                                    map.next_value::<Cow<'de, str>>()?
                                );
                        }

                        "text" => {
                            text =
                                Some(
                                    map.next_value::<Cow<'de, str>>()?
                                );
                        }

                        _ => {
                            let _: IgnoredAny =
                                map.next_value()?;
                        }
                    }
                }

                let role =
                    role.ok_or_else(|| {
                        serde::de::Error::missing_field(
                            "role"
                        )
                    })?;

                let text =
                    text.ok_or_else(|| {
                        serde::de::Error::missing_field(
                            "text"
                        )
                    })?;

                let selected =
                    self.state
                        .train_on
                        .iter()
                        .any(|wanted|
                            wanted.as_str()
                                == role.as_ref()
                        );

                let start =
                    self.state.tokens.len();

                // IMPORTANT:
                // Tokenize directly into the final dataset buffer.
                // No temporary Vec<u16>, and no second copy.
                (self.state.tokenizer)(
                    text.as_ref(),
                    &mut self.state.tokens,
                );

                let end =
                    self.state.tokens.len();

                if selected && start < end {
                    self.state
                        .ranges
                        .push(start..end);
                }

                Ok(())
            }
        }

        deserializer.deserialize_map(
            TurnVisitor {
                state: self.state,
            }
        )
    }
}

struct DatasetJsonVisitor<F> {
    state: JsonBuildState<F>,
}

impl<'de, F> Visitor<'de>
for DatasetJsonVisitor<F>
where
    F: FnMut(&str, &mut Vec<u16>),
{
    type Value = JsonLmDataset;

    fn expecting(
        &self,
        formatter: &mut fmt::Formatter,
    ) -> fmt::Result {
        formatter.write_str(
            "a JSON LM dataset"
        )
    }

    fn visit_map<A>(
        mut self,
        mut map: A,
    ) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(key) =
            map.next_key::<Cow<'de, str>>()?
        {
            match key.as_ref() {
                "train_on" => {
                    self.state.train_on =
                        map.next_value()?;
                }

                "turns" => {
                    map.next_value_seed(
                        TurnsSeed {
                            state:
                            &mut self.state,
                        }
                    )?;
                }

                _ => {
                    let _: IgnoredAny =
                        map.next_value()?;
                }
            }
        }

        let loss_mask =
            LmLossMask::new(
                std::mem::take(
                    &mut self.state.ranges
                )
            );

        Ok(JsonLmDataset {
            tokens:
            std::mem::take(
                &mut self.state.tokens
            ),

            loss_mask,
        })
    }
}

pub fn load_masked_json_lm_dataset<F>(
    path: impl AsRef<Path>,
    initial_capacity: usize,
    tokenizer: F,
) -> Result<
    JsonLmDataset,
    Box<dyn std::error::Error>,
>
where
    F: FnMut(&str, &mut Vec<u16>),
{
    let json =
        std::fs::read(path)?;

    let state =
        JsonBuildState {
            tokens:
            Vec::with_capacity(
                initial_capacity
            ),

            ranges:
            Vec::new(),

            train_on:
            Vec::new(),

            tokenizer,
        };

    let mut deserializer =
        serde_json::Deserializer::from_slice(
            &json
        );

    Ok(
        deserializer.deserialize_map(
            DatasetJsonVisitor {
                state,
            }
        )?
    )
}

// ============================================================================
// Trainer
// ============================================================================

pub struct Trainer {
    lr: f32,
    lr_schedule: LrSchedule,
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
        Self {
            lr,
            lr_schedule: LrSchedule::Constant,

            epochs,
            batch_size: batch_size.max(1),
            max_batches_per_epoch,
        }
    }

    pub fn reinit_lr_schedule(
        &mut self,
        schedule: LrSchedule,
    ) {
        self.lr_schedule =
            schedule;
    }

    pub fn reinit_lr(
        &mut self,
        lr: f32,
    ) {
        self.lr =
            lr;
    }

    pub fn reinit_epochs(
        &mut self,
        epochs: usize,
    ) {
        self.epochs =
            epochs;
    }

    pub fn reinit_batch(
        &mut self,
        batch_size: usize,
    ) {
        self.batch_size =
            batch_size.max(1);
    }

    pub fn reinit_batch_per_epoch(
        &mut self,
        max_batches_per_epoch: usize,
    ) {
        self.max_batches_per_epoch =
            max_batches_per_epoch;
    }

    #[inline]
    fn lr_for_step(
        &self,
        step: u64,
    ) -> f32 {
        let base_lr =
            self.lr.max(0.0);

        match self.lr_schedule {
            LrSchedule::Constant => {
                base_lr
            }

            LrSchedule::Cosine {
                warmup_steps,
                total_steps,
                min_lr_ratio,
            } => {
                let min_lr_ratio =
                    min_lr_ratio
                        .clamp(0.0, 1.0);

                let min_lr =
                    base_lr
                        * min_lr_ratio;

                // -----------------------------------------------------
                // Warmup
                // -----------------------------------------------------

                if warmup_steps > 0
                    && step <= warmup_steps
                {
                    return base_lr
                        * (
                        step as f32
                            / warmup_steps
                            as f32
                    );
                }

                // -----------------------------------------------------
                // Cosine decay
                // -----------------------------------------------------

                if total_steps
                    <= warmup_steps
                {
                    return min_lr;
                }

                let decay_steps =
                    total_steps
                        - warmup_steps;

                let decay_step =
                    step
                        .saturating_sub(
                            warmup_steps
                        );

                let progress =
                    (
                        decay_step
                            as f32
                            / decay_steps
                            as f32
                    )
                        .clamp(0.0, 1.0);

                let cosine =
                    0.5
                        * (
                        1.0
                            + (
                            std::f32::consts::PI
                                * progress
                        ).cos()
                    );

                min_lr
                    + (
                    base_lr
                        - min_lr
                )
                    * cosine
            }
        }
    }

    // ========================================================================
    // Simple dense MLP trainer
    // ========================================================================

    pub fn train(
        &mut self,
        update_frequency: usize,
        mlp: &mut MLP,
        dataset: &mut [(Vec<f32>, Vec<f32>)],
        threads: usize,
    ) -> TrainResult {
        let running =
            Arc::new(
                AtomicBool::new(true)
            );

        let running_flag =
            running.clone();

        ctrlc::set_handler(
            move || {
                println!(
                    "\nStopping after current batch..."
                );

                running_flag.store(
                    false,
                    Ordering::SeqCst,
                );
            },
        )
            .expect(
                "Error setting Ctrl+C handler"
            );

        let data_len = dataset.len();

        if data_len == 0 {
            println!("Dataset is empty!");
            return TrainResult::Finished;
        }

        if self.batch_size > data_len {
            println!("Batch size too high! Quitting...");
            return TrainResult::Finished;
        }

        let input_size =
            dataset[0].0.len();

        let output_size =
            dataset[0].1.len();

        if input_size == 0
            || output_size == 0
        {
            println!(
                "Input/output vectors cannot be empty!"
            );

            return TrainResult::Finished;
        }

        for (
            input,
            target,
        ) in dataset.iter()
        {
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

        mlp.set_num_threads(
            threads
        );

        let mut indices:
            Vec<usize> =
            (0..data_len)
                .collect();

        let mut lr =
            self.lr;

        let mut best_loss =
            f32::MAX;

        let mut plateau_count =
            0usize;

        let parameter_count =
            mlp.parameter_count();

        for epoch in
            1..=self.epochs
        {
            let now =
                Instant::now();

            let mut rng =
                rand::rng();

            indices.shuffle(
                &mut rng
            );

            let mut total_loss =
                0.0f32;

            let mut grad_sum =
                0.0f32;

            let mut count =
                0usize;

            let mut batches_done =
                0usize;

            while count < data_len {
                if self.max_batches_per_epoch
                    != 0
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

                let mut batch_input =
                    Vec::with_capacity(
                        current_batch
                            * input_size
                    );

                let mut targets =
                    Vec::with_capacity(
                        current_batch
                            * output_size
                    );

                for batch_index
                in 0..current_batch
                {
                    let sample =
                        indices[
                            count
                                + batch_index
                            ];

                    let (
                        input,
                        target,
                    ) =
                        &dataset[
                            sample
                            ];

                    batch_input
                        .extend_from_slice(
                            input
                        );

                    targets
                        .extend_from_slice(
                            target
                        );
                }

                let forward =
                    mlp.forward_batch(
                        &batch_input,
                        current_batch,
                        input_size,
                    );

                debug_assert_eq!(
                    forward.output_size,
                    output_size
                );

                let mut output_grads =
                    vec![
                        0.0f32;
                        current_batch
                            * output_size
                    ];

                let batch_loss =
                    crate::batched::mse_batch(
                        &forward.output,
                        &targets,
                        &mut output_grads,
                        current_batch,
                        output_size,
                    );

                total_loss +=
                    batch_loss
                        * current_batch
                        as f32;

                mlp.backward_batch(
                    &forward,
                    &output_grads,
                );

                let grads =
                    &mlp.params.grads;

                let mut batch_grad_sum =
                    0.0f64;

                for &g in grads {
                    if !g.is_finite() {
                        println!(
                            "Non-finite gradient detected. Stopping training."
                        );

                        mlp.params.zero_grads();

                        return TrainResult::Finished;
                    }

                    batch_grad_sum +=
                        g.abs() as f64;
                }

                grad_sum +=
                    batch_grad_sum as f32;

                {
                    let (
                        values,
                        grads,
                    ) = (
                        &mut mlp.params.values,
                        &mut mlp.params.grads,
                    );

                    for i in 0..values.len() {
                        values[i] -=
                            lr * grads[i];
                    }
                }

                mlp.params.zero_grads();

                count +=
                    current_batch;

                batches_done +=
                    1;

                if !running.load(
                    Ordering::SeqCst
                ) {
                    println!(
                        "Interrupted"
                    );

                    return TrainResult::Interrupted;
                }
            }

            let samples =
                count as f32;

            if samples == 0.0 {
                continue;
            }

            let avg_loss =
                total_loss
                    / samples;

            let grad_avg =
                grad_sum
                    / samples;

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
                        .max(
                            0.001
                        );

                plateau_count =
                    0;

                println!(
                    "LR reduced to {}",
                    lr
                );
            }

            let elapsed =
                now.elapsed();

            if update_frequency > 0
                && epoch
                % update_frequency
                == 0
            {
                let grad_per_param =
                    if parameter_count > 0 {
                        grad_avg
                            / parameter_count
                            as f32
                    } else {
                        0.0
                    };

                println!(
                    "Epoch {} | Loss (MSE) = {:.6} | \
                     Grad avg = {:.8} | Time elapsed: {:.2?}.",
                    epoch,
                    avg_loss,
                    grad_per_param,
                    elapsed,
                );
            }

            if grad_avg <= 1e-9 {
                println!(
                    "Early stopping, network will not learn anymore!"
                );

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
        variable_context: bool,
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
        loss_mask: Option<&LmLossMask>,
        sampler_seed: u64,
        threads: usize,
    ) -> TrainResult {
        const ADAM_BETA1: f32 = 0.9;
        const ADAM_BETA2: f32 = 0.999;
        const ADAM_EPSILON: f32 = 1.0e-8;
        const MAX_GRAD_NORM: f64 = 1.0;
        const IGNORE_TARGET: u16 = u16::MAX;

        // =============================================================
        // Ctrl+C
        // =============================================================

        let interrupt_requested =
            Arc::new(
                AtomicBool::new(false)
            );

        let interrupt_flag =
            interrupt_requested.clone();

        ctrlc::set_handler(move || {
            interrupt_flag.store(
                true,
                Ordering::SeqCst,
            );
        })
            .expect(
                "Error setting Ctrl+C handler"
            );

        // =============================================================
        // Dataset validation
        // =============================================================

        let data_len =
            tokens
                .len()
                .saturating_sub(context_len);

        if let Some(mask) =
            loss_mask
        {
            for range in mask.ranges() {
                assert!(
                    range.end <= tokens.len(),
                    "LM loss-mask range {:?} exceeds token dataset length {}",
                    range,
                    tokens.len(),
                );

                assert!(
                    range.start < range.end,
                    "LM loss-mask range must be non-empty: {:?}",
                    range,
                );
            }
        }

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

        mlp.set_num_threads(
            threads
        );

        let parameter_count =
            mlp.parameter_count();

        assert_eq!(
            mlp.params.values.len(),
            parameter_count
        );

        assert_eq!(
            mlp.params.grads.len(),
            parameter_count
        );

        let embedding_dim =
            embeddings.embedding_dim();

        // =============================================================
        // Optimizer state
        // =============================================================

        let mut lr =
            self.lr;

        let mut best_loss =
            f32::MAX;

        let mut plateau_count =
            0usize;

        let mut adam_first_moments =
            vec![
                0.0f32;
                parameter_count
            ];

        let mut adam_second_moments =
            vec![
                0.0f32;
                parameter_count
            ];

        let mut adam_step =
            0u64;

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
                    assert_eq!(
                        state.sampler_version,
                        PermutationSampler::VERSION,
                        "Unsupported permutation sampler version in checkpoint"
                    );

                    assert_eq!(
                        state.sampler_data_len,
                        data_len,
                        "Dataset length differs from checkpoint"
                    );

                    assert_eq!(
                        state.sampler_seed,
                        sampler_seed,
                        "Sampler seed differs from checkpoint"
                    );

                    lr =
                        state.lr;

                    best_loss =
                        state.best_loss;

                    plateau_count =
                        state.plateau_count;

                    if state.adam_first_moments.len()
                        == parameter_count
                        && state.adam_second_moments.len()
                        == parameter_count
                    {
                        adam_first_moments =
                            state.adam_first_moments.clone();

                        adam_second_moments =
                            state.adam_second_moments.clone();

                        adam_step =
                            state.adam_step;

                        if adam_step > 0 {
                            beta1_power =
                                ADAM_BETA1.powi(
                                    adam_step as i32
                                );

                            beta2_power =
                                ADAM_BETA2.powi(
                                    adam_step as i32
                                );
                        }

                        println!(
                            "Restored Adam state: {} parameters, step {}.",
                            parameter_count,
                            adam_step,
                        );
                    } else {
                        println!(
                            "Checkpoint has incompatible Adam state; \
                         starting Adam moments from zero."
                        );
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

                None => (
                    1,
                    None,
                ),
            };

        // =============================================================
        // Checkpoint helper
        // =============================================================

        fn emit_checkpoint(
            savefn: &mut Option<
                &mut dyn FnMut(
                    CheckpointState,
                    &MLP,
                    &Embeddings,
                ),
            >,
            kind: CheckpointKind,
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
            adam_step: u64,
            mlp: &MLP,
            embeddings: &Embeddings,
        ) {
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
        }

        // =============================================================
        // Reusable batch buffers
        // =============================================================

        let mut batch_ids =
            Vec::<u16>::with_capacity(
                self.batch_size
                    * context_len
            );

        let mut targets =
            Vec::<u16>::with_capacity(
                self.batch_size
                    * context_len
            );

        let mut batch_input =
            Vec::<f32>::with_capacity(
                self.batch_size
                    * context_len
                    * embedding_dim
            );

        let mut hidden_grads =
            Vec::<f32>::new();

        let mut embedding_output_grads =
            Vec::<f32>::new();

        let mut tied_row_max =
            Vec::<f32>::new();

        let mut tied_row_sum =
            Vec::<f32>::new();

        let mut tied_target_logits =
            Vec::<f32>::new();

        let mut tied_tile_logits =
            Vec::<f32>::new();

        let mut tied_tile_embedding_grads =
            Vec::<f32>::new();

        // =============================================================
        // Epochs
        // =============================================================

        for epoch
        in start_epoch..=self.epochs {
            let now =
                Instant::now();

            let sampler =
                PermutationSampler::new(
                    sampler_seed,
                    epoch,
                    data_len,
                );

            let (
                mut count,
                mut batches_done,
            ) =
                match resume_state.take() {
                    Some(state) => {
                        debug_assert_eq!(
                            state.epoch,
                            epoch
                        );

                        println!(
                            "Continuing epoch {} from batch {}.",
                            epoch,
                            state.batch,
                        );

                        (
                            state.sample,
                            state.batch,
                        )
                    }

                    None => (
                        0,
                        0,
                    ),
                };

            let epoch_start_count =
                count;

            let total_batches =
                (
                    data_len
                        + self.batch_size
                        - 1
                ) / self.batch_size;

            let mut total_loss =
                0.0f32;

            let mut total_valid_targets =
                0usize;

            let mut grad_sum =
                0.0f32;

            #[cfg(feature = "timing")]
            let mut encode_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut forward_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut tied_head_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut backward_time =
                Duration::ZERO;

            #[cfg(feature = "timing")]
            let mut embedding_grad_time =
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
                // Choose ONE context length for the whole batch.
                //
                // This is the key change:
                //
                // batch 1 -> e.g. 37 tokens
                // batch 2 -> e.g. 11 tokens
                // batch 3 -> e.g. 128 tokens
                //
                // Every sample inside a batch has the same length, so
                // the batch remains rectangular and no PAD token exists.
                // -----------------------------------------------------

                let batch_context_len =
                    if variable_context {
                        let first_sample =
                            sampler.index(
                                count
                            );

                        derive_context_len(
                            first_sample,
                            epoch,
                            context_len,
                        )
                    } else {
                        context_len
                    };

                let input_size =
                    batch_context_len
                        * embedding_dim;

                // -----------------------------------------------------
                // Build token batch
                // -----------------------------------------------------

                batch_ids.clear();
                targets.clear();

                for batch_index
                in 0..current_batch
                {
                    let sample =
                        sampler.index(
                            count
                                + batch_index
                        );

                    // -------------------------------------------------
                    // Inputs:
                    //
                    // [token0 token1 ... tokenN]
                    //
                    // No PAD.
                    // -------------------------------------------------

                    batch_ids.extend_from_slice(
                        &tokens[
                            sample
                                ..sample
                                + batch_context_len
                            ]
                    );

                    // -------------------------------------------------
                    // Targets:
                    //
                    // [token1 token2 ... tokenN+1]
                    //
                    // Or IGNORE_TARGET for masked positions.
                    // -------------------------------------------------

                    match loss_mask {
                        None => {
                            targets.extend_from_slice(
                                &tokens[
                                    sample + 1
                                        ..sample
                                        + batch_context_len
                                        + 1
                                    ]
                            );
                        }

                        Some(mask) => {
                            mask.write_targets(
                                &mut targets,
                                tokens,
                                sample,
                                batch_context_len,
                                IGNORE_TARGET,
                            );
                        }
                    }
                }

                // -----------------------------------------------------
                // Embeddings
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                embeddings.encode_batch_into(
                    &mlp.params,
                    &batch_ids,
                    current_batch,
                    batch_context_len,
                    &mut batch_input,
                );

                #[cfg(feature = "timing")]
                {
                    encode_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Forward: backbone only
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                let forward =
                    mlp.forward_lm_batch(
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
                // Tiled tied vocabulary head
                // -----------------------------------------------------

                let positions =
                    batch_context_len;

                let vocab_size =
                    embeddings.vocab_size();

                let rows =
                    current_batch
                        * positions;

                debug_assert_eq!(
                    batch_ids.len(),
                    rows
                );

                debug_assert_eq!(
                    targets.len(),
                    rows
                );

                debug_assert_eq!(
                    forward.output.len(),
                    rows * embedding_dim
                );

                debug_assert_eq!(
                    forward.output_size,
                    positions * embedding_dim,
                    "LM backbone must end at positions × embedding_dim"
                );

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                let (
                    batch_loss,
                    valid_count,
                ) =
                    weight_tying_softmax_cross_entropy_tiled(
                        &forward.output,
                        &targets,

                        &mut hidden_grads,
                        &mut embedding_output_grads,

                        &mut tied_row_max,
                        &mut tied_row_sum,
                        &mut tied_target_logits,

                        &mut tied_tile_logits,
                        &mut tied_tile_embedding_grads,

                        current_batch,
                        positions,
                        embedding_dim,
                        vocab_size,

                        mlp.params.values(
                            embeddings.parameter_range()
                        ),

                        IGNORE_TARGET,
                    );

                #[cfg(feature = "timing")]
                {
                    tied_head_time +=
                        timer.elapsed();
                }

                if !batch_loss.is_finite() {
                    println!("Invalid batch loss. Stopping training.");
                    mlp.params.zero_grads();
                    return TrainResult::Finished;
                }

                // -----------------------------------------------------
                // Empty masked batch
                // -----------------------------------------------------

                if valid_count == 0 {
                    mlp.params.zero_grads();

                    count +=
                        current_batch;

                    batches_done +=
                        1;

                    if let Some(frequency) =
                        batch_update_frequency
                    {
                        if frequency > 0
                            && batches_done
                            % frequency
                            == 0
                        {
                            println!(
                                "Epoch {} | Batch {}/{} | \
                 masked batch: 0 valid targets, skipped",
                                epoch,
                                batches_done,
                                total_batches,
                            );
                        }
                    }

                    if interrupt_requested.load(
                        Ordering::SeqCst
                    ) {
                        println!();
                        println!("Ctrl+C received. Current batch has finished.");
                        emit_checkpoint(
                            &mut savefn,
                            CheckpointKind::Batch,
                            epoch,
                            batches_done,
                            count,
                            sampler_seed,
                            data_len,
                            self.lr,
                            best_loss,
                            plateau_count,
                            &adam_first_moments,
                            &adam_second_moments,
                            adam_step,
                            mlp,
                            embeddings,
                        );

                        loop {
                            print!("Exit training? [y/N]: ");
                            io::stdout().flush().ok();

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

                                        _ => println!(
                                            "Please enter Y or N."
                                        ),
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

                    continue;
                }

                total_loss +=
                    batch_loss;

                total_valid_targets +=
                    valid_count;

                // -----------------------------------------------------
                // Add tied-output embedding gradients
                // -----------------------------------------------------

                {
                    let embedding_range =
                        embeddings.parameter_range();

                    let embedding_grads =
                        mlp.params.grads_mut(
                            embedding_range
                        );

                    debug_assert_eq!(
                        embedding_grads.len(),
                        embedding_output_grads.len()
                    );

                    for i in
                        0..embedding_grads.len()
                    {
                        embedding_grads[i] +=
                            embedding_output_grads[i];
                    }
                }

                // -----------------------------------------------------
                // Backward through backbone
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                let input_grads =
                    mlp.backward_lm_batch(
                        &forward,
                        &hidden_grads,
                    );

                #[cfg(feature = "timing")]
                {
                    backward_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Input embedding gradients
                // -----------------------------------------------------

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                embeddings.accumulate_batch_grads(
                    &mut mlp.params,
                    &batch_ids,
                    &input_grads,
                    current_batch,
                    batch_context_len,
                );

                #[cfg(feature = "timing")]
                {
                    embedding_grad_time +=
                        timer.elapsed();
                }

                // -----------------------------------------------------
                // Gradient validation
                // -----------------------------------------------------

                let batch_normalization =
                    1.0f32
                        / valid_count.max(1) as f32;

                let mut grad_sq_sum =
                    0.0f64;

                let mut batch_grad_sum =
                    0.0f64;

                for &g in
                    &mlp.params.grads
                {
                    if !g.is_finite() {
                        println!(
                            "Non-finite gradient detected. \
                         Stopping training."
                        );

                        mlp.params.zero_grads();

                        return TrainResult::Finished;
                    }

                    let mean_grad =
                        g
                            * batch_normalization;

                    let gf =
                        mean_grad as f64;

                    grad_sq_sum +=
                        gf * gf;

                    batch_grad_sum +=
                        gf.abs();
                }

                let grad_norm =
                    grad_sq_sum.sqrt();

                let grad_scale =
                    if grad_norm
                        > MAX_GRAD_NORM
                    {
                        (
                            MAX_GRAD_NORM
                                / grad_norm
                        ) as f32
                    } else {
                        1.0
                    };

                grad_sum +=
                    batch_grad_sum as f32
                        * valid_count as f32;

                // -----------------------------------------------------
                // Adam step + LR schedule
                // -----------------------------------------------------

                let next_adam_step =
                    adam_step + 1;

                lr =
                    self.lr_for_step(
                        next_adam_step
                    );

                adam_step =
                    next_adam_step;

                beta1_power *=
                    ADAM_BETA1;

                beta2_power *=
                    ADAM_BETA2;

                let beta1_correction_inv =
                    1.0f32
                        / (
                        1.0f32
                            - beta1_power
                    );

                let beta2_correction_inv =
                    1.0f32
                        / (
                        1.0f32
                            - beta2_power
                    );

                #[cfg(feature = "timing")]
                let timer =
                    Instant::now();

                for i in
                    0..parameter_count
                {
                    let gradient =
                        mlp.params.grads[i]
                            * batch_normalization
                            * grad_scale;

                    let m =
                        &mut adam_first_moments[
                            i
                            ];

                    let v =
                        &mut adam_second_moments[
                            i
                            ];

                    *m =
                        ADAM_BETA1
                            * *m
                            + (
                            1.0
                                - ADAM_BETA1
                        )
                            * gradient;

                    *v =
                        ADAM_BETA2
                            * *v
                            + (
                            1.0
                                - ADAM_BETA2
                        )
                            * gradient
                            * gradient;

                    let m_hat =
                        *m
                            * beta1_correction_inv;

                    let v_hat =
                        *v
                            * beta2_correction_inv;

                    mlp.params.values[i] -=
                        lr
                            * m_hat
                            / (
                            v_hat.sqrt()
                                + ADAM_EPSILON
                        );
                }

                mlp.params.zero_grads();

                #[cfg(feature = "timing")]
                {
                    update_time +=
                        timer.elapsed();
                }

                count +=
                    current_batch;

                batches_done +=
                    1;

                // -----------------------------------------------------
                // Batch checkpoint
                // -----------------------------------------------------

                let should_checkpoint =
                    match checkpoint_frequency {
                        CheckpointFrequency::EveryBatch(
                            frequency
                        ) =>
                            frequency > 0
                                && batches_done
                                % frequency
                                == 0,

                        _ =>
                            false,
                    };

                if should_checkpoint {
                    emit_checkpoint(
                        &mut savefn,
                        CheckpointKind::Batch,
                        epoch,
                        batches_done,
                        count,
                        sampler_seed,
                        data_len,
                        self.lr,
                        best_loss,
                        plateau_count,
                        &adam_first_moments,
                        &adam_second_moments,
                        adam_step,
                        mlp,
                        embeddings,
                    );
                }

                // -----------------------------------------------------
                // Batch progress
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

                        let processed_valid_targets =
                            total_valid_targets;

                        let running_loss =
                            if processed_valid_targets
                                > 0
                            {
                                total_loss
                                    / processed_valid_targets
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
                                processed_samples
                                    as f64
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

                        let batch_avg_loss =
                            batch_loss
                                / valid_count.max(1)
                                as f32;

                        println!(
                            "Epoch {} | Batch {}/{} | {:>6.2}% | \
                         Samples {}/{} | AvgLoss = {:.6} | \
                         AvgPPL = {:.6}\n\
                         Loss = {:.6} | PPL = {:.6} | \
                         {:.1} samples/s | \
                         Elapsed: {:.2?} | ETA: {:.2?} | LR: {}",
                            epoch,
                            batches_done,
                            total_batches,
                            progress * 100.0,
                            count,
                            data_len,
                            running_loss,
                            running_ppl,
                            batch_avg_loss,
                            batch_avg_loss.exp(),
                            samples_per_sec,
                            elapsed,
                            eta,
                            lr,
                        );

                        #[cfg(feature = "timing")]
                        {
                            let elapsed_secs =
                                elapsed.as_secs_f64();

                            let pct =
                                |duration: Duration| {
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
                                "  Tied LM head:    {:>10.3?} ({:>6.2}%)",
                                tied_head_time,
                                pct(tied_head_time)
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
                                "  Adam update:     {:>10.3?} ({:>6.2}%)",
                                update_time,
                                pct(update_time)
                            );
                        }
                    }
                }

                // -----------------------------------------------------
                // Ctrl+C
                // -----------------------------------------------------

                if interrupt_requested.load(
                    Ordering::SeqCst
                ) {
                    println!();
                    println!("Ctrl+C received. Current batch has finished.");

                    emit_checkpoint(
                        &mut savefn,
                        CheckpointKind::Batch,
                        epoch,
                        batches_done,
                        count,
                        sampler_seed,
                        data_len,
                        self.lr,
                        best_loss,
                        plateau_count,
                        &adam_first_moments,
                        &adam_second_moments,
                        adam_step,
                        mlp,
                        embeddings,
                    );

                    loop {
                        print!("Exit training? [y/N]: ");
                        io::stdout().flush().ok();

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

                                    _ => println!(
                                        "Please enter Y or N."
                                    ),
                                }
                            }

                            Err(_) => {
                                println!(
                                    "Could not read input. Exiting training."
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

            if total_valid_targets == 0 {
                continue;
            }

            let valid_targets =
                total_valid_targets as f32;

            let avg_loss =
                total_loss
                    / valid_targets;

            let perplexity =
                avg_loss.exp();

            let grad_avg =
                grad_sum
                    / valid_targets;

            // ---------------------------------------------------------
            // Best-loss tracking
            // ---------------------------------------------------------

            if avg_loss < best_loss {
                best_loss = avg_loss;
                plateau_count = 0;
            } else {
                plateau_count += 1;
            }

            // ---------------------------------------------------------
            // Epoch checkpoint
            // ---------------------------------------------------------

            let should_checkpoint =
                match checkpoint_frequency {
                    CheckpointFrequency::EveryEpoch(
                        frequency
                    ) =>
                        frequency > 0
                            && epoch % frequency
                            == 0,

                    _ => false,
                };

            if should_checkpoint {
                emit_checkpoint(
                    &mut savefn,
                    CheckpointKind::Epoch,
                    epoch,
                    batches_done,
                    count,
                    sampler_seed,
                    data_len,
                    self.lr,
                    best_loss,
                    plateau_count,
                    &adam_first_moments,
                    &adam_second_moments,
                    adam_step,
                    mlp,
                    embeddings,
                );
            }

            // ---------------------------------------------------------
            // Epoch logging
            // ---------------------------------------------------------

            let elapsed =
                now.elapsed();

            if update_frequency > 0
                && epoch % update_frequency
                == 0
            {
                let grad_per_param =
                    if parameter_count > 0 {
                        grad_avg
                            / parameter_count
                            as f32
                    } else {
                        0.0
                    };

                println!(
                    "Epoch {} | Loss (CE) = {:.6} | \
                 Grad avg per param = {:.8} | \
                 PPL = {:.6} | \
                 Time elapsed: {:.2?}.",
                    epoch,
                    avg_loss,
                    grad_per_param,
                    perplexity,
                    elapsed,
                );

                #[cfg(feature = "timing")]
                {
                    let elapsed_secs =
                        elapsed.as_secs_f64();

                    let pct =
                        |duration: Duration| {
                            if elapsed_secs > 0.0 {
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
                        "  Tied LM head:    {:>10.3?} ({:>6.2}%)",
                        tied_head_time,
                        pct(tied_head_time)
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
                        "  Adam update:     {:>10.3?} ({:>6.2}%)",
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
