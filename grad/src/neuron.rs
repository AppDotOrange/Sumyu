use std::sync::Arc;

use rand_distr::{
    Distribution,
    Normal as NormalDist,
};
use rayon::ThreadPool;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    embeddings::Embeddings,
    parameters::{
        ParamRange,
        ParameterStore,
    },
};

pub use crate::{
    backwards::*,
    forwards::*,
};

#[derive(
    Clone,
    Serialize,
    Deserialize,
    Copy,
)]
pub enum Activation {
    None,
    LeakyReLU {
        slope: f32,
    },
}

impl Activation {
    #[inline]
    pub(crate) fn apply(
        &self,
        x: f32,
    ) -> f32 {
        match self {
            Self::None => x,

            Self::LeakyReLU { slope } => {
                if x <= 0.0 {
                    x * slope
                } else {
                    x
                }
            }
        }
    }

    #[inline]
    pub fn backward(
        &self,
        x: f32,
        grad: &mut f32,
    ) {
        if let Self::LeakyReLU { slope } = self {
            if x <= 0.0 {
                *grad *= *slope;
            }
        }
    }
}

#[derive(
    Serialize,
    Deserialize,
)]
pub struct SavedMLP {
    pub version: u32,
    pub layers: Vec<SavedLayer>,
}

#[derive(
    Serialize,
    Deserialize,
)]
pub enum SavedLayer {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },

    Conv1D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        #[serde(default)]
        causal: bool,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },

    Residual {
        input_size: usize,
        output_size: usize,
        layers: Vec<SavedLayer>,
    },

    DepthwiseConv1D {
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        #[serde(default)]
        causal: bool,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },

    GroupedConv1D {
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        #[serde(default)]
        causal: bool,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },

    LowRankPointwise {
        in_channels: usize,
        rank: usize,
        out_channels: usize,
        activation: Activation,
        first_weights: Vec<f32>,
        first_biases: Vec<f32>,
        second_weights: Vec<f32>,
        second_biases: Vec<f32>,
    },

    ChannelScale {
        channels: usize,
        scales: Vec<f32>,
        biases: Vec<f32>,
    },

    WeightTying {
        embedding_dim: usize,
        vocab_size: usize,
    },

    LayerNorm {
        channels: usize,
        epsilon: f32,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    },

    GlobalMixer {
        channels: usize,
        global_dim: usize,

        write_weights: Vec<f32>,
        write_biases: Vec<f32>,

        read_weights: Vec<f32>,
        read_biases: Vec<f32>,

        #[serde(default)]
        write_positional_weights: Vec<f32>,

        #[serde(default)]
        read_positional_weights: Vec<f32>,
    },
}

#[derive(Clone)]
pub enum LayerSpec {
    Dense {
        output_size: usize,
        activation: Activation,
    },

    Conv1D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    },

    Residual {
        layers: Vec<LayerSpec>,
    },

    DepthwiseConv1D {
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    },

    GroupedConv1D {
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    },

    LowRankPointwise {
        in_channels: usize,
        rank: usize,
        out_channels: usize,
        activation: Activation,
    },

    ChannelScale {
        channels: usize,
    },

    WeightTying,

    LayerNorm {
        channels: usize,
        epsilon: f32,
    },

    GlobalMixer {
        channels: usize,
        global_dim: usize,
    },
}

pub struct BatchForward {
    pub output: Vec<f32>,
    pub batch_size: usize,
    pub output_size: usize,
    pub layers: Vec<BatchLayerCache>,
}

pub enum BatchLayerCache {
    Dense {
        input_size: usize,
        output_size: usize,
        input: Vec<f32>,
        activation_output: Option<Vec<f32>>,
        activation: Activation,
    },

    Conv1D {
        input: Vec<f32>,
        output: Vec<f32>,
        input_length: usize,
        output_length: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    },

    Residual {
        inner: Vec<BatchLayerCache>,
    },

    DepthwiseConv1D {
        input: Vec<f32>,
        output: Vec<f32>,
        input_length: usize,
        output_length: usize,
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    },

    GroupedConv1D {
        input: Vec<f32>,
        output: Vec<f32>,
        input_length: usize,
        output_length: usize,
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    },

    LowRankPointwise {
        input: Vec<f32>,
        hidden: Vec<f32>,
        output: Vec<f32>,

        rows: usize,
        in_channels: usize,
        rank: usize,
        out_channels: usize,

        first_weights: Vec<f32>,
        second_weights: Vec<f32>,

        activation: Activation,
    },

    ChannelScale {
        input: Vec<f32>,
        channels: usize,
    },

    WeightTying {
        input: Vec<f32>,
        embeddings: Arc<Embeddings>,
        batch_size: usize,
        embedding_dim: usize,
        vocab_size: usize,
    },

    LayerNorm {
        input: Vec<f32>,
        means: Vec<f32>,
        inv_stds: Vec<f32>,
        channels: usize,
    },

    GlobalMixer {
        input: Vec<f32>,

        write_probs: Vec<f32>,
        global_vectors: Vec<f32>,
        read_probs: Vec<f32>,

        positions: usize,
        channels: usize,
        global_dim: usize,
    },
}

#[derive(Clone)]
pub struct DenseLayer {
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,

    pub(crate) weights: ParamRange,
    pub(crate) biases: ParamRange,

    pub(crate) activation: Activation,
}

impl DenseLayer {
    fn new(
        params: &mut ParameterStore,
        num_inputs: usize,
        num_outputs: usize,
        activation: Activation,
    ) -> Self {
        let mut rng =
            rand::rng();

        let std_dev =
            (2.0 / num_inputs as f32).sqrt();

        let normal =
            NormalDist::new(
                0.0,
                std_dev,
            )
                .expect(
                    "Invalid standard deviation"
                );

        let weights =
            params.alloc_many(
                (0..num_inputs * num_outputs)
                    .map(|_| {
                        normal.sample(&mut rng)
                    }),
            );

        let biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.1f32,
                    num_outputs,
                ),
            );

        Self {
            input_size: num_inputs,
            output_size: num_outputs,
            weights,
            biases,
            activation,
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Dense {
            output_size: self.output_size,
            activation: self.activation,
        }
    }
}

#[derive(Clone)]
pub struct Conv1DLayer {
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) causal: bool,
    pub(crate) activation: Activation,

    pub(crate) weights: ParamRange,
    pub(crate) biases: ParamRange,
}

impl Conv1DLayer {
    pub fn new(
        params: &mut ParameterStore,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    ) -> Self {
        assert!(in_channels > 0);
        assert!(out_channels > 0);
        assert!(kernel_size > 0);
        assert!(stride > 0);

        let fan_in =
            in_channels * kernel_size;

        let std_dev =
            (2.0 / fan_in as f32).sqrt();

        let normal =
            NormalDist::new(
                0.0,
                std_dev,
            )
                .expect(
                    "Invalid standard deviation"
                );

        let mut rng =
            rand::rng();

        let weights =
            params.alloc_many(
                (0..out_channels * fan_in)
                    .map(|_| {
                        normal.sample(&mut rng)
                    }),
            );

        let biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.1f32,
                    out_channels,
                ),
            );

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
        }
    }

    #[inline]
    pub fn output_length(
        &self,
        input_length: usize,
    ) -> usize {
        if self.causal {
            assert!(input_length > 0);

            (input_length - 1)
                / self.stride
                + 1
        } else {
            assert!(
                input_length
                    + 2 * self.padding
                    >= self.kernel_size
            );

            (
                input_length
                    + 2 * self.padding
                    - self.kernel_size
            )
                / self.stride
                + 1
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Conv1D {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            causal: self.causal,
            activation: self.activation,
        }
    }
}

#[derive(Clone)]
pub struct DepthwiseConv1DLayer {
    pub(crate) in_channels: usize,
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) causal: bool,
    pub(crate) activation: Activation,

    pub(crate) weights: ParamRange,
    pub(crate) biases: ParamRange,
}

impl DepthwiseConv1DLayer {
    pub fn new(
        params: &mut ParameterStore,
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    ) -> Self {
        assert!(in_channels > 0);
        assert!(kernel_size > 0);
        assert!(stride > 0);

        let fan_in =
            kernel_size;

        let std_dev =
            (2.0 / fan_in as f32).sqrt();

        let normal =
            NormalDist::new(
                0.0,
                std_dev,
            )
                .expect(
                    "Invalid standard deviation"
                );

        let mut rng =
            rand::rng();

        let weights =
            params.alloc_many(
                (0..in_channels * kernel_size)
                    .map(|_| {
                        normal.sample(&mut rng)
                    }),
            );

        let biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.1f32,
                    in_channels,
                ),
            );

        Self {
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
        }
    }

    #[inline]
    pub fn output_length(
        &self,
        input_length: usize,
    ) -> usize {
        if self.causal {
            assert!(input_length > 0);

            (input_length - 1)
                / self.stride
                + 1
        } else {
            assert!(
                input_length
                    + 2 * self.padding
                    >= self.kernel_size
            );

            (
                input_length
                    + 2 * self.padding
                    - self.kernel_size
            )
                / self.stride
                + 1
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::DepthwiseConv1D {
            in_channels: self.in_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            causal: self.causal,
            activation: self.activation,
        }
    }
}

#[derive(Clone)]
pub struct GroupedConv1DLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub groups: usize,
    pub kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) causal: bool,
    pub(crate) activation: Activation,

    pub(crate) weights: ParamRange,
    pub(crate) biases: ParamRange,
}

impl GroupedConv1DLayer {
    pub fn new(
        params: &mut ParameterStore,
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        causal: bool,
        activation: Activation,
    ) -> Self {
        assert!(in_channels > 0);
        assert!(out_channels > 0);
        assert!(groups > 0);
        assert!(kernel_size > 0);
        assert!(stride > 0);

        assert_eq!(
            in_channels % groups,
            0,
            "in_channels must be divisible by groups"
        );

        assert_eq!(
            out_channels % groups,
            0,
            "out_channels must be divisible by groups"
        );

        let group_in =
            in_channels / groups;

        let fan_in =
            group_in * kernel_size;

        let std_dev =
            (2.0 / fan_in as f32).sqrt();

        let normal =
            NormalDist::new(
                0.0,
                std_dev,
            )
                .expect(
                    "Invalid standard deviation"
                );

        let mut rng =
            rand::rng();

        let weights =
            params.alloc_many(
                (0..out_channels * fan_in)
                    .map(|_| {
                        normal.sample(&mut rng)
                    }),
            );

        let biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.1f32,
                    out_channels,
                ),
            );

        Self {
            in_channels,
            out_channels,
            groups,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
        }
    }

    #[inline]
    pub(crate) fn output_length(
        &self,
        input_length: usize,
    ) -> usize {
        if self.causal {
            assert!(input_length > 0);

            (input_length - 1)
                / self.stride
                + 1
        } else {
            assert!(
                input_length
                    + 2 * self.padding
                    >= self.kernel_size
            );

            (
                input_length
                    + 2 * self.padding
                    - self.kernel_size
            )
                / self.stride
                + 1
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::GroupedConv1D {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            groups: self.groups,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            causal: self.causal,
            activation: self.activation,
        }
    }
}

#[derive(Clone)]
pub struct LowRankPointwiseLayer {
    pub(crate) in_channels: usize,
    pub(crate) rank: usize,
    pub(crate) out_channels: usize,
    pub(crate) activation: Activation,

    pub(crate) first_weights: ParamRange,
    pub(crate) first_biases: ParamRange,

    pub(crate) second_weights: ParamRange,
    pub(crate) second_biases: ParamRange,
}

impl LowRankPointwiseLayer {
    fn new(
        params: &mut ParameterStore,
        in_channels: usize,
        rank: usize,
        out_channels: usize,
        activation: Activation,
    ) -> Self {
        assert!(in_channels > 0);
        assert!(rank > 0);
        assert!(out_channels > 0);

        let mut rng =
            rand::rng();

        let first_std =
            (2.0 / in_channels as f32).sqrt();

        let second_std =
            (2.0 / rank as f32).sqrt();

        let first_normal =
            NormalDist::new(
                0.0,
                first_std,
            )
                .expect(
                    "Invalid standard deviation"
                );

        let second_normal =
            NormalDist::new(
                0.0,
                second_std,
            )
                .expect(
                    "Invalid standard deviation"
                );

        let first_weights =
            params.alloc_many(
                (0..rank * in_channels)
                    .map(|_| {
                        first_normal.sample(
                            &mut rng
                        )
                    }),
            );

        let first_biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.1f32,
                    rank,
                ),
            );

        let second_weights =
            params.alloc_many(
                (0..out_channels * rank)
                    .map(|_| {
                        second_normal.sample(
                            &mut rng
                        )
                    }),
            );

        let second_biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.1f32,
                    out_channels,
                ),
            );

        Self {
            in_channels,
            rank,
            out_channels,
            activation,

            first_weights,
            first_biases,

            second_weights,
            second_biases,
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::LowRankPointwise {
            in_channels: self.in_channels,
            rank: self.rank,
            out_channels: self.out_channels,
            activation: self.activation,
        }
    }
}

#[derive(Clone)]
pub struct ChannelScaleLayer {
    pub(crate) channels: usize,
    pub(crate) scales: ParamRange,
    pub(crate) biases: ParamRange,
}

impl ChannelScaleLayer {
    fn new(
        params: &mut ParameterStore,
        channels: usize,
    ) -> Self {
        assert!(channels > 0);

        let scales =
            params.alloc_many(
                std::iter::repeat_n(
                    1.0f32,
                    channels,
                ),
            );

        let biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.0f32,
                    channels,
                ),
            );

        Self {
            channels,
            scales,
            biases,
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::ChannelScale {
            channels: self.channels,
        }
    }
}

#[derive(Clone)]
pub struct WeightTyingLayer {
    pub(crate) embeddings: Arc<Embeddings>,
}

impl WeightTyingLayer {
    fn new(
        embeddings: Arc<Embeddings>,
    ) -> Self {
        assert!(
            embeddings.embedding_dim() > 0,
            "Weight tying requires a non-zero embedding dimension"
        );

        assert!(
            embeddings.vocab_size() > 0,
            "Weight tying requires a non-zero vocabulary size"
        );

        Self { embeddings }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::WeightTying
    }
}

#[derive(Clone)]
pub struct ResidualLayer {
    pub(crate) layers: Vec<Layer>,
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
}

impl ResidualLayer {
    fn new(
        input_size: usize,
        layers: Vec<Layer>,
        output_size: usize,
    ) -> Self {
        assert_eq!(
            input_size,
            output_size,
            "Residual input/output sizes must match"
        );

        Self {
            layers,
            input_size,
            output_size,
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Residual {
            layers: self.layers
                .iter()
                .map(|x| x.spec())
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct LayerNormLayer {
    pub(crate) channels: usize,
    pub(crate) epsilon: f32,

    pub(crate) gamma: ParamRange,
    pub(crate) beta: ParamRange,
}

impl LayerNormLayer {
    pub fn new(
        params: &mut ParameterStore,
        channels: usize,
        epsilon: f32,
    ) -> Self {
        assert!(
            channels > 0,
            "LayerNorm requires channels > 0",
        );

        assert!(
            epsilon > 0.0,
            "LayerNorm epsilon must be > 0",
        );

        let gamma =
            params.alloc_many(
                std::iter::repeat_n(
                    1.0f32,
                    channels,
                ),
            );

        let beta =
            params.alloc_many(
                std::iter::repeat_n(
                    0.0f32,
                    channels,
                ),
            );

        Self {
            channels,
            epsilon,
            gamma,
            beta,
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::LayerNorm {
            channels: self.channels,
            epsilon: self.epsilon,
        }
    }
}

#[derive(Clone)]
pub struct GlobalMixerLayer {
    pub(crate) channels: usize,
    pub(crate) global_dim: usize,

    pub(crate) write_weights: ParamRange,
    pub(crate) write_biases: ParamRange,

    pub(crate) read_weights: ParamRange,
    pub(crate) read_biases: ParamRange,

    pub(crate) write_positional_weights: ParamRange,
    pub(crate) read_positional_weights: ParamRange,
}

impl GlobalMixerLayer {
    fn new(
        params: &mut ParameterStore,
        channels: usize,
        global_dim: usize,
    ) -> Self {
        assert!(channels > 0);
        assert!(global_dim > 0);

        let mut rng =
            rand::rng();

        let scale =
            0.1f32
                / (channels as f32).sqrt();

        let dist =
            NormalDist::new(
                0.0,
                scale as f64,
            )
                .unwrap();

        let write_weights =
            params.alloc_many(
                (0..channels * global_dim)
                    .map(|_| {
                        dist.sample(&mut rng)
                            as f32
                    }),
            );

        let read_weights =
            params.alloc_many(
                (0..channels * global_dim)
                    .map(|_| {
                        dist.sample(&mut rng)
                            as f32
                    }),
            );

        let write_biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.0f32,
                    global_dim,
                ),
            );

        let read_biases =
            params.alloc_many(
                std::iter::repeat_n(
                    0.0f32,
                    global_dim,
                ),
            );

        const POS_FEATURES: usize = 4;

        let positional_count =
            global_dim * POS_FEATURES;

        let write_positional_weights =
            params.alloc_many(
                std::iter::repeat_n(
                    0.0f32,
                    positional_count,
                ),
            );

        let read_positional_weights =
            params.alloc_many(
                std::iter::repeat_n(
                    0.0f32,
                    positional_count,
                ),
            );

        Self {
            channels,
            global_dim,

            write_weights,
            write_biases,

            read_weights,
            read_biases,

            write_positional_weights,
            read_positional_weights,
        }
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::GlobalMixer {
            channels: self.channels,
            global_dim: self.global_dim,
        }
    }
}

#[derive(Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Conv1D(Conv1DLayer),
    Residual(ResidualLayer),
    DepthwiseConv1D(DepthwiseConv1DLayer),
    GroupedConv1D(GroupedConv1DLayer),
    LowRankPointwise(LowRankPointwiseLayer),
    ChannelScale(ChannelScaleLayer),
    LayerNorm(LayerNormLayer),
    WeightTying(WeightTyingLayer),
    GlobalMixer(GlobalMixerLayer),
}

impl Layer {
    fn spec(&self) -> LayerSpec {
        match self {
            Layer::Dense(layer) =>
                layer.spec(),

            Layer::Conv1D(layer) =>
                layer.spec(),

            Layer::Residual(layer) =>
                layer.spec(),

            Layer::DepthwiseConv1D(layer) =>
                layer.spec(),

            Layer::GroupedConv1D(layer) =>
                layer.spec(),

            Layer::LowRankPointwise(layer) =>
                layer.spec(),

            Layer::ChannelScale(layer) =>
                layer.spec(),

            Layer::LayerNorm(layer) =>
                layer.spec(),

            Layer::WeightTying(layer) =>
                layer.spec(),

            Layer::GlobalMixer(layer) =>
                layer.spec(),
        }
    }
}

fn build_layers(
    current_size: usize,
    specs: &[LayerSpec],
    embeddings: Option<&Arc<Embeddings>>,
    params: &mut ParameterStore,
) -> (Vec<Layer>, usize) {
    let mut layers =
        Vec::with_capacity(
            specs.len()
        );

    let mut current_size =
        current_size;

    for spec in specs {
        let layer =
            match spec {
                LayerSpec::Dense {
                    output_size,
                    activation,
                } => {
                    Layer::Dense(
                        DenseLayer::new(
                            params,
                            current_size,
                            *output_size,
                            *activation,
                        )
                    )
                }

                LayerSpec::Conv1D {
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                    activation,
                } => {
                    assert_eq!(
                        current_size
                            % in_channels,
                        0
                    );

                    Layer::Conv1D(
                        Conv1DLayer::new(
                            params,
                            *in_channels,
                            *out_channels,
                            *kernel_size,
                            *stride,
                            *padding,
                            *causal,
                            *activation,
                        )
                    )
                }

                LayerSpec::Residual {
                    layers: inner_specs,
                } => {
                    let residual_input_size =
                        current_size;

                    let (
                        inner_layers,
                        inner_output_size,
                    ) =
                        build_layers(
                            residual_input_size,
                            inner_specs,
                            embeddings,
                            params,
                        );

                    assert_eq!(
                        residual_input_size,
                        inner_output_size,
                        "Residual block changed tensor size"
                    );

                    Layer::Residual(
                        ResidualLayer::new(
                            residual_input_size,
                            inner_layers,
                            inner_output_size,
                        )
                    )
                }

                LayerSpec::DepthwiseConv1D {
                    in_channels,
                    kernel_size,
                    stride,
                    padding,
                    causal,
                    activation,
                } => {
                    assert_eq!(
                        current_size
                            % in_channels,
                        0
                    );

                    Layer::DepthwiseConv1D(
                        DepthwiseConv1DLayer::new(
                            params,
                            *in_channels,
                            *kernel_size,
                            *stride,
                            *padding,
                            *causal,
                            *activation,
                        )
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
                    activation,
                } => {
                    assert_eq!(
                        current_size
                            % in_channels,
                        0
                    );

                    Layer::GroupedConv1D(
                        GroupedConv1DLayer::new(
                            params,
                            *in_channels,
                            *out_channels,
                            *groups,
                            *kernel_size,
                            *stride,
                            *padding,
                            *causal,
                            *activation,
                        )
                    )
                }

                LayerSpec::LowRankPointwise {
                    in_channels,
                    rank,
                    out_channels,
                    activation,
                } => {
                    assert_eq!(
                        current_size
                            % in_channels,
                        0
                    );

                    Layer::LowRankPointwise(
                        LowRankPointwiseLayer::new(
                            params,
                            *in_channels,
                            *rank,
                            *out_channels,
                            *activation,
                        )
                    )
                }

                LayerSpec::ChannelScale {
                    channels,
                } => {
                    assert_eq!(
                        current_size
                            % channels,
                        0
                    );

                    Layer::ChannelScale(
                        ChannelScaleLayer::new(
                            params,
                            *channels,
                        )
                    )
                }

                LayerSpec::WeightTying => {
                    let embeddings =
                        embeddings.expect(
                            "WeightTying requires an Embeddings instance"
                        );

                    assert_eq!(
                        current_size,
                        embeddings.embedding_dim(),
                        "WeightTying input size must equal embedding dimension"
                    );

                    Layer::WeightTying(
                        WeightTyingLayer::new(
                            Arc::clone(
                                embeddings
                            )
                        )
                    )
                }

                LayerSpec::LayerNorm {
                    channels,
                    epsilon,
                } => {
                    assert_eq!(
                        current_size
                            % channels,
                        0,
                        "LayerNorm channels must divide layer size"
                    );

                    Layer::LayerNorm(
                        LayerNormLayer::new(
                            params,
                            *channels,
                            *epsilon,
                        )
                    )
                }

                LayerSpec::GlobalMixer {
                    channels,
                    global_dim,
                } => {
                    assert!(*channels > 0);
                    assert!(*global_dim > 0);

                    assert_eq!(
                        current_size
                            % channels,
                        0,
                        "GlobalMixer channels must divide layer size"
                    );

                    Layer::GlobalMixer(
                        GlobalMixerLayer::new(
                            params,
                            *channels,
                            *global_dim,
                        )
                    )
                }
            };

        current_size =
            layer_output_size(
                &layer,
                current_size,
            );

        layers.push(layer);
    }

    (layers, current_size)
}

fn layer_output_size(
    layer: &Layer,
    current_size: usize,
) -> usize {
    match layer {
        Layer::Dense(layer) =>
            layer.output_size,

        Layer::Conv1D(layer) => {
            let input_length =
                current_size
                    / layer.in_channels;

            layer.output_length(
                input_length
            )
                * layer.out_channels
        }

        Layer::Residual(layer) =>
            layer.output_size,

        Layer::DepthwiseConv1D(layer) => {
            let input_length =
                current_size
                    / layer.in_channels;

            layer.output_length(
                input_length
            )
                * layer.in_channels
        }

        Layer::GroupedConv1D(layer) => {
            let input_length =
                current_size
                    / layer.in_channels;

            layer.output_length(
                input_length
            )
                * layer.out_channels
        }

        Layer::LowRankPointwise(layer) => {
            let length =
                current_size
                    / layer.in_channels;

            length
                * layer.out_channels
        }

        Layer::ChannelScale(layer) => {
            assert_eq!(
                current_size
                    % layer.channels,
                0
            );

            current_size
        }

        Layer::LayerNorm(layer) => {
            assert_eq!(
                current_size
                    % layer.channels,
                0
            );

            current_size
        }

        Layer::GlobalMixer(layer) => {
            assert_eq!(
                current_size
                    % layer.channels,
                0
            );

            current_size
        }

        Layer::WeightTying(layer) =>
            layer.embeddings.vocab_size(),
    }
}

fn build_thread_pool(
    num_threads: usize,
) -> Arc<ThreadPool> {
    assert!(
        num_threads > 0,
        "MLP thread count must be greater than zero"
    );

    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect(
                "failed to build Rayon thread pool"
            )
    )
}

fn save_layer(
    layer: &Layer,
    params: &ParameterStore,
) -> SavedLayer {
    match layer {
        Layer::Dense(layer) => SavedLayer::Dense {
            input_size: layer.input_size,
            output_size: layer.output_size,
            activation: layer.activation,
            weights: params.values[layer.weights.start
                ..layer.weights.start + layer.weights.len]
                .to_vec(),
            biases: params.values[layer.biases.start
                ..layer.biases.start + layer.biases.len]
                .to_vec(),
        },

        Layer::Conv1D(layer) => SavedLayer::Conv1D {
            in_channels: layer.in_channels,
            out_channels: layer.out_channels,
            kernel_size: layer.kernel_size,
            stride: layer.stride,
            padding: layer.padding,
            causal: layer.causal,
            activation: layer.activation,
            weights: params.values[layer.weights.start
                ..layer.weights.start + layer.weights.len]
                .to_vec(),
            biases: params.values[layer.biases.start
                ..layer.biases.start + layer.biases.len]
                .to_vec(),
        },

        Layer::Residual(layer) => SavedLayer::Residual {
            input_size: layer.input_size,
            output_size: layer.output_size,
            layers: layer.layers
                .iter()
                .map(|x| save_layer(x, params))
                .collect(),
        },

        Layer::DepthwiseConv1D(layer) => SavedLayer::DepthwiseConv1D {
            in_channels: layer.in_channels,
            kernel_size: layer.kernel_size,
            stride: layer.stride,
            padding: layer.padding,
            causal: layer.causal,
            activation: layer.activation,
            weights: params.values[layer.weights.start
                ..layer.weights.start + layer.weights.len]
                .to_vec(),
            biases: params.values[layer.biases.start
                ..layer.biases.start + layer.biases.len]
                .to_vec(),
        },

        Layer::GroupedConv1D(layer) => SavedLayer::GroupedConv1D {
            in_channels: layer.in_channels,
            out_channels: layer.out_channels,
            groups: layer.groups,
            kernel_size: layer.kernel_size,
            stride: layer.stride,
            padding: layer.padding,
            causal: layer.causal,
            activation: layer.activation,
            weights: params.values[layer.weights.start
                ..layer.weights.start + layer.weights.len]
                .to_vec(),
            biases: params.values[layer.biases.start
                ..layer.biases.start + layer.biases.len]
                .to_vec(),
        },

        Layer::LowRankPointwise(layer) => {
            SavedLayer::LowRankPointwise {
                in_channels: layer.in_channels,
                rank: layer.rank,
                out_channels: layer.out_channels,
                activation: layer.activation,

                first_weights:
                params.values[
                    layer.first_weights.start
                        ..layer.first_weights.start
                        + layer.first_weights.len
                    ]
                    .to_vec(),

                first_biases:
                params.values[
                    layer.first_biases.start
                        ..layer.first_biases.start
                        + layer.first_biases.len
                    ]
                    .to_vec(),

                second_weights:
                params.values[
                    layer.second_weights.start
                        ..layer.second_weights.start
                        + layer.second_weights.len
                    ]
                    .to_vec(),

                second_biases:
                params.values[
                    layer.second_biases.start
                        ..layer.second_biases.start
                        + layer.second_biases.len
                    ]
                    .to_vec(),
            }
        }

        Layer::ChannelScale(layer) => {
            SavedLayer::ChannelScale {
                channels: layer.channels,

                scales:
                params.values[
                    layer.scales.start
                        ..layer.scales.start
                        + layer.scales.len
                    ]
                    .to_vec(),

                biases:
                params.values[
                    layer.biases.start
                        ..layer.biases.start
                        + layer.biases.len
                    ]
                    .to_vec(),
            }
        }

        Layer::WeightTying(layer) => {
            SavedLayer::WeightTying {
                embedding_dim:
                layer.embeddings.embedding_dim(),
                vocab_size:
                layer.embeddings.vocab_size(),
            }
        }

        Layer::LayerNorm(layer) => {
            SavedLayer::LayerNorm {
                channels: layer.channels,
                epsilon: layer.epsilon,

                gamma:
                params.values[
                    layer.gamma.start
                        ..layer.gamma.start
                        + layer.gamma.len
                    ]
                    .to_vec(),

                beta:
                params.values[
                    layer.beta.start
                        ..layer.beta.start
                        + layer.beta.len
                    ]
                    .to_vec(),
            }
        }

        Layer::GlobalMixer(layer) => {
            SavedLayer::GlobalMixer {
                channels: layer.channels,
                global_dim: layer.global_dim,

                write_weights:
                params.values[
                    layer.write_weights.start
                        ..layer.write_weights.start
                        + layer.write_weights.len
                    ]
                    .to_vec(),

                write_biases:
                params.values[
                    layer.write_biases.start
                        ..layer.write_biases.start
                        + layer.write_biases.len
                    ]
                    .to_vec(),

                read_weights:
                params.values[
                    layer.read_weights.start
                        ..layer.read_weights.start
                        + layer.read_weights.len
                    ]
                    .to_vec(),

                read_biases:
                params.values[
                    layer.read_biases.start
                        ..layer.read_biases.start
                        + layer.read_biases.len
                    ]
                    .to_vec(),

                write_positional_weights:
                params.values[
                    layer.write_positional_weights.start
                        ..layer.write_positional_weights.start
                        + layer.write_positional_weights.len
                    ]
                    .to_vec(),

                read_positional_weights:
                params.values[
                    layer.read_positional_weights.start
                        ..layer.read_positional_weights.start
                        + layer.read_positional_weights.len
                    ]
                    .to_vec(),
            }
        }
    }
}

fn load_layer(
    saved: &SavedLayer,
    embeddings: &Arc<Embeddings>,
    params: &mut ParameterStore,
) -> Layer {
    match saved {
        SavedLayer::Dense {
            input_size,
            output_size,
            activation,
            weights,
            biases,
        } => {
            assert_eq!(
                weights.len(),
                input_size * output_size
            );
            assert_eq!(
                biases.len(),
                *output_size
            );

            let weight_range =
                params.alloc_many(
                    weights.iter().copied()
                );

            let bias_range =
                params.alloc_many(
                    biases.iter().copied()
                );

            Layer::Dense(DenseLayer {
                input_size: *input_size,
                output_size: *output_size,
                weights: weight_range,
                biases: bias_range,
                activation: *activation,
            })
        }

        SavedLayer::Conv1D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
        } => {
            assert_eq!(
                weights.len(),
                in_channels
                    * out_channels
                    * kernel_size
            );
            assert_eq!(
                biases.len(),
                *out_channels
            );

            Layer::Conv1D(
                Conv1DLayer {
                    in_channels: *in_channels,
                    out_channels: *out_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    causal: *causal,
                    activation: *activation,

                    weights:
                    params.alloc_many(
                        weights.iter().copied()
                    ),

                    biases:
                    params.alloc_many(
                        biases.iter().copied()
                    ),
                }
            )
        }

        SavedLayer::Residual {
            input_size,
            output_size,
            layers,
        } => {
            let inner =
                layers
                    .iter()
                    .map(|x|
                        load_layer(
                            x,
                            embeddings,
                            params,
                        )
                    )
                    .collect::<Vec<_>>();

            Layer::Residual(
                ResidualLayer::new(
                    *input_size,
                    inner,
                    *output_size,
                )
            )
        }

        SavedLayer::DepthwiseConv1D {
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
        } => {
            assert_eq!(
                weights.len(),
                in_channels * kernel_size
            );
            assert_eq!(
                biases.len(),
                *in_channels
            );

            Layer::DepthwiseConv1D(
                DepthwiseConv1DLayer {
                    in_channels: *in_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    causal: *causal,
                    activation: *activation,

                    weights:
                    params.alloc_many(
                        weights.iter().copied()
                    ),

                    biases:
                    params.alloc_many(
                        biases.iter().copied()
                    ),
                }
            )
        }

        SavedLayer::GroupedConv1D {
            in_channels,
            out_channels,
            groups,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
        } => {
            let group_in =
                in_channels / groups;

            assert_eq!(
                weights.len(),
                out_channels
                    * group_in
                    * kernel_size
            );

            assert_eq!(
                biases.len(),
                *out_channels
            );

            Layer::GroupedConv1D(
                GroupedConv1DLayer {
                    in_channels: *in_channels,
                    out_channels: *out_channels,
                    groups: *groups,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    causal: *causal,
                    activation: *activation,

                    weights:
                    params.alloc_many(
                        weights.iter().copied()
                    ),

                    biases:
                    params.alloc_many(
                        biases.iter().copied()
                    ),
                }
            )
        }

        SavedLayer::LowRankPointwise {
            in_channels,
            rank,
            out_channels,
            activation,
            first_weights,
            first_biases,
            second_weights,
            second_biases,
        } => {
            assert_eq!(
                first_weights.len(),
                rank * in_channels
            );
            assert_eq!(
                first_biases.len(),
                *rank
            );
            assert_eq!(
                second_weights.len(),
                out_channels * rank
            );
            assert_eq!(
                second_biases.len(),
                *out_channels
            );

            Layer::LowRankPointwise(
                LowRankPointwiseLayer {
                    in_channels: *in_channels,
                    rank: *rank,
                    out_channels: *out_channels,
                    activation: *activation,

                    first_weights:
                    params.alloc_many(
                        first_weights
                            .iter()
                            .copied()
                    ),

                    first_biases:
                    params.alloc_many(
                        first_biases
                            .iter()
                            .copied()
                    ),

                    second_weights:
                    params.alloc_many(
                        second_weights
                            .iter()
                            .copied()
                    ),

                    second_biases:
                    params.alloc_many(
                        second_biases
                            .iter()
                            .copied()
                    ),
                }
            )
        }

        SavedLayer::ChannelScale {
            channels,
            scales,
            biases,
        } => {
            assert_eq!(
                scales.len(),
                *channels
            );
            assert_eq!(
                biases.len(),
                *channels
            );

            Layer::ChannelScale(
                ChannelScaleLayer {
                    channels: *channels,

                    scales:
                    params.alloc_many(
                        scales.iter().copied()
                    ),

                    biases:
                    params.alloc_many(
                        biases.iter().copied()
                    ),
                }
            )
        }

        SavedLayer::WeightTying {
            embedding_dim,
            vocab_size,
        } => {
            assert_eq!(
                *embedding_dim,
                embeddings.embedding_dim()
            );
            assert_eq!(
                *vocab_size,
                embeddings.vocab_size()
            );

            Layer::WeightTying(
                WeightTyingLayer::new(
                    Arc::clone(embeddings)
                )
            )
        }

        SavedLayer::LayerNorm {
            channels,
            epsilon,
            gamma,
            beta,
        } => {
            assert_eq!(
                gamma.len(),
                *channels
            );
            assert_eq!(
                beta.len(),
                *channels
            );

            Layer::LayerNorm(
                LayerNormLayer {
                    channels: *channels,
                    epsilon: *epsilon,

                    gamma:
                    params.alloc_many(
                        gamma.iter().copied()
                    ),

                    beta:
                    params.alloc_many(
                        beta.iter().copied()
                    ),
                }
            )
        }

        SavedLayer::GlobalMixer {
            channels,
            global_dim,
            write_weights,
            write_biases,
            read_weights,
            read_biases,
            write_positional_weights,
            read_positional_weights,
        } => {
            assert_eq!(
                write_weights.len(),
                channels * global_dim
            );
            assert_eq!(
                write_biases.len(),
                *global_dim
            );
            assert_eq!(
                read_weights.len(),
                channels * global_dim
            );
            assert_eq!(
                read_biases.len(),
                *global_dim
            );

            const POS_FEATURES: usize = 4;

            assert_eq!(
                write_positional_weights.len(),
                global_dim * POS_FEATURES
            );

            assert_eq!(
                read_positional_weights.len(),
                global_dim * POS_FEATURES
            );

            Layer::GlobalMixer(
                GlobalMixerLayer {
                    channels: *channels,
                    global_dim: *global_dim,

                    write_weights:
                    params.alloc_many(
                        write_weights
                            .iter()
                            .copied()
                    ),

                    write_biases:
                    params.alloc_many(
                        write_biases
                            .iter()
                            .copied()
                    ),

                    read_weights:
                    params.alloc_many(
                        read_weights
                            .iter()
                            .copied()
                    ),

                    read_biases:
                    params.alloc_many(
                        read_biases
                            .iter()
                            .copied()
                    ),

                    write_positional_weights:
                    params.alloc_many(
                        write_positional_weights
                            .iter()
                            .copied()
                    ),

                    read_positional_weights:
                    params.alloc_many(
                        read_positional_weights
                            .iter()
                            .copied()
                    ),
                }
            )
        }
    }
}

pub struct MLP {
    pub(crate) layers: Vec<Layer>,
    pub(crate) params: ParameterStore,
    thread_pool: Arc<ThreadPool>,
}

impl MLP {
    pub fn new(
        num_inputs: usize,
        layer_sizes: &[usize],
    ) -> Self {
        let specs =
            layer_sizes
                .iter()
                .enumerate()
                .map(
                    |(i, &output_size)| {
                        LayerSpec::Dense {
                            output_size,
                            activation:
                            if i + 1
                                == layer_sizes.len()
                            {
                                Activation::None
                            } else {
                                Activation::LeakyReLU {
                                    slope: 0.01,
                                }
                            },
                        }
                    },
                )
                .collect::<Vec<_>>();

        Self::from_layers(
            num_inputs,
            &specs,
        )
    }

    pub fn set_num_threads(
        &mut self,
        num_threads: usize,
    ) {
        self.thread_pool =
            build_thread_pool(
                num_threads
            );
    }

    pub fn from_layers(
        num_inputs: usize,
        specs: &[LayerSpec],
    ) -> Self {
        let mut params =
            ParameterStore::new();

        let (layers, _) =
            build_layers(
                num_inputs,
                specs,
                None,
                &mut params,
            );

        Self {
            layers,
            params,
            thread_pool:
            build_thread_pool(1),
        }
    }

    pub fn from_layers_with_embeddings(
        num_inputs: usize,
        specs: &[LayerSpec],
        embeddings: Arc<Embeddings>,
        mut params: ParameterStore,
    ) -> Self {
        let (layers, _) =
            build_layers(
                num_inputs,
                specs,
                Some(&embeddings),
                &mut params,
            );

        Self {
            layers,
            params,
            thread_pool:
            build_thread_pool(1),
        }
    }

    pub(crate) fn forward_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        input_size: usize,
    ) -> BatchForward {
        debug_assert_eq!(
            input.len(),
            batch_size * input_size
        );

        let (
            output,
            output_size,
            layers,
        ) =
            forward_layers_batch(
                &self.layers,
                &self.params,
                input,
                batch_size,
                input_size,
                &self.thread_pool,
            );

        BatchForward {
            output,
            batch_size,
            output_size,
            layers,
        }
    }

    pub(crate) fn backward_batch(
        &mut self,
        forward: &BatchForward,
        output_grads: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(
            output_grads.len(),
            forward.batch_size
                * forward.output_size
        );

        backward_layers_batch(
            &self.layers,
            &forward.layers,
            &mut self.params,
            output_grads.to_vec(),
            forward.batch_size,
        )
    }

    pub fn parameter_count(
        &self,
    ) -> usize {
        self.params.parameter_count()
    }

    pub fn layer_specs(
        &self,
    ) -> Vec<LayerSpec> {
        self.layers
            .iter()
            .map(|layer| layer.spec())
            .collect()
    }

    pub fn save(&self) -> SavedMLP {
        SavedMLP {
            version: 3,
            layers: self.layers
                .iter()
                .map(|layer| save_layer(layer, &self.params))
                .collect(),
        }
    }

    pub fn load_with_embeddings(
        saved: &SavedMLP,
        embeddings: Arc<Embeddings>,
        mut params: ParameterStore,
    ) -> Self {
        assert_eq!(
            saved.version,
            3,
            "Unsupported SavedMLP version"
        );

        let mut layers = Vec::with_capacity(
            saved.layers.len()
        );

        let mut current_size =
            embeddings.embedding_dim();

        for saved_layer in &saved.layers {
            let layer =
                load_layer(
                    saved_layer,
                    &embeddings,
                    &mut params,
                );

            current_size =
                layer_output_size(
                    &layer,
                    current_size,
                );

            layers.push(layer);
        }

        Self {
            layers,
            params,
            thread_pool:
            build_thread_pool(1),
        }
    }

    pub fn parameters(&self) -> ParameterStore {
        self.params.clone()
    }
}
