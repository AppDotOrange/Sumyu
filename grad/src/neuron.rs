use std::sync::Arc;
use rand_distr::{Distribution, Normal as NormalDist};
use serde::{Deserialize, Serialize};
use crate::{Tensor, TensorHandle, embeddings::Embeddings};
pub use crate::{backwards::*, forwards::*};

#[derive(Clone, Serialize, Deserialize, Copy)]
pub enum Activation {
    None,
    LeakyReLU { slope: f32 },
}
impl Activation {
    #[inline]
    pub(crate) fn apply(&self, x: f32) -> f32 {
        match self {
            Self::None => x,
            Self::LeakyReLU { slope } => if x <= 0.0 { x * slope } else { x },
        }
    }
    #[inline]
    pub fn backward(&self, x: f32, grad: &mut f32) {
        if let Self::LeakyReLU { slope } = self {
            if x <= 0.0 {
                *grad *= *slope;
            }
        }
    }
}
#[derive(Serialize, Deserialize)]
pub struct SavedMLP {
    version: u32,
    layers: Vec<SavedLayer>,
}
#[derive(Serialize, Deserialize)]
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
        weights: Vec<f32>,
        weight_handles: Arc<[TensorHandle]>,
        bias_handles: Arc<[TensorHandle]>,
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
        weight_handles: Arc<[TensorHandle]>,
        bias_handles: Arc<[TensorHandle]>,
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
        weight_handles: Arc<[TensorHandle]>,
        bias_handles: Arc<[TensorHandle]>,
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
        weight_handles: Arc<[TensorHandle]>,
        bias_handles: Arc<[TensorHandle]>,
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

        first_weight_handles: Arc<[TensorHandle]>,
        first_bias_handles: Arc<[TensorHandle]>,
        second_weight_handles: Arc<[TensorHandle]>,
        second_bias_handles: Arc<[TensorHandle]>,
        activation: Activation,
    },
    ChannelScale {
        input: Vec<f32>,
        channels: usize,
        scale_handles: Arc<[TensorHandle]>,
        bias_handles: Arc<[TensorHandle]>,
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
        gamma_handles: Arc<[TensorHandle]>,
        beta_handles: Arc<[TensorHandle]>,
    },
    GlobalMixer {
        input: Vec<f32>,

        write_probs: Vec<f32>,
        global_vectors: Vec<f32>,
        read_probs: Vec<f32>,

        positions: usize,
        channels: usize,
        global_dim: usize,

        write_weight_handles: Arc<[TensorHandle]>,
        write_bias_handles: Arc<[TensorHandle]>,

        read_weight_handles: Arc<[TensorHandle]>,
        read_bias_handles: Arc<[TensorHandle]>,
    },
}

#[derive(Clone)]
pub struct DenseLayer {
    pub(crate) weights: Vec<Tensor>,
    pub(crate) biases: Vec<Tensor>,
    pub(crate) fused_weights: Arc<[TensorHandle]>,
    pub(crate) fused_biases: Arc<[TensorHandle]>,
    pub(crate) activation: Activation,
}

impl DenseLayer {
    fn new(num_inputs: usize, num_outputs: usize, activation: Activation) -> Self {
        let mut rng = rand::rng();
        let std_dev = (2.0 / num_inputs as f32).sqrt();
        let normal =
            NormalDist::new(0.0, std_dev).expect("Invalid standard deviation");

        let weights = (0..num_inputs * num_outputs)
            .map(|_| Tensor::new(normal.sample(&mut rng)))
            .collect::<Vec<_>>();

        let biases = (0..num_outputs)
            .map(|_| Tensor::new(0.1))
            .collect::<Vec<_>>();

        let fused_weights = weights.iter().map(|x| x.handle).collect();
        let fused_biases = biases.iter().map(|x| x.handle).collect();

        Self {
            weights,
            biases,
            fused_weights,
            fused_biases,
            activation,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.weights.clone();
        params.extend_from_slice(&self.biases);
        params
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Dense {
            output_size: self.fused_biases.len(),
            activation: self.activation.clone(),
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
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    pub weight_handles: Arc<[TensorHandle]>,
    pub(crate) bias_handles: Arc<[TensorHandle]>,
}

impl Conv1DLayer {
    pub fn new(
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

        let fan_in = in_channels * kernel_size;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let normal = NormalDist::new(0.0, std_dev).expect("Invalid standard deviation");
        let mut rng = rand::rng();

        let weights = (0..out_channels * fan_in)
            .map(|_| Tensor::new(normal.sample(&mut rng)))
            .collect::<Vec<_>>();

        let biases = (0..out_channels)
            .map(|_| Tensor::new(0.1))
            .collect::<Vec<_>>();

        let weight_handles = weights.iter().map(|x| x.handle).collect();
        let bias_handles = biases.iter().map(|x| x.handle).collect();

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
            weight_handles,
            bias_handles,
        }
    }

    #[inline]
    pub fn output_length(&self, input_length: usize) -> usize {
        if self.causal {
            // Causal convolution pads only on the left by kernel_size - 1.
            // Output positions occur at 0, stride, 2*stride, ...
            assert!(input_length > 0);
            (input_length - 1) / self.stride + 1
        } else {
            assert!(input_length + 2 * self.padding >= self.kernel_size);
            (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.weights.clone();
        params.extend_from_slice(&self.biases);
        params
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Conv1D {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            causal: self.causal,
            activation: self.activation.clone(),
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

    weights: Vec<Tensor>,
    biases: Vec<Tensor>,

    pub(crate) weight_handles: Arc<[TensorHandle]>,
    pub(crate) bias_handles: Arc<[TensorHandle]>,
}

impl DepthwiseConv1DLayer {
    pub fn new(
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

        let fan_in = kernel_size;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let normal =
            NormalDist::new(0.0, std_dev).expect("Invalid standard deviation");
        let mut rng = rand::rng();

        let weights = (0..in_channels * kernel_size)
            .map(|_| Tensor::new(normal.sample(&mut rng)))
            .collect::<Vec<_>>();

        let biases = (0..in_channels)
            .map(|_| Tensor::new(0.1))
            .collect::<Vec<_>>();

        let weight_handles = weights.iter().map(|x| x.handle).collect();
        let bias_handles = biases.iter().map(|x| x.handle).collect();

        Self {
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            activation,
            weights,
            biases,
            weight_handles,
            bias_handles,
        }
    }

    #[inline]
    pub fn output_length(&self, input_length: usize) -> usize {
        if self.causal {
            assert!(input_length > 0);
            (input_length - 1) / self.stride + 1
        } else {
            assert!(input_length + 2 * self.padding >= self.kernel_size);
            (input_length + 2 * self.padding - self.kernel_size)
                / self.stride
                + 1
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.weights.clone();
        params.extend_from_slice(&self.biases);
        params
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::DepthwiseConv1D {
            in_channels: self.in_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            causal: self.causal,
            activation: self.activation.clone(),
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

    weights: Vec<Tensor>,
    biases: Vec<Tensor>,

    pub weight_handles: Arc<[TensorHandle]>,
    pub bias_handles: Arc<[TensorHandle]>,
}

impl GroupedConv1DLayer {
    pub fn new(
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

        let group_in = in_channels / groups;
        let fan_in = group_in * kernel_size;

        let std_dev = (2.0 / fan_in as f32).sqrt();
        let normal =
            NormalDist::new(0.0, std_dev).expect("Invalid standard deviation");
        let mut rng = rand::rng();

        // Layout is:
        // [out_channel][kernel][group_input_channel]
        let weights = (0..out_channels * fan_in)
            .map(|_| Tensor::new(normal.sample(&mut rng)))
            .collect::<Vec<_>>();

        let biases = (0..out_channels)
            .map(|_| Tensor::new(0.1))
            .collect::<Vec<_>>();

        let weight_handles = weights.iter().map(|x| x.handle).collect();
        let bias_handles = biases.iter().map(|x| x.handle).collect();

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
            weight_handles,
            bias_handles,
        }
    }

    #[inline]
    pub(crate) fn output_length(&self, input_length: usize) -> usize {
        if self.causal {
            assert!(input_length > 0);
            (input_length - 1) / self.stride + 1
        } else {
            assert!(input_length + 2 * self.padding >= self.kernel_size);
            (input_length + 2 * self.padding - self.kernel_size)
                / self.stride
                + 1
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.weights.clone();
        params.extend_from_slice(&self.biases);
        params
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
            activation: self.activation.clone(),
        }
    }
}

#[derive(Clone)]
pub struct LowRankPointwiseLayer {
    pub(crate) in_channels: usize,
    pub(crate) rank: usize,
    pub(crate) out_channels: usize,
    pub(crate) activation: Activation,

    first_weights: Vec<Tensor>,
    first_biases: Vec<Tensor>,

    second_weights: Vec<Tensor>,
    second_biases: Vec<Tensor>,

    pub(crate) first_weight_handles: Arc<[TensorHandle]>,
    pub(crate) first_bias_handles: Arc<[TensorHandle]>,
    pub(crate) second_weight_handles: Arc<[TensorHandle]>,
    pub(crate) second_bias_handles: Arc<[TensorHandle]>,
}

impl LowRankPointwiseLayer {
    fn new(
        in_channels: usize,
        rank: usize,
        out_channels: usize,
        activation: Activation,
    ) -> Self {
        assert!(in_channels > 0);
        assert!(rank > 0);
        assert!(out_channels > 0);

        let mut rng = rand::rng();

        let first_std = (2.0 / in_channels as f32).sqrt();
        let second_std = (2.0 / rank as f32).sqrt();

        let first_normal =
            NormalDist::new(0.0, first_std).expect("Invalid standard deviation");
        let second_normal =
            NormalDist::new(0.0, second_std).expect("Invalid standard deviation");

        let first_weights = (0..rank * in_channels)
            .map(|_| Tensor::new(first_normal.sample(&mut rng)))
            .collect::<Vec<_>>();

        let first_biases = (0..rank)
            .map(|_| Tensor::new(0.1))
            .collect::<Vec<_>>();

        let second_weights = (0..out_channels * rank)
            .map(|_| Tensor::new(second_normal.sample(&mut rng)))
            .collect::<Vec<_>>();

        let second_biases = (0..out_channels)
            .map(|_| Tensor::new(0.1))
            .collect::<Vec<_>>();

        let first_weight_handles =
            first_weights.iter().map(|x| x.handle).collect();

        let first_bias_handles =
            first_biases.iter().map(|x| x.handle).collect();

        let second_weight_handles =
            second_weights.iter().map(|x| x.handle).collect();

        let second_bias_handles =
            second_biases.iter().map(|x| x.handle).collect();

        Self {
            in_channels,
            rank,
            out_channels,
            activation,

            first_weights,
            first_biases,
            second_weights,
            second_biases,

            first_weight_handles,
            first_bias_handles,
            second_weight_handles,
            second_bias_handles,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        params.extend_from_slice(&self.first_weights);
        params.extend_from_slice(&self.first_biases);
        params.extend_from_slice(&self.second_weights);
        params.extend_from_slice(&self.second_biases);

        params
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::LowRankPointwise {
            in_channels: self.in_channels,
            rank: self.rank,
            out_channels: self.out_channels,
            activation: self.activation.clone(),
        }
    }
}

#[derive(Clone)]
pub struct ChannelScaleLayer {
    pub(crate) channels: usize,
    scales: Vec<Tensor>,
    biases: Vec<Tensor>,
    pub(crate) scale_handles: Arc<[TensorHandle]>,
    pub(crate) bias_handles: Arc<[TensorHandle]>,
}

impl ChannelScaleLayer {
    fn new(channels: usize) -> Self {
        assert!(channels > 0);

        let scales = (0..channels)
            .map(|_| Tensor::new(1.0))
            .collect::<Vec<_>>();

        let biases = (0..channels)
            .map(|_| Tensor::new(0.0))
            .collect::<Vec<_>>();

        let scale_handles = scales.iter().map(|x| x.handle).collect();
        let bias_handles = biases.iter().map(|x| x.handle).collect();

        Self {
            channels,
            scales,
            biases,
            scale_handles,
            bias_handles,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.scales.clone();
        params.extend_from_slice(&self.biases);
        params
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
    fn new(embeddings: Arc<Embeddings>) -> Self {
        assert!(
            embeddings.embedding_dim() > 0,
            "Weight tying requires a non-zero embedding dimension"
        );

        assert!(
            embeddings.vocab_size() > 0,
            "Weight tying requires a non-zero vocabulary size"
        );

        Self {
            embeddings,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        // The embedding parameters are owned by Embeddings.
        // Returning them here would double-count the parameters.
        Vec::new()
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::WeightTying
    }
}

#[derive(Clone)]
pub struct ResidualLayer {
    pub(crate) layers: Vec<Layer>,
    pub(crate) input_size: usize,
    output_size: usize,
}

impl ResidualLayer {
    fn new(
        input_size: usize,
        layers: Vec<Layer>,
        output_size: usize,
    ) -> Self {
        assert_eq!(
            input_size, output_size,
            "Residual input/output sizes must match"
        );

        Self {
            layers,
            input_size,
            output_size,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for layer in &self.layers {
            params.extend(layer.parameters());
        }

        params
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Residual {
            layers: self.layers.iter().map(|x| x.spec()).collect(),
        }
    }
}

#[derive(Clone)]
pub struct LayerNormLayer {
    pub(crate) channels: usize,
    pub(crate) epsilon: f32,

    pub(crate) gamma: Vec<Tensor>,
    pub(crate) beta: Vec<Tensor>,

    pub(crate) gamma_handles: Arc<[TensorHandle]>,
    pub(crate) beta_handles: Arc<[TensorHandle]>,
}

impl LayerNormLayer {
    pub fn new(
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

        let gamma = (0..channels)
            .map(|_| Tensor::new(1.0))
            .collect::<Vec<_>>();

        let beta = (0..channels)
            .map(|_| Tensor::new(0.0))
            .collect::<Vec<_>>();

        let gamma_handles =
            gamma
                .iter()
                .map(|x| x.handle)
                .collect();

        let beta_handles =
            beta
                .iter()
                .map(|x| x.handle)
                .collect();

        Self {
            channels,
            epsilon,
            gamma,
            beta,
            gamma_handles,
            beta_handles,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params =
            self.gamma.clone();

        params.extend_from_slice(
            &self.beta
        );

        params
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

    pub(crate) write_weights: Vec<Tensor>,
    pub(crate) write_biases: Vec<Tensor>,

    pub(crate) read_weights: Vec<Tensor>,
    pub(crate) read_biases: Vec<Tensor>,

    pub(crate) write_weight_handles: Arc<[TensorHandle]>,
    pub(crate) write_bias_handles: Arc<[TensorHandle]>,

    pub(crate) read_weight_handles: Arc<[TensorHandle]>,
    pub(crate) read_bias_handles: Arc<[TensorHandle]>,
}

impl GlobalMixerLayer {
    fn new(channels: usize, global_dim: usize) -> Self {
        assert!(channels > 0);
        assert!(global_dim > 0);

        let mut rng = rand::rng();

        // Start substantially closer to an identity mapping.
        //
        // Previous:
        //     1.0 / sqrt(channels)
        //
        // New:
        //     0.1 / sqrt(channels)
        //
        // This keeps the write/read logits small, making both softmaxes
        // close to uniform at initialization and greatly reducing the
        // magnitude of the initial global message.
        let scale = 0.1f32 / (channels as f32).sqrt();

        let dist =
            NormalDist::new(0.0, scale as f64).unwrap();

        let mut write_weights =
            Vec::with_capacity(channels * global_dim);

        let mut read_weights =
            Vec::with_capacity(channels * global_dim);

        for _ in 0..channels * global_dim {
            write_weights.push(
                Tensor::new(
                    dist.sample(&mut rng) as f32
                )
            );

            read_weights.push(
                Tensor::new(
                    dist.sample(&mut rng) as f32
                )
            );
        }

        // Zero biases keep the initial write/read logits centered.
        let mut write_biases =
            Vec::with_capacity(global_dim);

        let mut read_biases =
            Vec::with_capacity(global_dim);

        for _ in 0..global_dim {
            write_biases.push(
                Tensor::new(0.0)
            );

            read_biases.push(
                Tensor::new(0.0)
            );
        }

        let write_weight_handles: Arc<[TensorHandle]> =
            write_weights
                .iter()
                .map(|x| x.handle)
                .collect::<Vec<_>>()
                .into();

        let write_bias_handles: Arc<[TensorHandle]> =
            write_biases
                .iter()
                .map(|x| x.handle)
                .collect::<Vec<_>>()
                .into();

        let read_weight_handles: Arc<[TensorHandle]> =
            read_weights
                .iter()
                .map(|x| x.handle)
                .collect::<Vec<_>>()
                .into();

        let read_bias_handles: Arc<[TensorHandle]> =
            read_biases
                .iter()
                .map(|x| x.handle)
                .collect::<Vec<_>>()
                .into();

        Self {
            channels,
            global_dim,

            write_weights,
            write_biases,

            read_weights,
            read_biases,

            write_weight_handles,
            write_bias_handles,

            read_weight_handles,
            read_bias_handles,
        }
    }


    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::with_capacity(
            self.write_weights.len()
                + self.write_biases.len()
                + self.read_weights.len()
                + self.read_biases.len(),
        );

        params.extend_from_slice(&self.write_weights);
        params.extend_from_slice(&self.write_biases);
        params.extend_from_slice(&self.read_weights);
        params.extend_from_slice(&self.read_biases);

        params
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
            Layer::Dense(layer) => layer.spec(),
            Layer::Conv1D(layer) => layer.spec(),
            Layer::Residual(layer) => layer.spec(),
            Layer::DepthwiseConv1D(layer) => layer.spec(),
            Layer::GroupedConv1D(layer) => layer.spec(),
            Layer::LowRankPointwise(layer) => layer.spec(),
            Layer::ChannelScale(layer) => layer.spec(),
            Layer::LayerNorm(layer) => layer.spec(),
            Layer::WeightTying(layer) => layer.spec(),
            Layer::GlobalMixer(layer) => layer.spec(),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match self {
            Layer::Dense(layer) => layer.parameters(),
            Layer::Conv1D(layer) => layer.parameters(),
            Layer::Residual(layer) => layer.parameters(),
            Layer::DepthwiseConv1D(layer) => layer.parameters(),
            Layer::GroupedConv1D(layer) => layer.parameters(),
            Layer::LowRankPointwise(layer) => layer.parameters(),
            Layer::ChannelScale(layer) => layer.parameters(),
            Layer::LayerNorm(layer) => layer.parameters(),
            Layer::WeightTying(layer) => layer.parameters(),
            Layer::GlobalMixer(layer) => layer.parameters(),
        }
    }
}

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

fn build_layers(
    current_size: usize,
    specs: &[LayerSpec],
    embeddings: Option<&Arc<Embeddings>>,
) -> (Vec<Layer>, usize) {
    let mut layers = Vec::with_capacity(specs.len());
    let mut current_size = current_size;

    for spec in specs {
        let layer = match spec {
            LayerSpec::Dense {
                output_size,
                activation,
            } => Layer::Dense(DenseLayer::new(
                current_size,
                *output_size,
                activation.clone(),
            )),

            LayerSpec::Conv1D {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                causal,
                activation,
            } => {
                assert_eq!(current_size % in_channels, 0);

                Layer::Conv1D(Conv1DLayer::new(
                    *in_channels,
                    *out_channels,
                    *kernel_size,
                    *stride,
                    *padding,
                    *causal,
                    activation.clone(),
                ))
            }

            LayerSpec::Residual { layers: inner_specs } => {
                let residual_input_size = current_size;

                let (inner_layers, inner_output_size) =
                    build_layers(
                        residual_input_size,
                        inner_specs,
                        embeddings,
                    );

                assert_eq!(
                    residual_input_size,
                    inner_output_size,
                    "Residual block changed tensor size"
                );

                Layer::Residual(ResidualLayer::new(
                    residual_input_size,
                    inner_layers,
                    inner_output_size,
                ))
            }

            LayerSpec::DepthwiseConv1D {
                in_channels,
                kernel_size,
                stride,
                padding,
                causal,
                activation,
            } => {
                assert_eq!(current_size % in_channels, 0);

                Layer::DepthwiseConv1D(
                    DepthwiseConv1DLayer::new(
                        *in_channels,
                        *kernel_size,
                        *stride,
                        *padding,
                        *causal,
                        activation.clone(),
                    ),
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
                assert_eq!(current_size % in_channels, 0);

                Layer::GroupedConv1D(
                    GroupedConv1DLayer::new(
                        *in_channels,
                        *out_channels,
                        *groups,
                        *kernel_size,
                        *stride,
                        *padding,
                        *causal,
                        activation.clone(),
                    ),
                )
            }

            LayerSpec::LowRankPointwise {
                in_channels,
                rank,
                out_channels,
                activation,
            } => {
                assert_eq!(current_size % in_channels, 0);

                Layer::LowRankPointwise(
                    LowRankPointwiseLayer::new(
                        *in_channels,
                        *rank,
                        *out_channels,
                        activation.clone(),
                    ),
                )
            }

            LayerSpec::ChannelScale { channels } => {
                assert_eq!(current_size % channels, 0);

                Layer::ChannelScale(
                    ChannelScaleLayer::new(*channels),
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
                        Arc::clone(embeddings)
                    )
                )
            }
            LayerSpec::LayerNorm {
                channels,
                epsilon,
            } => {
                assert_eq!(current_size % channels,
                           0,
                           "LayerNorm channels ({}) must divide layer size ({})",
                           channels,
                           current_size);

                Layer::LayerNorm(
                    LayerNormLayer::new(
                        *channels,
                        *epsilon,
                    )
                )
            }
            LayerSpec::GlobalMixer {
                channels,
                global_dim,
            } => {
                assert!(
                    *channels > 0,
                    "GlobalMixer channels must be > 0"
                );

                assert!(
                    *global_dim > 0,
                    "GlobalMixer global_dim must be > 0"
                );

                assert_eq!(
                    current_size % *channels,
                    0,
                    "GlobalMixer channels ({}) must divide layer size ({})",
                    channels,
                    current_size
                );

                Layer::GlobalMixer(
                    GlobalMixerLayer::new(
                        *channels,
                        *global_dim,
                    )
                )
            }
        };

        current_size = layer_output_size(&layer, current_size);
        layers.push(layer);
    }

    (layers, current_size)
}

fn layer_output_size(layer: &Layer, current_size: usize) -> usize {
    match layer {
        Layer::Dense(layer) => layer.biases.len(),

        Layer::Conv1D(layer) => {
            let input_length = current_size / layer.in_channels;
            layer.output_length(input_length) * layer.out_channels
        }

        Layer::Residual(layer) => layer.output_size,

        Layer::DepthwiseConv1D(layer) => {
            let input_length = current_size / layer.in_channels;
            layer.output_length(input_length) * layer.in_channels
        }

        Layer::GroupedConv1D(layer) => {
            let input_length = current_size / layer.in_channels;
            layer.output_length(input_length) * layer.out_channels
        }

        Layer::LowRankPointwise(layer) => {
            let length = current_size / layer.in_channels;
            length * layer.out_channels
        }

        Layer::ChannelScale(layer) => {
            assert_eq!(current_size % layer.channels, 0);
            current_size
        }

        Layer::LayerNorm(layer) => {
            assert_eq!(
                current_size % layer.channels,
                0,
                "LayerNorm channels must divide input size"
            );

            current_size
        }

        Layer::GlobalMixer(layer) => {
            let channels = layer.channels;

            assert_eq!(
                current_size % channels,
                0,
                "GlobalMixer channels must divide input size"
            );

            current_size
        }

        Layer::WeightTying(layer) => {
            layer.embeddings.vocab_size()
        }
    }
}

impl MLP {
    pub fn new(num_inputs: usize, layer_sizes: &[usize]) -> Self {
        let specs = layer_sizes
            .iter()
            .enumerate()
            .map(|(i, &output_size)| LayerSpec::Dense {
                output_size,
                activation: if i + 1 == layer_sizes.len() {
                    Activation::None
                } else {
                    Activation::LeakyReLU { slope: 0.01 }
                },
            })
            .collect::<Vec<_>>();

        Self::from_layers(num_inputs, &specs)
    }

    pub fn from_layers(
        num_inputs: usize,
        specs: &[LayerSpec],
    ) -> Self {
        let (layers, _) =
            build_layers(
                num_inputs,
                specs,
                None,
            );

        Self { layers }
    }

    pub fn from_layers_with_embeddings(
        num_inputs: usize,
        specs: &[LayerSpec],
        embeddings: Arc<Embeddings>,
    ) -> Self {
        let (layers, _) =
            build_layers(
                num_inputs,
                specs,
                Some(&embeddings),
            );

        Self { layers }
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

        let (output, output_size, layers) =
            forward_layers_batch(
                &self.layers,
                input,
                batch_size,
                input_size,
            );

        BatchForward {
            output,
            batch_size,
            output_size,
            layers,
        }
    }

    pub(crate) fn backward_batch(
        &self,
        forward: &BatchForward,
        output_grads: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(
            output_grads.len(),
            forward.batch_size * forward.output_size
        );

        backward_layers_batch(
            &forward.layers,
            output_grads.to_vec(),
            forward.batch_size,
        )
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn save(&self) -> SavedMLP {
        SavedMLP {
            version: 2,
            layers: self.layers
                .iter()
                .map(save_layer)
                .collect(),
        }
    }

    pub fn load(saved: &SavedMLP) -> Self {
        assert!(
            saved.version == 1 || saved.version == 2,
            "Unsupported MLP save version: {}",
            saved.version
        );

        Self {
            layers: saved
                .layers
                .iter()
                .map(|layer| load_layer(layer, None))
                .collect(),
        }
    }

    pub fn load_with_embeddings(
        saved: &SavedMLP,
        embeddings: Arc<Embeddings>,
    ) -> Self {
        assert!(
            saved.version == 1 || saved.version == 2,
            "Unsupported MLP save version: {}",
            saved.version
        );

        Self {
            layers: saved
                .layers
                .iter()
                .map(|layer| {
                    load_layer(
                        layer,
                        Some(&embeddings),
                    )
                })
                .collect(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters().len()
    }

    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        self.layers
            .iter()
            .map(|layer| layer.spec())
            .collect()
    }
}

fn save_layer(layer: &Layer) -> SavedLayer {
    match layer {
        Layer::Dense(layer) => {
            SavedLayer::Dense {
                input_size: if layer.biases.is_empty() {
                    0
                } else {
                    layer.weights.len() / layer.biases.len()
                },

                output_size: layer.biases.len(),

                activation: layer.activation.clone(),

                weights: layer
                    .weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                biases: layer
                    .biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::Conv1D(layer) => {
            SavedLayer::Conv1D {
                in_channels: layer.in_channels,
                out_channels: layer.out_channels,
                kernel_size: layer.kernel_size,
                stride: layer.stride,
                padding: layer.padding,
                causal: layer.causal,
                activation: layer.activation.clone(),

                weights: layer
                    .weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                biases: layer
                    .biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::Residual(layer) => {
            SavedLayer::Residual {
                input_size: layer.input_size,
                output_size: layer.output_size,
                layers: layer
                    .layers
                    .iter()
                    .map(save_layer)
                    .collect(),
            }
        }

        Layer::DepthwiseConv1D(layer) => {
            SavedLayer::DepthwiseConv1D {
                in_channels: layer.in_channels,
                kernel_size: layer.kernel_size,
                stride: layer.stride,
                padding: layer.padding,
                causal: layer.causal,
                activation: layer.activation.clone(),

                weights: layer
                    .weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                biases: layer
                    .biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::GroupedConv1D(layer) => {
            SavedLayer::GroupedConv1D {
                in_channels: layer.in_channels,
                out_channels: layer.out_channels,
                groups: layer.groups,
                kernel_size: layer.kernel_size,
                stride: layer.stride,
                padding: layer.padding,
                causal: layer.causal,
                activation: layer.activation.clone(),

                weights: layer
                    .weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                biases: layer
                    .biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::LowRankPointwise(layer) => {
            SavedLayer::LowRankPointwise {
                in_channels: layer.in_channels,
                rank: layer.rank,
                out_channels: layer.out_channels,
                activation: layer.activation.clone(),

                first_weights: layer
                    .first_weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                first_biases: layer
                    .first_biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                second_weights: layer
                    .second_weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                second_biases: layer
                    .second_biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::ChannelScale(layer) => {
            SavedLayer::ChannelScale {
                channels: layer.channels,

                scales: layer
                    .scales
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                biases: layer
                    .biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::LayerNorm(layer) => {
            SavedLayer::LayerNorm {
                channels: layer.channels,

                epsilon: layer.epsilon,

                gamma: layer
                    .gamma
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                beta: layer
                    .beta
                    .iter()
                    .map(|x| x.data())
                    .collect(),
            }
        }

        Layer::GlobalMixer(layer) => {
            SavedLayer::GlobalMixer {
                channels: layer.channels,
                global_dim: layer.global_dim,

                write_weights: layer
                    .write_weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                write_biases: layer
                    .write_biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                read_weights: layer
                    .read_weights
                    .iter()
                    .map(|x| x.data())
                    .collect(),

                read_biases: layer
                    .read_biases
                    .iter()
                    .map(|x| x.data())
                    .collect(),
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
    }
}

fn load_layer(
    saved: &SavedLayer,
    embeddings: Option<&Arc<Embeddings>>,
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
                input_size * output_size,
                "Invalid Dense weight count"
            );

            assert_eq!(
                biases.len(),
                *output_size,
                "Invalid Dense bias count"
            );

            let weights = weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let biases = biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let fused_weights = weights
                .iter()
                .map(|x| x.handle)
                .collect();

            let fused_biases = biases
                .iter()
                .map(|x| x.handle)
                .collect();

            Layer::Dense(DenseLayer {
                weights,
                biases,
                fused_weights,
                fused_biases,
                activation: activation.clone(),
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
                in_channels * out_channels * kernel_size,
                "Invalid Conv1D weight count"
            );

            assert_eq!(
                biases.len(),
                *out_channels,
                "Invalid Conv1D bias count"
            );

            let weights = weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let biases = biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let weight_handles = weights
                .iter()
                .map(|x| x.handle)
                .collect();

            let bias_handles = biases
                .iter()
                .map(|x| x.handle)
                .collect();

            Layer::Conv1D(Conv1DLayer {
                in_channels: *in_channels,
                out_channels: *out_channels,
                kernel_size: *kernel_size,
                stride: *stride,
                padding: *padding,
                causal: *causal,
                activation: activation.clone(),
                weights,
                biases,
                weight_handles,
                bias_handles,
            })
        }

        SavedLayer::Residual {
            input_size,
            output_size,
            layers,
        } => {
            assert_eq!(
                input_size,
                output_size,
                "Invalid saved residual layer: input/output sizes differ"
            );

            let inner_layers = layers
                .iter()
                .map(|layer| load_layer(layer, embeddings))
                .collect::<Vec<_>>();

            Layer::Residual(
                ResidualLayer::new(
                    *input_size,
                    inner_layers,
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
                in_channels * kernel_size,
                "Invalid DepthwiseConv1D weight count"
            );

            assert_eq!(
                biases.len(),
                *in_channels,
                "Invalid DepthwiseConv1D bias count"
            );

            let weights = weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let biases = biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let weight_handles = weights
                .iter()
                .map(|x| x.handle)
                .collect();

            let bias_handles = biases
                .iter()
                .map(|x| x.handle)
                .collect();

            Layer::DepthwiseConv1D(
                DepthwiseConv1DLayer {
                    in_channels: *in_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    causal: *causal,
                    activation: activation.clone(),
                    weights,
                    biases,
                    weight_handles,
                    bias_handles,
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

            let group_in = in_channels / groups;

            assert_eq!(
                weights.len(),
                out_channels * group_in * kernel_size,
                "Invalid GroupedConv1D weight count"
            );

            assert_eq!(
                biases.len(),
                *out_channels,
                "Invalid GroupedConv1D bias count"
            );

            let weights = weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let biases = biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let weight_handles = weights
                .iter()
                .map(|x| x.handle)
                .collect();

            let bias_handles = biases
                .iter()
                .map(|x| x.handle)
                .collect();

            Layer::GroupedConv1D(
                GroupedConv1DLayer {
                    in_channels: *in_channels,
                    out_channels: *out_channels,
                    groups: *groups,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    causal: *causal,
                    activation: activation.clone(),
                    weights,
                    biases,
                    weight_handles,
                    bias_handles,
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
                rank * in_channels,
                "Invalid LowRankPointwise first weight count"
            );
            assert_eq!(
                first_biases.len(),
                *rank,
                "Invalid LowRankPointwise first bias count"
            );
            assert_eq!(
                second_weights.len(),
                out_channels * rank,
                "Invalid LowRankPointwise second weight count"
            );
            assert_eq!(
                second_biases.len(),
                *out_channels,
                "Invalid LowRankPointwise second bias count"
            );
            let first_weights = first_weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();
            let first_biases = first_biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();
            let second_weights = second_weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();
            let second_biases = second_biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();
            let first_weight_handles = first_weights
                .iter()
                .map(|x| x.handle)
                .collect();
            let first_bias_handles = first_biases
                .iter()
                .map(|x| x.handle)
                .collect();
            let second_weight_handles = second_weights
                .iter()
                .map(|x| x.handle)
                .collect();
            let second_bias_handles = second_biases
                .iter()
                .map(|x| x.handle)
                .collect();
            Layer::LowRankPointwise(
                LowRankPointwiseLayer {
                    in_channels: *in_channels,
                    rank: *rank,
                    out_channels: *out_channels,
                    activation: activation.clone(),
                    first_weights,
                    first_biases,
                    second_weights,
                    second_biases,
                    first_weight_handles,
                    first_bias_handles,
                    second_weight_handles,
                    second_bias_handles,
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
                *channels,
                "Invalid ChannelScale scale count"
            );
            assert_eq!(
                biases.len(),
                *channels,
                "Invalid ChannelScale bias count"
            );
            let scales = scales
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();
            let biases = biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();
            let scale_handles = scales
                .iter()
                .map(|x| x.handle)
                .collect();
            let bias_handles = biases
                .iter()
                .map(|x| x.handle)
                .collect();
            Layer::ChannelScale(
                ChannelScaleLayer {
                    channels: *channels,
                    scales,
                    biases,
                    scale_handles,
                    bias_handles,
                }
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
                *channels,
                "Invalid LayerNorm gamma count"
            );

            assert_eq!(
                beta.len(),
                *channels,
                "Invalid LayerNorm beta count"
            );

            let gamma =
                gamma
                    .iter()
                    .map(|&x| Tensor::new(x))
                    .collect::<Vec<_>>();

            let beta =
                beta
                    .iter()
                    .map(|&x| Tensor::new(x))
                    .collect::<Vec<_>>();

            let gamma_handles =
                gamma
                    .iter()
                    .map(|x| x.handle)
                    .collect();

            let beta_handles =
                beta
                    .iter()
                    .map(|x| x.handle)
                    .collect();

            Layer::LayerNorm(
                LayerNormLayer {
                    channels: *channels,
                    epsilon: *epsilon,
                    gamma,
                    beta,
                    gamma_handles,
                    beta_handles,
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
        } => {
            assert!(
                *channels > 0,
                "Invalid GlobalMixer channel count"
            );

            assert!(
                *global_dim > 0,
                "Invalid GlobalMixer global dimension"
            );

            assert_eq!(
                write_weights.len(),
                channels * global_dim,
                "Invalid GlobalMixer write weight count"
            );

            assert_eq!(
                write_biases.len(),
                *global_dim,
                "Invalid GlobalMixer write bias count"
            );

            assert_eq!(
                read_weights.len(),
                channels * global_dim,
                "Invalid GlobalMixer read weight count"
            );

            assert_eq!(
                read_biases.len(),
                *global_dim,
                "Invalid GlobalMixer read bias count"
            );

            let write_weights = write_weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let write_biases = write_biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let read_weights = read_weights
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let read_biases = read_biases
                .iter()
                .map(|&x| Tensor::new(x))
                .collect::<Vec<_>>();

            let write_weight_handles = write_weights
                .iter()
                .map(|x| x.handle)
                .collect();

            let write_bias_handles = write_biases
                .iter()
                .map(|x| x.handle)
                .collect();

            let read_weight_handles = read_weights
                .iter()
                .map(|x| x.handle)
                .collect();

            let read_bias_handles = read_biases
                .iter()
                .map(|x| x.handle)
                .collect();

            Layer::GlobalMixer(
                GlobalMixerLayer {
                    channels: *channels,
                    global_dim: *global_dim,

                    write_weights,
                    write_biases,

                    read_weights,
                    read_biases,

                    write_weight_handles,
                    write_bias_handles,

                    read_weight_handles,
                    read_bias_handles,
                }
            )
        }

        SavedLayer::WeightTying {
            embedding_dim,
            vocab_size,
        } => {
            let embeddings =
                embeddings.expect(
                    "Loading WeightTying requires Embeddings"
                );

            assert_eq!(
                embeddings.embedding_dim(),
                *embedding_dim,
                "Saved WeightTying embedding dimension does not match Embeddings"
            );

            assert_eq!(
                embeddings.vocab_size(),
                *vocab_size,
                "Saved WeightTying vocabulary size does not match Embeddings"
            );

            Layer::WeightTying(
                WeightTyingLayer::new(
                    Arc::clone(embeddings)
                )
            )
        }
    }
}
