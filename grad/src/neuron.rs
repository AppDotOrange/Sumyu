use std::sync::Arc;
use cblas::{Layout, Transpose};
use rand_distr::{Distribution, Normal as NormalDist};
use serde::{Deserialize, Serialize};
use crate::{Tensor, TensorHandle};

const CONV1D_TILE: usize = 64;

#[derive(Clone, Serialize, Deserialize)]
pub enum Activation {
    None,
    LeakyReLU { slope: f32 },
}

impl Activation {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        match self {
            Self::None => x,
            Self::LeakyReLU { slope } => if x <= 0.0 { x * slope } else { x },
        }
    }

    #[inline]
    fn backward(&self, x: f32, grad: &mut f32) {
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
}

pub(crate) struct BatchForward {
    pub output: Vec<f32>,
    pub batch_size: usize,
    pub output_size: usize,
    pub layers: Vec<BatchLayerCache>,
}

pub(crate) enum BatchLayerCache {
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
}

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Tensor>,
    bias: Tensor,
    is_output: bool,
}

impl Neuron {
    pub fn new(num_inputs: usize, is_output: bool) -> Self {
        let mut rng = rand::rng();
        let std_dev = (2.0 / num_inputs as f32).sqrt();
        let normal = NormalDist::new(0.0, std_dev).expect("Invalid standard deviation");

        let weights = (0..num_inputs)
            .map(|_| Tensor::new(normal.sample(&mut rng)))
            .collect();

        Self {
            weights,
            bias: Tensor::new(0.1),
            is_output,
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.weights.clone();
        params.push(self.bias);
        params
    }
}

#[derive(Clone)]
pub struct DenseLayer {
    neurons: Vec<Neuron>,
    fused_weights: Arc<[TensorHandle]>,
    fused_biases: Arc<[TensorHandle]>,
    activation: Activation,
}

impl DenseLayer {
    fn new(num_inputs: usize, num_outputs: usize, activation: Activation) -> Self {
        let is_output = matches!(activation, Activation::None);

        let neurons = (0..num_outputs)
            .map(|_| Neuron::new(num_inputs, is_output))
            .collect::<Vec<_>>();

        let fused_weights = neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.handle))
            .collect();

        let fused_biases = neurons.iter().map(|n| n.bias.handle).collect();

        Self {
            neurons,
            fused_weights,
            fused_biases,
            activation,
        }
    }

    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let input_handles: Vec<TensorHandle> = inputs.iter().map(|x| x.handle).collect();

        Tensor::fused_layer(
            Arc::clone(&self.fused_weights),
            &input_handles,
            Arc::clone(&self.fused_biases),
            matches!(self.activation, Activation::None),
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
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
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    activation: Activation,
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    weight_handles: Arc<[TensorHandle]>,
    bias_handles: Arc<[TensorHandle]>,
}

impl Conv1DLayer {
    fn new(
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
    fn output_length(&self, input_length: usize) -> usize {
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
    in_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    activation: Activation,

    weights: Vec<Tensor>,
    biases: Vec<Tensor>,

    weight_handles: Arc<[TensorHandle]>,
    bias_handles: Arc<[TensorHandle]>,
}

impl DepthwiseConv1DLayer {
    fn new(
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
    fn output_length(&self, input_length: usize) -> usize {
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
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    activation: Activation,

    weights: Vec<Tensor>,
    biases: Vec<Tensor>,

    weight_handles: Arc<[TensorHandle]>,
    bias_handles: Arc<[TensorHandle]>,
}

impl GroupedConv1DLayer {
    fn new(
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
    fn output_length(&self, input_length: usize) -> usize {
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
    in_channels: usize,
    rank: usize,
    out_channels: usize,
    activation: Activation,

    first_weights: Vec<Tensor>,
    first_biases: Vec<Tensor>,

    second_weights: Vec<Tensor>,
    second_biases: Vec<Tensor>,

    first_weight_handles: Arc<[TensorHandle]>,
    first_bias_handles: Arc<[TensorHandle]>,
    second_weight_handles: Arc<[TensorHandle]>,
    second_bias_handles: Arc<[TensorHandle]>,
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
    channels: usize,
    scales: Vec<Tensor>,
    biases: Vec<Tensor>,
    scale_handles: Arc<[TensorHandle]>,
    bias_handles: Arc<[TensorHandle]>,
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
pub struct ResidualLayer {
    layers: Vec<Layer>,
    input_size: usize,
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
pub enum Layer {
    Dense(DenseLayer),
    Conv1D(Conv1DLayer),
    Residual(ResidualLayer),
    DepthwiseConv1D(DepthwiseConv1DLayer),
    GroupedConv1D(GroupedConv1DLayer),
    LowRankPointwise(LowRankPointwiseLayer),
    ChannelScale(ChannelScaleLayer),
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
                    build_layers(residual_input_size, inner_specs);

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
        };

        current_size = layer_output_size(&layer, current_size);
        layers.push(layer);
    }

    (layers, current_size)
}

fn layer_output_size(layer: &Layer, current_size: usize) -> usize {
    match layer {
        Layer::Dense(layer) => layer.neurons.len(),

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
    }
}

fn forward_layers_batch(
    layers: &[Layer],
    input: &[f32],
    batch_size: usize,
    input_size: usize,
) -> (Vec<f32>, usize, Vec<BatchLayerCache>) {
    let mut current = input.to_vec();
    let mut current_size = input_size;
    let mut caches = Vec::with_capacity(layers.len());

    for layer in layers {
        let (output, output_size, cache) =
            forward_layer_batch(layer, &current, batch_size, current_size);

        current = output;
        current_size = output_size;
        caches.push(cache);
    }

    (current, current_size, caches)
}

fn forward_layer_batch(
    layer: &Layer,
    input: &[f32],
    batch_size: usize,
    input_size: usize,
) -> (Vec<f32>, usize, BatchLayerCache) {
    match layer {
        Layer::Dense(layer) => {
            let output_size = layer.neurons.len();

            assert_eq!(
                input.len(),
                batch_size * input_size
            );

            let weights = layer
                .fused_weights
                .iter()
                .map(|&h| crate::handle_data(h))
                .collect::<Vec<_>>();

            let biases = layer
                .fused_biases
                .iter()
                .map(|&h| crate::handle_data(h))
                .collect::<Vec<_>>();

            let mut output =
                vec![0.0; batch_size * output_size];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    batch_size as i32,
                    output_size as i32,
                    input_size as i32,
                    1.0,
                    input,
                    input_size as i32,
                    &weights,
                    input_size as i32,
                    0.0,
                    &mut output,
                    output_size as i32,
                );
            }

            for b in 0..batch_size {
                let base = b * output_size;

                for o in 0..output_size {
                    let i = base + o;

                    output[i] =
                        layer.activation.apply(
                            output[i] + biases[o]
                        );
                }
            }

            let activation_output =
                match layer.activation {
                    Activation::None => None,
                    _ => Some(output.clone()),
                };

            let cache = BatchLayerCache::Dense {
                input_size,
                output_size,
                input: input.to_vec(),
                activation_output,
                weights,
                weight_handles: Arc::clone(&layer.fused_weights),
                bias_handles: Arc::clone(&layer.fused_biases),
                activation: layer.activation.clone(),
            };

            (output, output_size, cache)
        }

        Layer::Conv1D(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let input_length =
                input_size / layer.in_channels;

            let output_length =
                layer.output_length(input_length);

            let output = conv1d_forward(
                layer,
                input,
                batch_size,
                input_length,
                output_length,
            );

            let output_size =
                output_length * layer.out_channels;

            let cache = BatchLayerCache::Conv1D {
                input: input.to_vec(),
                output: output.clone(),
                input_length,
                output_length,
                in_channels: layer.in_channels,
                out_channels: layer.out_channels,
                kernel_size: layer.kernel_size,
                stride: layer.stride,
                padding: layer.padding,
                causal: layer.causal,
                weight_handles: Arc::clone(
                    &layer.weight_handles
                ),
                bias_handles: Arc::clone(
                    &layer.bias_handles
                ),
                activation: layer.activation.clone(),
            };

            (output, output_size, cache)
        }

        Layer::Residual(layer) => {
            assert_eq!(
                input_size,
                layer.input_size
            );

            let (
                inner_output,
                inner_output_size,
                inner_caches,
            ) = forward_layers_batch(
                &layer.layers,
                input,
                batch_size,
                input_size,
            );

            assert_eq!(
                inner_output_size,
                input_size,
                "Residual block changed tensor size"
            );

            let mut output =
                inner_output.clone();

            for i in 0..output.len() {
                output[i] += input[i];
            }

            let cache = BatchLayerCache::Residual {
                inner: inner_caches,
            };

            (
                output,
                input_size,
                cache,
            )
        }

        Layer::DepthwiseConv1D(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let input_length =
                input_size / layer.in_channels;

            let output_length =
                layer.output_length(input_length);

            let output =
                depthwise_conv1d_forward(
                    layer,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                );

            let output_size =
                output_length * layer.in_channels;

            let cache =
                BatchLayerCache::DepthwiseConv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels: layer.in_channels,
                    kernel_size: layer.kernel_size,
                    stride: layer.stride,
                    padding: layer.padding,
                    causal: layer.causal,
                    weight_handles: Arc::clone(
                        &layer.weight_handles
                    ),
                    bias_handles: Arc::clone(
                        &layer.bias_handles
                    ),
                    activation: layer.activation.clone(),
                };

            (
                output,
                output_size,
                cache,
            )
        }

        Layer::GroupedConv1D(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let input_length =
                input_size / layer.in_channels;

            let output_length =
                layer.output_length(input_length);

            let output =
                grouped_conv1d_forward(
                    layer,
                    input,
                    batch_size,
                    input_length,
                    output_length,
                );

            let output_size =
                output_length * layer.out_channels;

            let cache =
                BatchLayerCache::GroupedConv1D {
                    input: input.to_vec(),
                    output: output.clone(),
                    input_length,
                    output_length,
                    in_channels: layer.in_channels,
                    out_channels: layer.out_channels,
                    groups: layer.groups,
                    kernel_size: layer.kernel_size,
                    stride: layer.stride,
                    padding: layer.padding,
                    causal: layer.causal,
                    weight_handles: Arc::clone(
                        &layer.weight_handles
                    ),
                    bias_handles: Arc::clone(
                        &layer.bias_handles
                    ),
                    activation: layer.activation.clone(),
                };

            (
                output,
                output_size,
                cache,
            )
        }

        Layer::LowRankPointwise(layer) => {
            assert_eq!(
                input_size % layer.in_channels,
                0
            );

            let positions =
                input_size / layer.in_channels;

            let mut first_weights =
                vec![0.0; layer.first_weight_handles.len()];

            let mut first_biases =
                vec![0.0; layer.first_bias_handles.len()];

            let mut second_weights =
                vec![0.0; layer.second_weight_handles.len()];

            let mut second_biases =
                vec![0.0; layer.second_bias_handles.len()];

            crate::handle_data_slice(
                &layer.first_weight_handles,
                &mut first_weights,
            );

            crate::handle_data_slice(
                &layer.first_bias_handles,
                &mut first_biases,
            );

            crate::handle_data_slice(
                &layer.second_weight_handles,
                &mut second_weights,
            );

            crate::handle_data_slice(
                &layer.second_bias_handles,
                &mut second_biases,
            );

            let (
                output,
                hidden,
            ) = low_rank_pointwise_forward(
                layer,
                input,
                batch_size,
                positions,
                &first_weights,
                &first_biases,
                &second_weights,
                &second_biases,
            );

            let output_size =
                positions * layer.out_channels;

            let rows =
                batch_size * positions;

            let cache =
                BatchLayerCache::LowRankPointwise {
                    input: input.to_vec(),
                    hidden,
                    output: output.clone(),
                    rows,
                    in_channels: layer.in_channels,
                    rank: layer.rank,
                    out_channels: layer.out_channels,
                    first_weights,
                    second_weights,
                    first_weight_handles: Arc::clone(
                        &layer.first_weight_handles
                    ),
                    first_bias_handles: Arc::clone(
                        &layer.first_bias_handles
                    ),
                    second_weight_handles: Arc::clone(
                        &layer.second_weight_handles
                    ),
                    second_bias_handles: Arc::clone(
                        &layer.second_bias_handles
                    ),
                    activation: layer.activation.clone(),
                };

            (
                output,
                output_size,
                cache,
            )
        }

        Layer::ChannelScale(layer) => {
            assert_eq!(
                input_size % layer.channels,
                0
            );

            let output =
                channel_scale_forward(
                    layer,
                    input,
                    batch_size,
                );

            let cache =
                BatchLayerCache::ChannelScale {
                    input: input.to_vec(),
                    channels: layer.channels,
                    scale_handles: Arc::clone(
                        &layer.scale_handles
                    ),
                    bias_handles: Arc::clone(
                        &layer.bias_handles
                    ),
                };

            (
                output,
                input_size,
                cache,
            )
        }
    }
}

fn backward_layers_batch(
    layers: &[BatchLayerCache],
    mut grad: Vec<f32>,
    batch_size: usize,
) -> Vec<f32> {
    for layer in layers.iter().rev() {
        grad = backward_layer_batch(
            layer,
            grad,
            batch_size,
        );
    }

    grad
}

fn backward_layer_batch(
    layer: &BatchLayerCache,
    mut grad: Vec<f32>,
    batch_size: usize,
) -> Vec<f32> {
    match layer {
        BatchLayerCache::Dense {
            input_size,
            output_size,
            input,
            activation_output,
            weights,
            weight_handles,
            bias_handles,
            activation,
        } => {
            if let Some(output) = activation_output {
                for b in 0..batch_size {
                    let base = b * *output_size;

                    for o in 0..*output_size {
                        activation.backward(
                            output[base + o],
                            &mut grad[base + o],
                        );
                    }
                }
            }

            let mut weight_grads =
                vec![0.0; output_size * input_size];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    *output_size as i32,
                    *input_size as i32,
                    batch_size as i32,
                    1.0,
                    &grad,
                    *output_size as i32,
                    input,
                    *input_size as i32,
                    0.0,
                    &mut weight_grads,
                    *input_size as i32,
                );
            }

            let mut bias_grads =
                vec![0.0; *output_size];

            for b in 0..batch_size {
                let base = b * *output_size;

                for o in 0..*output_size {
                    bias_grads[o] += grad[base + o];
                }
            }

            let mut input_grads =
                vec![0.0; batch_size * *input_size];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    batch_size as i32,
                    *input_size as i32,
                    *output_size as i32,
                    1.0,
                    &grad,
                    *output_size as i32,
                    weights,
                    *input_size as i32,
                    0.0,
                    &mut input_grads,
                    *input_size as i32,
                );
            }

            let mut parameter_grads =
                Vec::with_capacity(
                    weight_grads.len()
                        + bias_grads.len(),
                );

            for i in 0..weight_grads.len() {
                parameter_grads.push((
                    weight_handles[i],
                    weight_grads[i],
                ));
            }

            for i in 0..bias_grads.len() {
                parameter_grads.push((
                    bias_handles[i],
                    bias_grads[i],
                ));
            }

            crate::add_handle_grads(
                &parameter_grads
            );

            input_grads
        }

        BatchLayerCache::Conv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
            )
        }

        BatchLayerCache::Residual {
            inner,
        } => {
            let skip_grad = grad.clone();

            let mut result =
                backward_layers_batch(
                    inner,
                    grad,
                    batch_size,
                );

            debug_assert_eq!(
                result.len(),
                skip_grad.len()
            );

            for i in 0..result.len() {
                result[i] += skip_grad[i];
            }

            result
        }

        BatchLayerCache::DepthwiseConv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            depthwise_conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
            )
        }

        BatchLayerCache::GroupedConv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            out_channels,
            groups,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            grouped_conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *out_channels,
                *groups,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
            )
        }

        BatchLayerCache::LowRankPointwise {
            input,
            hidden,
            output,
            rows,
            in_channels,
            rank,
            out_channels,
            first_weights,
            second_weights,
            first_weight_handles,
            first_bias_handles,
            second_weight_handles,
            second_bias_handles,
            activation,
        } => {
            low_rank_pointwise_backward(
                input,
                hidden,
                output,
                &mut grad,
                *rows,
                *in_channels,
                *rank,
                *out_channels,
                first_weights,
                second_weights,
                first_weight_handles,
                first_bias_handles,
                second_weight_handles,
                second_bias_handles,
                activation,
            )
        }

        BatchLayerCache::ChannelScale {
            input,
            channels,
            scale_handles,
            bias_handles,
        } => {
            channel_scale_backward(
                input,
                &mut grad,
                batch_size,
                *channels,
                scale_handles,
                bias_handles,
            )
        }
    }
}

fn grouped_conv1d_forward(
    layer: &GroupedConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let group_in =
        layer.in_channels / layer.groups;

    let group_out =
        layer.out_channels / layer.groups;

    let kernel_width =
        group_in * layer.kernel_size;

    let mut weights =
        vec![0.0; layer.weight_handles.len()];

    let mut biases =
        vec![0.0; layer.bias_handles.len()];

    crate::handle_data_slice(
        &layer.weight_handles,
        &mut weights,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let mut output =
        vec![
            0.0;
            batch_size
                * output_length
                * layer.out_channels
        ];

    let mut col =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * kernel_width
        ];

    let mut group_output =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * group_out
        ];

    for batch_start
    in (0..batch_size).step_by(CONV1D_TILE)
    {
        let tile_batch =
            (batch_size - batch_start)
                .min(CONV1D_TILE);

        let rows =
            tile_batch * output_length;

        for group in 0..layer.groups {
            for b in 0..tile_batch {
                for out_pos in 0..output_length {
                    let row =
                        b * output_length + out_pos;

                    let row_start =
                        row * kernel_width;

                    for k in 0..layer.kernel_size {
                        let pos =
                            out_pos * layer.stride + k;

                        let src_pos =
                            if layer.causal {
                                pos as isize
                                    - (layer.kernel_size - 1)
                                    as isize
                            } else {
                                pos as isize
                                    - layer.padding as isize
                            };

                        let dst =
                            row_start
                                + k * group_in;

                        if src_pos >= 0
                            && (src_pos as usize)
                            < input_length
                        {
                            let src_channel =
                                group * group_in;

                            let src =
                                ((batch_start + b)
                                    * input_length
                                    + src_pos as usize)
                                    * layer.in_channels
                                    + src_channel;

                            col[
                                dst..dst + group_in
                                ].copy_from_slice(
                                &input[
                                    src..src + group_in
                                    ],
                            );
                        } else {
                            col[
                                dst..dst + group_in
                                ].fill(0.0);
                        }
                    }
                }
            }

            let weight_start =
                group
                    * group_out
                    * kernel_width;

            let weight_end =
                weight_start
                    + group_out * kernel_width;

            crate::batched::gemm(
                &col[..rows * kernel_width],
                &weights[
                    weight_start..weight_end
                    ],
                &mut group_output[
                    ..rows * group_out
                    ],
                rows,
                group_out,
                kernel_width,
                Transpose::None,
                Transpose::Ordinary,
                kernel_width,
                kernel_width,
                group_out,
            );

            let output_channel =
                group * group_out;

            for row in 0..rows {
                for oc in 0..group_out {
                    let src =
                        row * group_out + oc;

                    group_output[src] =
                        layer.activation.apply(
                            group_output[src]
                                + biases[
                                output_channel + oc
                                ],
                        );

                    let batch_row =
                        batch_start * output_length
                            * layer.out_channels;

                    output[
                        batch_row
                            + row * layer.out_channels
                            + output_channel
                            + oc
                        ] = group_output[src];
                }
            }
        }
    }

    output
}

fn grouped_conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
) -> Vec<f32> {
    let group_in =
        in_channels / groups;

    let group_out =
        out_channels / groups;

    let kernel_width =
        group_in * kernel_size;

    let mut weights =
        vec![0.0; weight_handles.len()];

    crate::handle_data_slice(
        weight_handles,
        &mut weights,
    );

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    let mut weight_grads =
        vec![0.0; weights.len()];

    let mut bias_grads =
        vec![0.0; out_channels];

    let mut input_grads =
        vec![0.0; input.len()];

    let mut col =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * kernel_width
        ];

    let mut col_grads =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * kernel_width
        ];

    let mut group_grad =
        vec![
            0.0;
            CONV1D_TILE
                * output_length
                * group_out
        ];

    for batch_start
    in (0..batch_size).step_by(CONV1D_TILE)
    {
        let tile_batch =
            (batch_size - batch_start)
                .min(CONV1D_TILE);

        let rows =
            tile_batch * output_length;

        for group in 0..groups {
            for b in 0..tile_batch {
                for out_pos in 0..output_length {
                    let row =
                        b * output_length + out_pos;

                    let row_start =
                        row * kernel_width;

                    for k in 0..kernel_size {
                        let pos =
                            out_pos * stride + k;

                        let src_pos =
                            if causal {
                                pos as isize
                                    - (kernel_size - 1)
                                    as isize
                            } else {
                                pos as isize
                                    - padding as isize
                            };

                        let dst =
                            row_start
                                + k * group_in;

                        if src_pos >= 0
                            && (src_pos as usize)
                            < input_length
                        {
                            let src_channel =
                                group * group_in;

                            let src =
                                ((batch_start + b)
                                    * input_length
                                    + src_pos as usize)
                                    * in_channels
                                    + src_channel;

                            col[
                                dst..dst + group_in
                                ].copy_from_slice(
                                &input[
                                    src..src + group_in
                                    ],
                            );
                        } else {
                            col[
                                dst..dst + group_in
                                ].fill(0.0);
                        }
                    }
                }
            }

            let output_channel =
                group * group_out;

            for row in 0..rows {
                let global_row =
                    batch_start * output_length
                        + row;

                for oc in 0..group_out {
                    group_grad[
                        row * group_out + oc
                        ] =
                        grad[
                            global_row
                                * out_channels
                                + output_channel
                                + oc
                            ];

                    bias_grads[
                        output_channel + oc
                        ] += group_grad[
                        row * group_out + oc
                        ];
                }
            }

            let weight_start =
                group
                    * group_out
                    * kernel_width;

            let weight_end =
                weight_start
                    + group_out * kernel_width;

            crate::batched::gemm_beta(
                &group_grad[..rows * group_out],
                &col[..rows * kernel_width],
                &mut weight_grads[
                    weight_start..weight_end
                    ],
                group_out,
                kernel_width,
                rows,
                Transpose::Ordinary,
                Transpose::None,
                group_out,
                kernel_width,
                kernel_width,
                1.0,
            );

            crate::batched::gemm(
                &group_grad[..rows * group_out],
                &weights[
                    weight_start..weight_end
                    ],
                &mut col_grads[
                    ..rows * kernel_width
                    ],
                rows,
                kernel_width,
                group_out,
                Transpose::None,
                Transpose::None,
                group_out,
                kernel_width,
                kernel_width,
            );

            for b in 0..tile_batch {
                for out_pos in 0..output_length {
                    let row =
                        b * output_length + out_pos;

                    for k in 0..kernel_size {
                        let pos =
                            out_pos * stride + k;

                        let src_pos =
                            if causal {
                                pos as isize
                                    - (kernel_size - 1)
                                    as isize
                            } else {
                                pos as isize
                                    - padding as isize
                            };

                        if src_pos >= 0
                            && (src_pos as usize)
                            < input_length
                        {
                            let dst =
                                ((batch_start + b)
                                    * input_length
                                    + src_pos as usize)
                                    * in_channels
                                    + group * group_in;

                            let src =
                                row * kernel_width
                                    + k * group_in;

                            for c in 0..group_in {
                                input_grads[
                                    dst + c
                                    ] += col_grads[
                                    src + c
                                    ];
                            }
                        }
                    }
                }
            }
        }
    }

    let mut parameter_grads =
        Vec::with_capacity(
            weight_grads.len()
                + bias_grads.len(),
        );

    for i in 0..weight_grads.len() {
        parameter_grads.push((
            weight_handles[i],
            weight_grads[i],
        ));
    }

    for i in 0..bias_grads.len() {
        parameter_grads.push((
            bias_handles[i],
            bias_grads[i],
        ));
    }

    crate::add_handle_grads(
        &parameter_grads
    );

    input_grads
}

fn low_rank_pointwise_forward(
    layer: &LowRankPointwiseLayer,
    input: &[f32],
    batch_size: usize,
    positions: usize,
    first_weights: &[f32],
    first_biases: &[f32],
    second_weights: &[f32],
    second_biases: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rows =
        batch_size * positions;

    let mut hidden =
        vec![0.0; rows * layer.rank];

    let mut output =
        vec![0.0; rows * layer.out_channels];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            layer.rank as i32,
            layer.in_channels as i32,
            1.0,
            input,
            layer.in_channels as i32,
            &first_weights,
            layer.in_channels as i32,
            0.0,
            &mut hidden,
            layer.rank as i32,
        );

        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            rows as i32,
            layer.out_channels as i32,
            layer.rank as i32,
            1.0,
            &hidden,
            layer.rank as i32,
            &second_weights,
            layer.rank as i32,
            0.0,
            &mut output,
            layer.out_channels as i32,
        );
    }

    for row in 0..rows {
        for r in 0..layer.rank {
            hidden[row * layer.rank + r] +=
                first_biases[r];
        }

        for oc in 0..layer.out_channels {
            let i =
                row * layer.out_channels + oc;

            output[i] =
                layer.activation.apply(
                    output[i] + second_biases[oc]
                );
        }
    }

    (output, hidden)
}

fn low_rank_pointwise_backward(
    input: &[f32],
    hidden: &[f32],
    output: &[f32],
    grad: &mut [f32],
    rows: usize,
    in_channels: usize,
    rank: usize,
    out_channels: usize,
    first_weights: &[f32],
    second_weights: &[f32],
    first_weight_handles: &[TensorHandle],
    first_bias_handles: &[TensorHandle],
    second_weight_handles: &[TensorHandle],
    second_bias_handles: &[TensorHandle],
    activation: &Activation,
) -> Vec<f32> {
    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    let mut second_weight_grads =
        vec![0.0; out_channels * rank];

    let mut second_bias_grads =
        vec![0.0; out_channels];

    let mut hidden_grads =
        vec![0.0; rows * rank];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            out_channels as i32,
            rank as i32,
            rows as i32,
            1.0,
            grad,
            out_channels as i32,
            hidden,
            rank as i32,
            0.0,
            &mut second_weight_grads,
            rank as i32,
        );

        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            rank as i32,
            out_channels as i32,
            1.0,
            grad,
            out_channels as i32,
            second_weights,
            rank as i32,
            0.0,
            &mut hidden_grads,
            rank as i32,
        );
    }

    for row in 0..rows {
        for r in 0..rank {
            // The first projection has no activation of its own.
            // Its bias therefore receives the complete hidden gradient.
            //
            // We intentionally accumulate this below.
            let _ = r;
        }

        for oc in 0..out_channels {
            second_bias_grads[oc] +=
                grad[row * out_channels + oc];
        }
    }

    let mut first_weight_grads =
        vec![0.0; rank * in_channels];

    let mut first_bias_grads =
        vec![0.0; rank];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            rank as i32,
            in_channels as i32,
            rows as i32,
            1.0,
            &hidden_grads,
            rank as i32,
            input,
            in_channels as i32,
            0.0,
            &mut first_weight_grads,
            in_channels as i32,
        );
    }

    for row in 0..rows {
        for r in 0..rank {
            first_bias_grads[r] +=
                hidden_grads[
                    row * rank + r
                    ];
        }
    }

    let mut input_grads =
        vec![0.0; rows * in_channels];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            in_channels as i32,
            rank as i32,
            1.0,
            &hidden_grads,
            rank as i32,
            first_weights,
            in_channels as i32,
            0.0,
            &mut input_grads,
            in_channels as i32,
        );
    }

    let mut parameter_grads =
        Vec::with_capacity(
            first_weight_grads.len()
                + first_bias_grads.len()
                + second_weight_grads.len()
                + second_bias_grads.len(),
        );

    for i in 0..first_weight_grads.len() {
        parameter_grads.push((
            first_weight_handles[i],
            first_weight_grads[i],
        ));
    }

    for i in 0..first_bias_grads.len() {
        parameter_grads.push((
            first_bias_handles[i],
            first_bias_grads[i],
        ));
    }

    for i in 0..second_weight_grads.len() {
        parameter_grads.push((
            second_weight_handles[i],
            second_weight_grads[i],
        ));
    }

    for i in 0..second_bias_grads.len() {
        parameter_grads.push((
            second_bias_handles[i],
            second_bias_grads[i],
        ));
    }

    crate::add_handle_grads(
        &parameter_grads
    );

    input_grads
}

fn channel_scale_forward(
    layer: &ChannelScaleLayer,
    input: &[f32],
    batch_size: usize,
) -> Vec<f32> {
    let mut scales =
        vec![0.0; layer.channels];

    let mut biases =
        vec![0.0; layer.channels];

    crate::handle_data_slice(
        &layer.scale_handles,
        &mut scales,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let sequence_size =
        input.len() / batch_size;

    let mut output =
        vec![0.0; input.len()];

    for b in 0..batch_size {
        let base =
            b * sequence_size;

        for i in 0..sequence_size {
            let c =
                i % layer.channels;

            let index =
                base + i;

            output[index] =
                input[index] * scales[c]
                    + biases[c];
        }
    }

    output
}

fn channel_scale_backward(
    input: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    channels: usize,
    scale_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
) -> Vec<f32> {
    let mut scales =
        vec![0.0; channels];

    crate::handle_data_slice(
        scale_handles,
        &mut scales,
    );

    let mut scale_grads =
        vec![0.0; channels];

    let mut bias_grads =
        vec![0.0; channels];

    let mut input_grads =
        vec![0.0; input.len()];

    let sequence_size =
        input.len() / batch_size;

    for b in 0..batch_size {
        let base =
            b * sequence_size;

        for i in 0..sequence_size {
            let c =
                i % channels;

            let index =
                base + i;

            let g =
                grad[index];

            scale_grads[c] +=
                g * input[index];

            bias_grads[c] += g;

            input_grads[index] =
                g * scales[c];
        }
    }

    let mut parameter_grads =
        Vec::with_capacity(
            channels * 2
        );

    for c in 0..channels {
        parameter_grads.push((
            scale_handles[c],
            scale_grads[c],
        ));

        parameter_grads.push((
            bias_handles[c],
            bias_grads[c],
        ));
    }

    crate::add_handle_grads(
        &parameter_grads
    );

    input_grads
}

fn depthwise_conv1d_forward(
    layer: &DepthwiseConv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let mut weights =
        vec![0.0; layer.weight_handles.len()];

    let mut biases =
        vec![0.0; layer.bias_handles.len()];

    crate::handle_data_slice(
        &layer.weight_handles,
        &mut weights,
    );

    crate::handle_data_slice(
        &layer.bias_handles,
        &mut biases,
    );

    let mut output =
        vec![
            0.0;
            batch_size
                * output_length
                * layer.in_channels
        ];

    for b in 0..batch_size {
        for out_pos in 0..output_length {
            for c in 0..layer.in_channels {
                let mut sum = biases[c];

                for k in 0..layer.kernel_size {
                    let pos =
                        out_pos * layer.stride + k;

                    let src_pos =
                        if layer.causal {
                            pos as isize
                                - (layer.kernel_size - 1)
                                as isize
                        } else {
                            pos as isize
                                - layer.padding as isize
                        };

                    if src_pos >= 0
                        && (src_pos as usize) < input_length
                    {
                        let src =
                            ((b * input_length
                                + src_pos as usize)
                                * layer.in_channels)
                                + c;

                        let weight =
                            weights[
                                c * layer.kernel_size + k
                                ];

                        sum += input[src] * weight;
                    }
                }

                let dst =
                    ((b * output_length + out_pos)
                        * layer.in_channels)
                        + c;

                output[dst] =
                    layer.activation.apply(sum);
            }
        }
    }

    output
}

fn depthwise_conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
) -> Vec<f32> {
    let mut weights =
        vec![0.0; weight_handles.len()];

    crate::handle_data_slice(
        weight_handles,
        &mut weights,
    );

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    let mut weight_grads =
        vec![0.0; in_channels * kernel_size];

    let mut bias_grads =
        vec![0.0; in_channels];

    let mut input_grads =
        vec![0.0; input.len()];

    for b in 0..batch_size {
        for out_pos in 0..output_length {
            for c in 0..in_channels {
                let dst =
                    (b * output_length + out_pos)
                        * in_channels
                        + c;

                let g = grad[dst];

                bias_grads[c] += g;

                for k in 0..kernel_size {
                    let pos =
                        out_pos * stride + k;

                    let src_pos =
                        if causal {
                            pos as isize
                                - (kernel_size - 1)
                                as isize
                        } else {
                            pos as isize
                                - padding as isize
                        };

                    if src_pos >= 0
                        && (src_pos as usize) < input_length
                    {
                        let src =
                            (b * input_length
                                + src_pos as usize)
                                * in_channels
                                + c;

                        let wi =
                            c * kernel_size + k;

                        weight_grads[wi] +=
                            input[src] * g;

                        input_grads[src] +=
                            weights[wi] * g;
                    }
                }
            }
        }
    }

    let mut parameter_grads =
        Vec::with_capacity(
            weight_grads.len()
                + bias_grads.len(),
        );

    for i in 0..weight_grads.len() {
        parameter_grads.push((
            weight_handles[i],
            weight_grads[i],
        ));
    }

    for i in 0..bias_grads.len() {
        parameter_grads.push((
            bias_handles[i],
            bias_grads[i],
        ));
    }

    crate::add_handle_grads(
        &parameter_grads
    );

    input_grads
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

    pub fn from_layers(num_inputs: usize, specs: &[LayerSpec]) -> Self {
        let (layers, _) = build_layers(num_inputs, specs);
        Self { layers }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut current = inputs.to_vec();

        for layer in &self.layers {
            match layer {
                Layer::Dense(layer) => {
                    current = layer.forward(&current);
                }

                Layer::Conv1D(_)
                | Layer::Residual(_)
                | Layer::DepthwiseConv1D(_)
                | Layer::GroupedConv1D(_)
                | Layer::LowRankPointwise(_)
                | Layer::ChannelScale(_) => {
                    panic!(
                        "This layer type is only supported by forward_batch"
                    );
                }
            }
        }

        current
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
            version: 1,
            layers: self.layers
                .iter()
                .map(save_layer)
                .collect(),
        }
    }

    pub fn load(saved: &SavedMLP) -> Self {
        assert_eq!(
            saved.version,
            1,
            "Unsupported MLP save version: {}",
            saved.version
        );

        Self {
            layers: saved
                .layers
                .iter()
                .map(load_layer)
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
            let input_size = layer
                .neurons
                .first()
                .map(|n| n.weights.len())
                .unwrap_or(0);

            SavedLayer::Dense {
                input_size,
                output_size: layer.neurons.len(),
                activation: layer.activation.clone(),

                weights: layer
                    .neurons
                    .iter()
                    .flat_map(|neuron| {
                        neuron.weights
                            .iter()
                            .map(|weight| weight.data())
                    })
                    .collect(),

                biases: layer
                    .neurons
                    .iter()
                    .map(|neuron| neuron.bias.data())
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
    }
}

fn load_layer(saved: &SavedLayer) -> Layer {
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

            let neurons = (0..*output_size)
                .map(|o| {
                    let weights = (0..*input_size)
                        .map(|i| {
                            Tensor::new(
                                weights[o * *input_size + i]
                            )
                        })
                        .collect();

                    Neuron {
                        weights,
                        bias: Tensor::new(biases[o]),
                        is_output: matches!(
                            activation,
                            Activation::None
                        ),
                    }
                })
                .collect::<Vec<_>>();

            let fused_weights = neurons
                .iter()
                .flat_map(|neuron| {
                    neuron.weights
                        .iter()
                        .map(|weight| weight.handle)
                })
                .collect();

            let fused_biases = neurons
                .iter()
                .map(|neuron| neuron.bias.handle)
                .collect();

            Layer::Dense(DenseLayer {
                neurons,
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
                .map(load_layer)
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
    }
}

fn conv1d_forward(
    layer: &Conv1DLayer,
    input: &[f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    let kernel_width = layer.kernel_size * layer.in_channels;
    let mut output = vec![0.0; batch_size * output_length * layer.out_channels];

    let mut weights = vec![0.0; layer.weight_handles.len()];
    let mut biases = vec![0.0; layer.bias_handles.len()];

    crate::handle_data_slice(&layer.weight_handles, &mut weights);
    crate::handle_data_slice(&layer.bias_handles, &mut biases);

    let mut col = vec![0.0; CONV1D_TILE * output_length * kernel_width];

    for batch_start in (0..batch_size).step_by(CONV1D_TILE) {
        let tile_batch = (batch_size - batch_start).min(CONV1D_TILE);
        let rows = tile_batch * output_length;

        for b in 0..tile_batch {
            for out_pos in 0..output_length {
                let row = b * output_length + out_pos;
                let row_start = row * kernel_width;

                for k in 0..layer.kernel_size {
                    let pos = out_pos * layer.stride + k;

                    let src_pos = if layer.causal {
                        // Causal convolution:
                        // output[t] may only read input positions <= t.
                        pos as isize - (layer.kernel_size - 1) as isize
                    } else {
                        // Existing symmetric-padding behavior.
                        pos as isize - layer.padding as isize
                    };

                    let dst = row_start + k * layer.in_channels;

                    if src_pos >= 0 && (src_pos as usize) < input_length {
                        let src = ((batch_start + b) * input_length + src_pos as usize)
                            * layer.in_channels;

                        col[dst..dst + layer.in_channels]
                            .copy_from_slice(&input[src..src + layer.in_channels]);
                    } else {
                        col[dst..dst + layer.in_channels].fill(0.0);
                    }
                }
            }
        }

        let output_offset = batch_start * output_length * layer.out_channels;
        let output_tile = &mut output[
            output_offset..output_offset + rows * layer.out_channels
            ];

        crate::batched::gemm(
            &col[..rows * kernel_width],
            &weights,
            output_tile,
            rows,
            layer.out_channels,
            kernel_width,
            Transpose::None,
            Transpose::Ordinary,
            kernel_width,
            kernel_width,
            layer.out_channels,
        );

        for row in 0..rows {
            for oc in 0..layer.out_channels {
                let i = row * layer.out_channels + oc;
                output_tile[i] = layer.activation.apply(output_tile[i] + biases[oc]);
            }
        }
    }

    output
}

fn conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
) -> Vec<f32> {
    let kernel_width = kernel_size * in_channels;

    let mut weights = vec![0.0; weight_handles.len()];
    crate::handle_data_slice(weight_handles, &mut weights);

    let mut weight_grads = vec![0.0; out_channels * kernel_width];
    let mut bias_grads = vec![0.0; out_channels];
    let mut input_grads = vec![0.0; input.len()];

    for i in 0..grad.len() {
        activation.backward(output[i], &mut grad[i]);
    }

    let mut col = vec![0.0; CONV1D_TILE * output_length * kernel_width];
    let mut col_grads = vec![0.0; CONV1D_TILE * output_length * kernel_width];

    for batch_start in (0..batch_size).step_by(CONV1D_TILE) {
        let tile_batch = (batch_size - batch_start).min(CONV1D_TILE);
        let rows = tile_batch * output_length;

        for b in 0..tile_batch {
            for out_pos in 0..output_length {
                let row = b * output_length + out_pos;
                let row_start = row * kernel_width;

                for k in 0..kernel_size {
                    let pos = out_pos * stride + k;

                    let src_pos = if causal {
                        pos as isize - (kernel_size - 1) as isize
                    } else {
                        pos as isize - padding as isize
                    };

                    let dst = row_start + k * in_channels;

                    if src_pos >= 0 && (src_pos as usize) < input_length {
                        let src = ((batch_start + b) * input_length + src_pos as usize)
                            * in_channels;

                        col[dst..dst + in_channels]
                            .copy_from_slice(&input[src..src + in_channels]);
                    } else {
                        col[dst..dst + in_channels].fill(0.0);
                    }
                }
            }
        }

        let grad_offset = batch_start * output_length * out_channels;
        let grad_tile =
            &grad[grad_offset..grad_offset + rows * out_channels];

        crate::batched::gemm_beta(
            grad_tile,
            &col[..rows * kernel_width],
            &mut weight_grads,
            out_channels,
            kernel_width,
            rows,
            Transpose::Ordinary,
            Transpose::None,
            out_channels,
            kernel_width,
            kernel_width,
            1.0,
        );

        for row in 0..rows {
            for oc in 0..out_channels {
                bias_grads[oc] += grad_tile[row * out_channels + oc];
            }
        }

        crate::batched::gemm(
            grad_tile,
            &weights,
            &mut col_grads[..rows * kernel_width],
            rows,
            kernel_width,
            out_channels,
            Transpose::None,
            Transpose::None,
            out_channels,
            kernel_width,
            kernel_width,
        );

        for b in 0..tile_batch {
            for out_pos in 0..output_length {
                let row = b * output_length + out_pos;

                for k in 0..kernel_size {
                    let pos = out_pos * stride + k;

                    let src_pos = if causal {
                        pos as isize - (kernel_size - 1) as isize
                    } else {
                        pos as isize - padding as isize
                    };

                    if src_pos >= 0 && (src_pos as usize) < input_length {
                        let dst = ((batch_start + b) * input_length + src_pos as usize)
                            * in_channels;
                        let src = row * kernel_width + k * in_channels;

                        for c in 0..in_channels {
                            input_grads[dst + c] += col_grads[src + c];
                        }
                    }
                }
            }
        }
    }

    let mut parameter_grads =
        Vec::with_capacity(weight_grads.len() + bias_grads.len());

    for i in 0..weight_grads.len() {
        parameter_grads.push((weight_handles[i], weight_grads[i]));
    }

    for i in 0..bias_grads.len() {
        parameter_grads.push((bias_handles[i], bias_grads[i]));
    }

    crate::add_handle_grads(&parameter_grads);

    input_grads
}
