use std::sync::Arc;

use cblas::{Layout, Transpose};
use rand_distr::{Distribution, Normal as NormalDist};
use serde::{Deserialize, Serialize};

use crate::{Tensor, TensorHandle};

const CONV1D_TILE: usize = 32;

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
        activation: Activation,
        weights: Vec<f32>,
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
        activation: Activation,
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
        output: Vec<f32>,
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
        weight_handles: Arc<[TensorHandle]>,
        bias_handles: Arc<[TensorHandle]>,
        activation: Activation,
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

    pub fn fwd(&self, inputs: &[Tensor]) -> Tensor {
        Tensor::linear_neuron(
            &self.weights,
            inputs,
            &self.bias,
            inputs.len(),
            !self.is_output,
        )
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
            activation,
            weights,
            biases,
            weight_handles,
            bias_handles,
        }
    }

    #[inline]
    fn output_length(&self, input_length: usize) -> usize {
        assert!(input_length + 2 * self.padding >= self.kernel_size);
        (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1
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
            activation: self.activation.clone(),
        }
    }
}

#[derive(Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Conv1D(Conv1DLayer),
}

impl Layer {
    fn spec(&self) -> LayerSpec {
        match self {
            Layer::Dense(layer) => layer.spec(),
            Layer::Conv1D(layer) => layer.spec(),
        }
    }
}

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
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
        let mut layers = Vec::with_capacity(specs.len());
        let mut current_size = num_inputs;

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
                    activation,
                } => {
                    assert_eq!(current_size % in_channels, 0);
                    Layer::Conv1D(Conv1DLayer::new(
                        *in_channels,
                        *out_channels,
                        *kernel_size,
                        *stride,
                        *padding,
                        activation.clone(),
                    ))
                }
            };

            current_size = match &layer {
                Layer::Dense(layer) => layer.neurons.len(),
                Layer::Conv1D(layer) => {
                    let input_length = current_size / layer.in_channels;
                    layer.output_length(input_length) * layer.out_channels
                }
            };

            layers.push(layer);
        }

        Self { layers }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut current = inputs.to_vec();

        for layer in &self.layers {
            match layer {
                Layer::Dense(layer) => current = layer.forward(&current),
                Layer::Conv1D(_) => {
                    panic!("Conv1D is only supported by forward_batch");
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
        debug_assert_eq!(input.len(), batch_size * input_size);

        let mut current = input.to_vec();
        let mut current_size = input_size;
        let mut layers = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            match layer {
                Layer::Dense(layer) => {
                    let output_size = layer.neurons.len();

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

                    let mut output = vec![0.0; batch_size * output_size];

                    unsafe {
                        cblas::sgemm(
                            Layout::RowMajor,
                            Transpose::None,
                            Transpose::Ordinary,
                            batch_size as i32,
                            output_size as i32,
                            current_size as i32,
                            1.0,
                            &current,
                            current_size as i32,
                            &weights,
                            current_size as i32,
                            0.0,
                            &mut output,
                            output_size as i32,
                        );
                    }

                    for b in 0..batch_size {
                        let base = b * output_size;
                        for o in 0..output_size {
                            let i = base + o;
                            output[i] = layer.activation.apply(output[i] + biases[o]);
                        }
                    }

                    layers.push(BatchLayerCache::Dense {
                        input_size: current_size,
                        output_size,
                        input: current,
                        output: output.clone(),
                        weights,
                        weight_handles: Arc::clone(&layer.fused_weights),
                        bias_handles: Arc::clone(&layer.fused_biases),
                        activation: layer.activation.clone(),
                    });

                    current = output;
                    current_size = output_size;
                }

                Layer::Conv1D(layer) => {
                    assert_eq!(current_size % layer.in_channels, 0);

                    let input_length = current_size / layer.in_channels;
                    let output_length = layer.output_length(input_length);

                    let output = conv1d_forward(
                        layer,
                        &current,
                        batch_size,
                        input_length,
                        output_length,
                    );

                    layers.push(BatchLayerCache::Conv1D {
                        input: current,
                        output: output.clone(),
                        input_length,
                        output_length,
                        in_channels: layer.in_channels,
                        out_channels: layer.out_channels,
                        kernel_size: layer.kernel_size,
                        stride: layer.stride,
                        padding: layer.padding,
                        weight_handles: Arc::clone(&layer.weight_handles),
                        bias_handles: Arc::clone(&layer.bias_handles),
                        activation: layer.activation.clone(),
                    });

                    current = output;
                    current_size = output_length * layer.out_channels;
                }
            }
        }

        BatchForward {
            output: current,
            batch_size,
            output_size: current_size,
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

        let mut grad = output_grads.to_vec();
        let batch_size = forward.batch_size;

        for layer in forward.layers.iter().rev() {
            match layer {
                BatchLayerCache::Dense {
                    input_size,
                    output_size,
                    input,
                    output,
                    weights,
                    weight_handles,
                    bias_handles,
                    activation,
                } => {
                    for b in 0..batch_size {
                        let base = b * *output_size;
                        for o in 0..*output_size {
                            activation.backward(output[base + o], &mut grad[base + o]);
                        }
                    }

                    let mut weight_grads = vec![0.0; output_size * input_size];

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

                    let mut bias_grads = vec![0.0; *output_size];

                    for b in 0..batch_size {
                        let base = b * *output_size;
                        for o in 0..*output_size {
                            bias_grads[o] += grad[base + o];
                        }
                    }

                    let mut input_grads = vec![0.0; batch_size * *input_size];

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
                        Vec::with_capacity(weight_grads.len() + bias_grads.len());

                    for i in 0..weight_grads.len() {
                        parameter_grads.push((weight_handles[i], weight_grads[i]));
                    }

                    for i in 0..bias_grads.len() {
                        parameter_grads.push((bias_handles[i], bias_grads[i]));
                    }

                    crate::add_handle_grads(&parameter_grads);
                    grad = input_grads;
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
                    weight_handles,
                    bias_handles,
                    activation,
                } => {
                    grad = conv1d_backward(
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
                        weight_handles,
                        bias_handles,
                        activation,
                    );
                }
            }
        }

        grad
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for layer in &self.layers {
            match layer {
                Layer::Dense(layer) => params.extend(layer.parameters()),
                Layer::Conv1D(layer) => params.extend(layer.parameters()),
            }
        }

        params
    }

    pub fn save(&self) -> SavedMLP {
        SavedMLP {
            version: 1,
            layers: self.layers.iter().map(|layer| match layer {
                Layer::Dense(layer) => SavedLayer::Dense {
                    input_size: layer.neurons[0].weights.len(),
                    output_size: layer.neurons.len(),
                    activation: layer.activation.clone(),
                    weights: layer.neurons
                        .iter()
                        .flat_map(|n| n.weights.iter().map(|w| w.data()))
                        .collect(),
                    biases: layer.neurons.iter().map(|n| n.bias.data()).collect(),
                },

                Layer::Conv1D(layer) => SavedLayer::Conv1D {
                    in_channels: layer.in_channels,
                    out_channels: layer.out_channels,
                    kernel_size: layer.kernel_size,
                    stride: layer.stride,
                    padding: layer.padding,
                    activation: layer.activation.clone(),
                    weights: layer.weights.iter().map(|x| x.data()).collect(),
                    biases: layer.biases.iter().map(|x| x.data()).collect(),
                },
            }).collect(),
        }
    }

    pub fn load(saved: &SavedMLP) -> Self {
        assert_eq!(saved.version, 1);

        let layers = saved.layers.iter().map(|layer| match layer {
            SavedLayer::Dense {
                input_size,
                output_size,
                activation,
                weights,
                biases,
            } => {
                assert_eq!(weights.len(), input_size * output_size);
                assert_eq!(biases.len(), *output_size);

                let neurons = (0..*output_size)
                    .map(|o| Neuron {
                        weights: (0..*input_size)
                            .map(|i| Tensor::new(weights[o * *input_size + i]))
                            .collect(),
                        bias: Tensor::new(biases[o]),
                        is_output: matches!(activation, Activation::None),
                    })
                    .collect::<Vec<_>>();

                let fused_weights = neurons
                    .iter()
                    .flat_map(|n| n.weights.iter().map(|w| w.handle))
                    .collect();

                let fused_biases = neurons.iter().map(|n| n.bias.handle).collect();

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
                activation,
                weights,
                biases,
            } => {
                assert_eq!(
                    weights.len(),
                    in_channels * out_channels * kernel_size
                );
                assert_eq!(biases.len(), *out_channels);

                let weights = weights.iter()
                    .map(|&x| Tensor::new(x))
                    .collect::<Vec<_>>();

                let biases = biases.iter()
                    .map(|&x| Tensor::new(x))
                    .collect::<Vec<_>>();

                let weight_handles = weights.iter().map(|x| x.handle).collect();
                let bias_handles = biases.iter().map(|x| x.handle).collect();

                Layer::Conv1D(Conv1DLayer {
                    in_channels: *in_channels,
                    out_channels: *out_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    activation: activation.clone(),
                    weights,
                    biases,
                    weight_handles,
                    bias_handles,
                })
            }
        }).collect();

        Self { layers }
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters().len()
    }

    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        self.layers.iter().map(|layer| layer.spec()).collect()
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

    let weights = layer.weight_handles
        .iter()
        .map(|&h| crate::handle_data(h))
        .collect::<Vec<_>>();

    let biases = layer.bias_handles
        .iter()
        .map(|&h| crate::handle_data(h))
        .collect::<Vec<_>>();

    let mut col = vec![0.0; CONV1D_TILE * output_length * kernel_width];

    for batch_start in (0..batch_size).step_by(CONV1D_TILE) {
        let tile_batch = (batch_size - batch_start).min(CONV1D_TILE);
        let rows = tile_batch * output_length;

        col[..rows * kernel_width].fill(0.0);

        for b in 0..tile_batch {
            for out_pos in 0..output_length {
                let row = b * output_length + out_pos;

                for k in 0..layer.kernel_size {
                    let pos = out_pos * layer.stride + k;
                    let src_pos = pos as isize - layer.padding as isize;

                    if src_pos >= 0 && (src_pos as usize) < input_length {
                        let src = ((batch_start + b) * input_length + src_pos as usize)
                            * layer.in_channels;
                        let dst = row * kernel_width + k * layer.in_channels;

                        col[dst..dst + layer.in_channels]
                            .copy_from_slice(&input[src..src + layer.in_channels]);
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
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
) -> Vec<f32> {
    let kernel_width = kernel_size * in_channels;

    let weights = weight_handles
        .iter()
        .map(|&h| crate::handle_data(h))
        .collect::<Vec<_>>();

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

        col[..rows * kernel_width].fill(0.0);

        for b in 0..tile_batch {
            for out_pos in 0..output_length {
                let row = b * output_length + out_pos;

                for k in 0..kernel_size {
                    let pos = out_pos * stride + k;
                    let src_pos = pos as isize - padding as isize;

                    if src_pos >= 0 && (src_pos as usize) < input_length {
                        let src = ((batch_start + b) * input_length + src_pos as usize)
                            * in_channels;
                        let dst = row * kernel_width + k * in_channels;

                        col[dst..dst + in_channels]
                            .copy_from_slice(&input[src..src + in_channels]);
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
                    let src_pos = pos as isize - padding as isize;

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
