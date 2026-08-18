use std::sync::Arc;
use crate::{Tensor, TensorHandle};
use rand_distr::{Distribution, Normal as NormalDist};
use serde::{Serialize, Deserialize};
use cblas::{Layout, Transpose};

pub(crate) struct BatchLayerCache {
    pub input_size: usize,
    pub output_size: usize,

    pub input: Vec<f32>,
    pub output: Vec<f32>,

    pub weights: Vec<f32>,

    pub weight_handles: Arc<[TensorHandle]>,
    pub bias_handles: Arc<[TensorHandle]>,

    pub relu: bool,
}

pub(crate) struct BatchForward {
    pub output: Vec<f32>,
    pub batch_size: usize,
    pub output_size: usize,
    pub layers: Vec<BatchLayerCache>,
}

#[derive(Serialize, Deserialize)]
pub struct SavedNeuron {
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Serialize, Deserialize)]
pub struct SavedLayer {
    neurons: Vec<SavedNeuron>,
}

#[derive(Serialize, Deserialize)]
pub struct SavedMLP {
    layers: Vec<SavedLayer>,
}

#[derive(Deserialize)]
struct OldSavedNeuron {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Deserialize)]
struct OldSavedLayer {
    neurons: Vec<OldSavedNeuron>,
}

#[derive(Deserialize)]
pub(crate) struct OldSavedMLP {
    layers: Vec<OldSavedLayer>,
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
        let std_dev = (2.0 / (num_inputs as f32)).sqrt();
        let normal = NormalDist::new(0.0, std_dev).expect("Invalid standard deviation");

        let weights: Vec<Tensor> = (0..num_inputs)
            .map(|_| {
                let val = normal.sample(&mut rng);
                Tensor::new(val)
            })
            .collect();

        let bias = Tensor::new(0.1);

        Neuron {
            weights,
            bias,
            is_output,
        }
    }

    pub fn fwd(
        &self,
        inputs: &[Tensor],
    ) -> Tensor {

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

    pub fn save(&self) -> SavedNeuron {
        SavedNeuron {
            weights: self.weights
                .iter()
                .map(|x| x.data())
                .collect(),

            bias: self.bias.data(),
        }
    }

    pub fn load(saved: &SavedNeuron) -> Self {
        Neuron {
            weights: saved.weights
                .iter()
                .map(|x| Tensor::new(*x))
                .collect(),

            bias: Tensor::new(saved.bias),

            is_output: false,
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
    fused_weights: Arc<[TensorHandle]>,
    fused_biases: Arc<[TensorHandle]>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_outputs: usize, is_output: bool) -> Self {
        let neurons: Vec<Neuron> = (0..num_outputs)
            .map(|_| Neuron::new(num_inputs, is_output))
            .collect();

        let fused_weights: Arc<[TensorHandle]> = neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.handle))
            .collect();

        let fused_biases: Arc<[TensorHandle]> = neurons
            .iter()
            .map(|n| n.bias.handle)
            .collect();

        Layer {
            neurons,
            fused_weights,
            fused_biases,
        }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let input_handles: Vec<TensorHandle> =
            inputs.iter().map(|x| x.handle).collect();

        Tensor::fused_layer(
            Arc::clone(&self.fused_weights),
            &input_handles,
            Arc::clone(&self.fused_biases),
            !self.neurons[0].is_output,
        )
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }

    pub fn save(&self) -> SavedLayer {
        SavedLayer {
            neurons: self.neurons
                .iter()
                .map(|n| n.save())
                .collect(),
        }
    }

    pub fn load(saved: &SavedLayer, is_output: bool) -> Self {
        let neurons: Vec<Neuron> = saved.neurons
            .iter()
            .map(|n| {
                let mut neuron = Neuron::load(n);
                neuron.is_output = is_output;
                neuron
            })
            .collect();

        let fused_weights: Arc<[TensorHandle]> = neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.handle))
            .collect();

        let fused_biases: Arc<[TensorHandle]> = neurons
            .iter()
            .map(|n| n.bias.handle)
            .collect();

        Layer {
            neurons,
            fused_weights,
            fused_biases,
        }
    }
}

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(num_inputs: usize, layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut input_size = num_inputs;

        for (i, &output_size) in layer_sizes.iter().enumerate() {
            let is_last_layer = i == layer_sizes.len() - 1;
            layers.push(Layer::new(input_size, output_size, is_last_layer));
            input_size = output_size;
        }

        MLP { layers }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut current = Vec::with_capacity(inputs.len());
        current.extend_from_slice(inputs);

        for layer in &self.layers {
            current = layer.forward(&current);
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

        let mut current = input.to_vec();
        let mut current_size = input_size;

        let mut layers = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let output_size = layer.neurons.len();

            debug_assert_eq!(
                layer.fused_weights.len(),
                output_size * current_size
            );

            let mut weights =
                Vec::with_capacity(output_size * current_size);

            for &handle in layer.fused_weights.iter() {
                weights.push(crate::handle_data(handle));
            }

            let mut biases =
                Vec::with_capacity(output_size);

            for &handle in layer.fused_biases.iter() {
                biases.push(crate::handle_data(handle));
            }

            let mut output =
                vec![0.0f32; batch_size * output_size];

            // X [batch × input]
            // Wᵀ [input × output]
            //
            // Y = X Wᵀ
            //
            // Our W is stored as:
            // [output × input]
            //
            // BLAS therefore computes:
            //
            // C = Wᵀ * Xᵀ
            //
            // but C is naturally column-major for that formulation.
            //
            // Instead, use row-major:
            //
            // C = X * Wᵀ
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

            // Bias + activation.
            for b in 0..batch_size {
                let base = b * output_size;

                for o in 0..output_size {
                    let idx = base + o;

                    output[idx] += biases[o];

                    if layer.neurons[0].is_output == false {
                        if output[idx] <= 0.0 {
                            output[idx] *= 0.01;
                        }
                    }
                }
            }

            layers.push(BatchLayerCache {
                input_size: current_size,
                output_size,
                input: current,
                output: output.clone(),
                weights,
                weight_handles: Arc::clone(&layer.fused_weights),
                bias_handles: Arc::clone(&layer.fused_biases),
                relu: !layer.neurons[0].is_output,
            });

            current = output;
            current_size = output_size;
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

        let batch_size = forward.batch_size;

        let mut grad = output_grads.to_vec();

        // Process layers backwards.
        for layer in forward.layers.iter().rev() {
            let input_size = layer.input_size;
            let output_size = layer.output_size;

            // ---------------------------------------------------------
            // Apply activation derivative.
            // ---------------------------------------------------------

            if layer.relu {
                for b in 0..batch_size {
                    let base = b * output_size;

                    for o in 0..output_size {
                        let idx = base + o;

                        if layer.output[idx] <= 0.0 {
                            grad[idx] *= 0.01;
                        }
                    }
                }
            }

            // ---------------------------------------------------------
            // dW = dYᵀ X
            //
            // dY = [batch × output]
            // X  = [batch × input]
            //
            // dW = [output × input]
            // ---------------------------------------------------------

            let mut weight_grads =
                vec![0.0f32; output_size * input_size];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    output_size as i32,
                    input_size as i32,
                    batch_size as i32,
                    1.0,
                    &grad,
                    output_size as i32,
                    &layer.input,
                    input_size as i32,
                    0.0,
                    &mut weight_grads,
                    input_size as i32,
                );
            }

            // ---------------------------------------------------------
            // db = sum(dY)
            // ---------------------------------------------------------

            let mut bias_grads =
                vec![0.0f32; output_size];

            for b in 0..batch_size {
                let base = b * output_size;

                for o in 0..output_size {
                    bias_grads[o] += grad[base + o];
                }
            }

            // ---------------------------------------------------------
            // dX = dY W
            //
            // dY [batch × output]
            // W  [output × input]
            //
            // dX [batch × input]
            // ---------------------------------------------------------

            let mut input_grads =
                vec![0.0f32; batch_size * input_size];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    batch_size as i32,
                    input_size as i32,
                    output_size as i32,
                    1.0,
                    &grad,
                    output_size as i32,
                    &layer.weights,
                    input_size as i32,
                    0.0,
                    &mut input_grads,
                    input_size as i32,
                );
            }

            // ---------------------------------------------------------
            // Accumulate parameter gradients.
            // ---------------------------------------------------------

            let mut parameter_grads =
                Vec::with_capacity(
                    weight_grads.len() + bias_grads.len()
                );

            for i in 0..weight_grads.len() {
                parameter_grads.push((
                    layer.weight_handles[i],
                    weight_grads[i],
                ));
            }

            for i in 0..bias_grads.len() {
                parameter_grads.push((
                    layer.bias_handles[i],
                    bias_grads[i],
                ));
            }

            crate::add_handle_grads(&parameter_grads);

            grad = input_grads;
        }

        grad
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for layer in &self.layers {
            params.extend(layer.parameters());
        }

        params
    }

    pub fn save(&self) -> SavedMLP {
        SavedMLP {
            layers: self.layers
                .iter()
                .map(|l| l.save())
                .collect(),
        }
    }

    pub fn load(saved: &SavedMLP) -> Self {
        let last = saved.layers.len() - 1;

        MLP {
            layers: saved.layers
                .iter()
                .enumerate()
                .map(|(i, layer)| {
                    Layer::load(layer, i == last)
                })
                .collect(),
        }
    }
}

impl From<OldSavedNeuron> for SavedNeuron {
    fn from(old: OldSavedNeuron) -> Self {
        SavedNeuron {
            weights: old.weights.into_iter()
                .map(|x| x as f32)
                .collect(),
            bias: old.bias as f32,
        }
    }
}

impl From<OldSavedLayer> for SavedLayer {
    fn from(old: OldSavedLayer) -> Self {
        SavedLayer {
            neurons: old.neurons.into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<OldSavedMLP> for SavedMLP {
    fn from(old: OldSavedMLP) -> Self {
        SavedMLP {
            layers: old.layers.into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}
