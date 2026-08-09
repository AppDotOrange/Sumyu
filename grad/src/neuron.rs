use crate::{Tensor, TensorHandle};
use rand_distr::{Distribution, Normal as NormalDist};
use serde::{Serialize, Deserialize};

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
    fused_weights: Vec<TensorHandle>,
    fused_biases: Vec<TensorHandle>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_outputs: usize, is_output: bool) -> Self {
        let neurons: Vec<Neuron> = (0..num_outputs)
            .map(|_| Neuron::new(num_inputs, is_output))
            .collect();

        let fused_weights = neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.handle))
            .collect();

        let fused_biases = neurons
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
            &self.fused_weights,
            &input_handles,
            &self.fused_biases,
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

        let fused_weights = neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.handle))
            .collect();

        let fused_biases = neurons
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
