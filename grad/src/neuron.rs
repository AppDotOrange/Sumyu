use crate::Tensor;
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
}

impl Layer {
    pub fn new(num_inputs: usize, num_outputs: usize, is_output: bool) -> Self {
        let neurons = (0..num_outputs)
            .map(|_| Neuron::new(num_inputs, is_output))
            .collect();

        Layer { neurons }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut weights = Vec::new();
        let mut biases = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            weights.extend_from_slice(&neuron.weights);
            biases.push(neuron.bias);
        }

        Tensor::fused_layer(
            &weights,
            inputs,
            &biases,
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
        Layer {
            neurons: saved.neurons
                .iter()
                .map(|n| {
                    let mut neuron = Neuron::load(n);
                    neuron.is_output = is_output;
                    neuron
                })
                .collect(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clear_tape_after;

    #[test]
    fn fused_layer_matches_individual_neurons() {
        let boundary = crate::tape_len();

        let inputs = vec![
            Tensor::new(0.5),
            Tensor::new(-0.2),
            Tensor::new(0.8),
        ];

        let layer = Layer::new(3, 2, false);

        // Old implementation: each neuron separately.
        let old_outputs: Vec<Tensor> = layer
            .neurons
            .iter()
            .map(|n| n.fwd(&inputs))
            .collect();

        // Fused implementation.
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for neuron in &layer.neurons {
            weights.extend_from_slice(&neuron.weights);
            biases.push(neuron.bias);
        }

        let fused_outputs =
            Tensor::fused_layer(
                &weights,
                &inputs,
                &biases,
                true,
            );

        for (old, fused) in
            old_outputs.iter().zip(fused_outputs.iter())
        {
            let diff = (old.data() - fused.data()).abs();

            assert!(
                diff < 1e-6,
                "Output mismatch: old={}, fused={}, diff={}",
                old.data(),
                fused.data(),
                diff
            );
        }

        clear_tape_after(boundary);
    }

    #[test]
    fn fused_layer_gradients_match_individual_neurons() {
        let boundary = crate::tape_len();

        // Use fixed values so both implementations receive
        // exactly the same parameters and inputs.
        let input_values = [0.5_f32, -0.2, 0.8];

        let layer = Layer::new(3, 2, false);

        let weight_values: Vec<f32> = layer
            .neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.data()))
            .collect();

        let bias_values: Vec<f32> = layer
            .neurons
            .iter()
            .map(|n| n.bias.data())
            .collect();

        // ============================================================
        // OLD IMPLEMENTATION
        // ============================================================

        let old_inputs: Vec<Tensor> = input_values
            .iter()
            .map(|&x| Tensor::new(x))
            .collect();

        let old_layer = Layer {
            neurons: layer
                .neurons
                .iter()
                .enumerate()
                .map(|(i, neuron)| {
                    Neuron {
                        weights: neuron
                            .weights
                            .iter()
                            .map(|w| Tensor::new(w.data()))
                            .collect(),
                        bias: Tensor::new(bias_values[i]),
                        is_output: neuron.is_output,
                    }
                })
                .collect(),
        };

        let old_outputs: Vec<Tensor> = old_layer
            .neurons
            .iter()
            .map(|n| n.fwd(&old_inputs))
            .collect();

        let old_loss = old_outputs[0]
            .add(&old_outputs[1]);

        old_loss.backward();

        let old_input_grads: Vec<f32> = old_inputs
            .iter()
            .map(|x| x.grad())
            .collect();

        let old_weight_grads: Vec<f32> = old_layer
            .neurons
            .iter()
            .flat_map(|n| n.weights.iter().map(|w| w.grad()))
            .collect();

        let old_bias_grads: Vec<f32> = old_layer
            .neurons
            .iter()
            .map(|n| n.bias.grad())
            .collect();

        let old_outputs_data: Vec<f32> = old_outputs
            .iter()
            .map(|x| x.data())
            .collect();

        // ============================================================
        // Clear the OLD graph.
        // Everything above this point has been copied into plain f32s.
        // ============================================================

        crate::clear_tape_after(boundary);

        // ============================================================
        // FUSED IMPLEMENTATION
        // ============================================================

        let fused_inputs: Vec<Tensor> = input_values
            .iter()
            .map(|&x| Tensor::new(x))
            .collect();

        let fused_weights: Vec<Tensor> = weight_values
            .iter()
            .map(|&x| Tensor::new(x))
            .collect();

        let fused_biases: Vec<Tensor> = bias_values
            .iter()
            .map(|&x| Tensor::new(x))
            .collect();

        let fused_outputs = Tensor::fused_layer(
            &fused_weights,
            &fused_inputs,
            &fused_biases,
            true,
        );

        let fused_loss = fused_outputs[0]
            .add(&fused_outputs[1]);

        fused_loss.backward();

        println!(
            "FUSED outputs: {:?}",
            fused_outputs
                .iter()
                .map(|x| x.data())
                .collect::<Vec<_>>()
        );

        println!(
            "FUSED output grads: {:?}",
            fused_outputs
                .iter()
                .map(|x| x.grad())
                .collect::<Vec<_>>()
        );

        let fused_input_grads: Vec<f32> = fused_inputs
            .iter()
            .map(|x| x.grad())
            .collect();

        let fused_weight_grads: Vec<f32> = fused_weights
            .iter()
            .map(|x| x.grad())
            .collect();

        let fused_bias_grads: Vec<f32> = fused_biases
            .iter()
            .map(|x| x.grad())
            .collect();

        let fused_outputs_data: Vec<f32> = fused_outputs
            .iter()
            .map(|x| x.data())
            .collect();

        // ============================================================
        // COMPARE OUTPUTS
        // ============================================================

        for (i, (old, fused)) in old_outputs_data
            .iter()
            .zip(fused_outputs_data.iter())
            .enumerate()
        {
            assert!(
                (old - fused).abs() < 1e-6,
                "Output mismatch at {}: old={}, fused={}",
                i,
                old,
                fused
            );
        }

        // ============================================================
        // COMPARE INPUT GRADIENTS
        // ============================================================

        for (i, (old, fused)) in old_input_grads
            .iter()
            .zip(fused_input_grads.iter())
            .enumerate()
        {
            assert!(
                (old - fused).abs() < 1e-6,
                "Input gradient mismatch at {}: old={}, fused={}",
                i,
                old,
                fused
            );
        }

        // ============================================================
        // COMPARE WEIGHT GRADIENTS
        // ============================================================

        for (i, (old, fused)) in old_weight_grads
            .iter()
            .zip(fused_weight_grads.iter())
            .enumerate()
        {
            assert!(
                (old - fused).abs() < 1e-6,
                "Weight gradient mismatch at {}: old={}, fused={}",
                i,
                old,
                fused
            );
        }

        // ============================================================
        // COMPARE BIAS GRADIENTS
        // ============================================================

        for (i, (old, fused)) in old_bias_grads
            .iter()
            .zip(fused_bias_grads.iter())
            .enumerate()
        {
            assert!(
                (old - fused).abs() < 1e-6,
                "Bias gradient mismatch at {}: old={}, fused={}",
                i,
                old,
                fused
            );
        }

        crate::clear_tape_after(boundary);
    }
}
