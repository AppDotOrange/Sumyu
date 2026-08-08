pub mod neuron;
pub mod trainer;
pub mod helper;
pub mod fnn_lm;
pub mod chatter;
pub mod embeddings;

use std::cell::RefCell;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
#[derive(Serialize, Deserialize)]
enum Op {
    Leaf,
    Add,
    Sub,
    Mul,
    Fma {
        weight: f32,
        input: f32,
    },
    Linear {
        input_size: usize,
        relu: bool,
        pre_activation: f32,
    },
    Pow(f32),
    Sigmoid,
    Relu,
    SoftmaxCrossEntropy {
        probs: Vec<f32>,
        target: usize,
    },
    SoftmaxCrossEntropyOld {
        probs: Vec<f32>,
        targets: Vec<f32>,
    },
}

struct TensorData {
    data: f32,
    grad: f32,
    prev: Vec<TensorHandle>,
    op: Op,
}

#[derive(Copy, Clone, Debug)]
struct TensorHandle {
    node: usize,
    index: usize,
}

struct FusedLayerData {
    outputs: Vec<f32>,
    grads: Vec<f32>,

    inputs: Vec<TensorHandle>,
    weights: Vec<TensorHandle>,
    biases: Vec<TensorHandle>,

    input_size: usize,
    output_size: usize,
    relu: bool,
}

enum Node {
    Scalar(TensorData),
    FusedLayer(FusedLayerData),
}

struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    fn alloc(
        &mut self,
        data: f32,
        prev: Vec<TensorHandle>,
        op: Op,
    ) -> TensorHandle {
        let id = self.nodes.len();

        self.nodes.push(Node::Scalar(
            TensorData {
                data,
                grad: 0.0,
                prev,
                op,
            }
        ));

        TensorHandle {
            node: id,
            index: 0,
        }
    }

    fn alloc_fused_layer(
        &mut self,
        data: FusedLayerData,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::FusedLayer(data));
        id
    }

    fn clear_after(&mut self, index: usize) {
        self.nodes.truncate(index);
    }
}

pub fn clear_tape_after(index: usize) {
    TAPE.with(|t| {
        t.borrow_mut()
            .clear_after(index);
    });
}

pub fn tape_len() -> usize {
    TAPE.with(|t| {
        t.borrow().nodes.len()
    })
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());

    static TOPO_VISITED: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static TOPO_STACK1: RefCell<Vec<TensorHandle>> = const { RefCell::new(Vec::new()) };
    static TOPO_STACK2: RefCell<Vec<TensorHandle>> = const { RefCell::new(Vec::new()) };
}

pub fn zero_grad_and_update(params: &[Tensor], lr: f32) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        for p in params {
            match &mut tape.nodes[p.handle.node] {
                Node::Scalar(node) => {
                    node.data -= lr * node.grad;
                    node.grad = 0.0;
                }
                Node::FusedLayer(_) => panic!("Parameter cannot be a fused-layer output.")
            }
        }
    });
}

#[derive(Copy)]
pub struct Tensor {
    handle: TensorHandle,
}

fn node_data(
    tape: &Tape,
    handle: TensorHandle,
) -> f32 {
    match &tape.nodes[handle.node] {
        Node::Scalar(node) => node.data,
        Node::FusedLayer(node) => node.outputs[handle.index],
    }
}

fn node_grad(
    tape: &Tape,
    handle: TensorHandle,
) -> f32 {
    match &tape.nodes[handle.node] {
        Node::Scalar(node) => node.grad,
        Node::FusedLayer(node) => node.grads[handle.index],
    }
}

fn add_node_grad(
    tape: &mut Tape,
    handle: TensorHandle,
    grad: f32,
) {
    match &mut tape.nodes[handle.node] {
        Node::Scalar(node) => {
            node.grad += grad;
        }

        Node::FusedLayer(node) => {
            node.grads[handle.index] += grad;
        }
    }
}

impl Tensor {
    pub fn new(data: f32) -> Self {
        let id = TAPE.with(|t| {
            t.borrow_mut().alloc(
                data,
                Vec::new(),
                Op::Leaf,
            )
        });
        Tensor {
            handle: id,
        }
    }

    fn from_op(
        data: f32,
        prev: Vec<TensorHandle>,
        op: Op,
    ) -> Self {
        let id = TAPE.with(|t| {
            t.borrow_mut().alloc(
                data,
                prev,
                op,
            )
        });
        Tensor {
            handle: id,
        }
    }

    pub fn data(&self) -> f32 {
        TAPE.with(|t| {
            match &t.borrow().nodes[self.handle.node] {
                Node::Scalar(node) => node.data,
                Node::FusedLayer(node) => node.outputs[self.handle.index],
            }
        })
    }

    pub fn grad(&self) -> f32 {
        TAPE.with(|t| {
            match &t.borrow().nodes[self.handle.node] {
                Node::Scalar(node) => node.grad,
                Node::FusedLayer(node) => node.grads[self.handle.index],
            }
        })
    }

    pub fn set_grad(&self, val: f32) {
        TAPE.with(|t| {
            match &mut t.borrow_mut().nodes[self.handle.node] {
                Node::Scalar(node) => node.grad = val,
                Node::FusedLayer(node) => node.grads[self.handle.index] = val
            }
        });
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a + b;

        let prev = vec![self.handle, other.handle];

        Tensor::from_op(data, prev, Op::Add)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a - b;

        let prev = vec![self.handle, other.handle];

        Tensor::from_op(data, prev, Op::Sub)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let left = self.data();
        let right = other.data();

        let data = left * right;

        let prev = vec![self.handle, other.handle];

        Tensor::from_op(
            data,
            prev,
            Op::Mul,
        )
    }

    pub fn fma(&self, weight: &Tensor, input: &Tensor) -> Tensor {

        let w = weight.data();
        let x = input.data();

        let data = self.data() + w * x;

        Tensor::from_op(
            data,
            vec![
                self.handle,
                weight.handle,
                input.handle,
            ],
            Op::Fma {
                weight: w,
                input: x,
            },
        )
    }

    pub fn pow(&self, n: f32) -> Tensor {
        let x = self.data();
        let data = x.powf(n);

        Tensor::from_op(data, vec![self.handle], Op::Pow(n))
    }

    pub fn sigmoid(&self) -> Tensor {
        // 1. Forward Pass: Compute 1 / (1 + e^-x)
        let data = 1.0 / (1.0 + (-self.data()).exp());

        // 2. Clone the input so we can use it in the closure

        // 4. Create the Tensor with the operation graph
        let prev = vec![self.handle];

        Tensor::from_op(data, prev, Op::Sigmoid)
    }

    pub fn linear_neuron(
        weights: &[Tensor],
        inputs: &[Tensor],
        bias: &Tensor,
        input_size: usize,
        relu: bool,
    ) -> Tensor {

        let mut sum = bias.data();


        for i in 0..input_size {
            sum += weights[i].data() * inputs[i].data();
        }


        let output =
            if relu {
                if sum > 0.0 {
                    sum
                } else {
                    sum * 0.01
                }
            }
            else {
                sum
            };


        let mut prev =
            Vec::with_capacity(
                input_size * 2 + 1
            );


        // weights
        for w in weights {
            prev.push(w.handle);
        }


        // inputs
        for x in inputs {
            prev.push(x.handle);
        }


        // bias
        prev.push(bias.handle);


        Tensor::from_op(
            output,
            prev,
            Op::Linear {
                input_size,
                relu,
                pre_activation: sum,
            }
        )
    }

    pub fn fused_layer(
        weights: &[Tensor],
        inputs: &[Tensor],
        biases: &[Tensor],
        relu: bool,
    ) -> Vec<Tensor> {
        let input_size = inputs.len();
        let output_size = biases.len();

        assert_eq!(
            weights.len(),
            input_size * output_size,
            "Fused layer weight count does not match dimensions"
        );

        let mut outputs = Vec::with_capacity(output_size);

        for o in 0..output_size {
            let mut sum = biases[o].data();

            let weight_base = o * input_size;

            for i in 0..input_size {
                sum += weights[weight_base + i].data()
                    * inputs[i].data();
            }

            let output = if relu {
                if sum > 0.0 {
                    sum
                } else {
                    sum * 0.01
                }
            } else {
                sum
            };

            outputs.push(output);
        }

        let node = TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            tape.alloc_fused_layer(
                FusedLayerData {
                    outputs,
                    grads: vec![0.0; output_size],
                    inputs: inputs.iter()
                        .map(|x| x.handle)
                        .collect(),
                    weights: weights.iter()
                        .map(|w| w.handle)
                        .collect(),
                    biases: biases.iter()
                        .map(|b| b.handle)
                        .collect(),
                    input_size,
                    output_size,
                    relu,
                }
            )
        });
        (0..output_size)
            .map(|index| Tensor {
                handle: TensorHandle {
                    node,
                    index,
                },
            })
            .collect()
    }

    fn build_reverse_top_order(&self) -> Vec<TensorHandle> {
        TAPE.with(|t| {
            let tape = t.borrow();
            let tape_size = tape.nodes.len();

            TOPO_VISITED.with(|v| {
                let mut visited = v.borrow_mut();

                visited.resize(tape_size, false);
                visited.fill(false);
            });

            TOPO_STACK1.with(|s1| {
                TOPO_STACK2.with(|s2| {
                    TOPO_VISITED.with(|v| {
                        let mut stack1 = s1.borrow_mut();
                        let mut stack2 = s2.borrow_mut();
                        let mut visited = v.borrow_mut();

                        stack1.clear();
                        stack2.clear();

                        stack1.push(self.handle);

                        while let Some(handle) = stack1.pop() {
                            let id = handle.node;

                            if visited[id] {
                                continue;
                            }

                            visited[id] = true;

                            match &tape.nodes[id] {
                                Node::Scalar(node) => {
                                    stack1.extend(
                                        node.prev.iter().rev().copied()
                                    );
                                }

                                Node::FusedLayer(node) => {
                                    stack1.extend(
                                        node.inputs.iter().rev().copied()
                                    );

                                    stack1.extend(
                                        node.weights.iter().rev().copied()
                                    );

                                    stack1.extend(
                                        node.biases.iter().rev().copied()
                                    );
                                }
                            }

                            stack2.push(handle);
                        }
                        stack2.clone()
                    })
                })
            })
        })
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            // Seed the output gradient.
            match &mut tape.nodes[self.handle.node] {
                Node::Scalar(node) => {
                    node.grad = 1.0;
                }

                Node::FusedLayer(node) => {
                    node.grads[self.handle.index] = 1.0;
                }
            }

            for &handle in topo.iter() {
                let id = handle.node;

                let grad = match &tape.nodes[id] {
                    Node::Scalar(node) => node.grad,
                    Node::FusedLayer(node) => node.grads[handle.index],
                };

                match &tape.nodes[id] {
                    Node::Scalar(node) => {
                        let prev = node.prev.clone();
                        let op = node.op.clone();

                        match op {
                            Op::Leaf => {}

                            Op::Add => {
                                add_node_grad(
                                    &mut tape,
                                    prev[0],
                                    grad,
                                );

                                add_node_grad(
                                    &mut tape,
                                    prev[1],
                                    grad,
                                );
                            }

                            Op::Sub => {
                                add_node_grad(
                                    &mut tape,
                                    prev[0],
                                    grad,
                                );

                                add_node_grad(
                                    &mut tape,
                                    prev[1],
                                    -grad,
                                );
                            }

                            Op::Mul => {
                                let a = prev[0];
                                let b = prev[1];

                                let a_data = node_data(&tape, a);
                                let b_data = node_data(&tape, b);

                                add_node_grad(
                                    &mut tape,
                                    a,
                                    grad * b_data,
                                );

                                add_node_grad(
                                    &mut tape,
                                    b,
                                    grad * a_data,
                                );
                            }

                            Op::Fma { weight, input } => {
                                add_node_grad(
                                    &mut tape,
                                    prev[0],
                                    grad,
                                );

                                add_node_grad(
                                    &mut tape,
                                    prev[1],
                                    grad * input,
                                );

                                add_node_grad(
                                    &mut tape,
                                    prev[2],
                                    grad * weight,
                                );
                            }

                            Op::Linear {
                                input_size,
                                relu,
                                pre_activation,
                            } => {
                                let mut grad_out = grad;

                                if relu && pre_activation <= 0.0 {
                                    grad_out *= 0.01;
                                }

                                let base = input_size;

                                for i in 0..base {
                                    let w_id = prev[i];
                                    let x_id = prev[base + i];

                                    let w = node_data(&tape, w_id);
                                    let x = node_data(&tape, x_id);

                                    add_node_grad(
                                        &mut tape,
                                        w_id,
                                        grad_out * x,
                                    );

                                    add_node_grad(
                                        &mut tape,
                                        x_id,
                                        grad_out * w,
                                    );
                                }

                                let bias = prev[base * 2];

                                add_node_grad(
                                    &mut tape,
                                    bias,
                                    grad_out,
                                );
                            }

                            Op::Pow(n) => {
                                let x_id = prev[0];
                                let x = node_data(&tape, x_id);

                                add_node_grad(
                                    &mut tape,
                                    x_id,
                                    grad * n * x.powf(n - 1.0),
                                );
                            }

                            Op::Sigmoid => {
                                let x_id = prev[0];
                                let y = node_data(&tape, handle);

                                add_node_grad(
                                    &mut tape,
                                    x_id,
                                    grad * y * (1.0 - y),
                                );
                            }

                            Op::Relu => {
                                let x_id = prev[0];
                                let x = node_data(&tape, x_id);

                                let local_grad = if x > 0.0 {
                                    1.0
                                } else {
                                    0.01
                                };

                                add_node_grad(
                                    &mut tape,
                                    x_id,
                                    grad * local_grad,
                                );
                            }

                            Op::SoftmaxCrossEntropyOld {
                                probs,
                                targets,
                            } => {
                                for i in 0..prev.len() {
                                    let local_grad =
                                        probs[i] - targets[i];

                                    add_node_grad(
                                        &mut tape,
                                        prev[i],
                                        grad * local_grad,
                                    );
                                }
                            }

                            Op::SoftmaxCrossEntropy {
                                probs,
                                target,
                            } => {
                                for (i, &prob) in probs.iter()
                                    .enumerate()
                                    .take(prev.len())
                                {
                                    let mut local_grad = prob;

                                    if i == target {
                                        local_grad -= 1.0;
                                    }

                                    add_node_grad(
                                        &mut tape,
                                        prev[i],
                                        grad * local_grad,
                                    );
                                }
                            }
                        }
                    }
                    Node::FusedLayer(node) => {
                        /*
                         * One fused node represents an entire dense layer.
                         *
                         * Layout:
                         *
                         * weights[o * input_size + i]
                         * inputs[i]
                         * biases[o]
                         * outputs[o]
                         */
                        let input_size = node.input_size;
                        let output_size = node.output_size;
                        let relu = node.relu;
                        let inputs = node.inputs.clone();
                        let weights = node.weights.clone();
                        let biases = node.biases.clone();
                        let outputs = node.outputs.clone();
                        // We need every output gradient because all outputs
                        // belong to this single graph node.
                        let output_grads = node.grads.clone();
                        for o in 0..output_size {
                            let mut grad_out = output_grads[o];
                            if relu && outputs[o] <= 0.0 { grad_out *= 0.01; }
                            let weight_base = o * input_size;
                            for i in 0..input_size {
                                let w_id =
                                    weights[weight_base + i];
                                let x_id = inputs[i];
                                let w = node_data(&tape, w_id);
                                let x = node_data(&tape, x_id);
                                if o == 0 && i == 0 {
                                }
                                // dL/dW = dL/dY * X
                                add_node_grad(
                                    &mut tape,
                                    w_id,
                                    grad_out * x,
                                );
                                // dL/dX = dL/dY * W
                                add_node_grad(
                                    &mut tape,
                                    x_id,
                                    grad_out * w,
                                );
                            }
                            // dL/dB = dL/dY
                            add_node_grad(
                                &mut tape,
                                biases[o],
                                grad_out,
                            );
                        }
                    }
                }
            }
        });
    }

    pub fn relu(&self) -> Tensor {
        let alpha = 0.01; // The "leak" factor
        let x = self.data();

        let data = if x > 0.0 {
            x
        } else {
            x * alpha
        };

        // Create the tensor with the new logic
        Tensor::from_op(data, vec![self.handle], Op::Relu)
    }

    pub fn zero_grad(&self) {
        TAPE.with(|t| {
            match &mut t.borrow_mut().nodes[self.handle.node] {
                Node::Scalar(node) => node.grad = 0.0,
                Node::FusedLayer(node) => {
                    node.grads[self.handle.index] = 0.0;
                }
            }
        });
    }

    pub fn update(&self, learning_rate: f32) {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            match &mut tape.nodes[self.handle.node] {
                Node::Scalar(node) => {
                    node.data -= learning_rate * node.grad;
                }
                Node::FusedLayer(_) => {
                    panic!("Cannot update a fused-layer output directly.");
                }
            }
        });
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        *self
    }
}
