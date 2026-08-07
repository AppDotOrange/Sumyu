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

type TensorHandle = usize;

struct Tape {
    nodes: Vec<TensorData>,
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

        self.nodes.push(
            TensorData {
                data,
                grad: 0.0,
                prev,
                op,
            }
        );

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

    static TOPO_VISITED: RefCell<Vec<bool>> = RefCell::new(Vec::new());
    static TOPO_STACK1: RefCell<Vec<TensorHandle>> = RefCell::new(Vec::new());
    static TOPO_STACK2: RefCell<Vec<TensorHandle>> = RefCell::new(Vec::new());
}

pub fn zero_grad_and_update(params: &[Tensor], lr: f32) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();

        for p in params {
            let node = &mut tape.nodes[p.handle];
            node.data -= lr * node.grad;
            tape.nodes[p.handle].grad = 0.0;
        }
    });
}

#[derive(Copy)]
pub struct Tensor {
    handle: TensorHandle,
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
            t.borrow()
                .nodes[self.handle]
                .data
        })
    }

    pub fn grad(&self) -> f32 {
        TAPE.with(|t| {
            t.borrow()
                .nodes[self.handle]
                .grad
        })
    }

    pub fn set_grad(&self, val: f32) {
        TAPE.with(|t| {
            t.borrow_mut()
                .nodes[self.handle]
                .grad = val;
        });
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a + b;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.handle);
        prev.push(other.handle);

        Tensor::from_op(data, prev, Op::Add)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a - b;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.handle);
        prev.push(other.handle);

        Tensor::from_op(data, prev, Op::Sub)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let left = self.data();
        let right = other.data();

        let data = left * right;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.handle);
        prev.push(other.handle);

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

    fn build_reverse_top_order(&self) -> Vec<Tensor> {
        TAPE.with(|t| {
            let tape = t.borrow();
            let tape_size = tape.nodes.len();

            TOPO_VISITED.with(|v| {
                let mut visited = v.borrow_mut();

                if visited.len() != tape_size {
                    visited.resize(tape_size, false);
                } else {
                    visited.fill(false);
                }
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

                        while let Some(id) = stack1.pop() {
                            if visited[id] {
                                continue;
                            }

                            visited[id] = true;
                            stack2.push(id);

                            for &parent in tape.nodes[id].prev.iter().rev() {
                                stack1.push(parent);
                            }
                        }
                        stack2
                            .iter()
                            .map(|&handle| Tensor { handle })
                            .collect()
                    })
                })
            })
        })
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            tape.nodes[self.handle].grad = 1.0;

            for tensor in topo.iter() {
                let id = tensor.handle;

                let grad = tape.nodes[id].grad;

                let op = tape.nodes[id].op.clone();

                match op {
                    Op::Leaf => {}

                    Op::Add => {
                        let a = tape.nodes[id].prev[0];
                        let b = tape.nodes[id].prev[1];

                        tape.nodes[a].grad += grad;
                        tape.nodes[b].grad += grad;
                    }

                    Op::Sub => {
                        let a = tape.nodes[id].prev[0];
                        let b = tape.nodes[id].prev[1];

                        tape.nodes[a].grad += grad;
                        tape.nodes[b].grad -= grad;
                    }

                    Op::Mul => {
                        let a = tape.nodes[id].prev[0];
                        let b = tape.nodes[id].prev[1];

                        let a_data = tape.nodes[a].data;
                        let b_data = tape.nodes[b].data;

                        tape.nodes[a].grad += grad * b_data;
                        tape.nodes[b].grad += grad * a_data;
                    }

                    Op::Fma { weight, input } => {
                        let sum = tape.nodes[id].prev[0];
                        let w = tape.nodes[id].prev[1];
                        let x = tape.nodes[id].prev[2];

                        tape.nodes[sum].grad += grad;
                        tape.nodes[w].grad += grad * input;
                        tape.nodes[x].grad += grad * weight;
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
                            let w_id = tape.nodes[id].prev[i];
                            let x_id = tape.nodes[id].prev[base + i];

                            let w = tape.nodes[w_id].data;
                            let x = tape.nodes[x_id].data;

                            tape.nodes[w_id].grad += grad_out * x;
                            tape.nodes[x_id].grad += grad_out * w;
                        }

                        let bias = tape.nodes[id].prev[base * 2];
                        tape.nodes[bias].grad += grad_out;
                    }

                    Op::Pow(n) => {
                        let x_id = tape.nodes[id].prev[0];
                        let x = tape.nodes[x_id].data;

                        tape.nodes[x_id].grad += grad * n * x.powf(n - 1.0);
                    }

                    Op::Sigmoid => {
                        let x_id = tape.nodes[id].prev[0];
                        let y = tape.nodes[id].data;

                        tape.nodes[x_id].grad += grad * y * (1.0 - y);
                    }

                    Op::Relu => {
                        let x_id = tape.nodes[id].prev[0];
                        let x = tape.nodes[x_id].data;

                        let local_grad = if x > 0.0 { 1.0 } else { 0.01 };

                        tape.nodes[x_id].grad += grad * local_grad;
                    }

                    Op::SoftmaxCrossEntropyOld { probs, targets } => {
                        let prev_len = tape.nodes[id].prev.len();

                        for i in 0..prev_len {
                            let parent = tape.nodes[id].prev[i];
                            tape.nodes[parent].grad += grad * (probs[i] - targets[i]);
                        }
                    }

                    Op::SoftmaxCrossEntropy { probs, target } => {
                        let prev_len = tape.nodes[id].prev.len();

                        for i in 0..prev_len {
                            let parent = tape.nodes[id].prev[i];

                            let mut local_grad = probs[i];

                            if i == target {
                                local_grad -= 1.0;
                            }

                            tape.nodes[parent].grad += grad * local_grad;
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
            t.borrow_mut()
                .nodes[self.handle]
                .grad = 0.0;
        });
    }

    pub fn update(&self, learning_rate: f32) {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();
            let node = &mut tape.nodes[self.handle];
            node.data -= learning_rate * node.grad;
        });
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            handle: self.handle,
        }
    }
}
