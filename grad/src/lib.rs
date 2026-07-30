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
        weight: f64,
        input: f64,
    },
    Linear {
        input_size: usize,
        relu: bool,
        pre_activation: f64,
    },
    Pow(f64),
    Sigmoid,
    Relu,
    SoftmaxCrossEntropy {
        probs: Vec<f64>,
        target: usize,
    },
    SoftmaxCrossEntropyOld {
        probs: Vec<f64>,
        targets: Vec<f64>,
    },
}

struct TensorData {
    data: f64,
    grad: f64,
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
        data: f64,
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

pub struct Tensor {
    handle: TensorHandle,
}

impl Tensor {
    pub fn new(data: f64) -> Self {
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
        data: f64,
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

    pub fn data(&self) -> f64 {
        TAPE.with(|t| {
            t.borrow()
                .nodes[self.handle]
                .data
        })
    }

    pub fn grad(&self) -> f64 {
        TAPE.with(|t| {
            t.borrow()
                .nodes[self.handle]
                .grad
        })
    }

    pub fn set_grad(&self, val: f64) {
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
        prev.push(self.handle.clone());
        prev.push(other.handle.clone());

        Tensor::from_op(data, prev, Op::Add)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a - b;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.handle.clone());
        prev.push(other.handle.clone());

        Tensor::from_op(data, prev, Op::Sub)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let left = self.data();
        let right = other.data();

        let data = left * right;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.handle.clone());
        prev.push(other.handle.clone());

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
                self.handle.clone(),
                weight.handle.clone(),
                input.handle.clone(),
            ],
            Op::Fma {
                weight: w,
                input: x,
            },
        )
    }

    pub fn pow(&self, n: f64) -> Tensor {
        let x = self.data();
        let data = x.powf(n);

        Tensor::from_op(data, vec![self.handle.clone()], Op::Pow(n))
    }

    pub fn sigmoid(&self) -> Tensor {
        // 1. Forward Pass: Compute 1 / (1 + e^-x)
        let data = 1.0 / (1.0 + (-self.data()).exp());

        // 2. Clone the input so we can use it in the closure

        // 4. Create the Tensor with the operation graph
        let prev = vec![self.handle.clone()];

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
            prev.push(w.handle.clone());
        }


        // inputs
        for x in inputs {
            prev.push(x.handle.clone());
        }


        // bias
        prev.push(bias.handle.clone());


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
        let tape_size = TAPE.with(|t| {
            t.borrow().nodes.len()
        });

        TOPO_VISITED.with(|v| {
            let mut visited = v.borrow_mut();

            if visited.len() != tape_size {
                visited.resize(tape_size, false);
            } else {
                visited.fill(false);
            }
        });

        TOPO_STACK1.with(|s| {
            let mut stack = s.borrow_mut();
            stack.clear();
            stack.push(self.handle);
        });

        TOPO_STACK2.with(|s| {
            let mut stack = s.borrow_mut();
            stack.clear();
        });

        loop {
            let current = TOPO_STACK1.with(|s| {
                s.borrow_mut().pop()
            });

            let id = match current {
                Some(x) => x,
                None => break,
            };

            let already_seen = TOPO_VISITED.with(|v| {
                let visited = v.borrow();
                visited[id]
            });

            if already_seen {
                continue;
            }

            TOPO_VISITED.with(|v| {
                v.borrow_mut()[id] = true;
            });

            TOPO_STACK2.with(|s| {
                s.borrow_mut().push(id);
            });

            TAPE.with(|t| {
                let tape = t.borrow();

                TOPO_STACK1.with(|s| {
                    let mut stack = s.borrow_mut();

                    for &parent in tape.nodes[id].prev.iter().rev() {
                        stack.push(parent);
                    }
                });
            });
        }
        TOPO_STACK2.with(|s| {
            s.borrow()
                .iter()
                .map(|&handle| Tensor { handle })
                .collect()
        })
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

        TAPE.with(|t| {
            t.borrow_mut()
                .nodes[self.handle]
                .grad = 1.0;
        });

        for tensor in topo.iter() {

            let id = tensor.handle;

            let (grad, op, prev) = TAPE.with(|t| {
                let tape = t.borrow();

                let node = &tape.nodes[id];

                (
                    node.grad,
                    node.op.clone(),
                    node.prev.clone(),
                )
            });

            match op {
                Op::Leaf => {}
                Op::Add => {
                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();
                        tape.nodes[prev[0]].grad += grad;
                        tape.nodes[prev[1]].grad += grad;
                    });
                }
                Op::Sub => {
                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();
                        tape.nodes[prev[0]].grad += grad;
                        tape.nodes[prev[1]].grad -= grad;
                    });
                }
                Op::Mul => {
                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();

                        let a = tape.nodes[prev[0]].data;
                        let b = tape.nodes[prev[1]].data;

                        tape.nodes[prev[0]].grad += grad * b;
                        tape.nodes[prev[1]].grad += grad * a;
                    });
                }
                Op::Fma { weight, input } => {
                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();

                        tape.nodes[prev[0]].grad += grad;
                        tape.nodes[prev[1]].grad += grad * input;
                        tape.nodes[prev[2]].grad += grad * weight;
                    });
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

                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();

                        for i in 0..input_size {
                            let w = tape.nodes[prev[i]].data;
                            let x = tape.nodes[prev[input_size + i]].data;

                            tape.nodes[prev[i]].grad += grad_out * x;
                            tape.nodes[prev[input_size + i]].grad += grad_out * w;
                        }

                        tape.nodes[prev[input_size * 2]].grad += grad_out;
                    });
                }
                Op::Pow(n) => {
                    let x = TAPE.with(|t| {
                        t.borrow()
                            .nodes[prev[0]]
                            .data
                    });
                    TAPE.with(|t| {
                        t.borrow_mut()
                            .nodes[prev[0]]
                            .grad += grad * n * x.powf(n - 1.0);
                    });
                }
                Op::Sigmoid => {
                    let y = TAPE.with(|t| {
                        t.borrow()
                            .nodes[id]
                            .data
                    });
                    TAPE.with(|t| {
                        t.borrow_mut()
                            .nodes[prev[0]]
                            .grad += grad * y * (1.0 - y);
                    });
                }
                Op::Relu => {
                    let x = TAPE.with(|t| {
                        t.borrow()
                            .nodes[prev[0]]
                            .data
                    });
                    let local_grad = if x > 0.0 {
                        1.0
                    } else {
                        0.01
                    };
                    TAPE.with(|t| {
                        t.borrow_mut()
                            .nodes[prev[0]]
                            .grad += grad * local_grad;
                    });
                }
                Op::SoftmaxCrossEntropyOld { probs, targets } => {
                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();
                        for ((parent, p), target) in prev
                            .iter()
                            .zip(probs.iter())
                            .zip(targets.iter())
                        {
                            tape.nodes[*parent].grad += grad * (p - target);
                        }
                    });
                }
                Op::SoftmaxCrossEntropy { probs, target } => {
                    TAPE.with(|t| {
                        let mut tape = t.borrow_mut();
                        for (i, (parent, p)) in prev
                            .iter()
                            .zip(probs.iter())
                            .enumerate()
                        {
                            let mut local_grad = *p;
                            if i == target {
                                local_grad -= 1.0;
                            }
                            tape.nodes[*parent].grad += grad * local_grad;
                        }
                    });
                }
            }
        }
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
        Tensor::from_op(data, vec![self.handle.clone()], Op::Relu)
    }

    pub fn zero_grad(&self) {
        TAPE.with(|t| {
            t.borrow_mut()
                .nodes[self.handle]
                .grad = 0.0;
        });
    }

    pub fn update(&self, learning_rate: f64) {
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
