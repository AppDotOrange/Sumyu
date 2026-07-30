pub mod neuron;
pub mod trainer;
pub mod helper;
pub mod fnn_lm;
pub mod chatter;
pub mod embeddings;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
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
    _Neuron {
        relu: bool,
        output: f64,
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
    prev: Vec<Rc<RefCell<TensorData>>>,
    op: Op,
}

pub struct Tensor {
    inner: Rc<RefCell<TensorData>>
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                data,
                grad: 0.0,
                prev: Vec::new(),
                op: Op::Leaf
            })),
        }
    }

    fn from_op(data: f64, prev: Vec<Rc<RefCell<TensorData>>>, op: Op) -> Self {
        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                data,
                grad: 0.0,
                prev,
                op,
            })),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    pub fn set_grad(&self, val: f64) {
        self.inner.borrow_mut().grad = val;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a + b;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.inner.clone());
        prev.push(other.inner.clone());

        Tensor::from_op(data, prev, Op::Add)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.data();
        let b = other.data();
        let data = a - b;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.inner.clone());
        prev.push(other.inner.clone());

        Tensor::from_op(data, prev, Op::Sub)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let left = self.data();
        let right = other.data();

        let data = left * right;

        let mut prev = Vec::with_capacity(2);
        prev.push(self.inner.clone());
        prev.push(other.inner.clone());

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
                self.inner.clone(),
                weight.inner.clone(),
                input.inner.clone(),
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

        Tensor::from_op(data, vec![self.inner.clone()], Op::Pow(n))
    }

    pub fn sigmoid(&self) -> Tensor {
        // 1. Forward Pass: Compute 1 / (1 + e^-x)
        let data = 1.0 / (1.0 + (-self.data()).exp());

        // 2. Clone the input so we can use it in the closure

        // 4. Create the Tensor with the operation graph
        let prev = vec![self.inner.clone()];

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
            prev.push(w.inner.clone());
        }


        // inputs
        for x in inputs {
            prev.push(x.inner.clone());
        }


        // bias
        prev.push(bias.inner.clone());


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
        let mut stack1 = Vec::new();
        let mut stack2 = Vec::new();
        let mut visited = HashSet::new();

        // Push the root Tensor
        stack1.push(self.clone());

        while let Some(node) = stack1.pop() {
            let ptr = Rc::as_ptr(&node.inner) as *const ();

            if visited.contains(&ptr) {
                continue;
            }
            visited.insert(ptr);

            stack2.push(node.clone());

            // Extract parents safely
            let parents: Vec<Rc<RefCell<TensorData>>> = {
                let inner = node.inner.borrow();
                inner.prev.clone()
            }; // Borrow guard drops here

            // Push parents as Tensors
            for parent_rc in parents.iter().rev() {
                // Dereference the Rc<Tensor> to get the Tensor, then access its inner Rc
                // parent_rc: &Rc<Tensor>
                // *parent_rc: Tensor
                // (*parent_rc).inner: &Rc<RefCell<TensorData>>
                // We need to clone the Rc<RefCell<TensorData>>
                stack1.push(Tensor {
                    inner: (*parent_rc).clone()
                });
            }
        }

        stack2
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

        self.inner.borrow_mut().grad = 1.0;

        for tensor in topo.iter() {

            let grad = tensor.grad();
            let inner = tensor.inner.borrow();

            match &inner.op {
                Op::Leaf => {}
                Op::Add => {
                    inner.prev[0].borrow_mut().grad += grad;
                    inner.prev[1].borrow_mut().grad += grad;
                }
                Op::Sub => {
                    inner.prev[0].borrow_mut().grad += grad;
                    inner.prev[1].borrow_mut().grad -= grad;
                }
                Op::Mul => {
                    let a = inner.prev[0].borrow().data;
                    let b = inner.prev[1].borrow().data;

                    inner.prev[0].borrow_mut().grad += grad * b;
                    inner.prev[1].borrow_mut().grad += grad * a;
                }
                Op::Fma { weight, input } => {

                    inner.prev[0].borrow_mut().grad += grad;

                    inner.prev[1].borrow_mut().grad += grad * input;

                    inner.prev[2].borrow_mut().grad += grad * weight;
                }
                Op::_Neuron {
                    relu,
                    output,
                } => {
                    let mut local_grad = grad;
                    if *relu && output <= &0.0 {
                        local_grad *= 0.01;
                    }
                    let weight_count =
                        (inner.prev.len() - 1) / 2;
                    let input_offset = weight_count;
                    // weights gradients
                    for i in 0..weight_count {
                        let input_value =
                            inner.prev[input_offset + i]
                                .borrow()
                                .data;
                        inner.prev[i]
                            .borrow_mut()
                            .grad += local_grad * input_value;
                    }
                    // input gradients
                    for i in 0..weight_count {
                        let weight_value =
                            inner.prev[i]
                                .borrow()
                                .data;
                        inner.prev[input_offset + i]
                            .borrow_mut()
                            .grad += local_grad * weight_value;
                    }
                    // bias gradient
                    inner.prev[inner.prev.len()-1]
                        .borrow_mut()
                        .grad += local_grad;
                }
                Op::Linear {
                    input_size,
                    relu,
                    pre_activation,
                } => {
                    let mut grad_out = grad;
                    if *relu && pre_activation <= &0.0 {
                        grad_out *= 0.01;
                    }
                    for i in 0..*input_size {
                        let w =
                            inner.prev[i].borrow().data;
                        let x =
                            inner.prev[input_size+i]
                                .borrow()
                                .data;
                        // dw
                        inner.prev[i]
                            .borrow_mut()
                            .grad += grad_out * x;
                        // dx
                        inner.prev[input_size+i]
                            .borrow_mut()
                            .grad += grad_out * w;
                    }
                    // bias
                    inner.prev[input_size*2]
                        .borrow_mut()
                        .grad += grad_out;
                }
                Op::Pow(n) => {
                    let x = inner.prev[0].borrow().data;
                    inner.prev[0].borrow_mut().grad +=
                        grad * n * x.powf(n - 1.0);
                }
                Op::Sigmoid => {
                    let y = inner.data;
                    inner.prev[0].borrow_mut().grad +=
                        grad * y * (1.0 - y);
                }
                Op::Relu => {
                    let x = inner.prev[0].borrow().data;
                    let local_grad = if x > 0.0 { 1.0 } else { 0.01 };
                    inner.prev[0].borrow_mut().grad += grad * local_grad;
                }
                Op::SoftmaxCrossEntropyOld { probs, targets } => {
                    for ((parent, p), t) in inner.prev
                        .iter()
                        .zip(probs.iter())
                        .zip(targets.iter())
                    {
                        parent.borrow_mut().grad += grad * (p - t);
                    }
                }
                Op::SoftmaxCrossEntropy { probs, target } => {
                    for (i, (parent, p)) in inner.prev
                        .iter()
                        .zip(probs.iter())
                        .enumerate()
                    {
                        let mut local_grad = *p;

                        if i == *target {
                            local_grad -= 1.0;
                        }

                        parent.borrow_mut().grad += grad * local_grad;
                    }
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
        Tensor::from_op(data, vec![self.inner.clone()], Op::Relu)
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = 0.0;
    }

    pub fn update(&self, learning_rate: f64) {
        let mut locked = self.inner.borrow_mut();
        locked.data -= learning_rate * locked.grad;
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            inner: Rc::clone(&self.inner),
        }
    }
}
