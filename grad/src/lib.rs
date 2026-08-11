pub mod neuron;
pub mod trainer;
pub mod helper;
pub mod fnn_lm;
pub mod chatter;
pub mod embeddings;
pub mod batched;

use std::cell::RefCell;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use cblas::{Layout, Transpose};

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

pub struct TensorData {
    data: f32,
    grad: f32,
    prev: Vec<TensorHandle>,
    op: Op,
}

#[derive(Copy, Clone, Debug)]
pub struct TensorHandle {
    node: usize,
    index: usize,
}

struct FusedLayerData {
    outputs: Vec<f32>,
    grads: Vec<f32>,

    inputs: Vec<TensorHandle>,
    weights: Arc<[TensorHandle]>,
    biases: Arc<[TensorHandle]>,

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

    scratch_inputs: Vec<f32>,
    scratch_weights: Vec<f32>,
    scratch_input_grads: Vec<f32>,
    scratch_weight_grads: Vec<f32>,
    scratch_bias_grads: Vec<f32>,
}

impl Tape {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            scratch_inputs: Vec::new(),
            scratch_weights: Vec::new(),
            scratch_input_grads: Vec::new(),
            scratch_weight_grads: Vec::new(),
            scratch_bias_grads: Vec::new(),
        }
    }

    #[inline]
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

#[inline]
pub(crate) fn handle_data(handle: TensorHandle) -> f32 {
    TAPE.with(|t| {
        node_data(&t.borrow(), handle)
    })
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

pub fn zero_grad_and_update(params: &[Tensor], lr: f32) -> f32 {
    let mut grad_sum = 0f32;
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        for p in params {
            match &mut tape.nodes[p.handle.node] {
                Node::Scalar(node) => {
                    node.data -= lr * node.grad;
                    grad_sum += node.grad.abs();
                    node.grad = 0.0;
                }
                Node::FusedLayer(_) => panic!("Parameter cannot be a fused-layer output.")
            }
        }
    });
    grad_sum
}

#[derive(Copy)]
pub struct Tensor {
    handle: TensorHandle,
}

#[inline(always)]
fn node_data(
    tape: &Tape,
    handle: TensorHandle,
) -> f32 {
    match &tape.nodes[handle.node] {
        Node::Scalar(node) => node.data,
        Node::FusedLayer(node) => node.outputs[handle.index],
    }
}

#[inline(always)]
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

pub(crate) fn add_handle_grads(
    grads: &[(TensorHandle, f32)],
) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();

        for &(handle, grad) in grads {
            match &mut tape.nodes[handle.node] {
                Node::Scalar(node) => {
                    node.grad += grad;
                }

                Node::FusedLayer(node) => {
                    node.grads[handle.index] += grad;
                }
            }
        }
    });
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

    #[inline(always)]
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

    #[inline(always)]
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
        weights: Arc<[TensorHandle]>,
        inputs: &[TensorHandle],
        biases: Arc<[TensorHandle]>,
        relu: bool,
    ) -> Vec<Tensor> {
        let input_size = inputs.len();
        let output_size = biases.len();

        debug_assert_eq!(
            weights.len(),
            input_size * output_size
        );

        let node = TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            // Resize reusable scratch buffers only when necessary.
            if tape.scratch_inputs.len() < input_size {
                tape.scratch_inputs.resize(input_size, 0.0);
            }

            if tape.scratch_weights.len() < weights.len() {
                tape.scratch_weights.resize(weights.len(), 0.0);
            }

            // Read inputs once.
            for i in 0..input_size {
                tape.scratch_inputs[i] =
                    node_data(&tape, inputs[i]);
            }

            // Read weights once.
            for i in 0..weights.len() {
                let h = weights[i];

                tape.scratch_weights[i] =
                    match &tape.nodes[h.node] {
                        Node::Scalar(node) => node.data,
                        Node::FusedLayer(_) => unreachable!(),
                    };
            }

            let mut outputs = vec![0.0f32; output_size];

            // BLAS: outputs = W * inputs
            //
            // W is stored row-major as:
            // [ w00 w01 ... w0n ]
            // [ w10 w11 ... w1n ]
            // [ ...           ]
            //
            // So this is:
            // output_size × input_size
            unsafe {
                cblas::sgemv(
                    Layout::RowMajor,
                    Transpose::None,
                    output_size as i32,
                    input_size as i32,
                    1.0,
                    &tape.scratch_weights[..weights.len()],
                    input_size as i32,
                    &tape.scratch_inputs[..input_size],
                    1,
                    0.0,
                    &mut outputs,
                    1,
                );
            }

            // Add biases.
            for o in 0..output_size {
                outputs[o] += match &tape.nodes[biases[o].node] {
                    Node::Scalar(node) => node.data,
                    Node::FusedLayer(_) => unreachable!(),
                };

                if relu {
                    outputs[o] = if outputs[o] > 0.0 {
                        outputs[o]
                    } else {
                        outputs[o] * 0.01
                    };
                }
            }

            let id = tape.nodes.len();

            tape.nodes.push(Node::FusedLayer(
                FusedLayerData {
                    outputs,
                    grads: vec![0.0; output_size],

                    inputs: inputs.to_vec(),
                    weights,
                    biases,

                    input_size,
                    output_size,
                    relu,
                }
            ));

            id
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

    pub fn backward(&self, parameter_boundary: usize) {
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            // Seed output gradient.
            match &mut tape.nodes[self.handle.node] {
                Node::Scalar(node) => {
                    node.grad = 1.0;
                }

                Node::FusedLayer(node) => {
                    node.grads[self.handle.index] = 1.0;
                }
            }

            // Every node after parameter_boundary belongs to the current
            // computation graph. Since nodes are allocated after their
            // dependencies, reverse node order is already reverse-topological.
            let tape_end = tape.nodes.len();

            for id in (parameter_boundary..tape_end).rev() {
                // ---------------------------------------------------------
                // Scalar node
                // ---------------------------------------------------------

                if matches!(&tape.nodes[id], Node::Scalar(_)) {
                    // Get gradient without keeping a borrow alive.
                    let grad = match &tape.nodes[id] {
                        Node::Scalar(node) => node.grad,
                        _ => unreachable!(),
                    };

                    // Extract everything we need from the node.
                    //
                    // We still clone these because the current Op representation
                    // owns its data. This can be optimized separately later.
                    let op = match &tape.nodes[id] {
                        Node::Scalar(node) => node.op.clone(),
                        _ => unreachable!(),
                    };

                    let prev = match &tape.nodes[id] {
                        Node::Scalar(node) => node.prev.clone(),
                        _ => unreachable!(),
                    };

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

                            // The output is already stored in this node.
                            let y = node_data(
                                &tape,
                                TensorHandle {
                                    node: id,
                                    index: 0,
                                },
                            );

                            add_node_grad(
                                &mut tape,
                                x_id,
                                grad * y * (1.0 - y),
                            );
                        }

                        Op::Relu => {
                            let x_id = prev[0];
                            let x = node_data(&tape, x_id);

                            let local_grad =
                                if x > 0.0 { 1.0 } else { 0.01 };

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
                            for i in 0..prev.len() {
                                let mut local_grad = probs[i];

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

                    continue;
                }

                // ---------------------------------------------------------
                // Fused layer
                // ---------------------------------------------------------

                // Temporarily move the fused layer out of the tape.
                //
                // This allows us to mutate parameter/input gradients while
                // still having ownership of the fused layer's data.
                let fused = match std::mem::replace(
                    &mut tape.nodes[id],
                    Node::Scalar(TensorData {
                        data: 0.0,
                        grad: 0.0,
                        prev: Vec::new(),
                        op: Op::Leaf,
                    }),
                ) {
                    Node::FusedLayer(node) => node,
                    _ => unreachable!(),
                };

                let input_size = fused.input_size;
                let output_size = fused.output_size;
                let relu = fused.relu;

                let weight_count = input_size * output_size;

                // ---------------------------------------------------------
                // Resize reusable scratch buffers.
                // ---------------------------------------------------------

                if tape.scratch_inputs.len() < input_size {
                    tape.scratch_inputs.resize(input_size, 0.0);
                }

                if tape.scratch_weights.len() < weight_count {
                    tape.scratch_weights.resize(weight_count, 0.0);
                }

                if tape.scratch_input_grads.len() < input_size {
                    tape.scratch_input_grads.resize(input_size, 0.0);
                }

                if tape.scratch_weight_grads.len() < weight_count {
                    tape.scratch_weight_grads.resize(weight_count, 0.0);
                }

                if tape.scratch_bias_grads.len() < output_size {
                    tape.scratch_bias_grads.resize(output_size, 0.0);
                }

                tape.scratch_weight_grads[..weight_count].fill(0.0);
                tape.scratch_bias_grads[..output_size].fill(0.0);

                // ---------------------------------------------------------
                // Read inputs into contiguous f32 scratch.
                // ---------------------------------------------------------

                for i in 0..input_size {
                    let h = fused.inputs[i];

                    tape.scratch_inputs[i] =
                        node_data(&tape, h);
                }

                // ---------------------------------------------------------
                // Read weights into contiguous f32 scratch.
                // ---------------------------------------------------------

                for i in 0..weight_count {
                    let h = fused.weights[i];

                    tape.scratch_weights[i] =
                        match &tape.nodes[h.node] {
                            Node::Scalar(node) => node.data,
                            Node::FusedLayer(_) => unreachable!(),
                        };
                }

                // ---------------------------------------------------------
                // Clear input gradients.
                // ---------------------------------------------------------

                tape.scratch_input_grads[..input_size]
                    .fill(0.0);

                // ---------------------------------------------------------
                // Prepare dY = gradient after activation derivative.
                // ---------------------------------------------------------

                if tape.scratch_bias_grads.len() < output_size {
                    tape.scratch_bias_grads.resize(output_size, 0.0);
                }

                for o in 0..output_size {
                    let mut grad_out = fused.grads[o];

                    // Leaky ReLU derivative.
                    if relu && fused.outputs[o] <= 0.0 {
                        grad_out *= 0.01;
                    }

                    tape.scratch_bias_grads[o] = grad_out;
                }

                // ---------------------------------------------------------
                // dX = W^T * dY
                // ---------------------------------------------------------

                tape.scratch_input_grads[..input_size].fill(0.0);

                let Tape {
                    scratch_weights,
                    scratch_bias_grads,
                    scratch_input_grads,
                    ..
                } = &mut *tape;

                unsafe {
                    cblas::sgemv(
                        Layout::RowMajor,
                        Transpose::Ordinary,
                        output_size as i32,
                        input_size as i32,
                        1.0,
                        &scratch_weights[..weight_count],
                        input_size as i32,
                        &scratch_bias_grads[..output_size],
                        1,
                        0.0,
                        &mut scratch_input_grads[..input_size],
                        1,
                    );
                }

                // ---------------------------------------------------------
                // dW = dY * X^T
                //
                // dY: output_size × 1
                // X^T: 1 × input_size
                //
                // Result: output_size × input_size
                // ---------------------------------------------------------

                tape.scratch_weight_grads[..weight_count].fill(0.0);

                let Tape {
                    scratch_bias_grads,
                    scratch_inputs,
                    scratch_weight_grads,
                    ..
                } = &mut *tape;

                unsafe {
                    cblas::sger(
                        Layout::RowMajor,
                        output_size as i32,
                        input_size as i32,
                        1.0,
                        &scratch_bias_grads[..output_size],
                        1,
                        &scratch_inputs[..input_size],
                        1,
                        &mut scratch_weight_grads[..weight_count],
                        input_size as i32,
                    );
                }

                // ---------------------------------------------------------
                // Accumulate weight gradients.
                // ---------------------------------------------------------

                for i in 0..weight_count {
                    let weight_id = fused.weights[i];
                    let grad = tape.scratch_weight_grads[i];

                    match &mut tape.nodes[weight_id.node] {
                        Node::Scalar(node) => {
                            node.grad += grad;
                        }

                        Node::FusedLayer(_) => {
                            unreachable!();
                        }
                    }
                }

                // ---------------------------------------------------------
                // Accumulate bias gradients.
                // ---------------------------------------------------------

                for o in 0..output_size {
                    let bias_id = fused.biases[o];
                    let grad = tape.scratch_bias_grads[o];

                    match &mut tape.nodes[bias_id.node] {
                        Node::Scalar(node) => {
                            node.grad += grad;
                        }

                        Node::FusedLayer(_) => {
                            unreachable!();
                        }
                    }
                }

                // ---------------------------------------------------------
                // Propagate input gradients.
                // ---------------------------------------------------------

                for i in 0..input_size {
                    let input_id = fused.inputs[i];
                    let grad = tape.scratch_input_grads[i];

                    match &mut tape.nodes[input_id.node] {
                        Node::Scalar(node) => {
                            node.grad += grad;
                        }

                        Node::FusedLayer(node) => {
                            node.grads[input_id.index] += grad;
                        }
                    }
                }
                // ---------------------------------------------------------
                // Put the fused layer back.
                // ---------------------------------------------------------
                tape.nodes[id] =
                    Node::FusedLayer(fused);
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
