pub mod neuron;
pub mod trainer;
pub mod helper;
pub mod fnn_lm;
pub mod chatter;
pub mod embeddings;

use std::cell::RefCell;
use std::sync::Arc;
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

    scratch_outputs: Vec<f32>,
    scratch_output_grads: Vec<f32>,

    // Reused handle buffers.
    scratch_inputs_handles: Vec<TensorHandle>,
    scratch_weights_handles: Vec<TensorHandle>,
    scratch_bias_handles: Vec<TensorHandle>,
}

impl Tape {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            scratch_inputs: Vec::new(),
            scratch_weights: Vec::new(),
            scratch_input_grads: Vec::new(),
            scratch_outputs: Vec::new(),
            scratch_output_grads: Vec::new(),
            scratch_inputs_handles: Vec::new(),
            scratch_weights_handles: Vec::new(),
            scratch_bias_handles: Vec::new(),
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
    static TOPO_VISITED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static TOPO_GENERATION: RefCell<u32> = const { RefCell::new(0) };
    static TOPO_STACK1: RefCell<Vec<TensorHandle>> = const { RefCell::new(Vec::new()) };
    static TOPO_STACK2: RefCell<Vec<TensorHandle>> = const { RefCell::new(Vec::new()) };
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

            let mut outputs = Vec::with_capacity(output_size);

            for o in 0..output_size {
                let bias = match &tape.nodes[biases[o].node] {
                    Node::Scalar(node) => node.data,
                    Node::FusedLayer(_) => unreachable!(),
                };

                let base = o * input_size;

                let mut sum = bias;

                // HOT LOOP
                for i in 0..input_size {
                    sum +=
                        tape.scratch_weights[base + i]
                            * tape.scratch_inputs[i];
                }

                if relu {
                    sum = if sum > 0.0 {
                        sum
                    } else {
                        sum * 0.01
                    };
                }

                outputs.push(sum);
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

    fn build_reverse_top_order(&self) -> Vec<TensorHandle> {
        TAPE.with(|t| {
            let tape = t.borrow();
            let tape_size = tape.nodes.len();

            TOPO_VISITED.with(|v| {
                TOPO_STACK1.with(|s1| {
                    TOPO_STACK2.with(|s2| {
                        let mut visited = v.borrow_mut();
                        let mut stack1 = s1.borrow_mut();
                        let mut stack2 = s2.borrow_mut();

                        if visited.len() < tape_size {
                            visited.resize(tape_size, 0);
                        }

                        let generation = TOPO_GENERATION.with(|g| {
                            let mut g = g.borrow_mut();
                            *g = g.wrapping_add(1);

                            if *g == 0 {
                                visited.fill(0);
                                *g = 1;
                            }

                            *g
                        });

                        stack1.clear();
                        stack2.clear();

                        stack1.push(self.handle);

                        while let Some(handle) = stack1.pop() {
                            let id = handle.node;

                            if visited[id] == generation {
                                continue;
                            }

                            visited[id] = generation;

                            match &tape.nodes[id] {
                                Node::Scalar(node) => {
                                    stack1.extend(
                                        node.prev.iter().rev().copied()
                                    );
                                }
                                Node::FusedLayer(node) => {
                                    // Only inputs are graph dependencies.
                                    // Weights and biases are leaves/parameters.
                                    stack1.extend(
                                        node.inputs.iter().rev().copied()
                                    );
                                }
                            }
                            stack2.push(handle);
                        }
                        let result = std::mem::take(&mut *stack2);
                        result
                    })
                })
            })
        })
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

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

            for &handle in topo.iter() {
                let id = handle.node;

                // ---------------------------------------------------------
                // Get gradient without keeping a borrow alive.
                // ---------------------------------------------------------

                let grad = match &tape.nodes[id] {
                    Node::Scalar(node) => node.grad,
                    Node::FusedLayer(node) => node.grads[handle.index],
                };

                // ---------------------------------------------------------
                // Scalar node
                // ---------------------------------------------------------

                if matches!(&tape.nodes[id], Node::Scalar(_)) {
                    // Extract everything we need from the node FIRST.
                    //
                    // This is the important part:
                    // after this scope ends, `tape.nodes[id]` is no longer
                    // immutably borrowed.

                    let op = match &tape.nodes[id] {
                        Node::Scalar(node) => node.op.clone(),
                        _ => unreachable!(),
                    };

                    // We unfortunately still clone Op here because your
                    // SoftmaxCrossEntropy variants contain Vecs.
                    //
                    // The common ops themselves are tiny, so the next
                    // optimization can remove this clone entirely.

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
                                handle,
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
                // This is the important optimization:
                //
                //   - no cloning inputs
                //   - no cloning weights
                //   - no cloning biases
                //   - no cloning outputs
                //   - no cloning grads
                //   - no unsafe
                //
                // Once the FusedLayerData is moved out, we can freely mutate
                // other nodes in `tape` without fighting the borrow checker.
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
                //
                // This removes the tape lookup from the matrix hot loop.
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
                // HOT BACKWARD LOOP
                // ---------------------------------------------------------

                for o in 0..output_size {
                    let mut grad_out = fused.grads[o];

                    // Leaky ReLU derivative.
                    if relu && fused.outputs[o] <= 0.0 {
                        grad_out *= 0.01;
                    }

                    let base = o * input_size;

                    for i in 0..input_size {
                        let w =
                            tape.scratch_weights[base + i];

                        let x =
                            tape.scratch_inputs[i];

                        // -----------------------------------------------------
                        // Weight gradient.
                        // -----------------------------------------------------

                        let weight_id =
                            fused.weights[base + i];

                        match &mut tape.nodes[weight_id.node] {
                            Node::Scalar(node) => {
                                node.grad += grad_out * x;
                            }

                            Node::FusedLayer(_) => {
                                unreachable!();
                            }
                        }

                        // -----------------------------------------------------
                        // Input gradient.
                        //
                        // Keep this local instead of touching the tape for every
                        // multiplication.
                        // -----------------------------------------------------

                        tape.scratch_input_grads[i] +=
                            grad_out * w;
                    }

                    // ---------------------------------------------------------
                    // Bias gradient.
                    // ---------------------------------------------------------

                    let bias_id = fused.biases[o];

                    match &mut tape.nodes[bias_id.node] {
                        Node::Scalar(node) => {
                            node.grad += grad_out;
                        }

                        Node::FusedLayer(_) => {
                            unreachable!();
                        }
                    }
                }

                // ---------------------------------------------------------
                // Propagate accumulated input gradients.
                // ---------------------------------------------------------

                for i in 0..input_size {
                    let input_id = fused.inputs[i];

                    let grad =
                        tape.scratch_input_grads[i];

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
                // Put the fused layer back into the tape.
                //
                // Its own `grads` are unchanged, just as before.
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
