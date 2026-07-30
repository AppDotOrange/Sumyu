pub mod neuron;
pub mod trainer;
pub mod helper;
pub mod fnn_lm;
pub mod chatter;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

type BackFn = Box<dyn Fn(f64) + Send>;

struct TensorData {
    data: f64,
    grad: f64,
    prev: Vec<Tensor>,
    back_fn: Option<BackFn>
}

pub struct Tensor {
    inner: Arc<Mutex<TensorData>>
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Tensor {
            inner: Arc::new(Mutex::new(TensorData {
                data,
                grad: 0.0,
                prev: Vec::new(),
                back_fn: None
            })),
        }
    }

    fn from_op(data: f64, prev: Vec<Tensor>, back_fn: BackFn) -> Self {
        Tensor {
            inner: Arc::new(Mutex::new(TensorData {
                data,
                grad: 0.0,
                prev,
                back_fn: Some(back_fn)
            })),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.lock().unwrap().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.lock().unwrap().grad
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let data = self.data()+other.data();

        let self_clone = self.clone();
        let other_clone = other.clone();

        let back_fn = Box::new(move |out_grad: f64| {
            self_clone.inner.lock().unwrap().grad += out_grad;
            other_clone.inner.lock().unwrap().grad += out_grad;
        });

        let prev = vec![self.clone(), other.clone()];

        Tensor::from_op(data, prev, back_fn)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let data = self.data()-other.data();

        let self_clone = self.clone();
        let other_clone = other.clone();

        let back_fn = Box::new(move |out_grad: f64| {
            self_clone.inner.lock().unwrap().grad += out_grad;
            other_clone.inner.lock().unwrap().grad -= out_grad;
        });

        let prev = vec![self.clone(), other.clone()];

        Tensor::from_op(data, prev, back_fn)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let data = self.data() * other.data();

        let self_clone = self.clone();
        let other_clone = other.clone();

        let backward_fn = Box::new(move |grad_output: f64| {
            self_clone.inner.lock().unwrap().grad += grad_output * other_clone.data();
            other_clone.inner.lock().unwrap().grad += grad_output * self_clone.data();
        });

        let prev = vec![self.clone(), other.clone()];

        Tensor::from_op(data, prev, backward_fn)
    }

    pub fn pow(&self, n: f64) -> Tensor {
        let data = self.data().powf(n);

        let self_clone = self.clone();

        let backward_fn = Box::new(move |grad_output: f64| {
            self_clone.inner.lock().unwrap().grad +=
                grad_output * n * self_clone.data().powf(n - 1.0);
        });

        let prev = vec![self.clone()];

        Tensor::from_op(data, prev, backward_fn)
    }

    pub fn sigmoid(&self) -> Tensor {
        // 1. Forward Pass: Compute 1 / (1 + e^-x)
        let data = 1.0 / (1.0 + (-self.data()).exp());

        // 2. Clone the input so we can use it in the closure
        let self_clone = self.clone();

        // 3. Backward Pass: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
        // Since we already computed 'data' (which is sigmoid(x)), we use that directly.
        let back_fn = Box::new(move |out_grad: f64| {
            let sigmoid_val = data;
            let derivative = sigmoid_val * (1.0 - sigmoid_val);
            self_clone.inner.lock().unwrap().grad += out_grad * derivative;
        });

        // 4. Create the Tensor with the operation graph
        let prev = vec![self.clone()];

        Tensor::from_op(data, prev, back_fn)
    }

    fn build_reverse_top_order(&self) -> Vec<Tensor> {
        let mut topo = Vec::new();  // the vec that we return
        let mut visited = HashSet::new();  // stores the nodes that we have already visited

        fn build_reverse_top_order_recursive(
            tensor: &Tensor,
            topo: &mut Vec<Tensor>,
            visited: &mut HashSet<*const ()>,
        ) {
            // create a unique pointer for this tensor
            let ptr = Arc::as_ptr(&tensor.inner) as *const ();

            if visited.contains(&ptr) {
                return;
            }

            visited.insert(ptr);

            let prev_unwrapped = tensor.inner.lock().unwrap().prev.clone();

            for parent in prev_unwrapped.iter() {
                build_reverse_top_order_recursive(parent, topo, visited);
            }
            // Add current node after all parents
            topo.push(tensor.clone());
        }
        build_reverse_top_order_recursive(self, &mut topo, &mut visited);
        topo
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

        self.inner.lock().unwrap().grad = 1.0;

        for tensor in topo.iter().rev() {
            let grad = tensor.grad();

            let should_call = tensor.inner.lock().unwrap().back_fn.is_some();

            if should_call {
                if let Some(ref func) = tensor.inner.lock().unwrap().back_fn {
                    func(grad);
                }
            }
        }
    }

    pub fn relu(&self) -> Tensor {
        let data = if self.data() > 0.0 { self.data() } else { 0.0 };

        let self_clone = self.clone();

        let backward_fn = Box::new(move |grad_output: f64| {
            let local_grad = if self_clone.data() > 0.0 { 1.0 } else { 0.0 };

            self_clone.inner.lock().unwrap().grad += grad_output * local_grad;
        });

        Tensor::from_op(data, vec![self.clone()], backward_fn)
    }

    pub fn zero_grad(&self) {
        self.inner.lock().unwrap().grad = 0.0;
    }

    pub fn update(&self, learning_rate: f64) {
        let mut locked = self.inner.lock().unwrap();
        locked.data -= learning_rate * locked.grad;
    }

    pub fn clear_prev(&self) {
        self.inner.lock().unwrap().prev.clear();
        // Also clear the back_fn to ensure no closures are held
        self.inner.lock().unwrap().back_fn = None;
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            inner: Arc::clone(&self.inner),
        }
    }
}