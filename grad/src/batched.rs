use cblas::{Layout, Transpose};

#[inline]
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    trans_a: Transpose,
    trans_b: Transpose,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    debug_assert!(m > 0);
    debug_assert!(n > 0);
    debug_assert!(k > 0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            lda as i32,
            b,
            ldb as i32,
            0.0,
            c,
            ldc as i32,
        );
    }
}

/// X: batch × input
/// W: output × input
/// Y: batch × output
///
/// Computes:
///
///     Y = X × Wᵀ
///
/// W is stored in the same row-major layout as the existing MLP.
pub fn dense_forward(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    output: &mut [f32],
    batch: usize,
    input_size: usize,
    output_size: usize,
    relu: bool,
) {
    debug_assert_eq!(
        input.len(),
        batch * input_size
    );

    debug_assert_eq!(
        weights.len(),
        output_size * input_size
    );

    debug_assert_eq!(
        biases.len(),
        output_size
    );

    debug_assert_eq!(
        output.len(),
        batch * output_size
    );

    // X × Wᵀ
    //
    // X: B × I
    // W: O × I
    // Wᵀ: I × O
    // Y: B × O
    gemm(
        input,
        weights,
        output,
        batch,
        output_size,
        input_size,
        Transpose::None,
        Transpose::Ordinary,
        input_size,
        input_size,
        output_size,
    );

    // Bias + Leaky ReLU.
    for b in 0..batch {
        let base = b * output_size;

        for o in 0..output_size {
            let index = base + o;

            let mut value =
                output[index] + biases[o];

            if relu && value <= 0.0 {
                value *= 0.01;
            }

            output[index] = value;
        }
    }
}

/// Batched softmax cross entropy.
///
/// `logits` is batch × classes.
/// `targets` contains one class index per sample.
/// `grad_logits` receives:
///
///     softmax(logits) - one_hot(target)
///
/// IMPORTANT:
/// The gradient is NOT divided by batch size.
///
/// That matches the existing trainer, which accumulates
/// per-sample gradients and then updates directly.
pub fn softmax_cross_entropy_batch(
    logits: &[f32],
    targets: &[usize],
    grad_logits: &mut [f32],
    batch: usize,
    classes: usize,
) -> f32 {
    debug_assert_eq!(
        logits.len(),
        batch * classes
    );

    debug_assert_eq!(
        targets.len(),
        batch
    );

    debug_assert_eq!(
        grad_logits.len(),
        batch * classes
    );

    let mut total_loss = 0.0f32;

    for b in 0..batch {
        let base = b * classes;

        // Numerically stable softmax.
        let mut max_value = f32::NEG_INFINITY;

        for c in 0..classes {
            max_value =
                max_value.max(logits[base + c]);
        }

        let mut sum_exp = 0.0f32;

        for c in 0..classes {
            let e =
                (logits[base + c] - max_value).exp();

            grad_logits[base + c] = e;
            sum_exp += e;
        }

        let target = targets[b];

        // Same protection as the existing implementation:
        //
        // -log(max(prob[target], 1e-7))
        let target_prob =
            (grad_logits[base + target] / sum_exp)
                .max(1e-7);

        total_loss -= target_prob.ln();

        // Convert exponentials into probabilities and
        // simultaneously create the CE gradient.
        for c in 0..classes {
            grad_logits[base + c] /= sum_exp;
        }

        grad_logits[base + target] -= 1.0;
    }

    total_loss
}

/// Applies the derivative of Leaky ReLU in-place.
///
/// The forward activation uses:
///
///     x       if x > 0
///     0.01*x  otherwise
///
/// Therefore:
///
///     1.0       if x > 0
///     0.01      otherwise
///
/// `activation` is the post-activation output. Since
/// negative outputs remain negative, checking <= 0 is
/// sufficient and matches the existing fused backward.
#[inline]
pub fn leaky_relu_backward(
    grad: &mut [f32],
    activation: &[f32],
) {
    debug_assert_eq!(
        grad.len(),
        activation.len()
    );

    for i in 0..grad.len() {
        if activation[i] <= 0.0 {
            grad[i] *= 0.01;
        }
    }
}

/// dX = dY × W
///
/// dY: batch × output
/// W:  output × input
/// dX: batch × input
pub fn dense_backward_input(
    grad_output: &[f32],
    weights: &[f32],
    grad_input: &mut [f32],
    batch: usize,
    input_size: usize,
    output_size: usize,
) {
    debug_assert_eq!(
        grad_output.len(),
        batch * output_size
    );

    debug_assert_eq!(
        weights.len(),
        output_size * input_size
    );

    debug_assert_eq!(
        grad_input.len(),
        batch * input_size
    );

    // dX = dY × W
    //
    // dY: B × O
    // W:  O × I
    // dX: B × I
    gemm(
        grad_output,
        weights,
        grad_input,
        batch,
        input_size,
        output_size,
        Transpose::None,
        Transpose::None,
        output_size,
        input_size,
        input_size,
    );
}

/// dW = dYᵀ × X
///
/// dY: B × output
/// X:  B × input
/// dW: output × input
pub fn dense_backward_weights(
    grad_output: &[f32],
    input: &[f32],
    grad_weights: &mut [f32],
    batch: usize,
    input_size: usize,
    output_size: usize,
) {
    debug_assert_eq!(
        grad_output.len(),
        batch * output_size
    );

    debug_assert_eq!(
        input.len(),
        batch * input_size
    );

    debug_assert_eq!(
        grad_weights.len(),
        output_size * input_size
    );

    // dW = dYᵀ × X
    //
    // dYᵀ: O × B
    // X:    B × I
    // dW:   O × I
    gemm(
        grad_output,
        input,
        grad_weights,
        output_size,
        input_size,
        batch,
        Transpose::Ordinary,
        Transpose::None,
        output_size,
        input_size,
        input_size,
    );
}

/// db = sum(dY over batch)
pub fn dense_backward_bias(
    grad_output: &[f32],
    grad_bias: &mut [f32],
    batch: usize,
    output_size: usize,
) {
    debug_assert_eq!(
        grad_output.len(),
        batch * output_size
    );

    debug_assert_eq!(
        grad_bias.len(),
        output_size
    );

    grad_bias.fill(0.0);

    for b in 0..batch {
        let base = b * output_size;

        for o in 0..output_size {
            grad_bias[o] +=
                grad_output[base + o];
        }
    }
}
