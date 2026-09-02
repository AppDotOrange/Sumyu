use crate::neuron::{Activation, BatchLayerCache, CONV1D_TILE};
use crate::TensorHandle;
use cblas::{Layout, Transpose};
use std::cell::RefCell;

pub struct BackwardWorkspace {
    weights: Vec<f32>,
    positions: Vec<i32>,

    weight_grads: Vec<f32>,
    bias_grads: Vec<f32>,

    col: Vec<f32>,
    col_grads: Vec<f32>,
    group_grad: Vec<f32>,

    first_weight_grads: Vec<f32>,
    first_bias_grads: Vec<f32>,

    second_weight_grads: Vec<f32>,
    second_bias_grads: Vec<f32>,

    hidden_grads: Vec<f32>,

    embedding_grads: Vec<f32>,
}

impl BackwardWorkspace {
    fn new() -> Self {
        Self {
            weights: Vec::new(),
            positions: Vec::new(),

            weight_grads: Vec::new(),
            bias_grads: Vec::new(),

            col: Vec::new(),
            col_grads: Vec::new(),
            group_grad: Vec::new(),

            first_weight_grads: Vec::new(),
            first_bias_grads: Vec::new(),

            second_weight_grads: Vec::new(),
            second_bias_grads: Vec::new(),

            hidden_grads: Vec::new(),

            embedding_grads: Vec::new(),
        }
    }
}

thread_local! {
    static BACKWARD_WORKSPACE:
        RefCell<BackwardWorkspace> =
            RefCell::new(
                BackwardWorkspace::new()
            );
}

#[inline]
pub fn make_conv_positions(
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
) -> Vec<i32> {
    let total =
        output_length * kernel_size;

    let mut positions =
        Vec::with_capacity(total);

    for out_pos in 0..output_length {
        let base =
            out_pos * stride;

        for k in 0..kernel_size {
            let pos =
                if causal {
                    base as i32
                        + k as i32
                        - (kernel_size - 1) as i32
                } else {
                    base as i32
                        + k as i32
                        - padding as i32
                };

            positions.push(pos);
        }
    }

    positions
}

pub fn conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    let kernel_width =
        kernel_size * in_channels;

    workspace.positions =
        make_conv_positions(
            output_length,
            kernel_size,
            stride,
            padding,
            causal,
        );

    workspace.weights.resize(
        weight_handles.len(),
        0.0,
    );

    crate::handle_data_slice(
        weight_handles,
        &mut workspace.weights,
    );

    workspace.weight_grads.resize(
        out_channels * kernel_width,
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        out_channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    let mut input_grads =
        vec![0.0; input.len()];

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    let col_len =
        CONV1D_TILE
            * output_length
            * kernel_width;

    workspace.col.resize(
        col_len,
        0.0,
    );

    workspace.col_grads.resize(
        col_len,
        0.0,
    );

    for batch_start in
        (0..batch_size).step_by(CONV1D_TILE)
    {
        let tile_batch =
            (batch_size - batch_start)
                .min(CONV1D_TILE);

        let rows =
            tile_batch * output_length;

        for b in 0..tile_batch {
            let input_batch_base =
                (batch_start + b)
                    * input_length
                    * in_channels;

            let row_base =
                b * output_length;

            for out_pos in 0..output_length {
                let row =
                    row_base + out_pos;

                let row_start =
                    row * kernel_width;

                let position_base =
                    out_pos * kernel_size;

                for k in 0..kernel_size {
                    let src_pos =
                        workspace.positions[
                            position_base + k
                            ];

                    let dst =
                        row_start
                            + k * in_channels;

                    if src_pos >= 0
                        && (src_pos as usize)
                        < input_length
                    {
                        let src =
                            input_batch_base
                                + src_pos as usize
                                * in_channels;

                        workspace.col[
                            dst..dst + in_channels
                            ]
                            .copy_from_slice(
                                &input[
                                    src..src + in_channels
                                    ],
                            );
                    } else {
                        workspace.col[
                            dst..dst + in_channels
                            ]
                            .fill(0.0);
                    }
                }
            }
        }

        let grad_offset =
            batch_start
                * output_length
                * out_channels;

        let grad_tile =
            &grad[
                grad_offset
                    ..grad_offset + rows * out_channels
                ];

        crate::batched::gemm_beta(
            grad_tile,
            &workspace.col[..rows * kernel_width],
            &mut workspace.weight_grads,
            out_channels,
            kernel_width,
            rows,
            Transpose::Ordinary,
            Transpose::None,
            out_channels,
            kernel_width,
            kernel_width,
            1.0,
        );

        for row in 0..rows {
            for oc in 0..out_channels {
                workspace.bias_grads[oc] +=
                    grad_tile[
                        row * out_channels + oc
                        ];
            }
        }

        crate::batched::gemm(
            grad_tile,
            &workspace.weights,
            &mut workspace.col_grads[
                ..rows * kernel_width
                ],
            rows,
            kernel_width,
            out_channels,
            Transpose::None,
            Transpose::None,
            out_channels,
            kernel_width,
            kernel_width,
        );

        for b in 0..tile_batch {
            let input_batch_base =
                (batch_start + b)
                    * input_length
                    * in_channels;

            let row_base =
                b * output_length;

            for out_pos in 0..output_length {
                let row =
                    row_base + out_pos;

                let col_grad_row =
                    row * kernel_width;

                let position_base =
                    out_pos * kernel_size;

                for k in 0..kernel_size {
                    let src_pos =
                        workspace.positions[
                            position_base + k
                            ];

                    if src_pos < 0
                        || src_pos as usize
                        >= input_length
                    {
                        continue;
                    }

                    let dst =
                        input_batch_base
                            + src_pos as usize
                            * in_channels;

                    let src =
                        col_grad_row
                            + k * in_channels;

                    for c in 0..in_channels {
                        input_grads[dst + c] +=
                            workspace.col_grads[
                                src + c
                                ];
                    }
                }
            }
        }
    }

    crate::add_handle_grad_slices_2(
        weight_handles,
        &workspace.weight_grads,
        bias_handles,
        &workspace.bias_grads,
    );

    input_grads
}

pub fn depthwise_conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    debug_assert_eq!(
        input.len(),
        batch_size * input_length * in_channels
    );

    debug_assert_eq!(
        output.len(),
        batch_size * output_length * in_channels
    );

    debug_assert_eq!(
        grad.len(),
        output.len()
    );

    debug_assert_eq!(
        weight_handles.len(),
        in_channels * kernel_size
    );

    debug_assert_eq!(
        bias_handles.len(),
        in_channels
    );

    workspace.weights.resize(
        weight_handles.len(),
        0.0,
    );

    crate::handle_data_slice(
        weight_handles,
        &mut workspace.weights,
    );

    workspace.positions =
        make_conv_positions(
            output_length,
            kernel_size,
            stride,
            padding,
            causal,
        );

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    workspace.weight_grads.resize(
        in_channels * kernel_size,
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        in_channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    let mut input_grads =
        vec![0.0; input.len()];

    for b in 0..batch_size {
        let input_batch_base =
            b * input_length * in_channels;

        let output_batch_base =
            b * output_length * in_channels;

        for c in 0..in_channels {
            let weight_offset =
                c * kernel_size;

            let weights_c =
                &workspace.weights[
                    weight_offset
                        ..weight_offset + kernel_size
                    ];

            let weight_grads_c =
                &mut workspace.weight_grads[
                    weight_offset
                        ..weight_offset + kernel_size
                    ];

            let mut bias_grad =
                0.0;

            for out_pos in 0..output_length {
                let grad_index =
                    output_batch_base
                        + out_pos * in_channels
                        + c;

                let g =
                    grad[grad_index];

                bias_grad += g;

                let position_base =
                    out_pos * kernel_size;

                for k in 0..kernel_size {
                    let src_pos =
                        workspace.positions[
                            position_base + k
                            ];

                    if src_pos < 0
                        || src_pos as usize
                        >= input_length
                    {
                        continue;
                    }

                    let src_index =
                        input_batch_base
                            + src_pos as usize
                            * in_channels
                            + c;

                    weight_grads_c[k] +=
                        input[src_index] * g;

                    input_grads[src_index] +=
                        weights_c[k] * g;
                }
            }

            workspace.bias_grads[c] +=
                bias_grad;
        }
    }

    crate::add_handle_grad_slices(
        weight_handles,
        &workspace.weight_grads,
    );

    crate::add_handle_grad_slices(
        bias_handles,
        &workspace.bias_grads,
    );

    input_grads
}

pub fn channel_scale_backward(
    input: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    channels: usize,
    scale_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    debug_assert_eq!(
        input.len(),
        grad.len()
    );

    debug_assert_eq!(
        input.len() % batch_size,
        0
    );

    debug_assert_eq!(
        scale_handles.len(),
        channels
    );

    debug_assert_eq!(
        bias_handles.len(),
        channels
    );

    workspace.weights.resize(
        channels,
        0.0,
    );

    crate::handle_data_slice(
        scale_handles,
        &mut workspace.weights,
    );

    workspace.weight_grads.resize(
        channels,
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    let mut input_grads =
        vec![0.0; input.len()];

    let sequence_size =
        input.len() / batch_size;

    debug_assert_eq!(
        sequence_size % channels,
        0
    );

    for b in 0..batch_size {
        let base =
            b * sequence_size;

        let sequence =
            &input[
                base..base + sequence_size
                ];

        let sequence_grad =
            &grad[
                base..base + sequence_size
                ];

        let sequence_input_grad =
            &mut input_grads[
                base..base + sequence_size
                ];

        for chunk_start in
            (0..sequence_size).step_by(channels)
        {
            let x =
                &sequence[
                    chunk_start
                        ..chunk_start + channels
                    ];

            let g =
                &sequence_grad[
                    chunk_start
                        ..chunk_start + channels
                    ];

            let dx =
                &mut sequence_input_grad[
                    chunk_start
                        ..chunk_start + channels
                    ];

            for c in 0..channels {
                let gc =
                    g[c];

                workspace.weight_grads[c] +=
                    gc * x[c];

                workspace.bias_grads[c] +=
                    gc;

                dx[c] =
                    gc * workspace.weights[c];
            }
        }
    }

    crate::add_handle_grad_slices_2(
        scale_handles,
        &workspace.weight_grads,
        bias_handles,
        &workspace.bias_grads,
    );

    input_grads
}

pub fn low_rank_pointwise_backward(
    input: &[f32],
    hidden: &[f32],
    output: &[f32],
    grad: &mut [f32],
    rows: usize,
    in_channels: usize,
    rank: usize,
    out_channels: usize,
    first_weights: &[f32],
    second_weights: &[f32],
    first_weight_handles: &[TensorHandle],
    first_bias_handles: &[TensorHandle],
    second_weight_handles: &[TensorHandle],
    second_bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    workspace.second_weight_grads.resize(
        out_channels * rank,
        0.0,
    );
    workspace.second_weight_grads.fill(0.0);

    workspace.second_bias_grads.resize(
        out_channels,
        0.0,
    );
    workspace.second_bias_grads.fill(0.0);

    workspace.hidden_grads.resize(
        rows * rank,
        0.0,
    );
    workspace.hidden_grads.fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            out_channels as i32,
            rank as i32,
            rows as i32,
            1.0,
            grad,
            out_channels as i32,
            hidden,
            rank as i32,
            0.0,
            &mut workspace.second_weight_grads,
            rank as i32,
        );

        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            rank as i32,
            out_channels as i32,
            1.0,
            grad,
            out_channels as i32,
            second_weights,
            rank as i32,
            0.0,
            &mut workspace.hidden_grads,
            rank as i32,
        );
    }

    for row in 0..rows {
        let base =
            row * out_channels;

        for oc in 0..out_channels {
            workspace.second_bias_grads[oc] +=
                grad[base + oc];
        }
    }

    workspace.first_weight_grads.resize(
        rank * in_channels,
        0.0,
    );
    workspace.first_weight_grads.fill(0.0);

    workspace.first_bias_grads.resize(
        rank,
        0.0,
    );
    workspace.first_bias_grads.fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            rank as i32,
            in_channels as i32,
            rows as i32,
            1.0,
            &workspace.hidden_grads,
            rank as i32,
            input,
            in_channels as i32,
            0.0,
            &mut workspace.first_weight_grads,
            in_channels as i32,
        );
    }

    for row in 0..rows {
        for r in 0..rank {
            workspace.first_bias_grads[r] +=
                workspace.hidden_grads[
                    row * rank + r
                    ];
        }
    }

    let mut input_grads =
        vec![0.0; rows * in_channels];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            rows as i32,
            in_channels as i32,
            rank as i32,
            1.0,
            &workspace.hidden_grads,
            rank as i32,
            first_weights,
            in_channels as i32,
            0.0,
            &mut input_grads,
            in_channels as i32,
        );
    }

    crate::add_handle_grad_slices(
        first_weight_handles,
        &workspace.first_weight_grads,
    );

    crate::add_handle_grad_slices(
        first_bias_handles,
        &workspace.first_bias_grads,
    );

    crate::add_handle_grad_slices(
        second_weight_handles,
        &workspace.second_weight_grads,
    );

    crate::add_handle_grad_slices(
        second_bias_handles,
        &workspace.second_bias_grads,
    );

    input_grads
}

pub fn grouped_conv1d_backward(
    input: &[f32],
    output: &[f32],
    grad: &mut [f32],
    batch_size: usize,
    input_length: usize,
    output_length: usize,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    causal: bool,
    weight_handles: &[TensorHandle],
    bias_handles: &[TensorHandle],
    activation: &Activation,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    let group_in =
        in_channels / groups;

    let group_out =
        out_channels / groups;

    let kernel_width =
        group_in * kernel_size;

    workspace.weights.resize(
        weight_handles.len(),
        0.0,
    );

    crate::handle_data_slice(
        weight_handles,
        &mut workspace.weights,
    );

    for i in 0..grad.len() {
        activation.backward(
            output[i],
            &mut grad[i],
        );
    }

    workspace.weight_grads.resize(
        workspace.weights.len(),
        0.0,
    );
    workspace.weight_grads.fill(0.0);

    workspace.bias_grads.resize(
        out_channels,
        0.0,
    );
    workspace.bias_grads.fill(0.0);

    let mut input_grads =
        vec![0.0; input.len()];

    workspace.col.resize(
        CONV1D_TILE
            * output_length
            * kernel_width,
        0.0,
    );

    workspace.col_grads.resize(
        CONV1D_TILE
            * output_length
            * kernel_width,
        0.0,
    );

    workspace.group_grad.resize(
        CONV1D_TILE
            * output_length
            * group_out,
        0.0,
    );

    workspace.positions =
        make_conv_positions(
            output_length,
            kernel_size,
            stride,
            padding,
            causal,
        );

    for batch_start in
        (0..batch_size).step_by(CONV1D_TILE)
    {
        let tile_batch =
            (batch_size - batch_start)
                .min(CONV1D_TILE);

        let rows =
            tile_batch * output_length;

        for group in 0..groups {
            for b in 0..tile_batch {
                let input_batch_base =
                    (batch_start + b)
                        * input_length
                        * in_channels;

                let row_base =
                    b * output_length;

                for out_pos in 0..output_length {
                    let row =
                        row_base + out_pos;

                    let row_start =
                        row * kernel_width;

                    let position_base =
                        out_pos * kernel_size;

                    for k in 0..kernel_size {
                        let src_pos =
                            workspace.positions[
                                position_base + k
                                ];

                        let dst =
                            row_start
                                + k * group_in;

                        if src_pos >= 0
                            && (src_pos as usize)
                            < input_length
                        {
                            let src =
                                input_batch_base
                                    + src_pos as usize
                                    * in_channels
                                    + group * group_in;

                            workspace.col[
                                dst..dst + group_in
                                ]
                                .copy_from_slice(
                                    &input[
                                        src..src + group_in
                                        ],
                                );
                        } else {
                            workspace.col[
                                dst..dst + group_in
                                ]
                                .fill(0.0);
                        }
                    }
                }
            }

            let output_channel =
                group * group_out;

            for row in 0..rows {
                let global_row =
                    batch_start
                        * output_length
                        + row;

                for oc in 0..group_out {
                    workspace.group_grad[
                        row * group_out + oc
                        ] =
                        grad[
                            global_row
                                * out_channels
                                + output_channel
                                + oc
                            ];

                    workspace.bias_grads[
                        output_channel + oc
                        ] +=
                        workspace.group_grad[
                            row * group_out + oc
                            ];
                }
            }

            let weight_start =
                group
                    * group_out
                    * kernel_width;

            let weight_end =
                weight_start
                    + group_out * kernel_width;

            crate::batched::gemm_beta(
                &workspace.group_grad[
                    ..rows * group_out
                    ],
                &workspace.col[
                    ..rows * kernel_width
                    ],
                &mut workspace.weight_grads[
                    weight_start..weight_end
                    ],
                group_out,
                kernel_width,
                rows,
                Transpose::Ordinary,
                Transpose::None,
                group_out,
                kernel_width,
                kernel_width,
                1.0,
            );

            crate::batched::gemm(
                &workspace.group_grad[
                    ..rows * group_out
                    ],
                &workspace.weights[
                    weight_start..weight_end
                    ],
                &mut workspace.col_grads[
                    ..rows * kernel_width
                    ],
                rows,
                kernel_width,
                group_out,
                Transpose::None,
                Transpose::None,
                group_out,
                kernel_width,
                kernel_width,
            );

            for b in 0..tile_batch {
                for out_pos in 0..output_length {
                    let row =
                        b * output_length
                            + out_pos;

                    let position_base =
                        out_pos * kernel_size;

                    for k in 0..kernel_size {
                        let src_pos =
                            workspace.positions[
                                position_base + k
                                ];

                        if src_pos >= 0
                            && (src_pos as usize)
                            < input_length
                        {
                            let dst =
                                (
                                    (batch_start + b)
                                        * input_length
                                        + src_pos as usize
                                )
                                    * in_channels
                                    + group * group_in;

                            let src =
                                row * kernel_width
                                    + k * group_in;

                            for c in 0..group_in {
                                input_grads[
                                    dst + c
                                    ] +=
                                    workspace.col_grads[
                                        src + c
                                        ];
                            }
                        }
                    }
                }
            }
        }
    }

    crate::add_handle_grad_slices(
        weight_handles,
        &workspace.weight_grads,
    );

    crate::add_handle_grad_slices(
        bias_handles,
        &workspace.bias_grads,
    );

    input_grads
}

pub fn weight_tying_backward(
    input: &[f32],
    grad: &[f32],
    batch_size: usize,
    embedding_dim: usize,
    vocab_size: usize,
    embeddings: &crate::embeddings::Embeddings,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        batch_size * embedding_dim,
        "Invalid WeightTying input size"
    );

    assert_eq!(
        grad.len(),
        batch_size * vocab_size,
        "Invalid WeightTying gradient size"
    );

    let weights =
        embeddings.flat_values();

    debug_assert_eq!(
        weights.len(),
        vocab_size * embedding_dim
    );

    workspace.embedding_grads.resize(
        vocab_size * embedding_dim,
        0.0,
    );
    workspace.embedding_grads.fill(0.0);

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            vocab_size as i32,
            embedding_dim as i32,
            batch_size as i32,
            1.0,
            grad,
            vocab_size as i32,
            input,
            embedding_dim as i32,
            0.0,
            &mut workspace.embedding_grads,
            embedding_dim as i32,
        );
    }

    embeddings.accumulate_flat_grads(
        &workspace.embedding_grads
    );

    let mut input_grads =
        vec![
            0.0;
            batch_size * embedding_dim
        ];

    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            batch_size as i32,
            embedding_dim as i32,
            vocab_size as i32,
            1.0,
            grad,
            vocab_size as i32,
            &weights,
            embedding_dim as i32,
            0.0,
            &mut input_grads,
            embedding_dim as i32,
        );
    }

    input_grads
}

pub fn backward_layers_batch(
    layers: &[BatchLayerCache],
    grad: Vec<f32>,
    batch_size: usize,
) -> Vec<f32> {
    BACKWARD_WORKSPACE.with(|cell| {
        let mut workspace =
            cell.borrow_mut();

        backward_layers_batch_inner(
            layers,
            grad,
            batch_size,
            &mut workspace,
        )
    })
}

fn backward_layers_batch_inner(
    layers: &[BatchLayerCache],
    mut grad: Vec<f32>,
    batch_size: usize,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    for layer in layers.iter().rev() {
        grad = backward_layer_batch(
            layer,
            grad,
            batch_size,
            workspace,
        );
    }

    grad
}

fn backward_layer_batch(
    layer: &BatchLayerCache,
    mut grad: Vec<f32>,
    batch_size: usize,
    workspace: &mut BackwardWorkspace,
) -> Vec<f32> {
    match layer {
        BatchLayerCache::Dense {
            input_size,
            output_size,
            input,
            activation_output,
            weights,
            weight_handles,
            bias_handles,
            activation,
        } => {
            if let Some(output) =
                activation_output
            {
                for b in 0..batch_size {
                    let base =
                        b * *output_size;

                    for o in 0..*output_size {
                        activation.backward(
                            output[base + o],
                            &mut grad[base + o],
                        );
                    }
                }
            }

            workspace.weight_grads.resize(
                output_size * input_size,
                0.0,
            );
            workspace.weight_grads.fill(0.0);

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    *output_size as i32,
                    *input_size as i32,
                    batch_size as i32,
                    1.0,
                    &grad,
                    *output_size as i32,
                    input,
                    *input_size as i32,
                    0.0,
                    &mut workspace.weight_grads,
                    *input_size as i32,
                );
            }

            workspace.bias_grads.resize(
                *output_size,
                0.0,
            );
            workspace.bias_grads.fill(0.0);

            for b in 0..batch_size {
                let base =
                    b * *output_size;

                for o in 0..*output_size {
                    workspace.bias_grads[o] +=
                        grad[base + o];
                }
            }

            let mut input_grads =
                vec![
                    0.0;
                    batch_size * *input_size
                ];

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    batch_size as i32,
                    *input_size as i32,
                    *output_size as i32,
                    1.0,
                    &grad,
                    *output_size as i32,
                    weights,
                    *input_size as i32,
                    0.0,
                    &mut input_grads,
                    *input_size as i32,
                );
            }

            crate::add_handle_grad_slices(
                weight_handles,
                &workspace.weight_grads,
            );

            crate::add_handle_grad_slices(
                bias_handles,
                &workspace.bias_grads,
            );

            input_grads
        }

        BatchLayerCache::Conv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::Residual {
            inner,
        } => {
            let skip_grad =
                grad.clone();

            let mut result =
                backward_layers_batch_inner(
                    inner,
                    grad,
                    batch_size,
                    workspace,
                );

            debug_assert_eq!(
                result.len(),
                skip_grad.len()
            );

            for i in 0..result.len() {
                result[i] +=
                    skip_grad[i];
            }

            result
        }

        BatchLayerCache::DepthwiseConv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            depthwise_conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::GroupedConv1D {
            input,
            output,
            input_length,
            output_length,
            in_channels,
            out_channels,
            groups,
            kernel_size,
            stride,
            padding,
            causal,
            weight_handles,
            bias_handles,
            activation,
        } => {
            grouped_conv1d_backward(
                input,
                output,
                &mut grad,
                batch_size,
                *input_length,
                *output_length,
                *in_channels,
                *out_channels,
                *groups,
                *kernel_size,
                *stride,
                *padding,
                *causal,
                weight_handles,
                bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::LowRankPointwise {
            input,
            hidden,
            output,
            rows,
            in_channels,
            rank,
            out_channels,
            first_weights,
            second_weights,
            first_weight_handles,
            first_bias_handles,
            second_weight_handles,
            second_bias_handles,
            activation,
        } => {
            low_rank_pointwise_backward(
                input,
                hidden,
                output,
                &mut grad,
                *rows,
                *in_channels,
                *rank,
                *out_channels,
                first_weights,
                second_weights,
                first_weight_handles,
                first_bias_handles,
                second_weight_handles,
                second_bias_handles,
                activation,
                workspace,
            )
        }

        BatchLayerCache::ChannelScale {
            input,
            channels,
            scale_handles,
            bias_handles,
        } => {
            channel_scale_backward(
                input,
                &mut grad,
                batch_size,
                *channels,
                scale_handles,
                bias_handles,
                workspace,
            )
        }

        BatchLayerCache::WeightTying {
            input,
            embeddings,
            batch_size,
            embedding_dim,
            vocab_size,
        } => {
            weight_tying_backward(
                input,
                &grad,
                *batch_size,
                *embedding_dim,
                *vocab_size,
                embeddings,
                workspace,
            )
        }
    }
}
