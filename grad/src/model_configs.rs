use crate::neuron::{Activation, LayerSpec};
use crate::vocabs::{ml_200_tok_vocab_v3, ml_v4, poke_v1, poke_v2, poke_v3, recipe_v1, recipe_v2, tale_v1, oasst1, fineweb, fineweb_v2, recipe_v3, fineweb_v3};

pub struct Config<'a> {
    pub lr: f32,
    pub batch_size: usize,
    pub max_batches_per_epoch: usize, // 0 means no limit
    pub vocab: Vec<String>,
    pub context_len: usize,
    pub emb_dim: usize,
    pub hidden_dim: &'a [usize],
    pub epochs: usize,
}

impl<'a> Config<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(lr: f32,
               batch_size: usize,
               max_batches_per_epoch: usize,
               vocab: Vec<String>,
               context_len: usize,
               emb_dim: usize,
               hidden_dim: &'a [usize],
               epochs: usize,) -> Self {
        Self {
            lr,
            batch_size,
            max_batches_per_epoch,
            vocab,
            context_len,
            emb_dim,
            hidden_dim,
            epochs,
        }
    }
}

pub struct HybridConfig {
    pub lr: f32,
    pub batch_size: usize,
    pub max_batches_per_epoch: usize, // 0 means no limit
    pub vocab: Vec<String>,
    pub context_len: usize,
    pub emb_dim: usize,
    pub layer_specs: Vec<LayerSpec>,
    pub epochs: usize,
}

impl<'a> HybridConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(lr: f32,
               batch_size: usize,
               max_batches_per_epoch: usize,
               vocab: Vec<String>,
               context_len: usize,
               emb_dim: usize,
               layer_specs: Vec<LayerSpec>,
               epochs: usize,) -> Self {
        Self {
            lr,
            batch_size,
            max_batches_per_epoch,
            vocab,
            context_len,
            emb_dim,
            layer_specs,
            epochs,
        }
    }
}

pub fn minimodel_config() -> Config<'static> {
    Config::new(0.01,
                32,
                10,
                ml_200_tok_vocab_v3(),
                8,
                20,
                &[200],
                500,
    )
}

pub fn rustception_optimized() -> Config<'static> {
    Config::new(0.01,
                32,
                0, // no limit
                ml_200_tok_vocab_v3(),
                32,
                32,
                &[200, 200, 100],
                500,
    )
}

pub fn rustception_optimized_v2() -> Config<'static> {
    Config::new(0.01,
                32,
                0, // no limit
                ml_200_tok_vocab_v3(),
                32,
                32,
                &[250, 200, 64],
                500,
    )
}

pub fn rustception_optimized_v2_train_options(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> {
    Config::new(lr,
                batch_size,
                0, // no limit
                ml_200_tok_vocab_v3(),
                32,
                32,
                &[250, 200, 64],
                epochs,
    )
}

pub fn rustception_optimized_v2_large_train_options(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> {
    Config::new(lr,
                batch_size,
                0, // no limit
                ml_200_tok_vocab_v3(),
                32,
                42,
                &[250, 200, 100, 64],
                epochs,
    )
}

pub fn rustception_v3_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                ml_v4(),
                32,
                42,
                &[250, 200, 100, 64],
                epochs,
    )
}

pub fn rustception_v4_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                ml_v4(),
                16,
                32,
                &[200, 100],
                epochs,
    )
}

pub fn poke_v1_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v1(),
                16,
                40,
                &[200, 100],
                epochs,
    )
}

pub fn tale_v1_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                tale_v1(),
                16,
                40,
                &[200, 100],
                epochs,
    )
}

pub fn tale_v1_mini_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                tale_v1(),
                16,
                32,
                &[100],
                epochs,
    )
}

pub fn tale_v1_scout_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                tale_v1(),
                32,
                30,
                &[90],
                epochs,
    )
}

pub fn poke_v2_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v1(),
                16,
                32,
                &[100],
                epochs,
    )
}

pub fn recipe_v1_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                recipe_v1(),
                16,
                40,
                &[200, 100],
                epochs,
    )
}

pub fn poke_v3_behemoth_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v2(),
                64,
                40,
                &[400, 150],
                epochs,
    )
}

pub fn poke_v3_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v2(), // 381 tokens, 300 are multi-char
                64,
                30,
                &[100],
                epochs,
    )
}

pub fn poke_v4_32_context_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v2(), // 381 tokens, 300 are multi-char
                32, // context length
                30, // emb_dim
                &[100, 100, 100], // hidden dim
                epochs,
    )
}

pub fn oasst1_v1_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                oasst1(),
                128,
                64,
                &[1024, 512, 512, 64],
                epochs,
    )
}

pub fn fineweb_v1_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                fineweb(),
                128,
                64,
                &[512, 512, 64],
                epochs,
    )
}

pub fn fineweb_v2_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                fineweb_v2(),
                128,
                64,
                &[512, 512, 64],
                epochs,
    )
}

pub fn recipe_v2_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                recipe_v2(),
                32,
                30,
                &[40, 30],
                epochs,
    )
}

pub fn recipe_v3_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                recipe_v3(),
                32,
                28,
                &[40, 30],
                epochs,
    )
}

pub fn fineweb_hybrid_v1_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v2(),
        128,
        64,
        vec![
            // ============================================================
            // Stage 1
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 64,
                out_channels: 96,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 96,
            },

            // ============================================================
            // Residual Block 1
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 96,
                        kernel_size: 3,
                        stride: 1,
                        padding: 1,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 96,
                        rank: 24,
                        out_channels: 96,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            // 128 -> 64 positions
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 96,
                out_channels: 64,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 64,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 64,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 64,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 64,
                        rank: 16,
                        out_channels: 64,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 64,
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            // 64 -> 32 positions
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 64,
                out_channels: 48,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 48,
            },

            // ============================================================
            // Residual Block 3
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 48,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 48,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 48,
                        rank: 12,
                        out_channels: 48,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 48,
                    },
                ],
            },

            // ============================================================
            // Global bottleneck
            // Input size here should become:
            // 32 positions × 48 channels = 1536
            // ============================================================
            LayerSpec::Dense {
                output_size: 32,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Vocabulary projection
            // ============================================================
            LayerSpec::Dense {
                output_size: fineweb_v2().len(),
                activation: Activation::None,
            },
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v2_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v2(),
        128, // context length: unchanged
        64,  // embedding dimension: unchanged
        vec![
            // ============================================================
            // Stage 1
            // 128 positions × 64 channels
            //
            // Initial full convolution mixes neighboring tokens AND
            // channels. After this, cheaper depthwise/low-rank blocks
            // do most of the feature processing.
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 64,
                out_channels: 96,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 96,
            },

            // ============================================================
            // Residual Block 1
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 96,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 96,
                        rank: 24,
                        out_channels: 96,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            // 128 → 64 positions
            // 96 → 128 channels
            //
            // Increase width as spatial/sequence resolution decreases.
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 96,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            // 64 → 32 positions
            //
            // Keep 128 channels instead of collapsing the representation.
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 3
            //
            // At this low sequence resolution, a larger depthwise kernel
            // is cheap and gives the model a wider effective receptive
            // field.
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Global representation
            //
            // 32 positions × 128 channels = 4096 inputs.
            //
            // Crucially, the representation bottleneck is 64,
            // exactly equal to emb_dim rather than smaller than it.
            // ============================================================
            LayerSpec::Dense {
                output_size: 64,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Vocabulary projection
            // ============================================================
            LayerSpec::Dense {
                output_size: fineweb_v2().len(),
                activation: Activation::None,
            },
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v3_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v2(),
        128, // context length: unchanged
        64,  // embedding dimension: unchanged
        vec![
            // ============================================================
            // Stage 1
            // 128 positions × 64 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 64,
                out_channels: 96,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 96,
            },

            // ============================================================
            // Residual Block 1
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 96,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 96,
                        rank: 24,
                        out_channels: 96,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            // 128 → 64 positions
            // 96 → 128 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 96,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            // 64 → 32 positions
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 3
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Global representation
            //
            // 32 positions × 128 channels = 4096 inputs.
            //
            // Bottleneck = 64 = embedding dimension.
            // ============================================================
            LayerSpec::Dense {
                output_size: 64,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Tied vocabulary projection
            //
            // Reuses the input embedding matrix:
            //     logits[token] = dot(hidden, embedding[token])
            //
            // Adds ZERO new parameters.
            // ============================================================
            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v4_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v3(),
        128, // context length: unchanged
        64,  // embedding dimension: unchanged
        vec![
            // ============================================================
            // Stage 1
            // 128 positions × 64 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 64,
                out_channels: 96,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 96,
            },

            // ============================================================
            // Residual Block 1
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 96,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 96,
                        rank: 24,
                        out_channels: 96,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            // 128 → 64 positions
            // 96 → 128 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 96,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            // 64 → 32 positions
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 3
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Global representation
            //
            // 32 positions × 128 channels = 4096 inputs.
            //
            // Bottleneck = 64 = embedding dimension.
            // ============================================================
            LayerSpec::Dense {
                output_size: 64,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Tied vocabulary projection
            //
            // Reuses the input embedding matrix:
            //     logits[token] = dot(hidden, embedding[token])
            //
            // Adds ZERO new parameters.
            // ============================================================
            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v5_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v3(),
        128, // context length
        128, // embedding dimension
        vec![
            // ============================================================
            // Stage 1
            // 128 positions × 128 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 1
            // Local token mixing + channel mixing
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 48,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            // 128 → 64 positions
            // 128 → 192 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 192,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 192,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 192,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 192,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 192,
                        rank: 64,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 192,
                    },
                ],
            },

            // ============================================================
            // Residual Block 3
            // Larger receptive field before compression
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 192,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 192,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 192,
                        rank: 64,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 192,
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            // 64 → 32 positions
            // 192 → 256 channels
            // ============================================================
            LayerSpec::Conv1D {
                in_channels: 192,
                out_channels: 256,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 256,
            },

            // ============================================================
            // Deep Residual Block 4
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 80,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },
                ],
            },

            // ============================================================
            // Deep Residual Block 5
            // More semantic mixing at the compressed resolution
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 13,
                        stride: 1,
                        padding: 6,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 80,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },
                ],
            },

            // ============================================================
            // Global representation
            //
            // 32 × 256 = 8192 inputs
            // → 128-dimensional language representation
            // ============================================================
            LayerSpec::Dense {
                output_size: 128,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Tied vocabulary projection
            //
            // 128 → 20,262 logits
            // shared embedding matrix
            // ============================================================
            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v6_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v3(),
        128, // context length
        128, // embedding dimension
        vec![
            // ============================================================
            // Stage 1
            // 128 positions × 128 channels
            //
            // Cheap initial local mixing.
            // ============================================================
            LayerSpec::LayerNorm {
                channels: 128,
                epsilon: 1e-5,
            },

            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 1
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 128,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 40,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            //
            // 128 positions → 64
            // 128 channels → 160
            //
            // Grouped convolution keeps this considerably cheaper than
            // a dense 128 → 160 convolution.
            // ============================================================
            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 160,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 160,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 48,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Residual Block 3
            //
            // Another local-mixing block before compression.
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 48,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            //
            // 64 positions → 32
            // 160 channels → 224
            // ============================================================
            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 224,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 224,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 224,
            },

            // ============================================================
            // Residual Block 4
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 224,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 224,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 224,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 224,
                        rank: 56,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Residual Block 5
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 224,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 224,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 224,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 224,
                        rank: 56,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Downsample 3
            //
            // 32 positions → 16
            // 224 channels → 256
            //
            // This is the important architectural change:
            // compress spatially BEFORE the expensive Dense.
            // ============================================================
            LayerSpec::GroupedConv1D {
                in_channels: 224,
                out_channels: 256,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 256,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 256,
            },

            // ============================================================
            // Deep Residual Block 6
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 64,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Deep Residual Block 7
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 64,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Global representation
            //
            // 16 × 256 = 4096 inputs
            // → 128-dimensional representation
            //
            // Half the input dimensionality of v5's 8192 → 128 Dense.
            // ============================================================
            LayerSpec::Dense {
                output_size: 128,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Tied vocabulary projection
            // ============================================================
            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v7_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0, // unlimited batches
        fineweb_v3(),
        128, // context length
        128, // embedding dimension
        vec![
            // ============================================================
            // Stage 1
            // 128 positions × 128 channels
            //
            // Cheap initial local mixing.
            // ============================================================
            LayerSpec::LayerNorm {
                channels: 128,
                epsilon: 1e-5,
            },

            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Residual Block 1
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 128,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 40,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Downsample 1
            //
            // 128 positions → 64
            // 128 channels → 160
            // ============================================================
            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 160,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 160,
            },

            // ============================================================
            // GLOBAL MIXER 1
            //
            // 64 positions × 160 channels
            //
            // Establishes global communication while the representation
            // still retains relatively fine-grained positional detail.
            // ============================================================
            LayerSpec::GlobalMixer {
                channels: 160,
                global_dim: 48,
            },

            // ============================================================
            // Residual Block 2
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 48,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Residual Block 3
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 48,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Downsample 2
            //
            // 64 positions → 32
            // 160 channels → 224
            // ============================================================
            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 224,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 224,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 224,
            },

            // ============================================================
            // GLOBAL MIXER 2
            //
            // 32 positions × 224 channels
            //
            // Stronger global bottleneck after substantial local
            // processing and two rounds of spatial compression.
            // ============================================================
            LayerSpec::GlobalMixer {
                channels: 224,
                global_dim: 64,
            },

            // ============================================================
            // Residual Block 4
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 224,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 224,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 224,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 224,
                        rank: 56,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Residual Block 5
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 224,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 224,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 224,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 224,
                        rank: 56,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Downsample 3
            //
            // 32 positions → 16
            // 224 channels → 256
            // ============================================================
            LayerSpec::GroupedConv1D {
                in_channels: 224,
                out_channels: 256,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 256,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 256,
            },

            // ============================================================
            // Deep Residual Block 6
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 64,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER 3
            //
            // 16 positions × 256 channels
            //
            // At this point the global operation is particularly cheap,
            // while the representation is highly semantic/compressed.
            // ============================================================
            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 96,
            },

            // ============================================================
            // Deep Residual Block 7
            //
            // Local refinement after the final global communication.
            // ============================================================
            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 64,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Global representation
            //
            // 16 × 256 = 4096
            // → 128
            // ============================================================
            LayerSpec::Dense {
                output_size: 128,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Tied vocabulary projection
            // ============================================================
            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v8_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        fineweb_v3(),
        128,
        96,
        vec![
            // ============================================================
            // 128 × 96
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 96,
                epsilon: 1e-5,
            },

            LayerSpec::Conv1D {
                in_channels: 96,
                out_channels: 96,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 96,
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 96,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 96,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 96,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 96,
                        rank: 32,
                        out_channels: 96,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 128 → 64 positions
            // 96 → 128 channels
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 96,
                out_channels: 128,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 128,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            LayerSpec::GlobalMixer {
                channels: 128,
                global_dim: 40,
            },

            // ============================================================
            // 64 × 128
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 128,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 40,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 64 → 32 positions
            // 128 → 160 channels
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 160,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 160,
            },

            LayerSpec::GlobalMixer {
                channels: 160,
                global_dim: 48,
            },

            // ============================================================
            // 32 × 160
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 48,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 32 → 16 positions
            // 160 → 192 channels
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 192,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 192,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 192,
            },

            // ============================================================
            // Final global communication
            // 16 × 192
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 192,
                global_dim: 64,
            },

            // ============================================================
            // Global representation
            // 16 × 192 = 3072
            // → 96
            // ============================================================

            LayerSpec::Dense {
                output_size: 96,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn poke_v5_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        poke_v3(),
        32, // Poke FNN already succeeds with 32 context
        40, // keep the proven-small embedding
        vec![
            // ============================================================
            // 32 × 40
            //
            // Small embedding, wider processing representation.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 40,
                epsilon: 1e-5,
            },

            LayerSpec::Conv1D {
                in_channels: 40,
                out_channels: 48,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 48,
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 48,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 48,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 48,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 48,
                        rank: 24,
                        out_channels: 48,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 32 → 16 positions
            // 48 → 72 channels
            //
            // Stronger representation expansion before global mixing.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 48,
                out_channels: 72,
                groups: 2,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 72,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 72,
            },

            // ============================================================
            // 16 × 72
            //
            // ONE global receptive-field expansion.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 72,
                global_dim: 16,
            },

            // ============================================================
            // 16 × 72
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 72,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 72,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 72,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 72,
                        rank: 24,
                        out_channels: 72,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 16 → 4 positions
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 72,
                out_channels: 72,
                groups: 4,
                kernel_size: 4,
                stride: 4,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 4 → 1 position
            // 72 → 40 channels
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 72,
                out_channels: 40,
                groups: 4,
                kernel_size: 4,
                stride: 1,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn fineweb_hybrid_v9_to(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        fineweb_v3(),
        128,
        128,
        vec![
            // ============================================================
            // 128 × 128
            //
            // Larger embedding space:
            //   20,262 × 128 = 2,593,536 params
            //
            // This is especially useful for the multilingual vocabulary.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 128,
                epsilon: 1e-5,
            },

            LayerSpec::Conv1D {
                in_channels: 128,
                out_channels: 128,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // Local residual processing ×2
            // 128 × 128
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 128,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 48,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 128,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 128,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 128,
                        rank: 48,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 128 → 160 positions / channels
            // 128 → 64 positions
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 160,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 160,
            },

            // ============================================================
            // Local residual processing ×2
            // 64 × 160
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 64,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 160,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 160,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 160,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 160,
                        rank: 64,
                        out_channels: 160,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 64 → 32 positions
            // 160 → 192 channels
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 192,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 192,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 192,
            },

            // ============================================================
            // Local residual processing ×2
            // 32 × 192
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 192,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 192,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 192,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 192,
                        rank: 80,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 192,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 192,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 192,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 192,
                        rank: 80,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // ONE global communication stage
            //
            // 32 × 192
            //
            // Deliberately only one GlobalMixer.
            // Once the representation has enough abstraction, one
            // full-context mixing stage should be more useful than
            // repeatedly paying for global communication.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 192,
                global_dim: 64,
            },

            // ============================================================
            // 32 → 16 positions
            // 192 → 224 channels
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 192,
                out_channels: 224,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 224,
                epsilon: 1e-5,
            },

            LayerSpec::ChannelScale {
                channels: 224,
            },

            // ============================================================
            // Final local refinement
            // 16 × 224
            //
            // Stronger rank here because this representation is about
            // to become the sequence-level language representation.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 224,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 224,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 224,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 224,
                        rank: 96,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // Sequence representation
            //
            // 16 × 224 = 3584
            //
            // Two-step nonlinear bottleneck:
            //   3584 → 256 → 128
            //
            // This is considerably more expressive than:
            //   3584 → 96
            //
            // while keeping the final representation compatible with
            // the tied 128-d embedding matrix.
            // ============================================================

            LayerSpec::Dense {
                output_size: 256,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::Dense {
                output_size: 128,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}
