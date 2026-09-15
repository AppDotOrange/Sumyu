use crate::neuron::{Activation, LayerSpec};
use crate::vocabs::{ml_200_tok_vocab_v3, ml_v4, poke_v1, poke_v2, poke_v3, recipe_v1, recipe_v2, tale_v1, oasst1, fineweb, fineweb_v2, recipe_v3, fineweb_v3, tiny_shakespeare_v1, tinychat_v1, tinychat_v2};

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

pub fn tiny_shakespeare_v1_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tiny_shakespeare_v1(),
        256, // Long enough to capture dialogue / sentence structure.
        64,  // Large enough to exploit the 607-token BPE-ish vocabulary.
        vec![

            // ============================================================
            // 256 × 64
            //
            // Small embedding, then expand into a richer processing space.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 64,
                epsilon: 1e-5,
            },

            LayerSpec::Conv1D {
                in_channels: 64,
                out_channels: 96,
                kernel_size: 5,
                stride: 1,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 96,
            },

            // ============================================================
            // 256 × 96
            //
            // Early local language processing.
            // Kernel 7 gives a fairly cheap local receptive field while
            // LowRankPointwise provides channel interaction.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![

                    LayerSpec::LayerNorm {
                        channels: 96,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 96,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
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
            // 256 → 128 positions
            // 96 → 128 channels
            //
            // Spend parameters here rather than keeping the sequence at
            // full resolution. 128 channels gives the mixer substantially
            // more representational capacity.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 96,
                out_channels: 128,
                groups: 2,
                kernel_size: 5,
                stride: 2,
                padding: 2,
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

            // ============================================================
            // 128 × 128
            //
            // FIRST GLOBAL MIXER
            //
            // At this point local features already contain useful
            // short-range language information, so the mixer does not
            // have to discover everything from raw token embeddings.
            //
            // global_dim = 16
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 128,
                global_dim: 16,
            },

            // ============================================================
            // 128 × 128
            //
            // Post-global local refinement.
            // Kernel 9 lets the network reorganize information gathered
            // from the global bottleneck using fairly cheap depthwise
            // convolution.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![

                    LayerSpec::LayerNorm {
                        channels: 128,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 128,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
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
                ],
            },

            // ============================================================
            // 128 → 32 positions
            //
            // Four-token-ish chunks are now compressed into a smaller
            // sequence. This dramatically cheapens another global pass.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 128,
                groups: 4,
                kernel_size: 4,
                stride: 4,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 32 × 128
            //
            // SECOND LOCAL REFINEMENT
            //
            // Now each position represents a much larger chunk of the
            // original 256-token context.
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
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 32 × 128
            //
            // SECOND GLOBAL MIXER
            //
            // This one is cheap because the sequence has already been
            // compressed from 128 → 32 positions.
            //
            // It gives the model another opportunity to connect distant
            // chunks after substantial local processing has occurred.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 128,
                global_dim: 16,
            },

            // ============================================================
            // 32 → 8 positions
            // 128 → 96 channels
            //
            // Aggressive late compression. At this point each position
            // represents a substantial portion of the context.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 96,
                groups: 4,
                kernel_size: 4,
                stride: 4,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 8 → 1 position
            // 96 → 64 channels
            //
            // Final learned aggregation of the whole context.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 96,
                out_channels: 64,
                groups: 4,
                kernel_size: 8,
                stride: 1,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 1 × 64
            //
            // Final normalization before weight tying.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 64,
                epsilon: 1e-5,
            },

            // ============================================================
            // 64 → vocab
            //
            // Weight tying means there is NO separate 64 × 607 output
            // parameter matrix.
            // ============================================================

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v1_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v1(),
        256,
        96,
        vec![

            // ============================================================
            // 256 × 96
            //
            // Downsample immediately to avoid expensive full-resolution
            // residual processing.
            //
            // 256 × 96 → 128 × 128
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 96,
                epsilon: 1e-5,
            },

            LayerSpec::GroupedConv1D {
                in_channels: 96,
                out_channels: 128,
                groups: 1,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // 128 × 128
            //
            // High-resolution local processing ×2.
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
            // 128 → 64 positions
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

            // ============================================================
            // 64 × 160
            //
            // Local processing ×2.
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
            // GLOBAL MIXER #1
            //
            // 64 × 160
            //
            // First full-context exchange after sufficient local
            // processing, while still retaining 64 spatial positions.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 160,
                global_dim: 32,
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
            // 32 × 192
            //
            // Strong local processing ×2.
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
                        rank: 56,
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
                        rank: 56,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #2
            //
            // 32 × 192
            //
            // Second full-context exchange at a highly compressed,
            // abstract representation.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 192,
                global_dim: 40,
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
            // 16 × 224
            //
            // One final local refinement block.
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
                        rank: 64,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // WHOLE-CONTEXT AGGREGATION
            //
            // 16 × 224 → 1 × 320
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 224,
                out_channels: 320,
                groups: 1,
                kernel_size: 16,
                stride: 1,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 320,
                epsilon: 1e-5,
            },

            // ============================================================
            // 320 → 192 → 96
            // ============================================================

            LayerSpec::Dense {
                output_size: 192,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::Dense {
                output_size: 96,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 96 → 8,260
            //
            // Tied to the input embedding matrix.
            // ============================================================

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v2_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v1(),
        256,
        96,
        vec![

            // ============================================================
            // 256 × 96
            // → 128 × 128
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 96,
                epsilon: 1e-5,
            },

            LayerSpec::GroupedConv1D {
                in_channels: 96,
                out_channels: 128,
                groups: 1,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 128,
            },

            // ============================================================
            // 128 × 128
            // Local processing ×2
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
            // 128 × 128
            // → 64 × 160
            //
            // LN already contains affine parameters, so do NOT put
            // ChannelScale immediately after it.
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

            // ============================================================
            // 64 × 160
            // Local processing ×2
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
            // GLOBAL MIXER #1
            // 64 × 160
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 160,
                global_dim: 32,
            },

            // ============================================================
            // 64 × 160
            // → 32 × 192
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

            // ============================================================
            // 32 × 192
            // Local processing ×2
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
                        rank: 56,
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
                        rank: 56,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #2
            // 32 × 192
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 192,
                global_dim: 40,
            },

            // ============================================================
            // 32 × 192
            // → 16 × 224
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

            // ============================================================
            // 16 × 224
            // Final local refinement
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
                        rank: 64,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #3
            //
            // Cheap full-context exchange immediately before collapsing
            // the remaining 16 positions.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 224,
                global_dim: 32,
            },

            // ============================================================
            // WHOLE-CONTEXT POOLING
            //
            // 16 × 224 → 1 × 224
            //
            // Learned depthwise temporal aggregation.
            //
            // Unlike the old 224→320 k16 grouped conv, this has only
            // 224 × 16 parameters.
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 224,
                kernel_size: 16,
                stride: 16,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 224 → 320
            //
            // Cheap low-rank channel expansion.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 224,
                rank: 64,
                out_channels: 320,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Final projection
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 320,
                epsilon: 1e-5,
            },

            LayerSpec::Dense {
                output_size: 192,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::Dense {
                output_size: 96,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v3_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v1(),
        256,
        160,
        vec![

            // ============================================================
            // 256 × 160
            //
            // Initial normalization.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            // ============================================================
            // 256 × 160
            // → 128 × 192
            //
            // Full channel mixing here is worthwhile because this is
            // the first major representation transition.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 192,
                groups: 1,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 192,
            },

            // ============================================================
            // 128 × 192
            // Local processing ×3
            //
            // Rank 80 gives substantially more channel mixing than the
            // old 192-wide rank-56-ish regime without paying for a dense
            // 192 × 192 projection.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 192,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 192,
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
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
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
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
                        kernel_size: 5,
                        stride: 1,
                        padding: 2,
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
            // GLOBAL MIXER #1
            //
            // 128 × 192
            //
            // This is intentionally the earliest global mixer.
            // It allows information to cross the entire context before
            // the next stride-2 compression destroys temporal resolution.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 192,
                global_dim: 40,
            },

            // ============================================================
            // 128 × 192
            // → 64 × 224
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

            // ============================================================
            // 64 × 224
            // Local processing ×3
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
                        rank: 96,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

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
                        rank: 96,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

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
                        rank: 96,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #2
            //
            // 64 × 224
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 224,
                global_dim: 56,
            },

            // ============================================================
            // 64 × 224
            // → 32 × 256
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

            // ============================================================
            // 32 × 256
            // Local processing ×4
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
                        rank: 112,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

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
                        rank: 112,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

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
                        rank: 112,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

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
                        rank: 112,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #3
            //
            // 32 × 256
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 64,
            },

            // ============================================================
            // 32 × 256
            // → 16 × 320
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 256,
                out_channels: 320,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 16 × 320
            // Local processing ×4
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 128,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 128,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 128,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 128,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // WHOLE-CONTEXT MIX
            //
            // 16 × 320
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 320,
                global_dim: 56,
            },

            // ============================================================
            // WHOLE-CONTEXT POOLING
            //
            // 16 × 320 → 1 × 320
            //
            // Still depthwise: only 320 * 16 weights + 320 biases.
            // The GlobalMixer immediately before this lets the final
            // representation contain information from the entire context.
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 320,
                kernel_size: 16,
                stride: 16,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 320 → 160
            //
            // Rank 160 keeps the final projection expressive without
            // introducing a large dense 320 × 160 matrix.
            //
            // 160 is also the tied embedding dimension.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 320,
                rank: 160,
                out_channels: 160,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Final normalization before tied vocabulary projection.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            // ============================================================
            // 160-dimensional representation
            // → tied 8,260-token vocabulary
            //
            // No extra Dense layers:
            // the embedding space itself is the output space.
            // ============================================================

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v4_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v1(),
        256,
        160,
        vec![

            // ============================================================
            // 256 × 160
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            // ============================================================
            // 256 × 160
            // → 128 × 176
            //
            // Cheap initial downsample.
            // 4 groups is enough here; there is no reason to use a dense
            // 160 → 176 convolution at this point.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 176,
                groups: 4,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 176,
            },

            // ============================================================
            // 128 × 176
            // Local block ×2
            //
            // Rank 56 is enough because the GlobalMixer supplies the
            // complementary global channel interaction.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 176,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 176,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 176,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 176,
                        rank: 56,
                        out_channels: 176,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 176,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 176,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 176,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 176,
                        rank: 56,
                        out_channels: 176,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::GlobalMixer {
                channels: 176,
                global_dim: 32,
            },

            // ============================================================
            // 128 × 176
            // → 64 × 208
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 176,
                out_channels: 208,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 208,
            },

            // ============================================================
            // 64 × 208
            // Local block ×2
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 208,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 208,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 208,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 208,
                        rank: 64,
                        out_channels: 208,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 208,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 208,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 208,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 208,
                        rank: 64,
                        out_channels: 208,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::GlobalMixer {
                channels: 208,
                global_dim: 40,
            },

            // ============================================================
            // 64 × 208
            // → 32 × 240
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 208,
                out_channels: 240,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 240,
            },

            // ============================================================
            // 32 × 240
            // Local block ×2
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 240,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 240,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 240,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 240,
                        rank: 80,
                        out_channels: 240,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 240,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 240,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 240,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 240,
                        rank: 80,
                        out_channels: 240,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::GlobalMixer {
                channels: 240,
                global_dim: 48,
            },

            // ============================================================
            // 32 × 240
            // → 16 × 256
            //
            // Final expansion is deliberately tiny. We don't need a
            // 320-wide representation when the output embedding is 160.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 240,
                out_channels: 256,
                groups: 4,
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
            // 16 × 256
            // One final local block.
            //
            // At this resolution, another 3–4 blocks aren't worth it.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
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
                ],
            },

            // ============================================================
            // Final global communication.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 48,
            },

            // ============================================================
            // 16 × 256 → 1 × 256
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 256,
                kernel_size: 16,
                stride: 16,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 256 → 160
            //
            // Small final bottleneck.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 256,
                rank: 96,
                out_channels: 160,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v5_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v1(),
        256,
        160,
        vec![

            // ============================================================
            // 256 × 160
            //
            // Normalize the tied embedding representation before entering
            // the convolutional hierarchy.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            // ============================================================
            // 256 × 160
            // → 128 × 176
            //
            // Cheap causal downsample. Four groups keep this inexpensive
            // while still allowing cross-channel mixing at later stages.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 160,
                out_channels: 176,
                groups: 4,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 176,
            },

            // ============================================================
            // 128 × 176
            //
            // Two local blocks at the highest useful resolution.
            //
            // This is where short-range syntax / phrase structure gets
            // built, so we deliberately spend compute here.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 176,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 176,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 176,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 176,
                        rank: 56,
                        out_channels: 176,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 176,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 176,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 176,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 176,
                        rank: 56,
                        out_channels: 176,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // First global communication.
            //
            // Position-sensitive mixer should be useful here because the
            // 128-token representation still retains substantial sequence
            // resolution.
            LayerSpec::GlobalMixer {
                channels: 176,
                global_dim: 32,
            },

            // ============================================================
            // 128 × 176
            // → 64 × 208
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 176,
                out_channels: 208,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 208,
            },

            // ============================================================
            // 64 × 208
            //
            // Two local blocks. This is the main "middle reasoning"
            // representation: enough resolution for local dependencies,
            // but much cheaper than working at 128 positions.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 208,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 208,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 208,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 208,
                        rank: 64,
                        out_channels: 208,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 208,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 208,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 208,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 208,
                        rank: 64,
                        out_channels: 208,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // Second global communication.
            LayerSpec::GlobalMixer {
                channels: 208,
                global_dim: 40,
            },

            // ============================================================
            // 64 × 208
            // → 32 × 240
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 208,
                out_channels: 240,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 240,
            },

            // ============================================================
            // 32 × 240
            //
            // One strong local block is enough here. By this point each
            // position already summarizes a substantial portion of the
            // original context, so another pair of blocks has poor
            // compute/value compared with widening the representation.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 240,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 240,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 240,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 240,
                        rank: 72,
                        out_channels: 240,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 32 × 240
            // → 16 × 256
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 240,
                out_channels: 256,
                groups: 4,
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
            // 16 × 256
            //
            // One final local refinement block. Large kernel gives this
            // stage a cheap way to mix neighboring coarse context before
            // the final global communication.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
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
                ],
            },

            // ============================================================
            // Final global communication.
            //
            // At 16 positions the mixer is extremely cheap compared with
            // running another high-resolution block, while allowing
            // information from the entire compressed context to interact.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 48,
            },

            // ============================================================
            // 16 × 256
            // → 1 × 256
            //
            // Collapse the coarse sequence into the final representation.
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 256,
                kernel_size: 16,
                stride: 16,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 256 → 160
            //
            // Moderate-rank output bottleneck. Keeping this below 256 → 160
            // dense is important because the embedding table already
            // dominates the parameter budget.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 256,
                rank: 96,
                out_channels: 160,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 160,
                epsilon: 1e-5,
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v6_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v2(), // 2260-token vocabulary
        256,
        144,
        vec![

            // ============================================================
            // 256 × 144
            //
            // Small embedding because the 2260-token vocabulary makes
            // this dramatically cheaper than the old 8k-vocab model.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 144,
                epsilon: 1e-5,
            },

            // ============================================================
            // 256 × 144
            // → 128 × 192
            //
            // Initial local feature extraction + downsampling.
            //
            // Keep this fairly cheap: the high-resolution representation
            // exists primarily to preserve token-local information before
            // the first global communication.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 144,
                out_channels: 192,
                groups: 4,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 192,
            },

            // ============================================================
            // 128 × 192
            //
            // First local coherence block.
            //
            // This is intentionally only ONE block. The first mixer then
            // gets to see reasonably rich local features instead of us
            // spending a huge amount of compute before communication.
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
                        rank: 40,
                        out_channels: 192,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // FIRST GLOBAL MIXER
            //
            // 128 positions is the most important mixer.
            //
            // It receives fairly local features but still retains enough
            // sequence resolution to potentially learn:
            //
            //   "this word/phrase relates to something much earlier"
            //
            // instead of waiting until the sequence has been compressed
            // to 16 positions.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 192,
                global_dim: 32,
            },

            // ============================================================
            // 128 × 192
            // → 64 × 224
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

            LayerSpec::ChannelScale {
                channels: 224,
            },

            // ============================================================
            // 64 × 224
            //
            // Main local coherence stage.
            //
            // Two relatively cheap blocks give the model enough depth to
            // compose the information received from the first mixer.
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
                        rank: 48,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

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
                        rank: 48,
                        out_channels: 224,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // SECOND GLOBAL MIXER
            //
            // This is deliberately at 64 positions.
            //
            // The first mixer provides early global communication.
            // The second allows the richer middle representation to
            // globally reorganize information after local processing.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 224,
                global_dim: 40,
            },

            // ============================================================
            // 64 × 224
            // → 32 × 256
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

            LayerSpec::ChannelScale {
                channels: 256,
            },

            // ============================================================
            // 32 × 256
            //
            // One strong local block.
            //
            // At this point each position represents a fairly large
            // portion of the original 256-token context, so one block
            // provides useful local refinement without wasting CPU.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 56,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // THIRD GLOBAL MIXER
            //
            // Important: don't wait until 16 positions for this.
            //
            // This gives the model another chance to establish global
            // relationships while there are still 32 distinct spatial
            // positions available.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 48,
            },

            // ============================================================
            // 32 × 256
            // → 16 × 256
            //
            // No widening here. Keeping the channel count fixed saves
            // parameters and, more importantly, memory bandwidth.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 256,
                out_channels: 256,
                groups: 4,
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
            // 16 × 256
            //
            // Final global refinement.
            //
            // This is intentionally cheap because the important global
            // communication has already happened at 128/64/32.
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
                        rank: 48,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // FOURTH GLOBAL MIXER
            //
            // Cheap insurance against the final compression destroying
            // relationships. At 16 positions this costs very little.
            //
            // This mixer is NOT the primary coherence mechanism; it is
            // the final global consolidation stage.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 40,
            },

            // ============================================================
            // 16 × 256
            // → 1 × 256
            //
            // Collapse the compressed sequence.
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 256,
                kernel_size: 16,
                stride: 16,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 256 → 144
            //
            // Small output bottleneck because the tied embedding/logit
            // matrix is only 2260 × 144 now.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 256,
                rank: 80,
                out_channels: 144,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 144,
                epsilon: 1e-5,
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v7_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v2(), // 2260-token vocabulary
        256,
        192,
        vec![

            // ============================================================
            // 256 × 192
            //
            // Larger than v6, but deliberately not 256.
            //
            // The 2260-token vocabulary makes the embedding affordable,
            // while 192 gives substantially more token representation
            // capacity without wasting too many parameters on embeddings.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 192,
                epsilon: 1e-5,
            },

            // ============================================================
            // 256 × 192
            // → 128 × 256
            //
            // Initial local feature extraction.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 192,
                out_channels: 256,
                groups: 4,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 256,
            },

            // ============================================================
            // 128 × 256
            //
            // Two substantial local blocks.
            //
            // We deliberately don't put five blocks here: this resolution
            // is expensive in CPU work. These two should establish local
            // phrase structure before the first global communication.
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

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
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
            // GLOBAL MIXER #1
            //
            // 128 positions is the highest-resolution global stage.
            //
            // This is important for letting information from distant
            // tokens interact before the sequence becomes heavily
            // compressed.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 48,
            },

            // ============================================================
            // 128 × 256
            // → 64 × 320
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 256,
                out_channels: 320,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 320,
            },

            // ============================================================
            // 64 × 320
            //
            // Main feature-composition stage.
            //
            // Three blocks gives the model enough depth to transform the
            // information received from Mixer #1 without the huge compute
            // cost of the previous 5-block design.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 80,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 80,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 15,
                        stride: 1,
                        padding: 7,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 88,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #2
            //
            // Larger channel representation and larger global bottleneck.
            //
            // This is the model's main middle-level global communication
            // stage.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 320,
                global_dim: 64,
            },

            // ============================================================
            // 64 × 320
            // → 32 × 384
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 320,
                out_channels: 384,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 384,
            },

            // ============================================================
            // 32 × 384
            //
            // Deepest stage.
            //
            // We STOP DOWNSAMPLING HERE.
            //
            // Keeping 32 spatial positions means the final mixer can still
            // distinguish many separate regions of the original 256-token
            // context instead of being forced through only 16 slots.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 384,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 384,
                        kernel_size: 9,
                        stride: 1,
                        padding: 4,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 384,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 384,
                        rank: 96,
                        out_channels: 384,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 384,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 384,
                        kernel_size: 11,
                        stride: 1,
                        padding: 5,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 384,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 384,
                        rank: 96,
                        out_channels: 384,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #3
            //
            // Final full-context communication before sequence collapse.
            //
            // This is the most important difference from v6's final stage:
            // the mixer still sees 32 distinct positions.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 384,
                global_dim: 72,
            },

            // ============================================================
            // 32 × 384
            // → 1 × 384
            //
            // Only now collapse the complete sequence.
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 384,
                kernel_size: 32,
                stride: 32,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 384 → 192
            //
            // Output representation for tied logits.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 384,
                rank: 96,
                out_channels: 192,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 192,
                epsilon: 1e-5,
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}

pub fn tinychat_v8_hybrid(
    lr: f32,
    batch_size: usize,
    epochs: usize,
) -> HybridConfig {
    HybridConfig::new(
        lr,
        batch_size,
        0,
        tinychat_v2(), // 2260-token vocabulary
        256,
        192,
        vec![

            // ============================================================
            // 256 × 192
            //
            // Keep the relatively wide embedding representation. With
            // only 2260 vocabulary entries, this is cheap.
            // ============================================================

            LayerSpec::LayerNorm {
                channels: 192,
                epsilon: 1e-5,
            },

            // ============================================================
            // 256 × 192 → 128 × 256
            //
            // First spatial/local transformation.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 192,
                out_channels: 256,
                groups: 4,
                kernel_size: 5,
                stride: 2,
                padding: 2,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 256,
            },

            // ============================================================
            // 128 × 256
            //
            // One local block. We do not waste this resolution on a pile
            // of expensive convolutional blocks.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 256,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 256,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 256,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 256,
                        rank: 48,
                        out_channels: 256,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #1
            //
            // Early global communication while there are still 128
            // temporal positions.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 256,
                global_dim: 48,
            },

            // ============================================================
            // 128 × 256 → 64 × 320
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 256,
                out_channels: 320,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 320,
            },

            // ============================================================
            // 64 × 320
            //
            // One stronger local feature block.
            // ============================================================

            LayerSpec::Residual {
                layers: vec![
                    LayerSpec::LayerNorm {
                        channels: 320,
                        epsilon: 1e-5,
                    },

                    LayerSpec::DepthwiseConv1D {
                        in_channels: 320,
                        kernel_size: 7,
                        stride: 1,
                        padding: 3,
                        causal: true,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },

                    LayerSpec::ChannelScale {
                        channels: 320,
                    },

                    LayerSpec::LowRankPointwise {
                        in_channels: 320,
                        rank: 64,
                        out_channels: 320,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // GLOBAL MIXER #2
            //
            // Main learned global mixing before the expensive fully
            // connected bottleneck.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 320,
                global_dim: 64,
            },

            // ============================================================
            // 64 × 320 → 32 × 320
            //
            // Preserve channel width while reducing spatial cost.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 320,
                out_channels: 320,
                groups: 4,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::ChannelScale {
                channels: 320,
            },

            // ============================================================
            // 32 × 320 → 16 × 256
            //
            // Final spatial compression before the global Dense bottleneck.
            //
            // 16 × 256 = 4096 values.
            //
            // This is the critical point: a true Dense is now affordable
            // enough to give the model full all-to-all interaction.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 320,
                out_channels: 256,
                groups: 4,
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
            // GLOBAL MLP BOTTLENECK
            //
            // 4096 → 1536 → 3072
            //
            // This is NOT pointwise.
            //
            // Every one of these neurons sees the entire 16 × 256
            // representation, so this is the model's strongest true
            // global interaction mechanism.
            //
            // 4096 × 1536 + 1536 × 3072 ≈ 11M parameters.
            //
            // The two GEMMs should also be substantially friendlier to
            // OpenBLAS than additional spatial convolutions.
            // ============================================================

            LayerSpec::Dense {
                output_size: 1536,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::Dense {
                // Interpret 3072 values as 24 × 128 after this layer.
                output_size: 3072,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 24 × 128
            //
            // Re-expand back into a reasonably fine-grained sequence
            // representation after global processing.
            // ============================================================

            LayerSpec::GlobalMixer {
                channels: 128,
                global_dim: 48,
            },

            // ============================================================
            // 24 × 128
            //
            // Cheap local cleanup after the global bottleneck.
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
                        rank: 32,
                        out_channels: 128,
                        activation: Activation::LeakyReLU { slope: 0.01 },
                    },
                ],
            },

            // ============================================================
            // 24 × 128 → 24 × 192
            //
            // Reintroduce the embedding width for the tied LM head.
            // ============================================================

            LayerSpec::GroupedConv1D {
                in_channels: 128,
                out_channels: 192,
                groups: 4,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                causal: true,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // 24 × 192 → 1 × 192
            //
            // Final sequence aggregation.
            // At this point the representation has already undergone:
            //
            // local → global mixer → local → global mixer
            // → full Dense global bottleneck → global mixer → local
            //
            // so this collapse is no longer responsible for doing all
            // of the global reasoning itself.
            // ============================================================

            LayerSpec::DepthwiseConv1D {
                in_channels: 192,
                kernel_size: 24,
                stride: 24,
                padding: 0,
                causal: false,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            // ============================================================
            // Final feature refinement before tied logits.
            // ============================================================

            LayerSpec::LowRankPointwise {
                in_channels: 192,
                rank: 64,
                out_channels: 192,
                activation: Activation::LeakyReLU { slope: 0.01 },
            },

            LayerSpec::LayerNorm {
                channels: 192,
                epsilon: 1e-5,
            },

            LayerSpec::WeightTying,
        ],
        epochs,
    )
}
