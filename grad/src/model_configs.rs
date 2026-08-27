use crate::vocabs::{ml_200_tok_vocab_v3, ml_v4, poke_v1, poke_v2, recipe_v1, recipe_v2, tale_v1, oasst1, fineweb, fineweb_v2, recipe_v3};

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
                32,
                30,
                &[100, 100, 100],
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
