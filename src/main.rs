use grad::fnn_lm::{SavedLM, LM};
use grad::helper;
use std::fs;
use bincode;

fn main() {
    let text = fs::read_to_string("corpus.txt").expect("Can't read corpus.txt!");
    let test = 3;
    if test == -1 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let lr = 0.01;
        let batch_size = 32;
        let max_batches_per_epoch = 10; // 0 means no limit
        let vocab = helper::ml_200_tok_vocab_v3();
        let context_len = 8;
        let emb_dim = 42;
        let hidden_dim: &[usize] = &[200];
        let epochs = 500;

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::new(context_len, vocab, hidden_dim, emb_dim);
        lm.params();
        lm.load_corpus(&*text);
        lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
        lm.train();
    } else if test == 0 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let lr = 0.01;
        let batch_size = 32;
        let max_batches_per_epoch = 0; // 0 means no limit
        let vocab = helper::ml_200_tok_vocab_v3();
        let context_len = 16;
        let emb_dim = 42;
        let hidden_dim: &[usize] = &[400, 200];
        let epochs = 500;

        let load = true;

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        if load {
            let bytes = fs::read("model.bin").unwrap();
            let (model, _): (SavedLM, usize) =
                bincode::serde::decode_from_slice(
                    &bytes,
                    bincode::config::standard(),
                ).unwrap();
            let mut lm = LM::from_saved(model);
            println!("Loaded!");
            lm.params();
            lm.load_corpus(&*text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            lm.train();
            let saved = lm.to_saved();
            let bytes = bincode::serde::encode_to_vec(
                &saved,
                bincode::config::standard(),
            ).unwrap();
            fs::write("model.bin", bytes).unwrap();
        } else {
            let mut lm = LM::new(context_len, vocab, hidden_dim, emb_dim);
            lm.params();
            lm.load_corpus(&*text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            lm.train();
            let saved = lm.to_saved();
            let bytes = bincode::serde::encode_to_vec(
                &saved,
                bincode::config::standard(),
            ).unwrap();
            fs::write("model.bin", bytes).unwrap();
        }
    } else if test == 1 {
        println!("{:?}", helper::make_vocab(&*text, 250))
    } else if test == 2 {
        let bytes = fs::read("Working_saved_models/Rustception V4 Finale.bin").unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        let lm = LM::from_saved(model);
        //println!("{}\n\n", lm.generate("pub ".to_string(), 10));
        //lm.generate_one_distribution("".to_string(), 10);
        //println!("\"{}\"", lm.generate("let x =".to_string(), 100))
        lm.embeds().find_clusters(0.50, helper::ml_200_tok_vocab_v3());
    } else if test == 3 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::rustception_v3_to(0.01, 64, 500);

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        //*
        let bytes = fs::read("Rustception_optimized.bin").unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        let mut lm = LM::from_saved(model);
        lm.train_options(0.1, 100000, 32, 0);
        //*/
        lm.params();
        lm.load_corpus(&*text);
        lm.train();
        let saved = lm.to_saved();
        let bytes = bincode::serde::encode_to_vec(
            &saved,
            bincode::config::standard(),
        ).unwrap();
        fs::write("Rustception_optimized.bin", bytes).unwrap();
    }
}
