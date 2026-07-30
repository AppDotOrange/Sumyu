use grad::fnn_lm::{SavedLM, LM};
use grad::helper;
use std::fs;
use bincode;

fn main() {
    let text = fs::read_to_string("corpus.txt").expect("Can't read corpus.txt!");
    let test = 0;
    if test == 0 {
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
        let epochs = 200;

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
        println!("{:?}", helper::make_vocab(&*text, 200))
    } else if test == 2 {
        let bytes = fs::read("model.bin").unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        let lm = LM::from_saved(model);
        //println!("{}\n\n", lm.generate("pub ".to_string(), 10));
        //lm.generate_one_distribution("".to_string(), 10);
        println!("\"{}\"", lm.generate("".to_string(), 1000))
    }
}
