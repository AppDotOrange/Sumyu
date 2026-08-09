use grad::fnn_lm::{SavedLM, LM};
use grad::helper;
use std::{fs, io};
use std::io::Write;

fn main() {
    //let text = fs::read_to_string("corpus.txt").expect("Can't read corpus.txt!");
    let text = fs::read_to_string("Datasets/Grimm's Fairy Tales").expect("Can't read Grimm's Fairy Tales!").replace("\r\n", "\n");
    let test = 6;
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
        lm.load_corpus(&text);
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
            let bytes = fs::read("model.sumyu").unwrap();
            let (model, _): (SavedLM, usize) =
                bincode::serde::decode_from_slice(
                    &bytes,
                    bincode::config::standard(),
                ).unwrap();
            let mut lm = LM::from_saved(model);
            println!("Loaded!");
            lm.params();
            lm.load_corpus(&text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            lm.train();
        } else {
            let mut lm = LM::new(context_len, vocab, hidden_dim, emb_dim);
            lm.params();
            lm.load_corpus(&text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            lm.train();
        let saved = lm.to_saved("");
        let bytes = bincode::serde::encode_to_vec(
            &saved,
            bincode::config::standard(),
        ).unwrap();
        fs::write("model.sumyu", bytes).unwrap();
        }
    } else if test == 1 {
        println!("{:?}", helper::make_vocab(&text, 250))
    } else if test == 2 {
        let bytes = fs::read("Production/PokeP1.sumyu").unwrap();
        //let bytes = fs::read("Production/Rustception_P1_mini.sumyu").unwrap();
        //let bytes = fs::read("Tests/RustceptionV3_2.977054CE.bin").unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        let lm = LM::from_saved(model);
        lm.params();
        println!("\"{}\"", lm.generate("".to_string(), 1000, 0.7));
        //lm.generate_one_distribution("".to_string(), 10);
        //println!("\"{}\"", lm.generate("pub fn ".to_string(), 100, 0.7))
        //lm.embeds().find_clusters(0.50, helper::ml_v4());
    } else if test == 3 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::rustception_v4_mini_to(0.01, 32, 500);

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        //*
        let (description, mut lm) = LM::load("Production/Rustception_P1_mini.sumyu");
        lm.train_options(0.01, 100_000, 32, 0);
        //*/
        lm.params();
        lm.load_corpus(&text);
        lm.train();
        lm.save(
            "Rustception_optimized.sumyu",
            &description,
        );
    } else if test == 4 {
        println!("Updating save format to .sumyu quality...");
        print!("Description: ");
        io::stdout().flush().expect("Failed to flush.");
        let mut desc = String::new();
        io::stdin().read_line(&mut desc).expect("Failed to read line!");
        let desc = desc.trim().to_string();
        let lm = LM::load_legacy("Poke_V1.sumyu");
        lm.save("Poke_V1_working.sumyu", &desc);
    } else if test == 5 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::poke_v1_mini_to(0.1, 32, 100_000);
        //let description = "A Sumyu model trained on a filtered Pokedex.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        //*
        let (description, mut lm) = LM::load("Tests/Poke_V1_134.sumyu");
        lm.train_options(0.001, 100_000, 32, 0);
        //*/
        lm.params();
        lm.load_corpus(&text);
        lm.train();
        lm.save(
            "Poke_V1.sumyu",
            &description,
        );
    } else if test == 6 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::tale_v1_mini_to(0.01, 32, 100_000);
        let description = "A Sumyu model trained on Grimm's Fairy Tales, obtained from Project Gutenberg.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_config(config);
        /*
        let (description, mut lm) = LM::load("Tests/Poke_V1_134.sumyu");
        lm.train_options(0.001, 100_000, 32, 0);
        */
        lm.params();
        lm.load_corpus(&text);
        lm.train();
        lm.save(
            "Tale_V1.sumyu",
            &description,
        );
    }
}
