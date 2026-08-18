use grad::fnn_lm::{SavedLM, LM};
use grad::helper;
use std::{fs, io};
use std::io::Write;
use grad::helper::poke_v2;
use grad::trainer::CheckpointFrequency;
use sumyu;

fn main() {
    let _rust = fs::read_to_string("Datasets/rust.txt").expect("Can't read rust.txt!");
    let text = fs::read_to_string("Datasets/Grimm's Fairy Tales").expect("Can't read Grimm's Fairy Tales!").replace("\r\n", "\n");
    let poke = fs::read_to_string("Datasets/pokedex.txt").expect("Can't read pokedex.txt!").replace("\r\n", "\n");
    let recipe = fs::read_to_string("Datasets/150recipes.txt").expect("Can't read 150recipes.txt!").replace("\r\n", "\n");
    let oasst1 = fs::read_to_string("Datasets/oasst1.txt").expect("Can't read oasst1.txt!").replace("\r\n", "\n");
    let test = 9;
    if test == -2 {
        println!("{}", poke_v2().len())
    } else if test == -1 {
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
        lm.load_corpus(&poke);
        lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
        //lm.train(None);
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
            //lm.train(None);
        } else {
            let mut lm = LM::new(context_len, vocab, hidden_dim, emb_dim);
            lm.params();
            lm.load_corpus(&text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            //lm.train(None);
        let saved = lm.to_saved("");
        let bytes = bincode::serde::encode_to_vec(
            &saved,
            bincode::config::standard(),
        ).unwrap();
        fs::write("model.sumyu", bytes).unwrap();
        }
    } else if test == 1 {
        helper::dump_vocab_as_rust(
            &helper::make_vocab(&oasst1, 10_000, 0),
            "oasst1_vocab.rs",
        ).expect("Failed to dump vocab!");
    } else if test == 2 {
        let bytes = fs::read("ChatterP1.sumyu").unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        let lm = LM::from_saved(model);
        lm.params();
        println!("\"{}\"", lm.generate("<USER>What are monopsonies?<EOT>\n".to_string(), 100, 0.7));
        //lm.generate_one_distribution("".to_string(), 25);
        //println!("\"{}\"", lm.generate("pub fn ".to_string(), 100, 0.7))
        //lm.embeds().find_clusters(0.50, helper::recipe_v1());
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
        //lm.train(None);
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
        let lm = LM::load_legacy("Production/Rustception_P1_mini.sumyu");
        lm.save("Production/Rustception_P1_mini.sumyu", &desc);
    } else if test == 5 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::poke_v4_32_context_to(0.1, 32, 100_000);
        let description = "A Sumyu model trained on a filtered Pokedex.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_config(config);
        /*
        let (description, mut lm) = LM::load("Production/PokeP1_64_minutes.sumyu");
        lm.train_options(0.05, 100_000, 32, 0);
        */

        lm.params();
        lm.load_corpus(&poke);
        //lm.train(None);
        lm.save(
            "PokeP4_32c.sumyu",
            &description,
        );
    } else if test == 6 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::tale_v1_scout_to(0.001, 32, 100_000);
        //let description = "A Sumyu model trained on Grimm's Fairy Tales, obtained from Project Gutenberg.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        //*
        let (description, mut lm) = LM::load("Tests/Tale_V1_scout_10.sumyu");
        lm.train_options(0.001, 100_000, 32, 0);
        //*/
        lm.params();
        lm.load_corpus(&text);
        //lm.train(None);
        lm.save(
            "Tale_V1.sumyu",
            &description,
        );
    } else if test == 7 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::recipe_v1_to(0.1, 32, 100_000);
        let description = "A Sumyu model trained on recipes.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_config(config);
        /*
        let (description, mut lm) = LM::load("Tests/Tale_V1_scout_10.sumyu");
        lm.train_options(0.001, 100_000, 32, 0);
        */
        lm.params();
        lm.load_corpus(&recipe);
        //lm.train(None);
        lm.save(
            "RecipeP2.sumyu",
            &description,
        );
    } else if test == 8 {
        let mut terminal = sumyu::Sumyu::new();
        terminal.start();
    } else if test == 9 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::oasst1_v1_to(0.1, 128, 10);
        //let description = "A huge Sumyu model trained on dialogue (oasst1 dataset).";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        //*
        let (description, mut lm) = LM::load("ChatterP1.sumyu");
        lm.train_options(0.01, 10, 128, 0);
        //*/
        lm.params();
        lm.load_corpus(&oasst1);
        lm.train(Some(10), Some("chatterP1_checks/ChatterV1_batch_4376.check".to_string()), Some("chatterP1_checks/ChatterV1".to_string()), CheckpointFrequency::EveryBatch(1000), Some(0.01));
        lm.save(
            "ChatterP1.sumyu",
            &description,
        );
    } else if test == 10 {
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(2);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
    }
}
