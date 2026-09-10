use std::{fs, io, io::Write};
use grad;
use grad::fnn_lm::LM;
use grad::helper;
use grad::trainer::CheckpointFrequency;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let poke = fs::read_to_string("Datasets/pokedex.txt").expect("Can't read pokedex.txt!").replace("\r\n", "\n");
    //------------------------------------------------------------------------------------------
    //     CONFIG
    //------------------------------------------------------------------------------------------

    let config = helper::poke_v5_to(0.1, 32, 10000);
    let description = "A tiny Sumyu Hybrid trained on the Pokedex.";

    //------------------------------------------------------------------------------------------
    //     DON'T TOUCH
    //------------------------------------------------------------------------------------------
    let mut lm = LM::from_hybrid_config(config);
    unsafe extern "C" {
        fn openblas_set_num_threads(num_threads: i32);
        fn openblas_get_num_threads() -> i32;
    }
    unsafe {
        openblas_set_num_threads(1);
        println!("OpenBLAS threads: {}", openblas_get_num_threads());
    }
    /*
    let (description, mut lm) = LM::load("pretrainV2.sumyu");
    lm.train_options(0.55, 1, 256, 0);
    */
    lm.params();
    lm.load_corpus(&*poke); //* REPLACE WITH REAL CORPUS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lm.train(
        Some(10),
        None,
        None,
        CheckpointFrequency::Disabled,
        None,
        None,
    );
    lm.save(
        "PokeP5.sumyu",
        &description,
    );
}