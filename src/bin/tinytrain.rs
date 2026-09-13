use std::{fs, io, io::Write};
use grad;
use grad::fnn_lm::LM;
use grad::helper;
use grad::trainer::CheckpointFrequency;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let shake = fs::read_to_string("Datasets/tiny_shakespeare.txt").expect("Can't read tiny_shakespeare.txt!").replace("\r\n", "\n");
    //------------------------------------------------------------------------------------------
    //     CONFIG
    //------------------------------------------------------------------------------------------

    let config = helper::poke_v5_to(1e-3, 32, 10000);
    let description = "A small Sumyu Hybrid trained on Tiny Shakespeare.";

    //------------------------------------------------------------------------------------------
    //     DON'T TOUCH
    //------------------------------------------------------------------------------------------
    let mut lm = LM::from_hybrid_config(config);
    lm.params();
    lm.load_corpus(&*shake);
    lm.train(
        Some(10),
        None,
        None,
        CheckpointFrequency::Disabled,
        None,
        None,
        true,
        1,
    );
    lm.save(
        "PoetP1.sumyu",
        &description,
    );
}
