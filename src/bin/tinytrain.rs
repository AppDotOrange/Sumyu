use grad;
use grad::fnn_lm::LM;
use grad::helper;
use grad::trainer::CheckpointFrequency;

fn main() {
    //------------------------------------------------------------------------------------------
    //     CONFIG
    //------------------------------------------------------------------------------------------

    let config = helper::fineweb_hybrid_v5_to(0.4, 256, 1);
    let description = "A large Sumyu Hybrid model pre-trained on the FineWeb dataset family.";

    //------------------------------------------------------------------------------------------
    //     DON'T TOUCH
    //------------------------------------------------------------------------------------------
    let mut lm = LM::from_hybrid_config(config);
    unsafe extern "C" {
        fn openblas_set_num_threads(num_threads: i32);
        fn openblas_get_num_threads() -> i32;
    }
    unsafe {
        openblas_set_num_threads(4);
        println!("OpenBLAS threads: {}", openblas_get_num_threads());
    }
    /*
    let (description, mut lm) = LM::load("pretrainV2.sumyu");
    lm.train_options(0.55, 1, 256, 0);
    */
    lm.params();
    lm.load_corpus(""); //* REPLACE WITH REAL CORPUS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lm.train(
        Some(10),
        Some("pretraining/pretrainConV5_batch_11692_epoch_1.check"),
        Some("pretraining/pretrainConV5"),
        CheckpointFrequency::EveryBatch(1000),
        Some(0.3),
        None,
    );
    lm.save(
        "pretrainV5.sumyu",
        &description,
    );
}