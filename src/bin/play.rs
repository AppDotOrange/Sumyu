#[allow(unused_imports)]
use grad::chatter::ChatFNN;
use grad::fnn_lm::{LM};

fn main() {
    let tokens = 10_000;
    let lm = LM::from_checkpoint("pretraining/pretrainConV1_batch_8000_epoch_1.check");
    //lm.params();
    let elapsed = std::time::Instant::now();
    lm.generate_gpt("".to_string(), tokens, 0.7);
    let time_elapsed = elapsed.elapsed();
    println!(
        "\n\n\nElapsed: {:?} ({} tokens/second)",
        time_elapsed,
        tokens as f64 / time_elapsed.as_secs_f64()
    );
    //let chatter = ChatFNN::new(lm);
    //chatter.start_chat(1000, 0.7);
}
