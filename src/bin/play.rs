use std::io;
use std::io::Write;
#[allow(unused_imports)]
use grad::chatter::ChatFNN;
use grad::fnn_lm::LM;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let tokens = 1_000;
    let (lm) = LM::from_checkpoint("tinychat/MiniChatterV1_batch_17064_epoch_1.check");
    lm.params();
    /*
    let elapsed = std::time::Instant::now();
    lm.generate_gpt("".to_string(), tokens, 0.7);
    let time_elapsed = elapsed.elapsed();
    println!(
        "\n\n\nElapsed: {:?} ({} tokens/second)",
        time_elapsed,
        tokens as f64 / time_elapsed.as_secs_f64()
    );
    */
    let chatter = ChatFNN::new(lm);
    chatter.start_chat(tokens, 0.7, 4);
}
