use std::io;
use std::io::Write;
#[allow(unused_imports)]
use grad::chatter::Chatter;
use grad::lm::LM;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let tokens = 1_000;
    let latest_tinychat_p1_small = LM::from_checkpoint("tinychat16/MiniChatterV16_batch_1000_epoch_1.check");
    /*
    let elapsed = std::time::Instant::now();
    lm.generate_gpt("".to_string(), tokens, 0.7, 2);
    let time_elapsed = elapsed.elapsed();
    println!(
        "\n\n\nElapsed: {:?} ({} tokens/second)",
        time_elapsed,
        tokens as f64 / time_elapsed.as_secs_f64()
    );
    */
    let chatter = Chatter::new(latest_tinychat_p1_small);
    chatter.start_chat(tokens, 0.7, 1);
}
