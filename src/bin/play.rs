use std::io;
use std::io::Write;
#[allow(unused_imports)]
use grad::chatter::ChatFNN;
use grad::fnn_lm::{LM};

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let tokens = 10_000;
    let lm = LM::from_checkpoint("pretraining_v8restart/pretrainConV8_batch_5000_epoch_1.check");
    lm.params();
    //lm.params();
    let elapsed = std::time::Instant::now();
    lm.generate_gpt("".to_string(), tokens, 0.7);
    // lm.generate_one_distribution("Ja se zovem Jakov. Ja sam novinar iz čukumbaba post-a.".to_string(), 25);
    let time_elapsed = elapsed.elapsed();
    println!(
        "\n\n\nElapsed: {:?} ({} tokens/second)",
        time_elapsed,
        tokens as f64 / time_elapsed.as_secs_f64()
    );
    //let chatter = ChatFNN::new(lm);
    //chatter.start_chat(1000, 0.7);
}
