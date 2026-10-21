use std::io;
use std::io::Write;
#[allow(unused_imports)]
use grad::chatter::Chatter;
use grad::chatter::ModelComparison;
use grad::lm::LM;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let tokens = 1_000;
    let old_tinychatp1_large = LM::from_checkpoint("tinychat15/MiniChatterV15_batch_10000_epoch_1.check");
    let latest_tinychatp1_large = LM::from_checkpoint("tinychat15/MiniChatterV15_batch_52687_epoch_1.check");
    let (_description, tinychatp1) = LM::load("Production/TinyChatP1-post.sumyu");
    //let (description, mut lm) = LM::load("Production/TinyChatP1.sumyu");
    //latest_tinychatp1_large.find_clusters(0.5);
    //latest_tinychatp1_large.params();
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
    let chatter = Chatter::new(latest_tinychatp1_large);
    chatter.start_chat(tokens, 0.7, 1);
    //let comp = ModelComparison::new(vec![latest_tinychatp1_large, old_tinychatp1_large, tinychatp1], vec!["P1-Large latest checkpoint", "P1-Large older checkpoint", "P1-post"]);
    //let outputs = comp.compare_one_message(tokens, 0.7, 1); // 1 thread to not disturb training really
    // 1 thread is fast enough.
    // temp 0.7 has a good history for Sumyu models
}
