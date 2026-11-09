use std::io;
use std::io::Write;
#[allow(unused_imports)]
use grad::chatter::Chatter;
use grad::inspection::inspect_text;
use grad::lm::LM;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    let tokens = 1000;
    let latest_tinychat_p1_small = LM::from_checkpoint("tinychat16/MiniChatterV16_batch_61600_epoch_1.check");
    /*
    let elapsed = std::time::Instant::now();
    latest_tinychat_p1_small.generate_gpt("<BOT>".to_string(), tokens, 0.7, 1);
    let time_elapsed = elapsed.elapsed();
    println!(
        "\n\n\nElapsed: {:?} ({} tokens/second)",
        time_elapsed,
        tokens as f64 / time_elapsed.as_secs_f64()
    );
    */
    //*
    latest_tinychat_p1_small.params();
    let chatter = Chatter::new(latest_tinychat_p1_small);
    chatter.start_chat(tokens, 0.1, 1); // tokens=max tokens/response before force quit, 0.1 or 0.7=temp, 1=threads
    //*/
    /*
    let inspection = inspect_text(&latest_tinychat_p1_small, "<USER>I am feeling quite happy today.<EOT><BOT>")
        .expect("Inspection failed");

    println!("Tokens: {}", inspection.tokens.len());
    println!("GlobalMixers: {}", inspection.global_mixers.len());

    for mixer in &inspection.global_mixers {
        println!(
            "Layer {}: {} positions, {} channels, {} slots, reconstruction error {:.8}",
            mixer.layer_path,
            mixer.positions,
            mixer.channels,
            mixer.global_dim,
            mixer.final_state_max_abs_error,
        );
    }

    inspection
        .write_csv("inspection_output_new")
        .expect("Failed to write inspection CSVs");

    println!("CSV files written to inspection_output_new/");
    */
}
