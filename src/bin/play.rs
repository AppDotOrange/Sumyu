use std::io;
use std::io::Write;
use std::path::PathBuf;
use grad::chatter::{Agent, ToolConfig};
#[allow(unused_imports)]
use grad::chatter::Chatter;
use grad::lm::LM;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();

    let tokens = 256;
    let mut lm = LM::from_checkpoint("calculator1/CalcV1_batch_38800_epoch_1.check");
    lm.user_bot_eot_to_blank_in_vocab();
    lm.reconfigure_context_length(1024);
    lm.params();
    let tools_dir = PathBuf::from("tools");
    //If I give each of my 4 friends 5 KM, how many KM do they have all together?
    //I am checking some probability figures. Could you work through these one at a time?
    //A box contains 10 red, 7 blue, and 5 green counters. What is the probability of drawing a blue counter, expressed as a percentage?

    let mut agent = Agent::new(
        lm,
        ToolConfig::calculator(tools_dir),
    ).expect("Agent construction err in src/bin/play.rs, lines 16-19 I think");

    agent.start_chat(
        tokens,
        0.4,
        1,
        false,
    ).expect("agent.start_chat err in src/bin/play.rs, lines 21-26 I think");
}
