use std::io;
use std::io::Write;
use grad::chatter::ChatConfig;
#[allow(unused_imports)]
use grad::chatter::Chatter;
use grad::inspection::inspect_text;
use grad::lm::LM;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();

    let tokens = 100;
    let mut lm = LM::from_checkpoint("tinychat16/MiniChatterV16_batch_70099_epoch_1.check");
    lm.user_bot_eot_to_blank_in_vocab();
    lm.params();
    let chatter = Chatter::new(lm, ChatConfig::tinychat());

    chatter.start_chat(tokens, 0.1, 1);
}
