use std::fs;
use grad::chatter::ChatFNN;
use grad::fnn_lm::{SavedLM, LM};

fn main() {
    let lm = LM::from_checkpoint("pretraining/pretrainV2_batch_13000_epoch_1.check");
    lm.params();
    lm.generate_gpt("".to_string(), 1000, 0.7);
    //let chatter = ChatFNN::new(lm);
    //chatter.start_chat(1000, 0.7);
}
