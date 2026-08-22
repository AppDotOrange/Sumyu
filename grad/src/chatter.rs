use crate::fnn_lm::LM;
use std::io;
use std::io::Write;

pub struct ChatFNN {
    lm: LM
}

impl ChatFNN {
    pub fn new(lm: LM) -> Self {
        ChatFNN {
            lm
        }
    }

    pub fn start_chat(&self, max_gen_length: usize, temperature: f32) {
        let mut context = "<USER>".to_string();
        loop {
            print!("USER: ");
            let _ = io::stdout().flush();
            let mut user = "".to_string();
            io::stdin().read_line(&mut user).unwrap();
            context.push_str(&user);
            context.push_str("<EOT>\n<BOT>");
            print!("BOT: ");
            let _ = io::stdout().flush();
            let bot = &*self.lm.generate_gpt_chatter(context.clone(), max_gen_length, temperature);
            context.push_str(bot);
            context.push_str("<EOT>\n\n<USER>");
        }
    }
}
