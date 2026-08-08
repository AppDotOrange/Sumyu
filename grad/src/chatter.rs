use crate::fnn_lm::LM;
use std::io;

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
            let mut user = "".to_string();
            io::stdin().read_line(&mut user).unwrap();
            context.push_str(&user);
            context.push_str("<EOT><BOT>");
            let mut bot = &*self.lm.generate(context.clone(), max_gen_length, temperature);
            bot = bot.split("<EOT>").collect::<Vec<&str>>()[0];
            context.push_str(bot);
            context.push_str("<EOT><USER>");
            println!("{}", bot)
        }
    }
}
