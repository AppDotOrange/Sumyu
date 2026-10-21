use crate::lm::LM;
use std::io;
use std::io::Write;

pub struct Chatter {
    lm: LM
}

impl Chatter {
    pub fn new(lm: LM) -> Self {
        Chatter {
            lm
        }
    }

    pub fn start_chat(&self, max_gen_length: usize, temperature: f32, threads: usize) {
        let mut context: String = "<USER>".to_string();
        loop {
            print!("USER: ");
            let _ = io::stdout().flush();
            let mut user: String = "".to_string();
            io::stdin().read_line(&mut user).unwrap();
            user = user.trim().to_string();
            context.push_str(&user);
            context.push_str("<EOT><BOT>");
            print!("BOT: ");
            let _ = io::stdout().flush();
            let bot = &*self.lm.generate_gpt_chatter(context.clone(), max_gen_length, temperature, threads);
            context.push_str(bot);
            context.push_str("<EOT><USER>");
            println!()
        }
    }
}

pub struct ModelComparison {
    models: Vec<(LM, String)>
}

impl ModelComparison {
    pub fn new(models: Vec<LM>, names: Vec<&str>) -> Self {
        ModelComparison {
            // joins models and names into Vec<(LM, String)> with some witchcraft
            models: models.iter().map(|x| {x.clone()}).zip(names.iter().map(|x1| {String::from(*x1)}).collect::<Vec<String>>()).collect()
        }
    }

    pub fn compare_one_message(&self, max_gen_length: usize, temperature: f32, threads: usize) -> Vec<String> {
        let mut context: String = "<USER>".to_string();
        print!("USER: ");
        let _ = io::stdout().flush();
        let mut user: String = "".to_string();
        io::stdin().read_line(&mut user).unwrap();
        user = user.trim().to_string();
        context.push_str(&user);
        context.push_str("<EOT><BOT>");
        let mut outputs = vec![];
        for (model, name) in self.models.clone() {
            print!("{name} (BOT): ");
            let _ = io::stdout().flush();
            let bot = model.generate_gpt_chatter(context.clone(), max_gen_length, temperature, threads);
            outputs.push(bot);
            println!();
            println!();
        }
        outputs
    }
}
