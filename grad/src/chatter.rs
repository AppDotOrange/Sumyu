use crate::lm::{stream_token, LM, LMInferenceState};
use std::io;
use std::io::Write;

fn feed_text(
    lm: &LM,
    state: &mut LMInferenceState,
    text: &str,
) -> Vec<f32> {
    let ids = lm.tokenize_text(text);

    assert!(
        !ids.is_empty(),
        "Cannot feed empty text into incremental inference"
    );

    let mut logits = Vec::new();

    for id in ids {
        logits = lm.forward_token_id(id, state);
    }

    logits
}

fn generate_incremental_reply(
    lm: &LM,
    state: &mut LMInferenceState,
    mut logits: Vec<f32>,
    max_gen_length: usize,
    temperature: f32,
) -> String {
    let mut output = String::new();
    let mut byte_buffer = Vec::<u8>::new();

    for step in 0..max_gen_length {
        let token_id = lm.sample_next_token(
            &logits,
            temperature,
        );

        let generation = lm.token_text(token_id);

        if generation.contains("<EOT>") {
            let before_eot = generation
                .split("<EOT>")
                .next()
                .unwrap();

            stream_token(before_eot, &mut byte_buffer);

            if !byte_buffer.is_empty() {
                print!(
                    "{}",
                    String::from_utf8_lossy(&byte_buffer)
                );
                byte_buffer.clear();
            }

            let _ = io::stdout().flush();

            output.push_str(before_eot);
            return output;
        }

        output.push_str(generation);

        stream_token(generation, &mut byte_buffer);
        let _ = io::stdout().flush();

        // The sampled token becomes the next input.
        // No need to process it if generation is finished.
        if step + 1 < max_gen_length {
            logits = lm.forward_token_id(
                token_id,
                state,
            );
        }
    }

    if !byte_buffer.is_empty() {
        print!(
            "{}",
            String::from_utf8_lossy(&byte_buffer)
        );
    }

    let _ = io::stdout().flush();
    println!();

    output
}

pub struct Chatter {
    lm: LM,
}

impl Chatter {
    pub fn new(lm: LM) -> Self {
        Chatter { lm }
    }

    pub fn start_chat(
        &self,
        max_gen_length: usize,
        temperature: f32,
        threads: usize,
    ) {
        let _ = threads;

        let mut state = self.lm
            .new_inference_state()
            .unwrap_or_else(|error| {
                panic!(
                    "Cannot start chat inference: {}",
                    error
                )
            });

        // First turn starts with <USER>.
        // Subsequent turns start with <EOT><USER>.
        let mut turn_prefix = "<USER>";

        loop {
            print!("USER: ");
            let _ = io::stdout().flush();

            let mut user = String::new();
            io::stdin().read_line(&mut user).unwrap();
            user = user.trim().to_string();

            let prompt = format!(
                "{}{}<EOT><BOT>",
                turn_prefix,
                user,
            );

            let logits = feed_text(
                &self.lm,
                &mut state,
                &prompt,
            );

            print!("BOT: ");
            let _ = io::stdout().flush();

            let _bot = generate_incremental_reply(
                &self.lm,
                &mut state,
                logits,
                max_gen_length,
                temperature,
            );

            // The next turn continues from the same persistent state.
            turn_prefix = "<EOT><USER>";
        }
    }
}

pub struct ModelComparison {
    models: Vec<(LM, String)>,
}

impl ModelComparison {
    pub fn new(
        models: Vec<LM>,
        names: Vec<&str>,
    ) -> Self {
        ModelComparison {
            models: models
                .iter()
                .map(|x| x.clone())
                .zip(
                    names.iter()
                        .map(|x| String::from(*x))
                        .collect::<Vec<String>>(),
                )
                .collect(),
        }
    }

    pub fn compare_one_message(
        &self,
        max_gen_length: usize,
        temperature: f32,
        threads: usize,
    ) -> Vec<String> {
        let _ = threads;

        print!("USER: ");
        let _ = io::stdout().flush();

        let mut user = String::new();
        io::stdin().read_line(&mut user).unwrap();
        user = user.trim().to_string();

        let prompt = format!(
            "<USER>{}<EOT><BOT>",
            user,
        );

        let mut outputs = Vec::new();

        for (model, name) in self.models.iter() {
            print!("{name} (BOT): ");
            let _ = io::stdout().flush();

            let mut state = model
                .new_inference_state()
                .unwrap_or_else(|error| {
                    panic!(
                        "Cannot start inference for {}: {}",
                        name,
                        error
                    )
                });

            let logits = feed_text(
                model,
                &mut state,
                &prompt,
            );

            let bot = generate_incremental_reply(
                model,
                &mut state,
                logits,
                max_gen_length,
                temperature,
            );

            outputs.push(bot);

            println!();
            println!();
        }

        outputs
    }
}
