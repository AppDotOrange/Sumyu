use std::{
    io,
    sync::mpsc,
    thread,
};

use grad::fnn_lm::LM;

use crate::{
    input,
    models::{self, DatasetFile, ModelFile},
    terminal::Terminal,
};

const WIDTH: usize = 50;
fn draw_training(
    terminal: &Terminal,
    dataset: &str,
    step: usize,
    total_steps: usize,
    loss: f32,
    perplexity: f32,
    time: std::time::Duration,
    done: bool,
) {
    terminal.cursor(2, 5);
    print!(
        "{}TRAINING · {}{}",
        terminal.theme.title,
        dataset,
        terminal.theme.reset
    );

    terminal.cursor(4, 5);
    print!("Step       {step} / {total_steps}");

    let progress = if total_steps == 0 {
        0.0
    } else {
        step as f32 / total_steps as f32
    };

    let bar_width = 30;
    let filled = ((progress * bar_width as f32) as usize).min(bar_width);

    terminal.cursor(5, 5);
    print!(
        "Progress   [{}{}] {:>5.1}%",
        "█".repeat(filled),
        "░".repeat(bar_width - filled),
        progress * 100.0,
    );

    terminal.cursor(7, 5);
    print!("Loss       {:.5}", loss);

    terminal.cursor(8, 5);
    print!("Perplexity {:.2}", perplexity);

    terminal.cursor(10, 5);
    print!("Time       {:.1?}", time);

    terminal.cursor(12, 5);
    print!(
        "{}Perplexity{} is roughly how many tokens the model",
        terminal.theme.subtitle,
        terminal.theme.reset,
    );

    terminal.cursor(13, 5);
    print!("considers plausible at each prediction.");

    terminal.cursor(15, 5);

    if done {
        print!(
            "{}✓ Training complete{}",
            terminal.theme.success,
            terminal.theme.reset
        );
    } else {
        print!(
            "{}Ctrl+C{}  Stop after current batch",
            terminal.theme.highlight,
            terminal.theme.reset
        );
    }
}

fn print_wrapped(
    terminal: &Terminal,
    row: u16,
    col: u16,
    text: &str,
    width: usize,
) -> u16 {
    let mut current_row = row;

    for paragraph in text.lines() {
        let mut line = String::new();

        for word in paragraph.split_whitespace() {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
                terminal.cursor(current_row, col);
                print!("{line}");
                current_row += 1;
                line.clear();
            }

            if !line.is_empty() {
                line.push(' ');
            }

            line.push_str(word);
        }

        if !line.is_empty() {
            terminal.cursor(current_row, col);
            print!("{line}");
            current_row += 1;
        }
    }

    current_row
}

pub fn start(terminal: &mut Terminal) -> io::Result<()> {
    loop {
        terminal.clear();
        draw_start(terminal);

        terminal.cursor(11, 5);
        print!("> ");
        terminal.flush();

        let choice = input::line()?;

        match choice.trim() {
            "1" => {
                model_screen(terminal)?;
            }

            "2" => {
                train_screen(terminal)?;
            }

            "q" | "Q" | "3" => {
                return Ok(());
            }

            _ => {}
        }
    }
}

fn draw_start(terminal: &Terminal) {
    terminal.cursor(2, 5);
    print!("{}SUMYU{}", terminal.theme.title, terminal.theme.reset);

    terminal.cursor(3, 5);
    print!(
        "{}A tiny neural language model playground{}",
        terminal.theme.subtitle,
        terminal.theme.reset
    );

    terminal.cursor(5, 5);
    terminal.box_top(WIDTH);

    terminal.cursor(6, 5);
    terminal.box_line("1. Models", WIDTH);

    terminal.cursor(7, 5);
    terminal.box_line("2. Train", WIDTH);

    terminal.cursor(8, 5);
    terminal.box_line("3. Quit", WIDTH);

    terminal.cursor(9, 5);
    terminal.box_bottom(WIDTH);
}

fn model_screen(terminal: &mut Terminal) -> io::Result<()> {
    loop {
        let models = models::find_models()?;

        terminal.clear();

        if models.is_empty() {
            terminal.cursor(3, 5);
            print!(
                "{}No models found.{}",
                terminal.theme.warning,
                terminal.theme.reset
            );

            terminal.cursor(5, 5);
            print!("Put .sumyu files inside the models/ directory.");

            terminal.cursor(7, 5);
            print!("Press Enter to go back.");

            terminal.flush();
            input::line()?;
            return Ok(());
        }

        draw_models(terminal, &models);

        terminal.cursor((models.len() + 7) as u16, 5);
        print!("Select a model, or q to go back: > ");
        terminal.flush();

        let choice = input::line()?;

        if choice.eq_ignore_ascii_case("q") {
            return Ok(());
        }

        let index = match choice.parse::<usize>() {
            Ok(value) if value >= 1 && value <= models.len() => value - 1,
            _ => continue,
        };

        model_menu(terminal, &models[index])?;
    }
}

fn draw_models(terminal: &Terminal, models: &[ModelFile]) {
    terminal.cursor(2, 5);
    print!(
        "{}MODELS{}",
        terminal.theme.title,
        terminal.theme.reset
    );

    for (index, model) in models.iter().enumerate() {
        terminal.cursor((4 + index) as u16, 5);

        print!(
            "{}{}.{} {}{}{}",
            terminal.theme.highlight,
            index + 1,
            terminal.theme.reset,
            terminal.theme.text,
            model.name,
            terminal.theme.reset,
        );
    }
}

fn model_menu(
    terminal: &mut Terminal,
    model: &ModelFile,
) -> io::Result<()> {
    let (description, mut lm) = match models::load_model(model) {
        Ok(value) => value,
        Err(error) => {
            terminal.clear();

            terminal.cursor(3, 5);
            print!(
                "{}Failed to load model:{} {}",
                terminal.theme.error,
                terminal.theme.reset,
                error
            );

            terminal.cursor(5, 5);
            print!("Press Enter to go back.");
            terminal.flush();

            input::line()?;
            return Ok(());
        }
    };

    loop {
        terminal.clear();

        terminal.cursor(2, 5);
        print!(
            "{}{}{}",
            terminal.theme.title,
            model.name,
            terminal.theme.reset
        );

        terminal.cursor(4, 5);
        print!("{}", terminal.theme.muted);

        let description = if description.is_empty() {
            "No description."
        } else {
            &description
        };

        println!("{description}");
        print!("{}", terminal.theme.reset);

        terminal.cursor(7, 5);
        print!("1. Prompt model");

        terminal.cursor(8, 5);
        print!("2. Model information");

        terminal.cursor(9, 5);
        print!("3. Back");

        terminal.cursor(11, 5);
        print!("> ");
        terminal.flush();

        match input::line()?.trim() {
            "1" => {
                prompt_screen(terminal, &mut lm, &model.name)?;
            }

            "2" => {
                information_screen(terminal, &mut lm)?;
            }

            "3" | "q" | "Q" => {
                return Ok(());
            }

            _ => {}
        }
    }
}

fn prompt_screen(
    terminal: &mut Terminal,
    lm: &mut LM,
    model_name: &str,
) -> io::Result<()> {
    let mut temperature = 0.7f32;
    let mut tokens = 50usize;

    loop {
        terminal.clear();

        terminal.cursor(2, 5);
        print!(
            "{}{} · PROMPT{}",
            terminal.theme.title,
            model_name,
            terminal.theme.reset
        );

        terminal.cursor(4, 5);
        print!("Prompt:");

        terminal.cursor(5, 5);
        print!("> ");
        terminal.flush();

        let prompt = input::line()?;

        if prompt.eq_ignore_ascii_case("q") {
            return Ok(());
        }

        terminal.cursor(7, 5);
        print!("Temperature [{temperature:.2}]: ");
        terminal.flush();

        temperature = input::float(temperature)?;

        terminal.cursor(8, 5);
        print!("Tokens [{tokens}]: ");
        terminal.flush();

        tokens = input::usize(tokens)?;

        terminal.cursor(10, 5);
        print!("Generating...");
        terminal.flush();

        let output = lm.generate(prompt.clone(), tokens, temperature);

        terminal.clear();

        terminal.cursor(2, 5);
        print!(
            "{}OUTPUT{}",
            terminal.theme.title,
            terminal.theme.reset
        );

        terminal.cursor(4, 5);
        print!("Prompt:");

        terminal.cursor(5, 5);
        print!("{prompt}");

        terminal.cursor(7, 5);
        print!("Output:");

        let end_row = print_wrapped(
            terminal,
            8,
            5,
            &output,
            WIDTH,
        );

        terminal.cursor(end_row + 1, 5);
        print!(
            "{}Enter{} again   {}q{} back",
            terminal.theme.highlight,
            terminal.theme.reset,
            terminal.theme.highlight,
            terminal.theme.reset,
        );

        terminal.flush();

        let choice = input::line()?;

        if choice.eq_ignore_ascii_case("q") {
            return Ok(());
        }
    }
}

fn information_screen(
    terminal: &mut Terminal,
    lm: &mut LM,
) -> io::Result<()> {
    terminal.clear();

    terminal.cursor(2, 5);
    print!(
        "{}MODEL INFORMATION{}",
        terminal.theme.title,
        terminal.theme.reset
    );

    terminal.cursor(4, 5);
    lm.params_sumyu();

    terminal.cursor(15, 5);
    print!("Press Enter to go back.");
    terminal.flush();

    input::line()?;

    Ok(())
}

fn train_screen(terminal: &mut Terminal) -> io::Result<()> {
    let datasets = models::find_datasets()?;

    terminal.clear();

    terminal.cursor(2, 5);
    print!(
        "{}TRAIN MODEL{}",
        terminal.theme.title,
        terminal.theme.reset
    );

    if datasets.is_empty() {
        terminal.cursor(4, 5);
        print!(
            "{}No datasets found.{}",
            terminal.theme.warning,
            terminal.theme.reset
        );

        terminal.cursor(6, 5);
        print!("Put .txt files inside datasets/.");

        terminal.cursor(8, 5);
        print!("Press Enter to go back.");
        terminal.flush();

        input::line()?;
        return Ok(());
    }

    terminal.cursor(4, 5);
    print!("Choose a dataset:");

    for (index, dataset) in datasets.iter().enumerate() {
        terminal.cursor((6 + index) as u16, 5);
        print!("{}. {}", index + 1, dataset.name);
    }

    let row = 7 + datasets.len() as u16;

    terminal.cursor(row, 5);
    print!("> ");
    terminal.flush();

    let index = match input::number(datasets.len())? {
        Some(index) => index,
        None => return Ok(()),
    };

    training_config(terminal, &datasets[index])
}

fn training_config(
    terminal: &mut Terminal,
    dataset: &DatasetFile,
) -> io::Result<()> {
    terminal.clear();

    terminal.cursor(2, 5);
    print!(
        "{}TRAINING CONFIGURATION{}",
        terminal.theme.title,
        terminal.theme.reset
    );

    terminal.cursor(4, 5);
    print!("Dataset: {}", dataset.name);

    terminal.cursor(6, 5);
    print!("Learning rate [0.1]: ");
    terminal.flush();

    let lr = input::float(0.1)?;

    terminal.cursor(7, 5);
    print!("Batch size [32]: ");
    terminal.flush();

    let batch_size = input::usize(32)?;

    terminal.cursor(8, 5);
    print!("Steps [10]: ");
    terminal.flush();

    let steps = input::usize(10)?;

    terminal.cursor(9, 5);
    print!("Max batches/step [0 = unlimited]: ");
    terminal.flush();

    let max_batches = input::usize(0)?;

    terminal.cursor(11, 5);
    print!("Press Enter to begin training.");
    terminal.flush();

    input::line()?;

    run_training(
        terminal,
        dataset,
        lr,
        batch_size,
        steps,
        max_batches,
    )
}

fn run_training(
    terminal: &mut Terminal,
    dataset: &DatasetFile,
    lr: f32,
    batch_size: usize,
    steps: usize,
    max_batches: usize,
) -> io::Result<()> {
    let corpus = std::fs::read_to_string(&dataset.path)?;
    let dataset_name = dataset.name.clone();

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vocab = grad::helper::make_vocab(&corpus, 200, 0);

        let mut lm = LM::new(
            16,
            vocab,
            &[200, 100],
            42,
        );

        lm.load_corpus_silent(&corpus);

        lm.train_options(
            lr,
            steps,
            batch_size,
            max_batches,
        );

        lm.train_sumyu(tx);

        // Find an unused model filename.
        let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models");

        std::fs::create_dir_all(&models_dir)
            .expect("Failed to create models directory");

        let base_name = format!("{}.sumyu", dataset_name);

        let mut model_path = models_dir.join(&base_name);
        let mut number = 1;

        while model_path.exists() {
            model_path = models_dir.join(format!(
                "{}_{}.sumyu",
                dataset_name,
                number
            ));

            number += 1;
        }

        let description = format!(
            "A .sumyu model trained on {}.",
            dataset_name
        );

        lm.save(
            model_path.to_str().unwrap(),
            &description,
        );

        println!(
            "\nModel saved to {}",
            model_path.display()
        );
    });

    terminal.hide_cursor();

    loop {
        let info = match rx.recv() {
            Ok(info) => info,
            Err(_) => break,
        };

        terminal.clear();

        draw_training(
            terminal,
            &dataset.name,
            info.epoch,
            steps,
            info.loss,
            info.perplexity,
            info.time,
            info.done,
        );

        terminal.flush();

        if info.done {
            break;
        }
    }

    terminal.show_cursor();

    terminal.cursor(20, 5);
    print!("Training finished. Press Enter.");
    terminal.flush();

    input::line()?;

    Ok(())
}
