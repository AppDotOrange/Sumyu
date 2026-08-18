mod input;
mod models;
mod screens;
mod terminal;
mod theme;

use std::io;

pub struct Sumyu {
    terminal: terminal::Terminal,
}

impl Sumyu {
    pub fn new() -> Self {
        Self {
            terminal: terminal::Terminal::new(),
        }
    }

    pub fn start(&mut self) {
        self.terminal.hide_cursor();

        let result = screens::start(&mut self.terminal);

        self.terminal.show_cursor();
        self.terminal.reset();

        if let Err(error) = result {
            eprintln!("Sumyu UI error: {error}");
        }

        let _ = io::Write::flush(&mut io::stdout());
    }
}