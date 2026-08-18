use std::io::{self, Write};

use crate::theme::Theme;

pub struct Terminal {
    pub theme: Theme,
}

impl Terminal {
    pub fn new() -> Self {
        Self {
            theme: Theme::normal(),
        }
    }

    pub fn flush(&self) {
        let _ = io::stdout().flush();
    }

    pub fn clear(&self) {
        print!("\x1B[2J\x1B[3J\x1B[1;1H");
        self.flush();
    }

    pub fn reset(&self) {
        print!("\x1b[0m\x1b[?25h");
        self.flush();
    }

    pub fn hide_cursor(&self) {
        print!("\x1b[?25l");
        self.flush();
    }

    pub fn show_cursor(&self) {
        print!("\x1b[?25h");
        self.flush();
    }

    pub fn cursor(&self, row: u16, col: u16) {
        print!("\x1b[{};{}H", row, col);
    }

    pub fn box_top(&self, width: usize) {
        print!("{}╭{}╮{}", self.theme.border, "─".repeat(width), self.theme.reset);
    }

    pub fn box_bottom(&self, width: usize) {
        print!("{}╰{}╯{}", self.theme.border, "─".repeat(width), self.theme.reset);
    }

    pub fn box_line(&self, text: &str, width: usize) {
        let available = width.saturating_sub(2);
        let text: String = text.chars().take(available).collect();
        let padding = available.saturating_sub(text.chars().count());

        print!(
            "{}│{}{}{}{}│{}",
            self.theme.border,
            self.theme.text,
            text,
            " ".repeat(padding+2),
            self.theme.border,
            self.theme.reset,
        );
    }
}