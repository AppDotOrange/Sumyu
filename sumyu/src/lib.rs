use std::io;
use std::io::Write;
use grad;

pub enum Screen {
    Start,
    Train,
    Chat,
    Prompt,
    ModelMaker,
}

pub struct Theme {
    pub title: &'static str,
    pub success: &'static str,
    pub warning: &'static str,
    pub error: &'static str,
}

pub struct Terminal {
    colorblind: bool,
    screen: Screen,
}

impl Terminal {
    pub fn new(colorblind: bool) -> Self {
        Terminal {
            colorblind,
            screen: Screen::Start,
        }
    }

    pub fn flush() {
        io::stdout().flush().unwrap_or(println!("UI error! (failed to flush)"));
    }

    pub fn clear(&self) {
        print!("\x1B[2J\x1B[H");
        Terminal::flush()
    }

    pub fn change_screen(&mut self, screen: Screen) {
        self.screen = screen;
    }

    pub fn draw(&self) {
        match self.screen {
            _ => todo!(),
        }
    }
}