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
    theme: Theme,
    screen: Screen,
}

impl Terminal {
    pub fn new(colorblind: bool) -> Self {
        let theme;
        if colorblind {
            theme = Theme {
                title: r"\e[0;36m",
                success: r"✓\e[0;32m",
                warning: r"⚠\e[0;33m",
                error: r"✗\e[0;31m",
            }
        } else {
            theme = Theme {
                title: r"\e[0;36m",
                success: r"\e[0;32m",
                warning: r"\e[0;33m",
                error: r"\e[0;31m",
            }
        }
        Terminal {
            theme,
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

    pub fn t(&self) -> &str {
        self.theme.title
    }

    pub fn draw(&self) {
        self.clear();
        match self.screen {
            Screen::Start => {
                println!("{}SUMYU----------------|", self.t());
                println!("|Welcome!            |");
                println!("|Instructions below. |");
                println!("|>                   |");
                println!("|____________________|");
                println!("Type in:");
                println!("1 for selecting models,");
                println!("2 for training models,");
                println!("3 for ")
            },
            _ => {}
        }
    }
}