pub struct Theme {
    pub reset: &'static str,

    pub title: &'static str,
    pub subtitle: &'static str,

    pub text: &'static str,
    pub muted: &'static str,

    pub success: &'static str,
    pub warning: &'static str,
    pub error: &'static str,

    pub border: &'static str,
    pub highlight: &'static str,
}

impl Theme {
    pub fn normal() -> Self {
        Self {
            reset: "\x1b[0m",

            title: "\x1b[1;36m",
            subtitle: "\x1b[36m",

            text: "\x1b[37m",
            muted: "\x1b[90m",

            success: "\x1b[1;32m",
            warning: "\x1b[1;33m",
            error: "\x1b[1;31m",

            border: "\x1b[36m",
            highlight: "\x1b[1;37m",
        }
    }

    /*
    pub fn colorblind() -> Self {
        Self {
            reset: "\x1b[0m",

            title: "\x1b[1;97m",
            subtitle: "\x1b[97m",

            text: "\x1b[97m",
            muted: "\x1b[90m",

            success: "\x1b[1;97m",
            warning: "\x1b[1;93m",
            error: "\x1b[1;91m",

            border: "\x1b[97m",
            highlight: "\x1b[1;97m",
        }
    }
    */
}