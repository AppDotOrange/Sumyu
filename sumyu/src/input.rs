use std::io::{self, Write};

pub fn line() -> io::Result<String> {
    let mut input = String::new();

    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;

    Ok(input.trim_end_matches(['\r', '\n']).to_string())
}

pub fn number(max: usize) -> io::Result<Option<usize>> {
    let input = line()?;

    if input.eq_ignore_ascii_case("q") {
        return Ok(None);
    }

    match input.parse::<usize>() {
        Ok(value) if value >= 1 && value <= max => Ok(Some(value - 1)),
        _ => Ok(None),
    }
}

pub fn float(default: f32) -> io::Result<f32> {
    let input = line()?;

    if input.trim().is_empty() {
        return Ok(default);
    }

    Ok(input.parse().unwrap_or(default))
}

pub fn usize(default: usize) -> io::Result<usize> {
    let input = line()?;

    if input.trim().is_empty() {
        return Ok(default);
    }

    Ok(input.parse().unwrap_or(default))
}