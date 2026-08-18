use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

use grad::fnn_lm::LM;

#[derive(Clone)]
pub struct ModelFile {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Clone)]
pub struct DatasetFile {
    pub name: String,
    pub path: PathBuf,
}

pub fn find_models() -> io::Result<Vec<ModelFile>> {
    find_files("models", "sumyu")
}

pub fn find_datasets() -> io::Result<Vec<DatasetFile>> {
    find_files("datasets", "txt")
}

fn find_files<T>(directory: &str, extension: &str) -> io::Result<Vec<T>>
where
    T: FromFile,
{
    let dir = &Path::new(env!("CARGO_MANIFEST_DIR")).join(directory);

    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }

    let mut files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;

        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        if path.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.eq_ignore_ascii_case(extension))
            != Some(true)
        {
            continue;
        }

        files.push(T::from_file(path));
    }

    files.sort_by_key(|x| x.name().to_lowercase());

    Ok(files)
}

pub trait FromFile {
    fn from_file(path: PathBuf) -> Self;
    fn name(&self) -> &str;
}

impl FromFile for ModelFile {
    fn from_file(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        Self { name, path }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl FromFile for DatasetFile {
    fn from_file(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        Self { name, path }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub fn load_model(model: &ModelFile) -> io::Result<(String, LM)> {
    let path = model.path.to_str().unwrap();
    Ok(LM::load_silent(path))
}
