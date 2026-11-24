//! Reference files from the `kindlingai/tessera-references` dataset.

use serde::Deserialize;
use std::io::Read;
use std::path::{Path, PathBuf};
use tessera_recipe::{Error, fetch_as, safe_path};

/// The dataset, as a hub path.
const DATASET: &str = "datasets/kindlingai/tessera-references";

/// The dataset commit this checkout reads.
const PINNED: &str = "f17c23f549adfc4bee1f4bfe2cbc32c199a7a3f5";

/// The file in the dataset that lists every reference file.
const MANIFEST: &str = "manifest.json";

#[derive(Deserialize)]
struct Manifest {
    files: Vec<Entry>,
}

/// One file of the reference tree.
#[derive(Deserialize)]
struct Entry {
    path: String,
    sha256: String,
    size: u64,
    /// The model repository that holds the file instead.
    #[serde(flatten)]
    source: Option<Source>,
}

#[derive(Deserialize)]
struct Source {
    repo: String,
    revision: String,
    file: String,
}

/// Every reference file into `dir`. Right-sized files stay.
pub(crate) fn fetch(dir: &Path) -> Result<(), Error> {
    let settings = tessera_frontend_settings::read();
    let token = settings.hf_token.as_deref();
    let list = dir.join(MANIFEST);
    fetch_as(DATASET, PINNED, MANIFEST, &list, token)?;
    let text = std::fs::read_to_string(&list).map_err(|e| Error::Io(list.clone(), e))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .map_err(|e| Error::Http(list.display().to_string(), e.to_string()))?;
    for entry in &manifest.files {
        if !safe_path(&entry.path) {
            return Err(Error::Source(entry.path.clone(), "unsafe file path"));
        }
        let path = dir.join(&entry.path);
        if size(&path) == Some(entry.size) {
            continue;
        }
        if let Some(blob) = cached(settings.hub.as_deref(), entry) {
            link(&blob, &path)?;
            continue;
        }
        let (repo, revision, file) = match &entry.source {
            Some(s) => (s.repo.as_str(), s.revision.as_str(), s.file.as_str()),
            None => (DATASET, PINNED, entry.path.as_str()),
        };
        fetch_as(repo, revision, file, &path, token)?;
        if sha256(&path)? != entry.sha256 {
            std::fs::remove_file(&path).map_err(|e| Error::Io(path.clone(), e))?;
            return Err(Error::Source(entry.path.clone(), "SHA-256 differs"));
        }
    }
    Ok(())
}

fn size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// The Hugging Face cache's copy of a model file.
fn cached(hub: Option<&Path>, entry: &Entry) -> Option<PathBuf> {
    let repo = &entry.source.as_ref()?.repo;
    let blob = hub?
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("blobs")
        .join(&entry.sha256);
    (size(&blob) == Some(entry.size)).then_some(blob)
}

fn link(blob: &Path, path: &Path) -> Result<(), Error> {
    let io = |e| Error::Io(path.to_path_buf(), e);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    // A stale file or a broken link goes first.
    if std::fs::symlink_metadata(path).is_ok() {
        std::fs::remove_file(path).map_err(io)?;
    }
    std::os::unix::fs::symlink(blob, path).map_err(io)
}

/// The lowercase hex SHA-256 of the file at `path`.
fn sha256(path: &Path) -> Result<String, Error> {
    let io = |e| Error::Io(path.to_path_buf(), e);
    let mut file = std::fs::File::open(path).map_err(io)?;
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        context.update(&buf[..n]);
    }
    let digest = context.finish();
    Ok(digest.as_ref().iter().map(|b| format!("{b:02x}")).collect())
}
