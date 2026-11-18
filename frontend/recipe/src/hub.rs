//! Model files from Hugging Face, kept by repository and commit.

use crate::{Error, io, pinned};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// File endings a model directory needs. Other files stay remote.
const KEEP: [&str; 5] = [".json", ".safetensors", ".jinja", ".txt", ".model"];

#[derive(Deserialize)]
struct Info {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

/// `owner/name[@revision]`. A loose revision is pinned and printed.
pub fn fetch(spec: &str, cache: &Path, token: Option<&str>) -> Result<PathBuf, Error> {
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    if pinned(revision) {
        return snapshot(&format!("{repo}@{revision}"), cache, token);
    }
    let info = lookup(repo, revision, token)?;
    let pinned = format!("{repo}@{}", info.sha);
    eprintln!("sensai: model {pinned}");
    snapshot(&pinned, cache, token)
}

/// The repository's files at `revision`.
fn lookup(repo: &str, revision: &str, token: Option<&str>) -> Result<Info, Error> {
    let api = format!("https://huggingface.co/api/models/{repo}/revision/{revision}");
    let response = get(&api, token).map_err(|e| match *e {
        ureq::Error::Status(401 | 404, _) => {
            let why = "not found, or gated: set HF_TOKEN";
            Error::Http(repo.into(), why.into())
        }
        ureq::Error::Transport(_) => Error::Http(repo.into(), "no network".into()),
        other => http(&api)(other),
    })?;
    let info: Info = response
        .into_json()
        .map_err(|e| Error::Http(repo.into(), e.to_string()))?;
    if !info
        .siblings
        .iter()
        .any(|s| s.rfilename.ends_with(".safetensors"))
    {
        return Err(Error::Source(repo.into(), "no safetensors files"));
    }
    Ok(info)
}

fn get(url: &str, token: Option<&str>) -> Result<ureq::Response, Box<ureq::Error>> {
    let request = ureq::get(url);
    match token {
        Some(t) => request.set("Authorization", &format!("Bearer {t}")).call(),
        None => request.call(),
    }
    .map_err(Box::new)
}

/// `owner/name@commit`, downloaded into `cache` once.
pub(crate) fn snapshot(spec: &str, cache: &Path, token: Option<&str>) -> Result<PathBuf, Error> {
    let refuse = || Error::Source(spec.into(), "hf weights are owner/name@<commit>");
    let (repo, commit) = spec.split_once('@').ok_or_else(refuse)?;
    if !pinned(commit) {
        return Err(refuse());
    }
    let dir = cache.join("hf").join(repo).join(commit);
    let done = dir.join(".complete");
    if done.is_file() {
        return Ok(dir);
    }
    let info = lookup(repo, commit, token)?;
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    for s in info.siblings {
        let name = s.rfilename;
        let wanted = !name.contains('/') && KEEP.iter().any(|k| name.ends_with(k));
        if wanted && !dir.join(&name).is_file() {
            let url = format!("https://huggingface.co/{repo}/resolve/{commit}/{name}");
            download(&url, &dir.join(&name), token)?;
        }
    }
    std::fs::write(&done, b"").map_err(io(&done))?;
    Ok(dir)
}

/// Streams `url` to `path` through a `.part` file.
fn download(url: &str, path: &Path, token: Option<&str>) -> Result<(), Error> {
    eprintln!("fetching {url}");
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let response = get(url, token).map_err(|e| http(url)(*e))?;
    let mut file = std::fs::File::create(&part).map_err(io(&part))?;
    std::io::copy(&mut response.into_reader(), &mut file).map_err(io(&part))?;
    std::fs::rename(&part, path).map_err(io(path))
}

fn http(url: &str) -> impl FnOnce(ureq::Error) -> Error + '_ {
    move |e| Error::Http(url.into(), e.to_string())
}
