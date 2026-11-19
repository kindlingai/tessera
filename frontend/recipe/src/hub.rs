//! Model files from Hugging Face, kept by repository and commit.

use crate::{Error, io, pinned};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// File endings a model directory needs. Other files stay remote.
const KEEP: [&str; 5] = [".json", ".safetensors", ".jinja", ".txt", ".model"];

/// PyTorch weights, kept only when a repository has no safetensors.
const PICKLED: [&str; 2] = [".pt", ".bin"];

#[derive(Deserialize)]
struct Info {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

/// The file that marks an exlr8 release.
const EXLR8: &str = "dense/exlr8-vocab.safetensors";

/// The expert width fetched when a recipe names none.
const WIDTH: u32 = 3;

/// `owner/name[@revision]`. A loose revision is pinned and printed.
pub fn fetch(spec: &str, cache: &Path, token: Option<&str>) -> Result<PathBuf, Error> {
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    if pinned(revision) {
        return snapshot(&format!("{repo}@{revision}"), cache, token, None);
    }
    let info = lookup(repo, revision, token)?;
    let pinned = format!("{repo}@{}", info.sha);
    eprintln!("sensai: model {pinned}");
    snapshot(&pinned, cache, token, None)
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

/// `owner/name@commit`, downloaded into `cache` once per `width`.
pub(crate) fn snapshot(
    spec: &str,
    cache: &Path,
    token: Option<&str>,
    width: Option<u32>,
) -> Result<PathBuf, Error> {
    let refuse = || Error::Source(spec.into(), "hf weights are owner/name@<commit>");
    let (repo, commit) = spec.split_once('@').ok_or_else(refuse)?;
    if !pinned(commit) {
        return Err(refuse());
    }
    let dir = cache.join("hf").join(repo).join(commit);
    let done = dir.join(match width {
        Some(k) => format!(".complete.k{k}"),
        None => ".complete".into(),
    });
    if done.is_file() {
        return Ok(dir);
    }
    let info = lookup(repo, commit, token)?;
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    let index = dir.join("model_index.json");
    let folders = if names.iter().any(|n| n == "model_index.json") {
        fetch_file(repo, commit, "model_index.json", &dir, token)?;
        Some(components(&index)?)
    } else {
        None
    };
    let files = match names.iter().any(|n| n == EXLR8) {
        true => exlr8(&names, width.unwrap_or(WIDTH)),
        false => wanted(&names, folders.as_deref()),
    };
    for name in files {
        if !safe_path(name) {
            return Err(Error::Source(spec.into(), "unsafe file path in repository"));
        }
        fetch_file(repo, commit, name, &dir, token)?;
    }
    std::fs::write(&done, b"").map_err(io(&done))?;
    Ok(dir)
}

/// Diffusers component folders that `model_index.json` names.
fn components(index: &Path) -> Result<Vec<String>, Error> {
    let text = std::fs::read_to_string(index).map_err(io(index))?;
    let map: BTreeMap<String, serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| Error::Http(index.display().to_string(), e.to_string()))?;
    let mut folders = Vec::new();
    for (key, value) in map {
        if value.as_array().is_some_and(|pair| pair.len() == 2) {
            folders.push(key);
        }
    }
    Ok(folders)
}

/// Files to fetch. With `folders`, root weights stay remote.
pub(crate) fn wanted<'a>(names: &'a [String], folders: Option<&[String]>) -> Vec<&'a str> {
    let ends = |name: &str, kinds: &[&str]| kinds.iter().any(|k| name.ends_with(k));
    let safe = names.iter().any(|n| n.ends_with(".safetensors"));
    let mut out = Vec::new();
    for name in names {
        if !(ends(name, &KEEP) || (!safe && ends(name, &PICKLED))) {
            continue;
        }
        let keep = match (name.split_once('/'), folders) {
            (None, None) => true,
            (None, Some(_)) => ends(name, &[".json"]),
            (Some((top, _)), Some(folders)) => folders.iter().any(|f| f == top),
            (Some(_), None) => false,
        };
        if keep {
            out.push(name.as_str());
        }
    }
    out
}

/// An exlr8 release's files, with experts at `width` only.
pub(crate) fn exlr8(names: &[String], width: u32) -> Vec<&str> {
    // The shared expert has no K2 codes.
    let codes = [
        format!(".k{width}.safetensors"),
        format!(".k{}.safetensors", width.max(3)),
    ];
    let mut out = Vec::new();
    for name in names {
        let keep = match name.split_once('/') {
            None => KEEP.iter().any(|k| name.ends_with(k)),
            Some(("dense", file)) => file.ends_with(".safetensors"),
            Some(("experts", file)) => {
                file.ends_with(".index.safetensors")
                    || codes.iter().any(|c| file.ends_with(c.as_str()))
            }
            Some(_) => false,
        };
        if keep {
            out.push(name.as_str());
        }
    }
    out
}

pub(crate) fn safe_path(name: &str) -> bool {
    !name.starts_with('/') && !name.contains('\\') && !name.split('/').any(|p| p == "..")
}

/// One repository file into `dir`, unless it is there.
fn fetch_file(
    repo: &str,
    commit: &str,
    name: &str,
    dir: &Path,
    token: Option<&str>,
) -> Result<(), Error> {
    let path = dir.join(name);
    if path.is_file() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/{commit}/{name}");
    download(&url, &path, token)
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
