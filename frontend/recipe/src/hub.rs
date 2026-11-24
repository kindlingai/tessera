//! Model files from Hugging Face, kept by repository and commit.

use crate::{Error, io, pinned};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tessera_frontend_settings::{HUB, resolve};

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

/// Dense files of the towers the engine does not run.
const TOWERS: [&str; 2] = ["exlr8-extra-audio.", "exlr8-extra-vision."];

/// The expert width fetched when a recipe names none.
const WIDTH: u32 = 3;

/// Tries without progress before a download fails.
const TRIES: u32 = 8;

/// How long a read waits before the download resumes.
const STALL: Duration = Duration::from_secs(60);

/// How often a download checks its speed.
const WINDOW: Duration = Duration::from_secs(20);

/// Bytes a connection must bring in each `WINDOW`.
const FLOOR: u64 = 8 << 20;

/// Files fetched at once.
const LANES: usize = 4;

/// `owner/name[@revision]`. A loose revision is pinned and printed.
pub fn fetch(spec: &str, cache: &Path, token: Option<&str>) -> Result<PathBuf, Error> {
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    if pinned(revision) {
        return snapshot(&format!("{repo}@{revision}"), cache, token, None);
    }
    let info = lookup(repo, revision, token)?;
    let pinned = format!("{repo}@{}", info.sha);
    eprintln!("tesserax: model {pinned}");
    snapshot(&pinned, cache, token, None)
}

/// Whether `owner/name[@revision]` exists with safetensors files.
pub fn check_hf(spec: &str, token: Option<&str>) -> Result<(), Error> {
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    lookup(repo, revision, token).map(|_| ())
}

#[derive(Deserialize)]
struct Found {
    id: String,
}

/// The top owner's shortest repository that matches `words`.
pub fn search_hf(words: &str, token: Option<&str>) -> Result<Option<String>, Error> {
    let api = format!("https://{HUB}/api/models");
    let mut request = ureq::get(&api).query("search", words);
    for (key, value) in [("sort", "downloads"), ("direction", "-1"), ("limit", "20")] {
        request = request.query(key, value);
    }
    request = request.query("filter", "safetensors");
    if let Some(t) = token {
        request = request.set("Authorization", &format!("Bearer {t}"));
    }
    let response = request.call().map_err(http(&api))?;
    let found: Vec<Found> = response
        .into_json()
        .map_err(|e| Error::Http(api.clone(), e.to_string()))?;
    let owner = |id: &str| id.split('/').next().unwrap_or_default().to_string();
    let Some(lead) = found.first().map(|f| owner(&f.id)) else {
        return Ok(None);
    };
    let mut best: Option<&str> = None;
    for f in &found {
        if owner(&f.id) == lead && best.is_none_or(|b| f.id.len() < b.len()) {
            best = Some(&f.id);
        }
    }
    Ok(best.map(String::from))
}

/// The repository's files at `revision`.
fn lookup(repo: &str, revision: &str, token: Option<&str>) -> Result<Info, Error> {
    let api = format!("https://{HUB}/api/models/{repo}/revision/{revision}");
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
    get_from(url, token, 0)
}

/// `url` from byte `from` on. A stalled read times out.
fn get_from(url: &str, token: Option<&str>, from: u64) -> Result<ureq::Response, Box<ureq::Error>> {
    let agent = ureq::AgentBuilder::new().timeout_read(STALL).build();
    let mut request = agent.get(url);
    if let Some(t) = token {
        request = request.set("Authorization", &format!("Bearer {t}"));
    }
    if from > 0 {
        request = request.set("Range", &format!("bytes={from}-"));
    }
    request.call().map_err(Box::new)
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
    if !files.iter().all(|name| safe_path(name)) {
        return Err(Error::Source(spec.into(), "unsafe file path in repository"));
    }
    fetch_all(repo, commit, &files, &dir, token)?;
    std::fs::write(&done, b"").map_err(io(&done))?;
    Ok(dir)
}

/// Each of `names` into `dir`, `LANES` at a time.
fn fetch_all(
    repo: &str,
    commit: &str,
    names: &[&str],
    dir: &Path,
    token: Option<&str>,
) -> Result<(), Error> {
    let next = AtomicUsize::new(0);
    let lane = || {
        while let Some(name) = names.get(next.fetch_add(1, Ordering::Relaxed)) {
            if let Err(e) = fetch_file(repo, commit, name, dir, token) {
                next.store(names.len(), Ordering::Relaxed);
                return Err(e);
            }
        }
        Ok(())
    };
    std::thread::scope(|s| {
        let lanes: Vec<_> = (0..LANES).map(|_| s.spawn(lane)).collect();
        let mut result = Ok(());
        for handle in lanes {
            let done = handle.join().expect("a fetch lane panicked");
            result = result.and(done);
        }
        result
    })
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
            Some(("dense", file)) => {
                file.ends_with(".safetensors") && !TOWERS.iter().any(|t| file.starts_with(t))
            }
            Some(("manifests", file)) => file.ends_with(".json"),
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

/// A relative path that stays inside its directory.
pub fn safe_path(name: &str) -> bool {
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
    fetch_as(repo, commit, name, &path, token)
}

/// `name` from `repo` at `revision`, saved as `path`.
pub fn fetch_as(
    repo: &str,
    revision: &str,
    name: &str,
    path: &Path,
    token: Option<&str>,
) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    download(&resolve(repo, revision, name), path, token)
}

/// Streams `url` to `path` through a `.part` file.
fn download(url: &str, path: &Path, token: Option<&str>) -> Result<(), Error> {
    eprintln!("fetching {url}");
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    // Tries since the file last grew.
    let mut tries = 1;
    loop {
        let before = held(&part);
        let Err(e) = resume(url, &part, token) else {
            break;
        };
        if held(&part) > before {
            tries = 0;
        } else if tries == TRIES {
            return Err(e);
        }
        eprintln!("tesserax: {e}: resuming");
        std::thread::sleep(Duration::from_secs(tries.into()));
        tries += 1;
    }
    std::fs::rename(&part, path).map_err(io(path))
}

fn held(part: &Path) -> u64 {
    std::fs::metadata(part).map_or(0, |m| m.len())
}

/// Adds the rest of `url` to `part`.
fn resume(url: &str, part: &Path, token: Option<&str>) -> Result<(), Error> {
    let have = held(part);
    let response = match get_from(url, token, have) {
        Err(e) if matches!(*e, ureq::Error::Status(416, _)) => return Ok(()),
        other => other.map_err(|e| http(url)(*e))?,
    };
    let mut file = match response.status() {
        206 => std::fs::OpenOptions::new().append(true).open(part),
        _ => std::fs::File::create(part),
    }
    .map_err(io(part))?;
    let mut reader = response.into_reader();
    let mut buf = vec![0; 1 << 16];
    let (mut since, mut brought) = (Instant::now(), 0);
    loop {
        let n = reader.read(&mut buf).map_err(io(part))?;
        if n == 0 {
            return Ok(());
        }
        file.write_all(&buf[..n]).map_err(io(part))?;
        brought += n as u64;
        if since.elapsed() >= WINDOW {
            if brought < FLOOR {
                return Err(Error::Http(url.into(), "slow connection".into()));
            }
            (since, brought) = (Instant::now(), 0);
        }
    }
}

fn http(url: &str) -> impl FnOnce(ureq::Error) -> Error + '_ {
    move |e| Error::Http(url.into(), e.to_string())
}
