//! Model files from Hugging Face, kept by repository and commit.

use crate::{Error, io, pinned};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tessera_engine::progress::{Progress, Step, Unit};
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
    size: Option<u64>,
}

/// The token and progress each request uses.
#[derive(Clone, Copy)]
struct Get<'a> {
    token: Option<&'a str>,
    progress: &'a Progress,
    /// The repository step that file bytes add to.
    whole: Option<&'a Step>,
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
pub fn fetch(
    spec: &str,
    cache: &Path,
    token: Option<&str>,
    progress: &Progress,
    (files, manifest): (Files, Option<&str>),
) -> Result<PathBuf, Error> {
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    let pinned = match pinned(revision) {
        true => format!("{repo}@{revision}"),
        false => {
            let pinned = format!("{repo}@{}", lookup(repo, revision, token)?.sha);
            eprintln!("tesserax: model {pinned}");
            pinned
        }
    };
    snapshot(&pinned, cache, token, progress, None, manifest, files)
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
    let api = format!("https://{HUB}/api/models/{repo}/revision/{revision}?blobs=true");
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
    let range = (from > 0).then(|| format!("bytes={from}-"));
    get_range(url, token, range.as_deref())
}

/// `url`, or a `range` of its bytes, timing out stalls.
fn get_range(
    url: &str,
    token: Option<&str>,
    range: Option<&str>,
) -> Result<ureq::Response, Box<ureq::Error>> {
    let agent = ureq::AgentBuilder::new().timeout_read(STALL).build();
    let mut request = agent.get(url);
    if let Some(t) = token {
        request = request.set("Authorization", &format!("Bearer {t}"));
    }
    if let Some(range) = range {
        request = request.set("Range", range);
    }
    request.call().map_err(Box::new)
}

/// `len` bytes of `url` from byte `from`.
fn read_range(url: &str, token: Option<&str>, from: u64, len: u64) -> Result<Vec<u8>, Error> {
    let range = format!("bytes={from}-{}", from + len - 1);
    let response = get_range(url, token, Some(&range)).map_err(|e| http(url)(*e))?;
    if response.status() != 206 {
        return Err(Error::Http(
            url.into(),
            "the server ignored the byte range".into(),
        ));
    }
    let mut bytes = Vec::new();
    let reader = response.into_reader().take(len);
    std::io::BufReader::new(reader)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Http(url.into(), e.to_string()))?;
    Ok(bytes)
}

/// The bytes of repository files in a snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Files {
    #[default]
    Whole,
    /// Large safetensors files hold only headers, then zeros.
    Headers,
}

/// Safetensors files at least this large keep only their header.
const LARGE: u64 = 16 << 20;

/// Whether `files` keeps only the header of `name`.
fn header_only(files: Files, name: &str, size: u64) -> bool {
    files == Files::Headers && name.ends_with(".safetensors") && size >= LARGE
}

/// `name`'s header at `path`, in a sparse file.
fn fetch_header(
    (repo, commit, name): (&str, &str, &str),
    path: &Path,
    token: Option<&str>,
    size: u64,
) -> Result<(), Error> {
    let url = resolve(repo, commit, name);
    let head = read_range(&url, token, 0, 8)?;
    let len = u64::from_le_bytes(head.as_slice().try_into().map_err(|_| short(&url))?);
    if 8 + len > size {
        return Err(short(&url));
    }
    let header = read_range(&url, token, 8, len)?;
    make_parent(path)?;
    let part = part_of(path);
    let mut file = std::fs::File::create(&part).map_err(io(&part))?;
    file.write_all(&head).map_err(io(&part))?;
    file.write_all(&header).map_err(io(&part))?;
    file.set_len(size).map_err(io(&part))?;
    std::fs::rename(&part, path).map_err(io(path))
}

fn short(url: &str) -> Error {
    Error::Http(url.into(), "short safetensors header".into())
}

/// `owner/name@commit`, downloaded into `cache` per width or manifest.
pub(crate) fn snapshot(
    spec: &str,
    cache: &Path,
    token: Option<&str>,
    progress: &Progress,
    width: Option<u32>,
    manifest: Option<&str>,
    files: Files,
) -> Result<PathBuf, Error> {
    let refuse = || Error::Source(spec.into(), "hf weights are owner/name@<commit>");
    let (repo, commit) = spec.split_once('@').ok_or_else(refuse)?;
    if !pinned(commit) {
        return Err(refuse());
    }
    if let Some(name) = manifest
        && (name.is_empty() || name.contains('/') || !safe_path(name))
    {
        return Err(Error::Source(spec.into(), "unsafe manifest name"));
    }
    let marker = match (manifest, width) {
        (Some(name), _) => format!(".complete.{name}"),
        (None, Some(k)) => format!(".complete.k{k}"),
        (None, None) => ".complete".into(),
    };
    let dir = cache.join("hf").join(repo).join(commit);
    if dir.join(&marker).is_file() {
        return Ok(dir);
    }
    // Apart, so header files never pass as whole.
    let dir = match files {
        Files::Whole => dir,
        Files::Headers => cache.join("hf-headers").join(repo).join(commit),
    };
    let done = dir.join(&marker);
    if done.is_file() {
        return Ok(dir);
    }
    let info = lookup(repo, commit, token)?;
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let mut sizes = BTreeMap::new();
    let mut names = Vec::new();
    for sibling in info.siblings {
        sizes.insert(sibling.rfilename.clone(), sibling.size);
        names.push(sibling.rfilename);
    }
    let get = Get {
        token,
        progress,
        whole: None,
    };
    let index = dir.join("model_index.json");
    let folders = if names.iter().any(|n| n == "model_index.json") {
        fetch_file(repo, commit, "model_index.json", &dir, get, None)?;
        Some(components(&index)?)
    } else {
        None
    };
    let layers = match manifest.filter(|_| names.iter().any(|n| n == EXLR8)) {
        Some(name) => Some(read_manifest(repo, commit, name, &names, &dir, get)?),
        None => None,
    };
    let want = match names.iter().any(|n| n == EXLR8) {
        true => exlr8_files(&names, width.unwrap_or(WIDTH), layers.as_ref()),
        false => wanted(&names, folders.as_deref()),
    };
    if !want.iter().all(|name| safe_path(name)) {
        return Err(Error::Source(spec.into(), "unsafe file path in repository"));
    }
    let mut left = 0;
    for name in &want {
        let size = sizes.get(*name).copied().flatten().unwrap_or(0);
        if !dir.join(name).is_file() && !header_only(files, name, size) {
            left += size.saturating_sub(held(&part_of(&dir.join(name))));
        }
    }
    let whole = progress.step(format!("download {repo}"), Unit::Bytes, Some(left));
    let inner = whole.progress();
    let get = Get {
        token,
        progress: &inner,
        whole: Some(&whole),
    };
    let sized: Vec<_> = want.iter().map(|n| (*n, sizes[*n])).collect();
    fetch_all((repo, commit), &sized, (&dir, files), get)?;
    whole.finish();
    std::fs::write(&done, b"").map_err(io(&done))?;
    Ok(dir)
}

/// Each of `names` into `dir`, `LANES` at a time.
fn fetch_all(
    (repo, commit): (&str, &str),
    names: &[(&str, Option<u64>)],
    (dir, files): (&Path, Files),
    get: Get,
) -> Result<(), Error> {
    let next = AtomicUsize::new(0);
    let one = |name: &str, size: Option<u64>| match size {
        Some(size) if header_only(files, name, size) && !dir.join(name).is_file() => {
            fetch_header((repo, commit, name), &dir.join(name), get.token, size)
        }
        _ => fetch_file(repo, commit, name, dir, get, size),
    };
    let lane = || {
        while let Some(&(name, size)) = names.get(next.fetch_add(1, Ordering::Relaxed)) {
            if let Err(e) = one(name, size) {
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

/// Fetches one exlr8 manifest and reads its expert widths.
fn read_manifest(
    repo: &str,
    commit: &str,
    name: &str,
    names: &[String],
    dir: &Path,
    get: Get,
) -> Result<BTreeMap<usize, BTreeSet<u32>>, Error> {
    let file = format!("manifests/exlr8-manifest-{name}.json");
    if !names.iter().any(|n| n == &file) {
        return Err(Error::Source(
            repo.into(),
            "manifest not found in repository",
        ));
    }
    fetch_file(repo, commit, &file, dir, get, None)?;
    let path = dir.join(&file);
    let text = std::fs::read_to_string(&path).map_err(io(&path))?;
    manifest_widths(&text).map_err(|e| Error::Http(path.display().to_string(), e))
}

/// Expert widths per layer from an exlr8 manifest JSON.
pub(crate) fn manifest_widths(text: &str) -> Result<BTreeMap<usize, BTreeSet<u32>>, String> {
    let manifest: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    let Some(layers) = manifest.get("layers").filter(|v| !v.is_null()) else {
        return Ok(out);
    };
    let layers = layers.as_object().ok_or("layers must be an object")?;
    for (key, spec) in layers {
        let layer: usize = key.parse().map_err(|_| format!("layer {key}"))?;
        let obj = spec
            .as_object()
            .ok_or_else(|| format!("layer {layer} must be an object"))?;
        let mut widths = BTreeSet::new();
        read_widths(obj, &mut widths)?;
        match obj.get("shared").filter(|v| !v.is_null()) {
            Some(shared) => {
                let shared = shared
                    .as_object()
                    .ok_or_else(|| format!("layer {layer}.shared must be an object"))?;
                read_widths(shared, &mut widths)?;
            }
            None => {
                let shared: Vec<u32> = widths.iter().map(|&k| k.max(3)).collect();
                widths.extend(shared);
            }
        }
        out.insert(layer, widths);
    }
    Ok(out)
}

/// Widths of `gate`, `up` and `down` in one object.
fn read_widths(
    obj: &serde_json::Map<String, serde_json::Value>,
    out: &mut BTreeSet<u32>,
) -> Result<(), String> {
    for proj in ["gate", "up", "down"] {
        if let Some(val) = obj.get(proj) {
            collect_widths(val, out)?;
        }
    }
    Ok(())
}

/// Collects positive integer widths from scalars or nested arrays.
fn collect_widths(val: &serde_json::Value, out: &mut BTreeSet<u32>) -> Result<(), String> {
    if let Some(n) = val.as_u64() {
        let k = u32::try_from(n)
            .ok()
            .filter(|&k| k > 0)
            .ok_or_else(|| format!("invalid width {n}"))?;
        out.insert(k);
        return Ok(());
    }
    let items = val.as_array().ok_or("expected integer or array width")?;
    for item in items {
        collect_widths(item, out)?;
    }
    Ok(())
}

/// Parses `(layer, width)` from `exlr8-layer-<NNN>.k<K>.safetensors`.
fn parse_expert(file: &str) -> Option<(usize, u32)> {
    let body = file
        .strip_prefix("exlr8-layer-")?
        .strip_suffix(".safetensors")?;
    let (layer, width) = body.split_once(".k")?;
    Some((layer.parse().ok()?, width.parse().ok()?))
}

/// An exlr8 release's files, filtered by width or manifest.
pub(crate) fn exlr8_files<'a>(
    names: &'a [String],
    width: u32,
    layers: Option<&BTreeMap<usize, BTreeSet<u32>>>,
) -> Vec<&'a str> {
    // The shared expert has no K2 codes.
    let fallback = [width, width.max(3)];
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
                    || parse_expert(file).is_some_and(|(layer, k)| {
                        layers
                            .and_then(|m| m.get(&layer))
                            .map_or_else(|| fallback.contains(&k), |ks| ks.contains(&k))
                    })
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
    get: Get,
    size: Option<u64>,
) -> Result<(), Error> {
    let path = dir.join(name);
    if path.is_file() {
        return Ok(());
    }
    make_parent(&path)?;
    download(repo, commit, name, &path, get, size)
}

fn make_parent(path: &Path) -> Result<(), Error> {
    match path.parent() {
        Some(parent) => std::fs::create_dir_all(parent).map_err(io(parent)),
        None => Ok(()),
    }
}

/// `name` from `repo` at `revision`, saved as `path`.
pub fn fetch_as(
    repo: &str,
    revision: &str,
    name: &str,
    path: &Path,
    token: Option<&str>,
    progress: &Progress,
) -> Result<(), Error> {
    make_parent(path)?;
    let get = Get {
        token,
        progress,
        whole: None,
    };
    download(repo, revision, name, path, get, None)
}

fn part_of(path: &Path) -> PathBuf {
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    PathBuf::from(part)
}

/// Streams `name` to `path` through a `.part` file.
fn download(
    repo: &str,
    revision: &str,
    name: &str,
    path: &Path,
    get: Get,
    size: Option<u64>,
) -> Result<(), Error> {
    let url = resolve(repo, revision, name);
    let part = part_of(path);
    let file = get
        .progress
        .step(format!("{repo}/{name}"), Unit::Bytes, size);
    file.set(held(&part));
    // Tries since the file last grew.
    let mut tries = 1;
    loop {
        let before = held(&part);
        let Err(e) = resume(&url, &part, get, &file) else {
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
    std::fs::rename(&part, path).map_err(io(path))?;
    file.finish();
    Ok(())
}

fn held(part: &Path) -> u64 {
    std::fs::metadata(part).map_or(0, |m| m.len())
}

/// Adds the rest of `url` to `part`, counting into `file`.
fn resume(url: &str, part: &Path, get: Get, file: &Step) -> Result<(), Error> {
    let have = held(part);
    let response = match get_from(url, get.token, have) {
        Err(e) if matches!(*e, ureq::Error::Status(416, _)) => return Ok(()),
        other => other.map_err(|e| http(url)(*e))?,
    };
    let mut out = match response.status() {
        206 => std::fs::OpenOptions::new().append(true).open(part),
        _ => std::fs::File::create(part),
    }
    .map_err(io(part))?;
    if response.status() != 206 {
        file.set(0);
    }
    let mut reader = response.into_reader();
    let mut buf = vec![0; 1 << 16];
    let (mut since, mut brought) = (Instant::now(), 0);
    loop {
        let n = reader.read(&mut buf).map_err(io(part))?;
        if n == 0 {
            return Ok(());
        }
        out.write_all(&buf[..n]).map_err(io(part))?;
        file.add(n as u64);
        if let Some(whole) = get.whole {
            whole.add(n as u64);
        }
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
