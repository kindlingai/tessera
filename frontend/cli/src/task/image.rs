//! `task image`: pictures from a prompt, as PNG files.

use super::{load, text};
use crate::source::{Source, Stop};
use std::path::PathBuf;
use tessera_serve::{fresh, square};
use tessera_tasks::png;

#[derive(Debug, Default, PartialEq)]
pub(super) struct Args {
    pub(super) n: Option<usize>,
    pub(super) seed: Option<u32>,
    pub(super) size: Option<usize>,
    /// Flags the image pipelines own, passed to the load.
    pub(super) serve: Vec<String>,
    pub(super) out: Option<PathBuf>,
    pub(super) words: Vec<String>,
}

/// The serve task this task loads.
const TASK: &str = "image";

pub(super) fn parse(args: &[String]) -> Result<Args, Stop> {
    let owned = tessera_serve::task::flags_of(TASK);
    let mut got = Args::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if arg == "--" {
            got.words.extend(rest.by_ref().cloned());
            break;
        }
        if !arg.starts_with('-') {
            got.words.push(arg.clone());
            continue;
        }
        let flag = owned.iter().find(|f| f.name == arg);
        if flag.is_some_and(|f| f.value.is_empty()) {
            got.serve.push(arg.clone());
            continue;
        }
        let value = rest.next().ok_or(Stop::Usage)?;
        match arg.as_str() {
            "--n" => got.n = Some(value.parse().map_err(|_| Stop::Usage)?),
            "--seed" => got.seed = Some(value.parse().map_err(|_| Stop::Usage)?),
            "-o" => got.out = Some(value.into()),
            _ if flag.is_none() => return Err(Stop::Usage),
            _ => got.serve.extend([arg.clone(), value.clone()]),
        }
        // The load's largest size is also the request's size.
        if arg == "--image-size" {
            got.size = Some(value.parse().map_err(|_| Stop::Usage)?);
        }
    }
    Ok(got)
}

/// This task's line in the task list.
pub(super) fn usage() -> String {
    let mut flags = String::new();
    for flag in tessera_serve::task::flags_of(TASK) {
        flags += &flag.usage();
        flags += " ";
    }
    format!("image [--n <k>] [--seed <s>] {flags}[-o <path>] [prompt...]")
}

/// Where picture `i` of `n` goes.
pub(super) fn path(out: &Option<PathBuf>, seed: u32, i: usize, n: usize) -> PathBuf {
    let Some(out) = out else {
        let suffix = if n > 1 {
            format!("-{}", i + 1)
        } else {
            String::new()
        };
        return format!("image-{seed}{suffix}.png").into();
    };
    if n == 1 {
        return out.clone();
    }
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    out.with_file_name(format!("{stem}-{}.png", i + 1))
}

pub fn run(source: Source, args: &[String]) -> Result<(), Stop> {
    let a = parse(args)?;
    let prompt = text(&a.words)?;
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(Stop::Usage);
    }
    let mut options = source.options(&a.serve)?;
    options.flag("--task", TASK.into()).ok_or(Stop::Usage)?;
    options.slots = 1;
    let slot = load(options);
    let image = slot.get().and_then(|m| m.image.as_ref());
    let image = image.ok_or_else(|| Stop::Failed("the model cannot make pictures".into()))?;
    let sizes = image.sizes();
    let size = square(a.size.unwrap_or(sizes.default), sizes).map_err(Stop::Failed)?;
    let (n, seed) = (a.n.unwrap_or(1), a.seed.unwrap_or_else(fresh));
    if n == 0 {
        return Err(Stop::Usage);
    }
    let images = image
        .generate(prompt, n, seed, size)
        .map_err(|e| Stop::Failed(e.to_string()))?;
    for (i, image) in images.iter().enumerate() {
        let to = path(&a.out, seed, i, n);
        std::fs::write(&to, png::encode(image))
            .map_err(|e| Stop::Failed(format!("{}: {e}", to.display())))?;
        println!("{}", to.display());
    }
    Ok(())
}
