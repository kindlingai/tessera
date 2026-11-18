//! `task image`: pictures from a prompt, as PNG files.

use super::{load, text};
use crate::source::{Source, Stop};
use sensai_serve::{fresh, square};
use sensai_tasks::png;
use std::path::PathBuf;

#[derive(Debug, Default, PartialEq)]
pub(super) struct Args {
    pub(super) n: Option<usize>,
    pub(super) seed: Option<u32>,
    pub(super) size: Option<usize>,
    pub(super) kind: Option<String>,
    pub(super) clip: Option<String>,
    pub(super) out: Option<PathBuf>,
    pub(super) words: Vec<String>,
}

pub(super) fn parse(args: &[String]) -> Result<Args, Stop> {
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
        let value = rest.next().ok_or(Stop::Usage)?;
        match arg.as_str() {
            "--n" => got.n = Some(value.parse().map_err(|_| Stop::Usage)?),
            "--seed" => got.seed = Some(value.parse().map_err(|_| Stop::Usage)?),
            "--image-size" => got.size = Some(value.parse().map_err(|_| Stop::Usage)?),
            "--task" => got.kind = Some(value.clone()),
            "--clip" => got.clip = Some(value.clone()),
            "-o" => got.out = Some(value.into()),
            _ => return Err(Stop::Usage),
        }
    }
    Ok(got)
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
    let mut flags = Vec::new();
    if let Some(size) = a.size {
        flags.extend(["--image-size".into(), size.to_string()]);
    }
    if let Some(clip) = &a.clip {
        flags.extend(["--clip".into(), clip.clone()]);
    }
    let mut options = source.options(&flags)?;
    let klein = options.model.join("model_index.json").exists();
    let kind = a
        .kind
        .as_deref()
        .unwrap_or(if klein { "klein" } else { "paint" });
    options.flag("--task", kind.into()).ok_or(Stop::Usage)?;
    options.slots = 1;
    let slot = load(options);
    let paint = slot.get().and_then(|m| m.paint.as_ref());
    let paint = paint.ok_or_else(|| Stop::Failed("the model does not paint pictures".into()))?;
    let sizes = paint.sizes();
    let size = square(a.size.unwrap_or(sizes.default), sizes).map_err(Stop::Failed)?;
    let (n, seed) = (a.n.unwrap_or(1), a.seed.unwrap_or_else(fresh));
    if n == 0 {
        return Err(Stop::Usage);
    }
    let images = paint
        .paint(prompt, n, seed, size)
        .map_err(|e| Stop::Failed(e.to_string()))?;
    for (i, image) in images.iter().enumerate() {
        let to = path(&a.out, seed, i, n);
        std::fs::write(&to, png::encode(image))
            .map_err(|e| Stop::Failed(format!("{}: {e}", to.display())))?;
        println!("{}", to.display());
    }
    Ok(())
}
