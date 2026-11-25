//! `image`: pictures from a prompt, as PNG files.

use super::{Run, Sink, Task, Usage};
use crate::Error;
use crate::api::{Model, fresh};
use crate::png;
use std::path::PathBuf;

/// Make PNG pictures from a prompt.
#[derive(Task)]
#[task(kind = "image", example = "draw two pictures of a red bicycle")]
pub struct Image {
    /// The subject of the pictures.
    #[arg(positional, min = 1, max = 500, example = "a red bicycle")]
    pub prompt: String,
    /// How many pictures to make.
    #[arg(value = "<k>", min = 1, max = 8, example = 2)]
    pub n: Option<usize>,
    /// The seed that repeats a set of pictures.
    #[arg(value = "<s>", cli_only)]
    pub seed: Option<u32>,
    /// Pixels a side, and the largest size the load plans.
    #[arg(flag = "--image-size", value = "<px>", load, cli_only)]
    pub size: Option<usize>,
    /// Flags the image pipelines own, passed to the load.
    #[arg(extra)]
    pub pipeline: Vec<String>,
    /// The PNG path. More pictures add a number to it.
    #[arg(flag = "-o", value = "<path>", cli_only)]
    pub out: Option<PathBuf>,
}

/// Where picture `i` of `n` goes.
pub fn path(out: &Option<PathBuf>, seed: u32, i: usize, n: usize) -> PathBuf {
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

impl Run<&Model> for Image {
    fn run(self, model: &Model, sink: &mut dyn Sink) -> Result<(), Error> {
        let (prompt, n) = (self.prompt.trim(), self.n.unwrap_or(1));
        if prompt.is_empty() || n == 0 {
            return Err(Usage("image needs a prompt and an --n above 0".into()).into());
        }
        let image = model.image.as_ref();
        let image = image.ok_or("the model cannot make pictures")?;
        let sizes = image.sizes();
        let size = sizes.square(self.size.unwrap_or(sizes.default))?;
        let seed = self.seed.unwrap_or_else(fresh);
        let pictures = image.generate(prompt, n, seed, size)?;
        for (i, picture) in pictures.iter().enumerate() {
            let to = path(&self.out, seed, i, n);
            let wrote = std::fs::write(&to, png::encode(picture));
            wrote.map_err(|e| format!("{}: {e}", to.display()))?;
            sink.file(&to);
        }
        Ok(())
    }
}
