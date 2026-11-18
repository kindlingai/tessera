//! Writes a prompt's sprites, side by side, to a PNG.

use sensai_tasks::api::Image;
use sensai_tasks::paint::{Options, Painter};
use sensai_tasks::png;
use std::time::Instant;

/// Each sprite pixel becomes this many.
const SCALE: usize = 8;
const N: usize = 4;

/// `images`, all one size, in a row.
fn strip(images: &[Image]) -> Image {
    let (w, h) = (images[0].width, images[0].height);
    let mut rgba = Vec::with_capacity(w * h * 4 * images.len());
    for y in 0..h {
        for i in images {
            rgba.extend_from_slice(&i.rgba[y * w * 4..][..w * 4]);
        }
    }
    Image {
        width: w * images.len(),
        height: h,
        rgba,
    }
}

fn main() -> Result<(), sensai_tasks::Error> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: paint <model dir> <out.png> <seed|greedy> <prompt>";
    let [_, dir, out, mode, prompt @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    let greedy = mode == "greedy";
    let seed = match greedy {
        true => 0,
        false => mode.parse().map_err(|_| usage)?,
    };
    let n = if greedy { 1 } else { N };
    let started = Instant::now();
    let mut painter = Painter::open(dir.as_ref(), None, &sensai_frontend_settings::read(), n)?;
    eprintln!("{}: opened in {:?}", painter.device(), started.elapsed());
    let started = Instant::now();
    let options = Options {
        greedy,
        seed,
        ..Options::default()
    };
    let sprites = painter.paint(&prompt.join(" "), n, &options)?;
    eprintln!("{n} sprites in {:?}", started.elapsed());
    if greedy {
        println!("{:?}", sprites[0]);
    }
    let images: Vec<Image> = sprites
        .iter()
        .map(|t| painter.image(t).scaled(SCALE))
        .collect();
    std::fs::write(out, png::encode(&strip(&images)))?;
    Ok(())
}
