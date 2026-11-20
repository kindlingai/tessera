//! A FLUX.2 klein picture for a prompt, written to PNG.

use sensai_bundle::Overrides;
use sensai_checkpoint::File;
use sensai_recipe::Recipe;
use sensai_tasks::klein::{Klein, noise};
use sensai_tasks::png;
use std::time::Instant;

fn main() -> Result<(), sensai_tasks::Error> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: klein <recipe> <out.png> <latent side> <seed | noise.safetensors> <prompt>";
    let [_, recipe, out, side, from, prompt @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    let side: usize = side.parse()?;
    let start = match from.parse::<u64>() {
        Ok(seed) => noise(seed, side * side * 128),
        Err(_) => File::open(from.as_ref())?.floats("noise")?,
    };
    let settings = sensai_frontend_settings::read();
    let (cache, token) = (settings.recipes.clone(), settings.hf_token.clone());
    let recipe = Recipe::open(recipe, cache.as_deref())?;
    let dir = recipe.weights(cache.as_deref(), token.as_deref())?;
    let config = Overrides::new(recipe.overrides());
    let started = Instant::now();
    let mut klein = Klein::open(dir, settings, side * 16, config)?;
    eprintln!("opened in {:?}", started.elapsed());
    let started = Instant::now();
    let picture = klein.paint(&prompt.join(" "), side, &start)?;
    eprintln!(
        "painted {}x{} in {:?}",
        picture.width,
        picture.height,
        started.elapsed()
    );
    std::fs::write(out, png::encode(&picture))?;
    Ok(())
}
