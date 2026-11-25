//! A FLUX.2 klein picture for a prompt, written to PNG.

use std::sync::Arc;
use std::time::Instant;
use tessera_bundle::Overrides;
use tessera_checkpoint::File;
use tessera_recipe::Recipe;
use tessera_recipe::progress::{Log, Progress};
use tessera_tasks::klein::{Klein, noise};
use tessera_tasks::png;

fn main() -> Result<(), tessera_tasks::Error> {
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
    let settings = tessera_frontend_settings::read();
    let (cache, token) = (settings.recipes.clone(), settings.hf_token.clone());
    let recipe = Recipe::open(recipe, cache.as_deref())?;
    let progress = Progress::new(Arc::new(Log));
    let dir = recipe.weights(cache.as_deref(), token.as_deref(), &progress)?;
    let config = Overrides::new(recipe.overrides());
    let started = Instant::now();
    let mut klein = Klein::open(dir, settings, side * 16, config, &progress)?;
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
