//! `fetch`: download a recipe and its weights.

use crate::source::{Source, Stop};

pub fn run(source: Source, rest: &[String]) -> Result<(), Stop> {
    if let (Some(dir), true) = (&source.fetched, rest.is_empty()) {
        println!("weights {}", dir.display());
        return Ok(());
    }
    let (Some(recipe), true) = (&source.recipe, rest.is_empty()) else {
        return Err(Stop::Usage);
    };
    println!("recipe {}", recipe.dir.display());
    let (cache, token) = (source.cache.as_deref(), source.token.as_deref());
    println!("weights {}", recipe.weights(cache, token)?.display());
    for (flag, dir) in recipe.extras(cache, token)? {
        println!("{} {}", flag.trim_start_matches('-'), dir.display());
    }
    Ok(())
}
