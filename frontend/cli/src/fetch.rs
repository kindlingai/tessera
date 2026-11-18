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
    println!(
        "weights {}",
        recipe
            .weights(source.cache.as_deref(), source.token.as_deref())?
            .display()
    );
    Ok(())
}
