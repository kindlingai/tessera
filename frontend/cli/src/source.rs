//! Where the model comes from, and the options it gives.

use sensai_recipe::Recipe;
use sensai_serve::start::Options;
use std::path::{Path, PathBuf};

/// Why a run stops early.
pub enum Stop {
    Usage,
    /// The task list, after an unknown task name if given.
    Tasks(Option<String>),
    Failed(String),
}

impl From<sensai_recipe::Error> for Stop {
    fn from(e: sensai_recipe::Error) -> Self {
        Stop::Failed(e.to_string())
    }
}

/// Where the model comes from.
pub enum Origin {
    Recipe(String),
    Model(String),
}

/// Splits out the model flags, up to `--`.
pub fn split(given: &[String]) -> Result<(Option<Origin>, Vec<String>), Stop> {
    let mut origin = None;
    let mut rest = Vec::new();
    let mut args = given.iter();
    while let Some(arg) = args.next() {
        if arg == "--" {
            rest.push(arg.clone());
            rest.extend(args.by_ref().cloned());
            break;
        }
        if arg != "--recipe" && arg != "--model" {
            rest.push(arg.clone());
            continue;
        }
        let value = args.next().ok_or(Stop::Usage)?.clone();
        let found = match arg.as_str() {
            "--recipe" => Origin::Recipe(value),
            _ => Origin::Model(value),
        };
        if origin.replace(found).is_some() {
            return Err(Stop::Usage);
        }
    }
    Ok((origin, rest))
}

/// An opened origin.
pub struct Source {
    origin: Origin,
    pub recipe: Option<Recipe>,
    pub cache: Option<PathBuf>,
}

impl Source {
    /// Opens the recipe, if the origin is one.
    pub fn open(origin: Option<Origin>) -> Result<Source, Stop> {
        let origin = origin.ok_or(Stop::Usage)?;
        let cache = sensai_frontend_settings::read().recipes;
        let recipe = match &origin {
            Origin::Recipe(src) => Some(Recipe::open(src, cache.as_deref())?),
            Origin::Model(_) => None,
        };
        Ok(Source {
            origin,
            recipe,
            cache,
        })
    }

    /// Serve options from the recipe, then from `rest`.
    pub fn options(&self, rest: &[String]) -> Result<Options, Stop> {
        let mut options = Options::default();
        if let Some(r) = &self.recipe {
            let cache = self.cache.as_deref();
            options.model = r.weights(cache)?;
            options.name = Some(r.name.clone());
            options.lower = r.lower.clone();
            options.names = r.names();
            if let Some(path) = r.module() {
                let failed = |e: String| Stop::Failed(format!("{}: {e}", path.display()));
                let bytes = std::fs::read(&path).map_err(|e| failed(e.to_string()))?;
                let rules = sensai_engine::wasm::load(&bytes).map_err(|e| failed(e.to_string()))?;
                options.lower.get_or_insert_default().rules.extend(rules);
            }
            for (flag, value) in r.flags() {
                apply(&mut options, &flag, value)?;
            }
        }
        let mut given = rest.iter().cloned();
        while let Some(flag) = given.next() {
            if options.switch(&flag) {
                continue;
            }
            let value = given.next().ok_or(Stop::Usage)?;
            apply(&mut options, &flag, Some(value))?;
        }
        if let Origin::Model(dir) = &self.origin {
            options.model = Path::new(dir).into();
        }
        Ok(options)
    }
}

/// Takes one serve flag. `Usage` when it is bad.
fn apply(options: &mut Options, flag: &str, value: Option<String>) -> Result<(), Stop> {
    match value {
        None if options.switch(flag) => Ok(()),
        Some(value) => options.flag(flag, value).ok_or(Stop::Usage),
        None => Err(Stop::Usage),
    }
}
