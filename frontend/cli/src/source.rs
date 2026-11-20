//! Where the model comes from, and the options it gives.

use sensai_engine::wasm::Module;
use sensai_recipe::Recipe;
use sensai_serve::start::Options;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// `owner/name` or `owner/name@revision` on Hugging Face.
    Hub(String),
}

/// The repository in a `--model hf:<owner>/<name>[@<revision>]` value.
fn hub_spec(value: &str) -> Option<Result<String, Stop>> {
    let spec = value.strip_prefix("hf:")?;
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    let valid = |part: &str| !part.is_empty() && !part.contains(['/', '@']);
    let named = matches!(repo.split_once('/'), Some((o, n)) if valid(o) && valid(n));
    Some(if named && !revision.is_empty() {
        Ok(spec.to_string())
    } else {
        Err(Stop::Failed(format!(
            "{value}: use hf:<owner>/<name>[@<revision>]"
        )))
    })
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
            _ => match hub_spec(&value) {
                Some(spec) => Origin::Hub(spec?),
                None => Origin::Model(value),
            },
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
    pub token: Option<String>,
    /// The snapshot directory, for a Hugging Face origin.
    pub fetched: Option<PathBuf>,
}

impl Source {
    /// Opens the recipe, if the origin is one.
    pub fn open(origin: Option<Origin>) -> Result<Source, Stop> {
        let origin = origin.ok_or(Stop::Usage)?;
        let settings = sensai_frontend_settings::read();
        let (cache, token) = (settings.recipes, settings.hf_token);
        let mut fetched = None;
        let recipe = match &origin {
            Origin::Recipe(src) => Some(Recipe::open(src, cache.as_deref())?),
            Origin::Model(_) => None,
            Origin::Hub(spec) => {
                let dir = cache.as_deref().ok_or(sensai_recipe::Error::NoCache)?;
                fetched = Some(sensai_recipe::fetch_hf(spec, dir, token.as_deref())?);
                None
            }
        };
        Ok(Source {
            origin,
            recipe,
            cache,
            token,
            fetched,
        })
    }

    /// Serve options from the recipe, then from `rest`.
    pub fn options(&self, rest: &[String]) -> Result<Options, Stop> {
        let mut options = Options::default();
        if let Some(r) = &self.recipe {
            let cache = self.cache.as_deref();
            options.model = r.weights(cache, self.token.as_deref())?;
            options.name = Some(r.name.clone());
            options.lower = r.lower.clone();
            options.names = r.names();
            options.config = sensai_bundle::Overrides::new(r.overrides());
            if let Some(bytes) = r.module_bytes(cache)? {
                let module = Module::open(&bytes).map_err(|e| Stop::Failed(e.to_string()))?;
                let rules = module.rules.iter().copied();
                options.lower.get_or_insert_default().rules.extend(rules);
                options.module = Some(Arc::new(module));
            }
            for (flag, dir) in r.extras(cache, self.token.as_deref())? {
                apply(&mut options, &flag, Some(dir.display().to_string()))?;
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
        match (&self.origin, &self.fetched) {
            (Origin::Model(dir), _) => options.model = Path::new(dir).into(),
            (_, Some(dir)) => options.model = dir.clone(),
            _ => {}
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
