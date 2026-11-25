//! Where the model comes from, and the options it gives.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tessera_engine::wasm::{Module, Trace};
use tessera_recipe::Recipe;
use tessera_serve::start::Options;

/// Why a run stops early.
#[derive(Debug)]
pub enum Stop {
    Usage,
    /// The task list, after a message if given.
    Tasks(Option<String>),
    Failed(String),
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::Usage | Stop::Tasks(None) => f.write_str("bad usage"),
            Stop::Tasks(Some(text)) | Stop::Failed(text) => f.write_str(text),
        }
    }
}

impl std::error::Error for Stop {}

impl From<tessera_recipe::Error> for Stop {
    fn from(e: tessera_recipe::Error) -> Self {
        Stop::Failed(e.to_string())
    }
}

/// Where the model comes from.
pub enum Origin {
    Recipe(String),
    Model(String),
    /// `owner/name` or `owner/name@revision` on Hugging Face.
    Hub(String),
    /// A recipe built into tesserax, by catalog name.
    Builtin(&'static str),
}

macro_rules! catalog {
    ($($name:literal),* $(,)?) => {
        [$(($name, include_str!(concat!("../recipe/", $name, ".toml")))),*]
    };
}

/// Built-in recipes by name. The first is the default.
pub const CATALOG: [(&str, &str); 4] = catalog![
    "gemma-4-e4b",
    "qwen3-0.6b",
    "flux2-klein",
    "embeddinggemma-2",
];

/// Catalog recipe `name`.
pub fn builtin(name: &str) -> Option<Result<Recipe, Stop>> {
    let (_, text) = CATALOG.iter().find(|(n, _)| *n == name)?;
    let path = PathBuf::from(format!("builtin/{name}.toml"));
    Some(Recipe::parse(text, &path).map_err(Stop::from))
}

/// A recipe's kind: its serve task, or text generation.
pub fn kind(r: &Recipe) -> &str {
    r.serve
        .get("task")
        .and_then(|v| v.as_str())
        .unwrap_or("generate")
}

/// Every catalog recipe that reads.
pub fn catalog() -> Vec<Recipe> {
    let read = |(name, _): &(&str, &str)| builtin(name)?.ok();
    CATALOG.iter().filter_map(read).collect()
}

/// The default recipe's name and Hugging Face repository.
pub fn default_model() -> String {
    let Some(Ok(recipe)) = builtin(CATALOG[0].0) else {
        return CATALOG[0].0.into();
    };
    let hf = recipe.weights.hf.unwrap_or_default();
    let repo = hf.split('@').next().unwrap_or_default();
    format!("{} from {repo}", recipe.name)
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
            // A directory or source wins over a catalog name.
            "--recipe" => match CATALOG.iter().find(|(n, _)| *n == value) {
                Some((name, _)) if !Path::new(&value).exists() => Origin::Builtin(name),
                _ => Origin::Recipe(value),
            },
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
    dump_sites: Option<PathBuf>,
}

impl Source {
    /// Opens the recipe, if the origin is one.
    pub fn open(origin: Option<Origin>) -> Result<Source, Stop> {
        let origin = origin.ok_or(Stop::Usage)?;
        let settings = tessera_frontend_settings::read();
        let (cache, token, dump_sites) = (settings.recipes, settings.hf_token, settings.dump_sites);
        let mut fetched = None;
        let recipe = match &origin {
            Origin::Recipe(src) => Some(Recipe::open(src, cache.as_deref())?),
            Origin::Builtin(name) => Some(builtin(name).ok_or(Stop::Usage)??),
            Origin::Model(_) => None,
            Origin::Hub(spec) => {
                let dir = cache.as_deref().ok_or(tessera_recipe::Error::NoCache)?;
                fetched = Some(tessera_recipe::fetch_hf(spec, dir, token.as_deref())?);
                None
            }
        };
        Ok(Source {
            origin,
            recipe,
            cache,
            token,
            fetched,
            dump_sites,
        })
    }

    /// Opens `origin`, else the first catalog recipe of `of`.
    pub fn open_or_kind(origin: Option<Origin>, of: Option<&str>) -> Result<Source, Stop> {
        if origin.is_some() {
            return Self::open(origin);
        }
        for (name, _) in CATALOG {
            let Some(Ok(recipe)) = builtin(name) else {
                continue;
            };
            if of.is_none_or(|k| k == kind(&recipe)) {
                return Self::open(Some(Origin::Builtin(name)));
            }
        }
        let of = of.unwrap_or_default();
        let none = format!("no built-in model does {of}: give --recipe or --model");
        Err(Stop::Failed(none))
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
            options.config = tessera_bundle::Overrides::new(r.overrides());
            if let Some(bytes) = r.module_bytes(cache)? {
                let trace = Trace {
                    dump: self.dump_sites.clone(),
                    favor: r.lower.as_ref().and_then(|l| l.favor.clone()),
                };
                let module =
                    Module::open_with(&bytes, trace).map_err(|e| Stop::Failed(e.to_string()))?;
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
