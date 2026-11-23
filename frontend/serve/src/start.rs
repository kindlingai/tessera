//! Loads one model on a thread and serves HTTP.

use crate::{Model, Slot};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;
use tessera_bundle::Overrides;
use tessera_engine::wasm::Module;
use tessera_engine::{Batch, Drafter, Engine, Fair, Pins, Recipe, Settings, Speculate, Steps};
use tessera_tasks::api::{Embed, GenerateImage, Info, Score, Tokens};
use tessera_tasks::klein::Klein;
use tessera_tasks::paint::Painter;
use tessera_tasks::{Generate, Job, Queued, Threaded, Worker};

/// The flags for `Options::flag` and `Options::switch`.
pub const FLAGS: &str = "[--host <addr>] [--port <n>] [--max-model-len <tokens>] [--slots <n>] \
[--served-model-name <name>] [--task generate|embed|image] [--image-size <px>] [--clip <dir>] \
[--draft <dir>] [--max-connections <n>] [--reject-unknown-fields] [--guided] [--reuse-prefix] \
[--require-favored] [--manifest <name>]";

/// The loaded model's task.
#[derive(Clone, Copy, PartialEq)]
enum Task {
    Generate,
    Embed,
    /// Pictures: FLUX.2 klein from diffusers, else sprites.
    Image,
}

/// One image model as both traits the routes read.
fn painter<P: GenerateImage + Tokens + 'static>(
    p: P,
    set: impl FnOnce(Arc<dyn Tokens>, Arc<dyn GenerateImage>),
) {
    let p = Arc::new(p);
    set(p.clone(), p);
}

pub struct Options {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    /// Tokens one answer holds, prompt included.
    pub max_model_len: usize,
    /// Answers in flight at once.
    pub slots: usize,
    pub name: Option<String>,
    task: Task,
    /// Unknown request fields refuse the request.
    pub strict: bool,
    /// Answers may ask for a `response_format`.
    pub guided: bool,
    /// A prompt that continues an answer keeps its rows.
    pub reuse: bool,
    /// Fail lowering unless the favored rule covers a step.
    pub require_favored: bool,
    pub max_connections: usize,
    /// The largest picture klein paints, in pixels a side.
    pub image_size: usize,
    /// The CLIP text tower for `paint`. `None` takes `model/clip`.
    pub clip: Option<PathBuf>,
    /// DFlash drafter directory for speculative decoding.
    pub draft: Option<PathBuf>,
    /// A recipe module that may build the model's graph.
    pub module: Option<Arc<Module>>,
    /// Lowering choices. Empty pins take the model's own.
    pub lower: Option<Recipe>,
    /// Weight-name rules, ahead of `TESSERA_NAMES`.
    pub names: Option<PathBuf>,
    /// This rank and its peers. Empty serves alone.
    pub ranks: crate::ranks::Ranks,
    /// Config values the recipe sets, by file.
    pub config: Overrides,
    /// The exlr8 manifest that picks the widths.
    pub manifest: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            host: "127.0.0.1".into(),
            port: 8080,
            max_model_len: 4096,
            slots: 4,
            name: None,
            task: Task::Generate,
            strict: false,
            guided: false,
            reuse: false,
            require_favored: false,
            max_connections: 64,
            image_size: 512,
            clip: None,
            draft: None,
            module: None,
            lower: None,
            names: None,
            ranks: Default::default(),
            config: Overrides::default(),
            manifest: None,
        }
    }
}

impl Options {
    /// Takes a flag with no value. `false` if unknown.
    pub fn switch(&mut self, flag: &str) -> bool {
        match flag {
            "--reject-unknown-fields" => self.strict = true,
            "--guided" => self.guided = true,
            "--reuse-prefix" => self.reuse = true,
            "--require-favored" => self.require_favored = true,
            _ => return false,
        }
        true
    }

    /// Takes `flag value`. `None` if either is bad.
    pub fn flag(&mut self, flag: &str, value: String) -> Option<()> {
        match flag {
            "--host" => self.host = value,
            "--port" => self.port = value.parse().ok()?,
            "--max-model-len" => self.max_model_len = value.parse().ok()?,
            "--slots" => self.slots = value.parse().ok()?,
            "--manifest" => self.manifest = Some(value),
            "--max-connections" => self.max_connections = value.parse().ok()?,
            "--served-model-name" => self.name = Some(value),
            "--task" if value == "generate" => self.task = Task::Generate,
            "--task" if value == "embed" => self.task = Task::Embed,
            "--task" if value == "image" => self.task = Task::Image,
            "--image-size" => self.image_size = value.parse().ok()?,
            "--clip" => self.clip = Some(value.into()),
            "--draft" => self.draft = Some(value.into()),
            _ => return self.ranks.flag(flag, &value),
        }
        Some(())
    }
}

/// Loads the model on a thread. The slot fills after.
pub fn spawn(options: Options) -> Slot {
    let slot: Slot = Arc::new(OnceLock::new());
    let loading = slot.clone();
    std::thread::spawn(move || load(options, loading));
    slot
}

/// Loads the model, then answers requests until killed.
pub fn run(options: Options) -> Result<(), String> {
    let (host, port) = (options.host.clone(), options.port);
    let limits = crate::http::Limits {
        max_connections: options.max_connections,
        ..Default::default()
    };
    if options.ranks.follows() {
        load(options, Slot::default());
        return Ok(());
    }
    let slot = spawn(options);
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|e| e.to_string())?;
    eprintln!("listening on {host}:{port}");
    crate::http::run(listener, limits, move |request| {
        crate::route(&slot, request)
    });
    Ok(())
}

/// The lowering choices, with the model's pins by default.
fn lowering(options: &Options, pins: Option<Pins>) -> Recipe {
    let mut recipe = options.lower.clone().unwrap_or_default();
    if recipe.pins == Pins::default() {
        recipe.pins = pins.unwrap_or_else(|| Pins::all(tessera_engine::form::BF16_FORM));
    }
    if options.require_favored {
        recipe.require_favored = true;
    }
    recipe
}

/// The image pipelines this server runs, by `_class_name`.
const KLEIN: &str = "Flux2KleinPipeline";
const SPRITE: &str = "TesseraSmallPipeline";

/// The image pipeline's class: the recipe's, else `model_index.json`'s.
pub(crate) fn pipeline(options: &Options) -> Result<String, String> {
    const FILE: &str = "model_index.json";
    let index = options.model.join(FILE);
    let path = index.is_file().then_some(index.as_path());
    let class = (options.config).read(FILE, path, |k| k.need::<String>(&["_class_name"]));
    class.map_err(|e| format!("--task image needs a class in {FILE} or the recipe:\n{e}"))
}

/// Loads optional user weight-name rules from `settings`.
fn user_rules(settings: &Settings) -> Result<Vec<tessera_bundle::names::Rule>, String> {
    match &settings.names {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| tessera_bundle::names::rules(&text).map_err(|e| e.to_string())),
        None => Ok(Vec::new()),
    }
}

/// Loads a target and DFlash drafter, then answers requests.
fn draft_thread(
    o: &Options,
    draft_dir: &std::path::Path,
    settings: &Settings,
    info: Info,
    slot: &Slot,
) -> Result<(), String> {
    if o.guided || !o.ranks.alone() {
        return Err("--draft cannot combine with --guided or multi-rank yet".into());
    }
    let start = Instant::now();
    let rules = user_rules(settings)?;
    let at = [1u64, o.max_model_len as u64];
    let manifest = o.manifest.as_deref();
    let pair = tessera_bundle::dflash::load([&o.model, draft_dir], at, &rules, &o.config, manifest)
        .map_err(|e| e.to_string())?;
    let mut recipe = lowering(o, pair.target.pins);
    if recipe.config.is_none() {
        recipe.config = pair.target.config;
    }
    let mut engine = Engine::open(pair.target.graph, o.max_model_len, &recipe, settings)
        .map_err(|e| e.to_string())?;
    pair.target
        .weights
        .each(|idx, w| engine.load_weight(idx, w))
        .map_err(|e| e.to_string())?;
    let draft_recipe = lowering(o, Some(pair.pins));
    let mut drafter = Drafter::open(
        pair.graph,
        o.max_model_len,
        pair.block,
        pair.mask_token,
        &draft_recipe,
        settings,
    )
    .map_err(|e| e.to_string())?;
    drafter.load(&pair.weights).map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: loaded target + dflash (block {}) in {seconds:.1} s",
        engine.device(),
        pair.block
    );
    let generate = Arc::new(Generate {
        tokenizer: pair.target.tokenizer,
        template: pair.target.template,
        stop: pair.target.stop,
        guides: None,
    });
    let (queue, jobs) = mpsc::channel::<Job>();
    let queued = Arc::new(Queued {
        generate: generate.clone(),
        jobs: Mutex::new(queue),
    });
    let _ = slot.set(Model {
        info,
        tokens: generate.clone(),
        chat: Some(generate),
        generator: Some(queued),
        embed: None,
        image: None,
        score: None,
        strict: o.strict,
    });
    let mut spec = Speculate::new(&mut engine, &mut drafter).map_err(|e| e.to_string())?;
    while let Ok((request, mut sink)) = jobs.recv() {
        spec.generate(request, &mut sink)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Loads the generator, fills `slot`, then steps jobs.
fn engine_thread(o: &Options, settings: &Settings, info: Info, slot: &Slot) -> Result<(), String> {
    if let Some(draft_dir) = &o.draft {
        return draft_thread(o, draft_dir, settings, info, slot);
    }
    let start = Instant::now();
    let links = o.ranks.join(settings)?;
    let rules = user_rules(settings)?;
    let (slots, context) = (o.slots as u64, o.max_model_len as u64);
    // Ranks keep one entry, as they did before scoring.
    let steps = match (o.guided, o.ranks.alone()) {
        (true, _) => Steps::Masked,
        (false, true) => Steps::Scored,
        (false, false) => Steps::Plain,
    };
    let at = [slots, context];
    let manifest = o.manifest.as_deref();
    let loaded = tessera_bundle::load_named(&o.model, at, &rules, steps, &o.config, manifest)
        .map_err(|e| e.to_string())?;
    let scores = loaded.graph.entries.len() > 1;
    let mut recipe = lowering(o, loaded.pins);
    if recipe.config.is_none() {
        recipe.config = loaded.config;
    }
    let at = (&recipe, settings);
    let count = loaded.graph.weights.len() as u32;
    let mut engine = o.ranks.open(links, loaded.graph, o.max_model_len, at)?;
    let shares: Vec<_> = (0..count).map(|at| engine.share(at)).collect();
    let shares = |at: u32| shares[at as usize];
    let loading = (loaded.weights).each_share(shares, |at, w| engine.load_weight(at, w));
    loading.map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!("{}: loaded in {seconds:.1} s", engine.device());
    if o.ranks.follows() {
        return engine.follow().map_err(|e| e.to_string());
    }
    let guides = match (o.guided, engine.words()) {
        (false, _) => None,
        (true, None) => return Err("the model has no mask port".into()),
        (true, Some(words)) => Some(
            tessera_tasks::guides(&loaded.tokenizer, &loaded.stop, words)
                .map_err(|e| e.to_string())?,
        ),
    };
    let generate = Arc::new(Generate {
        tokenizer: loaded.tokenizer,
        template: loaded.template,
        stop: loaded.stop,
        guides,
    });
    let (queue, jobs) = mpsc::channel::<Job>();
    let queued = Arc::new(Queued {
        generate: generate.clone(),
        jobs: Mutex::new(queue),
    });
    let model = Model {
        info,
        tokens: generate.clone(),
        chat: Some(generate),
        generator: Some(queued.clone()),
        embed: None,
        image: None,
        score: scores.then_some(queued as Arc<dyn Score>),
        strict: o.strict,
    };
    let _ = slot.set(model);
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
    if o.reuse {
        batch.reuse();
    }
    loop {
        // Takes every job that has arrived. Waits only when idle.
        let job = match batch.busy() {
            true => jobs.try_recv().ok(),
            false => match jobs.recv() {
                Ok(job) => Some(job),
                Err(_) => return Ok(()),
            },
        };
        if let Some((request, sink)) = job {
            batch.submit(request, sink);
            continue;
        }
        batch.step().map_err(|e| e.to_string())?;
    }
}

/// Loads the model into `slot`. A failure ends the process.
fn load(options: Options, slot: Slot) {
    let mut settings = tessera_frontend_settings::read();
    settings.names = options.names.clone().or(settings.names);
    let info = Info {
        name: options
            .name
            .clone()
            .unwrap_or_else(|| options.model.display().to_string()),
        context: options.max_model_len,
    };
    let strict = options.strict;
    // A task the worker thread serves, by its traits.
    let set = |info, tokens: Arc<dyn Tokens>, embed, image| {
        let chat = None;
        let generator = None;
        let model = Model {
            info,
            tokens,
            chat,
            generator,
            embed,
            image,
            score: None,
            strict,
        };
        let _ = slot.set(model);
    };
    let failed = match options.task {
        Task::Generate => engine_thread(&options, &settings, info, &slot),
        Task::Embed => Threaded::open(options.model, settings, options.config)
            .map(|e| {
                let e = Arc::new(e);
                set(info, e.clone(), Some(e as Arc<dyn Embed>), None);
            })
            .map_err(|e| e.to_string()),
        Task::Image => match pipeline(&options) {
            Ok(class) if class == KLEIN => {
                Worker::<Klein>::open(options.model, settings, options.image_size, options.config)
                    .map(|p| painter(p, |t, p| set(info, t, None, Some(p))))
                    .map_err(|e| e.to_string())
            }
            Ok(class) if class == SPRITE => Worker::<Painter>::open(
                options.model,
                options.clip,
                options.module,
                settings,
                options.slots,
                options.config,
            )
            .map(|p| painter(p, |t, p| set(info, t, None, Some(p))))
            .map_err(|e| e.to_string()),
            Ok(class) => Err(format!("{class} is not an image pipeline this server runs")),
            Err(e) => Err(e),
        },
    };
    if let Err(e) = failed {
        eprintln!("tesserax: {e}");
        std::process::exit(1);
    }
}
