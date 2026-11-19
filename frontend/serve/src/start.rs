//! Loads one model on a thread and serves HTTP.

use crate::{Model, Slot};
use sensai_engine::wasm::Module;
use sensai_engine::{Batch, Fair, Pins, Recipe, Settings};
use sensai_tasks::api::{Embed, GenerateImage, Info, Tokens};
use sensai_tasks::klein::Klein;
use sensai_tasks::paint::Painter;
use sensai_tasks::{Generate, Job, Queued, Threaded, Worker};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;

/// The flags `Options::flag` and `Options::switch` take.
pub const FLAGS: &str = "[--host <addr>] [--port <n>] [--max-model-len <tokens>] [--slots <n>] \
[--served-model-name <name>] [--task generate|embed|image] [--image-size <px>] [--clip <dir>] \
[--max-connections <n>] [--reject-unknown-fields] [--guided]";

/// What the loaded model serves.
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
    pub max_connections: usize,
    /// The largest picture klein paints, in pixels a side.
    pub image_size: usize,
    /// The CLIP text tower for `paint`. `None` takes `model/clip`.
    pub clip: Option<PathBuf>,
    /// A recipe module that may build the model's graph.
    pub module: Option<Arc<Module>>,
    /// Lowering choices. Empty pins take the model's own.
    pub lower: Option<Recipe>,
    /// Weight-name rules, ahead of `TESSERA_NAMES`.
    pub names: Option<PathBuf>,
    /// This rank and its peers. Empty serves alone.
    pub ranks: crate::ranks::Ranks,
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
            max_connections: 64,
            image_size: 512,
            clip: None,
            module: None,
            lower: None,
            names: None,
            ranks: Default::default(),
        }
    }
}

impl Options {
    /// Takes a flag with no value. `false` if unknown.
    pub fn switch(&mut self, flag: &str) -> bool {
        match flag {
            "--reject-unknown-fields" => self.strict = true,
            "--guided" => self.guided = true,
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
            "--max-connections" => self.max_connections = value.parse().ok()?,
            "--served-model-name" => self.name = Some(value),
            "--task" if value == "generate" => self.task = Task::Generate,
            "--task" if value == "embed" => self.task = Task::Embed,
            "--task" if value == "image" => self.task = Task::Image,
            "--image-size" => self.image_size = value.parse().ok()?,
            "--clip" => self.clip = Some(value.into()),
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
        recipe.pins = pins.unwrap_or_else(|| Pins::all(sensai_engine::form::BF16_FORM));
    }
    recipe
}

/// Loads the generator, fills `slot`, then steps jobs.
fn engine_thread(o: &Options, settings: &Settings, info: Info, slot: &Slot) -> Result<(), String> {
    let start = Instant::now();
    let links = o.ranks.join(settings)?;
    let rules = match &settings.names {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| sensai_bundle::names::rules(&text).map_err(|e| e.to_string()))?,
        None => Vec::new(),
    };
    let (slots, context) = (o.slots as u64, o.max_model_len as u64);
    let loaded = sensai_bundle::load_named(&o.model, slots, context, &rules, o.guided)
        .map_err(|e| e.to_string())?;
    let recipe = lowering(o, loaded.pins);
    let at = (&recipe, settings);
    let mut engine = o.ranks.open(links, loaded.graph, o.max_model_len, at)?;
    engine.load(&loaded.weights).map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!("{}: loaded in {seconds:.1} s", engine.device());
    if o.ranks.follows() {
        return engine.follow().map_err(|e| e.to_string());
    }
    let guides = match (o.guided, engine.words()) {
        (false, _) => None,
        (true, None) => return Err("the model has no mask port".into()),
        (true, Some(words)) => Some(
            sensai_tasks::guides(&loaded.tokenizer, &loaded.stop, words)
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
    let queued = Queued {
        generate: generate.clone(),
        jobs: Mutex::new(queue),
    };
    let model = Model {
        info,
        tokens: generate.clone(),
        chat: Some(generate),
        generator: Some(Arc::new(queued)),
        embed: None,
        image: None,
        strict: o.strict,
    };
    let _ = slot.set(model);
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
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
    let mut settings = sensai_frontend_settings::read();
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
            strict,
        };
        let _ = slot.set(model);
    };
    let failed = match options.task {
        Task::Generate => engine_thread(&options, &settings, info, &slot),
        Task::Embed => Threaded::open(options.model, settings)
            .map(|e| {
                let e = Arc::new(e);
                set(info, e.clone(), Some(e as Arc<dyn Embed>), None);
            })
            .map_err(|e| e.to_string()),
        Task::Image if options.model.join("model_index.json").is_file() => {
            Worker::<Klein>::open(options.model, settings, options.image_size)
                .map(|p| painter(p, |t, p| set(info, t, None, Some(p))))
                .map_err(|e| e.to_string())
        }
        Task::Image => Worker::<Painter>::open(
            options.model,
            options.clip,
            options.module,
            settings,
            options.slots,
        )
        .map(|p| painter(p, |t, p| set(info, t, None, Some(p))))
        .map_err(|e| e.to_string()),
    };
    if let Err(e) = failed {
        eprintln!("sensai: {e}");
        std::process::exit(1);
    }
}
