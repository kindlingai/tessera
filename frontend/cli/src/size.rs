//! `size`: memory a model needs, and memory this machine has.

use crate::source::{Origin, Source, Stop};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tessera_engine::Settings;
use tessera_engine::progress::{Event, Report, State};
use tessera_engine::sizer;
use tessera_recipe::Files;
use tessera_serve::Model;

pub fn run(origin: Option<Origin>, rest: &[String]) -> Result<(), Stop> {
    let boot = rest.iter().any(|a| a == "--boot");
    let mut flags = Vec::new();
    for arg in rest.iter().filter(|a| *a != "--boot") {
        flags.push(match arg.as_str() {
            "--ctx" => "--max-model-len".into(),
            _ => arg.clone(),
        });
    }
    let files = match boot {
        true => Files::Whole,
        false => Files::Headers,
    };
    let at = flags.iter().position(|f| f == "--manifest");
    let manifest = at.and_then(|at| flags.get(at + 1)).map(String::as_str);
    let source = Source::open_as(origin, crate::progress::stderr(), (files, manifest))?;
    let options = source.options(&flags)?;
    let settings = tessera_frontend_settings::read();
    let platform = sizer::platform(&settings);
    let [total, available] = sizer::host();
    let free = sizer::free(&settings);
    let mut out = json!({
        "model": options.name.clone().unwrap_or_else(|| options.model.display().to_string()),
        "ctx": options.max_model_len,
        "slots": options.slots,
        "device": {
            "backend": platform.backend,
            "name": platform.name,
            "integrated": platform.integrated,
            "total": platform.memory,
            "free": free,
        },
        "host": {"total": total, "available": available},
    });
    if boot {
        out["boot"] = self::boot(options, &settings);
    } else {
        let free = free.unwrap_or(platform.memory);
        plan(options, (free, platform.integrated), &mut out);
    }
    println!("{out}");
    Ok(())
}

/// Loads with runners that count bytes, then compares with `free`.
fn plan(options: tessera_serve::start::Options, (free, integrated): (u64, bool), out: &mut Value) {
    sizer::sizing();
    crate::task::load(options);
    let sizes = sizer::sized();
    // Integrated devices share host memory with load copies.
    let (free, copy) = match integrated {
        true => (free.min(sizer::host()[1]), sizes.largest),
        false => (free, 0),
    };
    let need = sizes.weights + sizes.kept + sizes.scratch + copy;
    out["weights"] = sizes.weights.into();
    out["kv"] = sizes.kept.into();
    out["scratch"] = sizes.scratch.into();
    out["load_copy"] = copy.into();
    out["need"] = need.into();
    out["fits"] = (need <= free).into();
    out["headroom"] = (free as i64 - need as i64).into();
}

/// A step's name, start and seconds once it ends.
type Phase = (Arc<str>, Instant, Option<f64>);

/// Each step by id, so in the order they start.
#[derive(Default)]
struct Phases(Mutex<BTreeMap<u64, Phase>>);

impl Report for Phases {
    fn report(&self, e: &Event) {
        // One file's download is part of a whole repository's.
        if e.name.contains('/') && !e.name.starts_with("download") {
            return;
        }
        let mut held = self.0.lock().expect("phases");
        match e.state {
            State::Start => {
                held.insert(e.id, (e.name.clone(), Instant::now(), None));
            }
            State::Done | State::Stopped => {
                if let Some(step) = held.get_mut(&e.id) {
                    step.2 = Some(step.1.elapsed().as_secs_f64());
                }
            }
            State::Move => {}
        }
    }
}

/// The least host available and device free memory seen.
struct Lows {
    host: AtomicU64,
    device: AtomicU64,
    done: AtomicBool,
}

impl Lows {
    fn sample(&self, settings: &Settings) {
        self.host.fetch_min(sizer::host()[1], Ordering::Relaxed);
        if let Some(free) = sizer::free(settings) {
            self.device.fetch_min(free, Ordering::Relaxed);
        }
    }
}

/// Loads, runs one step, and measures memory and time.
fn boot(mut options: tessera_serve::start::Options, settings: &Settings) -> Value {
    let phases = Arc::new(Phases::default());
    options.progress = options.progress.also(phases.clone());
    let before = (sizer::host()[1], sizer::free(settings));
    let lows = Arc::new(Lows {
        host: AtomicU64::new(before.0),
        device: AtomicU64::new(before.1.unwrap_or(u64::MAX)),
        done: AtomicBool::new(false),
    });
    let watch = {
        let (lows, settings) = (lows.clone(), settings.clone());
        std::thread::spawn(move || {
            while !lows.done.load(Ordering::Relaxed) {
                lows.sample(&settings);
                std::thread::sleep(Duration::from_millis(20));
            }
        })
    };
    let start = Instant::now();
    let slot = crate::task::load(options);
    let loaded = start.elapsed().as_secs_f64();
    let model = slot.get().expect("the model loaded");
    let stepping = Instant::now();
    let stepped = first_step(model);
    let step = stepping.elapsed().as_secs_f64();
    lows.sample(settings);
    lows.done.store(true, Ordering::Relaxed);
    let _ = watch.join();
    let held = phases.0.lock().expect("phases");
    let mut steps = Vec::new();
    for (name, _, seconds) in held.values() {
        steps.push(json!({"name": &**name, "seconds": seconds}));
    }
    let device = before
        .1
        .map(|b| b - lows.device.load(Ordering::Relaxed).min(b));
    json!({
        "phases": steps,
        "load_seconds": loaded,
        "first_step_seconds": step,
        "first_step": stepped,
        "peak_rss": peak_rss(),
        "peak_host_used": before.0 - lows.host.load(Ordering::Relaxed).min(before.0),
        "peak_device_used": device,
    })
}

/// The kind of model step that ran, or its error.
fn first_step(model: &Model) -> Value {
    let ran = if model.generator.is_some() {
        model
            .answer("Hello", Some(1), None, &mut |_| {})
            .map(|_| "generate")
    } else if let Some(image) = &model.image {
        let size = image.sizes().default;
        image.generate("a red circle", 1, 0, size).map(|_| "image")
    } else if let Some(embed) = &model.embed {
        embed.embed(&["Hello".into()]).map(|_| "embed")
    } else {
        Ok("none")
    };
    match ran {
        Ok(kind) => kind.into(),
        Err(e) => json!({"error": e.to_string()}),
    }
}

/// The most resident memory this process held, in bytes.
#[cfg(target_os = "linux")]
fn peak_rss() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let line = status.lines().find_map(|l| l.strip_prefix("VmHWM:"));
    let kib = line.and_then(|l| l.trim().trim_end_matches("kB").trim().parse::<u64>().ok());
    kib.unwrap_or(0) * 1024
}

/// The most resident memory this process held, in bytes.
#[cfg(target_os = "macos")]
fn peak_rss() -> u64 {
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32;
    }
    let mut usage = [0i64; 18];
    // SAFETY: `rusage` fits in 144 bytes.
    let failed = unsafe { getrusage(0, &mut usage) };
    // After the two times, `ru_maxrss` counts bytes on macOS.
    if failed != 0 { 0 } else { usage[4] as u64 }
}
