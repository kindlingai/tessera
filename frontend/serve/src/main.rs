//! `tessera-serve --model <dir> [--port 8080] [--context 4096]`

mod server;
mod text;
#[cfg(test)]
mod text_test;

use engine::Engine;
use lower::{Pins, Recipe};
use server::{Job, Shared};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

const USAGE: &str =
    "usage: tessera-serve --model <dir> [--host <addr>] [--port <n>] [--context <rows>]";

struct Args {
    model: PathBuf,
    host: String,
    port: u16,
    context: usize,
}

fn args() -> Option<Args> {
    let mut args = Args {
        model: PathBuf::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        context: 4096,
    };
    let mut given = std::env::args().skip(1);
    while let Some(flag) = given.next() {
        let value = given.next()?;
        match flag.as_str() {
            "--model" => args.model = value.into(),
            "--host" => args.host = value,
            "--port" => args.port = value.parse().ok()?,
            "--context" => args.context = value.parse().ok()?,
            _ => return None,
        }
    }
    (!args.model.as_os_str().is_empty()).then_some(args)
}

/// Loads the model, then answers jobs until the queue closes.
fn engine_thread(
    args: &Args,
    jobs: mpsc::Receiver<Job>,
    ready: mpsc::Sender<Result<Arc<Shared>, String>>,
    queue: mpsc::Sender<Job>,
) {
    let start = Instant::now();
    let loaded = match hf::load(&args.model, 1, args.context as u64) {
        Ok(loaded) => loaded,
        Err(e) => return drop(ready.send(Err(e.to_string()))),
    };
    let settings = settings::read();
    let recipe = Recipe {
        pins: Pins::all(lower::form::BF16_FORM),
        ..Recipe::default()
    };
    let opened = Engine::open(loaded.graph, args.context, &recipe, &settings);
    let mut engine = match opened {
        Ok(engine) => engine,
        Err(e) => return drop(ready.send(Err(e.to_string()))),
    };
    if let Err(e) = engine.load(&loaded.weights) {
        return drop(ready.send(Err(e.to_string())));
    }
    let seconds = start.elapsed().as_secs_f64();
    eprintln!("{}: loaded in {seconds:.1} s", engine.device());
    let shared = Shared {
        tokenizer: loaded.tokenizer,
        template: loaded.template,
        stop: loaded.stop,
        jobs: Mutex::new(queue),
    };
    let _ = ready.send(Ok(Arc::new(shared)));
    for (request, events) in jobs {
        // A client that left closed its channel: the answer stops.
        let mut send = |e| {
            let _ = events.blocking_send(e);
        };
        if let Err(e) = engine.generate(&request, &mut send) {
            eprintln!("tessera-serve: {e}");
        }
    }
}

fn serve(shared: Arc<Shared>, host: String, port: u16) -> ExitCode {
    let served = tokio::runtime::Runtime::new().and_then(|runtime| {
        runtime.block_on(async {
            let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
            eprintln!("listening on {host}:{port}");
            axum::serve(listener, server::router(shared)).await
        })
    });
    match served {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-serve: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let Some(args) = args() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let (queue, jobs) = mpsc::channel();
    let (ready, loaded) = mpsc::channel();
    let (host, port) = (args.host.clone(), args.port);
    std::thread::spawn(move || engine_thread(&args, jobs, ready, queue));
    match loaded.recv() {
        Ok(Ok(shared)) => serve(shared, host, port),
        Ok(Err(e)) => {
            eprintln!("tessera-serve: {e}");
            ExitCode::FAILURE
        }
        Err(_) => {
            eprintln!("tessera-serve: the engine thread stopped before loading");
            ExitCode::FAILURE
        }
    }
}
