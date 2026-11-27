//! Verifies nested `Worker::spawn` inherits offline check scope.

use tessera_engine::check::{self, CheckConfig};
use tessera_engine::decoder::{Config, Swiglu, stepper};
use tessera_engine::{Backend, Engine, Norm, Pick, Recipe, Settings};
use tessera_tasks::Worker;

fn open_small(settings: &Settings) -> Result<Engine, tessera_tasks::Error> {
    let graph = stepper::<Norm, Swiglu>(&Config::qwen3_small());
    let mut engine = Engine::open(graph, 64, &Recipe::default(), settings)?;
    engine.warm(1, false)?;
    Ok(engine)
}

#[test]
fn forwards_check_scope() {
    let session = check::start("sm_121").unwrap();

    // Unrelated threads stay outside the active check session.
    let outside = std::thread::spawn(|| {
        assert!(check::current_scope().is_none());
        let engine = open_small(&Settings::default()).unwrap();
        assert!(!engine.device().contains("offline check"));
    });
    outside.join().unwrap();

    // Nested forwarding (`spawn_all` -> `Worker::spawn`) captures kernels.
    let scope = check::current_scope().expect("active check scope");
    let device = std::thread::spawn(move || {
        scope.run(|| {
            let gpu = Settings {
                backend: Pick::Gpu(Backend::Cuda),
                ..Settings::default()
            };
            let worker = Worker::spawn(move || open_small(&gpu)).unwrap();
            worker.run(|e| e.device().to_string()).unwrap()
        })
    })
    .join()
    .unwrap();
    assert!(device.contains("offline check"), "{device}");

    let cfg = CheckConfig {
        run_ptxas: false,
        ..CheckConfig::default()
    };
    let report = session.finish(&cfg).unwrap();
    assert!(!report.kernels.is_empty());

    // An unrelated thread alone must leave a fresh session empty.
    let quiet = check::start("sm_121").unwrap();
    std::thread::spawn(|| {
        let _ = open_small(&Settings::default()).unwrap();
    })
    .join()
    .unwrap();
    let empty = quiet.finish(&cfg).unwrap();
    assert!(empty.kernels.is_empty() && !empty.ok());
}
