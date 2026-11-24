use super::graph::{SMALL, answer, module, steps_agree};
use tessera_bundle::Overrides;
use tessera_models::decoder::{Swiglu, stepper};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

#[test]
fn module_steps() {
    let want = stepper::<Norm, Swiglu>(&tessera_models::decoder::Config::qwen3_small());
    let got = module().graph(SMALL, 4, Some(16)).unwrap().unwrap();
    let noise = weights(&want, &mut Noise(1));
    for settings in runs() {
        steps_agree(&settings, [&want, &got], &noise);
    }
}

#[test]
fn module_generates() {
    let Some(dir) = tessera_testkit::reference("qwen3-taps") else {
        return eprintln!("no qwen3-taps reference here; skipped");
    };
    let (model, prompt) = (dir.join("model"), "The capital of France is");
    let want = answer(tessera_bundle::load(&model, 1, 64).unwrap(), prompt);
    let built = tessera_bundle::load_module(&model, [1, 64], &[], &Overrides::default(), &module());
    assert_eq!(answer(built.unwrap().unwrap(), prompt), want);
}
