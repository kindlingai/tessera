use tessera_cpu as _;
use tessera_graph::{Act, Dim, DimName::Rows};
use tessera_lower::{Device, Pick, Pins, Settings};
use tessera_testkit::{Noise, graphs, weights};

/// Compiled `cpu_tiled` matches `cpu_dense` across float forms.
#[test]
fn tiled_matches_forms() {
    let settings = Settings {
        backend: Pick::Cpu,
        ..tessera_run::testing::host()
    };
    let g = graphs::finished(&[Dim::bounded(Rows, 8)], [40, 37], Act::Silu);
    let sizes = [(Rows, 5)];
    let mut noise = Noise(19);
    let stored = weights(&g, &mut noise);
    let inputs: Vec<_> = g.entries[0]
        .inputs
        .iter()
        .map(|p| tessera_testkit::fill(&g, &p.ty, &sizes, &mut noise).unwrap())
        .collect();
    let feeds: Vec<_> = inputs.iter().map(|t| t.view()).collect();
    for name in ["f32", "bf16", "f16", "fp8"] {
        let form = tessera_codegen::form::named(name).unwrap();
        let base = Device::guess(&settings).pinned(Pins::all(form));
        let dense_plan =
            tessera_lower::lower(&g, 0, &sizes, &base.clone().favoring("cpu_dense")).unwrap();
        let tiled_plan = tessera_lower::lower(&g, 0, &sizes, &base.favoring("cpu_tiled")).unwrap();
        assert!(
            tiled_plan.steps.iter().any(|s| s.rule == "cpu_tiled"),
            "{name}"
        );
        let run = |plan| {
            let mut r = tessera_run::open(plan, &settings).unwrap();
            assert_eq!(r.compiled().0, r.compiled().1, "{name}: must compile");
            for (at, w) in stored.iter().enumerate() {
                r.load(at as u32, w.view()).unwrap();
            }
            r.run(&feeds).unwrap().remove(0)
        };
        let want = run(dense_plan);
        let got = run(tiled_plan);
        let off = got
            .floats()
            .unwrap()
            .iter()
            .zip(want.floats().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(off < 1e-4, "{name}: off by {off:e}");
    }
}
