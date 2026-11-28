//! Parity and vectorization tests for kept-row scatter.

use tessera_graph::{DType, Dim, DimName::*, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::{Device, Draft, Pins, Source, T, Tree, lower};
use tessera_run::{
    open,
    testing::{host, runs},
};
use tessera_testkit::{
    Noise, fill,
    graphs::{floats, ids},
    max_abs_diff,
};
use tessera_trace::{Act, EntryKind, Model, Role};

fn nothing(_: &Tree, _: T) -> Option<Vec<Draft>> {
    None
}
tessera_lower::rule!("nothing", nothing);

#[test]
fn scatter_wide_parity() {
    let cases: [(u64, &'static dyn Stored, &'static dyn Stored, bool, f32); 6] = [
        (16, F32_FORM, F32_FORM, true, 1e-5),
        (17, F32_FORM, F32_FORM, false, 1e-5),
        (64, BF16_FORM, BF16_FORM, true, 2e-2),
        (64, F32_FORM, BF16_FORM, true, 2e-2),
        (64, BF16_FORM, F32_FORM, true, 2e-2),
        (64, F32_FORM, FP8_FORM, true, 1e-1),
    ];
    let layouts = [
        (
            "dense",
            2u64,
            3u64,
            [0u32, 1, 0, 1, 0, 1],
            [0u32, 0, 1, 1, 2, 2],
        ),
        ("paged16", 1, 6, [0; 6], [0, 1, 2, 3, 4, 5]),
    ];
    for (w, act, kf, x4, tol) in cases {
        for (lay, seqs, ctx, slots, pos) in layouts {
            let dims = vec![
                Dim::bounded(Seqs, seqs),
                Dim::bounded(Rows, 16),
                Dim::bounded(Context, ctx),
                Dim::fixed(Hidden, w),
            ];
            let m = Model::new(dims);
            let [x, sl, ps] = m.entry(
                EntryKind::Step,
                [floats([Rows, Hidden]), ids([Rows]), ids([Rows])],
            );
            let kept = m.kept(TensorType::new([Seqs, Context, Hidden], DType::F32));
            let g = m.finish([(Role::Logits, &x.act(Act::Silu).append(&kept, [&sl, &ps]))]);
            let sizes = [(Seqs, seqs), (Rows, 6), (Context, ctx)];
            let x = fill(&g, &g.entries[0].inputs[0].ty, &sizes, &mut Noise(41)).unwrap();
            let s = Tensor::u32(vec![6], slots.into());
            let p = Tensor::u32(vec![6], pos.into());
            let ins = [x, s, p];
            let want = run(&g, &[], 0, &ins).unwrap();
            let pins = Pins::all(act).activations(act).outputs(kf).kept(kf);
            let mut dev = Device::guess(&host()).pinned(pins);
            dev.kept = tessera_codegen::kept::named(lay).unwrap();
            let plan = lower(&g, 0, &sizes, &dev).unwrap();
            let step = plan.steps.iter().find(|s| s.kernel.name() == "scatter");
            if let Source::Made(body) = step.unwrap().kernel.source() {
                assert_eq!(body.entry.contains("_x4_"), x4, "{lay} w={w}");
            }
            let feeds = ins.each_ref().map(Tensor::view);
            for settings in runs() {
                let mut runner = open(plan.clone(), &settings).unwrap();
                let got = runner.run(&feeds).unwrap();
                let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
                assert!(diff < tol, "{lay} w={w} diff={diff:e}");
            }
        }
    }
}
