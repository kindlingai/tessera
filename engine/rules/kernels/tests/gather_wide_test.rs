//! Parity tests for vectorized gather and kept_gather across forms.

use tessera_kernels as _;

use tessera_graph::DimName::{Context, Hidden, Rows, Seqs, Vocab};
use tessera_graph::{DType, Dim, TensorType};
use tessera_interp::{Store, Tensor, run, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored, named};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Model, Port, Role};

fn draw(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn gather_wide_matches_cpu() {
    let mut noise = Noise(101);
    let forms: [&'static dyn Stored; 5] = [
        F32_FORM,
        BF16_FORM,
        FP8_FORM,
        named("q8_0").unwrap(),
        named("q4_0").unwrap(),
    ];
    for &(vocab, width, rows) in &[(16usize, 64usize, 7usize), (12, 7, 5)] {
        let model = Model::new(vec![
            Dim::fixed(Vocab, vocab as u64),
            Dim::fixed(Hidden, width as u64),
            Dim::bounded(Rows, 32),
        ]);
        let [ids] = model.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows], DType::U32),
            }],
        );
        let table = model.weight(
            &[tessera_graph::PathSeg::Field("embed".into())],
            TensorType::new([Vocab, Hidden], DType::F32),
        );
        let out = table.gather(&ids);
        let g = model.finish([(Role::Logits, &out)]);
        let w_raw = draw(vec![vocab, width], &mut noise);
        // Include an out-of-bounds id to verify zero-fill.
        let id_vals: Vec<u32> = (0..rows)
            .map(|r| {
                if r == rows - 1 {
                    vocab as u32 + 2
                } else {
                    (r * 3) as u32 % vocab as u32
                }
            })
            .collect();
        let ids_t = Tensor::u32(vec![rows], id_vals);

        for &form in &forms {
            if width == 7 && form.name().starts_with('q') {
                continue;
            }
            let w_ref = Tensor::f32(w_raw.shape.clone(), form.roundtrip(w_raw.floats().unwrap()));
            let want = run(
                &g,
                std::slice::from_ref(&w_ref),
                0,
                std::slice::from_ref(&ids_t),
            )
            .unwrap();
            for settings in runs() {
                let device = Device::guess(&settings).pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
                let mut runner = open(plan, &settings).unwrap();
                runner.load(0, w_ref.view()).unwrap();
                let got = runner.run(&[ids_t.view()]).unwrap().remove(0);
                let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    diff < 1e-4,
                    "gather w={width} {}: diff={diff:e}",
                    form.name()
                );
            }
        }
    }
}

#[test]
fn kept_gather_wide_matches() {
    let mut noise = Noise(107);
    let layouts: [&'static dyn Layout; 2] = [kept::DENSE, &kept::PAGED16];
    let forms: [&'static dyn Stored; 3] = [F32_FORM, BF16_FORM, FP8_FORM];
    for &feat in &[64usize, 7usize] {
        let model = Model::new(vec![
            Dim::bounded(Seqs, 4),
            Dim::bounded(Rows, 16),
            Dim::bounded(Context, 32),
            Dim::fixed(Hidden, feat as u64),
        ]);
        let u32_row = TensorType::new([Rows], DType::U32);
        let [vals, slots, positions] = model.entry(
            EntryKind::Step,
            [
                Port {
                    role: Role::Tokens,
                    ty: TensorType::new([Rows, Hidden], DType::F32),
                },
                Port {
                    role: Role::Slots,
                    ty: u32_row.clone(),
                },
                Port {
                    role: Role::Positions,
                    ty: u32_row,
                },
            ],
        );
        let cache = model.kept(TensorType::new([Seqs, Context, Hidden], DType::F32));
        let out = vals.append(&cache, [&slots, &positions]).gather(&slots);
        let g = model.finish([(Role::Logits, &out)]);
        for &form in &forms {
            let tol = match form.name() {
                "f32" => 1e-4,
                "bf16" => 5e-2,
                _ => 0.3,
            };
            let pins = if form.name() == "f32" {
                Pins::default()
            } else {
                Pins::default().kept(form)
            };
            for &layout in &layouts {
                for settings in runs() {
                    let mut store = Store::default();
                    let slots_t = Tensor::u32(vec![6], vec![0, 0, 0, 1, 1, 2]);
                    let pos_t = Tensor::u32(vec![6], vec![0, 1, 2, 0, 1, 0]);
                    let v_t = draw(vec![6, feat], &mut noise);
                    let inputs = [v_t, slots_t, pos_t];
                    let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                    let mut device = Device::guess(&settings).pinned(pins.clone());
                    device.kept = layout;
                    let sizes = [(Seqs, 4), (Rows, 6), (Context, 3)];
                    let plan = lower(&g, 0, &sizes, &device).unwrap();
                    let table = plan.table;
                    let mut runner = open(plan, &settings).unwrap();
                    if let Some(slot) = table {
                        let raw: Vec<u8> = (0..slot.len as u32)
                            .rev()
                            .flat_map(u32::to_le_bytes)
                            .collect();
                        runner.held_mut().upload(slot, &raw).unwrap();
                    }
                    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
                    let got = runner.run(&views).unwrap().remove(0);
                    let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                    assert!(
                        diff < tol,
                        "kept_gather feat={feat} {} {}: diff={diff:e}",
                        form.name(),
                        layout.name()
                    );
                }
            }
        }
    }
}
