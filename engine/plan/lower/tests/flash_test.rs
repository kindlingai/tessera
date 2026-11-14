//! Attention on tiles against the CPU interpreter, on the GPU.

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

use gpukit::testing::{Noise, max_abs_diff};
use graph::DimName::{Context, HeadDim, Heads, KvHeads, Rows, Segments, Seqs};
use graph::{DType, Dim, Graph, Size, TensorType};
use interp::{Store, Tensor, run, run_kept};
use lower::{Device, lower};
use run::{Run, open};
use settings::read as settings;
use trace::{EntryKind, Mask, Model, Port, Role};

/// Causal attention over `heads` query heads of `width`.
fn attention(heads: u64, width: u64) -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 128), (Context, 256)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([
        dim(Heads, Size::Fixed(heads)),
        dim(HeadDim, Size::Fixed(width)),
    ]);
    let m = Model::new(dims);
    let port = |rows| Port {
        role: Role::Tokens,
        ty: TensorType::new([Seqs, rows, Heads, HeadDim], DType::F32),
    };
    let [q, k, v] = m.entry(
        EntryKind::Forward,
        [port(Rows), port(Context), port(Context)],
    );
    let out = m.attend(Mask::Causal, 0.125, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn flash_matches_cpu() {
    let mut draws = Noise(41);
    // Rows off, on and past the block, with older keys.
    let cases = [
        (1, 1, 1),
        (1, 1, 38),
        (2, 7, 7),
        (1, 33, 33),
        (2, 70, 93),
        (1, 128, 128),
    ];
    for (heads, width) in [(2u64, 16u64), (4, 64), (2, 128)] {
        let g = attention(heads, width);
        for (seqs, rows, keys) in cases {
            let (h, w) = (heads as usize, width as usize);
            let q = noise(vec![seqs, rows, h, w], &mut draws);
            let k = noise(vec![seqs, keys, h, w], &mut draws);
            let v = noise(vec![seqs, keys, h, w], &mut draws);
            let inputs = [q, k, v];
            let want = run(&g, &[], 0, &inputs).unwrap();
            let sizes = [
                (Seqs, seqs as u64),
                (Rows, rows as u64),
                (Context, keys as u64),
            ];
            let device = Device::guess(&settings()).favoring("flash");
            let plan = lower(&g, 0, &sizes, &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert_eq!(names, ["map", "flash"], "{seqs}x{rows}x{keys} at {width}");
            let mut runner = open(plan, &settings()).unwrap();
            let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
            let got = runner.run(&views).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(
                off < 1e-3,
                "{seqs}x{rows}x{keys} at {width}: off by {off:e}"
            );
        }
    }
}

/// Ragged rows over kept keys: `heads` query heads over `kv_heads`.
fn ragged(heads: u64, kv_heads: u64, width: u64) -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 128), (Context, 256)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([
        dim(Heads, Size::Fixed(heads)),
        dim(KvHeads, Size::Fixed(kv_heads)),
        dim(HeadDim, Size::Fixed(width)),
    ]);
    let m = Model::new(dims);
    let floats = |heads| TensorType::new([Rows, heads, HeadDim], DType::F32);
    let port = |role, ty| Port { role, ty };
    let ids = TensorType::new([Rows], DType::U32);
    let [q, k, v, slots, positions] = m.entry(
        EntryKind::Step,
        [
            port(Role::Tokens, floats(Heads)),
            port(Role::Tokens, floats(KvHeads)),
            port(Role::Tokens, floats(KvHeads)),
            port(Role::Slots, ids.clone()),
            port(Role::Positions, ids),
        ],
    );
    let kept = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let (keys, values) = (m.kept(kept.clone()), m.kept(kept));
    let by = [&slots, &positions];
    let (k, v) = (k.append(&keys, by), v.append(&values, by));
    let out = m.attend_rows(0.125, [&q, &k, &v], by);
    m.finish([(Role::Logits, &out)])
}

/// Rows for `runs` of (slot, first position, count).
fn rows(runs: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let slots = runs.iter().flat_map(|(s, _, n)| (0..*n).map(move |_| *s));
    let positions = runs
        .iter()
        .flat_map(|(_, p, n)| (0..*n).map(move |i| p + i));
    (slots.collect(), positions.collect())
}

#[test]
fn ragged_rows_match_cpu() {
    let mut draws = Noise(43);
    // A block over three slots, then rows past kept ones.
    let launches: [&[(u32, u32, u32)]; 2] = [
        &[(0, 0, 20), (1, 0, 5), (2, 0, 15)],
        &[(0, 20, 1), (1, 5, 1), (3, 0, 3), (2, 15, 1)],
    ];
    for (heads, kv_heads, width) in [(2u64, 2u64, 16u64), (4, 2, 64)] {
        let g = ragged(heads, kv_heads, width);
        for favored in ["flash", "attend"] {
            let (mut store, mut runner): (Store, Option<Box<dyn Run>>) = (Store::default(), None);
            for runs in launches {
                let (slots, positions) = rows(runs);
                let (n, h, kv, w) = (
                    slots.len(),
                    heads as usize,
                    kv_heads as usize,
                    width as usize,
                );
                let furthest = *positions.iter().max().unwrap() as u64 + 1;
                let inputs = [
                    noise(vec![n, h, w], &mut draws),
                    noise(vec![n, kv, w], &mut draws),
                    noise(vec![n, kv, w], &mut draws),
                    Tensor::u32(vec![n], slots),
                    Tensor::u32(vec![n], positions),
                ];
                let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                let sizes = [
                    (Seqs, 4),
                    (Rows, n as u64),
                    (Context, furthest),
                    (Segments, runs.len() as u64),
                ];
                let device = Device::guess(&settings()).favoring(favored);
                let plan = lower(&g, 0, &sizes, &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let kernel = format!("{favored}_rows");
                assert!(names.contains(&kernel.as_str()), "{names:?}");
                let r = match &mut runner {
                    Some(r) => {
                        r.replan(&plan).unwrap();
                        r
                    }
                    None => runner.insert(open(plan, &settings()).unwrap()),
                };
                let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
                let got = r.run(&views).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    off < 1e-3,
                    "{favored} on {runs:?} at {width}: off by {off:e}"
                );
            }
        }
    }
}
