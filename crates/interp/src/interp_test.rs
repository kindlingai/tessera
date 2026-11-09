use crate::*;
use trace::DimName::*;
use trace::{DType, Dim, EntryKind, Graph, Mask, Model, Port, Role, Size, TensorType};

fn model() -> Model {
    Model::new(vec![
        Dim {
            name: Seqs,
            size: Size::Bounded(2),
        },
        Dim {
            name: Rows,
            size: Size::Bounded(4),
        },
        Dim {
            name: Context,
            size: Size::Bounded(4),
        },
        Dim {
            name: Heads,
            size: Size::Fixed(2),
        },
        Dim {
            name: HeadDim,
            size: Size::Fixed(3),
        },
    ])
}

fn port(shape: impl Into<trace::Shape>) -> Port {
    Port {
        role: Role::Tokens,
        ty: TensorType::new(shape, DType::F32),
    }
}

fn ramp(shape: Vec<usize>) -> Tensor {
    let n = shape.iter().product::<usize>();
    Tensor::f32(shape, (0..n).map(|i| i as f32).collect())
}

fn views() -> Graph {
    let m = model();
    let [x] = m.entry(EntryKind::Forward, [port([Context, Heads, HeadDim])]);
    let sliced = x.slice(2, 1, 2);
    let swapped = x.permute(&[2, 0, 1]);
    let flat = x.reshape(vec![Context * Heads * HeadDim]);
    m.finish([
        (Role::Logits, &sliced),
        (Role::Logits, &swapped),
        (Role::Logits, &flat),
    ])
}

#[test]
fn views_move_data() {
    let out = run(&views(), &mut Cpu, &[], 0, &[ramp(vec![2, 2, 3])]).unwrap();
    assert_eq!(
        out[0].data,
        Data::F32(vec![1., 2., 4., 5., 7., 8., 10., 11.])
    );
    assert_eq!(out[1].shape, [3, 2, 2]);
    assert_eq!(out[1].floats().unwrap()[..4], [0., 3., 6., 9.]);
    assert_eq!(out[2].shape, [12]);
}

#[test]
fn launch_sizes_checked() {
    let g = views();
    let too_long = run(&g, &mut Cpu, &[], 0, &[ramp(vec![5, 2, 3])]);
    assert!(matches!(too_long, Err(Error::Mismatch(_))));
    let wrong_heads = run(&g, &mut Cpu, &[], 0, &[ramp(vec![2, 3, 3])]);
    assert!(matches!(wrong_heads, Err(Error::Mismatch(_))));
}

fn attention(mask: Mask) -> Graph {
    let m = model();
    let shape = [Seqs, Context, Heads, HeadDim];
    let [q, k, v] = m.entry(EntryKind::Forward, [port(shape), port(shape), port(shape)]);
    let out = m.attend(mask, 0.5, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

fn qkv() -> [Tensor; 3] {
    let wave = |seed: f32| {
        let v = (0..48).map(|i| (i as f32 * seed).sin()).collect();
        Tensor::f32(vec![2, 4, 2, 3], v)
    };
    [wave(0.3), wave(0.7), wave(1.1)]
}

#[test]
fn causal_first_row() {
    let ins = qkv();
    let out = run(&attention(Mask::Causal), &mut Cpu, &[], 0, &ins).unwrap();
    let (o, v) = (out[0].floats().unwrap(), ins[2].floats().unwrap());
    assert_eq!(o[..6], v[..6]);
    assert_eq!(o[24..30], v[24..30]);
    let both = run(&attention(Mask::Bidirectional), &mut Cpu, &[], 0, &ins).unwrap();
    assert_ne!(both[0].floats().unwrap()[..6], v[..6]);
}

#[test]
fn bind_checks_weights() {
    let m = model();
    let w = m.weight(
        &[trace::PathSeg::Index(0)],
        TensorType::new([Heads], DType::F32),
    );
    let [] = m.entry(EntryKind::Forward, []);
    let g = m.finish([(Role::Logits, &w)]);
    assert!(bind(&g, |_| None).is_err());
    assert!(bind(&g, |_| Some(Data::F32(vec![1.0; 3]))).is_err());
    let ok = bind(&g, |_| Some(Data::F32(vec![1.0; 2]))).unwrap();
    assert_eq!(ok[0].shape, [2]);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_matches_cpu() {
    if !gpu_ready() {
        return;
    }
    let cases = [
        (attention(Mask::Causal), qkv()),
        (decode(Mask::Causal), decode_inputs(2)),
    ];
    for (g, ins) in cases {
        let cpu = run(&g, &mut Cpu, &[], 0, &ins).unwrap();
        let mut k = Gpu::default();
        let gpu = run(&g, &mut k, &[], 0, &ins).unwrap();
        let (a, b) = (cpu[0].floats().unwrap(), gpu[0].floats().unwrap());
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-5, "attention differs by {worst}");
        assert_eq!(k.fallbacks, 0);
    }
}

fn decode(mask: Mask) -> Graph {
    let m = model();
    let keys = [Seqs, Context, Heads, HeadDim];
    let ports = [port([Seqs, Rows, Heads, HeadDim]), port(keys), port(keys)];
    let [q, k, v] = m.entry(EntryKind::Forward, ports);
    let out = m.attend(mask, 0.5, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

fn decode_inputs(rows: usize) -> [Tensor; 3] {
    let [q, k, v] = qkv();
    let q = Tensor::f32(
        vec![2, rows, 2, 3],
        q.floats().unwrap()[..12 * rows].to_vec(),
    );
    [q, k, v]
}

#[test]
fn decode_sees_past_keys() {
    let ins = decode_inputs(1);
    let causal = run(&decode(Mask::Causal), &mut Cpu, &[], 0, &ins).unwrap();
    let both = run(&decode(Mask::Bidirectional), &mut Cpu, &[], 0, &ins).unwrap();
    assert_eq!(causal, both);
    let short = Tensor::f32(vec![2, 4, 2, 3], vec![0.0; 47]);
    let [q, _, v] = qkv();
    assert!(run(&attention(Mask::Causal), &mut Cpu, &[], 0, &[q, short, v]).is_err());
}

#[test]
fn entries_run_alone() {
    let m = model();
    let [a] = m.entry(EntryKind::Forward, [port([Heads])]);
    let first = m.finish([(Role::Logits, &a.scale(2.0))]);
    let [b] = m.entry(EntryKind::Forward, [port([Heads])]);
    let g = m.finish([(Role::Logits, &b.scale(3.0))]);
    assert_eq!(first.entries.len(), 1);
    let x = Tensor::f32(vec![2], vec![1.0, 2.0]);
    let out = run(&g, &mut Cpu, &[], 0, std::slice::from_ref(&x)).unwrap();
    assert_eq!(out[0].data, Data::F32(vec![2.0, 4.0]));
    let out = run(&g, &mut Cpu, &[], 1, &[x]).unwrap();
    assert_eq!(out[0].data, Data::F32(vec![3.0, 6.0]));
}

#[derive(trace::Weights)]
struct Proj {
    #[shape(HeadDim, Heads)]
    w: trace::Linear,
}

/// A matmul summing in `acc`, as a pass derives it.
fn summing(acc: DType) -> Graph {
    let m = model();
    let p: Proj = trace::declare(&m);
    let [x] = m.entry(EntryKind::Forward, [port([Rows, HeadDim])]);
    let mut g = m.finish([(Role::Logits, &x.matmul(&p.w))]);
    let n = g.nodes.last_mut().unwrap().numerics.as_mut().unwrap();
    n.accumulate = acc;
    graph::verify(&g).unwrap();
    g
}

#[test]
fn accumulators_follow_numerics() {
    let ones = bind(&summing(DType::F32), |_| Some(Data::F32(vec![1.0; 6]))).unwrap();
    let x = Tensor::f32(vec![1, 3], vec![2048.0, 1.0, 1.0]);
    let sums = |acc| {
        let g = summing(acc);
        let out = run(&g, &mut Cpu, &ones, 0, std::slice::from_ref(&x)).unwrap();
        out[0].floats().unwrap().to_vec()
    };
    assert_eq!(sums(DType::F32), [2050.0, 2050.0]);
    assert_eq!(sums(DType::F16), [2048.0, 2048.0]);
    assert_eq!(sums(DType::Bf16), [2048.0, 2048.0]);
    #[cfg(feature = "gpu")]
    if gpu_ready() {
        let mut k = Gpu::default();
        let out = run(&summing(DType::F16), &mut k, &ones, 0, &[x]).unwrap();
        assert_eq!(out[0].floats().unwrap(), [2048.0, 2048.0]);
        assert_eq!(k.fallbacks, 1);
    }
}
