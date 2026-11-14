use crate::cpu_test::{close, eval, port as typed};
use crate::*;
use trace::DimName::*;
use trace::{DType, Dim, EntryKind, Graph, Mask, Model, Port, Role, TensorType};

fn model() -> Model {
    Model::new(vec![
        Dim::bounded(Seqs, 2),
        Dim::bounded(Rows, 4),
        Dim::bounded(Context, 4),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 3),
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
    let out = run(&views(), &[], 0, &[ramp(vec![2, 2, 3])]).unwrap();
    assert_eq!(out[0].floats().unwrap(), [1., 2., 4., 5., 7., 8., 10., 11.]);
    assert_eq!(out[1].shape, [3, 2, 2]);
    assert_eq!(out[1].floats().unwrap()[..4], [0., 3., 6., 9.]);
    assert_eq!(out[2].shape, [12]);
}

#[test]
fn launch_sizes_checked() {
    let g = views();
    let too_long = run(&g, &[], 0, &[ramp(vec![5, 2, 3])]);
    assert!(matches!(too_long, Err(Error::Mismatch(_))));
    let wrong_heads = run(&g, &[], 0, &[ramp(vec![2, 3, 3])]);
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
    let out = run(&attention(Mask::Causal), &[], 0, &ins).unwrap();
    let (o, v) = (out[0].floats().unwrap(), ins[2].floats().unwrap());
    assert_eq!(o[..6], v[..6]);
    assert_eq!(o[24..30], v[24..30]);
    let both = run(&attention(Mask::Bidirectional), &[], 0, &ins).unwrap();
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
    assert!(bind(&g, |_| Some(Tensor::f32(vec![3], vec![1.0; 3]))).is_err());
    let ok = bind(&g, |_| Some(Tensor::f32(vec![2], vec![1.0; 2]))).unwrap();
    assert_eq!(ok[0].shape, [2]);
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
    let causal = run(&decode(Mask::Causal), &[], 0, &ins).unwrap();
    let both = run(&decode(Mask::Bidirectional), &[], 0, &ins).unwrap();
    assert_eq!(causal, both);
    let short = Tensor::f32(vec![2, 4, 2, 3], vec![0.0; 47]);
    let [q, _, v] = qkv();
    assert!(run(&attention(Mask::Causal), &[], 0, &[q, short, v]).is_err());
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
    let out = run(&g, &[], 0, std::slice::from_ref(&x)).unwrap();
    assert_eq!(out[0].floats().unwrap(), [2.0, 4.0]);
    let out = run(&g, &[], 1, &[x]).unwrap();
    assert_eq!(out[0].floats().unwrap(), [3.0, 6.0]);
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
    let ones = bind(&summing(DType::F32), |_| {
        Some(Tensor::f32(vec![6], vec![1.0; 6]))
    })
    .unwrap();
    let x = Tensor::f32(vec![1, 3], vec![2048.0, 1.0, 1.0]);
    let sums = |acc| {
        let g = summing(acc);
        let out = run(&g, &ones, 0, std::slice::from_ref(&x)).unwrap();
        out[0].floats().unwrap().to_vec()
    };
    assert_eq!(sums(DType::F32), [2050.0, 2050.0]);
    assert_eq!(sums(DType::F16), [2048.0, 2048.0]);
    assert_eq!(sums(DType::Bf16), [2048.0, 2048.0]);
}

/// Concat, center and unit sum over `[Context, Heads, HeadDim]` rows.
fn row_ops() -> Graph {
    let m = model();
    let [x] = m.entry(EntryKind::Forward, [port([Context, Heads, HeadDim])]);
    let outs = [x.concat(&x), x.center(), x.unit_sum()];
    m.finish([
        (Role::Logits, &outs[0]),
        (Role::Logits, &outs[1]),
        (Role::Logits, &outs[2]),
    ])
}

#[test]
fn join_center_sum() {
    let out = run(&row_ops(), &[], 0, &[ramp(vec![1, 2, 3])]).unwrap();
    assert_eq!(
        out[0].floats().unwrap(),
        [0., 1., 2., 0., 1., 2., 3., 4., 5., 3., 4., 5.]
    );
    assert_eq!(out[1].floats().unwrap(), [-1., 0., 1., -1., 0., 1.]);
    assert_eq!(
        out[2].floats().unwrap(),
        [0., 1. / 3., 2. / 3., 0.25, 1. / 3., 5. / 12.]
    );
}

/// Keys and values `[Seqs, Context, heads, HeadDim]`, as one port.
fn kv_port(heads: trace::DimName) -> Port {
    typed([Seqs, Context, heads, HeadDim], DType::F32)
}

/// Zero queries average the keys they see in `v`.
fn window(rows: usize, v: &[f32]) -> Tensor {
    let q = Tensor::f32(vec![1, rows, 1, 2], vec![0.0; rows * 2]);
    let kv = Tensor::f32(vec![1, v.len() / 2, 1, 2], v.to_vec());
    let q_port = typed([Seqs, Rows, KvHeads, HeadDim], DType::F32);
    let ports = [q_port, kv_port(KvHeads), kv_port(KvHeads)];
    let out = eval(ports, &[q, kv.clone(), kv], None, |m, [q, k, v]| {
        [m.attend(Mask::Within(1), 1.0, [&q, &k, &v], None)]
    });
    out.unwrap().remove(0)
}

/// Two rows, 4 query heads over 2 key heads.
fn rows_over(positions: Vec<u32>) -> Result<Vec<Tensor>, Error> {
    let ln3 = 3f32.ln();
    let mut k = vec![0., ln3, 0., 0., ln3, 0., 0., 0., 99., 99., 99., 99.];
    k.extend([9.; 12]);
    let mut v = vec![4., -8., 0., 2., 8., 4., 4., 0., 99., 99., 99., 99.];
    v.extend([-1., 2., 3., -3., 50., 50., 50., 50., 60., 60., 60., 60.]);
    let ids = |v| Tensor::u32(vec![2], v);
    let inputs = [
        Tensor::f32(vec![2, 4, 2], [1., 0., 0., 1.].repeat(4)),
        Tensor::f32(vec![2, 3, 2, 2], k),
        Tensor::f32(vec![2, 3, 2, 2], v),
        ids(vec![1, 0]),
        ids(positions),
    ];
    let idx = || typed([Rows], DType::U32);
    let q = typed([Rows, Hidden, HeadDim], DType::F32);
    let ports = [q, kv_port(Heads), kv_port(Heads), idx(), idx()];
    eval(ports, &inputs, None, |m, [q, k, v, s, p]| {
        [m.attend_rows(1.0, [&q, &k, &v], [&s, &p])]
    })
}

#[test]
fn window_edges() {
    let v: Vec<f32> = (0..4).flat_map(|t| [(1 << t) as f32, -t as f32]).collect();
    // One key either side, cut at both ends.
    let want = [1.5, -0.5, 7. / 3., -1., 14. / 3., -2., 6., -2.5];
    close(&window(4, &v), &want);
    // A lone query sits at the last position.
    close(&window(1, &v), &want[6..]);
}

#[test]
fn rows_read_slots() {
    let out = rows_over(vec![0, 1]).unwrap();
    // Row 0 sees one key. Row 1 weighs two.
    let want = [
        -1., 2., -1., 2., 3., -3., 3., -3., 7., 1., 5., -5., 2., 1., 2., 1.,
    ];
    close(&out[0], &want);
    assert!(matches!(rows_over(vec![0, 3]), Err(Error::Mismatch(_))));
}
