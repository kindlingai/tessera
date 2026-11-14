use crate::*;
use trace::DType::{F32, U32};
use trace::DimName::*;
use trace::{Act, DType, Dim, EntryKind, Model, Pairs, Port, Role, TensorType};

pub(crate) type Traced = trace::Tensor;

pub(crate) fn model() -> Model {
    Model::new(vec![
        Dim::bounded(Seqs, 2),
        Dim::bounded(Rows, 4),
        Dim::bounded(Context, 4),
        Dim::fixed(Heads, 2),
        Dim::fixed(KvHeads, 1),
        Dim::fixed(HeadDim, 2),
        Dim::fixed(Hidden, 4),
        Dim::fixed(Ffn, 8),
        Dim::fixed(Embedding, 128),
        Dim::fixed(TopK, 3),
    ])
}

pub(crate) fn port(shape: impl Into<trace::Shape>, dtype: DType) -> Port {
    Port {
        role: Role::Tokens,
        ty: TensorType::new(shape, dtype),
    }
}

/// Traces `f` over `ports`, then runs it on `inputs`.
pub(crate) fn eval<const I: usize, const N: usize>(
    ports: [Port; I],
    inputs: &[Tensor],
    weight: Option<&Tensor>,
    f: impl FnOnce(&Model, [Traced; I]) -> [Traced; N],
) -> Result<Vec<Tensor>, Error> {
    let m = model();
    let outs = f(&m, m.entry(EntryKind::Forward, ports));
    let g = m.finish(outs.each_ref().map(|t| (Role::Logits, t)));
    let weights = bind(&g, |_| weight.cloned())?;
    run(&g, &weights, 0, inputs)
}

/// One float input shaped `[Rows, width]`, through `f`.
fn rows<const N: usize>(
    width: trace::DimName,
    x: Vec<f32>,
    f: impl FnOnce(Traced) -> [Traced; N],
) -> Vec<Tensor> {
    let n = model().size(width) as usize;
    let x = Tensor::f32(vec![x.len() / n, n], x);
    eval([port([Rows, width], F32)], &[x], None, |_, [x]| f(x)).unwrap()
}

pub(crate) fn close(got: &Tensor, want: &[f32]) {
    let got = got.floats().unwrap();
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "{got:?} vs {want:?}"
        );
    }
}

/// Rows at position 0, then at 3: angles 3, 1.5.
fn turned(x: &[f32], pairs: Pairs) -> Vec<f32> {
    let (s0, c0) = 3f32.sin_cos();
    let (s1, c1) = 1.5f32.sin_cos();
    let mut want = x[..8].to_vec();
    for r in x[8..].chunks(4) {
        let &[a, b, c, d] = r else { unreachable!() };
        want.extend(match pairs {
            Pairs::Adjacent => [
                a * c0 - b * s0,
                a * s0 + b * c0,
                c * c1 - d * s1,
                c * s1 + d * c1,
            ],
            Pairs::Halves => [
                a * c0 - c * s0,
                b * c1 - d * s1,
                a * s0 + c * c0,
                b * s1 + d * c1,
            ],
        });
    }
    want
}

/// Signs from the parity of `i & j`, then scaled.
fn sylvester(x: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (x.len() as f32).sqrt();
    let mut out = Vec::new();
    for i in 0..x.len() {
        let mut sum = 0.0;
        for (j, v) in x.iter().enumerate() {
            sum += if (i & j).count_ones() % 2 == 0 {
                *v
            } else {
                -*v
            };
        }
        out.push(sum * scale);
    }
    out
}

#[derive(trace::Weights)]
struct Gain {
    #[shape(Hidden)]
    w: trace::Norm,
}

#[test]
fn rope_both_pairs() {
    let x: Vec<f32> = (0..16).map(|i| i as f32 - 5.5).collect();
    let ports = [port([Rows, Heads, Hidden], F32), port([Rows], U32)];
    let at = Tensor::u32(vec![2], vec![0, 3]);
    for pairs in [Pairs::Adjacent, Pairs::Halves] {
        let inputs = [Tensor::f32(vec![2, 2, 4], x.clone()), at.clone()];
        let out = eval(ports.clone(), &inputs, None, |_, [x, at]| {
            [x.rope(&at, 4.0, pairs)]
        });
        close(&out.unwrap()[0], &turned(&x, pairs));
    }
}

#[test]
fn rms_divides_rows() {
    let x = Tensor::f32(vec![2, 4], vec![1., -3., 5., -1., 2., 0., 0., -2.]);
    let w = Tensor::f32(vec![4], vec![1., 2., -1., 0.5]);
    let out = eval([port([Rows, Hidden], F32)], &[x], Some(&w), |m, [x]| {
        let g: Gain = trace::declare(m);
        [x.rms(7.0), x.rms_norm(&g.w, 7.0)]
    });
    let out = out.unwrap();
    // Mean squares 9 and 2, plus 7: roots 4, 3.
    close(
        &out[0],
        &[0.25, -0.75, 1.25, -0.25, 2. / 3., 0., 0., -2. / 3.],
    );
    close(
        &out[1],
        &[0.25, -1.5, -1.25, -0.125, 2. / 3., 0., 0., -1. / 3.],
    );
}

#[test]
fn hadamard_blocks() {
    let x = vec![1., 2., 3., 4., -1., 0., 2., 5.];
    // Each block of four through the 4x4 matrix, halved.
    let out = rows(Ffn, x, |x| [x.hadamard(4)]);
    close(&out[0], &[5., -1., -2., 0., 3., -2., -4., 1.]);
    let x: Vec<f32> = (0..128).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
    let out = rows(Embedding, x.clone(), |x| [x.hadamard(128)]);
    close(&out[0], &sylvester(&x));
}

#[test]
fn softmax_rows() {
    let (ln2, ln3, ln4) = (2f32.ln(), 3f32.ln(), 4f32.ln());
    let x = vec![0., ln2, ln3, ln4, 500., 500., f32::NEG_INFINITY, 500.];
    // Exponents 1 to 4 over 10. Masked gives zero.
    let out = rows(Hidden, x, |x| [x.softmax()]);
    close(
        &out[0],
        &[0.1, 0.2, 0.3, 0.4, 1. / 3., 1. / 3., 0., 1. / 3.],
    );
}

#[test]
fn act_values() {
    let acts = [Act::Relu, Act::Sigmoid, Act::Silu, Act::GeluTanh];
    let out = rows(Hidden, vec![-2., 0., 1., 3.], |x| acts.map(|a| x.act(a)));
    close(&out[0], &[0., 0., 1., 3.]);
    close(&out[1], &[0.119_202_92, 0.5, 0.731_058_6, 0.952_574_1]);
    close(&out[2], &[-0.238_405_84, 0., 0.731_058_6, 2.857_722_4]);
    // The tanh form. Exact erf gives 0.841_345 at 1.
    close(&out[3], &[-0.045_402_3, 0., 0.841_192, 2.996_362_6]);
}

#[test]
fn gather_rows() {
    let table = Tensor::f32(vec![4, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let pick = |ids: Vec<u32>| {
        let ports = [port([Context, HeadDim], F32), port([Rows], U32)];
        let inputs = [table.clone(), Tensor::u32(vec![ids.len()], ids)];
        eval(ports, &inputs, None, |_, [t, ids]| [t.gather(&ids)])
    };
    // Repeats, and both ends of the table.
    close(&pick(vec![3, 0, 3]).unwrap()[0], &[7., 8., 1., 2., 7., 8.]);
    assert!(matches!(pick(vec![4]), Err(Error::Mismatch(_))));
}

#[test]
fn mean_each_axis() {
    // Element `[a, b, c]` holds `6a + 2b + c`.
    let x = Tensor::f32(vec![2, 3, 2], (0..12).map(|i| i as f32).collect());
    let ports = [port([Seqs, Rows, HeadDim], F32)];
    let out = eval(ports, &[x], None, |_, [x]| [0, 1, 2].map(|a| x.mean(a))).unwrap();
    close(&out[0], &[3., 4., 5., 6., 7., 8.]);
    close(&out[1], &[2., 3., 8., 9.]);
    close(&out[2], &[0.5, 2.5, 4.5, 6.5, 8.5, 10.5]);
}
