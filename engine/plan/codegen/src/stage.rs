//! Stages over a row of finished sums, for host code.

/// One stage a row call applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    /// Adds the operand row.
    Add,
    /// Multiplies by the operand row.
    Mul,
    Scale,
    GeluTanh,
    Silu,
    Sigmoid,
    Relu,
    Softplus,
    Exp,
    Tanh,
    SignedSqrt,
}

impl Row {
    const ALL: [Row; 11] = [
        Row::Add,
        Row::Mul,
        Row::Scale,
        Row::GeluTanh,
        Row::Silu,
        Row::Sigmoid,
        Row::Relu,
        Row::Softplus,
        Row::Exp,
        Row::Tanh,
        Row::SignedSqrt,
    ];

    pub fn code(self) -> u32 {
        Self::ALL.iter().position(|r| *r == self).expect("listed") as u32
    }

    pub fn from_code(code: u32) -> Option<Row> {
        Self::ALL.get(code as usize).copied()
    }

    /// Whether the stage reads an operand row.
    pub fn reads(self) -> bool {
        matches!(self, Row::Add | Row::Mul)
    }
}

/// Past this `tanh` rounds to one.
pub const TANH_CLAMP: f32 = 7.905_311;
/// `tanh`'s numerator over `x^2`, lowest first, times `x`.
pub const TANH_P: [f32; 7] = [
    4.893_524_6e-3,
    6.372_619_3e-4,
    1.485_722_4e-5,
    5.122_297e-8,
    -8.604_672e-11,
    2.000_188e-13,
    -2.760_768_5e-16,
];
/// `tanh`'s denominator over `x^2`, lowest first.
pub const TANH_Q: [f32; 4] = [4.893_525e-3, 2.268_434_6e-3, 1.185_347e-4, 1.198_258_4e-6];

/// `tanh(x)` as a rational polynomial, within 3e-7 of f32 `tanh`.
pub fn tanh(x: f32) -> f32 {
    let x = x.clamp(-TANH_CLAMP, TANH_CLAMP);
    let x2 = x * x;
    let horner = |c: &[f32]| {
        c.iter()
            .rev()
            .skip(1)
            .fold(c[c.len() - 1], |a, k| x2 * a + k)
    };
    x * horner(&TANH_P) / horner(&TANH_Q)
}

/// GELU with the tanh estimate, in the graph's spelling.
fn gelu(v: f32) -> f32 {
    let inner = v + 0.044_715 * (v * v * v);
    let t = 1.0 + tanh((0.797_884_6 * inner).clamp(-8.0, 8.0));
    0.5 * v * t
}

/// `out` through `stage`, with operand row `b` or scale `s`.
pub fn apply(stage: Row, out: &mut [f32], b: &[f32], s: f32) {
    let each = |out: &mut [f32], f: fn(f32) -> f32| out.iter_mut().for_each(|o| *o = f(*o));
    match stage {
        Row::Add => out.iter_mut().zip(b).for_each(|(o, x)| *o += x),
        Row::Mul => out.iter_mut().zip(b).for_each(|(o, x)| *o *= x),
        Row::Scale => out.iter_mut().for_each(|o| *o *= s),
        Row::GeluTanh => each(out, gelu),
        Row::Silu => each(out, |v| v / (1.0 + (-v).exp())),
        Row::Sigmoid => each(out, |v| 1.0 / (1.0 + (-v).exp())),
        Row::Relu => each(out, |v| v.max(0.0)),
        Row::Softplus => each(out, |v| v.max(0.0) + (-v.abs()).exp().ln_1p()),
        Row::Exp => each(out, f32::exp),
        Row::Tanh => each(out, tanh),
        Row::SignedSqrt => each(out, |v| {
            if v > 0.0 {
                v.max(1e-6).sqrt()
            } else if v < 0.0 {
                -(-v).max(1e-6).sqrt()
            } else {
                0.0
            }
        }),
    }
}
