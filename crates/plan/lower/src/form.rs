//! How values are stored, and which weights are pinned.

use graph::PathSeg;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Form {
    #[default]
    F32,
    /// The top half of an f32.
    Bf16,
}

impl Form {
    pub fn bytes(self) -> usize {
        match self {
            Form::F32 => 4,
            Form::Bf16 => 2,
        }
    }

    /// Four-byte words that hold `elements`.
    pub fn words(self, elements: usize) -> usize {
        (elements * self.bytes()).div_ceil(4)
    }

    pub fn name(self) -> &'static str {
        match self {
            Form::F32 => "f32",
            Form::Bf16 => "bf16",
        }
    }

    /// A generated kernel's type for one element.
    pub fn ty(self) -> &'static str {
        match self {
            Form::F32 => "float",
            Form::Bf16 => "u16",
        }
    }

    /// Element `index` of buffer `name`, as a float.
    pub fn load(self, name: &str, index: &str) -> String {
        match self {
            Form::F32 => format!("{name}[{index}]"),
            Form::Bf16 => format!("f_bits((u32){name}[{index}] << 16)"),
        }
    }

    /// Whether `input` reads as `form`, when `weight` is this form.
    pub fn taken(self, input: usize, weight: usize, form: Form) -> bool {
        form == Form::F32 || (input == weight && form == self)
    }

    /// Bytes a matmul reads: the weight, then the rows.
    pub fn matmul_bytes(self, m: usize, n: usize, k: usize) -> Vec<f64> {
        vec![(self.bytes() * n * k) as f64, (4 * m * k) as f64]
    }

    /// `x` as this form holds it, back in f32.
    pub fn round(self, x: f32) -> f32 {
        match self {
            Form::F32 => x,
            Form::Bf16 => f32::from_bits(u32::from(bf16(x)) << 16),
        }
    }

    /// `data` packed as this form, in whole words.
    pub fn narrow(self, data: &[f32]) -> Vec<u8> {
        let mut bytes: Vec<u8> = match self {
            Form::F32 => data.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            Form::Bf16 => data.iter().flat_map(|x| bf16(*x).to_ne_bytes()).collect(),
        };
        bytes.resize(4 * self.words(data.len()), 0);
        bytes
    }
}

/// The nearest bf16, ties to even. NaN stays NaN.
fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return (bits >> 16) as u16 | 0x40;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// The form of each weight, as the user says.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pins {
    /// The form of every weight no pattern names.
    pub weights: Form,
    /// Patterns over weight paths, the last match winning.
    pub named: Vec<(String, Form)>,
}

impl Pins {
    pub fn all(weights: Form) -> Self {
        Self {
            weights,
            named: Vec::new(),
        }
    }

    /// Pins weights whose path fits `pattern`. `*` is any run.
    pub fn pin(mut self, pattern: &str, form: Form) -> Self {
        self.named.push((pattern.into(), form));
        self
    }

    /// The form of the weight at `path`.
    pub fn form(&self, path: &[PathSeg]) -> Form {
        let seg = |s: &PathSeg| match s {
            PathSeg::Field(f) => f.clone(),
            PathSeg::Index(i) => i.to_string(),
        };
        let name = path.iter().map(seg).collect::<Vec<_>>().join(".");
        let hit = self.named.iter().rev().find(|(p, _)| matches(p, &name));
        hit.map_or(self.weights, |(_, form)| *form)
    }
}

/// Whether `name` fits `pattern`. A star stands for anything.
fn matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or("");
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let mut parts = parts.peekable();
    while let Some(part) = parts.next() {
        let last = parts.peek().is_none();
        rest = match last {
            true => return rest.ends_with(part),
            false => match rest.find(part) {
                Some(at) => &rest[at + part.len()..],
                None => return false,
            },
        };
    }
    rest.is_empty()
}
