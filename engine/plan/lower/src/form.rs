//! Pinned weight forms, and how kernels read a form.

use graph::{DType, PathSeg};

/// The form of each weight, as the user says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pins {
    /// The form of every weight no pattern names.
    pub weights: DType,
    /// Patterns over weight paths, the last match winning.
    pub named: Vec<(String, DType)>,
}

impl Default for Pins {
    fn default() -> Self {
        Self::all(DType::F32)
    }
}

impl Pins {
    pub fn all(weights: DType) -> Self {
        Self {
            weights,
            named: Vec::new(),
        }
    }

    /// Pins weights whose path fits `pattern`. `*` is any run.
    pub fn pin(mut self, pattern: &str, form: DType) -> Self {
        self.named.push((pattern.into(), form));
        self
    }

    /// The form of the weight at `path`.
    pub fn form(&self, path: &[PathSeg]) -> DType {
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

/// A stored form, as a generated kernel reads it.
pub trait Stored: Copy {
    /// Whether a float form needs widening before an f32 read.
    fn narrow(self) -> bool;
    fn name(self) -> &'static str;
    /// A generated kernel's type for one element.
    fn ty(self) -> &'static str;
    /// Element `index` of buffer `name`, as a float.
    fn load(self, name: &str, index: &str) -> String;
    /// Whether `input` reads as `form`, when `weight` is this form.
    fn taken(self, input: usize, weight: usize, form: DType) -> bool;
    /// Bytes a matmul reads: the weight, then the rows.
    fn matmul_bytes(self, m: usize, n: usize, k: usize) -> Vec<f64>;
}

impl Stored for DType {
    fn narrow(self) -> bool {
        self.is_float() && self != DType::F32
    }

    fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Bf16 => "bf16",
            DType::F16 => "f16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::Bool => "bool",
        }
    }

    fn ty(self) -> &'static str {
        match self {
            DType::Bf16 | DType::F16 => "u16",
            DType::U32 | DType::Bool => "u32",
            DType::I32 => "int",
            DType::F32 => "float",
        }
    }

    fn load(self, name: &str, index: &str) -> String {
        match self {
            DType::Bf16 => format!("f_bits((u32){name}[{index}] << 16)"),
            _ => format!("{name}[{index}]"),
        }
    }

    fn taken(self, input: usize, weight: usize, form: DType) -> bool {
        !form.narrow() || (input == weight && form == self)
    }

    fn matmul_bytes(self, m: usize, n: usize, k: usize) -> Vec<f64> {
        vec![(self.bytes() * n * k) as f64, (4 * m * k) as f64]
    }
}
