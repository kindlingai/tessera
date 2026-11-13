//! Symbolic dimensions, shape expressions and their normal form.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul};

/// Every shape is written in these names.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimName {
    Seqs,
    Rows,
    Context,
    /// Slots one launch's rows span. A size with no axis.
    Segments,
    /// Length of a learned position table.
    Positions,
    Hidden,
    Heads,
    KvHeads,
    HeadDim,
    /// A head's width outside its rotary part.
    Nope,
    /// A head's rotary width.
    Rope,
    /// Width of the low-rank query.
    QLatent,
    /// Width of the low-rank keys and values.
    Latent,
    Ffn,
    /// A dense layer's MLP width. Experts use `Ffn`.
    DenseFfn,
    Vocab,
    Experts,
    TopK,
    IndexHeads,
    IndexDim,
    /// Keys an indexer keeps for each query.
    IndexKeep,
    Layers,
    DenseLayers,
    MoeLayers,
    /// Layers that run their own indexer.
    IndexLayers,
}

/// How a dimension gets its size.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Fixed(u64),
    /// Varies per launch, up to this.
    Bounded(u64),
}

impl Size {
    pub fn max(self) -> u64 {
        match self {
            Size::Fixed(n) | Size::Bounded(n) => n,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Dim {
    pub name: DimName,
    pub size: Size,
}

impl Dim {
    pub const fn fixed(name: DimName, n: u64) -> Self {
        Self {
            name,
            size: Size::Fixed(n),
        }
    }

    /// Varies per launch, up to `max`.
    pub const fn bounded(name: DimName, max: u64) -> Self {
        Self {
            name,
            size: Size::Bounded(max),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum DimExpr {
    Dim(DimName),
    Const(u64),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Add(Box<DimExpr>, Box<DimExpr>),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<DimExpr>);

/// Coefficients keyed by sorted bounded-dimension monomials.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Poly(BTreeMap<Vec<DimName>, u64>);

impl Poly {
    pub fn constant(c: u64) -> Self {
        let mut p = Poly::default();
        if c != 0 {
            p.0.insert(Vec::new(), c);
        }
        p
    }

    fn var(name: DimName) -> Self {
        Poly(BTreeMap::from([(vec![name], 1)]))
    }

    pub fn sum(&self, other: &Poly) -> Poly {
        let mut out = self.clone();
        for (m, c) in &other.0 {
            *out.0.entry(m.clone()).or_insert(0) += c;
        }
        out
    }

    pub fn product(&self, other: &Poly) -> Poly {
        let mut out = Poly::default();
        for (ma, ca) in &self.0 {
            for (mb, cb) in &other.0 {
                let mut m: Vec<DimName> = ma.iter().chain(mb).copied().collect();
                m.sort();
                *out.0.entry(m).or_insert(0) += ca * cb;
            }
        }
        out
    }

    pub fn as_const(&self) -> Option<u64> {
        match self.0.len() {
            0 => Some(0),
            1 => self.0.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    /// The value under `size`. `None` if a dimension is unbound.
    pub fn eval(&self, size: &dyn Fn(DimName) -> Option<u64>) -> Option<u64> {
        let mut total = 0;
        for (m, c) in &self.0 {
            let mut term = *c;
            for name in m {
                term *= size(*name)?;
            }
            total += term;
        }
        Some(total)
    }
}

impl DimExpr {
    /// Its value, given each bounded name's size.
    pub fn value(&self, dims: &[Dim], size: &dyn Fn(DimName) -> Option<u64>) -> Option<u64> {
        Some(match self {
            DimExpr::Const(c) => *c,
            DimExpr::Dim(name) => match dims.iter().find(|d| d.name == *name)?.size {
                Size::Fixed(n) => n,
                Size::Bounded(_) => size(*name)?,
            },
            DimExpr::Mul(a, b) => a.value(dims, size)? * b.value(dims, size)?,
            DimExpr::Add(a, b) => a.value(dims, size)? + b.value(dims, size)?,
        })
    }

    /// Fixed dimensions become constants. Bounded ones stay symbolic.
    pub fn poly(&self, dims: &[Dim]) -> Result<Poly, DimName> {
        Ok(match self {
            DimExpr::Const(c) => Poly::constant(*c),
            DimExpr::Dim(name) => match dims.iter().find(|d| d.name == *name) {
                Some(Dim {
                    size: Size::Fixed(n),
                    ..
                }) => Poly::constant(*n),
                Some(_) => Poly::var(*name),
                None => return Err(*name),
            },
            DimExpr::Mul(a, b) => a.poly(dims)?.product(&b.poly(dims)?),
            DimExpr::Add(a, b) => a.poly(dims)?.sum(&b.poly(dims)?),
        })
    }
}

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn elements(&self, dims: &[Dim]) -> Result<Poly, DimName> {
        self.0
            .iter()
            .try_fold(Poly::constant(1), |acc, e| Ok(acc.product(&e.poly(dims)?)))
    }
}

impl From<DimName> for DimExpr {
    fn from(name: DimName) -> Self {
        DimExpr::Dim(name)
    }
}

impl From<u64> for DimExpr {
    fn from(c: u64) -> Self {
        DimExpr::Const(c)
    }
}

impl<T: Into<DimExpr>> Mul<T> for DimExpr {
    type Output = DimExpr;
    fn mul(self, rhs: T) -> DimExpr {
        DimExpr::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<DimExpr>> Add<T> for DimExpr {
    type Output = DimExpr;
    fn add(self, rhs: T) -> DimExpr {
        DimExpr::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<DimExpr>> Mul<T> for DimName {
    type Output = DimExpr;
    fn mul(self, rhs: T) -> DimExpr {
        DimExpr::from(self) * rhs
    }
}

impl<T: Into<DimExpr>> Add<T> for DimName {
    type Output = DimExpr;
    fn add(self, rhs: T) -> DimExpr {
        DimExpr::from(self) + rhs
    }
}

impl<const N: usize> From<[DimName; N]> for Shape {
    fn from(names: [DimName; N]) -> Self {
        Shape(names.into_iter().map(DimExpr::Dim).collect())
    }
}

impl From<Vec<DimExpr>> for Shape {
    fn from(exprs: Vec<DimExpr>) -> Self {
        Shape(exprs)
    }
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Dim(n) => write!(f, "{n:?}"),
            DimExpr::Const(c) => write!(f, "{c}"),
            DimExpr::Mul(a, b) => write!(f, "{a}*{b}"),
            DimExpr::Add(a, b) => write!(f, "({a}+{b})"),
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|e| e.to_string()).collect();
        write!(f, "[{}]", parts.join(", "))
    }
}
