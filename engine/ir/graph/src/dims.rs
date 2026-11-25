//! Symbolic dimensions, shape expressions and their normal form.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Add, Mul};
use std::sync::Mutex;

/// Declares the built-in names and their spellings.
macro_rules! dim_names {
    ($($(#[$doc:meta])* $name:ident,)*) => {
        /// Every shape is written in these names.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum DimName {
            $($(#[$doc])* $name,)*
            /// A name a model declares with [`DimName::named`].
            Named(Name),
        }

        impl DimName {
            const BUILT_IN: &[DimName] = &[$(DimName::$name),*];

            /// The spelling in graph JSON and recipe modules.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(DimName::$name => stringify!($name),)*
                    DimName::Named(name) => name.0,
                }
            }
        }
    };
}

dim_names! {
    Seqs,
    Rows,
    Context,
    /// Slots one launch's rows span. A size with no axis.
    Segments,
    /// Rows a step returns results for.
    Picks,
    /// Length of a learned position table.
    Positions,
    Hidden,
    Heads,
    KvHeads,
    HeadDim,
    /// Width of a pooled embedding.
    Embedding,
    Ffn,
    Vocab,
    Experts,
    TopK,
    Layers,
    /// `u32` words of a one-bit-per-token mask.
    VocabWords,
    /// Token ids one sparse pick chooses from.
    Candidates,
    /// `u32` words of one pick's packed logit edits.
    EditWords,
}

/// The spelling of a model's own dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(&'static str);

impl DimName {
    /// A model's own name. A built-in spelling panics.
    pub const fn named(name: &'static str) -> DimName {
        let mut i = 0;
        while i < Self::BUILT_IN.len() {
            assert!(
                !same_text(Self::BUILT_IN[i].as_str(), name),
                "use the built-in DimName"
            );
            i += 1;
        }
        DimName::Named(Name(name))
    }

    /// The name a spelling stands for, built-in or declared.
    pub fn parse(text: &str) -> DimName {
        for name in Self::BUILT_IN {
            if name.as_str() == text {
                return *name;
            }
        }
        DimName::Named(Name(intern(text)))
    }
}

const fn same_text(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// One leaked copy of each spelling read at run time.
fn intern(text: &str) -> &'static str {
    static SEEN: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(known) = seen.get(text) {
        return known;
    }
    let leaked: &'static str = Box::leak(text.into());
    seen.insert(leaked);
    leaked
}

impl Serialize for DimName {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DimName {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(DimName::parse(&String::deserialize(d)?))
    }
}

impl fmt::Display for DimName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for DimName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, derive_more::From)]
pub enum DimExpr {
    Dim(DimName),
    Const(u64),
    #[from(ignore)]
    Mul(Box<DimExpr>, Box<DimExpr>),
    #[from(ignore)]
    Add(Box<DimExpr>, Box<DimExpr>),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, derive_more::From)]
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

    /// Whether this is at least `other` at every size.
    pub fn covers(&self, other: &Poly) -> bool {
        (other.0.iter()).all(|(m, c)| self.0.get(m).is_some_and(|own| own >= c))
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
