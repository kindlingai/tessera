//! Symbolic dimensions, shape expressions and their normal form.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeSet;
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

            fn built_in(text: &str) -> Option<DimName> {
                match text {
                    $(stringify!($name) => Some(DimName::$name),)*
                    _ => None,
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
        Self::built_in(text).unwrap_or_else(|| DimName::Named(Name(intern(text))))
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
        let text = Cow::<'de, str>::deserialize(d)?;
        Ok(DimName::parse(&text))
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

/// Sorted bounded-dimension factors, stored inline up to four.
#[derive(Clone, Debug)]
enum Mono {
    Small { len: u8, dims: [DimName; 4] },
    Many(Vec<DimName>),
}

impl Mono {
    const EMPTY: Self = Self::Small {
        len: 0,
        dims: [DimName::Seqs; 4],
    };

    fn var(name: DimName) -> Self {
        Self::Small {
            len: 1,
            dims: [name, DimName::Seqs, DimName::Seqs, DimName::Seqs],
        }
    }

    fn as_slice(&self) -> &[DimName] {
        match self {
            Self::Small { len, dims } => &dims[..*len as usize],
            Self::Many(v) => v.as_slice(),
        }
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn mul(&self, other: &Self) -> Self {
        let (a, b) = (self.as_slice(), other.as_slice());
        if a.is_empty() {
            return other.clone();
        }
        if b.is_empty() {
            return self.clone();
        }
        let total = a.len() + b.len();
        let (mut i, mut j) = (0, 0);
        if total <= 4 {
            let mut dims = [DimName::Seqs; 4];
            let mut k = 0;
            while i < a.len() && j < b.len() {
                if a[i] <= b[j] {
                    dims[k] = a[i];
                    i += 1;
                } else {
                    dims[k] = b[j];
                    j += 1;
                }
                k += 1;
            }
            for &d in &a[i..] {
                dims[k] = d;
                k += 1;
            }
            for &d in &b[j..] {
                dims[k] = d;
                k += 1;
            }
            return Self::Small {
                len: total as u8,
                dims,
            };
        }
        let mut out = Vec::with_capacity(total);
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                out.push(a[i]);
                i += 1;
            } else {
                out.push(b[j]);
                j += 1;
            }
        }
        out.extend_from_slice(&a[i..]);
        out.extend_from_slice(&b[j..]);
        Self::Many(out)
    }
}

impl PartialEq for Mono {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Mono {}

impl PartialOrd for Mono {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Mono {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

/// Canonical non-zero polynomial terms, inline for 0 or 1 term.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
enum Terms {
    #[default]
    Zero,
    One((Mono, u64)),
    Many(Vec<(Mono, u64)>),
}

impl Terms {
    fn as_slice(&self) -> &[(Mono, u64)] {
        match self {
            Self::Zero => &[],
            Self::One(pair) => std::slice::from_ref(pair),
            Self::Many(pairs) => pairs,
        }
    }

    fn from_sorted(mut pairs: Vec<(Mono, u64)>) -> Self {
        match pairs.len() {
            0 => Self::Zero,
            1 => Self::One(pairs.pop().expect("len 1")),
            _ => Self::Many(pairs),
        }
    }
}

/// Coefficients keyed by sorted bounded-dimension monomials.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Poly(Terms);

impl Poly {
    pub fn constant(c: u64) -> Self {
        if c == 0 {
            Poly(Terms::Zero)
        } else {
            Poly(Terms::One((Mono::EMPTY, c)))
        }
    }

    fn var(name: DimName) -> Self {
        Poly(Terms::One((Mono::var(name), 1)))
    }

    pub fn sum(&self, other: &Poly) -> Poly {
        match (&self.0, &other.0) {
            (Terms::Zero, _) => other.clone(),
            (_, Terms::Zero) => self.clone(),
            (Terms::One((ma, ca)), Terms::One((mb, cb))) => match ma.cmp(mb) {
                Ordering::Equal => Poly(Terms::One((ma.clone(), ca + cb))),
                Ordering::Less => Poly(Terms::Many(vec![(ma.clone(), *ca), (mb.clone(), *cb)])),
                Ordering::Greater => Poly(Terms::Many(vec![(mb.clone(), *cb), (ma.clone(), *ca)])),
            },
            (a, b) => {
                let (a, b) = (a.as_slice(), b.as_slice());
                let mut out = Vec::with_capacity(a.len() + b.len());
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    match a[i].0.cmp(&b[j].0) {
                        Ordering::Less => {
                            out.push(a[i].clone());
                            i += 1;
                        }
                        Ordering::Greater => {
                            out.push(b[j].clone());
                            j += 1;
                        }
                        Ordering::Equal => {
                            out.push((a[i].0.clone(), a[i].1 + b[j].1));
                            i += 1;
                            j += 1;
                        }
                    }
                }
                out.extend_from_slice(&a[i..]);
                out.extend_from_slice(&b[j..]);
                Poly(Terms::from_sorted(out))
            }
        }
    }

    pub fn product(&self, other: &Poly) -> Poly {
        match (&self.0, &other.0) {
            (Terms::Zero, _) | (_, Terms::Zero) => Poly(Terms::Zero),
            (Terms::One((ma, ca)), Terms::One((mb, cb))) => Poly(Terms::One((ma.mul(mb), ca * cb))),
            (a, b) => {
                let (a, b) = (a.as_slice(), b.as_slice());
                let mut raw = Vec::with_capacity(a.len() * b.len());
                for (ma, ca) in a {
                    for (mb, cb) in b {
                        raw.push((ma.mul(mb), ca * cb));
                    }
                }
                raw.sort_unstable_by(|x, y| x.0.cmp(&y.0));
                let mut out: Vec<(Mono, u64)> = Vec::with_capacity(raw.len());
                for (m, c) in raw {
                    if let Some(last) = out.last_mut()
                        && last.0 == m
                    {
                        last.1 += c;
                    } else {
                        out.push((m, c));
                    }
                }
                Poly(Terms::from_sorted(out))
            }
        }
    }

    /// Whether this is at least `other` at every size.
    pub fn covers(&self, other: &Poly) -> bool {
        let (a, b) = (self.0.as_slice(), other.0.as_slice());
        if b.is_empty() {
            return true;
        }
        let mut i = 0;
        for (mb, cb) in b {
            while i < a.len() && a[i].0 < *mb {
                i += 1;
            }
            if i == a.len() || a[i].0 != *mb || a[i].1 < *cb {
                return false;
            }
            i += 1;
        }
        true
    }

    pub fn as_const(&self) -> Option<u64> {
        match &self.0 {
            Terms::Zero => Some(0),
            Terms::One((m, c)) if m.is_empty() => Some(*c),
            _ => None,
        }
    }

    /// The value under `size`. `None` if a dimension is unbound.
    pub fn eval(&self, size: &dyn Fn(DimName) -> Option<u64>) -> Option<u64> {
        let mut total = 0;
        for (m, c) in self.0.as_slice() {
            let mut term = *c;
            for &name in m.as_slice() {
                term *= size(name)?;
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
