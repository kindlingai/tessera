//! Typed helpers that offer logic reads sites with.

use crate::site::{Count, Input, Kept, Kind, Lane, Offer, Place, Resident, Scalar, Shape, Site};
use std::collections::HashMap;
use std::ops::Deref;

/// A region site's fields.
#[derive(Clone, Copy)]
pub struct Region<'a> {
    pub inputs: &'a [Input],
    /// Elements per output.
    pub outputs: &'a [usize],
    pub kept: &'a [Kept],
    pub sums: &'a [usize],
    pub scalars: &'a [Scalar],
}

/// A model region's step: one output, kept rows by slot.
pub struct Step<'a> {
    pub inputs: &'a [Input],
    /// Elements of the output.
    pub len: usize,
    pub kept: &'a [Kept],
    pub sums: &'a [usize],
    pub scalars: &'a [Scalar],
    /// Input places of tokens, positions, slots and picks.
    pub ids: [usize; 4],
}

impl Site {
    /// This site as a region of `want`.
    pub fn region(&self, want: Kind) -> Option<Region<'_>> {
        let Site::Region {
            kind,
            inputs,
            outputs,
            kept,
            sums,
            scalars,
        } = self
        else {
            return None;
        };
        (*kind == want).then_some(Region {
            inputs,
            outputs,
            kept,
            sums,
            scalars,
        })
    }

    /// This site as a model region's step.
    pub fn step(&self) -> Option<Step<'_>> {
        let r = self.region(Kind::Model)?;
        let (&[len], Some(first)) = (r.outputs, r.kept.first()) else {
            return None;
        };
        let [Some(slots), Some(positions)] = first.by else {
            return None;
        };
        let inputs = r.inputs;
        let ids = |i: &usize| inputs[*i].form == "u32" && ![slots, positions].contains(i);
        // Exactly tokens then picks, else the site is new.
        let &[tokens, picks] = &(0..inputs.len()).filter(ids).collect::<Vec<_>>()[..] else {
            return None;
        };
        Some(Step {
            inputs,
            len,
            kept: r.kept,
            sums: r.sums,
            scalars: r.scalars,
            ids: [tokens, positions, slots, picks],
        })
    }
}

impl Region<'_> {}

impl Step<'_> {
    /// The value of `op`, if present and unique.
    pub fn scalar(&self, op: &str) -> Option<f32> {
        crate::site::scalar(self.scalars, op)
    }

    /// Rows the step reads.
    pub fn rows(&self) -> usize {
        self.inputs[self.ids[0]].shape[0]
    }

    /// Reads ids, `reads`, kept rows. Counts ids, `sizes`, op bits.
    pub fn resident(
        &self,
        kernel: &'static str,
        reads: impl IntoIterator<Item = Place>,
        sizes: &[usize],
        ops: &[&str],
        work: f64,
        resident: Resident,
    ) -> Option<Offer> {
        let [tokens, _, _, picks] = self.ids;
        let mut counts = vec![self.inputs[tokens].count(), self.inputs[picks].count()];
        counts.extend(fixed(sizes));
        for op in ops {
            counts.push(Count::bits(self.scalar(op)?));
        }
        let mut all: Vec<Place> = self.ids.map(Place::Read).into();
        all.extend(reads);
        all.extend((0..self.kept.len()).map(|kept| Place::Kept { kept }));
        Some(Offer {
            kernel,
            reads: all,
            len: self.len,
            counts,
            work,
            resident: Some(resident),
            ..Offer::default()
        })
    }
}

impl Default for Offer {
    fn default() -> Self {
        Offer {
            kernel: "",
            reads: Vec::new(),
            len: 0,
            counts: Vec::new(),
            shape: Shape::Line(1),
            work: 0.0,
            guess: 1e-12,
            writes: None,
            resident: None,
        }
    }
}

impl Resident {
    /// `threads` per block, `scratch` bytes and one `lane`.
    pub fn grid(threads: u32, scratch: usize, lane: Lane) -> Self {
        Resident {
            threads,
            shared: 0,
            scratch,
            lanes: vec![lane],
        }
    }
}

impl Shape {
    /// Blocks `across` by `down`, `threads` each.
    pub fn grid(across: usize, down: usize, threads: usize) -> Self {
        Shape::Grid {
            across,
            down,
            threads,
        }
    }
}

impl Input {
    /// Its first axis: the launch dimension, else its size.
    pub fn count(&self) -> Count {
        match self.dims.first() {
            Some(Some(dim)) => Count::Dim {
                dim: dim.clone(),
                mul: 1,
            },
            _ => Count::Fixed(self.shape[0] as u32),
        }
    }

    /// The whole size of `axis`, from share or shape.
    pub fn whole(&self, axis: usize) -> usize {
        match self.share {
            Some([a, _, _, w]) if a == axis => w,
            _ => self.shape[axis],
        }
    }

    /// This rank's first and count along `axis`, in `unit`s.
    pub fn split(&self, axis: usize, unit: usize) -> Option<Option<[usize; 2]>> {
        match self.share {
            None => Some(None),
            Some([a, first, count, _]) if a == axis && first % unit == 0 && count % unit == 0 => {
                Some(Some([first / unit, count / unit]))
            }
            Some(_) => None,
        }
    }
}

impl From<usize> for Count {
    fn from(n: usize) -> Self {
        Count::Fixed(n as u32)
    }
}

impl Count {
    /// An f32 constant, passed as its bits.
    pub fn bits(value: f32) -> Self {
        Count::Fixed(value.to_bits())
    }
}

/// Fixed counts of `sizes`, in order.
pub fn fixed(sizes: &[usize]) -> Vec<Count> {
    sizes.iter().map(|&n| n.into()).collect()
}

/// Weights by dotted path, less the prefix all share.
pub struct Weights<'a> {
    inputs: &'a [Input],
    at: HashMap<&'a str, usize>,
}

impl<'a> Weights<'a> {
    pub fn new(inputs: &'a [Input]) -> Self {
        let mut paths = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            if let Some(path) = input.weight.as_deref() {
                paths.push((i, path));
            }
        }
        let cut = prefix(&paths);
        let mut at = HashMap::with_capacity(paths.len());
        for (i, path) in paths {
            at.entry(&path[cut..]).or_insert(i);
        }
        Weights { inputs, at }
    }

    /// The weight at `path`, after the shared prefix.
    pub fn get(&self, path: &str) -> Option<Found<'a>> {
        let at = *self.at.get(path)?;
        Some(Found {
            at,
            input: &self.inputs[at],
        })
    }

    /// Each of `paths`, else `None`.
    pub fn all<const N: usize>(&self, paths: [&str; N]) -> Option<[Found<'a>; N]> {
        self.group("", &paths)?.try_into().ok()
    }

    /// Each of `fields` under `prefix`, in order, else `None`.
    pub fn group(&self, prefix: &str, fields: &[&str]) -> Option<Vec<Found<'a>>> {
        let path = |f: &&str| self.get(&format!("{prefix}{f}"));
        fields.iter().map(path).collect()
    }

    /// Layers from 0 up whose `pattern` path is a weight.
    pub fn layers(&self, pattern: &str) -> usize {
        let has = |l| self.get(&layer(pattern, l)).is_some();
        (0..).take_while(|&l| has(l)).count()
    }

    /// Distinct weight paths.
    pub fn len(&self) -> usize {
        self.at.len()
    }

    #[allow(dead_code, reason = "len_without_is_empty asks for is_empty")]
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
}

/// `pattern` with each `{l}` set to layer `l`.
pub fn layer(pattern: &str, l: usize) -> String {
    pattern.replace("{l}", &l.to_string())
}

/// One value for every entry of `parts`, else `None`.
pub fn same<T: PartialEq>(parts: impl IntoIterator<Item = T>) -> Option<T> {
    let mut parts = parts.into_iter();
    let first = parts.next()?;
    parts.all(|p| p == first).then_some(first)
}

/// The part all split entries share, `whole` if none split.
pub fn agreed<T: PartialEq + Copy>(parts: &[Option<T>], whole: T) -> Option<T> {
    let one = same(parts.iter().flatten()).copied();
    one.or(parts.iter().all(Option::is_none).then_some(whole))
}

/// Trellis bits, codebook (0 mcg, 1 mul1, 2 pmcgx), rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trellis {
    pub k: u32,
    pub book: u32,
    pub rotated: bool,
}

/// The trellis form `form` names, `k` from 2 to 4.
pub fn trellis(form: &str) -> Option<Trellis> {
    let rest = form.strip_suffix("_rot");
    let rotated = rest.is_some();
    let rest = rest.unwrap_or(form).strip_prefix("trellis_k")?;
    let (k, book) = match (rest.strip_suffix("_mul1"), rest.strip_suffix("_pmcgx")) {
        (Some(k), _) => (k, 1),
        (_, Some(k)) => (k, 2),
        _ => (rest, 0),
    };
    let k: u32 = k.parse().ok()?;
    (2..=4).contains(&k).then_some(Trellis { k, book, rotated })
}

/// Bytes of the dotted prefix all `paths` share.
fn prefix(paths: &[(usize, &str)]) -> usize {
    let Some((_, first)) = paths.first() else {
        return 0;
    };
    let mut cut = 0;
    for (dot, _) in first.match_indices('.') {
        let head = &first[..=dot];
        if !paths.iter().all(|(_, p)| p.starts_with(head)) {
            break;
        }
        cut = dot + 1;
    }
    cut
}

/// One input and its place.
#[derive(Clone, Copy, Debug)]
pub struct Found<'a> {
    pub at: usize,
    pub input: &'a Input,
}

impl Deref for Found<'_> {
    type Target = Input;

    fn deref(&self) -> &Input {
        self.input
    }
}

impl Found<'_> {
    /// The input itself as a read.
    pub fn read(&self) -> Place {
        Place::Read(self.at)
    }

    /// Part `p` of a rotated weight: codes, `suh` or `svh`.
    pub fn part(&self, p: usize) -> Place {
        Place::Part { part: [self.at, p] }
    }

    /// The codes, `suh` and `svh` of a rotated weight.
    pub fn parts(&self) -> [Place; 3] {
        [0, 1, 2].map(|p| self.part(p))
    }
}
