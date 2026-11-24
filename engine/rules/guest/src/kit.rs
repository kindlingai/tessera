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

impl Region<'_> {
    /// The value of `op`, if present and unique.
    pub fn scalar(&self, op: &str) -> Option<f32> {
        crate::site::scalar(self.scalars, op)
    }
}

impl Step<'_> {
    /// The value of `op`, if present and unique.
    pub fn scalar(&self, op: &str) -> Option<f32> {
        crate::site::scalar(self.scalars, op)
    }

    /// Rows the step reads.
    pub fn rows(&self) -> usize {
        self.inputs[self.ids[0]].shape[0]
    }

    /// Tokens, positions, slots and picks, as reads.
    pub fn id_reads(&self) -> Vec<Place> {
        self.ids.map(Place::Read).into()
    }

    /// Launch counts of tokens and picks.
    pub fn id_counts(&self) -> Vec<Count> {
        let [tokens, _, _, picks] = self.ids;
        vec![self.inputs[tokens].count(), self.inputs[picks].count()]
    }

    /// One persistent grid: `reads`, then every kept row.
    pub fn resident(
        &self,
        kernel: &'static str,
        mut reads: Vec<Place>,
        counts: Vec<Count>,
        work: f64,
        resident: Resident,
    ) -> Offer {
        reads.extend((0..self.kept.len()).map(|kept| Place::Kept { kept }));
        Offer {
            kernel,
            reads,
            len: self.len,
            counts,
            shape: Shape::Line(1),
            work,
            guess: 1e-12,
            writes: None,
            resident: Some(resident),
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

    /// As `get`, if the weight's form is `form`.
    pub fn typed(&self, path: &str, form: &str) -> Option<Found<'a>> {
        self.get(path).filter(|f| f.form == form)
    }

    /// Each of `paths`, else `None`.
    pub fn all<const N: usize>(&self, paths: [&str; N]) -> Option<[Found<'a>; N]> {
        let found = paths.map(|p| self.get(p));
        found
            .iter()
            .all(Option::is_some)
            .then(|| found.map(Option::unwrap))
    }

    /// Distinct weight paths.
    pub fn len(&self) -> usize {
        self.at.len()
    }

    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
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
