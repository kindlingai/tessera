//! A module's kernel text, launched with plain arguments.

use crate::abi::{Offer, Shape};
use sensai_lower::kept::Layout;
use sensai_lower::{Arg, Kernel, Launch, Source, Step};
use std::fmt;

/// A module kernel's text, never freed.
#[derive(Debug)]
pub struct Fixed {
    pub name: &'static str,
    pub entry: &'static str,
    pub metal: &'static str,
    pub cuda: &'static str,
}

/// One launch of a module kernel: reads, output, then `args`.
#[derive(Clone)]
pub struct Text {
    pub fixed: &'static Fixed,
    pub args: Vec<Arg>,
    pub shape: sensai_lower::Shape,
    pub work: f64,
    pub guess: f64,
    /// For kept-row launches: the layout and its scalars.
    pub kept: Option<(&'static dyn Layout, Vec<Arg>)>,
}

impl Text {
    pub fn new(fixed: &'static Fixed, offer: &Offer, args: Vec<Arg>) -> Self {
        let shape = match offer.shape {
            Shape::Line(n) => sensai_lower::Shape::Line(n),
            Shape::Grid {
                across,
                down,
                threads,
            } => sensai_lower::Shape::Grid {
                across,
                down,
                threads,
            },
        };
        Self {
            fixed,
            args,
            shape,
            work: offer.work,
            guess: offer.guess,
            kept: None,
        }
    }
}

/// Name and launch, not text. Compiled kernels key on it.
impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Fixed { name, .. } = self.fixed;
        let layout = self.kept.as_ref().map(|(l, _)| l.name());
        write!(f, "{name}{:?}{:?}{layout:?}", self.args, self.shape)
    }
}

impl Kernel for Text {
    fn name(&self) -> &'static str {
        self.fixed.name
    }

    fn source(&self) -> Source {
        let Fixed {
            entry, metal, cuda, ..
        } = *self.fixed;
        Source::Fixed { metal, cuda, entry }
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend(self.args.iter().cloned());
        // Kept rows add any page table, then the layout's scalars.
        if let Some((layout, values)) = &self.kept {
            args.extend(step.kept(*layout));
            args.extend(values.iter().cloned());
        }
        vec![Launch {
            args,
            shape: self.shape,
        }]
    }

    fn work(&self, _: usize) -> f64 {
        self.work
    }

    fn sizes(&self) -> Vec<usize> {
        let size = |a: &Arg| match a {
            Arg::Size(e) => Some(e.most()),
            other => other.u32().map(|n| n as usize),
        };
        self.args.iter().filter_map(size).collect()
    }

    fn guess(&self) -> f64 {
        self.guess
    }

    /// The module saw each read's form before it offered.
    fn accepts(&self, _: usize, _: &dyn sensai_lower::form::Load) -> bool {
        true
    }
}
