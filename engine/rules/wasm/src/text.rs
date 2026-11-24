//! A module's kernel text, launched with plain arguments.

use crate::abi::{File, Lane, Offer, Shape};
use crate::leak;
use std::fmt;
use tessera_device::{Backend, Options, Problem, flatten};
use tessera_lower::kept::Layout;
use tessera_lower::{
    Arg, ByBackend, Collective, Dialect, Kernel, Launch, Needs, Resident, Source, Step,
};

/// A module kernel's text, never freed.
#[derive(Debug)]
pub struct Fixed {
    pub name: &'static str,
    pub entry: &'static str,
    pub text: ByBackend<&'static str>,
    /// Platform backends with no text.
    pub(crate) lacks: &'static [Backend],
}

/// Each backend's text of `file`, or empty.
fn flat(file: &str, files: &[File]) -> Result<ByBackend<String>, tessera_device::Error> {
    let load = |path: &str| {
        files
            .iter()
            .find(|f| f.path == path)
            .map(|f| f.text.clone())
    };
    let text = load(file).ok_or_else(|| tessera_device::Error {
        file: file.into(),
        line: 0,
        problem: Problem::Missing(file.into()),
    })?;
    let mut out = ByBackend::<String>::default();
    let mut skipped = None;
    for (slot, &backend) in out.0.iter_mut().zip(Backend::ALL) {
        // No remote rows or marks yet.
        let options = Options {
            backend,
            remote_kv: false,
            marks: false,
        };
        match flatten(file, &text, &load, &options) {
            Ok(flat) => *slot = flat.text,
            Err(e) if matches!(e.problem, Problem::Extension(_) | Problem::Unsupported(..)) => {
                skipped = Some(e)
            }
            Err(e) => return Err(e),
        }
    }
    match skipped {
        Some(e) if out.0.iter().all(String::is_empty) => Err(e),
        _ => Ok(out),
    }
}

impl Fixed {
    /// Kernel `k`, with any recipe file flattened per backend.
    pub fn open(k: crate::abi::Text, files: &[File]) -> Result<Self, tessera_device::Error> {
        let text = if k.file.is_empty() {
            // CUDA text also serves HIP.
            let (metal, cuda) = (leak(k.metal), leak(k.cuda));
            ByBackend::from_fn(|b| match Dialect::of(b) {
                Dialect::Metal => metal,
                Dialect::Cuda => cuda,
            })
        } else {
            ByBackend(flat(&k.file, files)?.0.map(leak))
        };
        let lacks = Backend::ALL.iter().filter(|b| text[**b].is_empty());
        Ok(Fixed {
            name: leak(k.name),
            entry: leak(k.entry),
            text,
            lacks: lacks.copied().collect::<Vec<_>>().leak(),
        })
    }
}

/// One launch of a module kernel: reads, output, then `args`.
#[derive(Clone)]
pub struct Text {
    pub fixed: &'static Fixed,
    pub args: Vec<Arg>,
    pub shape: tessera_lower::Shape,
    pub work: f64,
    pub guess: f64,
    /// For kept-row launches: the layout and its scalars.
    pub kept: Option<(&'static dyn Layout, Vec<Arg>)>,
    pub resident: Option<Resident>,
}

impl Text {
    pub fn new(fixed: &'static Fixed, offer: &Offer, args: Vec<Arg>) -> Self {
        let shape = match offer.shape {
            Shape::Line(n) => tessera_lower::Shape::Line(n),
            Shape::Grid {
                across,
                down,
                threads,
            } => tessera_lower::Shape::Grid {
                across,
                down,
                threads,
            },
        };
        let lane = |l: &Lane| match l {
            Lane::Sum => Collective::Sum,
            Lane::SumBf16 => Collective::SumBf16,
        };
        let resident = offer.resident.as_ref().map(|r| Resident {
            threads: r.threads,
            shared: r.shared,
            scratch: r.scratch,
            lanes: r.lanes.iter().map(lane).collect(),
        });
        Self {
            fixed,
            args,
            shape,
            work: offer.work,
            guess: offer.guess,
            kept: None,
            resident,
        }
    }
}

/// Name and launch, not text. Compiled kernels key on it.
impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Fixed { name, .. } = self.fixed;
        let layout = self.kept.as_ref().map(|(l, _)| l.name());
        write!(f, "{name}{:?}{:?}{layout:?}", self.args, self.shape)?;
        match &self.resident {
            Some(r) => write!(f, "{r:?}"),
            None => Ok(()),
        }
    }
}

impl Kernel for Text {
    fn name(&self) -> &'static str {
        self.fixed.name
    }

    fn source(&self) -> Source {
        let Fixed { entry, text, .. } = *self.fixed;
        Source::Fixed { text, entry }
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
    fn accepts(&self, _: usize, _: &dyn tessera_lower::form::Load) -> bool {
        true
    }

    fn resident(&self) -> Option<Resident> {
        self.resident.clone()
    }

    /// A module may bring text for some backends only.
    fn needs(&self) -> Needs {
        Needs {
            lacks: self.fixed.lacks,
            ..Needs::text()
        }
    }
}
