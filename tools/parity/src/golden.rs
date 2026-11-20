//! Recorded references, and the outputs each model must match.

use super::{Spec, inputs, interpreted, loaded, runs};
use sensai_bundle::names::repo;
use sensai_checkpoint::{File, Offer};
use sensai_interp::Tensor;
use sensai_lower::form::{BF16_FORM, Load};
use sensai_lower::{Device, Pins, Weight, lower};
use sensai_run::Run;
use sensai_testkit::{argmax, cosine_f32, max_abs_diff};
use sensai_trace::DimName::{self, Context, Seqs};
use sensai_trace::Graph;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// A recorded reference, and how closely to match it.
pub struct Golden {
    /// Values the test itself holds, else the model's reference directory.
    pub record: Option<fn(&Graph) -> Reference>,
    /// The form the plan keeps weights in.
    pub form: &'static dyn Load,
    pub checks: &'static [Check],
    /// The plan runs on a host with no GPU too.
    pub anywhere: bool,
}

/// One recorded output, and the graph output it matches.
pub struct Check {
    /// The output's file name, less `.f32`.
    pub name: &'static str,
    pub compare: Compare,
    pub tolerance: Tolerance,
}

/// How recorded rows relate to a graph's output.
pub enum Compare {
    /// The release graph's output, a row per token.
    Tokens,
    /// The release graph's output, scaled to unit length.
    Unit,
    /// Row `n - 1` is the mean row of `probe(n)`.
    Layers(fn(u64) -> Graph, &'static [u64]),
}

/// Each position must pick the reference's top token.
const TOP_TOKEN: Check = Check {
    name: "logits",
    compare: Compare::Tokens,
    tolerance: Tolerance {
        max_abs: 1e-3,
        cos: 0.99999,
        top: true,
    },
};

/// A bf16 checkpoint, checked by its logits.
pub const LOGITS: Golden = Golden {
    record: None,
    form: BF16_FORM,
    checks: &[TOP_TOKEN],
    anywhere: false,
};

/// A reference's contents: weights, and inputs with outputs.
pub struct Reference {
    pub weights: Vec<Tensor>,
    pub cases: Vec<Case>,
    /// The checkpoint, which probes read again.
    pub checkpoint: Option<PathBuf>,
    /// Held while the reference lives: one loads at a time.
    pub turn: MutexGuard<'static, ()>,
}

/// Sequences of one length, and the outputs recorded for them.
pub struct Case {
    pub name: String,
    pub seqs: usize,
    pub tokens: Vec<u32>,
    pub outputs: BTreeMap<String, Vec<f32>>,
}

impl Case {
    /// `tokens.txt`, a sequence per line, and each `<name>.f32`.
    fn read(dir: &Path) -> Self {
        let text = std::fs::read_to_string(dir.join("tokens.txt")).unwrap();
        let (mut seqs, mut tokens) = (0, Vec::new());
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            seqs += 1;
            for id in line.split_whitespace() {
                tokens.push(id.parse().unwrap());
            }
        }
        assert_eq!(tokens.len() % seqs, 0, "{dir:?}: ragged sequences");
        let mut outputs = BTreeMap::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "f32") {
                let name = path.file_stem().unwrap().to_string_lossy();
                outputs.insert(name.into_owned(), floats(&path));
            }
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        Self {
            name,
            seqs,
            tokens,
            outputs,
        }
    }

    /// The output recorded as `name`.
    pub fn output(&self, name: &str) -> &[f32] {
        let found = self.outputs.get(name);
        found.unwrap_or_else(|| panic!("{}: no {name}.f32", self.name))
    }

    pub fn inputs(&self) -> [Tensor; 2] {
        inputs(self.seqs, &self.tokens)
    }

    pub fn sizes(&self) -> [(DimName, u64); 2] {
        let len = self.tokens.len() / self.seqs;
        [(Seqs, self.seqs as u64), (Context, len as u64)]
    }
}

/// Each `cases/<name>`, else `dir` alone, shortest first.
fn cases(dir: &Path) -> Vec<Case> {
    let mut out = Vec::new();
    match std::fs::read_dir(dir.join("cases")) {
        Ok(entries) => {
            for entry in entries {
                out.push(Case::read(&entry.unwrap().path()));
            }
        }
        Err(_) => out.push(Case::read(dir)),
    }
    out.sort_by_key(|c| c.tokens.len());
    out
}

/// Floats from a file of little-endian f32.
fn floats(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let wide = |b: &[u8; 4]| f32::from_le_bytes(*b);
    bytes.as_chunks::<4>().0.iter().map(wide).collect()
}

/// `g`'s weights from `offer`. Packed forms other than bf16 decode.
pub fn pulled(model: &str, g: &Graph, offer: &dyn Offer) -> Vec<Tensor> {
    let rules = repo(model).unwrap_or_default();
    let mut out = Vec::with_capacity(g.weights.len());
    let shape = |at: usize| -> Vec<usize> {
        let axis = |e: &sensai_graph::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        g.weights[at].ty.shape.0.iter().map(axis).collect()
    };
    sensai_bundle::each(g, offer, &[&rules], &|_| None, |w, _| {
        let shape = shape(out.len());
        out.push(match w {
            Weight::Floats(t) => t,
            Weight::Packed(f, bytes) if f.block() == 1 => {
                Tensor::from_bytes(shape, f.dtype(), &bytes)
            }
            Weight::Packed(f, bytes) => Tensor::f32(shape, f.unpack_all(&bytes)),
        });
        Ok(())
    })
    .unwrap();
    out
}

/// `g`'s weights from the safetensors file at `path`.
fn safetensors(model: &str, g: &Graph, path: &Path) -> Vec<Tensor> {
    pulled(model, g, &File::open(path).unwrap())
}

/// The one turn to hold a real checkpoint.
pub fn turn() -> MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    // A failed test still hands the turn on.
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// How far an output may stray from the reference.
pub struct Tolerance {
    pub max_abs: f32,
    /// The least cosine any row may have.
    pub cos: f32,
    /// Each row's largest entry must match.
    pub top: bool,
}

impl Tolerance {
    pub fn check(&self, label: &str, got: &[f32], want: &[f32], rows: usize) {
        let width = got.len() / rows;
        let (max, mut cos) = (max_abs_diff(got, want), 1.0f32);
        for (g, w) in got.chunks(width).zip(want.chunks(width)) {
            cos = cos.min(cosine_f32(g, w));
            assert!(!self.top || argmax(g) == argmax(w), "{label}: top differs");
        }
        eprintln!("{label}: max_abs {max:.3e} worst cos {cos:.7}");
        assert!(
            max < self.max_abs && cos > self.cos,
            "{label}: {max:.3e} {cos}"
        );
    }
}

impl Golden {
    /// The reference for `g`, where this host has it.
    pub fn reference(&self, model: &str, g: &Graph) -> Option<Reference> {
        if let Some(read) = self.record {
            return Some(read(g));
        }
        let dir = sensai_testkit::reference(model)?;
        let file = dir.join("model.safetensors");
        // The turn comes first, so the weights load in it.
        let turn = turn();
        Some(Reference {
            weights: safetensors(model, g, &file),
            cases: cases(&dir),
            checkpoint: Some(file),
            turn,
        })
    }

    /// `got`, the release graph's output, against `case`.
    pub fn release(&self, label: &str, got: &[f32], case: &Case) {
        for check in self.checks {
            let (rows, scaled) = match check.compare {
                Compare::Tokens => (case.tokens.len(), got.to_vec()),
                Compare::Unit => (1, unit(got)),
                Compare::Layers(..) => continue,
            };
            let want = case.output(check.name);
            check.tolerance.check(label, &scaled, want, rows);
        }
    }

    /// Each probe's mean row on the interpreter, shortest case only.
    fn probes(&self, model: &str, reference: &Reference) {
        let case = &reference.cases[0];
        for check in self.checks {
            let Compare::Layers(probe, layers) = check.compare else {
                continue;
            };
            let file = reference.checkpoint.as_ref().expect("probes read a file");
            let want = case.output(check.name);
            for &n in layers {
                let g = probe(n);
                let w = safetensors(model, &g, file);
                let out = interpreted(&g, &w, case.seqs, &case.tokens);
                let got = mean_rows(&out, case.tokens.len());
                let at = (n as usize - 1) * got.len();
                let label = format!("{} after layer {n}", case.name);
                check
                    .tolerance
                    .check(&label, &got, &want[at..at + got.len()], 1);
            }
        }
    }
}

pub fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

/// The mean of `rows` rows.
fn mean_rows(out: &[f32], rows: usize) -> Vec<f32> {
    let width = out.len() / rows;
    let mut mean = vec![0.0f64; width];
    for row in out.chunks(width) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += f64::from(*v);
        }
    }
    mean.iter().map(|m| (m / rows as f64) as f32).collect()
}

/// The interpreter and the plan against the reference.
pub fn golden(spec: &Spec) {
    let Some(golden) = &spec.golden else { return };
    let g = (spec.release)();
    let Some(reference) = golden.reference(spec.name, &g) else {
        return eprintln!("{}: no reference here; skipped", spec.name);
    };
    for case in &reference.cases {
        let out = interpreted(&g, &reference.weights, case.seqs, &case.tokens);
        golden.release(&format!("{} cpu", case.name), &out, case);
    }
    golden.probes(spec.name, &reference);
    for settings in runs() {
        if settings.gpu().is_none() && !golden.anywhere {
            continue;
        }
        let device = Device::guess(&settings).pinned(Pins::all(golden.form));
        let mut runner: Option<Box<dyn Run>> = None;
        for case in &reference.cases {
            let plan = lower(&g, 0, &case.sizes(), &device).unwrap();
            let r = match runner.as_mut() {
                Some(r) => r,
                None => runner.insert(loaded(plan.clone(), &g, &reference.weights, &settings)),
            };
            r.replan(&plan).unwrap();
            let inputs = case.inputs();
            let out = r.run(&[inputs[0].view(), inputs[1].view()]).unwrap();
            let label = format!("{} on {}", case.name, r.device());
            golden.release(&label, out[0].floats().unwrap(), case);
        }
    }
}
