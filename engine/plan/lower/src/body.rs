//! Helpers for rules that build a source from a body.

use crate::{Shape, Source};
use tessera_codegen::hash;
use tessera_codegen::ir::{Body, Param};

/// Guessed seconds per unit of work, under lockstep trees.
pub const GUESS: f64 = 1e-12;

/// `body` under an entry name that hashes text and forms.
pub fn source(kind: &str, mut body: Body) -> Source {
    let mut key = body.text().body;
    for p in &body.params {
        if let Param::Buffer { form, .. } = p {
            key.push(':');
            key.push_str(form.name());
        }
    }
    let hashed = hash(&key);
    body.entry = format!("{kind}_{hashed:x}");
    Source::Made(body)
}

/// Small groups: workers split a launch by group.
pub fn small(jobs: usize) -> Shape {
    Shape::Grid {
        across: jobs.div_ceil(8),
        down: 1,
        threads: 8,
    }
}
