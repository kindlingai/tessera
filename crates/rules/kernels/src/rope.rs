//! Rotary positions: a table of angles, then the rotation.

use crate::apply_rope_heads;
use codegen::prelude::*;
use graph::{Pairs, Stage};
use lower::{Arg, Draft, Kernel, Launch, Read, Shape, Site, Source, Step, T, Tree};

/// One thread per position and pair: its cosine and sine.
#[kernel]
fn angles(positions: Buf<U32>, mut out: BufMut<F32>, count: u32, width: u32, theta: f32) {
    let pairs = width / 2;
    let at = gid() / pairs;
    let pair = gid() % pairs;
    if at >= count {
        return;
    }
    let turn = pow(theta, -((2 * pair) as f32) / width as f32);
    let angle = positions[at as u64] as f32 * turn;
    out[(at * width + 2 * pair) as u64] = cos(angle);
    out[(at * width + 2 * pair + 1) as u64] = sin(angle);
}

/// Each position's cosine and sine, per pair.
#[derive(Clone, Copy, Debug)]
pub struct Angles {
    pub width: usize,
    pub theta: f32,
}

/// Rotates halves in place, by the second input's angles.
#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub positions: usize,
    pub heads: usize,
    pub width: usize,
}

impl Kernel for Angles {
    fn name(&self) -> &'static str {
        "angles"
    }

    fn source(&self) -> Source {
        let mut text = angles_ir().text();
        text.entry = "plan_angles".into();
        Source::Made(text)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let positions = step.reads[0].len;
        let sizes = [
            Arg::count(positions),
            Arg::count(self.width),
            Arg::F32(self.theta),
        ];
        let buffers = [step.read(0), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(positions * self.width / 2),
        }]
    }

    /// One table serves every rotation that matches.
    fn shared(&self) -> bool {
        true
    }
}

impl Kernel for Rope {
    fn name(&self) -> &'static str {
        "rope"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.positions, self.heads, self.width]
    }

    fn source(&self) -> Source {
        use apply_rope_heads::{CUDA, ENTRY, METAL};
        Source::Fixed {
            metal: METAL,
            cuda: CUDA,
            entry: ENTRY,
        }
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let shape = apply_rope_heads::Fixture {
            x: Vec::new(),
            freqs: Vec::new(),
            seq_len: self.positions,
            num_heads: self.heads,
            head_dim: self.width,
            rotary_dim: self.width,
            elem_offset: 0,
        };
        let sizes = Arg::pair(self.heads, self.positions);
        vec![Launch {
            args: vec![
                step.read(0),
                step.read(1),
                Arg::bytes(&shape.params()),
                sizes,
            ],
            shape: Shape::Line(self.heads * self.positions),
        }]
    }

    fn overwrites(&self) -> bool {
        true
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Stage {
        stage: Stage::Rope { theta, pairs },
        x,
        operand: Some((at, positions)),
        len,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    if pairs != Pairs::Halves {
        return None;
    }
    let angles = Angles { width, theta };
    let rope = Rope {
        positions,
        heads: len / (positions * width),
        width,
    };
    Some(vec![
        Draft::new(angles, [at], positions * width),
        Draft::new(rope, [Read::Node(x), Read::Draft(0)], len),
    ])
}

lower::rule!("rope", propose);
