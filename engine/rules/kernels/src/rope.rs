//! Rotary positions: a table of angles, then the rotation.

use codegen::prelude::*;
use graph::{Pairs, Stage};
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Read, Shape, Source, Step, T, Tree};

/// One thread per position, head and pair, rotating in place.
#[kernel]
fn rope(mut x: BufMut<F32>, freqs: Buf<F32>, positions: u32, heads: u32, width: u32) {
    let pairs = width / 2;
    if gid() >= positions * heads * pairs {
        return;
    }
    let pair = gid() % pairs;
    let h = (gid() / pairs) % heads;
    let s = gid() / pairs / heads;
    let at = ((s * heads + h) * width) as u64;
    let c = freqs[(s * width + 2 * pair) as u64];
    let sn = freqs[(s * width + 2 * pair + 1) as u64];
    let x0 = x[at + pair as u64];
    let x1 = x[at + (pairs + pair) as u64];
    x[at + pair as u64] = x0 * c - x1 * sn;
    x[at + (pairs + pair) as u64] = x0 * sn + x1 * c;
}

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
        let mut body = angles_ir();
        body.entry = "plan_angles".into();
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let positions = step.reads[0].len;
        let sizes = [
            Arg::count(positions),
            Arg::count(self.width),
            Arg::value(&self.theta),
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
        let mut body = rope_ir();
        body.entry = "plan_rope".into();
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.positions, self.heads, self.width].map(Arg::count);
        let buffers = [step.read(0), step.read(1)];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.positions * self.heads * self.width / 2),
        }]
    }

    fn overwrites(&self) -> bool {
        true
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Stage {
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
