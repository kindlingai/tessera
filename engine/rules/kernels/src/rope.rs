//! Rotary positions: a table of angles, then the rotation.

use tessera_codegen::prelude::*;
use tessera_graph::{Pairs, Stage};
use tessera_lower::sites;
use tessera_lower::{Draft, Kernel, Read, Shape, T, Tree, knob};

knob!("rope", "threads", 1, 1024);

fn line_or_grid(len: usize, threads: usize) -> Shape {
    match threads {
        0 => Shape::Line(len),
        t => Shape::Grid {
            across: len.div_ceil(t),
            down: 1,
            threads: t,
        },
    }
}

/// One thread per position, head and pair, rotating in place.
#[kernel]
fn rope<const ADJACENT: bool>(
    mut x: BufMut<F32>,
    freqs: Buf<F32>,
    positions: u32,
    heads: u32,
    width: u32,
) {
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
    // Pairs lie side by side, or one in each half.
    let a = if ADJACENT { 2 * pair } else { pair };
    let b = if ADJACENT { a + 1 } else { pairs + pair };
    let x0 = x[at + a as u64];
    let x1 = x[at + b as u64];
    x[at + a as u64] = x0 * c - x1 * sn;
    x[at + b as u64] = x0 * sn + x1 * c;
}

/// One thread per position and pair: its cosine and sine.
#[kernel]
fn angles(
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    count: u32,
    width: u32,
    theta: f32,
    turned: u32,
) {
    let pairs = width / 2;
    let at = gid() / pairs;
    let pair = gid() % pairs;
    if at >= count {
        return;
    }
    let turn = pow(theta, -((2 * pair) as f32) / width as f32);
    // Pairs past `turned` keep angle zero.
    let angle = if pair < turned {
        positions[at as u64] as f32 * turn
    } else {
        0.0
    };
    out[(at * width + 2 * pair) as u64] = cos(angle);
    out[(at * width + 2 * pair + 1) as u64] = sin(angle);
}

/// Each position's cosine and sine, per pair.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    angles,
    threads = line_or_grid(self.count * self.width / 2, self.threads),
    shared,
)]
pub(crate) struct Angles {
    pub count: usize,
    pub width: usize,
    pub theta: f32,
    pub turned: usize,
    pub threads: usize,
}

/// Rotates pairs in place, by the second input's angles.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    rope,
    variant = self.adjacent,
    name = match self.adjacent {
        true => "rope_adjacent",
        false => "rope",
    },
    threads = line_or_grid(self.positions * self.heads * self.width / 2, self.threads),
    overwrites,
)]
pub struct Rope {
    pub positions: usize,
    pub heads: usize,
    pub width: usize,
    pub adjacent: bool,
    pub threads: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Stage {
        stage: Stage::Rope {
            theta,
            pairs,
            turned,
        },
        x,
        operand: Some((at, positions)),
        len,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    let threads = tree.knobs.get("rope", "threads").unwrap_or(0);
    let angles = Angles {
        count: positions,
        width,
        theta,
        turned: turned.map_or(width / 2, |t| t as usize),
        threads,
    };
    let rope = Rope {
        positions,
        heads: len / (positions * width),
        width,
        adjacent: pairs == Pairs::Adjacent,
        threads,
    };
    Some(vec![
        Draft::new(angles, [at], positions * width),
        Draft::new(rope, [Read::Node(x), Read::Draft(0)], len),
    ])
}

tessera_lower::rule!("rope", propose);
