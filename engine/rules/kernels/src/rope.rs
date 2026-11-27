//! Rotary positions: a table of angles, then the rotation.

use crate::io_name;
use tessera_codegen::prelude::*;
use tessera_graph::{Pairs, Stage};
use tessera_lower::form::{F32_FORM, Load as Stored, plain, writable};
use tessera_lower::sites;
use tessera_lower::{Draft, Kernel, Read, Shape, Source, T, Tree, knob};

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
    #[form(form)] mut x: BufMut<Dyn>,
    freqs: Buf<F32>,
    positions: u32,
    heads: u32,
    width: u32,
    #[build] form: &'static dyn Stored,
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
    rope(self.form),
    variant = self.adjacent,
    name = match self.adjacent {
        true => "rope_adjacent",
        false => "rope",
    },
    threads = line_or_grid(self.positions * self.heads * self.width / 2, self.threads),
    accepts = rope_accepts,
    emits = rope_emits,
    source = rope_source,
    overwrites,
)]
pub struct Rope {
    pub positions: usize,
    pub heads: usize,
    pub width: usize,
    pub adjacent: bool,
    pub threads: usize,
    pub form: &'static dyn Stored,
}

fn rope_accepts(k: &Rope, input: usize, form: &dyn Stored) -> bool {
    form.name() == if input == 0 { k.form.name() } else { "f32" }
}

fn rope_emits(k: &Rope, form: &dyn Stored) -> bool {
    form.name() == k.form.name()
}

fn rope_source(k: &Rope) -> Source {
    let base = if k.adjacent { "rope_adjacent" } else { "rope" };
    let body = if k.adjacent {
        rope_ir::<true>(k.form)
    } else {
        rope_ir::<false>(k.form)
    };
    Source::named(body, io_name(base, k.form, k.form))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Stage {
        stage: Stage::Rope {
            theta,
            pairs,
            turned,
        },
        x,
        operand: Some((pos_t, positions)),
        len,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    let xf = plain(tree.node(x).form);
    let of = writable(tree.node(at).form);
    let form = if xf.name() == of.name() { of } else { F32_FORM };
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
        form,
    };
    Some(vec![
        Draft::new(angles, [pos_t], positions * width),
        Draft::new(rope, [Read::Node(x), Read::Draft(0)], len),
    ])
}

tessera_lower::rule!("rope", propose);
