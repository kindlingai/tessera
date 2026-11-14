//! Each row's largest entry, found in two launches.

use codegen::prelude::*;
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Read, Shape, Step, T, Tree};

/// The most threads that share one row's scan.
const PARTS: usize = 256;

/// Rows of `width`, each scanned in `parts` pieces.
#[derive(Clone, Copy, Debug)]
pub struct Rows {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
}

/// The best index in each piece of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    peaks,
    name = "peaks",
    sizes = self.0.sizes(),
    work = self.0.rows * self.0.width,
    launches = self.0.launch(step, self.0.rows * self.0.parts),
)]
pub struct Peaks(pub Rows);

/// The best of each row's peaks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    argmax,
    name = "argmax",
    sizes = self.0.sizes(),
    work = self.0.rows * self.0.parts,
    launches = self.0.launch(step, self.0.rows),
)]
pub struct Argmax(pub Rows);

/// Strictly greater, so the lowest index wins ties.
#[kernel]
fn peaks(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32) {
    if gid() >= rows * parts {
        return;
    }
    let span = (width + parts - 1) / parts;
    let from = (gid() % parts) * span;
    let to = if from + span < width {
        from + span
    } else {
        width
    };
    let at = (gid() / parts) as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for i in from..to {
        if x[at + i as u64] > best {
            best = x[at + i as u64];
            arg = i;
        }
    }
    out[gid() as u64] = arg;
}

/// A piece with nothing finite holds the width.
#[kernel]
fn argmax(x: Buf<F32>, peaks: Buf<U32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for p in 0..parts {
        let i = peaks[(gid() * parts + p) as u64];
        if i < width && x[at + i as u64] > best {
            best = x[at + i as u64];
            arg = i;
        }
    }
    out[gid() as u64] = arg;
}

impl Rows {
    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    /// Every input, the output, then the three sizes.
    fn launch(&self, step: &Step, threads: usize) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.rows, self.width, self.parts].map(Arg::count));
        vec![Launch {
            args,
            shape: Shape::Line(threads),
        }]
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top {
        x,
        rows,
        width,
        k: 1,
    } = tree.site(at)?
    else {
        return None;
    };
    let parts = PARTS.min(width).max(1);
    let sizes = Rows { rows, width, parts };
    let both = [Read::Node(x), Read::Draft(0)];
    Some(vec![
        Draft::new(Peaks(sizes), [x], rows * parts),
        Draft::new(Argmax(sizes), both, rows),
    ])
}

lower::rule!("argmax", propose);
