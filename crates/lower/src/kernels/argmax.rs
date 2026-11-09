//! Each row's largest entry, found in two launches.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Read, Site};

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
#[derive(Clone, Copy, Debug)]
pub struct Peaks(pub Rows);

/// The best of each row's peaks.
#[derive(Clone, Copy, Debug)]
pub struct Argmax(pub Rows);

// Strictly greater, so the lowest index wins ties.
const PEAKS: &str = "    if (gid >= rows * parts) {
        return;
    }
    u32 span = (width + parts - 1) / parts;
    u32 from = (gid % parts) * span;
    u32 to = from + span < width ? from + span : width;
    u64 at = (u64)(gid / parts) * width;
    u32 arg = width;
    float best = f_bits(0xff800000u);
    for (u32 i = from; i < to; i++) {
        if (x[at + i] > best) {
            best = x[at + i];
            arg = i;
        }
    }
    out[gid] = arg;
";

// A piece with nothing finite holds the width.
const ARGMAX: &str = "    if (gid >= rows) {
        return;
    }
    u64 at = (u64)gid * width;
    u32 arg = width;
    float best = f_bits(0xff800000u);
    for (u32 p = 0; p < parts; p++) {
        u32 i = peaks[gid * parts + p];
        if (i < width && x[at + i] > best) {
            best = x[at + i];
            arg = i;
        }
    }
    out[gid] = arg;
";

impl Rows {
    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn text(&self, entry: &str, peaks: bool, body: &str) -> Source {
        let ids = |name: &str, writes| (name.to_string(), "u32", writes);
        let mut buffers = vec![float("x", false)];
        buffers.extend(peaks.then(|| ids("peaks", false)));
        buffers.push(ids("out", true));
        Source::Made(Text {
            entry: entry.into(),
            buffers,
            scalars: vec![("rows", "u32"), ("width", "u32"), ("parts", "u32")],
            body: body.into(),
        })
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

impl Kernel for Peaks {
    fn name(&self) -> &'static str {
        "peaks"
    }

    fn sizes(&self) -> Vec<usize> {
        self.0.sizes()
    }

    fn source(&self) -> Source {
        self.0.text("plan_peaks", false, PEAKS)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        self.0.launch(step, self.0.rows * self.0.parts)
    }

    fn work(&self, _: usize) -> f64 {
        (self.0.rows * self.0.width) as f64
    }
}

impl Kernel for Argmax {
    fn name(&self) -> &'static str {
        "argmax"
    }

    fn sizes(&self) -> Vec<usize> {
        self.0.sizes()
    }

    fn source(&self) -> Source {
        self.0.text("plan_argmax", true, ARGMAX)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        self.0.launch(step, self.0.rows)
    }

    fn work(&self, _: usize) -> f64 {
        (self.0.rows * self.0.parts) as f64
    }
}

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Top {
        x,
        rows,
        width,
        k: 1,
    } = *site
    else {
        return None;
    };
    let parts = PARTS.min(width).max(1);
    let sizes = Rows { rows, width, parts };
    let both = [Read::Value(x), Read::Draft(0)];
    Some(vec![
        Draft::new(Peaks(sizes), [x], rows * parts),
        Draft::new(Argmax(sizes), both, rows),
    ])
}

crate::rule!("argmax", propose);
