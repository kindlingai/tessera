//! The gated delta rule: one thread per value column.

use tessera_codegen::prelude::*;
use tessera_lower::{Arg, Draft, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};
use tessera_lower::{GUESS, small, source};

/// Keys wider than this do not fit a thread's column.
const WIDEST: usize = 128;

/// Threads in a GPU group.
const THREADS: usize = 128;

/// Per row: decay, write the value, answer the query.
#[kernel]
fn delta_rule(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    decay: Buf<F32>,
    rate: Buf<F32>,
    mut out: BufMut<F32>,
    scale: f32,
    jobs: u32,
    #[build] rows: u32,
    #[build] heads: u32,
    #[build] kv_heads: u32,
    #[build] key_width: u32,
    #[build] width: u32,
) {
    if gid() >= jobs {
        return;
    }
    let j = gid() % width;
    let sh = gid() / width;
    let s = sh / heads;
    let h = sh % heads;
    let kh = h / (heads / kv_heads);
    let mut col = [0.0f32; 128];
    for i in 0..key_width {
        col[i as usize] = 0.0;
    }
    for t in 0..rows {
        let row = (s * rows + t) * heads + h;
        let at = ((s * rows + t) * kv_heads + kh) as u64 * key_width as u64;
        let fade = exp(decay[row as u64]);
        let mut seen = 0.0f32;
        for i in 0..key_width {
            col[i as usize] = col[i as usize] * fade;
            seen += col[i as usize] * k[at + i as u64];
        }
        let o = row as u64 * width as u64 + j as u64;
        let change = (v[o] - seen) * rate[row as u64];
        let mut read = 0.0f32;
        for i in 0..key_width {
            col[i as usize] += k[at + i as u64] * change;
            read += col[i as usize] * q[at + i as u64];
        }
        out[o] = read * scale;
    }
}

/// Whole sequences, on the CPU or a GPU.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = self.name(),
    sizes = vec![self.site.seqs, self.site.rows, self.site.heads, self.site.width],
    work = self.site.out() * self.site.key_width * 4,
    guess = GUESS,
    runs_on = self.zone,
    source = source(self.name(), self.body()),
    launches = launches,
)]
pub(crate) struct Delta {
    site: sites::Delta,
    zone: ZoneKind,
}

impl Delta {
    fn name(&self) -> &'static str {
        match self.zone {
            ZoneKind::Cpu => "cpu_delta",
            _ => "delta",
        }
    }

    fn jobs(&self) -> usize {
        self.site.seqs * self.site.heads * self.site.width
    }

    fn body(&self) -> tessera_codegen::ir::Body {
        let d = &self.site;
        let [rows, heads, kv_heads] = [d.rows, d.heads, d.kv_heads].map(|n| n as u32);
        let [key_width, width] = [d.key_width, d.width].map(|n| n as u32);
        delta_rule_ir(rows, heads, kv_heads, key_width, width)
    }
}

/// One launch, the scale before the job count.
fn launches(k: &Delta, step: &Step) -> Vec<Launch> {
    let jobs = k.jobs();
    let shape = match k.zone {
        ZoneKind::Cpu => small(jobs),
        _ => Shape::Grid {
            across: jobs.div_ceil(THREADS),
            down: 1,
            threads: THREADS,
        },
    };
    let mut launch = Launch::plain(step, &[jobs], shape);
    let at = launch.args.len() - 1;
    launch.args.insert(at, Arg::value(&k.site.scale));
    vec![launch]
}

fn propose(zone: ZoneKind, tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Delta = tree.site(at)?;
    if site.key_width > WIDEST {
        return None;
    }
    let len = site.out();
    Some(vec![Draft::new(Delta { site, zone }, site.reads, len)])
}

fn on_cpu(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose(ZoneKind::Cpu, tree, at)
}

fn on_gpu(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose(ZoneKind::Gpu, tree, at)
}

tessera_lower::rule!("cpu_delta", on_cpu);
tessera_lower::rule!("delta", on_gpu);
