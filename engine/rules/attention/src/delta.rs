//! The gated delta rule: one thread per value column.

use tessera_codegen::prelude::*;
use tessera_lower::{Arg, Draft, Kernel, Launch, Read, Shape, Step, T, Tree, ZoneKind, sites};
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
    rows: u32,
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

/// Ragged delta rule updating kept per-slot state in place.
#[kernel]
fn delta_rule_rows(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    decay: Buf<F32>,
    rate: Buf<F32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    mut state: BufMut<F32>,
    mut out: BufMut<F32>,
    scale: f32,
    jobs: u32,
    rows: u32,
    #[build] heads: u32,
    #[build] kv_heads: u32,
    #[build] key_width: u32,
    #[build] width: u32,
    #[build] seqs: u32,
    #[build] snap: u32,
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
    let mut loaded = 0u32;
    let mut prev = 0xffff_ffffu32;
    let mut last = 0.0f32;
    let base = (sh * key_width) as u64 * width as u64 + j as u64;
    for r in 0..rows {
        if slots[r as u64] == s {
            let pos = positions[r as u64];
            if loaded == 0 {
                loaded = 1;
                if pos == 0 {
                    for i in 0..key_width {
                        col[i as usize] = 0.0;
                    }
                } else {
                    for i in 0..key_width {
                        col[i as usize] = state[base + (i * width) as u64];
                    }
                }
            } else if pos == 0 && prev != 0 {
                for i in 0..key_width {
                    col[i as usize] = 0.0;
                }
            }
            let row = r * heads + h;
            let o = row as u64 * width as u64 + j as u64;
            if pos == prev {
                out[o] = last;
            } else {
                let at = (r * kv_heads + kh) as u64 * key_width as u64;
                let fade = exp(decay[row as u64]);
                let mut seen = 0.0f32;
                for i in 0..key_width {
                    col[i as usize] = col[i as usize] * fade;
                    seen += col[i as usize] * k[at + i as u64];
                }
                let change = (v[o] - seen) * rate[row as u64];
                let mut read = 0.0f32;
                for i in 0..key_width {
                    col[i as usize] += k[at + i as u64] * change;
                    read += col[i as usize] * q[at + i as u64];
                }
                last = read * scale;
                prev = pos;
                out[o] = last;
                if snap != 0 && s == 0 && rows > 1 && rows < seqs {
                    let sb = (((r + 1) * heads + h) * key_width) as u64 * width as u64 + j as u64;
                    for i in 0..key_width {
                        state[sb + (i * width) as u64] = col[i as usize];
                    }
                }
            }
        }
    }
    if loaded != 0 {
        for i in 0..key_width {
            state[base + (i * width) as u64] = col[i as usize];
        }
    }
}

/// Whole or ragged sequences, on the CPU or a GPU.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = self.name(),
    sizes = vec![self.site.seqs, self.site.rows.most(), self.site.heads, self.site.width],
    work = self.site.out() * self.site.key_width * 4,
    guess = GUESS,
    runs_on = self.zone,
    source = source(self.name(), self.body()),
    launches = launches,
)]
pub(crate) struct Delta {
    site: sites::Delta,
    zone: ZoneKind,
    snap: bool,
}

impl Delta {
    fn name(&self) -> &'static str {
        match (self.zone, self.site.state.is_some()) {
            (ZoneKind::Cpu, true) => "cpu_delta_rows",
            (ZoneKind::Cpu, false) => "cpu_delta",
            (_, true) => "delta_rows",
            (_, false) => "delta",
        }
    }

    fn jobs(&self) -> usize {
        self.site.seqs * self.site.heads * self.site.width
    }

    fn body(&self) -> tessera_codegen::ir::Body {
        let d = &self.site;
        let [heads, kv_heads] = [d.heads, d.kv_heads].map(|n| n as u32);
        let [key_width, width] = [d.key_width, d.width].map(|n| n as u32);
        match d.state {
            Some(_) => delta_rule_rows_ir(
                heads,
                kv_heads,
                key_width,
                width,
                d.seqs as u32,
                u32::from(self.snap),
            ),
            None => delta_rule_ir(heads, kv_heads, key_width, width),
        }
    }
}

/// One launch: the scale, the job count, then the rows.
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
    launch.args.push(Arg::size(k.site.rows));
    vec![launch]
}

fn propose(zone: ZoneKind, tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Delta = tree.site(at)?;
    if site.key_width > WIDEST {
        return None;
    }
    let len = site.out();
    let mut reads: Vec<Read> = site.reads.into_iter().map(Read::Node).collect();
    if let Some((idx, [slots, positions])) = site.state {
        reads.extend([Read::Node(slots), Read::Node(positions), Read::Kept(idx)]);
    }
    let snap = tree.has_taps() && site.seqs > 1;
    Some(vec![Draft::new(Delta { site, zone, snap }, reads, len)])
}

fn on_cpu(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose(ZoneKind::Cpu, tree, at)
}

fn on_gpu(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose(ZoneKind::Gpu, tree, at)
}

tessera_lower::rule!("cpu_delta", on_cpu);
tessera_lower::rule!("delta", on_gpu);
