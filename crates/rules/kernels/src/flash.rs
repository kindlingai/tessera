//! Causal attention on tiles, the keys streamed in blocks.

use codegen::ir::Body;
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{Map, MapStage};
use lower::{Arg, Attn, Draft, Kernel, Launch, Load, Read, Shape, Site, Source, Step, T, Tree};
use std::cell::RefCell;
use std::collections::HashMap;

/// Queries one group handles.
const BQ: u32 = 32;

/// Keys one block of the stream holds.
const BK: u32 = 16;

/// Four warps: two form the scores, all four apply them.
const THREADS: u32 = 128;

/// Floats of padding after each staged row, against bank conflicts.
const PAD: u32 = 4;

/// The widest head this schedule stages.
const WIDEST: usize = 128;

thread_local! {
    /// The kernel per head width, built once.
    static TREES: RefCell<HashMap<u32, Body>> = RefCell::new(HashMap::new());
}

/// Lanes keep their output shares across the key stream.
#[kernel]
fn flash(
    q: Buf<F32>,
    k: Buf<F32>,
    values: Buf<F32>,
    mut out: BufMut<F32>,
    queries: u32,
    keys: u32,
    past: u32,
    heads: u32,
    kv_heads: u32,
    #[build] w: u32,
) {
    let warp = lid() / LANES as u32;
    let group = gid() / THREADS;
    let h = group % heads;
    let q0 = (group / heads) * BQ;
    let kvh = h / (heads / kv_heads);
    let row = lid();
    let mut q_s = shared::<f32>(BQ * (w + PAD));
    // Keys, then the values transposed, in one slot.
    let mut kv_s = shared::<f32>(w * (BK + PAD));
    let mut p_s = shared::<f32>(BQ * (BK + PAD));
    let mut alpha_s = shared::<f32>(BQ);
    let mut l_s = shared::<f32>(BQ);
    for i in (lid()..BQ * w).by(THREADS) {
        let r = i / w;
        let d = i % w;
        q_s[r * (w + PAD) + d] = if q0 + r < queries {
            q[((q0 + r) * heads + h) as u64 * w as u64 + d as u64]
        } else {
            0.0
        };
    }
    let mut s_t = tiles(16, 16);
    let mut o00 = tiles(16, 16);
    let mut o01 = tiles(16, 16);
    let mut o10 = tiles(16, 16);
    let mut o11 = tiles(16, 16);
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    // Each warp owns up to two column pieces of sixteen.
    let first = warp * 16 < w;
    let second = (warp + 4) * 16 < w;
    for b in 0..(keys + BK - 1) / BK {
        let k0 = b * BK;
        for i in (lid()..BK * w).by(THREADS) {
            let t = i / w;
            let d = i % w;
            kv_s[t * (w + PAD) + d] = if k0 + t < keys {
                k[((k0 + t) * kv_heads + kvh) as u64 * w as u64 + d as u64]
            } else {
                0.0
            };
        }
        barrier();
        if warp < 2 {
            s_t.zero();
            s_t.block(&q_s, warp * 16 * (w + PAD), &kv_s, 0, w + PAD, w);
            s_t.each(|r, c, v| {
                let t = k0 + c;
                let seen = t < keys && t <= past + q0 + warp * 16 + r;
                p_s[(warp * 16 + r) * (BK + PAD) + c] = if seen { v } else { f32::NEG_INFINITY };
            });
        }
        barrier();
        if lid() < BQ {
            let mut top = f32::NEG_INFINITY;
            for c in 0..BK {
                top = max(top, p_s[row * (BK + PAD) + c]);
            }
            let m_new = max(m, top);
            // A row that has seen nothing keeps zero sums.
            let safe = if m_new == f32::NEG_INFINITY {
                0.0
            } else {
                m_new
            };
            let alpha = exp(m - safe);
            let mut sum = 0.0f32;
            for c in 0..BK {
                let p = exp(p_s[row * (BK + PAD) + c] - safe);
                p_s[row * (BK + PAD) + c] = p;
                sum += p;
            }
            l = l * alpha + sum;
            m = m_new;
            alpha_s[row] = alpha;
        }
        barrier();
        for i in (lid()..BK * w).by(THREADS) {
            let t = i / w;
            let d = i % w;
            kv_s[d * (BK + PAD) + t] = if k0 + t < keys {
                values[((k0 + t) * kv_heads + kvh) as u64 * w as u64 + d as u64]
            } else {
                0.0
            };
        }
        barrier();
        if first {
            o00.scale(|r| alpha_s[r]);
            o01.scale(|r| alpha_s[16 + r]);
            o00.block(&p_s, 0, &kv_s, warp * 16 * (BK + PAD), BK + PAD, BK);
            o01.block(
                &p_s,
                16 * (BK + PAD),
                &kv_s,
                warp * 16 * (BK + PAD),
                BK + PAD,
                BK,
            );
        }
        if second {
            o10.scale(|r| alpha_s[r]);
            o11.scale(|r| alpha_s[16 + r]);
            o10.block(&p_s, 0, &kv_s, (warp + 4) * 16 * (BK + PAD), BK + PAD, BK);
            o11.block(
                &p_s,
                16 * (BK + PAD),
                &kv_s,
                (warp + 4) * 16 * (BK + PAD),
                BK + PAD,
                BK,
            );
        }
        barrier();
    }
    if lid() < BQ {
        l_s[row] = l;
    }
    barrier();
    if first {
        o00.each(|r, c, v| {
            if q0 + r < queries {
                out[((q0 + r) * heads + h) as u64 * w as u64 + (warp * 16 + c) as u64] = v / l_s[r];
            }
        });
        o01.each(|r, c, v| {
            if q0 + 16 + r < queries {
                out[((q0 + 16 + r) * heads + h) as u64 * w as u64 + (warp * 16 + c) as u64] =
                    v / l_s[16 + r];
            }
        });
    }
    if second {
        o10.each(|r, c, v| {
            if q0 + r < queries {
                out[((q0 + r) * heads + h) as u64 * w as u64 + ((warp + 4) * 16 + c) as u64] =
                    v / l_s[r];
            }
        });
        o11.each(|r, c, v| {
            if q0 + 16 + r < queries {
                out[((q0 + 16 + r) * heads + h) as u64 * w as u64 + ((warp + 4) * 16 + c) as u64] =
                    v / l_s[16 + r];
            }
        });
    }
}

/// Reads scaled queries, and keys and values apart.
#[derive(Clone, Copy, Debug)]
pub struct Flash(pub Attn);

impl Flash {
    fn tree(&self) -> Body {
        let w = self.0.width as u32;
        TREES.with(|t| {
            t.borrow_mut()
                .entry(w)
                .or_insert_with(|| flash_ir(w))
                .clone()
        })
    }

    fn groups(&self) -> usize {
        self.0.heads * self.0.queries.div_ceil(BQ as usize)
    }
}

impl Kernel for Flash {
    fn name(&self) -> &'static str {
        "flash"
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn sizes(&self) -> Vec<usize> {
        self.0.sizes()
    }

    fn load(&self, len: usize) -> Option<Load> {
        let a = &self.0;
        let (keys, side) = (a.keys.most() as f64, (a.kv_heads * a.width) as f64);
        // Keys and values are read per block of queries.
        let blocks = a.queries.div_ceil(BQ as usize) as f64;
        let per_seq = 4.0 * keys * side * blocks;
        Some(Load {
            threads: a.seqs * self.groups() * THREADS as usize,
            group: THREADS as usize,
            trips: keys / BK as f64,
            trip: self.tree().trip(),
            reads: vec![
                4.0 * len as f64,
                per_seq * a.seqs as f64,
                per_seq * a.seqs as f64,
            ],
            written: 4.0 * len as f64,
            flight: 16.0,
        })
    }

    fn source(&self) -> Source {
        let mut text = self.tree().text();
        text.entry = format!("flash_{}", self.0.width);
        Source::Made(text)
    }

    /// One launch per sequence, each at its own offset.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let per_q = a.queries * a.heads * a.width;
        let per_kv = a.keys.most() * a.kv_heads * a.width;
        let launch = |s: usize| Launch {
            args: vec![
                Arg::Buffer(step.reads[0], s * per_q),
                Arg::Buffer(step.reads[1], s * per_kv),
                Arg::Buffer(step.reads[2], s * per_kv),
                Arg::Buffer(step.write, s * per_q),
                Arg::count(a.queries),
                Arg::size(a.keys),
                Arg::size(a.keys.less(a.queries)),
                Arg::count(a.heads),
                Arg::count(a.kv_heads),
            ],
            shape: Shape::Grid {
                across: self.groups(),
                down: 1,
                threads: THREADS as usize,
            },
        };
        (0..a.seqs).map(launch).collect()
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * a.queries * a.keys.most() * a.width) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }
}

/// Scale the queries, then attend by tiles.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Attend {
        scale,
        q,
        k,
        v,
        sizes,
    } = tree.site(at)?
    else {
        return None;
    };
    // Sequences would sit at moving offsets as the keys grow.
    if sizes.keys.runtime() && sizes.seqs > 1 {
        return None;
    }
    // Tiles of sixteen along the head, within one group's memory.
    if !sizes.width.is_multiple_of(16) || sizes.width > WIDEST {
        return None;
    }
    let out = sizes.seqs * sizes.queries * sizes.heads * sizes.width;
    let reads = [Read::Draft(0), Read::Node(k), Read::Node(v)];
    Some(vec![
        Draft::new(Map(vec![MapStage::Scale(scale)]), [q], out),
        Draft::new(Flash(sizes), reads, out),
    ])
}

lower::rule!("flash", propose);
