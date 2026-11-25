//! Fused residual add and RMS norm, per warp or block.

use crate::lane_norm::{BLOCK_MIN, BLOCK_ROWS, BLOCK_THREADS, WARPS, row_needs, row_shape};
use crate::{AHEAD, WIDE};
use tessera_codegen::prelude::*;
use tessera_graph::{Order, Stage};
use tessera_lower::form::{F32_FORM, Load as Stored, plain, writable};
use tessera_lower::{Draft, Kernel, Launch, Load, Source, Step, T, Term, ToArg, Tree, sites};

/// Maximum wide trips held in registers across passes.
const CACHE_TRIPS: u32 = 8;

/// Per-thread register slots for cached pass-1 sums.
const CACHE_SLOTS: usize = (CACHE_TRIPS * WIDE) as usize;

/// Trip count cached in registers, or zero when too wide.
fn cache_trips(width: usize, block: bool) -> u32 {
    let stride = if block { BLOCK_THREADS as usize } else { LANES } * WIDE as usize;
    let trips = width.div_ceil(stride);
    if trips <= CACHE_TRIPS as usize {
        trips as u32
    } else {
        0
    }
}

/// Sums two or three rows, optionally saves sum, norms sum.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn add_norm_lanes<const WIDE: u32, const AHEAD: u32>(
    #[form(a_form)] a: Buf<Dyn>,
    #[form(b_form)] b: Buf<Dyn>,
    #[form(c_form)] c: Buf<Dyn>,
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    #[form(sum_form)] mut sum_out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    eps: f32,
    #[build] a_form: &'static dyn Stored,
    #[build] b_form: &'static dyn Stored,
    #[build] c_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] sum_form: &'static dyn Stored,
    #[build] add3: bool,
    #[build] weighted: bool,
    #[build] save_sum: bool,
    #[build] reload_sum: bool,
    #[build] block: bool,
    #[build] use_cache: bool,
    #[build] cached: u32,
) {
    let group_w = if block { BLOCK_THREADS } else { LANES as u32 };
    let row = gid() / group_w;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let first = if block { lid() } else { lane() };
    let mut buf = [0.0f32; CACHE_SLOTS];
    let mut acc = 0.0f32;
    if use_cache {
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let va = a.load4(at + i as u64);
                let vb = b.load4(at + i as u64);
                let mut s = F32x4::zero();
                s.x = va.x + vb.x;
                s.y = va.y + vb.y;
                s.z = va.z + vb.z;
                s.w = va.w + vb.w;
                if add3 {
                    let vc = c.load4(at + i as u64);
                    s.x += vc.x;
                    s.y += vc.y;
                    s.z += vc.z;
                    s.w += vc.w;
                }
                if save_sum {
                    sum_out.store4(at + i as u64, s);
                }
                let idx = (t * WIDE) as usize;
                buf[idx] = s.x;
                buf[idx + 1] = s.y;
                buf[idx + 2] = s.z;
                buf[idx + 3] = s.w;
                acc += s.x * s.x;
                acc += s.y * s.y;
                acc += s.z * s.z;
                acc += s.w * s.w;
            }
        }
    } else {
        for i in (first * WIDE..width).by(group_w * WIDE).unroll(AHEAD) {
            let va = a.load4(at + i as u64);
            let vb = b.load4(at + i as u64);
            let mut s = F32x4::zero();
            s.x = va.x + vb.x;
            s.y = va.y + vb.y;
            s.z = va.z + vb.z;
            s.w = va.w + vb.w;
            if add3 {
                let vc = c.load4(at + i as u64);
                s.x += vc.x;
                s.y += vc.y;
                s.z += vc.z;
                s.w += vc.w;
            }
            if save_sum {
                sum_out.store4(at + i as u64, s);
            }
            acc += s.x * s.x;
            acc += s.y * s.y;
            acc += s.z * s.z;
            acc += s.w * s.w;
        }
    }
    acc = sum32(acc);
    let mut tot = acc;
    if block {
        let mut red = shared::<f32>(WARPS);
        if lane() == 0 {
            red[lid() / LANES as u32] = acc;
        }
        barrier();
        tot = 0.0;
        for w_i in 0..WARPS {
            tot += red[w_i];
        }
    }
    let inv = rsqrt(tot / width as f32 + eps);
    if use_cache {
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let idx = (t * WIDE) as usize;
                let mut g = F32x4::zero();
                if weighted {
                    g = w.load4(i as u64);
                } else {
                    g.x = 1.0;
                    g.y = 1.0;
                    g.z = 1.0;
                    g.w = 1.0;
                }
                let mut o = F32x4::zero();
                o.x = buf[idx] * inv * g.x;
                o.y = buf[idx + 1] * inv * g.y;
                o.z = buf[idx + 2] * inv * g.z;
                o.w = buf[idx + 3] * inv * g.w;
                out.store4(at + i as u64, o);
            }
        }
    } else {
        for i in (first * WIDE..width).by(group_w * WIDE).unroll(AHEAD) {
            let mut s = F32x4::zero();
            if reload_sum {
                s = sum_out.load4(at + i as u64);
            } else {
                let va = a.load4(at + i as u64);
                let vb = b.load4(at + i as u64);
                s.x = va.x + vb.x;
                s.y = va.y + vb.y;
                s.z = va.z + vb.z;
                s.w = va.w + vb.w;
                if add3 {
                    let vc = c.load4(at + i as u64);
                    s.x += vc.x;
                    s.y += vc.y;
                    s.z += vc.z;
                    s.w += vc.w;
                }
            }
            let mut g = F32x4::zero();
            if weighted {
                g = w.load4(i as u64);
            } else {
                g.x = 1.0;
                g.y = 1.0;
                g.z = 1.0;
                g.w = 1.0;
            }
            let mut o = F32x4::zero();
            o.x = s.x * inv * g.x;
            o.y = s.y * inv * g.y;
            o.z = s.z * inv * g.z;
            o.w = s.w * inv * g.w;
            out.store4(at + i as u64, o);
        }
    }
}

/// Fused add + RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = add_norm_name(self),
    launches = add_norm_launches,
    work = self.rows * self.width,
    needs = row_needs(self.block),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    accepts = add_norm_accepts,
    emits = add_norm_emits,
    emits_at = add_norm_emits_at,
    source = add_norm_source,
    load = add_norm_load,
    unproven,
)]
pub(crate) struct AddLaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub add3: bool,
    pub weighted: bool,
    pub save_sum: bool,
    pub block: bool,
    pub a_form: &'static dyn Stored,
    pub b_form: &'static dyn Stored,
    pub c_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub sum_form: &'static dyn Stored,
}

impl AddLaneNorm {
    /// Whether pass 2 reloads the f32 sum from pass 1.
    fn reload_sum(&self) -> bool {
        self.save_sum && self.sum_form.name() == F32_FORM.name()
    }
}

fn add_norm_name(k: &AddLaneNorm) -> &'static str {
    match (k.block, k.add3, k.weighted, k.save_sum) {
        (true, false, true, true) => "block_add_norm_save",
        (true, false, false, true) => "block_add_rms_save",
        (true, false, true, false) => "block_add_norm",
        (true, false, false, false) => "block_add_rms",
        (true, true, true, true) => "block_add3_norm_save",
        (true, true, false, true) => "block_add3_rms_save",
        (true, true, true, false) => "block_add3_norm",
        (true, true, false, false) => "block_add3_rms",
        (false, false, true, true) => "lane_add_norm_save",
        (false, false, false, true) => "lane_add_rms_save",
        (false, false, true, false) => "lane_add_norm",
        (false, false, false, false) => "lane_add_rms",
        (false, true, true, true) => "lane_add3_norm_save",
        (false, true, false, true) => "lane_add3_rms_save",
        (false, true, true, false) => "lane_add3_norm",
        (false, true, false, false) => "lane_add3_rms",
    }
}

fn add_norm_accepts(k: &AddLaneNorm, input: usize, form: &dyn Stored) -> bool {
    let want = match input {
        0 => k.a_form,
        1 => k.b_form,
        2 if k.add3 => k.c_form,
        2 if k.weighted => k.w_form,
        3 if k.add3 && k.weighted => k.w_form,
        _ => return !form.narrow(),
    };
    form.name() == want.name()
}

fn add_norm_emits(k: &AddLaneNorm, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn add_norm_emits_at(k: &AddLaneNorm, output: usize, form: &dyn Stored) -> bool {
    match output {
        0 => form.name() == k.out_form.name(),
        1 if k.save_sum => form.name() == k.sum_form.name(),
        _ => !form.narrow(),
    }
}

fn add_norm_body(k: &AddLaneNorm) -> tessera_codegen::ir::Body {
    let cached = cache_trips(k.width, k.block);
    let mut body = add_norm_lanes_ir::<WIDE, AHEAD>(
        k.a_form,
        k.b_form,
        k.c_form,
        k.w_form,
        k.out_form,
        k.sum_form,
        k.add3,
        k.weighted,
        k.save_sum,
        k.reload_sum(),
        k.block,
        cached > 0,
        cached,
    );
    if !k.add3 {
        body.params.retain(|p| p.name() != "c");
    }
    if !k.weighted {
        body.params.retain(|p| p.name() != "w");
    }
    if !k.save_sum {
        body.params.retain(|p| p.name() != "sum_out");
    }
    body
}

fn add_norm_launches(k: &AddLaneNorm, step: &Step) -> Vec<Launch> {
    let body = add_norm_body(k);
    let scalar = |name: &str| match name {
        "rows" => k.rows.arg(),
        "width" => k.width.arg(),
        "eps" => k.eps.arg(),
        other => panic!("AddLaneNorm has no field `{other}`"),
    };
    vec![Launch::of(
        &body,
        step,
        false,
        &scalar,
        row_shape(k.rows, k.block),
    )]
}

fn add_norm_source(k: &AddLaneNorm) -> Source {
    let body = add_norm_body(k);
    let base = k.name();
    let cached = cache_trips(k.width, k.block);
    let tag = match cached {
        0 => String::new(),
        n => format!("_c{n}"),
    };
    let mut forms = vec![k.a_form.name(), k.b_form.name()];
    if k.add3 {
        forms.push(k.c_form.name());
    }
    if k.weighted {
        forms.push(k.w_form.name());
    }
    forms.push(k.out_form.name());
    if k.save_sum {
        forms.push(k.sum_form.name());
    }
    let entry = match forms.iter().all(|&f| f == "f32") {
        true => format!("plan_{base}{tag}"),
        false => format!("plan_{base}{tag}_{}", forms.join("_")),
    };
    Source::named(body, entry)
}

fn add_norm_load(k: &AddLaneNorm, _: usize) -> Option<Load> {
    let body = add_norm_body(k);
    let elems = k.rows * k.width;
    let mut reads = vec![k.a_form.bytes(elems) as f64, k.b_form.bytes(elems) as f64];
    if k.add3 {
        reads.push(k.c_form.bytes(elems) as f64);
    }
    if k.weighted {
        reads.push(k.w_form.bytes(k.width) as f64);
    }
    let mut written = k.out_form.bytes(elems) as f64;
    if k.save_sum {
        written += k.sum_form.bytes(elems) as f64;
    }
    let mut flight = k.a_form.bytes((WIDE * AHEAD) as usize) as f64
        + k.b_form.bytes((WIDE * AHEAD) as usize) as f64;
    if k.add3 {
        flight += k.c_form.bytes((WIDE * AHEAD) as usize) as f64;
    }
    let per_row = if k.block {
        BLOCK_THREADS as usize
    } else {
        LANES
    };
    let trips = if cache_trips(k.width, k.block) > 0 {
        1.0
    } else {
        (k.width as f64 / (per_row * WIDE as usize) as f64).ceil()
    };
    Some(Load {
        threads: k.rows * per_row,
        group: BLOCK_THREADS as usize,
        trips,
        trip: body.trip(),
        reads,
        written,
        flight,
        ..Load::default()
    })
}

/// Full-length `Stage::Add` operands of `t`, if elementwise add.
fn add_operands(tree: &Tree, t: T, len: usize) -> Option<(T, T)> {
    let node = tree.node(t);
    let Term::Stage(Stage::Add) = &node.term else {
        return None;
    };
    let [a, b] = node.args.as_slice() else {
        return None;
    };
    let (a, b) = (*a, *b);
    (tree.elements(a) == len && tree.elements(b) == len).then_some((a, b))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    if !site.width.is_multiple_of(WIDE as usize) {
        return None;
    }
    let sum = site.x;
    let (a, b) = add_operands(tree, sum, site.len)?;
    let save_sum = tree.root(sum);
    let out_form = writable(tree.node(at).form);
    let sum_form = match save_sum {
        true => writable(tree.node(sum).form),
        false => F32_FORM,
    };
    let w_form = site
        .operand
        .map_or(F32_FORM, |(o, _)| plain(tree.node(o).form));
    let rows = site.len / site.width;
    let block = rows <= BLOCK_ROWS && site.width >= BLOCK_MIN;
    let make = |reads: Vec<T>, forms: [&'static dyn Stored; 3], add3: bool, choice: usize| {
        let kernel = AddLaneNorm {
            rows,
            width: site.width,
            eps,
            add3,
            weighted,
            save_sum,
            block,
            a_form: forms[0],
            b_form: forms[1],
            c_form: forms[2],
            w_form,
            out_form,
            sum_form,
        };
        let mut draft = Draft::new(kernel, reads, site.len).choice(choice);
        if save_sum {
            draft = draft.also(sum, site.len);
        }
        draft
    };
    let w = site.operand.map(|(o, _)| o);
    let reads2: Vec<T> = [a, b].into_iter().chain(w).collect();
    let forms2 = [plain(tree.node(a).form), plain(tree.node(b).form), F32_FORM];
    let mut drafts = vec![make(reads2, forms2, false, 0)];
    // Fuse a 3-way add (`x + (a + b)`) keeping order.
    let triple = (!tree.root(b))
        .then(|| add_operands(tree, b, site.len).map(|(u, v)| [u, v, a]))
        .flatten()
        .or_else(|| {
            (!tree.root(a))
                .then(|| add_operands(tree, a, site.len).map(|(u, v)| [u, v, b]))
                .flatten()
        });
    if let Some([x0, x1, x2]) = triple {
        let reads3: Vec<T> = [x0, x1, x2].into_iter().chain(w).collect();
        let forms3 = [
            plain(tree.node(x0).form),
            plain(tree.node(x1).form),
            plain(tree.node(x2).form),
        ];
        drafts.push(make(reads3, forms3, true, 1));
    }
    Some(drafts)
}

tessera_lower::rule!("add_rms_norm", propose);
