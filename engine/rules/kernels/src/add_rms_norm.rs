//! Fused residual add and RMS norm, per warp or block.

use crate::lane_norm::{
    RowSpec, block_threads, cache_trips, row_body, row_launches, row_needs, row_shape, row_tag,
    use_block,
};
use crate::{AHEAD, WIDE};
use tessera_codegen::prelude::*;
use tessera_graph::{Order, Stage};
use tessera_lower::form::{F32_FORM, Load as Stored, plain, writable};
use tessera_lower::{Draft, Kernel, Launch, Load, Source, Step, T, Term, Tree, sites};

/// Fused add + RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = add_norm_name(self),
    launches = add_norm_launches,
    shape = row_shape(self.rows, self.width, self.block),
    work = self.rows * self.width,
    needs = row_needs(self.width, self.block),
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
    row_body(RowSpec {
        width: k.width,
        block: k.block,
        a_form: k.a_form,
        b_form: k.b_form,
        c_form: k.c_form,
        w_form: k.w_form,
        out_form: k.out_form,
        sum_form: k.sum_form,
        add2: true,
        add3: k.add3,
        weighted: k.weighted,
        keep_w: k.weighted,
        save_sum: k.save_sum,
        reload_sum: k.reload_sum(),
    })
}

fn add_norm_launches(k: &AddLaneNorm, step: &Step) -> Vec<Launch> {
    let body = add_norm_body(k);
    row_launches(&body, step, k.rows, k.width, k.eps, k.block)
}

fn add_norm_source(k: &AddLaneNorm) -> Source {
    let body = add_norm_body(k);
    let base = k.name();
    let tag = row_tag(k.width, k.block);
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
    let threads = block_threads(k.width) as usize;
    let per_row = if k.block { threads } else { LANES };
    let trips = if cache_trips(k.width, k.block) > 0 {
        1.0
    } else {
        (k.width as f64 / (per_row * WIDE as usize) as f64).ceil()
    };
    Some(Load {
        threads: k.rows * per_row,
        group: threads,
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
    let block = use_block(rows, site.width);
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
