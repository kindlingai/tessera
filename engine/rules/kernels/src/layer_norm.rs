//! Fused single-pass layer norm with optional residual add and bias.

use crate::WIDE;
use crate::add_rms_norm::add_operands;
use crate::lane_norm::{
    RowCfg, RowSpec, row_body, row_launches, row_load, row_needs, row_shape, row_tag,
};
use tessera_graph::{Order, Stage};
use tessera_lower::form::{F32_FORM, Load as Stored, plain, writable};
use tessera_lower::{Draft, Kernel, Launch, Load, Source, Step, T, Term, Tree};

/// Fused center, RMS norm, optional residual add and bias.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = ln_name(self),
    launches = ln_launches,
    shape = row_shape(self.rows, self.block, self.warps),
    work = self.rows * self.width,
    needs = row_needs(self.block, self.warps),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    accepts = ln_accepts,
    emits = ln_emits,
    emits_at = ln_emits_at,
    source = ln_source,
    load = ln_load,
    unproven,
)]
pub(crate) struct LayerNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub add2: bool,
    pub add_bias: bool,
    pub weighted: bool,
    pub biased: bool,
    pub save_sum: bool,
    pub block: bool,
    pub warps: u32,
    pub cache_trips: u32,
    pub ahead: u32,
    pub a_form: &'static dyn Stored,
    pub b_form: &'static dyn Stored,
    pub in_bias_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub bias_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub sum_form: &'static dyn Stored,
}

fn ln_name(k: &LayerNorm) -> &'static str {
    match (k.block, k.add2, k.biased, k.save_sum) {
        (true, false, false, _) => "block_layer_norm",
        (true, false, true, _) => "block_layer_bias",
        (true, true, false, false) => "block_add_ln",
        (true, true, false, true) => "block_add_ln_save",
        (true, true, true, false) => "block_add_ln_bias",
        (true, true, true, true) => "block_add_ln_bsave",
        (false, false, false, _) => "lane_layer_norm",
        (false, false, true, _) => "lane_layer_bias",
        (false, true, false, false) => "lane_add_ln",
        (false, true, false, true) => "lane_add_ln_save",
        (false, true, true, false) => "lane_add_ln_bias",
        (false, true, true, true) => "lane_add_ln_bsave",
    }
}

fn ln_accepts(k: &LayerNorm, input: usize, form: &dyn Stored) -> bool {
    let mut idx = 0;
    if input == idx {
        return form.name() == k.a_form.name();
    }
    idx += 1;
    if k.add2 {
        if input == idx {
            return form.name() == k.b_form.name();
        }
        idx += 1;
    }
    if k.add_bias {
        if input == idx {
            return form.name() == k.in_bias_form.name();
        }
        idx += 1;
    }
    if k.weighted {
        if input == idx {
            return form.name() == k.w_form.name();
        }
        idx += 1;
    }
    if k.biased && input == idx {
        return form.name() == k.bias_form.name();
    }
    !form.narrow()
}

fn ln_emits(k: &LayerNorm, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn ln_emits_at(k: &LayerNorm, output: usize, form: &dyn Stored) -> bool {
    match output {
        0 => form.name() == k.out_form.name(),
        1 if k.save_sum => form.name() == k.sum_form.name(),
        _ => !form.narrow(),
    }
}

fn ln_spec(k: &LayerNorm) -> RowSpec {
    let reload_sum = k.save_sum && k.sum_form.name() == F32_FORM.name();
    RowSpec {
        b_form: k.b_form,
        in_bias_form: k.in_bias_form,
        bias_form: k.bias_form,
        sum_form: k.sum_form,
        add2: k.add2,
        add_bias: k.add_bias,
        center: true,
        biased: k.biased,
        save_sum: k.save_sum,
        reload_sum,
        ..RowSpec::base(
            k.width,
            k.block,
            k.warps,
            k.cache_trips,
            k.ahead,
            k.a_form,
            k.w_form,
            k.out_form,
            k.weighted,
        )
    }
}

fn ln_body(k: &LayerNorm) -> tessera_codegen::ir::Body {
    row_body(ln_spec(k))
}

fn ln_launches(k: &LayerNorm, step: &Step) -> Vec<Launch> {
    let body = ln_body(k);
    row_launches(&body, step, k.rows, k.width, k.eps, k.block, k.warps)
}

fn ln_source(k: &LayerNorm) -> Source {
    let body = ln_body(k);
    let base = k.name();
    let tag = row_tag(k.width, k.block, k.warps, k.cache_trips, k.ahead);
    let w_tag = if k.weighted { "" } else { "_nw" };
    let b_tag = if k.add_bias { "_ib" } else { "" };
    let mut forms = vec![k.a_form.name()];
    if k.add2 {
        forms.push(k.b_form.name());
    }
    if k.add_bias {
        forms.push(k.in_bias_form.name());
    }
    if k.weighted {
        forms.push(k.w_form.name());
    }
    if k.biased {
        forms.push(k.bias_form.name());
    }
    forms.push(k.out_form.name());
    if k.save_sum {
        forms.push(k.sum_form.name());
    }
    let entry = match forms.iter().all(|&f| f == "f32") {
        true => format!("plan_{base}{w_tag}{b_tag}{tag}"),
        false => format!("plan_{base}{w_tag}{b_tag}{tag}_{}", forms.join("_")),
    };
    Source::named(body, entry)
}

fn ln_load(k: &LayerNorm, _: usize) -> Option<Load> {
    row_load(&ln_spec(k), k.rows, 3)
}

/// Matches `center -> rms_norm/rms` at `norm_t`, returning `(x, weight, eps)`.
fn match_norm(tree: &Tree, norm_t: T, len: usize, width: usize) -> Option<(T, Option<T>, f32)> {
    let norm = tree.node(norm_t);
    if norm.shape.last() != Some(&width) {
        return None;
    }
    let (center_t, weight, eps) = match norm.term {
        Term::Stage(Stage::RmsNorm { eps }) => (norm.args[0], Some(norm.args[1]), eps),
        Term::Stage(Stage::Rms { eps }) => (norm.args[0], None, eps),
        _ => return None,
    };
    if tree.root(center_t) {
        return None;
    }
    let center = tree.node(center_t);
    if center.term != Term::Stage(Stage::Center) || center.shape.last() != Some(&width) {
        return None;
    }
    let x = center.args[0];
    if tree.elements(x) != len || tree.node(x).shape.last() != Some(&width) {
        return None;
    }
    Some((x, weight, eps))
}

/// Unwraps optional trailing 1D bias `Stage::Add` at `at`.
fn match_bias(tree: &Tree, at: T, len: usize, width: usize) -> Option<(T, Option<T>)> {
    let node = tree.node(at);
    if node.term == Term::Stage(Stage::Add) {
        let [lhs, rhs] = node.args.as_slice() else {
            return None;
        };
        let (norm_t, bias_t) = if tree.elements(*lhs) == len && tree.elements(*rhs) == width {
            (*lhs, *rhs)
        } else if tree.elements(*rhs) == len && tree.elements(*lhs) == width {
            (*rhs, *lhs)
        } else {
            return None;
        };
        if tree.root(norm_t) {
            return None;
        }
        return Some((norm_t, Some(bias_t)));
    }
    Some((at, None))
}

/// Matches an unshared 1D bias add `(tensor, bias)` at `t`.
fn in_bias_pair(tree: &Tree, t: T, len: usize, width: usize) -> Option<(T, T)> {
    if tree.root(t) || len == width {
        return None;
    }
    let node = tree.node(t);
    if node.term != Term::Stage(Stage::Add) {
        return None;
    }
    let [lhs, rhs] = node.args.as_slice() else {
        return None;
    };
    if tree.elements(*lhs) == len && tree.elements(*rhs) == width {
        Some((*lhs, *rhs))
    } else if tree.elements(*rhs) == len && tree.elements(*lhs) == width {
        Some((*rhs, *lhs))
    } else {
        None
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let len = tree.elements(at);
    let width = *tree.node(at).shape.last()?;
    if width == 0 || !width.is_multiple_of(WIDE as usize) || !len.is_multiple_of(width) {
        return None;
    }
    let (norm_t, bias) = match_bias(tree, at, len, width)?;
    let (x, weight, eps) = match_norm(tree, norm_t, len, width)?;
    let rows = len / width;
    let cfg = RowCfg::of(&tree.knobs, "layer_norm");
    let block = cfg.use_block(rows, width);
    let warps = cfg.warps(width);
    let out_form = writable(tree.node(at).form);
    let w_form = weight.map_or(F32_FORM, |w| plain(tree.node(w).form));
    let bias_form = bias.map_or(F32_FORM, |b| plain(tree.node(b).form));
    let build = |inputs: &[T],
                 add2: bool,
                 add_bias: bool,
                 save_sum: bool,
                 sum_t: Option<T>,
                 choice: usize| {
        let a_form = plain(tree.node(inputs[0]).form);
        let b_form = if add2 {
            plain(tree.node(inputs[1]).form)
        } else {
            F32_FORM
        };
        let in_bias_form = if add_bias {
            plain(tree.node(inputs[2]).form)
        } else {
            F32_FORM
        };
        let sum_form = match (save_sum, sum_t) {
            (true, Some(s)) => writable(tree.node(s).form),
            _ => F32_FORM,
        };
        let kernel = LayerNorm {
            rows,
            width,
            eps,
            add2,
            add_bias,
            weighted: weight.is_some(),
            biased: bias.is_some(),
            save_sum,
            block,
            warps,
            cache_trips: cfg.cache_trips,
            ahead: cfg.ahead,
            a_form,
            b_form,
            in_bias_form,
            w_form,
            bias_form,
            out_form,
            sum_form,
        };
        let mut reads: Vec<T> = inputs.to_vec();
        reads.extend(weight);
        reads.extend(bias);
        let mut draft = Draft::new(kernel, reads, len).choice(choice);
        if let (true, Some(s)) = (save_sum, sum_t) {
            draft = draft.also(s, len);
        }
        draft
    };
    let mut drafts = vec![build(&[x], false, false, false, None, 0)];
    if let Some((a, b)) = add_operands(tree, x, len) {
        let save_sum = tree.root(x);
        drafts.push(build(&[a, b], true, false, save_sum, Some(x), 1));
        let biased_add = in_bias_pair(tree, b, len, width)
            .map(|(raw, ib)| [a, raw, ib])
            .or_else(|| in_bias_pair(tree, a, len, width).map(|(raw, ib)| [b, raw, ib]));
        if let Some(triple) = biased_add {
            drafts.push(build(&triple, true, true, save_sum, Some(x), 2));
        }
    }
    Some(drafts)
}

tessera_lower::rule!("layer_norm", propose);
