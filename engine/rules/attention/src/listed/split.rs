//! Split-K decode: many groups per list, then a merge.

use super::*;
use tessera_lower::form::F32_FORM;
use tessera_lower::{Backend, GUESS};

/// The split is tuned for Metal's many small cores.
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];

/// More rows go to the tiled kernel.
const MAX_ROWS: usize = 16;

/// Fewer keys leave each group too little work.
const MIN_KEYS: usize = 256;

/// Groups sharing one query row and head.
const SPLITS: u32 = 16;

/// SIMD groups in each of those groups.
const PART_WARPS: u32 = 4;

knob!("listed_split", "splits", 2, 64);
knob!("listed_split", "warps", 4, 8);

/// The fused kernel's cost curve, whole or split.
pub(super) fn name(splits: u32) -> &'static str {
    match splits {
        1 => "listed_attend",
        _ => "listed_split",
    }
}

/// Seconds per multiply-add, until measured.
pub(super) fn guess(splits: u32) -> f64 {
    match splits {
        1 => 1e-10,
        _ => GUESS,
    }
}

/// Backends a kernel of `splits` leaves out.
pub(super) fn lacks(splits: u32) -> &'static [Backend] {
    match splits {
        1 => &[],
        _ => LACKS,
    }
}

/// One thread per output: merges a list's parts.
#[kernel]
fn listed_merge(
    parts: Buf<F32>,
    sinks: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    heads: u32,
    width: u32,
    splits: u32,
    sink_mode: u32,
) {
    if gid() >= len {
        return;
    }
    let d = gid() % width;
    let list = gid() / width;
    let sink_at = if sink_mode == 1 { list % heads } else { list };
    let mut best = f32::NEG_INFINITY;
    if sink_mode > 0 {
        best = sinks[sink_at as u64];
    }
    let mut denom = 0.0f32;
    let mut sum = 0.0f32;
    // A dead part that saw keys holds a NaN.
    let mut hole = 0.0f32;
    let stride = (width + 2) as u64;
    let at = list as u64 * splits as u64 * stride;
    // Only parts that saw keys carry a sum.
    for s in 0..splits {
        let p = at + s as u64 * stride;
        if (parts[p + 1].to_bits() & 0x7fff_ffff) != 0 {
            best = max(best, parts[p]);
        } else {
            hole += parts[p];
        }
    }
    if sink_mode > 0 && sinks[sink_at as u64].to_bits() != 0xff80_0000 {
        denom = exp(sinks[sink_at as u64] - best);
    }
    for s in 0..splits {
        let p = at + s as u64 * stride;
        if (parts[p + 1].to_bits() & 0x7fff_ffff) != 0 {
            denom += parts[p + 1] * exp(parts[p] - best);
        }
    }
    let inv = if (denom.to_bits() & 0x7fff_ffff) != 0 {
        1.0 / denom
    } else {
        hole
    };
    // Each part's scaled mean joins by its share.
    for s in 0..splits {
        let p = at + s as u64 * stride;
        // A dead part weighs zero, yet passes on a NaN.
        let live = (parts[p + 1].to_bits() & 0x7fff_ffff) != 0;
        let share = if live { exp(parts[p] - best) } else { 1.0 };
        sum += parts[p + 2 + d as u64] * ((parts[p + 1] * share) * inv);
    }
    // A finite mean stays finite at full scale.
    let r = sum * 4.0;
    let finite = (sum.to_bits() & 0x7f80_0000) != 0x7f80_0000;
    out[gid() as u64] = if finite {
        clamp(r, f32::MIN, f32::MAX)
    } else {
        r
    };
}

/// Each list's parts merged, with its sink.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed_merge,
    name = "listed_merge",
    sizes = [self.0.len, self.1 as usize],
    work = self.0.len * self.1 as usize,
    guess = GUESS,
    accepts = merge_accepts,
    launches = merge_launches,
)]
pub(crate) struct ListedMerge(pub Sizes, pub u32);

fn merge_accepts(k: &ListedMerge, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 if k.0.sink_mode == 0 => true,
        _ => !form.narrow(),
    }
}

fn merge_launches(k: &ListedMerge, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let counts = [s.len, s.heads, s.width, k.1 as usize, s.sink_mode];
    vec![Launch::plain(step, &counts, Shape::Line(s.len))]
}

/// Checks the rows, keys, forms and head width.
fn check(s: &Sizes) -> Result<(), String> {
    if s.from_list {
        return Err("needs every key, not a list".into());
    }
    let rows = s.lists() / s.heads;
    if rows >= MAX_ROWS {
        return Err(format!("{rows} rows are not fewer than {MAX_ROWS}"));
    }
    if s.keys.most() < MIN_KEYS {
        return Err(format!("{} keys are under {MIN_KEYS}", s.keys.most()));
    }
    let forms = [s.q_form, s.k_form, s.v_form, s.out_form];
    if forms.iter().any(|f| f.name() != F32_FORM.name()) {
        return Err("needs f32 queries, keys, values and output".into());
    }
    s.check_fused()
}

/// Parts of each list in their own groups, then merged.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (sizes, reads) = site(tree, at)?;
    check(&sizes).ok()?;
    let knob = |name, or: u32| {
        tree.knobs
            .get("listed_split", name)
            .map_or(or, |v| v as u32)
    };
    let (splits, warps) = (knob("splits", SPLITS), knob("warps", PART_WARPS));
    let parts = sizes.lists() * splits as usize * (sizes.width + 2);
    let merge = [Read::Draft(0), Read::Node(reads[6])];
    Some(vec![
        Draft::new(ListedAttend(sizes, warps, splits), reads, parts),
        Draft::new(ListedMerge(sizes, splits), merge, sizes.len),
    ])
}

tessera_lower::rule!("listed_split", propose);
