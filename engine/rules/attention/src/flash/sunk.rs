//! Tiled ragged attention with sinks, bias and a window.

use super::*;
use crate::listed::{self, Sizes};
use tessera_codegen::kept;
use tessera_graph::DimName;
use tessera_lower::form::F32_FORM;
use tessera_lower::{Backend, Shape};

/// The tiles are tuned for Metal's matrix unit.
pub(super) const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];

/// Seconds per multiply-add, until measured.
pub(super) const GUESS: f64 = 1e-12;

/// Fewer rows go to the split-K kernel.
const MIN_ROWS: usize = 16;

/// Keys per block.
const BK: u32 = WAYS[0].0;

thread_local! {
    /// Each kernel tree, by head width.
    static SUNK: RefCell<HashMap<u32, Body>> = RefCell::new(HashMap::new());
}

/// Writes biased scores to `p_s`. Unseen keys get -inf.
#[kernel]
fn scores_near(
    st: &mut Stream,
    pos_s: &Shared<u32>,
    rel_bias: Buf<F32>,
    hd: &Head,
    simd: u32,
    k0: u32,
    scale: f32,
    rel_win: u32,
    #[build] bk: u32,
) {
    if simd < bk / 8 {
        st.s_t.each(|r, c, v| {
            let col = (simd / 2) * 16 + c;
            let t = k0 + col;
            let row = (simd % 2) * 16 + r;
            let p = pos_s[row];
            let seen = t >= st.from_s[row] && t < st.to_s[row];
            let near = seen && rel_win > 0 && t <= p && p - t < rel_win;
            let mut s = if seen { v * scale } else { f32::NEG_INFINITY };
            if near {
                let list = ((hd.q0 + row) * hd.heads + hd.head) as u64;
                s = s + rel_bias[list * rel_win as u64 + (p - t) as u64];
            }
            st.p_s[row * (bk + PAD) + col] = s;
        });
    }
}

/// Adds each row's sink to its softmax sum.
#[kernel]
fn sink_rows(st: &mut Stream, sinks: Buf<F32>, hd: &Head, sink_mode: u32) {
    let simd = lid() / LANES as u32;
    for i in 0..ROWS {
        let r = hd.q0 + simd * ROWS + i;
        if sink_mode > 0 && r < hd.queries {
            let list = r * hd.heads + hd.head;
            let at = if sink_mode == 1 { hd.head } else { list };
            let sink = sinks[at as u64];
            if sink > f32::NEG_INFINITY {
                let sum = st.state[(ROWS + i) as usize];
                st.state[(ROWS + i) as usize] = sum + exp(sink - st.state[i as usize]);
            }
        }
    }
}

/// The key blocks a pass reads.
#[derive(Group)]
struct Span {
    lo: u32,
    hi: u32,
}

/// The slot `pass` serves, or `u32::MAX` for none.
#[kernel]
fn lead_slot(slots: Buf<U32>, hd: &Head, pass: u32) -> u32 {
    let lead = hd.q0 + pass;
    let mut slot = u32::MAX;
    if lead < hd.queries {
        slot = slots[lead as u64];
        // An earlier row's slot was served on its pass.
        for j in 0..pass {
            if slots[(hd.q0 + j) as u64] == slot {
                slot = u32::MAX;
            }
        }
    }
    slot
}

/// Sets the keys each row of `slot` sees.
#[kernel]
fn see_rows(
    st: &mut Stream,
    pos_s: &mut Shared<u32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    hd: &Head,
    slot: u32,
    keys: u32,
    before: u32,
    after: u32,
) {
    if lid() < BQ {
        let r = hd.q0 + lid();
        let mine = r < hd.queries && slots[r as u64] == slot;
        let p = if mine { positions[r as u64] } else { 0 };
        let from = if p > before { p - before } else { 0 };
        let to = if p < keys && keys - p - 1 > after {
            p + after + 1
        } else {
            keys
        };
        pos_s[lid()] = p;
        st.from_s[lid()] = if mine { from } else { 0 };
        st.to_s[lid()] = if mine { to } else { 0 };
    }
}

/// The keys any row of the pass sees.
#[kernel]
fn span_of(st: &Stream, keys: u32) -> Span {
    let mut lo = keys;
    let mut hi = 0u32;
    for r in 0..BQ {
        if st.to_s[r] > st.from_s[r] {
            lo = if st.from_s[r] < lo { st.from_s[r] } else { lo };
            hi = if st.to_s[r] > hi { st.to_s[r] } else { hi };
        }
    }
    Span { lo, hi }
}

/// Starts a pass: no peaks, no sums.
#[kernel]
fn clear(st: &mut Stream) {
    restart(st);
    st.o00.zero();
    st.o01.zero();
    st.o10.zero();
    st.o11.zero();
}

/// Ragged rows in a window, with sinks and bias.
#[kernel]
fn flash_sunk(
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] values: Buf<Dyn>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    sinks: Buf<F32>,
    rel_bias: Buf<F32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    scale: f32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    before: u32,
    after: u32,
    sink_mode: u32,
    rel_win: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let hd = head_at(gid() / THREADS, queries, heads, kv_heads);
    let mut st = stream(w, bk, held_q);
    let mut pos_s = shared::<u32>(BQ);
    stage_q(&mut st.q_s, q, &hd, 0, w, w);
    for pass in 0..BQ {
        let slot = lead_slot(slots, &hd, pass);
        if slot != u32::MAX {
            see_rows(
                &mut st, &mut pos_s, slots, positions, &hd, slot, keys, before, after,
            );
            barrier();
            let span = span_of(&st, keys);
            clear(&mut st);
            for b in span.lo / bk..(span.hi + bk - 1) / bk {
                let k0 = b * bk;
                let at = kept_row(slot, k0);
                let simd = lid() / LANES as u32;
                stage_kv::<false>(&mut st.kv_s, k, &hd, at, k0, span.hi, 0, w, w, bk, k_form);
                barrier();
                st.s_t.zero();
                score_chunk(&mut st, simd, w, bk, held_q);
                scores_near(&mut st, &pos_s, rel_bias, &hd, simd, k0, scale, rel_win, bk);
                barrier();
                fold_block(
                    &mut st, values, &hd, at, k0, span.hi, 0, w, w, bk, held_q, v_form,
                );
            }
            sink_rows(&mut st, sinks, &hd, sink_mode);
            write_all(&mut st, out, &hd, 0, w, w, out_form);
            barrier();
        }
    }
}

/// Tiled ragged rows: scale, sinks, bias, a window.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "flash_sunk",
    order = Order::PerConfig,
    sizes = [self.0.len, self.0.keys.most(), self.seen(), self.1],
    work = self.0.len * self.seen() * self.1,
    guess = GUESS,
    needs = sunk_needs(&self.0),
    accepts = sunk_accepts,
    emits = sunk_emits,
    source = sunk_source,
    launches = sunk_launches,
)]
pub(crate) struct FlashSunk(pub Sizes, pub usize);

impl FlashSunk {
    /// The most keys one query sees.
    pub(super) fn seen(&self) -> usize {
        let (keys, (back, ahead)) = (self.0.keys.most(), self.0.window);
        keys.min(back.saturating_add(ahead).saturating_add(1))
    }

    /// A group per head and block of rows.
    fn groups(&self) -> usize {
        let s = &self.0;
        s.heads * (s.lists() / s.heads).div_ceil(BQ as usize)
    }
}

pub(super) fn sunk_accepts(k: &FlashSunk, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.0.k_form.name(),
        2 => form.name() == k.0.v_form.name(),
        5 if k.0.sink_mode == 0 => true,
        6 if k.0.rel_win == 0 => true,
        _ => !form.narrow(),
    }
}

pub(super) fn sunk_emits(k: &FlashSunk, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

fn sunk_source(k: &FlashSunk) -> Source {
    let s = &k.0;
    let w = s.width as u32;
    let make = || flash_sunk_ir(w, BK, false, s.k_form, s.v_form, s.out_form);
    let tree = SUNK.with(|t| t.borrow_mut().entry(w).or_insert_with(make).clone());
    let entry = format!("flash_sunk_{w}_{}", s.kept.name());
    Source::named(kept::attach(tree, s.kept), entry)
}

/// Seven reads, the output, any table, then scalars.
pub(super) fn sunk_launches(k: &FlashSunk, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let rows = s.lists() / s.heads;
    let reads = (0..step.reads.len()).map(|i| step.read(i));
    let mut args: Vec<Arg> = reads.chain([step.out()]).collect();
    args.extend(step.kept_for(s.kept, true));
    args.push(Arg::value(&s.scale));
    args.push(Arg::count(rows));
    args.push(Arg::size(s.keys));
    let (back, ahead) = s.window;
    let counts = [s.heads, s.kv_heads, back, ahead, s.sink_mode, s.rel_win];
    args.extend(counts.into_iter().map(Arg::count));
    let kept = s.kept.values(s.seqs, s.context);
    args.extend(kept.into_iter().map(|v| Arg::count(v as usize)));
    vec![Launch {
        args,
        shape: Shape::Grid {
            across: k.groups(),
            down: 1,
            threads: THREADS as usize,
        },
    }]
}

/// The stream, and each row's position.
fn sunk_needs(s: &Sizes) -> Needs {
    let floats = stream_floats(s.width, BK, false) + BQ as usize;
    Needs {
        lanes: LANES,
        group_bytes: 4 * floats,
        group_threads: THREADS as usize,
        tile: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

/// Checks the rows, head width, forms and kept layout.
pub(super) fn check_sunk(s: &Sizes, bk: u32) -> Result<(), String> {
    if !s.ragged || s.from_list {
        return Err("needs ragged rows over every key".into());
    }
    let (w, rows) = (s.width, s.lists() / s.heads);
    if !w.is_multiple_of(16) || w > CW as usize {
        return Err(format!("head width {w} is not 16 to {CW} by 16"));
    }
    if rows < MIN_ROWS {
        return Err(format!("{rows} rows are fewer than {MIN_ROWS}"));
    }
    let forms = [s.q_form, s.k_form, s.v_form, s.out_form];
    if forms.iter().any(|f| f.name() != F32_FORM.name()) {
        return Err("needs f32 queries, keys, values and output".into());
    }
    // A ring of whole blocks keeps each block unbroken.
    let run = match s.kept.name() {
        "ring" => s.context,
        _ => s.kept.run(s.context),
    };
    if !run.is_multiple_of(bk as usize) {
        return Err(format!("kept run {run} is not a multiple of {bk}"));
    }
    Ok(())
}

/// The kernel and reads of a site that `fits`.
pub(super) fn sited(
    tree: &Tree,
    at: T,
    fits: fn(&Sizes) -> bool,
) -> Option<(FlashSunk, [Read; 7])> {
    let (sizes, [q, k, v, _, slots, positions, sinks, rel]) = listed::site(tree, at)?;
    if !fits(&sizes) {
        return None;
    }
    let rows = sizes.lists() / sizes.heads;
    // Slots the rows span, if the launch says. Else any.
    let segments = tree.size(DimName::Segments).map_or(rows, |n| n as usize);
    let passes = segments.clamp(1, rows.min(BQ as usize));
    let reads = [q, k, v, slots, positions, sinks, rel].map(Read::Node);
    Some((FlashSunk(sizes, passes), reads))
}

/// One launch scores, softens and folds by tiles.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (sunk, reads) = sited(tree, at, |s| check_sunk(s, BK).is_ok())?;
    let mut ways = vec![Draft::new(sunk, reads, sunk.0.len).choice(0)];
    if super::sunk_metal::fits(&sunk.0) {
        ways.push(super::sunk_metal::draft(sunk, reads).choice(1));
    }
    Some(ways)
}

fn why_not(tree: &Tree, at: T) -> Option<String> {
    let (sizes, _) = listed::site(tree, at)?;
    check_sunk(&sizes, BK).err()
}

tessera_lower::rule!("flash_sunk", propose, why = why_not, lacks = LACKS);
