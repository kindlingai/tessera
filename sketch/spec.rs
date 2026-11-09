//! Speculative rounds: MTP and DFlash2 drafting, verify, commit. Sketch.

use crate::core::*;
use crate::glm53::{self, Glm53, Taps};
use crate::library::{fit_budget, greedy_accept, keep_confident, rejection_accept};

/// One extra block; embedding and head are the target's.
#[derive(Weights)]
pub struct Mtp {
    pub enorm: Norm,
    pub hnorm: Norm,
    pub proj: Linear,
    pub block: glm53::Block,
    pub embed: Shared<Embed>,
    pub head: Shared<Linear>,
}

/// Block-diffusion drafter fed by the target's tapped layers.
#[derive(Weights)]
pub struct Dflash {
    pub fuse: Linear,
    pub blocks: Vec<glm53::Block>,
    pub norm: Norm,
    pub head: Shared<Linear>,
}

pub enum Drafter {
    Mtp,
    Dflash,
}

/// Thresholds and costs: per-round inputs and tuning data.
pub struct DraftPolicy {
    pub mtp_len: usize,
    pub dflash_len: usize,
    pub mtp_cost: f32,
    pub dflash_cost: f32,
    pub margin: f32,
    pub min_dwell: usize,
    pub row_budget: Rows,
    pub confidence_floor: f32,
}

/// Per-sequence history the Accept step updates.
pub struct SeqStats {
    pub accept_mtp: Tensor,
    pub accept_dflash: Tensor,
    pub drafter: Tensor,
    pub dwell: Tensor,
}

pub struct Drafts {
    pub tokens: Tensor,
    pub logprobs: Tensor,
}

pub enum Rule {
    Greedy,
    Rejection,
}

/// Draft, verify, accept, commit: one captured graph.
pub fn round(
    r: &mut Model,
    t: &Glm53,
    mtp: &Mtp,
    dfl: &Dflash,
    p: &DraftPolicy,
    s: &SeqStats,
    rule: Rule,
) -> Counts {
    let plans = choose_drafter(r, s, p);
    let taps = r.taps_of_last_round();
    let a = draft_mtp(r, mtp, &taps, &plans.part(Drafter::Mtp), p);
    let b = draft_dflash(r, dfl, &taps, &plans.part(Drafter::Dflash), p);
    let drafts = a.join(&b);
    let rows = r.verify_rows(&drafts.tokens);
    let (logits, next_taps) = glm53::forward(r, &r.config(), t, &rows);
    let n = match rule {
        Rule::Greedy => greedy_accept(r, &logits, &drafts.tokens),
        Rule::Rejection => {
            rejection_accept(r, &logits.softmax(), &drafts.logprobs.exp(), &drafts.tokens)
        }
    };
    r.update_stats(s, &plans, &n);
    r.keep_taps(&next_taps, &n);
    r.commit_all(&n);
    n
}

/// A policy written in the description language, not a kernel.
fn choose_drafter(r: &mut Model, s: &SeqStats, p: &DraftPolicy) -> Plans<Drafter> {
    let dflash = s
        .accept_dflash
        .scale(p.dflash_len as f32)
        .sub_scalar(p.dflash_cost);
    let mtp = s.accept_mtp.scale(p.mtp_len as f32).sub_scalar(p.mtp_cost);
    let want = dflash
        .compare(Cmp::Gt, &mtp.add_scalar(p.margin))
        .choose(Drafter::Dflash, Drafter::Mtp);
    let stay = s.dwell.compare_scalar(Cmp::Lt, p.min_dwell as f32);
    let choice = stay.choose_tensor(&s.drafter, &want);
    let wanted = choice.map_len(p.mtp_len, p.dflash_len);
    let rows = fit_budget(r, &wanted, p.row_budget);
    r.partition_rows(&choice, &rows)
}

/// Chained drafts; each pass drops rows below the confidence floor.
fn draft_mtp(r: &mut Model, w: &Mtp, taps: &Taps, rows: &Plan, p: &DraftPolicy) -> Drafts {
    let mut live = rows.clone();
    let mut h = taps.last.clone();
    let mut tok = r.last_tokens(&live);
    let mut out = Drafts::empty();
    for _ in 0..p.mtp_len {
        let e = r.embed(&tok, &w.embed).rms_norm(&w.enorm);
        let x = e.concat(&h.rms_norm(&w.hnorm)).matmul(&w.proj);
        h = r.with_rows(&live, |r| r.block(&w.block, &x));
        let logits = h.matmul(&w.head);
        tok = r.sample(&logits, Sampler::Argmax);
        out.push(&tok, &logits.log_softmax_at(&tok));
        live = keep_confident(r, &logits.softmax().max_per_row(), p.confidence_floor);
    }
    out
}

/// One pass drafts a whole block from the fused taps.
fn draft_dflash(r: &mut Model, w: &Dflash, taps: &Taps, rows: &Plan, p: &DraftPolicy) -> Drafts {
    let ctx = r.dflash_context(Sync::Eager);
    ctx.append(&r.concat_all(&taps.layers).matmul(&w.fuse));
    let mut x = r.mask_block(rows, p.dflash_len);
    for b in &w.blocks {
        x = r.block_over(
            b,
            &x,
            &ctx,
            Mask::BlockCausal {
                block: p.dflash_len,
            },
        );
    }
    let logits = x.rms_norm(&w.norm).matmul(&w.head);
    let tok = r.sample(&logits, Sampler::Argmax);
    Drafts {
        logprobs: logits.log_softmax_at(&tok),
        tokens: tok,
    }
}
