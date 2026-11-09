//! Speculative rounds: MTP, DFlash2 and DSpine drafts, verify, commit. Sketch.

use crate::core::*;
use crate::glm53::{self, Glm53};
use crate::library::{fit_budget, greedy_accept, rejection_accept};

/// A drafter names the target sites it reads.
pub trait Subscriber {
    fn taps(&self) -> Vec<(Site, TapRequest)>;
}

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

/// Which target layers a block drafter fuses.
pub struct TapLayers {
    pub layers: Vec<Layer>,
    pub window: Option<Rows>,
    pub context: ResourceId,
}

/// Block-diffusion drafter fed by fused target layers.
#[derive(Weights)]
pub struct Dflash {
    pub fuse: Linear,
    pub blocks: Vec<glm53::Block>,
    pub norm: Norm,
    pub head: Shared<Linear>,
}

/// Shape guessed from the config and tensor names only.
pub struct DspineShape {
    pub block: usize,
    pub head_topk: usize,
    pub first_exact: usize,
}

/// Each layer predicts the token before each position.
#[derive(Weights)]
pub struct CodeBridge {
    pub codebook: Embed,
    pub read: Linear,
    pub early_read: Vec<Linear>,
    pub write: Vec<Linear>,
    pub gate: Linear,
    pub prev: Linear,
    pub predecessor_gate: Linear,
    pub out: Linear,
}

/// A DFlash-family drafter: sliding window, block 16, code bridge.
#[derive(Weights)]
pub struct Dspine {
    pub fuse: Linear,
    pub hidden_norm: Norm,
    pub blocks: Vec<glm53::Block>,
    pub norm: Norm,
    pub bridge: CodeBridge,
    pub embed: Shared<Embed>,
    pub head: Shared<Linear>,
}

pub struct Drafters {
    pub mtp: (Mtp, ResourceId),
    pub dflash: (Dflash, TapLayers),
    pub dspine: (Dspine, TapLayers, DspineShape),
}

impl Subscriber for Drafters {
    fn taps(&self) -> Vec<(Site, TapRequest)> {
        let mut out = vec![(Site::PreHead, request(self.mtp.1, Some(Rows(1))))];
        for t in [&self.dflash.1, &self.dspine.1] {
            for l in &t.layers {
                out.push((Site::Residual(*l), request(t.context, t.window)));
            }
        }
        out
    }
}

fn request(to: ResourceId, window: Option<Rows>) -> TapRequest {
    TapRequest {
        to,
        rows: TapRows::Accepted,
        window,
        format: Format::Bf16,
    }
}

/// Wires every drafter's taps into the target's sites, once.
pub fn subscribe(m: &mut Model, d: &Drafters) {
    for (site, req) in d.taps() {
        m.hook(site, Hook::Tap(req));
    }
}

pub enum Drafter {
    Mtp,
    Dflash,
    Dspine,
}

/// Lengths, costs and limits: per-round inputs and tuning data.
pub struct DraftPolicy {
    pub max_rounds: usize,
    pub len: PerDrafter<usize>,
    pub cost: PerDrafter<f32>,
    pub min_dwell: usize,
    pub row_budget: Rows,
    pub confidence_floor: f32,
}

/// Acceptance per drafter, current drafter and dwell, per sequence.
pub struct SeqStats {
    pub accept: Tensor,
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

/// Rounds until each row stops; accepted tokens stream out.
pub fn decode(
    m: &mut Model,
    t: &Glm53,
    d: &Drafters,
    p: &DraftPolicy,
    s: &SeqStats,
    rule: Rule,
    left: Tensor,
    out: &Port,
) -> Tensor {
    // Every round emits a token, so one left means last.
    let last = |left: &Tensor, _: &Iter| left.compare_scalar(Cmp::Lt, 2.0);
    let until = Until {
        max: p.max_rounds,
        last: Box::new(last),
    };
    m.repeat(until, left, |m, left, it| {
        let n = round(m, t, d, p, s, &rule, it, &left);
        let tokens = m.accepted_tokens(&n);
        m.emit(out, &tokens);
        let left = left.sub(&n.add_one());
        let more = left.compare_scalar(Cmp::Gt, 0.0).and(&tokens.no_stop());
        Next { carry: left, more }
    })
}

/// Draft, verify, accept, commit: one loop iteration.
pub fn round(
    r: &mut Model,
    t: &Glm53,
    d: &Drafters,
    p: &DraftPolicy,
    s: &SeqStats,
    rule: &Rule,
    it: &Iter,
    left: &Tensor,
) -> Counts {
    // A row's last round drafts nothing; one verify row suffices.
    let drafting = it.live.minus(&r.filter(&it.last));
    let plans = choose_drafter(r, s, p, &drafting, &left.sub_scalar(1.0));
    let a = draft_mtp(r, &d.mtp.0, d.mtp.1, &plans.part(Drafter::Mtp), p);
    let b = draft_dflash(r, &d.dflash.0, &d.dflash.1, &plans.part(Drafter::Dflash), p);
    let (w, taps, shape) = &d.dspine;
    let c = draft_dspine(r, w, taps, shape, &plans.part(Drafter::Dspine));
    let drafts = a.join(&b).join(&c);
    // Verify taps every row; commit keeps only the accepted ones.
    let logits = glm53::forward(r, &r.config(), t, &r.verify_rows(&drafts.tokens));
    let n = match rule {
        Rule::Greedy => greedy_accept(r, &logits, &drafts.tokens),
        Rule::Rejection => {
            rejection_accept(r, &logits.softmax(), &drafts.logprobs.exp(), &drafts.tokens)
        }
    };
    r.update_stats(s, &plans, &n);
    r.commit_all(&n);
    // No later round drafts from a last row's taps.
    r.drop_taps(&it.last);
    n
}

/// Best expected gain per sequence, held for a minimum dwell.
fn choose_drafter(
    r: &mut Model,
    s: &SeqStats,
    p: &DraftPolicy,
    rows: &Plan,
    max_len: &Tensor,
) -> Plans<Drafter> {
    let gains = s
        .accept
        .mul_cols(&p.len.as_tensor())
        .sub_cols(&p.cost.as_tensor());
    let want = r.sample(&gains, Sampler::Argmax);
    let stay = s.dwell.compare_scalar(Cmp::Lt, p.min_dwell as f32);
    let choice = stay.choose_tensor(&s.drafter, &want);
    let wanted = choice.map_len(&p.len).min(max_len);
    let lens = fit_budget(r, &wanted, p.row_budget);
    r.partition_rows(&choice.only(rows), &lens)
}

/// Chained drafts as a nested loop; KStop decides who continues.
fn draft_mtp(r: &mut Model, w: &Mtp, ctx: ResourceId, rows: &Plan, p: &DraftPolicy) -> Drafts {
    let len = p.len.get(Drafter::Mtp);
    let start = (r.context(ctx).last(), r.last_tokens(rows), Drafts::empty());
    // The last link's hidden state feeds nothing.
    let last = move |_: &_, it: &Iter| it.step.compare_scalar(Cmp::Gt, len as f32 - 2.0);
    let until = Until {
        max: len,
        last: Box::new(last),
    };
    let (_, _, out) = r.repeat(until, start, |r, (h, tok, mut out), it| {
        let e = r.embed(&tok, &w.embed).rms_norm(&w.enorm);
        let x = e.concat(&h.rms_norm(&w.hnorm)).matmul(&w.proj);
        let h = r.with_rows(&it.live, |r| r.block(&w.block, &x));
        let logits = h.matmul(&w.head);
        let tok = r.sample(&logits, Sampler::Argmax);
        out.push(&tok, &logits.log_softmax_at(&tok));
        let sure = logits.softmax().max_per_row();
        let more = sure.compare_scalar(Cmp::Gt, p.confidence_floor);
        Next {
            carry: (h, tok, out),
            more,
        }
    });
    out
}

/// One pass drafts a whole block from the fused taps.
fn draft_dflash(r: &mut Model, w: &Dflash, t: &TapLayers, rows: &Plan, p: &DraftPolicy) -> Drafts {
    let len = p.len.get(Drafter::Dflash);
    let ctx = r.context(t.context).fused(&w.fuse);
    let mut x = r.mask_block(rows, len);
    for b in &w.blocks {
        x = r.block_over(b, &x, &ctx, Mask::BlockCausal { block: len });
    }
    sample_drafts(r, &x.rms_norm(&w.norm).matmul(&w.head))
}

/// Each layer passes its predecessor guess to the next position.
fn draft_dspine(r: &mut Model, w: &Dspine, t: &TapLayers, s: &DspineShape, rows: &Plan) -> Drafts {
    let ctx = r.context(t.context).fused(&w.fuse).rms_norm(&w.hidden_norm);
    let mut x = r.mask_block(rows, s.block).embed(&w.embed);
    let mut prev = x.zeros_like();
    for (i, b) in w.blocks.iter().enumerate() {
        x = r.block_over(b, &x.add(&prev), &ctx, Mask::BlockCausal { block: s.block });
        let guess = r.top_k(&x.matmul(&w.head), s.head_topk);
        let code = r.gather(&w.bridge.codebook, &guess.ids, Tier::Device);
        let feat = code.weighted(&guess.weights).matmul(&w.bridge.write[i]);
        prev = feat
            .gate(Act::Sigmoid, &code.matmul(&w.bridge.gate))
            .shift_rows(1);
    }
    let first = sample_drafts(r, &x.rms_norm(&w.norm).matmul(&w.head));
    refine(r, w, s, &ctx, first)
}

/// Second pass: keep the first drafts exact, redo the rest.
fn refine(r: &mut Model, w: &Dspine, s: &DspineShape, ctx: &Tensor, d: Drafts) -> Drafts {
    let fixed = d.tokens.first_rows(s.first_exact);
    let x = r.embed(&fixed.pad_to(s.block), &w.embed);
    let x = w.blocks.iter().fold(x, |x, b| {
        r.block_over(b, &x, ctx, Mask::BlockCausal { block: s.block })
    });
    d.splice(
        s.first_exact,
        sample_drafts(r, &x.rms_norm(&w.norm).matmul(&w.head)),
    )
}

fn sample_drafts(r: &mut Model, logits: &Tensor) -> Drafts {
    let tok = r.sample(logits, Sampler::Argmax);
    Drafts {
        logprobs: logits.log_softmax_at(&tok),
        tokens: tok,
    }
}
