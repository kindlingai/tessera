//! Speculative rounds: MTP, DFlash2 and DSpine drafts, verify, commit. Sketch.

use crate::core::*;
use crate::glm53::{self, Glm53};
use crate::library::{greedy_accept, rejection_accept};

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

/// The graph's own limits; the host picks the drafter.
pub struct DraftLimits {
    pub max_rounds: usize,
    pub max_len: PerDrafter<usize>,
    pub confidence_floor: f32,
}

/// The host's side of the drafter choice.
pub struct DraftPorts {
    /// Mailbox: each sequence's drafter and draft length.
    pub choice: Port,
    /// Stream: acceptance per drafter, each round.
    pub stats: Port,
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
    lim: &DraftLimits,
    ports: &DraftPorts,
    rule: Rule,
    left: Tensor,
    out: &Port,
) -> Tensor {
    // Every round emits a token, so one left means last.
    let last = |left: &Tensor, _: &Iter| left.compare_scalar(Cmp::Lt, 2.0);
    let until = Until {
        max: lim.max_rounds,
        last: Box::new(last),
    };
    m.repeat(until, left, |m, left, it| {
        let n = round(m, t, d, lim, ports, &rule, it, &left);
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
    lim: &DraftLimits,
    ports: &DraftPorts,
    rule: &Rule,
    it: &Iter,
    left: &Tensor,
) -> Counts {
    let plans = apply_choice(r, &r.read(&ports.choice), lim, it, left);
    let a = draft_mtp(r, &d.mtp.0, d.mtp.1, &plans.part(Drafter::Mtp), lim);
    let b = draft_dflash(r, &d.dflash.0, &d.dflash.1, &plans.part(Drafter::Dflash));
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
    r.emit(&ports.stats, &r.acceptance(&plans, &n));
    r.commit_all(&n);
    // No later round drafts from a last row's taps.
    r.drop_taps(&it.last);
    n
}

/// The host's choice, clamped by what only the graph knows.
fn apply_choice(
    r: &mut Model,
    choice: &Tensor,
    lim: &DraftLimits,
    it: &Iter,
    left: &Tensor,
) -> Plans<Drafter> {
    // A last round drafts nothing; one verify row suffices.
    let drafting = it.live.minus(&r.filter(&it.last));
    let len = choice
        .len()
        .min(&left.sub_scalar(1.0))
        .min_each(&lim.max_len);
    r.partition_rows(&choice.drafter().only(&drafting), &len)
}

/// Chained drafts as a nested loop; KStop decides who continues.
fn draft_mtp(r: &mut Model, w: &Mtp, ctx: ResourceId, rows: &Plan, lim: &DraftLimits) -> Drafts {
    let len = lim.max_len.get(Drafter::Mtp);
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
        let more = sure.compare_scalar(Cmp::Gt, lim.confidence_floor);
        Next {
            carry: (h, tok, out),
            more,
        }
    });
    out
}

/// One pass drafts a whole block from the fused taps.
fn draft_dflash(r: &mut Model, w: &Dflash, t: &TapLayers, rows: &Plan) -> Drafts {
    let len = rows.max_len();
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

// Host side: plain Rust, not a description.

/// One sequence as the host's policy sees it.
pub struct SeqView {
    pub accept: PerDrafter<f32>,
    pub drafter: Drafter,
    pub dwell: usize,
    /// Request facts only the host has, like a code hint.
    pub hint: Option<Drafter>,
}

/// The host's costs and limits, from tuning data and load.
pub struct HostPolicy {
    pub len: PerDrafter<usize>,
    pub cost: PerDrafter<f32>,
    pub min_dwell: usize,
    pub row_budget: usize,
}

/// Best expected gain, held for a dwell, within budget.
pub fn choose(seqs: &mut [SeqView], p: &HostPolicy) -> Vec<(Drafter, usize)> {
    let mut rows = 0;
    let mut out = Vec::with_capacity(seqs.len());
    for s in seqs.iter_mut() {
        let best = s.hint.unwrap_or_else(|| p.best(&s.accept));
        if s.dwell >= p.min_dwell && best != s.drafter {
            s.drafter = best;
            s.dwell = 0;
        }
        s.dwell += 1;
        let len = p.len.get(s.drafter).min(p.row_budget.saturating_sub(rows));
        rows += len;
        out.push((s.drafter, len));
    }
    out
}
