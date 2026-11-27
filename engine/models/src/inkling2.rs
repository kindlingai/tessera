//! Inkling-2: looped decoder, depth attention, short convs, routed experts.

use crate::keys::{At, Keys};
use crate::shared::{CodedExperts, ExpertSet, PlainExperts, Stack, Steps, part, rows, sizes};
use tessera_trace::DimName::*;
use tessera_trace::hf::EPS;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, DimName, Embed, Graph, Kept, Linear, Mask, Model, Norm,
    PathSeg, Role, Shape, Size, Taps, Tensor, TensorType, Vector,
};
use tessera_trace::{pick, step_ports, tokens};

const REL_DIM: DimName = DimName::named("RelDim");
const REL_WIN: DimName = DimName::named("RelWin");
const SCONV_K: DimName = DimName::named("SconvK");
const SCONV_HIST: DimName = DimName::named("SconvHist");
const GATE_RANK: DimName = DimName::named("GateRank");
const GATE_IN: DimName = DimName::named("GateIn");
const UNPADDED: DimName = DimName::named("UnpaddedVocab");
const CROSS_ROWS: DimName = DimName::named("CrossPassRows");
const Q_DIM: DimName = DimName::named("QDim");
const KV_DIM: DimName = DimName::named("KvDim");
const REL_ALL: DimName = DimName::named("RelAll");
const DYN_TAPS: DimName = DimName::named("DynTaps");

/// Experts as plain matrices.
pub type Plain = PlainExperts;
/// Experts as rotated codes.
pub type Coded = CodedExperts;

#[derive(Clone, Debug)]
pub struct Config {
    pub stack: Stack,
    pub unpadded_vocab: u64,
    pub kv_heads: u64,
    pub d_rel: u64,
    pub rel_extent: u64,
    pub log_scaling_n_floor: f32,
    pub log_scaling_alpha: f32,
    pub eps: f32,
    pub qk_l2_clip: f32,
    pub rel_l2_clip: f32,
    /// True where a layer attends over the whole context.
    pub global: Vec<bool>,
    pub sliding_window: u64,
    pub sconv_kernel: u64,
    pub attn_res_block: u64,
    pub n_experts: u64,
    pub top_k: u64,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    pub final_logit_softcap: Option<f32>,
    pub loop_start: u64,
    pub loop_end: u64,
    pub loop_passes: u64,
    pub gate_rank: u64,
}

impl Config {
    /// Parses a Hugging Face `inkling2_loop` `config.json`.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let at = At::text_or_top();
        let stack = Stack::sized(sizes(k, &at), max_seqs, max_context);
        let local: Vec<u64> = k.read(&at, "local_layer_ids");
        let d_rel: u64 = k.read_or(&at, "d_rel", 16);
        let cap: f32 = k.read_or(&at, "final_logit_softcapping", 0.0);
        let clip = |n: u64| 2.0 * (n as f32).sqrt();
        Self {
            unpadded_vocab: k.read_or(&at, "unpadded_vocab_size", stack.vocab),
            kv_heads: k.read(&at, "num_key_value_heads"),
            d_rel,
            rel_extent: k.read_or(&at, "rel_extent", 1024),
            log_scaling_n_floor: k.read_or(&at, "log_scaling_n_floor", 8192.0),
            log_scaling_alpha: k.read_or(&at, "log_scaling_alpha", 0.1),
            eps: k.read_or(&at, EPS, 1e-6),
            qk_l2_clip: k.read_or(&at, "qk_l2_clip_threshold", clip(stack.head_dim)),
            rel_l2_clip: k.read_or(&at, "rel_l2_clip_threshold", clip(d_rel)),
            global: (0..stack.layers).map(|l| !local.contains(&l)).collect(),
            sliding_window: k.read_or(&at, "sliding_window_size", 1024),
            sconv_kernel: k.read_or(&at, "sconv_kernel_size", 4),
            attn_res_block: k.read_or(&at, "n_layers_per_attn_res_block", 6),
            n_experts: k.read_or(&at, "n_routed_experts", 128),
            top_k: k.read_or(&at, "num_experts_per_tok", 8),
            mlp_gate_softcap: k.read_or(&at, "mlp_gate_softcap", 4.0),
            mlp_up_softcap: k.read_or(&at, "mlp_up_softcap", 25.0),
            route_scale: k.read_or(&at, "route_scale", 8.0),
            final_logit_softcap: (cap > 0.0).then_some(cap),
            loop_start: k.read_or(&at, "loop_config.core_start", 0),
            loop_end: k.read_or(&at, "loop_config.core_end", 0),
            loop_passes: k.read_or(&at, "loop_config.num_passes", 1),
            gate_rank: k.read_or(&at, "loop_config.gate_rank", 0),
            stack,
        }
    }

    /// Inkling-2-Mini at its released sizes.
    pub fn mini(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([201_024, 2304, 24, 24, 128, 1152], max_seqs, max_context),
            unpadded_vocab: 200_058,
            kv_heads: 8,
            d_rel: 16,
            rel_extent: 1024,
            log_scaling_n_floor: 8192.0,
            log_scaling_alpha: 0.1,
            eps: 1e-6,
            qk_l2_clip: 2.0 * 128f32.sqrt(),
            rel_l2_clip: 8.0,
            global: (0..24).map(|l| l % 6 == 5).collect(),
            sliding_window: 1024,
            sconv_kernel: 4,
            attn_res_block: 6,
            n_experts: 128,
            top_k: 8,
            mlp_gate_softcap: 4.0,
            mlp_up_softcap: 25.0,
            route_scale: 8.0,
            final_logit_softcap: Some(30.0),
            loop_start: 6,
            loop_end: 18,
            loop_passes: 3,
            gate_rank: 64,
        }
    }

    /// Eight small layers, a core of four run thrice.
    pub fn small() -> Self {
        Self {
            stack: Stack::sized([64, 16, 8, 4, 4, 16], 2, 16),
            unpadded_vocab: 60,
            kv_heads: 2,
            d_rel: 4,
            rel_extent: 8,
            log_scaling_n_floor: 4.0,
            qk_l2_clip: 4.0,
            rel_l2_clip: 4.0,
            global: (0..8).map(|l| l % 2 == 1).collect(),
            sliding_window: 4,
            attn_res_block: 2,
            n_experts: 4,
            top_k: 2,
            loop_start: 2,
            loop_end: 6,
            gate_rank: 4,
            ..Self::mini(2, 16)
        }
    }

    pub fn is_looped(&self) -> bool {
        self.loop_passes > 1 && self.loop_end > self.loop_start
    }

    /// Layers run per token, counting every loop pass.
    pub fn n_virtual_layers(&self) -> usize {
        let core = self.loop_end.saturating_sub(self.loop_start);
        let extra = if self.is_looped() {
            (self.loop_passes - 1) * core
        } else {
            0
        };
        (self.stack.layers + extra) as usize
    }

    fn core(&self) -> std::ops::Range<usize> {
        if self.is_looped() {
            self.loop_start as usize..self.loop_end as usize
        } else {
            0..0
        }
    }

    fn dims(&self) -> Vec<Dim> {
        let s = &self.stack;
        let kv = self.kv_heads * s.head_dim;
        let mut own = vec![
            (KvHeads, self.kv_heads),
            (REL_DIM, self.d_rel),
            (REL_WIN, self.rel_extent),
            (SCONV_K, self.sconv_kernel),
            (SCONV_HIST, self.sconv_kernel - 1),
            (Experts, self.n_experts),
            (TopK, self.top_k),
            (UNPADDED, self.unpadded_vocab),
            (Q_DIM, s.heads * s.head_dim),
            (KV_DIM, kv),
            (REL_ALL, s.heads * self.d_rel),
            (DYN_TAPS, self.kv_heads * self.sconv_kernel),
        ];
        if self.is_looped() {
            own.extend([
                (GATE_RANK, self.gate_rank),
                (GATE_IN, 2 * s.hidden),
                (CROSS_ROWS, 2 * (self.loop_end - self.loop_start)),
            ]);
        }
        let mut dims = s.dims(&own);
        for d in &mut dims {
            if d.name == VocabWords {
                *d = Dim::fixed(VocabWords, self.unpadded_vocab.div_ceil(32));
            }
        }
        dims
    }
}

/// A cross-pass message gate.
struct Gate {
    down: Linear,
    forget: Linear,
    gain: Linear,
}

impl Gate {
    fn declare(m: &Model, p: &mut Vec<PathSeg>, name: &str) -> Self {
        p.push(PathSeg::Field(name.into()));
        let wide = [GATE_RANK.into(), Hidden.into()];
        let gate = Self {
            down: part(m, p, "down", &[GATE_IN.into(), GATE_RANK.into()]),
            forget: part(m, p, "forget", &wide),
            gain: part(m, p, "gain", &wide),
        };
        p.pop();
        gate
    }

    /// `(1 - forget) * h + gain * msg`.
    fn apply(&self, h: &Tensor, msg: &Tensor, lead: &[DimExpr], eps: f32) -> Tensor {
        let both = h.rms(eps).concat(&msg.rms(eps));
        let g = both.reshape(cat(lead, [GATE_IN.into()])).matmul(&self.down);
        let fg = g.matmul(&self.forget).act(Act::Tanh);
        let mg = g.matmul(&self.gain).act(Act::Tanh);
        h.add(&h.mul(&fg).scale(-1.0)).add(&msg.mul(&mg))
    }
}

/// A key or value short conv.
enum KvConv {
    Fixed(Taps),
    /// Fixed taps plus taps projected per head.
    Dynamic {
        fixed: Linear,
        proj: Linear,
    },
}

impl KvConv {
    fn declare(m: &Model, p: &mut Vec<PathSeg>, side: &str, glob: bool) -> Self {
        let name = |tail: &str| format!("{side}_{tail}");
        if glob {
            Self::Dynamic {
                fixed: part(m, p, &name("static"), &[SCONV_K.into(), KV_DIM.into()]),
                proj: part(m, p, &name("wproj"), &[Hidden.into(), DYN_TAPS.into()]),
            }
        } else {
            Self::Fixed(part(m, p, &name("sconv"), &[KV_DIM.into(), SCONV_K.into()]))
        }
    }
}

struct Attn {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    wr: Linear,
    gate: Linear,
    sink: Linear,
    q_clip: Tensor,
    rel_clip: Tensor,
    rel_proj: Linear,
    k_conv: KvConv,
    v_conv: KvConv,
}

fn raw(m: &Model, p: &mut Vec<PathSeg>, name: &str, shape: Vec<DimExpr>) -> Tensor {
    p.push(PathSeg::Field(name.into()));
    let t = m.weight(p, TensorType::new(Shape(shape), DType::F32));
    p.pop();
    t
}

impl Attn {
    fn declare(m: &Model, p: &mut Vec<PathSeg>, glob: bool) -> Self {
        p.push(PathSeg::Field("attn".into()));
        let h: DimExpr = Hidden.into();
        let mut lin = |name: &str, shape: [DimExpr; 2]| -> Linear { part(m, p, name, &shape) };
        let attn = Self {
            wq: lin("wq", [h.clone(), Q_DIM.into()]),
            wk: lin("wk", [h.clone(), KV_DIM.into()]),
            wv: lin("wv", [h.clone(), KV_DIM.into()]),
            wo: lin("wo", [Q_DIM.into(), h.clone()]),
            wr: lin("wr", [h.clone(), REL_ALL.into()]),
            gate: lin("gate", [h.clone(), Q_DIM.into()]),
            sink: lin("sink", [h, Heads.into()]),
            rel_proj: lin("rel_proj", [REL_DIM.into(), REL_WIN.into()]),
            q_clip: raw(m, p, "q_clip", vec![Heads.into(), HeadDim.into()]),
            rel_clip: raw(m, p, "rel_clip", vec![Heads.into(), REL_DIM.into()]),
            k_conv: KvConv::declare(m, p, "k", glob),
            v_conv: KvConv::declare(m, p, "v", glob),
        };
        p.pop();
        attn
    }
}

struct Layer<E> {
    /// One depth query when local, three when global.
    attn_query: Option<Linear>,
    mlp_query: Linear,
    /// Query, key, value input norms, or one shared.
    norms: Vec<Norm>,
    attn: Attn,
    attn_sconv: Taps,
    mlp_sconv: Taps,
    mlp_norm: Norm,
    router: Linear,
    router_bias: Vector,
    experts: E,
    /// Message gates for query, key, value, and MLP.
    ctrl: Option<[Gate; 4]>,
}

impl<E: ExpertSet> Layer<E> {
    fn declare(m: &Model, c: &Config, l: usize) -> Self {
        let p = &mut vec![PathSeg::Field("layers".into()), PathSeg::Index(l as u32)];
        let glob = c.global[l];
        let h: DimExpr = Hidden.into();
        let one = DimExpr::Const(1);
        let width = if glob { DimExpr::Const(3) } else { one.clone() };
        let taps = [h.clone(), SCONV_K.into()];
        let names: &[&str] = if glob {
            &["q_norm", "k_norm", "v_norm"]
        } else {
            &["attn_norm"]
        };
        let attn_query = (l > 0).then(|| part(m, p, "attn_query", &[h.clone(), width]));
        let mlp_query = part(m, p, "mlp_query", &[h.clone(), one]);
        let norms = names
            .iter()
            .map(|n| part(m, p, n, std::slice::from_ref(&h)))
            .collect();
        let attn = Attn::declare(m, p, glob);
        let attn_sconv = part(m, p, "attn_sconv", &taps);
        let mlp_sconv = part(m, p, "mlp_sconv", &taps);
        let mlp_norm = part(m, p, "mlp_norm", std::slice::from_ref(&h));
        p.push(PathSeg::Field("moe".into()));
        let router = part(m, p, "router", &[h, Experts.into()]);
        let router_bias = part(m, p, "router_bias", &[Experts.into()]);
        let experts = part(m, p, "experts", &[]);
        p.pop();
        let ctrl = c.core().contains(&l).then(|| {
            p.push(PathSeg::Field("ctrl".into()));
            let gates = ["q", "k", "v", "mlp"].map(|name| Gate::declare(m, p, name));
            p.pop();
            gates
        });
        Self {
            attn_query,
            mlp_query,
            norms,
            attn,
            attn_sconv,
            mlp_sconv,
            mlp_norm,
            router,
            router_bias,
            experts,
            ctrl,
        }
    }
}

struct Inkling2<E> {
    embed: Embed,
    embed_norm: Norm,
    norm: Norm,
    head: Embed,
    output_query: Linear,
    /// Query and MLP message queries, two rows per core layer.
    cross_pass: Option<[Linear; 2]>,
    layers: Vec<Layer<E>>,
}

impl<E: ExpertSet> Inkling2<E> {
    fn declare(m: &Model, c: &Config) -> Self {
        let p = &mut Vec::new();
        let table = [Vocab.into(), Hidden.into()];
        let cross = [Hidden.into(), CROSS_ROWS.into()];
        Self {
            embed: part(m, p, "embed", &table),
            embed_norm: part(m, p, "embed_norm", &[Hidden.into()]),
            norm: part(m, p, "norm", &[Hidden.into()]),
            head: part(m, p, "head", &table),
            output_query: part(m, p, "output_query", &[Hidden.into(), DimExpr::Const(1)]),
            cross_pass: c.is_looped().then(|| {
                ["cross_pass_queries", "cross_pass_kv_queries"].map(|n| part(m, p, n, &cross))
            }),
            layers: (0..c.stack.layers as usize)
                .map(|l| Layer::declare(m, c, l))
                .collect(),
        }
    }
}

/// One virtual layer's kept rows while stepping.
struct Virt {
    hist_a: Kept,
    hist_m: Kept,
    hist_k: Kept,
    hist_v: Kept,
    keys: Kept,
    values: Kept,
}

fn cat<const N: usize>(lead: &[DimExpr], rest: [DimExpr; N]) -> Vec<DimExpr> {
    lead.iter().cloned().chain(rest).collect()
}

struct Run<'a, E> {
    m: &'a Model,
    c: &'a Config,
    w: &'a Inkling2<E>,
    lead: Vec<DimExpr>,
    positions: &'a Tensor,
    /// Slots, then each virtual layer's state.
    step: Option<(&'a Tensor, &'a [Virt])>,
}

impl<E: ExpertSet> Run<'_, E> {
    fn shape<const N: usize>(&self, rest: [DimExpr; N]) -> Vec<DimExpr> {
        cat(&self.lead, rest)
    }

    /// Unit-RMS `states` and `partial`, stacked by depth.
    fn stack(&self, states: &[Tensor], partial: &Tensor) -> Tensor {
        let depth = DimExpr::Const(states.len() as u64 + 1);
        let unit = states.iter().chain([partial]).map(|s| s.rms(self.c.eps));
        let all = unit.reduce(|a, b| a.concat(&b)).expect("partial present");
        all.reshape(self.shape([depth, Hidden.into()]))
    }

    /// One depth blend of `u` per query column of `scores`.
    fn blend(&self, u: &Tensor, scores: &Tensor, queries: u64) -> Vec<Tensor> {
        let at = self.lead.len() as u32;
        let depth = u.ty().shape.0[at as usize].clone();
        let order: Vec<u32> = (0..at).chain([at + 1, at]).collect();
        let scale = (self.c.stack.hidden as f32).sqrt().recip();
        let weights = scores.permute(&order).scale(scale).softmax();
        (0..queries)
            .map(|q| {
                let one = weights.slice(at, q, 1u64);
                u.mix(at, &one.reshape(self.shape([depth.clone()])))
            })
            .collect()
    }

    fn attn_res(&self, states: &[Tensor], partial: &Tensor, query: &Linear, n: u64) -> Vec<Tensor> {
        let u = self.stack(states, partial);
        self.blend(&u, &u.matmul(query), n)
    }

    /// `x` plus its causal filter, scaled.
    fn sconv(&self, w: &Taps, hist: Option<&Kept>, x: &Tensor) -> Tensor {
        let y = match (hist, self.step) {
            (Some(kept), Some((slots, _))) => x.conv_rows(w, 1, kept, [slots, self.positions]),
            _ => x.conv(w),
        };
        x.add(&y).scale(5f32.sqrt().recip())
    }

    fn kv_conv(&self, w: &KvConv, hist: Option<&Kept>, x: &Tensor, input: &Tensor) -> Tensor {
        let (fixed, proj) = match w {
            KvConv::Fixed(taps) => return self.sconv(taps, hist, x),
            KvConv::Dynamic { fixed, proj } => (fixed, proj),
        };
        let at = self.lead.len() as u32;
        let per_head = self.shape([KvHeads.into(), DimExpr::Const(1), SCONV_K.into()]);
        let taps = input
            .matmul(proj)
            .reshape(per_head)
            .expand(at + 1, HeadDim)
            .reshape(self.shape([KV_DIM.into(), SCONV_K.into()]))
            .add(fixed.stored());
        let y = match (hist, self.step) {
            (Some(kept), Some((slots, _))) => {
                x.conv_rows_dynamic(&taps, 1, kept, [slots, self.positions])
            }
            _ => x.conv_dynamic(&taps, 1),
        };
        x.add(&y).scale(1.0 / 3.0)
    }

    fn attend(&self, v: usize, l: usize, [qi, ki, vi]: [&Tensor; 3]) -> Tensor {
        let (c, a) = (self.c, &self.w.layers[l].attn);
        let glob = c.global[l];
        let virt = self.step.map(|(_, virt)| &virt[v]);
        let heads =
            |t: Tensor, n: DimName, width: DimName| t.reshape(self.shape([n.into(), width.into()]));
        let kc = self.kv_conv(&a.k_conv, virt.map(|s| &s.hist_k), &ki.matmul(&a.wk), ki);
        let vc = self.kv_conv(&a.v_conv, virt.map(|s| &s.hist_v), &vi.matmul(&a.wv), vi);
        let k = heads(kc, KvHeads, HeadDim).l2_clip(c.qk_l2_clip, c.eps);
        let val = heads(vc, KvHeads, HeadDim).rms(c.eps);
        let mut q = heads(qi.matmul(&a.wq), Heads, HeadDim)
            .mul(&a.q_clip)
            .l2_clip(c.qk_l2_clip, c.eps);
        let mut r = heads(qi.matmul(&a.wr), Heads, REL_DIM)
            .mul(&a.rel_clip)
            .l2_clip(c.rel_l2_clip, c.eps);
        if glob {
            let [floor, alpha] = [c.log_scaling_n_floor, c.log_scaling_alpha];
            q = q.log_scale(self.positions, floor, alpha);
            r = r.log_scale(self.positions, floor, alpha);
        }
        let rel = r.matmul(&a.rel_proj);
        let sink = qi.matmul(&a.sink);
        let mask = if glob {
            Mask::Causal
        } else {
            Mask::Back(c.sliding_window.saturating_sub(1))
        };
        let scale = (c.stack.head_dim as f32).sqrt().recip();
        let extras = [Some(&sink), Some(&rel)];
        let out = match (virt, self.step) {
            (Some(s), Some((slots, _))) => {
                let by = [slots, self.positions];
                let (k, val) = (k.append(&s.keys, by), val.append(&s.values, by));
                self.m
                    .attend_rows_biased(mask, scale, [&q, &k, &val], by, None, extras)
            }
            _ => self
                .m
                .attend_biased(mask, scale, [&q, &k, &val], None, extras),
        };
        let open = qi.matmul(&a.gate).act(Act::Sigmoid);
        let gated = out.reshape(self.shape([Q_DIM.into()])).mul(&open);
        gated.matmul(&a.wo)
    }

    fn moe(&self, w: &Layer<E>, x: &Tensor) -> Tensor {
        let c = self.c;
        let score = x.matmul(&w.router).act(Act::Sigmoid);
        let ids = score.add_vector(&w.router_bias).top_k(TopK);
        let gates = score.take(&ids).unit_sum().scale(c.route_scale);
        let [gcap, ucap] = [c.mlp_gate_softcap, c.mlp_up_softcap];
        let capped = |t: &Tensor, cap: f32| t.scale(cap.recip()).act(Act::Tanh).scale(cap);
        // Sigmoid as tanh, so zero rows stay zero.
        let join = |gate: Tensor, up: Tensor| {
            let half = capped(&gate, gcap).scale(0.5);
            let swung = half.mul(&gate.scale(0.5).act(Act::Tanh));
            half.add(&swung).mul(&capped(&up, ucap))
        };
        w.experts.apply_with(self.m, x, [&ids, &gates], &join).0
    }

    /// Query, key, and value inputs of real layer `l`.
    fn inputs(
        &self,
        l: usize,
        states: &[Tensor],
        h: &Tensor,
        msgs: Option<&[Tensor; 4]>,
    ) -> [Tensor; 3] {
        let (c, w) = (self.c, &self.w.layers[l]);
        let split = if c.global[l] { 3 } else { 1 };
        let x = match &w.attn_query {
            Some(query) => self.attn_res(states, h, query, split),
            None => vec![h.clone()],
        };
        let normed = |i: usize, t: &Tensor| t.rms_norm(&w.norms[i.min(w.norms.len() - 1)], c.eps);
        let side = |i: usize| &x[i.min(x.len() - 1)];
        match (msgs, &w.ctrl) {
            (Some(m4), Some(ctrl)) => [0, 1, 2].map(|i| {
                let gated = ctrl[i].apply(side(i), &m4[i], &self.lead, c.eps);
                normed(i, &gated)
            }),
            _ if x.len() == 1 => {
                let one = normed(0, &x[0]);
                [one.clone(), one.clone(), one]
            }
            _ => [0, 1, 2].map(|i| normed(i, side(i))),
        }
    }

    /// Virtual layer `v` on real layer `l`'s weights.
    fn layer(
        &self,
        [v, l]: [usize; 2],
        states: &mut Vec<Tensor>,
        h: &mut Tensor,
        msgs: Option<&[Tensor; 4]>,
    ) {
        let (c, w) = (self.c, &self.w.layers[l]);
        let virt = self.step.map(|(_, virt)| &virt[v]);
        let [a] = self.m.composite(Composite::Attention, || {
            let [qi, ki, vi] = self.inputs(l, states, h, msgs);
            let out = self.attend(v, l, [&qi, &ki, &vi]);
            [self.sconv(&w.attn_sconv, virt.map(|s| &s.hist_a), &out)]
        });
        let mid = if (l as u64).is_multiple_of(c.attn_res_block) {
            states.push(h.clone());
            a
        } else {
            h.add(&a)
        };
        let [next] = self.m.composite(Composite::Moe, || {
            let mut x = self.attn_res(states, &mid, &w.mlp_query, 1).remove(0);
            if let (Some(m4), Some(ctrl)) = (msgs, &w.ctrl) {
                x = ctrl[3].apply(&x, &m4[3], &self.lead, c.eps);
            }
            let out = self.moe(w, &x.rms_norm(&w.mlp_norm, c.eps));
            let ms = self.sconv(&w.mlp_sconv, virt.map(|s| &s.hist_m), &out);
            [mid.add(&ms)]
        });
        *h = next;
    }

    /// Every core layer's messages from the last pass.
    fn messages(&self, states: &[Tensor], h: &Tensor) -> Vec<[Tensor; 4]> {
        let [main, kv] = self.w.cross_pass.as_ref().expect("looped");
        let u = self.stack(states, h);
        let n = 2 * self.c.core().len();
        let all = |w: &Linear| self.blend(&u, &u.matmul(w), n as u64);
        let (qm, kv) = (all(main), all(kv));
        let four = |j: usize| [&qm[j], &kv[j], &kv[j + 1], &qm[j + 1]].map(Tensor::clone);
        (0..n).step_by(2).map(four).collect()
    }

    fn logits(&self, tokens: &Tensor) -> Tensor {
        let (c, w) = (self.c, self.w);
        let mut h = self
            .m
            .embed(tokens, &w.embed)
            .rms_norm(&w.embed_norm, c.eps);
        let mut states = Vec::new();
        let core = c.core();
        let layers = c.stack.layers as usize;
        let mut v = 0;
        let mut plain =
            |range: std::ops::Range<usize>, states: &mut Vec<Tensor>, h: &mut Tensor| {
                for l in range {
                    self.layer([v, l], states, h, None);
                    v += 1;
                }
            };
        if core.is_empty() {
            plain(0..layers, &mut states, &mut h);
        } else {
            plain(0..core.start, &mut states, &mut h);
            let (pre_h, pre_states) = (h.clone(), states.clone());
            plain(core.clone(), &mut states, &mut h);
            let mut v = core.end;
            for _ in 1..c.loop_passes {
                let prev = states.split_off((pre_states.len() + 1).min(states.len()));
                let prev_h = std::mem::replace(&mut h, pre_h.clone());
                states = pre_states.clone();
                let msgs = self.messages(&prev, &prev_h);
                for (l, m4) in core.clone().zip(&msgs) {
                    self.layer([v, l], &mut states, &mut h, Some(m4));
                    v += 1;
                }
            }
            for l in core.end..layers {
                self.layer([v, l], &mut states, &mut h, None);
                v += 1;
            }
        }
        let x = self.attn_res(&states, &h, &w.output_query, 1).remove(0);
        let mut logits = x.rms_norm(&w.norm, c.eps).unembed(&w.head);
        if c.unpadded_vocab < c.stack.vocab {
            logits = logits.slice(self.lead.len() as u32, 0u64, UNPADDED);
        }
        match c.final_logit_softcap {
            Some(cap) => logits.scale(cap.recip()).act(Act::Tanh).scale(cap),
            None => logits,
        }
    }
}

/// Whole-sequence forward graph.
pub fn program<E: ExpertSet>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w = Inkling2::<E>::declare(&m, c);
    let [tokens, positions] = tokens(&m);
    let run = Run {
        m: &m,
        c,
        w: &w,
        lead: rows([]),
        positions: &positions,
        step: None,
    };
    let [logits] = m.composite(Composite::Model, || [run.logits(&tokens)]);
    m.finish([(Role::Logits, &logits)])
}

/// Ragged step graph, state kept per virtual layer.
pub fn stepper<E: ExpertSet>(c: &Config) -> Graph {
    let mut dims = c.dims();
    dims.push(Dim {
        name: Rows,
        size: Size::Bounded(c.stack.max_context),
    });
    let m = Model::new(dims);
    let w = Inkling2::<E>::declare(&m, c);
    let ([tokens, positions, slots], allowed) = step_ports(&m, c.stack.steps == Steps::Masked);
    let hist = |width: DimName| {
        let shape = vec![Seqs.into(), SCONV_HIST.into(), width.into()];
        m.kept(TensorType::new(Shape(shape), DType::F32))
    };
    let cache = || {
        m.kept(TensorType::new(
            Shape(rows([KvHeads.into(), HeadDim.into()])),
            DType::F32,
        ))
    };
    let virt: Vec<Virt> = (0..c.n_virtual_layers())
        .map(|_| Virt {
            hist_a: hist(Hidden),
            hist_m: hist(Hidden),
            hist_k: hist(KV_DIM),
            hist_v: hist(KV_DIM),
            keys: cache(),
            values: cache(),
        })
        .collect();
    let run = Run {
        m: &m,
        c,
        w: &w,
        lead: vec![Rows.into()],
        positions: &positions,
        step: Some((&slots, &virt)),
    };
    let [logits] = m.composite(Composite::Model, || [run.logits(&tokens)]);
    pick(&m, &logits, allowed.as_ref(), None)
}

crate::preset::traced!("inkling2", Config::small, Config::mini, [Plain]);
