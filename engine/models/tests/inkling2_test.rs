//! Inkling-2 graphs against a scalar reference, and their lowering.

use std::collections::HashMap;
use tessera_gpukit::platform::{Asked, Platform};
use tessera_graph::DimName::{Context, Rows, Seqs};
use tessera_graph::{Graph, PathSeg};
use tessera_interp::{Store, Tensor, run, run_kept};
use tessera_lower::{Device, lower};
use tessera_models::inkling2::{Coded, Config, Plain, program, stepper};
use tessera_rules as _;
use tessera_run::testing::host;
use tessera_testkit::{Noise, weights_with};

type Mat = Vec<Vec<f32>>;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn lin(w: &[f32], x: &[f32]) -> Vec<f32> {
    w.chunks(x.len()).map(|row| dot(row, x)).collect()
}

fn rows(w: &[f32], x: &Mat) -> Mat {
    x.iter().map(|r| lin(w, r)).collect()
}

fn rms(x: &[f32], eps: f32) -> Vec<f32> {
    let r = (dot(x, x) / x.len() as f32 + eps).sqrt().recip();
    x.iter().map(|v| v * r).collect()
}

fn rms_norm(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    rms(x, eps).iter().zip(gain).map(|(v, g)| v * g).collect()
}

fn l2_clip(x: &[f32], threshold: f32, eps: f32) -> Vec<f32> {
    let by = (threshold / (dot(x, x).sqrt() + eps)).min(1.0);
    x.iter().map(|v| v * by).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let top = x.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - top).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

/// `x` plus taps over its past: `taps(t, c)[3 - lag]`.
fn sconv(x: &Mat, by: f32, taps: impl Fn(usize, usize) -> Vec<f32>) -> Mat {
    let each = |t: usize, c: usize| {
        let w = taps(t, c);
        let k = w.len();
        let past: f32 = (0..k.min(t + 1))
            .map(|lag| w[k - 1 - lag] * x[t - lag][c])
            .sum();
        (x[t][c] + past) * by
    };
    (0..x.len())
        .map(|t| (0..x[t].len()).map(|c| each(t, c)).collect())
        .collect()
}

struct Ref {
    c: Config,
    w: HashMap<String, Vec<f32>>,
}

impl Ref {
    fn new(c: &Config, g: &Graph, ws: &[Tensor]) -> Self {
        let name = |path: &[PathSeg]| {
            let parts = path.iter().map(|seg| match seg {
                PathSeg::Field(f) => f.to_string(),
                PathSeg::Index(i) => i.to_string(),
            });
            parts.collect::<Vec<_>>().join(".")
        };
        let named = g.weights.iter().zip(ws);
        let w = named.map(|(w, t)| (name(&w.path), t.floats().unwrap().to_vec()));
        Self {
            c: c.clone(),
            w: w.collect(),
        }
    }

    fn top(&self, name: &str) -> &[f32] {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("no weight {name}"))
    }

    fn at(&self, l: usize, name: &str) -> &[f32] {
        self.top(&format!("layers.{l}.{name}"))
    }

    fn static_conv(&self, l: usize, name: &str, x: &Mat) -> Mat {
        let w = self.at(l, name);
        let k = self.c.sconv_kernel as usize;
        sconv(x, 5f32.sqrt().recip(), |_, c| w[c * k..][..k].to_vec())
    }

    fn kv_conv(&self, l: usize, side: &str, x: &Mat, input: &Mat) -> Mat {
        if !self.c.global[l] {
            return self.static_conv(l, &format!("attn.{side}_sconv"), x);
        }
        let k = self.c.sconv_kernel as usize;
        let hd = self.c.stack.head_dim as usize;
        let fixed = self.at(l, &format!("attn.{side}_static"));
        let moved = rows(self.at(l, &format!("attn.{side}_wproj")), input);
        sconv(x, 1.0 / 3.0, |t, c| {
            let head = &moved[t][c / hd * k..][..k];
            (0..k).map(|j| fixed[c * k + j] + head[j]).collect()
        })
    }

    /// One blend over depth per query row.
    fn attn_res(&self, states: &[Mat], partial: &Mat, query: &[f32]) -> Vec<Mat> {
        let (eps, h) = (self.c.eps, self.c.stack.hidden as usize);
        let token = |t: usize, q: &[f32]| {
            let depth = states.iter().chain([partial]);
            let u: Mat = depth.map(|s| rms(&s[t], eps)).collect();
            let scores: Vec<f32> = u.iter().map(|r| dot(q, r) / (h as f32).sqrt()).collect();
            let p = softmax(&scores);
            (0..h)
                .map(|c| (0..u.len()).map(|d| p[d] * u[d][c]).sum())
                .collect()
        };
        let each = |q: &[f32]| (0..partial.len()).map(|t| token(t, q)).collect();
        query.chunks(h).map(each).collect()
    }

    fn gate(&self, l: usize, name: &str, x: &Mat, msg: &Mat) -> Mat {
        let eps = self.c.eps;
        let part = |tail: &str| self.at(l, &format!("ctrl.{name}.{tail}"));
        let one = |h: &Vec<f32>, m: &Vec<f32>| {
            let both = [rms(h, eps), rms(m, eps)].concat();
            let g = lin(part("down"), &both);
            let (fg, mg) = (lin(part("forget"), &g), lin(part("gain"), &g));
            (0..h.len())
                .map(|c| (1.0 - fg[c].tanh()) * h[c] + mg[c].tanh() * m[c])
                .collect()
        };
        x.iter().zip(msg).map(|(h, m)| one(h, m)).collect()
    }

    fn attention(&self, l: usize, [qi, ki, vi]: [&Mat; 3]) -> Mat {
        let (c, s) = (&self.c, &self.c.stack);
        let [nh, hd, rd] = [s.heads, s.head_dim, c.d_rel].map(|n| n as usize);
        let [kvh, win] = [c.kv_heads, c.rel_extent].map(|n| n as usize);
        let (glob, eps) = (c.global[l], c.eps);
        let part = |name: &str| self.at(l, &format!("attn.{name}"));
        let kc = self.kv_conv(l, "k", &rows(part("wk"), ki), ki);
        let vc = self.kv_conv(l, "v", &rows(part("wv"), vi), vi);
        let heads = |m: &Mat, f: &dyn Fn(&[f32]) -> Vec<f32>| -> Mat {
            m.iter()
                .map(|r| r.chunks(hd).flat_map(f).collect())
                .collect()
        };
        let keys = heads(&kc, &|k| l2_clip(k, c.qk_l2_clip, eps));
        let vals = heads(&vc, &|v| rms(v, eps));
        let clipped = |x: &[f32], [w, scale]: [&str; 2], width: usize, threshold: f32| {
            let scaled: Vec<f32> = lin(part(w), x)
                .iter()
                .zip(part(scale))
                .map(|(a, b)| a * b)
                .collect();
            let clip = |r: &[f32]| l2_clip(r, threshold, eps);
            scaled.chunks(width).flat_map(clip).collect::<Vec<f32>>()
        };
        let token = |p: usize| {
            let x = &qi[p];
            let tau = if glob {
                let over = ((p + 1) as f32 / c.log_scaling_n_floor).max(1.0);
                1.0 + c.log_scaling_alpha * over.ln()
            } else {
                1.0
            };
            let q = clipped(x, ["wq", "q_clip"], hd, c.qk_l2_clip);
            let r = clipped(x, ["wr", "rel_clip"], rd, c.rel_l2_clip);
            let (sinks, open) = (lin(part("sink"), x), lin(part("gate"), x));
            let lo = if glob {
                0
            } else {
                (p + 1).saturating_sub(c.sliding_window as usize)
            };
            let mut out = vec![0.0; nh * hd];
            for h in 0..nh {
                let qh: Vec<f32> = q[h * hd..][..hd].iter().map(|v| v * tau).collect();
                let rh: Vec<f32> = r[h * rd..][..rd].iter().map(|v| v * tau).collect();
                let kv = h / (nh / kvh) * hd;
                let mut scores = vec![sinks[h]];
                for (at, key) in keys.iter().enumerate().take(p + 1).skip(lo) {
                    let mut z = dot(&qh, &key[kv..][..hd]) / (hd as f32).sqrt();
                    if p - at < win {
                        z += dot(&rh, &part("rel_proj")[(p - at) * rd..][..rd]);
                    }
                    scores.push(z);
                }
                let probs = softmax(&scores);
                for d in 0..hd {
                    let seen = (lo..=p).map(|at| probs[at - lo + 1] * vals[at][kv + d]);
                    out[h * hd + d] = seen.sum::<f32>() * sigmoid(open[h * hd + d]);
                }
            }
            lin(part("wo"), &out)
        };
        (0..qi.len()).map(token).collect()
    }

    fn moe(&self, l: usize, x: &[f32]) -> Vec<f32> {
        let c = &self.c;
        let [h, ffn] = [c.stack.hidden, c.stack.ffn].map(|n| n as usize);
        let part = |name: &str| self.at(l, &format!("moe.{name}"));
        let score: Vec<f32> = lin(part("router"), x).into_iter().map(sigmoid).collect();
        let mut order: Vec<usize> = (0..score.len()).collect();
        let ranked = |e: usize| score[e] + part("router_bias")[e];
        order.sort_by(|&a, &b| ranked(b).total_cmp(&ranked(a)));
        order.truncate(c.top_k as usize);
        let sum: f32 = order.iter().map(|&e| score[e]).sum();
        let capped = |v: f32, cap: f32| cap * (v / cap).tanh();
        let mut out = vec![0.0; h];
        for e in order {
            let of =
                |name: &str, n: usize| &self.at(l, &format!("moe.experts.{name}"))[e * n..][..n];
            let (gate, up) = (lin(of("gate", ffn * h), x), lin(of("up", ffn * h), x));
            let inner: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| {
                    capped(g, c.mlp_gate_softcap) * sigmoid(g) * capped(u, c.mlp_up_softcap)
                })
                .collect();
            let weight = c.route_scale * score[e] / sum;
            for (o, d) in out.iter_mut().zip(lin(of("down", ffn * h), &inner)) {
                *o += weight * d;
            }
        }
        out
    }

    fn inputs(&self, l: usize, states: &[Mat], h: &Mat, msgs: Option<&[Mat; 4]>) -> [Mat; 3] {
        let (glob, eps) = (self.c.global[l], self.c.eps);
        let x = if l > 0 {
            self.attn_res(states, h, self.at(l, "attn_query"))
        } else {
            vec![h.clone()]
        };
        [("q", 0), ("k", 1), ("v", 2)].map(|(name, i)| {
            let side = &x[i.min(x.len() - 1)];
            let gated = match msgs {
                Some(m4) => self.gate(l, name, side, &m4[i]),
                None => side.clone(),
            };
            let norm = if glob {
                format!("{name}_norm")
            } else {
                "attn_norm".into()
            };
            let gain = self.at(l, &norm);
            gated.iter().map(|r| rms_norm(r, gain, eps)).collect()
        })
    }

    fn layer(&self, l: usize, states: &mut Vec<Mat>, h: &mut Mat, msgs: Option<&[Mat; 4]>) {
        let c = &self.c;
        let add = |a: &Mat, b: &Mat| -> Mat {
            let row =
                |(x, y): (&Vec<f32>, &Vec<f32>)| x.iter().zip(y).map(|(p, q)| p + q).collect();
            a.iter().zip(b).map(row).collect()
        };
        let [qi, ki, vi] = self.inputs(l, states, h, msgs);
        let att = self.attention(l, [&qi, &ki, &vi]);
        let a = self.static_conv(l, "attn_sconv", &att);
        let mid = if (l as u64).is_multiple_of(c.attn_res_block) {
            states.push(h.clone());
            a
        } else {
            add(h, &a)
        };
        let mut x = self
            .attn_res(states, &mid, self.at(l, "mlp_query"))
            .remove(0);
        if let Some(m4) = msgs {
            x = self.gate(l, "mlp", &x, &m4[3]);
        }
        let gain = self.at(l, "mlp_norm");
        let moe = |r: &Vec<f32>| self.moe(l, &rms_norm(r, gain, c.eps));
        let out: Mat = x.iter().map(moe).collect();
        *h = add(&mid, &self.static_conv(l, "mlp_sconv", &out));
    }

    /// Logits of one sequence from position zero.
    fn logits(&self, tokens: &[u32]) -> Mat {
        let c = &self.c;
        let [width, layers] = [c.stack.hidden, c.stack.layers].map(|n| n as usize);
        let [start, end] = [c.loop_start, c.loop_end].map(|n| n as usize);
        let embed = |&t: &u32| {
            let row = &self.top("embed")[t as usize * width..][..width];
            rms_norm(row, self.top("embed_norm"), c.eps)
        };
        let mut h: Mat = tokens.iter().map(embed).collect();
        let mut states = Vec::new();
        for l in 0..start {
            self.layer(l, &mut states, &mut h, None);
        }
        let (pre_h, pre_states) = (h.clone(), states.clone());
        for l in start..end {
            self.layer(l, &mut states, &mut h, None);
        }
        for _ in 1..c.loop_passes {
            let prev = states.split_off((pre_states.len() + 1).min(states.len()));
            let prev_h = std::mem::replace(&mut h, pre_h.clone());
            states = pre_states.clone();
            for (j, l) in (start..end).enumerate() {
                let two = |name: &str| {
                    let query = &self.top(name)[2 * j * width..][..2 * width];
                    self.attn_res(&prev, &prev_h, query)
                };
                let [q, mlp] = <[Mat; 2]>::try_from(two("cross_pass_queries")).unwrap();
                let [k, v] = <[Mat; 2]>::try_from(two("cross_pass_kv_queries")).unwrap();
                self.layer(l, &mut states, &mut h, Some(&[q, k, v, mlp]));
            }
        }
        for l in end..layers {
            self.layer(l, &mut states, &mut h, None);
        }
        let x = self
            .attn_res(&states, &h, self.top("output_query"))
            .remove(0);
        let cap = c.final_logit_softcap.unwrap();
        let head = &self.top("head")[..c.unpadded_vocab as usize * width];
        let token = |r: &Vec<f32>| {
            let scores = lin(head, &rms_norm(r, self.top("norm"), c.eps));
            scores.iter().map(|z| cap * (z / cap).tanh()).collect()
        };
        x.iter().map(token).collect()
    }
}

fn filled(g: &Graph) -> Vec<Tensor> {
    weights_with(g, &mut Noise(7), |w| {
        let last = w.path.last().map(|s| format!("{s:?}")).unwrap_or_default();
        if last.contains("norm") || last.contains("clip") {
            (1.0, 0.2)
        } else {
            (0.0, 0.4)
        }
    })
}

fn tokens(seq: u32, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|t| (seq * 17 + t * 7 + t * t) % 60)
        .collect()
}

fn worst(got: &[f32], want: &Mat) -> f32 {
    let flat = want.iter().flatten();
    assert_eq!(got.len(), flat.clone().count());
    got.iter()
        .zip(flat)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn forward_matches_reference() {
    let c = Config::small();
    let g = program::<Plain>(&c);
    let ws = filled(&g);
    let reference = Ref::new(&c, &g, &ws);
    let n = 12;
    let ids: Vec<u32> = (0..2).flat_map(|s| tokens(s, n)).collect();
    let pos: Vec<u32> = (0..2).flat_map(|_| 0..n as u32).collect();
    let feeds = [Tensor::u32(vec![2, n], ids), Tensor::u32(vec![2, n], pos)];
    let out = run(&g, &ws, 0, &feeds).unwrap();
    let got = out[0].floats().unwrap();
    let want: Mat = (0..2)
        .flat_map(|s| reference.logits(&tokens(s, n)))
        .collect();
    let err = worst(got, &want);
    assert!(err <= 1e-4, "forward off by {err}");
}

#[test]
fn step_matches_reference() {
    let c = Config::small();
    let g = stepper::<Plain>(&c);
    let ws = filled(&g);
    let reference = Ref::new(&c, &g, &ws);
    let n = 12;
    let want: Vec<Mat> = (0..2).map(|s| reference.logits(&tokens(s, n))).collect();
    let mut store = Store::default();
    // Uneven launches, the two slots interleaved.
    let launches: [&[(u32, usize)]; 5] = [
        &[(0, 0), (0, 1), (0, 2), (1, 0)],
        &[(1, 1), (0, 3), (1, 2), (1, 3), (1, 4)],
        &[(0, 4), (0, 5), (0, 6), (0, 7), (1, 5), (1, 6)],
        &[(0, 8), (1, 7), (0, 9), (1, 8), (1, 9), (0, 10)],
        &[(0, 11), (1, 10), (1, 11)],
    ];
    for rows in launches {
        let column = |f: &dyn Fn(u32, usize) -> u32| {
            Tensor::u32(
                vec![rows.len()],
                rows.iter().map(|&(s, t)| f(s, t)).collect(),
            )
        };
        let feeds = [
            column(&|s, t| tokens(s, n)[t]),
            column(&|_, t| t as u32),
            column(&|s, _| s),
        ];
        let out = run_kept(&g, &ws, &mut store, 0, &feeds).unwrap();
        let logits = out.iter().rev().find_map(|t| t.floats().ok()).unwrap();
        let rows: Mat = rows
            .iter()
            .map(|&(s, t)| want[s as usize][t].clone())
            .collect();
        let err = worst(logits, &rows);
        assert!(err <= 1e-4, "step off by {err}");
    }
}

fn metal() -> Platform {
    Platform::new(Asked {
        backend: "metal",
        arch: "apple9".into(),
        name: "apple9".into(),
        driver: String::new(),
        limits: tessera_gpukit::Limits::assumed(),
        units: 0,
        bandwidth: 0.0,
        integrated: true,
        memory: 0,
    })
}

#[test]
fn graphs_lower() {
    let c = Config::small();
    let wide = Config {
        stack: tessera_models::Stack::sized([64, 128, c.stack.layers, 4, 4, 128], 2, 16),
        ..Config::small()
    };
    let whole = [(Seqs, 2), (Context, 16)];
    let ragged = [(Seqs, 2), (Rows, 4), (Context, 16)];
    for device in [Device::guess(&host()), Device::guess(&host()).on(metal())] {
        lower(&program::<Plain>(&c), 0, &whole, &device).unwrap();
        lower(&stepper::<Plain>(&c), 0, &ragged, &device).unwrap();
        lower(&stepper::<Coded>(&wide), 0, &ragged, &device).unwrap();
    }
}

/// Experts at mixed widths take the fast Metal launch.
#[test]
fn metal_experts_fast() {
    let small = Config::small();
    let wide = Config {
        stack: tessera_models::Stack::sized([64, 128, small.stack.layers, 4, 4, 128], 2, 16),
        ..small
    };
    let g = stepper::<Coded>(&wide);
    let mixed = tessera_lower::form::named("trellis_mix_pmcgx").unwrap();
    let mut pins = tessera_lower::Pins::default();
    for set in ["gate", "up", "down"] {
        pins = pins.pin(&format!("*.{set}"), mixed);
    }
    let device = Device::guess(&host()).on(metal()).pinned(pins);
    let plan = lower(&g, 0, &[(Seqs, 2), (Rows, 4), (Context, 16)], &device).unwrap();
    let fast = |s: &&tessera_lower::Step| match s.kernel.source() {
        tessera_lower::Source::Fixed { entry, .. } => entry == "tm_pickmix_pmcgx_t",
        tessera_lower::Source::Made(_) => false,
    };
    // Gate, up and down of each layer pass.
    let count = plan.steps.iter().filter(fast).count() as u64;
    assert!(count >= 3 * wide.stack.layers && count.is_multiple_of(3));
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(!names.iter().any(|n| n.ends_with("picked")), "{names:?}");
}
