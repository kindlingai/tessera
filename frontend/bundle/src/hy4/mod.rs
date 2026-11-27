//! Tencent Hy4-flash (`hy_v4_flash`) directory loading and weight rules.

#[cfg(test)]
mod hy4_test;

use crate::arch::{Arch, Ask, Built, Spot, arch, open_streamed, parse_keys, push_routed_mlp};
use crate::names::Rule;
use crate::parts::{Cut, Part, from_streamed};
use std::error::Error;
use tessera_engine::hy4::{Config, PlainLatent, RotatedLatent, program, stepper};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

arch!(
    "hy4",
    [
        Spot::Type("hy_v4_flash"),
        Spot::Class("HYV4FlashForCausalLM"),
        Spot::Class("HYV4FlashForConditionalGeneration"),
    ],
    step: step,
    forward: forward,
    part: part,
);

const MLP: [(&str, &str); 3] = [
    ("gate", "gate_proj"),
    ("up", "up_proj"),
    ("down", "down_proj"),
];

fn build(ask: &Ask, is_step: bool) -> Result<Built> {
    let cfg = parse_config(ask.json, ask.sizes)?;
    open_streamed(
        ask,
        is_step,
        cfg,
        |c| &mut c.stack,
        rules,
        [
            program::<PlainLatent>,
            stepper::<PlainLatent>,
            program::<RotatedLatent>,
            stepper::<RotatedLatent>,
        ],
    )
}

fn step(_: &Arch, ask: &Ask) -> Result<Built> {
    build(ask, true)
}

fn forward(_: &Arch, ask: &Ask) -> Result<Built> {
    build(ask, false)
}

fn part(_: &Arch, cut: &Cut) -> Result<Part> {
    let cfg = parse_config(cut.json, cut.sized(256))?;
    let layers = cfg.stack.layers;
    from_streamed(
        cut,
        cfg,
        layers,
        rules,
        [stepper::<PlainLatent>, stepper::<RotatedLatent>],
    )
}

pub(crate) fn parse_config(json: &str, [seqs, context]: [u64; 2]) -> Result<Config> {
    parse_keys(json, |keys| Config::read(keys, seqs, context))
}

/// Maps traced Hy4-flash weights to checkpoint tensor names.
pub fn rules(c: &Config, rotated: bool) -> Vec<Rule> {
    let mut rules = vec![
        ("embed".into(), "model.embed_tokens".into()),
        ("norm".into(), "model.norm".into()),
        ("hc_head.fn_w".into(), "model.hc_head.hc_head_fn".into()),
        ("hc_head.base".into(), "model.hc_head.hc_head_base".into()),
        ("hc_head.scale".into(), "model.hc_head.hc_head_scale".into()),
    ];
    if !c.tied {
        rules.push(("head".into(), "lm_head".into()));
    }
    let (mut di, mut si, mut mi) = (0usize, 0usize, 0usize);
    for l in 0..c.stack.layers as usize {
        let at = format!("model.layers.{l}");
        for (dst, src) in [
            (format!("attn_hc.{l}"), format!("{at}.hc_attn_layer.hc_pre")),
            (format!("mlp_hc.{l}"), format!("{at}.hc_mlp_layer.hc_pre")),
        ] {
            rules.push((format!("{dst}.fn_w"), format!("{src}.hc_fn")));
            rules.push((format!("{dst}.base"), format!("{src}.hc_base")));
            rules.push((format!("{dst}.scale"), format!("{src}.hc_scale")));
        }
        if let Some(slot) = c.engram_slot(l) {
            let ep = format!("engram.{slot}");
            let src = format!("{at}.engram_layer");
            rules.push((
                format!("{ep}.head_norms"),
                format!("{src}.lookup_per_head_norms.{{}}"),
            ));
            rules.push((format!("{ep}.val_w"), format!("{src}.value_proj")));
            rules.push((format!("{ep}.val_b"), format!("{src}.value_proj.bias")));
            rules.push((
                format!("{ep}.pre_gamma"),
                format!("{src}.unified_pre_gamma"),
            ));
            for hc in 0..c.hc_mult as usize {
                rules.push((format!("{ep}.key_w.{hc}"), format!("{src}.key_projs.{hc}")));
                rules.push((
                    format!("{ep}.key_b.{hc}"),
                    format!("{src}.key_projs.{hc}.bias"),
                ));
                rules.push((
                    format!("{ep}.norm_key.{hc}"),
                    format!("{src}.norm_key.{hc}"),
                ));
                rules.push((
                    format!("{ep}.norm_query.{hc}"),
                    format!("{src}.norm_query.{hc}"),
                ));
                rules.push((
                    format!("{ep}.hc_scales.{hc}"),
                    format!("{src}.unified_per_hc_scales.{hc}"),
                ));
            }
        }
        let ap = if c.swa_pattern[l] {
            let p = format!("swa.{si}");
            si += 1;
            p
        } else {
            let p = format!("dsa.{di}");
            di += 1;
            p
        };
        rules.push((format!("{ap}.input_norm"), format!("{at}.input_layernorm")));
        for (dst, src) in [
            ("q_a", "q_a_proj"),
            ("q_norm", "q_a_layernorm"),
            ("q_b", "q_b_proj"),
            ("kv_a", "kv_a_proj_with_mqa"),
            ("kv_norm", "kv_a_layernorm"),
            ("kv_b", "kv_b_proj"),
            ("linear_gate", "linear_gate"),
            ("o", "o_proj"),
            ("post_proj_norm", "post_proj_norm"),
            ("sinks", "learnable_sink_param"),
        ] {
            rules.push((format!("{ap}.{dst}"), format!("{at}.self_attn.{src}")));
        }
        if (l as u64) < c.dense_layers {
            let dp = format!("dense.{l}");
            rules.push((
                format!("{dp}.post_attn_norm"),
                format!("{at}.post_attention_layernorm"),
            ));
            rules.push((
                format!("{dp}.post_fc2_norm"),
                format!("{at}.mlp.post_fc2_norm"),
            ));
            for (f, s) in MLP {
                rules.push((format!("{dp}.mlp.{f}"), format!("{at}.mlp.{s}")));
            }
        } else {
            let mp = format!("moe.{mi}");
            mi += 1;
            rules.push((
                format!("{mp}.post_attn_norm"),
                format!("{at}.post_attention_layernorm"),
            ));
            rules.push((format!("{mp}.router"), format!("{at}.mlp.gate")));
            rules.push((
                format!("{mp}.router_bias"),
                format!("{at}.mlp.gate.e_score_correction_bias"),
            ));
            rules.push((
                format!("{mp}.routed_down"),
                format!("{at}.mlp.routed_expert_down_proj"),
            ));
            rules.push((
                format!("{mp}.routed_up"),
                format!("{at}.mlp.routed_expert_up_proj"),
            ));
            rules.push((
                format!("{mp}.combine_norm"),
                format!("{at}.mlp.combine_output_norm"),
            ));
            push_routed_mlp(
                &mut rules,
                (&format!("{mp}.shared"), &format!("{at}.mlp.shared_experts")),
                (&format!("{mp}.routed"), "ex_", &format!("{at}.mlp.experts")),
                rotated,
            );
        }
    }
    rules
}
