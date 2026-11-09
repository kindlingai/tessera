//! Transformer layer ops: embedding, RoPE, attention, MoE routing.

// Fixtures are never empty, so `is_empty` would mean nothing.
#![allow(clippy::len_without_is_empty)]

pub mod apply_rope_heads;
pub mod embed_gather;
pub mod gather_rows;
pub mod gqa_attention;
pub mod moe_router_topk;
