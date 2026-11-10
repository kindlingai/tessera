//! Transformer layer ops: row gather, RoPE, attention.

// Fixtures are never empty, so `is_empty` would mean nothing.
#![allow(clippy::len_without_is_empty)]

pub mod apply_rope_heads;
pub mod gather_rows;
pub mod gqa_attention;
