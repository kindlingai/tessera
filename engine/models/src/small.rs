//! Small configurations: two layers, wide enough for every block form.

use crate::decoder::{Config, Scales};
use crate::mla::{self, Index};
use crate::shared::Stack;
use crate::{eg2, gemma4, nanogpt, tagger};

/// Two layers of two heads over a vocabulary of 64.
pub fn stack(hidden: u64, head_dim: u64, ffn: u64) -> Stack {
    Stack::sized([64, hidden, 2, 2, head_dim, ffn], 4, 16)
}

pub fn qwen3() -> Config {
    Config {
        stack: stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-6,
        experts: 0,
        top_k: 0,
        scales: Scales::default(),
    }
}

/// Four experts, two per row, two KV heads.
pub fn granite() -> Config {
    Config {
        stack: stack(64, 32, 128),
        kv_heads: 2,
        rope_theta: 1e4,
        eps: 1e-6,
        experts: 4,
        top_k: 2,
        scales: Scales {
            embed: 2.0,
            attention: Some(0.1),
            residual: 0.5,
            logits: 0.25,
        },
    }
}

pub fn smollm2() -> Config {
    Config {
        stack: stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-5,
        experts: 0,
        top_k: 0,
        scales: Scales::default(),
    }
}

/// One sliding layer and one full.
pub fn eg2() -> eg2::Config {
    eg2::Config {
        stack: stack(256, 128, 256),
        kv_heads: 1,
        full_every: 2,
        full_head_dim: 256,
        per_layer: 128,
        embedding: 128,
        window: 4,
        theta: 1e4,
        full_theta: 1e6,
        eps: 1e-6,
    }
}

/// Four layers, sliding then full. The last two share keys.
pub fn gemma4() -> gemma4::Config {
    let mut stack = stack(256, 128, 256);
    stack.layers = 4;
    gemma4::Config {
        stack,
        kv_heads: 1,
        full_every: 2,
        full_head_dim: 256,
        full_turned: 32,
        per_layer: 128,
        window: 4,
        theta: 1e4,
        full_theta: 1e6,
        first_shared: 2,
        eps: 1e-6,
    }
}

/// Two features of eight rows, one window layer.
pub fn tagger() -> tagger::Config {
    tagger::Config {
        rows: vec![8, 8],
        width: 32,
        depth: 1,
        tags: 32,
        max_context: 16,
    }
}

/// Two layers of two heads, four wide.
pub fn nanogpt() -> nanogpt::Config {
    nanogpt::Config {
        stack: Stack::sized([8, 8, 2, 2, 4, 32], 4, 8),
        eps: 1e-5,
    }
}

/// One dense layer, two routed. The middle reuses picked keys.
pub fn glm53() -> mla::Config {
    let index = Index {
        heads: 2,
        dim: 32,
        keep: 4,
        every: 2,
    };
    // Rotations turn 128 channels at a time.
    mla::Config {
        stack: Stack::sized([64, 128, 3, 2, 32, 128], 4, 16),
        index: Some(index),
        ..deepseek()
    }
}

/// Three layers of low-rank attention. Weight rows hold whole blocks.
pub fn deepseek() -> mla::Config {
    mla::Config {
        stack: Stack::sized([64, 64, 3, 2, 32, 32], 4, 16),
        dense_layers: 1,
        nope: 16,
        rope: 16,
        rope_theta: 1e4,
        q_latent: 32,
        latent: 32,
        dense_ffn: 64,
        experts: 32,
        top_k: 2,
        route_scale: 2.5,
        index: None,
        eps: 1e-6,
    }
}
