//! Rows scaled by the log of their position.

use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::{Kernel, stage_rule};

/// One thread per row. `per` rows share a position.
#[kernel]
fn log_scale(
    x: Buf<F32>,
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    floor: f32,
    alpha: f32,
    rows: u32,
    per: u32,
    width: u32,
) {
    if gid() >= rows {
        return;
    }
    let count = (positions[(gid() / per) as u64] + 1) as f32;
    let tau = 1.0 + alpha * log(max(1.0, count / floor));
    let at = gid() as u64 * width as u64;
    for i in 0..width {
        out[at + i as u64] = x[at + i as u64] * tau;
    }
}

/// Rows scaled by position, `per` rows to a position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(log_scale, threads = self.rows)]
pub struct LogScale {
    pub floor: f32,
    pub alpha: f32,
    pub rows: usize,
    pub per: usize,
    pub width: usize,
}

stage_rule!("log_scale", |site| {
    let (Stage::LogScale { floor, alpha }, Some((_, positions))) = (&site.stage, site.operand)
    else {
        return None;
    };
    let rows = site.len / site.width;
    Some(LogScale {
        floor: *floor,
        alpha: *alpha,
        rows,
        per: rows / positions.max(1),
        width: site.width,
    })
});
