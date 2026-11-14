//! The Walsh-Hadamard transform of 128-wide blocks, a SIMD group each.

use codegen::prelude::*;
use graph::Stage;
use lower::sites;
use lower::{Draft, Kernel, Needs, T, Tree};

/// Each lane holds four values of the block.
const PER_LANE: u32 = 4;

/// The block this kernel transforms.
pub const BLOCK: u32 = LANES as u32 * PER_LANE;

/// Lower keeps the sum, upper the difference.
#[kernel]
fn butterfly(a: f32, b: f32, upper: bool) -> f32 {
    if upper { b - a } else { a + b }
}

/// Four values a lane, then five exchanges across the lanes.
#[kernel]
fn hadamard(x: Buf<F32>, mut out: BufMut<F32>, blocks: u32) {
    let block = gid() / LANES as u32;
    if block >= blocks {
        return;
    }
    let at = block as u64 * BLOCK as u64 + lane() as u64 * PER_LANE as u64;
    let v = x.load4(at);
    // Distances one and two stay in the lane.
    let xy = v.x + v.y;
    let xy_minus = v.x - v.y;
    let zw = v.z + v.w;
    let zw_minus = v.z - v.w;
    let mut a = xy + zw;
    let mut b = xy_minus + zw_minus;
    let mut c = xy - zw;
    let mut d = xy_minus - zw_minus;
    for level in 0..5 {
        let mask = 1u32 << level;
        let upper = (lane() & mask) != 0;
        a = butterfly(a, shfl_xor(a, mask), upper);
        b = butterfly(b, shfl_xor(b, mask), upper);
        c = butterfly(c, shfl_xor(c, mask), upper);
        d = butterfly(d, shfl_xor(d, mask), upper);
    }
    // One over the square root of the block, 128.
    let scale = rsqrt(128.0);
    let mut o = F32x4::zero();
    o.x = a * scale;
    o.y = b * scale;
    o.z = c * scale;
    o.w = d * scale;
    out.store4(at, o);
}

/// `blocks` transforms of `BLOCK` values, one SIMD group each.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(hadamard, threads = self.blocks * LANES, needs = Needs::lanes())]
pub struct Hadamard {
    pub blocks: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let Stage::Hadamard { block } = site.stage else {
        return None;
    };
    if block != BLOCK {
        return None;
    }
    let hadamard = Hadamard {
        blocks: site.len / BLOCK as usize,
    };
    Some(vec![Draft::new(hadamard, [site.x], site.len)])
}

lower::rule!("hadamard", propose);
