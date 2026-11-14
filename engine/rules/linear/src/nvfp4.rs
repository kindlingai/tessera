//! Native nvfp4 blocks across one SIMD group.

use crate::lanes::block_kernel;

block_kernel! {
    NVFP4, 256, 2, "lanes_nvfp4";
    #[kernel]
    fn linear(x, w, k, col, s, m, form) {
        let head = bits_to_f32(w.u32_le(0));
        let sub = lane() & 1;
        for bk in ((lane() / 2) * 16..k).by(256).unroll(2) {
            let blk = bk / 16;
            let base = 4 + (col as u64 * (k / 16) as u64 + blk as u64) * 9;
            let sc = head * e4m3_to_f32(w.byte(base));
            let p = w.pack4(base + 1 + (sub * 4) as u64);
            let mut codes = [0.0f32; 8];
            for j in (0..8).unroll(8) {
                codes[j as usize] = sc * e2m1_to_f32((p >> (4 * j)) & 15);
            }
            let at = bk + sub * 8;
            accumulate(x, s, k, at, codes[0], codes[1], codes[2], codes[3], m);
            accumulate(x, s, k, at + 4, codes[4], codes[5], codes[6], codes[7], m);
        }
    }
}
