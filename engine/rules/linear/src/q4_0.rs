//! Native q4_0 blocks across one SIMD group.

use crate::lanes::f16_kernel;

f16_kernel! {
    Q4_0, 18, 256, 2, 4, "lanes_q4_0";
    #[kernel]
    fn linear(x, w, k, col, s, m, form)
    for bk in [(lane() / 4) * 32..k] => (sub, sc, p) {
        let lo0 = sc * (((p & 15) as f32) - 8.0);
        let lo1 = sc * ((((p >> 8) & 15) as f32) - 8.0);
        let lo2 = sc * ((((p >> 16) & 15) as f32) - 8.0);
        let lo3 = sc * ((((p >> 24) & 15) as f32) - 8.0);
        let hi0 = sc * ((((p >> 4) & 15) as f32) - 8.0);
        let hi1 = sc * ((((p >> 12) & 15) as f32) - 8.0);
        let hi2 = sc * ((((p >> 20) & 15) as f32) - 8.0);
        let hi3 = sc * ((((p >> 28) & 15) as f32) - 8.0);
        let at = bk + sub * 4;
        accumulate(x, s, k, at, lo0, lo1, lo2, lo3, m);
        accumulate(x, s, k, at + 16, hi0, hi1, hi2, hi3, m);
    }
}
