//! Native q8_0 blocks across one SIMD group.

use crate::lanes::f16_kernel;

f16_kernel! {
    Q8_0, 34, 128, 4, 8, "lanes_q8_0";
    #[kernel]
    fn linear(x, w, k, col, s, m, form)
    for i in [lane() * 4..k] => (sub, sc, p) {
        let c0 = sc * s8_to_f32(p);
        let c1 = sc * s8_to_f32(p >> 8);
        let c2 = sc * s8_to_f32(p >> 16);
        let c3 = sc * s8_to_f32(p >> 24);
        accumulate(x, s, k, i, c0, c1, c2, c3, m);
    }
}
