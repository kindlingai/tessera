use tessera_codegen::numerics::{Operand, Precision, Terms};

fn ref_tf32(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let v = f64::from(x);
    let ulp = 2f64.powi(v.abs().log2().floor() as i32 - 11 + 1);
    ((v / ulp).round() * ulp) as f32
}

fn ref_f16(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let exp = x.abs().log2().floor() as i32;
    let ulp = 2f32.powi((exp - 10).max(-24));
    let rounded = (x / ulp).round_ties_even() * ulp;
    if rounded.abs() > 65504.0 {
        return f32::INFINITY.copysign(x);
    }
    rounded
}

fn ref_acc(p: Precision, acc: f32, sum: f32) -> f32 {
    let total = acc + sum;
    if !p.truncates || total == 0.0 || !total.is_finite() {
        return total;
    }
    let v = f64::from(total);
    let ulp = 2f64.powi(v.abs().log2().floor() as i32 - p.acc_bits as i32 + 1);
    ((v / ulp).trunc() * ulp) as f32
}

fn same_bits(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[test]
fn split_bit_exact() {
    let mut samples = vec![
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc1_2345),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::from_bits(1),
        f32::from_bits(0x0040_0000),
        2f32.powi(-25),
        2f32.powi(-24),
        1.5 * 2f32.powi(-24),
        2.5 * 2f32.powi(-24),
        2f32.powi(-14) - 2f32.powi(-25),
        2f32.powi(-14),
        1.0 + 2f32.powi(-11),
        -(1.0 + 2f32.powi(-11)),
        2.0 - 2f32.powi(-12),
        65504.0,
        65519.996,
        65520.0,
        -65520.0,
        65536.0,
    ];
    // Sweep exponents and mantissa tie regions across all finite ranges.
    for exp in 0u32..255 {
        for &mant in &[
            0x0000_0000u32,
            0x0000_0fff,
            0x0000_1000,
            0x0000_1001,
            0x0000_3000,
            0x003f_f000,
            0x007f_e000,
            0x007f_efff,
            0x007f_f000,
            0x007f_ffff,
        ] {
            samples.push(f32::from_bits((exp << 23) | mant));
            samples.push(f32::from_bits(0x8000_0000 | (exp << 23) | mant));
        }
    }
    for &x in &samples {
        let (hi, lo) = Precision::CUDA.split(x);
        let rhi = ref_tf32(x);
        let rlo = ref_tf32(x - rhi);
        assert!(same_bits(hi, rhi), "tf32 hi {x:e}: {hi:?} vs {rhi:?}");
        assert!(same_bits(lo, rlo), "tf32 lo {x:e}: {lo:?} vs {rlo:?}");

        let (fhi, flo) = Precision::HIP.split(x);
        let rfhi = ref_f16(x);
        let rflo = ref_f16(x - rfhi);
        assert!(same_bits(fhi, rfhi), "f16 hi {x:e}: {fhi:?} vs {rfhi:?}");
        assert!(same_bits(flo, rflo), "f16 lo {x:e}: {flo:?} vs {rflo:?}");

        for bits in [16u32, 22, 24] {
            let p = Precision {
                operand: Operand::Tf32x2,
                depth: 8,
                acc_bits: bits,
                truncates: true,
            };
            let got = p.accumulate(0.0, x);
            let want = ref_acc(p, 0.0, x);
            assert!(
                same_bits(got, want),
                "acc({bits}) {x:e}: {got:?} vs {want:?}",
            );
        }
    }
}

#[test]
fn matmul_bound_modes() {
    let k = 13usize;
    let x: Vec<f32> = (0..3 * k).map(|i| (i as f32 * 0.37).sin()).collect();
    let w: Vec<f32> = (0..4 * k).map(|i| (i as f32 * 0.53).cos()).collect();
    let terms = Terms {
        tile: 16.0,
        lane: 5.0,
    };
    for p in [
        Precision::EXACT,
        Precision::METAL,
        Precision::CUDA,
        Precision::HIP,
    ] {
        let got = terms.matmul(&p, &x, &w, k);
        let adds = terms.tile as f32 * p.acc_error() + terms.lane as f32 * 2f32.powi(-24);
        let mut want = 0.0f32;
        for xi in x.chunks(k) {
            for wj in w.chunks(k) {
                let mut bound = 0.0f32;
                for i in 0..k {
                    let (a, b) = (xi[i].abs(), wj[i].abs());
                    bound += a * p.operand_error(wj[i]) + b * p.operand_error(xi[i]) + a * b * adds;
                }
                want = want.max(bound);
            }
        }
        let rel = (got - want).abs() / want.max(1e-30);
        assert!(rel < 1e-5, "{p:?}: got {got} vs want {want}");
    }
}
