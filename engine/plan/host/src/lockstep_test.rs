use crate::lockstep::{self, BLOCK, Dots, dots_form};
use tessera_codegen::form::{BF16_FORM, F32_FORM, Load};

#[test]
fn odd_length_dots() {
    let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..19).map(|i| (i % 3) as f32).collect();
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert_eq!(lockstep::dot(&a, &b), want);
    let halves = BF16_FORM.pack(&b);
    assert_eq!(lockstep::dot_bf16(&a, &halves), want);
    let bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(
        lockstep::dot_form(BF16_FORM, &bytes, 0, &halves, 0, 19),
        want
    );
}

#[test]
fn narrow_dots_match() {
    use tessera_codegen::form::{FP8_FORM, named};
    let f16 = named("f16").unwrap();
    let k = 131usize;
    let x: Vec<f32> = (0..(BLOCK + 3) * k)
        .map(|i| ((i * 19) % 23) as f32 * 0.25 - 2.75)
        .collect();
    let w: Vec<f32> = (0..BLOCK * k)
        .map(|i| ((i * 11) % 17) as f32 * 0.5 - 4.0)
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wf16 = f16.pack(&w);
    let wf16_ref = f16.unpack_all(&wf16);
    let wfp8 = FP8_FORM.pack(&w);
    let wfp8_ref = FP8_FORM.unpack_all(&wfp8);

    // Odd-length dot_form and direct helpers for f16 and fp8.
    let want_f16: f32 = x[..k].iter().zip(&wf16_ref[..k]).map(|(a, b)| a * b).sum();
    let want_fp8: f32 = x[..k].iter().zip(&wfp8_ref[..k]).map(|(a, b)| a * b).sum();
    assert!((lockstep::dot_form(f16, &xb, 0, &wf16, 0, k) - want_f16).abs() < 1e-3);
    assert!((lockstep::dot_form(FP8_FORM, &xb, 0, &wfp8, 0, k) - want_fp8).abs() < 1e-3);

    // Exhaustive binary16 widening check via tail and NEON dot_f16.
    let one = [1.0f32];
    let mut mask = [0.0f32; 8];
    for bits in 0u16..=u16::MAX {
        let raw = bits.to_le_bytes();
        let got = lockstep::dot_f16(&one, &raw);
        let want = f16.unpack(&raw, 0);
        if want.is_nan() {
            assert!(got.is_nan(), "f16 {bits:#06x}");
        } else {
            assert_eq!(got, want, "f16 {bits:#06x}");
        }
    }
    for range in [0u16..0x7C00, 0x8000u16..0xFC00] {
        for base in range.step_by(8) {
            let mut raw8 = [0u8; 16];
            for lane in 0..8 {
                let b = base + lane as u16;
                raw8[lane * 2..lane * 2 + 2].copy_from_slice(&b.to_le_bytes());
            }
            for lane in 0..8 {
                mask.fill(0.0);
                mask[lane] = 1.0;
                let got = lockstep::dot_f16(&mask, &raw8);
                let want = f16.unpack(&raw8[lane * 2..lane * 2 + 2], 0);
                assert_eq!(got, want, "neon f16 lane {lane} base {base:#06x}");
            }
        }
    }

    // Tile dots over f16, fp8, partial cols, and rows > BLOCK.
    let cases: [(&dyn Load, &[u8], &[f32]); 2] =
        [(f16, &wf16, &wf16_ref), (FP8_FORM, &wfp8, &wfp8_ref)];
    for (form, wb, wref) in cases {
        for (rows, cols) in [(3, 13), (BLOCK + 3, 21)] {
            let mut out = vec![0u8; rows * BLOCK * 4];
            let d = Dots {
                at: 0,
                stride: BLOCK,
                x_at: 0,
                w_at: 0,
                k,
                cols,
            };
            dots_form(form, &mut out, &xb, wb, &d, rows);
            for r in 0..rows {
                for c in 0..BLOCK {
                    let want: f32 = if c < cols {
                        (0..k).map(|i| x[r * k + i] * wref[c * k + i]).sum()
                    } else {
                        0.0
                    };
                    let at = (r * BLOCK + c) * 4;
                    let got = f32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                    assert!(
                        (got - want).abs() < 1e-2,
                        "{} r={r} c={c}: {got} vs {want}",
                        form.name()
                    );
                }
            }
        }
    }
}

/// Blocks of odd rows and depths against plain sums.
#[test]
fn dots_match_sums() {
    let (k, n) = (300, BLOCK);
    // Halves of small integers: exact in bf16 too.
    let value = |i: usize| (i % 13) as f32 * 0.5 - 3.0;
    let x: Vec<f32> = (0..BLOCK * k).map(value).collect();
    let w: Vec<f32> = (0..n * k).map(|i| value(i * 7)).collect();
    let wide: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let narrow: Vec<u8> = w
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let forms: [(&dyn Load, &[u8]); 2] = [(F32_FORM, &wide), (BF16_FORM, &narrow)];
    for (form, w_bytes) in forms {
        for (rows, cols) in [(1, n), (3, 13), (13, n), (BLOCK, 13)] {
            let mut out = vec![0u8; BLOCK * n * 4];
            let d = Dots {
                at: 0,
                stride: n,
                x_at: 0,
                w_at: 0,
                k,
                cols,
            };
            dots_form(form, &mut out, &xb, w_bytes, &d, rows);
            for (r, c) in (0..rows).flat_map(|r| (0..n).map(move |c| (r, c))) {
                // Columns past `cols` stay untouched.
                let want: f32 = match c < cols {
                    true => (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum(),
                    false => 0.0,
                };
                let at = (r * n + c) * 4;
                let got = f32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                assert!(
                    (got - want).abs() < 1e-3,
                    "{} rows {rows}: {got} {want}",
                    form.name()
                );
            }
        }
    }
}

#[test]
fn attend_matches_reference() {
    use crate::attend::{Query, Rows, attend_f32};

    type Case<'a> = (usize, usize, usize, usize, &'a [u32]);
    let cases: &[Case<'_>] = &[
        (13, 21, 0, 11, &[]),
        (16, 16, 3, 17, &[]),
        (128, 256, 2, 19, &[]),
        (13, 13, 5, 5, &[]),
        (13, 19, 2, 15, &[0, 3, 7, 14, 15, 11, 4, 2, 9]),
        (32, 64, 1, 18, &[1, 5, 9, 13, 17, 2, 6, 10, 14, 3]),
    ];
    for &(width, stride, from, to, picks) in cases {
        let total_keys = to.max(20);
        let q: Vec<f32> = (0..width)
            .map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.5)
            .collect();
        let k: Vec<f32> = (0..total_keys * stride)
            .map(|i| ((i * 11) % 17) as f32 * 0.07 - 0.4)
            .collect();
        let v: Vec<f32> = (0..total_keys * stride)
            .map(|i| ((i * 5) % 19) as f32 * 0.06 - 0.3)
            .collect();
        let ids: Vec<u8> = picks.iter().flat_map(|x| x.to_le_bytes()).collect();
        let scale = (width as f32).sqrt().recip();

        let keys: Vec<usize> = if picks.is_empty() {
            (from..to).collect()
        } else {
            picks
                .iter()
                .map(|&j| j as usize)
                .filter(|j| (from..to).contains(j))
                .collect()
        };
        let mut scores: Vec<f32> = keys
            .iter()
            .map(|&j| {
                let d: f32 = (0..width).map(|c| q[c] * k[j * stride + c]).sum();
                d * scale
            })
            .collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for s in &mut scores {
            *s = (*s - top).exp();
            denom += *s;
        }
        let mut want = vec![0.0f32; width];
        for (&j, &p) in keys.iter().zip(&scores) {
            for c in 0..width {
                want[c] += (p / denom) * v[j * stride + c];
            }
        }

        let qb: Vec<u8> = q.iter().flat_map(|x| x.to_le_bytes()).collect();
        let kb: Vec<u8> = k.iter().flat_map(|x| x.to_le_bytes()).collect();
        let vb: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let query = Query {
            out_at: 0,
            q_at: 0,
            k_at: 0,
            v_at: 0,
            stride,
            from,
            to,
            width,
            scale,
            list_at: 0,
            picks: picks.len(),
        };
        // Check both aligned and unaligned byte slices.
        for pad in [0usize, 1] {
            let mut ob = vec![0u8; pad + width * 4];
            let mut qp = vec![0u8; pad];
            qp.extend_from_slice(&qb);
            let mut kp = vec![0u8; pad];
            kp.extend_from_slice(&kb);
            let mut vp = vec![0u8; pad];
            vp.extend_from_slice(&vb);
            attend_f32(
                Rows {
                    out: &mut ob[pad..],
                    q: &qp[pad..],
                    k: &kp[pad..],
                    v: &vp[pad..],
                    ids: &ids,
                },
                &query,
            );
            for (c, &expected) in want.iter().enumerate() {
                let at = pad + c * 4;
                let got = f32::from_le_bytes(ob[at..at + 4].try_into().unwrap());
                assert!(
                    (got - expected).abs() < 1e-5,
                    "w={width} pad={pad} c={c}: {got} vs {expected}"
                );
            }
        }
    }
}
