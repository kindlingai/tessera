use crate::repack::{self, Repack};
use tessera_engine::form::{BF16_FORM, named};
use tessera_engine::{Bf16, Tensor, Weight};

#[test]
fn repacks_forms() {
    let mxfp8 = named("mxfp8").unwrap();
    let int4 = named("int4_g32").unwrap();
    let (rows, cols) = (32usize, 512usize);
    let bf16_f32: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            f32::from(Bf16::from(s * (((i * 37) % 1000) as f32 / 250.0)))
        })
        .collect();
    let bf16_bytes: Vec<u8> = bf16_f32
        .iter()
        .flat_map(|&v| Bf16::from(v).0.to_le_bytes())
        .collect();
    let pack = |w: Weight, share, target| {
        let Weight::Packed(_, bytes) =
            repack::repack_weight(w, Some([rows, cols]), share, target).unwrap()
        else {
            panic!("expected packed weight");
        };
        bytes
    };
    let full = pack(Weight::Packed(BF16_FORM, bf16_bytes.clone()), None, mxfp8);
    assert_eq!(full, mxfp8.pack(&bf16_f32));
    let round = mxfp8.unpack_all(&full);
    let (mut dot, mut na, mut nb, mut max_rel) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&x, &y) in bf16_f32.iter().zip(&round) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
        if x.abs() > 1e-3 {
            max_rel = max_rel.max((x - y).abs() / x.abs());
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    assert!(cos > 0.999 && max_rel < 0.08, "cos={cos} max_rel={max_rel}");
    let r_slice = pack(
        Weight::Packed(BF16_FORM, bf16_bytes.clone()),
        Some((0, 8, 16)),
        mxfp8,
    );
    assert_eq!(r_slice, mxfp8.pack(&bf16_f32[8 * cols..24 * cols]));
    let c_slice = pack(
        Weight::Packed(BF16_FORM, bf16_bytes.clone()),
        Some((1, 256, 256)),
        mxfp8,
    );
    let want_col: Vec<f32> = (0..rows)
        .flat_map(|r| bf16_f32[r * cols + 256..r * cols + 512].iter().copied())
        .collect();
    assert_eq!(c_slice, mxfp8.pack(&want_col));
    let from_f32 = pack(
        Weight::Floats(Tensor::f32(vec![rows, cols], bf16_f32.clone())),
        Some((0, 8, 16)),
        mxfp8,
    );
    assert_eq!(from_f32, r_slice);
    let as_int4 = pack(Weight::Packed(BF16_FORM, bf16_bytes), None, int4);
    assert_eq!(as_int4, int4.pack(&bf16_f32));
    assert!(Repack::into(int4).fits([32, 64]));
    assert!(!Repack::into(int4).fits([32, 16]));
}

#[test]
fn repacks_mxfp8_edges() {
    let mxfp8 = named("mxfp8").unwrap();
    let f16 = named("f16").unwrap();
    let int4 = named("int4_g32").unwrap();
    let (rows, cols) = (16usize, 256usize);
    let vals: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            s * (((i * 29) % 1000) as f32 / 128.0)
        })
        .collect();
    let f16_bytes = f16.pack(&vals);
    let f16_f32 = f16.unpack_all(&f16_bytes);
    let unpack = |w: Weight, share, target| {
        let Weight::Packed(_, bytes) =
            repack::repack_weight(w, Some([rows, cols]), share, target).unwrap()
        else {
            panic!("expected packed weight");
        };
        bytes
    };
    assert_eq!(
        unpack(Weight::Packed(f16, f16_bytes.clone()), None, mxfp8),
        mxfp8.pack(&f16_f32),
    );
    assert_eq!(
        unpack(
            Weight::Packed(f16, f16_bytes.clone()),
            Some((0, 4, 8)),
            mxfp8
        ),
        mxfp8.pack(&f16_f32[4 * cols..12 * cols]),
    );
    assert_eq!(
        unpack(
            Weight::Packed(f16, f16_bytes.clone()),
            Some((1, 64, 128)),
            mxfp8
        ),
        mxfp8.pack(
            &(0..rows)
                .flat_map(|r| f16_f32[r * cols + 64..r * cols + 192].iter().copied())
                .collect::<Vec<_>>()
        ),
    );
    assert_eq!(
        unpack(Weight::Packed(f16, f16_bytes), None, int4),
        int4.pack(&f16_f32),
    );

    // Exhaustive binary16 widening parity.
    for bits in 0u16..=u16::MAX {
        let got = repack::widen_f16(bits);
        let want = f16.unpack(&bits.to_le_bytes(), 0);
        if want.is_nan() {
            assert!(got.is_nan(), "f16 {bits:#06x} expected NaN, got {got}");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "f16 {bits:#06x}");
        }
    }

    // Edge MXFP8 blocks: zeros, subnormals, ties, overflow, Inf, NaN.
    let mut blocks: Vec<[f32; 32]> = vec![
        [0.0; 32],
        [-0.0; 32],
        [f32::MAX; 32],
        [f32::INFINITY; 32],
        [f32::NAN; 32],
    ];
    let mut mixed = [0.0f32; 32];
    for (i, slot) in mixed.iter_mut().enumerate() {
        *slot = match i {
            0 => 448.0,
            1 => -448.0,
            2 => 0.5 / 512.0,
            3 => 1.5 / 512.0,
            4 => 7.5 / 512.0,
            5 => 1.0 / 64.0,
            6 => f32::NAN,
            7 => -0.0,
            _ => ((i as f32) - 16.0) * 1.375,
        };
    }
    blocks.push(mixed);
    let mut huge_mix = [1.0e38f32; 32];
    huge_mix[0] = -3.0e38;
    huge_mix[1] = 0.0;
    huge_mix[2] = f32::NAN;
    blocks.push(huge_mix);

    let flat: Vec<f32> = blocks.iter().flatten().copied().collect();
    let expected = mxfp8.pack(&flat);
    let Weight::Packed(_, got_repack) = repack::repack_weight(
        Weight::Floats(Tensor::f32(vec![blocks.len(), 32], flat)),
        Some([blocks.len(), 32]),
        None,
        mxfp8,
    )
    .unwrap() else {
        panic!("expected packed weight");
    };
    assert_eq!(got_repack, expected);
    let mut scalar = Vec::with_capacity(blocks.len() * 33);
    for blk in &blocks {
        let mut dst = [0u8; 33];
        repack::pack_mxfp8_scalar(blk, &mut dst);
        scalar.extend_from_slice(&dst);
    }
    assert_eq!(scalar, expected);
}
