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
