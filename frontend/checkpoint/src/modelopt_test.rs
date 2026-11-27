use crate::modelopt::ModelOpt;
use crate::{Offer, Shards, Stored};
use serde_json::json;
use std::path::PathBuf;
use tessera_engine::storage::Form;
use tessera_engine::{Bf16, E4M3, block};

/// Writes a safetensors file from `(name, dtype, shape, bytes)` specs.
fn write_st(name: &str, specs: &[(&str, &str, &[usize], &[u8])]) -> PathBuf {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for &(key, dtype, shape, bytes) in specs {
        let from = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            key.into(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [from, data.len()]}),
        );
    }
    let text = serde_json::to_string(&header).unwrap();
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&data);
    tessera_testkit::stage(&format!("modelopt-{name}.safetensors"), &file)
}

/// Encodes `vals` into ModelOpt NVFP4 `(u8_weights, e4m3_scales, f32_scale2)`.
fn make_nvfp4(vals: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
    let packed = block::NVFP4.pack(vals);
    let head = f32::from_le_bytes(packed[..4].try_into().unwrap());
    let mut scales = Vec::new();
    let mut codes = Vec::new();
    for blk in packed[4..].as_chunks::<9>().0 {
        scales.push(blk[0]);
        codes.extend_from_slice(&blk[1..]);
    }
    (codes, scales, head)
}

/// Encodes `vals` into ModelOpt MXFP8 `(e4m3_weights, e8m0_scales)`.
fn make_mxfp8(vals: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let packed = block::MXFP8.pack(vals);
    let mut scales = Vec::new();
    let mut codes = Vec::new();
    for blk in packed.as_chunks::<33>().0 {
        scales.push(blk[0]);
        codes.extend_from_slice(&blk[1..]);
    }
    (codes, scales)
}

#[test]
fn decodes_mixed_formats() {
    // 2x32 NVFP4 matrix.
    let nv_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.03).collect();
    let (nv_w, nv_s, nv_s2) = make_nvfp4(&nv_vals);
    let nv_s2_b = nv_s2.to_le_bytes();
    let inp_s = 1.0f32.to_le_bytes();

    // 2x32 MXFP8 matrix.
    let mx_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 30.0) * 0.05).collect();
    let (mx_w, mx_s) = make_mxfp8(&mx_vals);

    // FP8 per-tensor and per-row scaled matrices.
    let fp8_raw: Vec<u8> = [0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 3.0]
        .iter()
        .map(|&v| E4M3::from(v).0)
        .collect();
    let fp8_ts = Bf16::from(0.25).0.to_le_bytes();
    let fp8_rs: Vec<u8> = [0.5f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();

    // Plain BF16 and I64 pass-through tensors.
    let bf16_b: Vec<u8> = [1.0f32, -0.5]
        .iter()
        .flat_map(|&v| Bf16::from(v).0.to_le_bytes())
        .collect();
    let i64_b: Vec<u8> = [11i64, 22].iter().flat_map(|v| v.to_le_bytes()).collect();

    let path = write_st(
        "mixed",
        &[
            ("a.weight", "U8", &[2, 16], &nv_w),
            ("a.weight_scale", "F8_E4M3", &[2, 2], &nv_s),
            ("a.weight_scale_2", "F32", &[], &nv_s2_b),
            ("a.input_scale", "F32", &[], &inp_s),
            ("b.weight", "F8_E4M3", &[2, 32], &mx_w),
            ("b.weight_scale", "F8_E8M0", &[2, 1], &mx_s),
            ("c.weight", "F8_E4M3", &[2, 4], &fp8_raw),
            ("c.weight_scale", "BF16", &[1], &fp8_ts),
            ("d.weight", "F8_E4M3", &[2, 4], &fp8_raw),
            ("d.weight_scale_inv", "F32", &[2, 1], &fp8_rs),
            ("e.weight", "BF16", &[2], &bf16_b),
            ("ids", "I64", &[2], &i64_b),
        ],
    );

    let offer = crate::open(&path, None).unwrap();
    let slices = offer.slices();
    let find = |n: &str| slices.iter().find(|s| s.name == n).cloned();
    assert!(find("a.weight_scale").is_none() && find("a.input_scale").is_none());
    assert_eq!(find("a.weight").unwrap().shape, [2, 32]);
    assert_eq!(
        find("a.weight").unwrap().stored,
        Stored::Form(&block::NVFP4)
    );
    assert_eq!(
        find("b.weight").unwrap().stored,
        Stored::Form(&block::MXFP8)
    );
    assert_eq!(find("c.weight").unwrap().stored, Stored::BF16);
    assert_eq!(find("d.weight").unwrap().stored, Stored::BF16);
    assert!(offer.filed("e.weight").is_some());
    assert!(offer.filed("a.weight").is_none());

    // Exact bit match with block::NVFP4 and block::MXFP8 packing.
    let nv_packed = offer.packed("a.weight").unwrap();
    assert_eq!(nv_packed, block::NVFP4.pack(&nv_vals));
    assert_eq!(
        offer.floats("a.weight").unwrap(),
        block::NVFP4.unpack_all(&nv_packed)
    );
    let nv_row1 = offer.packed_rows("a.weight", (1, 1)).unwrap();
    assert_eq!(
        block::NVFP4.unpack_all(&nv_row1),
        block::NVFP4.unpack_all(&nv_packed)[32..]
    );
    let nv_col = offer.packed_axis("a.weight", 1, (16, 16)).unwrap();
    let full_nv = block::NVFP4.unpack_all(&nv_packed);
    let want_col: Vec<f32> = full_nv[16..32]
        .iter()
        .chain(&full_nv[48..64])
        .copied()
        .collect();
    assert_eq!(block::NVFP4.unpack_all(&nv_col), want_col);

    assert_eq!(
        offer.packed("b.weight").unwrap(),
        block::MXFP8.pack(&mx_vals)
    );
    let want_c: Vec<f32> = fp8_raw.iter().map(|&b| f32::from(E4M3(b)) * 0.25).collect();
    assert_eq!(offer.floats("c.weight").unwrap(), want_c);
    let want_d: Vec<f32> = fp8_raw
        .iter()
        .enumerate()
        .map(|(i, &b)| f32::from(E4M3(b)) * if i < 4 { 0.5 } else { 2.0 })
        .collect();
    assert_eq!(offer.floats("d.weight").unwrap(), want_d);
    assert_eq!(offer.i64s("ids").unwrap(), [11, 22]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn splits_fused_experts() {
    // Two NVFP4 experts fused as [2, 4, 16].
    let e0: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.04).collect();
    let e1: Vec<f32> = (0..64).map(|i| (i as f32 - 40.0) * 0.09).collect();
    let (w0, s0, g0) = make_nvfp4(&e0);
    let (w1, s1, g1) = make_nvfp4(&e1);
    let mut w = w0;
    w.extend(w1);
    let mut s = s0;
    s.extend(s1);
    let s2: Vec<u8> = [g0, g1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = write_st(
        "fused",
        &[
            ("m.experts.gate_up_proj", "U8", &[2, 4, 8], &w),
            ("m.experts.gate_up_proj_scale", "F8_E4M3", &[2, 4, 1], &s),
            ("m.experts.gate_up_proj_scale_2", "F32", &[2], &s2),
            ("m.experts.down_proj", "U8", &[2, 4, 8], &w),
            ("m.experts.down_proj_scale", "F8_E4M3", &[2, 4, 1], &s),
            ("m.experts.down_proj_scale_2", "F32", &[2], &s2),
        ],
    );
    let mo = ModelOpt::new(Shards::from_files(vec![crate::File::open(&path).unwrap()]));
    let slices = mo.slices();
    let shape = |n: &str| slices.iter().find(|x| x.name == n).unwrap().shape.clone();
    assert_eq!(shape("m.experts.gate_proj.weight"), [2, 2, 16]);
    assert_eq!(shape("m.experts.up_proj.weight"), [2, 2, 16]);
    assert_eq!(shape("m.experts.down_proj.weight"), [2, 4, 16]);

    // Expert 1 slice uses its own exact scale2 without rescaling.
    let up_e1 = mo
        .packed_axis("m.experts.up_proj.weight", 0, (1, 1))
        .unwrap();
    let want_e1 = block::NVFP4.unpack_all(&block::NVFP4.pack(&e1))[32..64].to_vec();
    assert_eq!(block::NVFP4.unpack_all(&up_e1), want_e1);

    // Full gate tensor agrees within E4M3 rescaling tolerance.
    let gate_all = mo.floats("m.experts.gate_proj.weight").unwrap();
    let want_g0 = &block::NVFP4.unpack_all(&block::NVFP4.pack(&e0))[..32];
    let want_g1 = &block::NVFP4.unpack_all(&block::NVFP4.pack(&e1))[..32];
    for (&got, &exp) in gate_all.iter().zip(want_g0.iter().chain(want_g1)) {
        assert!((got - exp).abs() <= 0.1, "{got} vs {exp}");
    }
    std::fs::remove_file(path).unwrap();
}
