use crate::quant::{Quant, Scheme};
use crate::{File, Offer, Shards, Stored};
use std::path::PathBuf;
use tessera_engine::storage::Form;
use tessera_engine::{Bf16, E4M3, block};

struct TensorSpec<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [usize],
    data: Vec<u8>,
}

fn spec<'a>(name: &'a str, dtype: &'a str, shape: &'a [usize], data: Vec<u8>) -> TensorSpec<'a> {
    TensorSpec {
        name,
        dtype,
        shape,
        data,
    }
}

/// Builds a safetensors directory with optional `config.json`.
fn stage_dir(tag: &str, config: Option<&str>, tensors: &[TensorSpec<'_>]) -> PathBuf {
    let dir = tessera_testkit::stage(&format!("quant_{tag}.stamp"), b"")
        .parent()
        .unwrap()
        .join(format!("quant_{tag}_dir"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(cfg) = config {
        std::fs::write(dir.join("config.json"), cfg).unwrap();
    }
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for t in tensors {
        let start = data.len();
        data.extend_from_slice(&t.data);
        let end = data.len();
        header.insert(
            t.name.to_string(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    dir
}

/// Splits interleaved block bytes into `(head, scales, codes)`.
fn split_blocks(
    packed: &[u8],
    head_bytes: usize,
    scale_bytes: usize,
    code_bytes: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let head = packed[..head_bytes].to_vec();
    let stride = scale_bytes + code_bytes;
    let groups = (packed.len() - head_bytes) / stride;
    let mut scales = Vec::with_capacity(groups * scale_bytes);
    let mut codes = Vec::with_capacity(groups * code_bytes);
    for g in 0..groups {
        let at = head_bytes + g * stride;
        scales.extend_from_slice(&packed[at..at + scale_bytes]);
        codes.extend_from_slice(&packed[at + scale_bytes..at + stride]);
    }
    (head, scales, codes)
}

fn sample_data(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * (((i * 29 + 7) % 100) as f32 / 25.0)
        })
        .collect()
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| Bf16::from(v).0.to_le_bytes())
        .collect()
}

fn verify_block_pair(offer: &dyn Offer, name: &str, want: &[u8], row_bytes: usize, head: usize) {
    assert_eq!(offer.packed(name).unwrap(), want);
    let mut want_rows = want[..head].to_vec();
    want_rows.extend_from_slice(&want[head + row_bytes..head + 3 * row_bytes]);
    assert_eq!(offer.packed_rows(name, (1, 2)).unwrap(), want_rows);
    assert_eq!(offer.packed_axis(name, 0, (1, 2)).unwrap(), want_rows);
    assert!(offer.filed(name).is_none());
}

#[test]
fn parses_quant_schemes() {
    assert_eq!(
        Scheme::parse(
            r#"{"quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]}}"#
        ),
        Some(Scheme::Fp8 {
            block: Some([128, 128])
        })
    );
    assert_eq!(
        Scheme::parse(r#"{"text_config":{"quantization_config":{"quant_method":"mxfp4"}}}"#),
        Some(Scheme::MxFp4)
    );
    assert_eq!(
        Scheme::parse(
            r#"{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}"#
        ),
        Some(Scheme::NvFp4)
    );
    assert_eq!(
        Scheme::parse(r#"{"quantization_config":{"quant_method":"compressed-tensors"}}"#),
        Some(Scheme::Mixed)
    );
    assert_eq!(Scheme::parse(r#"{"hidden_size":64}"#), None);
}

#[test]
fn pairs_quant_forms() {
    let vals = sample_data(4 * 64);
    let mxfp8_want = block::MXFP8.pack(&vals);
    let (_, mx8_s, mx8_c) = split_blocks(&mxfp8_want, 0, 1, 32);
    let mxfp4_want = block::MXFP4.pack(&vals);
    let (_, mx4_s, mx4_c) = split_blocks(&mxfp4_want, 0, 1, 16);
    let nvfp4_want = block::NVFP4.pack(&vals);
    let (nv_h, nv_s, nv_c) = split_blocks(&nvfp4_want, 4, 1, 8);
    let inv_head = (1.0f32 / f32::from_le_bytes(nv_h.clone().try_into().unwrap()))
        .to_le_bytes()
        .to_vec();
    let fp8_codes: Vec<u8> = (0..4 * 4).map(|i| E4M3::from((i as f32) - 7.5).0).collect();
    let fp8_scales: Vec<u8> = [0.5f32, 2.0, 1.5, 0.25]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let norm = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);

    let cfg = r#"{"quantization_config":{"quant_method":"compressed-tensors"}}"#;
    let dir = stage_dir(
        "mixed",
        Some(cfg),
        &[
            spec("l0.q_proj.weight", "F8_E4M3", &[4, 64], mx8_c),
            spec("l0.q_proj.weight_scale", "U8", &[4, 2], mx8_s),
            spec("l0.k_proj_blocks", "U8", &[4, 2, 16], mx4_c),
            spec("l0.k_proj_scales", "U8", &[4, 2], mx4_s),
            spec("l0.v_proj.weight_packed", "U8", &[4, 32], nv_c.clone()),
            spec("l0.v_proj.weight_scale", "F8_E4M3", &[4, 4], nv_s.clone()),
            spec("l0.v_proj.weight_scale_2", "F32", &[1], nv_h),
            spec("l0.o_proj.weight", "U8", &[4, 32], nv_c),
            spec("l0.o_proj.weight_scale", "F8_E4M3", &[4, 4], nv_s),
            spec("l0.o_proj.weight_global_scale", "F32", &[1], inv_head),
            spec("l0.gate_proj.weight", "F8_E4M3", &[4, 4], fp8_codes.clone()),
            spec("l0.gate_proj.weight_scale_inv", "F32", &[2, 2], fp8_scales),
            spec("l0.norm.weight", "BF16", &[4], norm.clone()),
        ],
    );

    let offer = crate::open(&dir, None).unwrap();
    let slices = offer.slices();
    let find = |n: &str| slices.iter().find(|s| s.name == n).unwrap().clone();
    assert_eq!(slices.len(), 6);
    assert_eq!(find("l0.q_proj.weight").stored, Stored::Form(&block::MXFP8));
    assert_eq!(find("l0.k_proj.weight").stored, Stored::Form(&block::MXFP4));
    assert_eq!(find("l0.v_proj.weight").stored, Stored::Form(&block::NVFP4));
    assert_eq!(find("l0.o_proj.weight").stored, Stored::Form(&block::NVFP4));
    assert_eq!(find("l0.gate_proj.weight").stored, Stored::BF16);
    assert_eq!(find("l0.norm.weight").stored, Stored::BF16);
    assert_eq!(find("l0.k_proj.weight").shape, vec![4, 64]);
    assert_eq!(find("l0.v_proj.weight").shape, vec![4, 64]);

    verify_block_pair(&*offer, "l0.q_proj.weight", &mxfp8_want, 2 * 33, 0);
    verify_block_pair(&*offer, "l0.k_proj.weight", &mxfp4_want, 2 * 17, 0);
    verify_block_pair(&*offer, "l0.v_proj.weight", &nvfp4_want, 4 * 9, 4);
    assert_eq!(
        offer.floats("l0.o_proj.weight").unwrap(),
        block::NVFP4.unpack_all(&nvfp4_want)
    );

    let col_share = offer.packed_axis("l0.q_proj.weight", 1, (32, 32)).unwrap();
    let want_col = block::MXFP8
        .share(&mxfp8_want, [4, 64, 1], [32, 32])
        .unwrap();
    assert_eq!(col_share, want_col);

    let got_fp8 = offer.floats("l0.gate_proj.weight").unwrap();
    let scales = [0.5f32, 2.0, 1.5, 0.25];
    let want_fp8: Vec<f32> = (0..16)
        .map(|i| f32::from(E4M3(fp8_codes[i])) * scales[(i / 4 / 2) * 2 + (i % 4) / 2])
        .collect();
    assert_eq!(got_fp8, want_fp8);
    assert_eq!(
        offer.packed_rows("l0.gate_proj.weight", (2, 2)).unwrap(),
        bf16_bytes(&want_fp8[8..16])
    );
    assert!(offer.filed("l0.norm.weight").is_some());
    assert_eq!(offer.packed("l0.norm.weight").unwrap(), norm);

    // Direct `Quant::new` with explicit `Scheme::Fp8`.
    let shards = Shards::from_files(vec![File::open(&dir.join("model.safetensors")).unwrap()]);
    let fp8_only = Quant::new(
        shards,
        Scheme::Fp8 {
            block: Some([2, 2]),
        },
    );
    assert_eq!(fp8_only.floats("l0.gate_proj.weight").unwrap(), want_fp8);

    std::fs::remove_dir_all(dir).unwrap();
}
