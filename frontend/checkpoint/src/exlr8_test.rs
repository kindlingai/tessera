use crate::exlr8::Release;
use crate::exlr8_fixture::{
    e4m3, floats, halves, ramp, record, serve, trellis_record, write, write_meta,
};
use crate::{Offer, Stored};
use tessera_engine::block::{INT4_G32, MXFP8};
use tessera_engine::storage::Form;
use tessera_engine::trellis;

/// Each format decodes as the release's README says.
#[test]
fn glm_dense_formats() {
    let dir = tessera_testkit::scratch(&format!("{}-glm-exlr8", std::process::id()));
    let fp8 = ramp(2 * 128, 7);
    let (e8m0, block) = (vec![126u8, 128, 127, 129], [0.5f32]);
    let (q4, s4) = (ramp(2 * 32, 11), [0.5f32, 1.0, 2.0, 4.0]);
    let (q4_in, s4_in) = (
        ramp(2 * 64, 13),
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let mut tr = record(1, 128, 128, 1);
    tr[24..32].copy_from_slice(&2i64.to_le_bytes());
    write(
        &dir.join("dense/exlr8-dense-000.safetensors"),
        &[
            ("mx.record", "I64", vec![8], record(3, 2, 64, 0)),
            ("mx.fp8", "U8", vec![2, 64], fp8[..128].to_vec()),
            ("mx.e8m0", "U8", vec![2, 2], e8m0.clone()),
            ("blk.record", "I64", vec![8], record(4, 2, 128, 0)),
            ("blk.fp8", "U8", vec![2, 128], fp8.clone()),
            ("blk.scale", "F32", vec![1, 1], floats(&block)),
            ("q.record", "I64", vec![8], record(2, 2, 64, 1)),
            ("q.q4", "U8", vec![2, 32], q4.clone()),
            ("q.s4", "F16", vec![2, 2], halves(&s4)),
            ("qi.record", "I64", vec![8], record(2, 2, 128, 2)),
            ("qi.q4", "U8", vec![1, 2, 64], q4_in.clone()),
            ("qi.s4", "F16", vec![1, 2, 4], halves(&s4_in)),
            ("tr.record", "I64", vec![8], tr),
            ("tr.trellis", "U8", vec![1, 4096], ramp(4096, 3)),
            ("tr.suh", "F16", vec![128], halves(&[0.5; 128])),
            ("tr.svh", "F16", vec![128], halves(&[2.0; 128])),
        ],
    );
    write(
        &dir.join("dense/exlr8-vocab.safetensors"),
        &[
            ("lm_head.record", "I64", vec![8], record(7, 2, 2, 0)),
            (
                "lm_head.fp32",
                "F32",
                vec![2, 2],
                floats(&[1.0, 2.0, 3.0, 4.0]),
            ),
            ("norm.weight", "F32", vec![2], floats(&[0.5, 0.25])),
        ],
    );
    let release = Release::open(&dir);
    let slices = release.slices();
    let found = |name: &str| slices.iter().find(|s| s.name == name).unwrap();
    let shape = |name: &str| found(name).shape.clone();
    // Each matrix comes in the most compact form tessera reads.
    assert_eq!(
        found("model.layers.0.qi.weight").stored,
        Stored::Form(&INT4_G32)
    );
    assert_eq!(
        found("model.layers.0.mx.weight").stored,
        Stored::Form(&MXFP8)
    );
    assert_eq!(found("model.layers.0.blk.weight").stored, Stored::BF16);
    assert_eq!(found("lm_head.weight").stored, Stored::F32);
    let turned = Stored::Form(&tessera_engine::trellis::TRELLIS_K2_ROT);
    assert_eq!(found("model.layers.0.tr.weight").stored, turned);
    assert_eq!(shape("model.layers.0.mx.weight"), [2, 64]);
    assert_eq!(shape("model.layers.0.qi.weight"), [2, 128]);
    assert_eq!(shape("lm_head.weight"), [2, 2]);
    assert_eq!(shape("model.norm.weight"), [2]);
    let read = |name: &str| release.floats(name).unwrap();
    let mx = read("model.layers.0.mx.weight");
    let blk = read("model.layers.0.blk.weight");
    let q = read("model.layers.0.q.weight");
    let qi = read("model.layers.0.qi.weight");
    for o in 0..2 {
        for i in 0..128 {
            let at = o * 128 + i;
            assert_eq!(blk[at], e4m3(fp8[at]) * 0.5, "block {o} {i}");
            let nibble = |b: u8| f32::from((b >> (4 * (i % 2))) & 0xF) - 8.0;
            let byte = q4_in[o * 64 + i / 2];
            assert_eq!(
                qi[at],
                nibble(byte) * s4_in[o * 4 + i / 32],
                "int4 by input {o} {i}"
            );
            if i >= 64 {
                continue;
            }
            let scale = 2f32.powi(i32::from(e8m0[o * 2 + i / 32]) - 127);
            assert_eq!(
                mx[o * 64 + i],
                e4m3(fp8[o * 64 + i]) * scale,
                "mxfp8 {o} {i}"
            );
            let byte = q4[o * 32 + i / 2];
            assert_eq!(
                q[o * 64 + i],
                nibble(byte) * s4[o * 2 + i / 32],
                "int4 {o} {i}"
            );
        }
    }
    assert_eq!(read("lm_head.weight"), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read("model.norm.weight"), [0.5, 0.25]);
    // Trellis stays packed, its rotations after the codes.
    let file = release.packed("model.layers.0.tr.weight", None).unwrap();
    let codes = trellis::packed(128, 128, 1, &ramp(4096, 3));
    let whole = [codes, halves(&[0.5; 128]), halves(&[2.0; 128])].concat();
    assert_eq!(file, whole);
    let cut = release
        .packed("model.layers.0.tr.weight", Some((0, 128)))
        .unwrap();
    assert_eq!(cut, whole);
    // Block formats stay packed, whole or by row.
    let forms = [
        ("q", &q, &INT4_G32 as &dyn Form),
        ("qi", &qi, &INT4_G32),
        ("mx", &mx, &MXFP8),
    ];
    for (name, want, form) in forms {
        let name = format!("model.layers.0.{name}.weight");
        let bytes = release.packed(&name, None).unwrap();
        assert_eq!(&form.unpack_all(&bytes), want, "{name}");
        let row = release.packed(&name, Some((1, 1))).unwrap();
        assert_eq!(form.unpack_all(&row), want[want.len() / 2..], "{name} row");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// v4: the manifest picks widths, and metadata gives names.
#[test]
fn v4_release() {
    let dir = tessera_testkit::scratch(&format!("{}-v4-exlr8", std::process::id()));
    let width = trellis_record;
    let (k2, k3) = (ramp(4096, 3), ramp(6144, 5));
    let bf16 = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let side = halves(&[0.5; 128]);
    write_meta(
        &dir.join("dense/exlr8-dense-000.safetensors"),
        &[("tensor_prefix", "model.text.layers.")],
        &[
            ("q.k2.record", "I64", vec![8], width(2)),
            ("q.k2.trellis.pmcgx", "U8", vec![1, 4096], k2),
            ("q.k2.suh", "F16", vec![128], side.clone()),
            ("q.k2.svh", "F16", vec![128], side.clone()),
            ("q.k3.record", "I64", vec![8], width(3)),
            ("q.k3.trellis.pmcgx", "U8", vec![1, 6144], k3.clone()),
            ("q.k3.suh", "F16", vec![128], side.clone()),
            ("q.k3.svh", "F16", vec![128], side.clone()),
            ("g.record", "I64", vec![8], record(6, 2, 2, 0)),
            ("g.bf16", "BF16", vec![2, 2], bf16.clone()),
            ("b.record", "I64", vec![8], record(4, 2, 2, 0)),
            ("b.fp8", "U8", vec![2, 2], vec![0x38, 0x40, 0x38, 0x40]),
            ("b.scale", "F32", vec![1, 1], floats(&[2.0])),
        ],
    );
    let names = r#"{"embed_tokens": "model.text.embed_tokens"}"#;
    write_meta(
        &dir.join("dense/exlr8-vocab.safetensors"),
        &[("hf_names", names)],
        &[
            ("embed_tokens.record", "I64", vec![8], record(7, 1, 2, 0)),
            ("embed_tokens.fp32", "F32", vec![1, 2], floats(&[1.0, 2.0])),
        ],
    );
    let extra = "model.text.extra.weight";
    write(
        &dir.join("dense/exlr8-extra-more.safetensors"),
        &[(extra, "F32", vec![2], floats(&[3.0, 4.0]))],
    );
    let manifest = |name: &str, needs: &str| {
        let text =
            format!(r#"{{"requires": {{"trellis": ["{needs}"]}}, "dense": {{"0": {{"q": 3}}}}}}"#);
        let path = dir.join(format!("manifests/exlr8-manifest-{name}.json"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    manifest("k3", "k3 pmcgx");
    manifest("k5", "k5 pmcgx");
    assert_eq!(Release::open(&dir).manifests(), ["k3", "k5"]);
    assert!(Release::open(&dir).manifest("k5").is_err());
    let release = Release::open(&dir).manifest("k3").unwrap();
    let slices = release.slices();
    let found = |name: &str| slices.iter().find(|s| s.name == name).map(|s| s.stored);
    let q = "model.text.layers.0.q.weight";
    assert_eq!(found(q), Some(Stored::Form(&trellis::TRELLIS_K3_PMCGX_ROT)));
    assert_eq!(
        found("model.text.layers.0.q.codes"),
        Some(Stored::Form(&trellis::TRELLIS_K3_PMCGX))
    );
    assert!(slices.iter().all(|s| !s.name.contains(".k2")));
    assert_eq!(
        release.floats("model.text.embed_tokens.weight").unwrap(),
        [1.0, 2.0]
    );
    assert_eq!(release.floats(extra).unwrap(), [3.0, 4.0]);
    assert_eq!(
        Offer::packed(&release, "model.text.layers.0.g.weight").unwrap(),
        bf16
    );
    let fp8_all = Offer::packed(&release, "model.text.layers.0.b.weight").unwrap();
    let fp8_row = Offer::packed_rows(&release, "model.text.layers.0.b.weight", (1, 1)).unwrap();
    assert_eq!(fp8_all.len(), 8);
    assert_eq!(fp8_row, &fp8_all[4..8]);
    let codes = trellis::TRELLIS_K3_PMCGX.unpack_all(&trellis::packed(128, 128, 1, &k3));
    let want = trellis::rotated(&codes, 128, 128, &[0.5; 128], &[0.5; 128]);
    assert_eq!(release.floats(q).unwrap(), want);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn remote_manifest_loads() {
    let dir = tessera_testkit::scratch(&format!("{}-remote-exlr8", std::process::id()));
    let width = trellis_record;
    let (k2, k3) = (ramp(4096, 3), ramp(6144, 5));
    let side = halves(&[0.5; 128]);
    write(
        &dir.join("dense/exlr8-vocab.safetensors"),
        &[("embed_tokens.weight", "F32", vec![2], floats(&[1.0, 2.0]))],
    );
    write(
        &dir.join("dense/exlr8-dense-000.safetensors"),
        &[
            ("q.k2.record", "I64", vec![8], width(2)),
            ("q.k2.trellis.pmcgx", "U8", vec![1, 4096], k2),
            ("q.k2.suh", "F16", vec![128], side.clone()),
            ("q.k2.svh", "F16", vec![128], side.clone()),
            ("q.k3.record", "I64", vec![8], width(3)),
            ("q.k3.trellis.pmcgx", "U8", vec![1, 6144], k3),
            ("q.k3.suh", "F16", vec![128], side.clone()),
            ("q.k3.svh", "F16", vec![128], side),
        ],
    );
    let mpath = dir.join("manifests/exlr8-manifest-k3.json");
    std::fs::create_dir_all(mpath.parent().unwrap()).unwrap();
    std::fs::write(&mpath, r#"{"dense": {"0": {"q": 3}}}"#).unwrap();
    let (base, server) = serve(dir.clone());
    let release = Release::from_url(&base).manifest("k3").unwrap();
    let slices = release.slices();
    let found = slices
        .iter()
        .find(|s| s.name == "model.layers.0.q.weight")
        .map(|s| s.stored);
    assert_eq!(found, Some(Stored::Form(&trellis::TRELLIS_K3_PMCGX_ROT)));
    assert!(slices.iter().all(|s| !s.name.contains(".k2")));
    let _ = ureq::get(&format!("{base}/stop")).call();
    server.join().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
