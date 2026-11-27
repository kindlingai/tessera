use crate::exlr8::{self, Release};
use crate::{Offer, Stored};
use std::path::Path;
use tessera_engine::block::{INT4_G32, MXFP8};
use tessera_engine::storage::Form;
use tessera_engine::{Bf16, E4M3, F16, trellis};

/// One tensor: name, safetensors dtype, shape, bytes.
pub(crate) type Tensor<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

/// `tensors` as one safetensors file at `path`.
pub(crate) fn write(path: &Path, tensors: &[Tensor]) {
    write_meta(path, &[], tensors);
}

/// `tensors` and `meta` strings as one safetensors file at `path`.
fn write_meta(path: &Path, meta: &[(&str, &str)], tensors: &[Tensor]) {
    let (mut header, mut data): (Vec<String>, Vec<u8>) = (Vec::new(), Vec::new());
    if !meta.is_empty() {
        let map: serde_json::Map<_, _> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), (*v).into()))
            .collect();
        header.push(format!(
            r#""__metadata__":{}"#,
            serde_json::Value::Object(map)
        ));
    }
    for (name, dtype, shape, bytes) in tensors {
        let at = data.len();
        data.extend(bytes);
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        let end = data.len();
        header.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{at},{end}]}}"#,
            shape.join(",")
        ));
    }
    let header = format!("{{{}}}", header.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// A record: format, out, in, K, side, then zeros.
fn record(format: i64, out: usize, inp: usize, side: i64) -> Vec<u8> {
    let fields = [format, out as i64, inp as i64, 0, side, 0, 0, 0];
    fields.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// `n` bytes below 0x7F, the E4M3 NaN.
pub(crate) fn ramp(n: usize, step: usize) -> Vec<u8> {
    (0..n).map(|i| (i * step % 120) as u8).collect()
}

pub(crate) fn halves(values: &[f32]) -> Vec<u8> {
    let half = |v: &f32| F16::from(*v).0.to_le_bytes();
    values.iter().flat_map(half).collect()
}

pub(crate) fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn e4m3(b: u8) -> f32 {
    f32::from(E4M3(b))
}

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
    let width = |k: i64| {
        let mut r = record(1, 128, 128, 1);
        r[24..32].copy_from_slice(&k.to_le_bytes());
        r
    };
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let dir = tessera_testkit::scratch(&format!("{}-remote-exlr8", std::process::id()));
    let width = |k: i64| {
        let mut r = record(1, 128, 128, 1);
        r[24..32].copy_from_slice(&k.to_le_bytes());
        r
    };
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
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let root = dir.clone();
    let server = std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut s = stream;
            let mut buf = [0u8; 2048];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let mut lines = req.lines();
            let path = lines
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("/");
            if path == "/stop" {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                break;
            }
            let range = lines.find_map(|l| {
                l.strip_prefix("Range: bytes=")
                    .or_else(|| l.strip_prefix("range: bytes="))
            });
            let full = root.join(path.trim_start_matches('/'));
            let Ok(bytes) = std::fs::read(&full) else {
                let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                continue;
            };
            if let Some(r) = range {
                let (a, b) = r.split_once('-').unwrap();
                let start: usize = a.parse().unwrap();
                let end = b.parse::<usize>().unwrap().min(bytes.len() - 1);
                let slice = &bytes[start..=end];
                let head = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\r\n",
                    slice.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(slice);
            } else {
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", bytes.len());
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&bytes);
            }
        }
    });
    let base = format!("http://{addr}");
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

/// Builds a synthetic `qwen3_vl` multi-group exlr8 v2 release.
fn write_qwen3_vl(dir: &Path) {
    let bf16 = |xs: &[f32]| -> Vec<u8> {
        xs.iter()
            .flat_map(|x| Bf16::from(*x).0.to_le_bytes())
            .collect()
    };
    let tr4 = |side: i64| {
        let mut r = record(1, 128, 128, side);
        r[24..32].copy_from_slice(&4i64.to_le_bytes());
        r
    };
    let meta = [("format", "exlr8-v2"), ("model_profile", "qwen3_vl")];
    let k4_out = ramp(8192, 3);
    let k4_in = ramp(8192, 5);
    let side = halves(&[0.5; 128]);
    write_meta(
        &dir.join("exlr8-vocab.safetensors"),
        &meta,
        &[
            ("embed_tokens.record", "I64", vec![8], record(5, 4, 4, 1)),
            ("embed_tokens.checksum", "U64", vec![1], vec![0u8; 8]),
            ("__pad.0", "U8", vec![8], vec![0u8; 8]),
            ("embed_tokens.fp8", "U8", vec![4, 4], ramp(16, 7)),
            (
                "embed_tokens.scale",
                "F32",
                vec![4],
                floats(&[0.5, 1.0, 2.0, 4.0]),
            ),
            ("norm.weight", "BF16", vec![4], bf16(&[1.0, 2.0, 3.0, 4.0])),
        ],
    );
    write_meta(
        &dir.join("exlr8-dense-text-000.safetensors"),
        &meta,
        &[
            ("self_attn.q_proj.record", "I64", vec![8], tr4(1)),
            ("self_attn.q_proj.trellis", "U8", vec![1, 8192], k4_out),
            ("self_attn.q_proj.suh", "F16", vec![128], side.clone()),
            ("self_attn.q_proj.svh", "F16", vec![128], side.clone()),
            ("self_attn.o_proj.record", "I64", vec![8], tr4(2)),
            ("self_attn.o_proj.trellis", "U8", vec![1, 8192], k4_in),
            ("self_attn.o_proj.suh", "F16", vec![128], side.clone()),
            ("self_attn.o_proj.svh", "F16", vec![128], side.clone()),
            ("self_attn.q_norm.weight", "BF16", vec![4], bf16(&[1.5; 4])),
        ],
    );
    write_meta(
        &dir.join("exlr8-dense-text-001.safetensors"),
        &meta,
        &[
            ("mlp.gate_proj.record", "I64", vec![8], record(2, 2, 64, 1)),
            ("mlp.gate_proj.q4", "U8", vec![2, 32], ramp(64, 11)),
            (
                "mlp.gate_proj.s4",
                "F16",
                vec![2, 2],
                halves(&[0.5, 1.0, 2.0, 4.0]),
            ),
            ("mlp.down_proj.record", "I64", vec![8], record(2, 2, 128, 2)),
            ("mlp.down_proj.q4", "U8", vec![1, 2, 64], ramp(128, 13)),
            (
                "mlp.down_proj.s4",
                "F16",
                vec![1, 2, 4],
                halves(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ),
        ],
    );
    write_meta(
        &dir.join("exlr8-dense-vision-000.safetensors"),
        &meta,
        &[
            ("attn.qkv.record", "I64", vec![8], tr4(1)),
            ("attn.qkv.trellis", "U8", vec![1, 8192], ramp(8192, 9)),
            ("attn.qkv.suh", "F16", vec![128], side.clone()),
            ("attn.qkv.svh", "F16", vec![128], side.clone()),
            ("mlp.linear_fc1.record", "I64", vec![8], record(5, 6, 4, 1)),
            ("mlp.linear_fc1.fp8", "U8", vec![6, 4], ramp(24, 5)),
            ("mlp.linear_fc1.scale", "F32", vec![6], floats(&[0.25; 6])),
            ("norm1.bias", "BF16", vec![4], bf16(&[0.5; 4])),
        ],
    );
    write_meta(
        &dir.join("exlr8-dense-global-000.safetensors"),
        &meta,
        &[
            (
                "model.visual.patch_embed.proj.weight",
                "BF16",
                vec![2, 3, 2, 2, 2],
                bf16(&[0.25; 48]),
            ),
            (
                "model.visual.pos_embed.record",
                "I64",
                vec![8],
                record(6, 4, 4, 0),
            ),
            (
                "model.visual.pos_embed.bf16",
                "BF16",
                vec![4, 4],
                bf16(&[2.0; 16]),
            ),
            (
                "model.visual.merger.linear_fc2.record",
                "I64",
                vec![8],
                tr4(2),
            ),
            (
                "model.visual.merger.linear_fc2.trellis",
                "U8",
                vec![1, 8192],
                ramp(8192, 11),
            ),
            (
                "model.visual.merger.linear_fc2.suh",
                "F16",
                vec![128],
                side.clone(),
            ),
            ("model.visual.merger.linear_fc2.svh", "F16", vec![128], side),
        ],
    );
}

fn check_qwen3_vl(release: &Release) {
    let slices = release.slices();
    let find = |n: &str| slices.iter().find(|s| s.name == n).expect(n);
    assert!(
        slices
            .iter()
            .all(|s| !s.name.contains("checksum") && !s.name.contains("__pad"))
    );
    let k4 = Stored::Form(&trellis::TRELLIS_K4_ROT);
    assert_eq!(
        find("model.language_model.embed_tokens.weight").stored,
        Stored::BF16
    );
    assert_eq!(
        find("model.language_model.norm.weight").stored,
        Stored::BF16
    );
    assert_eq!(
        find("model.language_model.layers.0.self_attn.q_proj.weight").stored,
        k4
    );
    assert_eq!(
        find("model.language_model.layers.0.self_attn.o_proj.weight").stored,
        k4
    );
    assert_eq!(
        find("model.language_model.layers.1.mlp.gate_proj.weight").stored,
        Stored::Form(&INT4_G32)
    );
    assert_eq!(
        find("model.language_model.layers.1.mlp.down_proj.weight").stored,
        Stored::Form(&INT4_G32)
    );
    assert_eq!(find("model.visual.blocks.0.attn.qkv.weight").stored, k4);
    assert_eq!(
        find("model.visual.blocks.0.mlp.linear_fc1.weight").stored,
        Stored::BF16
    );
    assert_eq!(
        find("model.visual.blocks.0.mlp.linear_fc1.weight").shape,
        [6, 4]
    );
    assert_eq!(
        find("model.visual.patch_embed.proj.weight").shape,
        [2, 3, 2, 2, 2]
    );
    assert_eq!(find("model.visual.pos_embed.weight").shape, [4, 4]);
    assert_eq!(find("model.visual.merger.linear_fc2.weight").stored, k4);
    // Streaming FP8_ROW matches floats, whole and row-sliced.
    let emb_f = release
        .floats("model.language_model.embed_tokens.weight")
        .unwrap();
    let emb_p = Offer::packed(release, "model.language_model.embed_tokens.weight").unwrap();
    let emb_r =
        Offer::packed_rows(release, "model.language_model.embed_tokens.weight", (1, 2)).unwrap();
    assert_eq!(emb_p.len(), 32);
    assert_eq!(emb_r, &emb_p[8..24]);
    let got: Vec<f32> = emb_p
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(Bf16(u16::from_le_bytes(*b))))
        .collect();
    let want: Vec<f32> = emb_f.iter().map(|x| f32::from(Bf16::from(*x))).collect();
    assert_eq!(got, want);
    assert_eq!(
        release.floats("model.language_model.norm.weight").unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        Offer::packed(release, "model.visual.patch_embed.proj.weight")
            .unwrap()
            .len(),
        96
    );
    assert_eq!(
        release.floats("model.visual.pos_embed.weight").unwrap(),
        [2.0; 16]
    );
    // Side-IN trellis and int4_g32 unpack bit-exact.
    let o_name = "model.language_model.layers.0.self_attn.o_proj.weight";
    let o_packed = Offer::packed(release, o_name).unwrap();
    assert_eq!(
        trellis::TRELLIS_K4_ROT.unpack_all(&o_packed),
        release.floats(o_name).unwrap()
    );
    let d_name = "model.language_model.layers.1.mlp.down_proj.weight";
    let d_packed = Offer::packed(release, d_name).unwrap();
    assert_eq!(
        INT4_G32.unpack_all(&d_packed),
        release.floats(d_name).unwrap()
    );
    assert_eq!(
        release.floats("model.visual.blocks.0.norm1.bias").unwrap(),
        [0.5; 4]
    );
}

fn check_fp8_lut(dir: &Path) {
    let (rows, inp) = (257usize, 384usize);
    let codes: Vec<u8> = (0..rows * inp).map(|i| ((i * 37) % 256) as u8).collect();
    let scales: Vec<f32> = (0..rows).map(|r| 0.01 * (r as f32 + 1.0)).collect();
    let pdir = dir.join("lut");
    write(
        &pdir.join("dense/exlr8-vocab.safetensors"),
        &[
            (
                "embed_tokens.record",
                "I64",
                vec![8],
                record(5, rows, inp, 1),
            ),
            ("embed_tokens.fp8", "U8", vec![rows, inp], codes.clone()),
            ("embed_tokens.scale", "F32", vec![rows], floats(&scales)),
        ],
    );
    let packed = Offer::packed(&Release::open(&pdir), "model.embed_tokens.weight").unwrap();
    let want: Vec<u8> = (0..rows * inp)
        .flat_map(|i| Bf16::from(e4m3(codes[i]) * scales[i / inp]).0.to_le_bytes())
        .collect();
    assert_eq!(packed, want);
}

#[test]
fn qwen3_vl_release() {
    let dir = tessera_testkit::scratch(&format!("{}-qwen3vl-exlr8", std::process::id()));
    write_qwen3_vl(&dir);
    assert!(exlr8::is_release(&dir));
    check_qwen3_vl(&Release::open(&dir));
    check_fp8_lut(&dir);
    std::fs::remove_dir_all(dir).unwrap();
}
