use crate::exlr8::Release;
use crate::{Offer, Stored};
use sensai_engine::block::{INT4_G32, MXFP8};
use sensai_engine::storage::Form;
use sensai_engine::{E4M3, F16, trellis};
use std::path::Path;

/// One tensor: name, safetensors dtype, shape, bytes.
type Tensor<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

/// `tensors` as one safetensors file at `path`.
fn write(path: &Path, tensors: &[Tensor]) {
    let (mut header, mut data): (Vec<String>, Vec<u8>) = (Vec::new(), Vec::new());
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
fn ramp(n: usize, step: usize) -> Vec<u8> {
    (0..n).map(|i| (i * step % 120) as u8).collect()
}

fn halves(values: &[f32]) -> Vec<u8> {
    let half = |v: &f32| F16::from(*v).0.to_le_bytes();
    values.iter().flat_map(half).collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn e4m3(b: u8) -> f32 {
    f32::from(E4M3(b))
}

/// Each format decodes as the release's README says.
#[test]
fn glm_dense_formats() {
    let dir = sensai_testkit::scratch(&format!("{}-glm-exlr8", std::process::id()));
    let fp8 = ramp(2 * 128, 7);
    let (e8m0, block) = (vec![126u8, 128, 127, 129], [0.5f32]);
    let (q4, s4) = (ramp(2 * 32, 11), [0.5f32, 1.0, 2.0, 4.0]);
    let (q4_in, s4_in) = (
        ramp(2 * 64, 13),
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
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
            ("tr.record", "I64", vec![8], record(1, 128, 128, 1)),
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
    assert_eq!(found("model.layers.0.blk.weight").stored, Stored::Bf16);
    assert_eq!(found("lm_head.weight").stored, Stored::F32);
    let turned = Stored::Form(&sensai_engine::trellis::TRELLIS_K2_ROT);
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
    assert_eq!(
        file,
        [codes, halves(&[0.5; 128]), halves(&[2.0; 128])].concat()
    );
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

/// A routed layer's sets and rotations, at one width.
#[test]
fn glm_experts() {
    let dir = sensai_testkit::scratch(&format!("{}-glm-experts", std::process::id()));
    let norm = || ("norm.weight", "F32", vec![1], floats(&[1.0]));
    for file in ["exlr8-vocab", "exlr8-dense-000", "exlr8-dense-001"] {
        write(&dir.join(format!("dense/{file}.safetensors")), &[norm()]);
    }
    let signs: Vec<u8> = (0..128)
        .map(|i| if i % 3 == 0 { 0xFF } else { 1 })
        .collect();
    let ends = [
        "gate.su_block",
        "up.su_block",
        "down.sv_block",
        "gate.svh",
        "up.svh",
        "down.suh",
    ];
    let names: Vec<[String; 2]> = (ends.iter())
        .map(|e| [format!("k3.{e}"), format!("shared.k3.{e}")])
        .collect();
    let mut index: Vec<Tensor> = vec![
        ("signs.in", "I8", vec![128], signs.clone()),
        ("signs.out", "I8", vec![128], signs),
    ];
    for (i, [own, shared]) in names.iter().enumerate() {
        let (width, value) = (if i < 3 { 1 } else { 128 }, 0.5 * (i + 1) as f32);
        index.push((own, "F16", vec![2, width], halves(&vec![value; 2 * width])));
        index.push((shared, "F16", vec![1, width], halves(&vec![4.0; width])));
    }
    write(
        &dir.join("experts/exlr8-layer-001.index.safetensors"),
        &index,
    );
    let (set, one) = (ramp(2 * 128 * 128 * 3 / 8, 5), ramp(128 * 128 * 3 / 8, 9));
    let codes: Vec<[String; 2]> = (["gate", "up", "down"].iter())
        .map(|p| [format!("trellis.{p}"), format!("shared.trellis.{p}")])
        .collect();
    let mut stored: Vec<Tensor> = Vec::new();
    for [own, shared] in &codes {
        stored.push((own, "U8", vec![set.len()], set.clone()));
        stored.push((shared, "U8", vec![one.len()], one.clone()));
    }
    write(&dir.join("experts/exlr8-layer-001.k3.safetensors"), &stored);
    let release = Release::open(&dir).width(3);
    let at = "model.layers.1.mlp.";
    let gate = format!("{at}experts.gate.weight");
    let offered = release
        .slices()
        .into_iter()
        .find(|s| s.name == gate)
        .unwrap();
    assert_eq!(offered.shape, [2, 128, 128]);
    let read = |name: &str| release.floats(&format!("{at}{name}")).unwrap();
    let su = read("experts.gate.su");
    assert_eq!((su.len(), su[0], su[128]), (256, 0.5, 0.5));
    assert_eq!(read("experts.down.sv")[200], 1.5);
    // The shared expert's input scales carry the signs.
    let shared = read("shared_experts.gate_proj.suh");
    assert_eq!((shared[0], shared[1]), (-4.0, 4.0));
    let gate = format!("{at}shared_experts.gate_proj.weight");
    let file = release.packed(&gate, None).unwrap();
    let parts = sensai_engine::trellis::TRELLIS_K3_ROT.split(&file).unwrap();
    assert_eq!(parts[1][..8], [0, 0, 128, 192, 0, 0, 128, 64]);
    let down = release
        .packed(&format!("{at}experts.down.weight"), None)
        .unwrap();
    assert_eq!(down, trellis::packed(128, 128, trellis::INPUT_SIDE, &set));
    // A rank's experts, read alone, match its share.
    let name = format!("{at}experts.down.weight");
    let second = release.packed_share(&name, (1, 1)).unwrap();
    let k3 = trellis::form_for(3).unwrap();
    assert_eq!(Some(second), k3.share(&down, [1, 2, 128 * 128], [1, 1]));
    // The shared expert is not a set: it reads whole.
    assert_eq!(release.packed_share(&gate, (0, 64)).unwrap(), file);
    std::fs::remove_dir_all(dir).unwrap();
}
