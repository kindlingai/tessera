use crate::exlr8::Release;
use crate::exlr8_test::{Tensor, floats, halves, ramp, write};
use crate::{Offer, Stored};
use std::path::Path;
use tessera_engine::storage::Form;
use tessera_engine::trellis;

const ENDS: [&str; 6] = [
    "gate.su_block",
    "up.su_block",
    "down.sv_block",
    "gate.svh",
    "up.svh",
    "down.suh",
];

fn write_layer(dir: &Path, layer: usize, widths: &[u32], suffix: &str) {
    let signs: Vec<u8> = (0..128)
        .map(|i| if i % 3 == 0 { 0xFF } else { 1 })
        .collect();
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for &k in widths {
        for (i, end) in ENDS.iter().enumerate() {
            let width = if i < 3 { 1 } else { 128 };
            let value = k as f32 + 0.5 * (i + 1) as f32;
            owned.push((
                format!("k{k}.{end}"),
                vec![2, width],
                halves(&vec![value; 2 * width]),
            ));
            if k >= 3 {
                owned.push((
                    format!("shared.k{k}.{end}"),
                    vec![1, width],
                    halves(&vec![k as f32 * 2.0; width]),
                ));
            }
        }
    }
    let mut index: Vec<Tensor> = vec![
        ("signs.in", "I8", vec![128], signs.clone()),
        ("signs.out", "I8", vec![128], signs),
    ];
    for (name, shape, bytes) in &owned {
        index.push((name, "F16", shape.clone(), bytes.clone()));
    }
    let idx_path = dir.join(format!("experts/exlr8-layer-{layer:03}.index.safetensors"));
    write(&idx_path, &index);
    for &k in widths {
        let n = 128 * 128 * k as usize / 8;
        let (set, one) = (ramp(2 * n, 3 + k as usize), ramp(n, 7 + k as usize));
        let mut names = Vec::new();
        for p in ["gate", "up", "down"] {
            names.push((format!("trellis.{p}{suffix}"), set.clone()));
            if k >= 3 {
                names.push((format!("shared.trellis.{p}{suffix}"), one.clone()));
            }
        }
        let stored: Vec<Tensor> = names
            .iter()
            .map(|(n, b)| (n.as_str(), "U8", vec![b.len()], b.clone()))
            .collect();
        let path = dir.join(format!("experts/exlr8-layer-{layer:03}.k{k}.safetensors"));
        write(&path, &stored);
    }
}

fn write_dense_stubs(dir: &Path, layers: usize) {
    let norm = || ("norm.weight", "F32", vec![1], floats(&[1.0]));
    write(&dir.join("dense/exlr8-vocab.safetensors"), &[norm()]);
    for l in 0..layers {
        let p = dir.join(format!("dense/exlr8-dense-{l:03}.safetensors"));
        write(&p, &[norm()]);
    }
}

fn write_manifest(dir: &Path, name: &str, json: &str) {
    let path = dir.join(format!("manifests/exlr8-manifest-{name}.json"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, json).unwrap();
}

fn form_name(release: &Release, name: &str) -> &'static str {
    let slices = release.slices();
    match slices.iter().find(|s| s.name == name).map(|s| s.stored) {
        Some(Stored::Form(f)) => f.name(),
        other => panic!("{name}: expected form, got {other:?}"),
    }
}

fn check_single_width(dir: &Path) {
    write_dense_stubs(dir, 2);
    write_layer(dir, 1, &[3], "");
    let release = Release::open(dir).width(3);
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
    assert_eq!((su.len(), su[0], su[128]), (256, 3.5, 3.5));
    assert_eq!(read("experts.down.sv")[200], 4.5);
    // The shared expert's input scales carry the signs.
    let shared = read("shared_experts.gate_proj.suh");
    assert_eq!((shared[0], shared[1]), (-6.0, 6.0));
    let shared_gate = format!("{at}shared_experts.gate_proj.weight");
    let file = release.packed(&shared_gate, None).unwrap();
    let parts = trellis::TRELLIS_K3_ROT.split(&file).unwrap();
    assert_eq!(parts[1][..8], floats(&[-6.0, 6.0]));
    let set = ramp(2 * 128 * 128 * 3 / 8, 6);
    let down = release
        .packed(&format!("{at}experts.down.weight"), None)
        .unwrap();
    assert_eq!(down, trellis::packed(128, 128, trellis::INPUT_SIDE, &set));
    let name = format!("{at}experts.down.weight");
    let second = release.packed_share(&name, (1, 1)).unwrap();
    let k3 = trellis::form_for(3, trellis::Codebook::Mcg).unwrap();
    assert_eq!(Some(second), k3.share(&down, [1, 2, 128 * 128], [1, 1]));
    // The shared expert is not a set: it reads whole.
    assert_eq!(release.packed_share(&shared_gate, (0, 64)).unwrap(), file);
    // A `.mul1` suffix selects mul1 forms for the same bytes.
    write_layer(dir, 1, &[3], ".mul1");
    let release = Release::open(dir).width(3);
    assert_eq!(form_name(&release, &name), "trellis_k3_mul1");
    assert_eq!(release.packed(&name, None).unwrap(), down);
}

fn check_axis_slicing(dir: &Path) {
    let big = dir.join("ffn");
    write(&big.join("dense/exlr8-vocab.safetensors"), &[]);
    let idx = vec![
        ("signs.in", "I8", vec![128], vec![1u8; 128]),
        ("k3.gate.svh", "F16", vec![2, 256], halves(&vec![1.0; 512])),
    ];
    write(&big.join("experts/exlr8-layer-001.index.safetensors"), &idx);
    let big_set = ramp(2 * 128 * 256 * 3 / 8, 13);
    write(
        &big.join("experts/exlr8-layer-001.k3.safetensors"),
        &[
            ("trellis.gate", "U8", vec![big_set.len()], big_set.clone()),
            ("trellis.down", "U8", vec![big_set.len()], big_set.clone()),
        ],
    );
    let brel = Release::open(&big).width(3);
    let at = "model.layers.1.mlp.";
    let (bg, bd) = (
        format!("{at}experts.gate.weight"),
        format!("{at}experts.down.weight"),
    );
    let (gw, dw) = (
        brel.packed(&bg, None).unwrap(),
        brel.packed(&bd, None).unwrap(),
    );
    let k3 = trellis::form_for(3, trellis::Codebook::Mcg).unwrap();
    assert_eq!(
        Some(brel.packed_axis(&bg, 1, (128, 128)).unwrap()),
        k3.share(&gw, [2, 256, 128], [128, 128])
    );
    assert_eq!(
        Some(brel.packed_axis(&bd, 2, (128, 128)).unwrap()),
        k3.share(&dw, [2 * 128, 256, 1], [128, 128])
    );
    // Sparse zero-filled experts fail closed under TP slicing.
    let mut sparse = big_set;
    let half = sparse.len() / 2;
    sparse[half..].fill(0);
    write(
        &big.join("experts/exlr8-layer-001.k3.safetensors"),
        &[("trellis.gate", "U8", vec![sparse.len()], sparse)],
    );
    let srel = Release::open(&big).width(3);
    assert!(srel.packed_axis(&bg, 1, (0, 128)).is_err());
}

fn check_manifest_widths(dir: &Path) {
    write_dense_stubs(dir, 3);
    write_layer(dir, 1, &[2, 3, 4], ".mul1");
    write_layer(dir, 2, &[3], ".mul1");
    write_manifest(
        dir,
        "bit200",
        r#"{
            "format": "exlr8-v2-manifest",
            "layers": {
                "1": {
                    "gate": [[2, 2], [2, 2]],
                    "up": [[2, 2], [2, 2]],
                    "down": [2, 2],
                    "shared": {"gate": [4, 4], "up": [4, 4], "down": 4}
                },
                "2": {
                    "gate": [[3, 3], [3, 3]],
                    "up": [[3, 3], [3, 3]],
                    "down": [3, 3],
                    "shared": {"gate": [3, 3], "up": [3, 3], "down": 3}
                }
            }
        }"#,
    );
    // Calling `.width(3)` after `.manifest()` must not clobber manifest widths.
    let r = Release::open(dir).manifest("bit200").unwrap().width(3);
    let l1 = "model.layers.1.mlp.";
    let l2 = "model.layers.2.mlp.";
    assert_eq!(
        form_name(&r, &format!("{l1}experts.gate.weight")),
        "trellis_k2_mul1"
    );
    assert_eq!(
        form_name(&r, &format!("{l1}experts.down.weight")),
        "trellis_k2_mul1"
    );
    assert_eq!(
        form_name(&r, &format!("{l1}shared_experts.gate_proj.weight")),
        "trellis_k4_mul1_rot"
    );
    assert_eq!(
        form_name(&r, &format!("{l2}experts.gate.weight")),
        "trellis_k3_mul1"
    );
    assert_eq!(
        form_name(&r, &format!("{l2}shared_experts.down_proj.weight")),
        "trellis_k3_mul1_rot"
    );
    assert_eq!(r.floats(&format!("{l1}experts.gate.su")).unwrap()[0], 2.5);
    assert_eq!(
        r.floats(&format!("{l1}shared_experts.gate_proj.suh"))
            .unwrap()[0],
        -8.0
    );
    let k2_set = ramp(2 * 128 * 128 * 2 / 8, 5);
    assert_eq!(
        r.packed(&format!("{l1}experts.gate.weight"), None).unwrap(),
        trellis::packed(128, 128, 1, &k2_set)
    );
    // Mixed projection widths on one layer, shared omitted.
    write_manifest(
        dir,
        "mixed",
        r#"{
            "layers": {
                "1": {
                    "gate": [[2, 2]],
                    "up": [[2, 2]],
                    "down": [3, 3]
                }
            }
        }"#,
    );
    let m = Release::open(dir).manifest("mixed").unwrap();
    assert_eq!(
        form_name(&m, &format!("{l1}experts.gate.weight")),
        "trellis_k2_mul1"
    );
    assert_eq!(
        form_name(&m, &format!("{l1}experts.down.weight")),
        "trellis_k3_mul1"
    );
    assert_eq!(
        form_name(&m, &format!("{l1}shared_experts.gate_proj.weight")),
        "trellis_k3_mul1_rot"
    );
    assert_eq!(m.floats(&format!("{l1}experts.gate.su")).unwrap()[0], 2.5);
    assert_eq!(m.floats(&format!("{l1}experts.down.sv")).unwrap()[0], 4.5);
    // Fallback without manifest uses `.width(k)` and `k.max(3)` for shared.
    let fb = Release::open(dir).width(2);
    assert_eq!(
        form_name(&fb, &format!("{l1}experts.gate.weight")),
        "trellis_k2_mul1"
    );
    assert_eq!(
        form_name(&fb, &format!("{l1}shared_experts.gate_proj.weight")),
        "trellis_k3_mul1_rot"
    );
    // Mixed widths inside one tensor fail closed with an error.
    write_manifest(dir, "bad", r#"{"layers": {"1": {"gate": [[2, 3]]}}}"#);
    let err = Release::open(dir).manifest("bad").err().unwrap();
    assert!(err.to_string().contains("mixed widths"));
}

fn check_latent_moe(dir: &Path) {
    let lm = "model.layers.";
    write(
        &dir.join("dense/exlr8-vocab.safetensors"),
        &[("norm.weight", "F32", vec![1], floats(&[1.0]))],
    );
    write(
        &dir.join("dense/exlr8-dense.safetensors"),
        &[
            (
                &format!("{lm}0.norm.weight"),
                "F32",
                vec![1],
                floats(&[1.0]),
            ),
            (
                &format!("{lm}1.norm.weight"),
                "F32",
                vec![1],
                floats(&[2.0]),
            ),
        ],
    );
    let mk = |n, m| {
        (0..n)
            .map(|i| if i % m == 0 { 0xFF } else { 1 })
            .collect::<Vec<u8>>()
    };
    let (rs, ss) = (mk(128, 2), mk(256, 3));
    let mut index: Vec<Tensor> = vec![
        ("signs.in", "I8", vec![128], rs.clone()),
        ("signs.out", "I8", vec![128], rs),
        ("shared.signs.in", "I8", vec![256], ss.clone()),
        ("shared.signs.out", "I8", vec![256], ss),
    ];
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (i, end) in ENDS.iter().enumerate() {
        let (rw, sw) = if i < 3 { (1, 2) } else { (128, 128) };
        owned.push((format!("k4.{end}"), vec![2, rw], halves(&vec![1.5; 2 * rw])));
        owned.push((
            format!("shared.k4.{end}"),
            vec![1, sw],
            halves(&vec![2.5; sw]),
        ));
    }
    for (name, shape, bytes) in &owned {
        index.push((name, "F16", shape.clone(), bytes.clone()));
    }
    write(
        &dir.join("experts/exlr8-layer-001.index.safetensors"),
        &index,
    );
    let (rset, sone) = (ramp(2 * 8192, 5), ramp(16384, 9));
    let mut names = Vec::new();
    for proj in ["gate", "up", "down"] {
        names.push((format!("trellis.{proj}"), rset.clone()));
        names.push((format!("shared.trellis.{proj}"), sone.clone()));
    }
    let stored: Vec<Tensor> = names
        .iter()
        .map(|(n, b)| (n.as_str(), "U8", vec![b.len()], b.clone()))
        .collect();
    write(&dir.join("experts/exlr8-layer-001.k4.safetensors"), &stored);
    let rel = Release::open(dir).width(4);
    let slices = rel.slices();
    let shape = |n: &str| slices.iter().find(|s| s.name == n).unwrap().shape.clone();
    let at = "model.layers.1.mlp.";
    assert_eq!(shape(&format!("{at}experts.gate.weight")), [2, 128, 128]);
    assert_eq!(
        shape(&format!("{at}shared_experts.gate_proj.weight")),
        [128, 256]
    );
    assert_eq!(shape(&format!("{at}shared_experts.gate_proj.suh")), [256]);
    assert_eq!(shape(&format!("{at}shared_experts.down_proj.svh")), [256]);
    let suh = rel
        .floats(&format!("{at}shared_experts.gate_proj.suh"))
        .unwrap();
    assert_eq!((suh.len(), suh[0], suh[1]), (256, -2.5, 2.5));
    let svh = rel
        .floats(&format!("{at}shared_experts.down_proj.svh"))
        .unwrap();
    assert_eq!((svh.len(), svh[0], svh[1]), (256, -2.5, 2.5));
}

/// One whole-tensor file, and a width for each expert.
fn check_mixed_widths(dir: &Path) {
    let scale = "model.layers.0.attn.q_proj.scale";
    write(
        &dir.join("model-nonexperts.safetensors"),
        &[(scale, "F32", vec![1], floats(&[7.0]))],
    );
    write_layer(dir, 0, &[2, 3, 4], "");
    assert!(crate::exlr8::is_release(dir));
    write_manifest(
        dir,
        "mix",
        r#"{"layers": {"0": {
            "gate": [[3, 3], [2, 2]],
            "up": [[4, 4], [4, 4]],
            "down": [3, 2]
        }}}"#,
    );
    let r = Release::open(dir).manifest("mix").unwrap();
    assert_eq!(r.floats(scale).unwrap(), [7.0]);
    let at = "model.layers.0.mlp.experts.";
    assert_eq!(form_name(&r, &format!("{at}up.weight")), "trellis_k4");
    let su = r.floats(&format!("{at}gate.su")).unwrap();
    assert_eq!((su[0], su[128]), (3.5, 2.5));
    let sv = r.floats(&format!("{at}down.sv")).unwrap();
    assert_eq!((sv[0], sv[128]), (4.5, 3.5));
    let mix = trellis::mixed_for(trellis::Codebook::Mcg).unwrap();
    for (p, shape) in [("gate", [2, 128, 128]), ("down", [2 * 128, 128, 1])] {
        let name = format!("{at}{p}.weight");
        assert_eq!(form_name(&r, &name), "trellis_mix");
        let set = r.packed(&name, None).unwrap();
        let mut want = Vec::new();
        for (e, k) in [3, 2].into_iter().enumerate() {
            let one = Release::open(dir).width(k).packed(&name, None).unwrap();
            let form = trellis::form_for(k, trellis::Codebook::Mcg).unwrap();
            let n = 128 * 128;
            want.extend_from_slice(&form.unpack_all(&one)[e * n..(e + 1) * n]);
        }
        assert_eq!(mix.unpack_all(&set), want, "{p}");
        let second = r.packed_share(&name, (1, 1)).unwrap();
        assert_eq!(Some(second), mix.share(&set, [1, 2, 128 * 128], [1, 1]));
        let axis = if p == "down" { 2 } else { 1 };
        let inner = r.packed_axis(&name, axis, (0, 128)).unwrap();
        assert_eq!(Some(inner), mix.share(&set, shape, [0, 128]), "{p}");
    }
    write_manifest(dir, "short", r#"{"layers": {"0": {"down": [3, 2, 4]}}}"#);
    let short = Release::open(dir).manifest("short").unwrap();
    assert!(short.packed(&format!("{at}down.weight"), None).is_err());
}

#[test]
fn experts_and_manifests() {
    let dir = tessera_testkit::scratch(&format!("{}-exlr8-experts", std::process::id()));
    check_single_width(&dir.join("single"));
    check_axis_slicing(&dir.join("axis"));
    check_manifest_widths(&dir.join("manifest"));
    check_latent_moe(&dir.join("latent"));
    check_mixed_widths(&dir.join("mixed"));
    std::fs::remove_dir_all(dir).unwrap();
}

/// A lone dense file's common layer stem names the experts.
#[test]
fn single_file_prefix() {
    let dir = tessera_testkit::scratch(&format!("{}-exlr8-single", std::process::id()));
    let lm = "model.language_model.layers.";
    let (norm0, norm1) = (format!("{lm}0.norm.weight"), format!("{lm}1.norm.weight"));
    let mults = format!("{lm}1.ple.ple_embedding.layer_multipliers");
    let ints: Vec<u8> = [3i64, -5, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
    write(
        &dir.join("dense/exlr8-dense.safetensors"),
        &[
            (&norm0, "F32", vec![1], floats(&[1.0])),
            (&norm1, "F32", vec![1], floats(&[2.0])),
            (&mults, "I64", vec![3], ints),
            ("mtp.layers.0.norm.weight", "F32", vec![1], floats(&[3.0])),
        ],
    );
    let norm = ("norm.weight", "F32", vec![1], floats(&[1.0]));
    write(&dir.join("dense/exlr8-vocab.safetensors"), &[norm]);
    write_layer(&dir, 0, &[3], "");
    let release = Release::open(&dir).width(3);
    let names: Vec<String> = release.slices().into_iter().map(|s| s.name).collect();
    assert!(
        names.contains(&format!("{lm}0.mlp.experts.gate.weight")),
        "{names:?}"
    );
    assert_eq!(release.i64s(&mults).unwrap(), [3, -5, 7]);
    assert_eq!(release.floats("mtp.layers.0.norm.weight").unwrap(), [3.0]);
    std::fs::remove_dir_all(dir).unwrap();
}
