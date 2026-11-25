use crate::File;
use crate::hessian::{
    DenseEntry, HessianMeta, dense_name, expert_name, inputs_name, pack_tiles, route_name,
    triu_len, weights_id, weights_id_bytes, write_dense, write_experts, write_inputs, write_route,
};

fn pack_triu(full: &[f32], d: usize) -> Vec<f32> {
    assert_eq!(full.len(), d * d);
    let mut out = Vec::with_capacity(triu_len(d));
    for r in 0..d {
        out.extend_from_slice(&full[r * d + r..r * d + d]);
    }
    out
}

fn unpack_triu(tri: &[f32], d: usize) -> Vec<f32> {
    assert_eq!(tri.len(), triu_len(d));
    let mut out = vec![0.0f32; d * d];
    let mut p = 0usize;
    for r in 0..d {
        for c in r..d {
            let v = tri[p];
            out[r * d + c] = v;
            out[c * d + r] = v;
            p += 1;
        }
    }
    out
}

#[test]
fn hessian_vectors() {
    let cfg = b"{\"hidden_size\":128}\n";
    let idx = b"{\"weight_map\":{}}\n";
    let want = "24b0d8c953197327441b27e43ecf94a2f3a7adbaf84b22c9ab0123b7c22519a1";
    assert_eq!(weights_id_bytes(cfg, idx), want);

    let full = [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];
    let tri = pack_triu(&full, 3);
    assert_eq!(tri, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(unpack_triu(&tri, 3), full);

    let mut mat = vec![0.0f32; 16];
    for r in 0..4 {
        for c in r..4 {
            let v = (r * 10 + c + 1) as f32;
            mat[r * 4 + c] = v;
            mat[c * 4 + r] = v;
        }
    }
    let mut tiles = Vec::new();
    for br in 0..2 {
        for bc in br..2 {
            for lr in 0..2 {
                for lc in 0..2 {
                    tiles.push(mat[(br * 2 + lr) * 4 + (bc * 2 + lc)]);
                }
            }
        }
    }
    assert_eq!(pack_tiles(&tiles, 4, 2), pack_triu(&mat, 4));
}

#[test]
fn hessian_roundtrip() {
    let dir = tessera_testkit::scratch(&format!("{}-hess-w5", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), b"{\"hidden_size\":128}\n").unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        b"{\"weight_map\":{}}\n",
    )
    .unwrap();
    let wid = weights_id(&dir).unwrap();
    let meta = HessianMeta {
        hessian_id: "glm53-calib-v1".into(),
        weights_id: wid.clone(),
    };

    let gu_e0 = pack_triu(
        &[
            1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 6.0, 7.0, 3.0, 6.0, 8.0, 9.0, 4.0, 7.0, 9.0, 10.0,
        ],
        4,
    );
    let dn_e0 = pack_triu(&[2.0, 1.0, 1.0, 3.0], 2);
    let mut gu = gu_e0.clone();
    gu.extend_from_slice(&gu_e0);
    let mut dn = dn_e0.clone();
    dn.extend_from_slice(&dn_e0);
    let exp_path = dir.join(expert_name(3, 0, 1));
    write_experts(&exp_path, 3, 0, &gu, &dn, &[100, 200], (4, 2), &meta).unwrap();

    let raw = std::fs::read(&exp_path).unwrap();
    let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    assert_eq!(hlen % 8, 0);
    let f = File::open(&exp_path).unwrap();
    assert_eq!(f.meta("layer"), Some("3"));
    assert_eq!(f.meta("experts"), Some("0-1"));
    assert_eq!(f.meta("layout"), Some("upper triangle, row-major, fp32"));
    assert_eq!(f.meta("weights_id"), Some(wid.as_str()));
    assert_eq!(f.i64s("count").unwrap(), [100, 200]);
    assert_eq!(f.floats("gu").unwrap().len(), 2 * triu_len(4));

    let mlp = [1.0f32; 16];
    let sh = [2.0f32; 4];
    let dense_path = dir.join(dense_name(3, false));
    write_dense(
        &dense_path,
        3,
        &[
            DenseEntry {
                name: "mlp_in",
                dim: 4,
                data: &mlp,
                count: 300,
            },
            DenseEntry {
                name: "shared_down_in",
                dim: 2,
                data: &sh,
                count: 300,
            },
        ],
        false,
        &meta,
    )
    .unwrap();
    let df = File::open(&dense_path).unwrap();
    assert_eq!(df.meta("sequences"), Some("calibration"));
    assert_eq!(df.i64s("count_mlp_in").unwrap(), [300]);

    let in_path = dir.join(inputs_name(3));
    write_inputs(
        &in_path,
        3,
        2,
        &[0x3f80; 16],
        &[0, 1, 2, 3, 1, 2, 3, 0],
        &[0.5; 8],
        4,
        2,
        &meta,
    )
    .unwrap();
    let inf = File::open(&in_path).unwrap();
    assert_eq!(inf.meta("heldout_sequences"), Some("2"));

    let rt_path = dir.join(route_name(3));
    write_route(
        &rt_path,
        3,
        "all",
        &[0, 1, 2, 3],
        &[0x3c00; 4],
        [1, 2, 2],
        &meta,
    )
    .unwrap();
    let rtf = File::open(&rt_path).unwrap();
    assert_eq!(rtf.meta("rows"), Some("all"));

    std::fs::remove_dir_all(&dir).unwrap();
}
