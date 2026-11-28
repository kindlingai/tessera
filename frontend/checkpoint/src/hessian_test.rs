use crate::File;
use crate::hessian::{
    DenseEntry, ExpertShard, HessianMeta, Inputs, dense_name, expert_name, inputs_name,
    pack_tiles_into, route_name, triu_len, weights_id, weights_id_bytes, write_dense,
    write_experts, write_inputs, write_route,
};
use crate::st_write::as_le_bytes;

fn pack_triu(full: &[f32], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(triu_len(d));
    for r in 0..d {
        out.extend_from_slice(&full[r * d + r..r * d + d]);
    }
    out
}

#[test]
fn hessian_vectors() {
    let cfg = b"{\"hidden_size\":128}\n";
    let idx = b"{\"weight_map\":{}}\n";
    let want = "24b0d8c953197327441b27e43ecf94a2f3a7adbaf84b22c9ab0123b7c22519a1";
    assert_eq!(weights_id_bytes(cfg, idx), want);

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
                let row = (br * 2 + lr) * 4 + bc * 2;
                tiles.extend_from_slice(&mat[row..row + 2]);
            }
        }
    }
    let mut packed = vec![0.0f32; triu_len(4)];
    pack_tiles_into(&tiles, 4, 2, &mut packed);
    assert_eq!(packed, pack_triu(&mat, 4));
}

#[test]
fn hessian_roundtrip() {
    let dir = tessera_testkit::scratch(&format!("{}-hess-w5", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), b"{\"hidden_size\":128}\n").unwrap();
    let index = b"{\"weight_map\":{}}\n";
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
    let wid = weights_id(&dir).unwrap();
    let meta = HessianMeta {
        hessian_id: "glm53-calib-v1".into(),
        weights_id: wid.clone(),
    };

    let gu = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].repeat(2);
    let dn = [2.0, 1.0, 3.0].repeat(2);
    let exp_path = dir.join(expert_name(3, 0, 1));
    let shard = ExpertShard {
        layer: 3,
        first: 0,
        gu: &as_le_bytes(&gu),
        down: &as_le_bytes(&dn),
        counts: &[100, 200],
        dims: (3, 2),
    };
    write_experts(&exp_path, &shard, &meta).unwrap();
    let f = File::open(&exp_path).unwrap();
    assert_eq!(f.meta("experts"), Some("0-1"));
    assert_eq!(f.meta("weights_id"), Some(wid.as_str()));
    assert_eq!(f.i64s("count").unwrap(), [100, 200]);
    assert_eq!(f.floats("gu").unwrap(), gu);

    let dense_path = dir.join(dense_name(3, false));
    let entry = DenseEntry {
        name: "mlp_in",
        dim: 2,
        data: &[1.0, 2.0, 2.0, 3.0],
        count: 300,
    };
    write_dense(&dense_path, 3, &[entry], false, &meta).unwrap();
    let df = File::open(&dense_path).unwrap();
    assert_eq!(df.meta("sequences"), Some("calibration"));
    assert_eq!(df.i64s("count_mlp_in").unwrap(), [300]);
    assert_eq!(df.floats("mlp_in").unwrap(), [1.0, 2.0, 2.0, 3.0]);

    let inp_path = dir.join(inputs_name(3));
    let inp = Inputs {
        layer: 3,
        heldout_seqs: 2,
        x_bf16: &[0x3f80, 0xc000, 0x3f00, 0x0000],
        topk_idx: &[1, 3, 0, 2],
        topk_w: &[0.75, 0.25, 0.5, 0.5],
        hidden: 2,
        top_k: 2,
    };
    write_inputs(&inp_path, &inp, &meta).unwrap();
    let inf = File::open(&inp_path).unwrap();
    assert_eq!(inf.floats("x").unwrap(), [1.0, -2.0, 0.5, 0.0]);
    assert_eq!(inf.floats("topk_w").unwrap(), [0.75, 0.25, 0.5, 0.5]);

    let rt_path = dir.join(route_name(3));
    write_route(
        &rt_path,
        3,
        "pre",
        &[1, -2],
        &[0x3c00, 0xbc00],
        [1, 1, 2],
        &meta,
    )
    .unwrap();
    let rf = File::open(&rt_path).unwrap();
    assert_eq!(rf.floats("w").unwrap(), [1.0, -1.0]);
    std::fs::remove_dir_all(&dir).unwrap();
}
