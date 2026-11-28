use crate::pickle::parse;
use crate::{Error, File, Offer, Torch};
use std::path::PathBuf;

fn text(s: &str) -> Vec<u8> {
    let mut out = vec![b'X'];
    out.extend((s.len() as u32).to_le_bytes());
    out.extend(s.as_bytes());
    out
}

/// `{"w": tensor}` over a `storage` of `count` elements from `offset`.
fn pickle_as(storage: &str, count: u32, offset: u32, shape: &[u32], stride: &[u32]) -> Vec<u8> {
    let int = |p: &mut Vec<u8>, v: u32| {
        p.push(b'J');
        p.extend(v.to_le_bytes());
    };
    let tup = |p: &mut Vec<u8>, vals: &[u32]| {
        p.push(b'(');
        vals.iter().for_each(|&v| int(p, v));
        p.push(b't');
    };
    let mut p = b"\x80\x02}".to_vec();
    p.extend(text("w"));
    p.extend(b"ctorch._utils\n_rebuild_tensor_v2\n((");
    p.extend(text("storage"));
    p.extend(format!("ctorch\n{storage}\n").as_bytes());
    p.extend(text("0"));
    p.extend(text("cpu"));
    int(&mut p, count);
    p.extend(b"tQ");
    int(&mut p, offset);
    tup(&mut p, shape);
    tup(&mut p, stride);
    p.extend(b"\x89ccollections\nOrderedDict\n)RtRs.");
    p
}

fn pickle(count: u8, shape: &[u8], stride: &[u8]) -> Vec<u8> {
    let s: Vec<u32> = shape.iter().map(|&v| u32::from(v)).collect();
    let st: Vec<u32> = stride.iter().map(|&v| u32::from(v)).collect();
    pickle_as("FloatStorage", u32::from(count), 0, &s, &st)
}

/// A stored zip of `files`, as `torch.save` lays one out.
fn archive(name: &str, files: &[(&str, Vec<u8>)]) -> PathBuf {
    let (mut out, mut dir) = (Vec::new(), Vec::new());
    for (file, body) in files {
        let at = out.len() as u32;
        let name = format!("archive/{file}");
        let fixed = |sig: u32, len: usize| {
            let mut h = sig.to_le_bytes().to_vec();
            h.extend([0u8; 14]);
            h.extend((body.len() as u32).to_le_bytes().repeat(2));
            h.extend((name.len() as u16).to_le_bytes());
            h.resize(len, 0);
            h
        };
        out.extend(fixed(0x0403_4b50, 30));
        out.extend(name.as_bytes());
        out.extend(body);
        let mut central = fixed(0x0201_4b50, 46);
        central.copy_within(4..28, 6);
        central[42..46].copy_from_slice(&at.to_le_bytes());
        dir.extend(central);
        dir.extend(name.as_bytes());
    }
    let mut end = 0x0605_4b50u32.to_le_bytes().to_vec();
    end.extend([0u8; 4]);
    end.extend((files.len() as u16).to_le_bytes().repeat(2));
    end.extend((dir.len() as u32).to_le_bytes());
    end.extend((out.len() as u32).to_le_bytes());
    end.extend([0u8; 2]);
    out.extend(dir);
    out.extend(end);
    tessera_testkit::stage(&format!("{name}.pt"), &out)
}

#[test]
fn refuses_code_globals() {
    let named = b"\x80\x02cos\nsystem\nX\x02\x00\x00\x00ls\x85R.";
    let stacked = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93.";
    for bytes in [&named[..], &stacked[..]] {
        let err = parse(bytes).unwrap_err().to_string();
        assert!(err.contains("refused"), "{err}");
    }
}

#[test]
fn reads_strides_within_bounds() {
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let open = |name, count, stride: &[u8], data: &[u8]| {
        let pkl = pickle(count, &[2, 2], stride);
        Torch::open(&archive(
            name,
            &[("data.pkl", pkl), ("data/0", data.to_vec())],
        ))
    };
    let t = open("strided", 4, &[1, 2], &data).unwrap();
    assert_eq!(t.floats("w").unwrap(), [1.0, 3.0, 2.0, 4.0]);
    assert!(matches!(
        open("past", 4, &[4, 1], &data),
        Err(Error::Format(_))
    ));
    assert!(matches!(
        open("short", 4, &[2, 1], &data[..8]),
        Err(Error::Format(_))
    ));
}

/// The cached snapshot `refs/main` names for `repo`.
fn snapshot(repo: &str) -> Option<PathBuf> {
    let hub = tessera_frontend_settings::read().hub?;
    let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let main = std::fs::read_to_string(dir.join("refs/main")).ok()?;
    Some(dir.join("snapshots").join(main.trim()))
}

/// Every tensor in `want`, with the same shape and values.
fn same(got: &Torch, want: &File) {
    for slice in want.slices() {
        let name = &slice.name;
        assert_eq!(got.shape(name), Some(&slice.shape[..]), "{name}");
        let values = got.floats(name).unwrap();
        assert!(values == want.floats(name).unwrap(), "{name}");
    }
}

#[test]
fn clip_matches_safetensors() {
    let (Some(dir), Some(clip)) = (
        tessera_testkit::reference("sprite"),
        snapshot("openai/clip-vit-base-patch32"),
    ) else {
        return eprintln!("no CLIP checkpoints here; skipped");
    };
    let torch = Torch::open(&clip.join("pytorch_model.bin")).unwrap();
    same(
        &torch,
        &File::open(&dir.join("clip/model.safetensors")).unwrap(),
    );
}

#[test]
fn reads_contiguous_and_strided() {
    use tessera_engine::{Bf16, F16};
    let vals: Vec<f32> = (0..18).map(|i| i as f32 * 0.5 - 4.0).collect();
    let bf16: Vec<u8> = vals
        .iter()
        .flat_map(|&v| Bf16::from(v).0.to_le_bytes())
        .collect();
    let open = |name, st, meta: (u32, u32, &[u32], &[u32]), body: Vec<u8>| {
        let pkl = pickle_as(st, meta.0, meta.1, meta.2, meta.3);
        Torch::open(&archive(name, &[("data.pkl", pkl), ("data/0", body)])).unwrap()
    };
    let t = open(
        "c_bf16",
        "BFloat16Storage",
        (18, 2, &[4, 4], &[4, 1]),
        bf16.clone(),
    );
    let want: Vec<f32> = vals[2..18]
        .iter()
        .map(|&v| f32::from(Bf16::from(v)))
        .collect();
    assert_eq!(t.floats("w").unwrap(), want);
    assert_eq!(t.packed("w").unwrap(), &bf16[4..36]);
    assert_eq!(t.packed_rows("w", (1, 2)).unwrap(), &bf16[12..28]);
    assert!(t.packed_rows("w", (0, 0)).unwrap().is_empty());
    assert!(matches!(t.packed_rows("w", (3, 2)), Err(Error::Format(_))));

    let s = open(
        "s_bf16",
        "BFloat16Storage",
        (6, 0, &[3, 2], &[1, 3]),
        bf16[..12].to_vec(),
    );
    let ord: Vec<u8> = [0, 3, 1, 4, 2, 5]
        .iter()
        .flat_map(|&i| bf16[i * 2..i * 2 + 2].iter().copied())
        .collect();
    assert_eq!(s.packed("w").unwrap(), ord);
    assert_eq!(s.packed_rows("w", (1, 2)).unwrap(), &ord[4..12]);

    let all_f16: Vec<u8> = (0u16..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let m = (65_536, 0, &[1, 256, 256][..], &[99_999, 256, 1][..]);
    let f = open("all_f16", "HalfStorage", m, all_f16.clone());
    assert_eq!(f.packed("w").unwrap(), all_f16);
    for (bits, got) in f.floats("w").unwrap().into_iter().enumerate() {
        let w = f32::from(F16(bits as u16));
        assert!(if w.is_nan() {
            got.is_nan()
        } else {
            got.to_bits() == w.to_bits()
        });
    }
    let f32b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f32t = open("c_f32", "FloatStorage", (18, 2, &[4, 4], &[4, 1]), f32b);
    assert_eq!(f32t.floats("w").unwrap(), vals[2..18]);
}
