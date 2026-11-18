use crate::pickle::parse;
use crate::{Error, Torch};
use std::path::PathBuf;

fn text(s: &str) -> Vec<u8> {
    let mut out = vec![b'X'];
    out.extend((s.len() as u32).to_le_bytes());
    out.extend(s.as_bytes());
    out
}

fn ints(values: &[u8]) -> Vec<u8> {
    let mut out = vec![b'('];
    for &v in values {
        out.extend([b'K', v]);
    }
    out.push(b't');
    out
}

/// `{"w": tensor}` over a float storage of `count` elements.
fn pickle(count: u8, shape: &[u8], stride: &[u8]) -> Vec<u8> {
    let mut p = b"\x80\x02}".to_vec();
    p.extend(text("w"));
    p.extend(b"ctorch._utils\n_rebuild_tensor_v2\n((");
    p.extend(text("storage"));
    p.extend(b"ctorch\nFloatStorage\n");
    p.extend(text("0"));
    p.extend(text("cpu"));
    p.extend([b'K', count, b't', b'Q', b'K', 0]);
    p.extend(ints(shape));
    p.extend(ints(stride));
    p.extend(b"\x89ccollections\nOrderedDict\n)RtRs.");
    p
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
    sensai_testkit::stage(&format!("{name}.pt"), &out)
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
