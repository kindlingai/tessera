use super::*;
use tessera_guest::site::Input;

fn input(shape: Vec<usize>, form: &str, weight: Option<&str>, parts: Option<[usize; 2]>) -> Input {
    Input {
        dims: vec![None; shape.len()],
        shape,
        form: form.into(),
        weight: weight.map(|w| format!("moe.{w}")),
        share: None,
        parts,
    }
}

fn site(rows: usize, experts: usize, hidden: usize, ffn: usize) -> Site {
    let mut inputs = vec![input(vec![rows, hidden], "f32", None, None)];
    for (i, field) in ROUTED.iter().enumerate() {
        let shape = match i {
            0 => vec![experts, hidden],
            5 => vec![experts, ffn],
            _ => vec![experts, hidden],
        };
        let form = match i {
            0 => "bf16",
            4 | 7 | 10 => "trellis_k3_mul1",
            _ => "f32",
        };
        inputs.push(input(shape, form, Some(field), None));
    }
    for (i, field) in SHARED.iter().enumerate() {
        let parts = if i == 2 { [ffn, hidden] } else { [hidden, ffn] };
        inputs.push(input(vec![], "trellis_k3_mul1", Some(field), Some(parts)));
    }
    Site::Region {
        kind: Kind::Moe,
        inputs,
        outputs: vec![rows * hidden],
        kept: vec![],
        sums: vec![],
        scalars: vec![],
    }
}

fn define(src: &str, name: &str) -> usize {
    let prefix = format!("#define {name} ");
    let line = src.lines().find(|l| l.starts_with(&prefix)).unwrap();
    line[prefix.len()..].trim().parse().unwrap()
}

fn plan_bytes(src: &str) -> usize {
    let from = src.find("struct Plan {").unwrap();
    let body = &src[from..];
    let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
    let (mut off, mut max_align) = (0usize, 1usize);
    for raw in body.lines() {
        let line = raw.trim().trim_end_matches(';');
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (elem, align) = match parts.next().unwrap() {
            "u8" => (1, 1),
            "u16" => (2, 2),
            "u32" | "float" => (4, 4),
            other => panic!("unexpected Plan field type {other}"),
        };
        let mut count = 1usize;
        for seg in parts.next().unwrap().split('[').skip(1) {
            let dim = seg.trim_end_matches(']');
            count *= match dim {
                "MAX_ROWS" => define(src, "MAX_ROWS"),
                "MAX_SLOTS" => define(src, "MAX_SLOTS"),
                "PICKS" => define(src, "PICKS"),
                n => n.parse().unwrap(),
            };
        }
        off = off.div_ceil(align) * align + elem * count;
        max_align = max_align.max(align);
    }
    off.div_ceil(max_align) * max_align
}

#[test]
fn offer_row_bounds() {
    for rows in [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        for experts in [256, 32] {
            let got = offer(&site(rows, experts, 512, 256)).expect("must accept");
            assert_eq!(got.len(), 4);
            let rks = if rows > 1 && !(rows >= 64 && experts % 64 == 0) {
                let nb = experts / 16;
                let rm = rows.div_ceil(16);
                split_dim(512 / 256, 128usize.div_ceil(nb * rm).max(1))
            } else {
                1
            };
            assert_eq!(got[0].len, SCORES_OFF + rows * experts * rks);
            let Shape::Grid { down, .. } = got[1].shape else {
                panic!("fc1 must be grid");
            };
            let cap = if rows > 32 { 64 } else { 8 };
            assert!(down <= rows.div_ceil(cap) + MAX_SLOTS);
        }
    }
    assert!(offer(&site(0, 256, 512, 256)).is_none());
    assert!(offer(&site(2049, 256, 512, 256)).is_none());
    assert!(offer(&site(2049, 32, 512, 256)).is_none());
    let Shape::Grid { down: d512, .. } = offer(&site(512, 256, 512, 256)).unwrap()[1].shape else {
        panic!("fc1 must be grid");
    };
    assert_eq!(d512, 8 + 316);
    let Shape::Grid { down: d2048, .. } = offer(&site(2048, 256, 512, 256)).unwrap()[1].shape
    else {
        panic!("fc1 must be grid");
    };
    assert_eq!(d2048, 32 + 508);
    let Shape::Grid { down: d32, .. } = offer(&site(512, 32, 512, 256)).unwrap()[1].shape else {
        panic!("fc1 must be grid");
    };
    assert_eq!(d32, 8 + 47);
}

#[test]
fn plan_words_match() {
    let cu = include_str!("moe.cu");
    assert_eq!(define(cu, "MAX_ROWS"), MOST_ROWS);
    assert_eq!(define(cu, "MAX_SLOTS"), MAX_SLOTS);
    assert_eq!(define(cu, "PICKS"), PICKS);
    assert_eq!(plan_bytes(cu), PLAN_WORDS * 4);
}

#[test]
fn mixed_fc_widths() {
    let mut s = site(1, 256, 512, 256);
    let Site::Region { inputs, .. } = &mut s else {
        panic!("region");
    };
    // Routed gate and up at K2, down at K3.
    inputs[5].form = "trellis_k2_mul1".into();
    inputs[8].form = "trellis_k2_mul1".into();
    inputs[11].form = "trellis_k3_mul1".into();
    let got = offer(&s).expect("must accept mixed fc1/fc2");
    assert!(matches!(
        got[1].counts[4],
        tessera_guest::site::Count::Fixed(2)
    ));
    assert!(matches!(
        got[2].counts[4],
        tessera_guest::site::Count::Fixed(3)
    ));
}
