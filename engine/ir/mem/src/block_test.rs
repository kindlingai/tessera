use crate::block::{INT2_G32, INT4_G32, INT4_G128, MXFP4, MXFP8, NVFP4, Q4_0, Q8_0};
use crate::form::Form;

/// Each form keeps `data` within a share of its largest.
#[test]
fn blocks_round_trip() {
    let data: Vec<f32> = (0..128)
        .map(|i| ((i * 37) % 23) as f32 * 0.01 - 0.1)
        .collect();
    let forms = [
        (&Q8_0 as &dyn Form, 0.005),
        (&Q4_0, 0.07),
        (&NVFP4, 0.17),
        (&MXFP8, 0.07),
        (&MXFP4, 0.17),
        (&INT4_G32, 0.07),
        (&INT4_G128, 0.07),
        (&INT2_G32, 0.35),
    ];
    for (form, share) in forms {
        let back = form.roundtrip(&data);
        let top = data.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        for (x, y) in data.iter().zip(&back) {
            assert!(
                (x - y).abs() <= top * share,
                "{}: {x} came back {y}",
                form.name()
            );
        }
        let bytes = form.pack(&data);
        assert_eq!(bytes.len(), form.bytes(128));
        // A dot matches the plain sum over the unpacked elements.
        let x: Vec<f32> = (0..128).map(|i| (i % 7) as f32 - 3.0).collect();
        let want: f32 = (0..128).map(|i| x[i] * form.unpack(&bytes, i)).sum();
        let got = form.dot(&x, &bytes, 0);
        assert!(
            (got - want).abs() <= 1e-4,
            "{}: {got} for {want}",
            form.name()
        );
    }
    assert_eq!(
        (Q8_0.bytes(64), Q4_0.bytes(64), NVFP4.bytes(64)),
        (68, 36, 40)
    );
    assert_eq!(
        (MXFP8.bytes(64), MXFP4.bytes(64), INT4_G128.bytes(128)),
        (66, 34, 66)
    );
    assert_eq!(
        (Q4_0.elements(36), NVFP4.offset(32), NVFP4.offset(0)),
        (64, 22, 0)
    );
}

/// A part decodes as the same elements of the whole.
pub(crate) fn same_part(
    form: &dyn Form,
    bytes: &[u8],
    [outer, whole, inner]: [usize; 3],
    [first, count]: [usize; 2],
) {
    let part = form
        .share(bytes, [outer, whole, inner], [first, count])
        .unwrap();
    for o in 0..outer {
        for i in 0..count * inner {
            let at = (o * whole + first) * inner + i;
            let got = form.unpack(&part, o * count * inner + i);
            assert_eq!(got, form.unpack(bytes, at), "{} element {at}", form.name());
        }
    }
}

#[test]
fn shares_round_trip() {
    let data: Vec<f32> = (0..256)
        .map(|i| ((i * 37) % 23) as f32 * 0.01 - 0.1)
        .collect();
    for form in [&Q8_0 as &dyn Form, &Q4_0, &NVFP4, &MXFP8, &INT4_G32] {
        let bytes = form.pack(&data);
        same_part(form, &bytes, [2, 4, 32], [1, 2]);
        same_part(form, &bytes, [4, 2, 32], [1, 1]);
        // Half a block is no part.
        assert!(form.share(&bytes, [16, 2, 8], [1, 1]).is_none());
    }
}

#[test]
fn pack_and_dot_edges() {
    use crate::block::{INT2_G128, Nibble};

    let forms: &[&dyn Form] = &[
        &Q8_0, &Q4_0, &NVFP4, &MXFP8, &MXFP4, &INT4_G32, &INT4_G128, &INT2_G32, &INT2_G128,
    ];
    // Zero blocks must unpack to zero for every code bias.
    let zeros = vec![0.0f32; 128];
    for &form in forms {
        let packed = form.pack(&zeros);
        for i in 0..128 {
            assert_eq!(form.unpack(&packed, i), 0.0, "{} zero {i}", form.name());
        }
        assert_eq!(
            form.dot(&zeros, &packed, 0),
            0.0,
            "{} zero dot",
            form.name()
        );
    }
    // Partial trailing blocks and non-zero block offsets match unpack.
    let data: Vec<f32> = (0..165)
        .map(|i| ((i * 53) % 29) as f32 * 0.05 - 0.7)
        .collect();
    let x: Vec<f32> = (0..165).map(|i| ((i * 19) % 11) as f32 - 5.0).collect();
    for &form in forms {
        let packed = form.pack(&data);
        let at = form.block();
        let tail = &x[at..];
        let want: f32 = tail
            .iter()
            .enumerate()
            .map(|(i, xi)| xi * form.unpack(&packed, at + i))
            .sum();
        let got = form.dot(tail, &packed, at);
        assert!(
            (got - want).abs() <= 1e-3,
            "{} offset tail: {got} vs {want}",
            form.name()
        );
    }
    // Symmetric nibble scale preserves first-occurrence sign on ties.
    use crate::block::Encode;
    assert!(Nibble::scale(&[2.0, -2.0, 0.5]) < 0.0);
    assert!(Nibble::scale(&[-2.0, 2.0, 0.5]) > 0.0);
}
