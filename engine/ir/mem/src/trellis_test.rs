use crate::block_test::same_part;
use crate::form::Form;
use crate::trellis::{
    Codebook, HEAD, INPUT_SIDE, MIX_ROW, MIX_TABLE, TRELLIS_K2, TRELLIS_K2_ROT, TRELLIS_MIX_PMCGX,
    alloc_mixed, form_for, mixed_for, packed, parse_form, parse_mix_form, rotated,
};
use tessera_graph::F16;

/// Each side's cuts, and a set's, decode as the whole.
#[test]
fn trellis_shares() {
    // Noise is valid codes under any header.
    let mut state = 7u32;
    let mut noise = |n: usize| -> Vec<u8> {
        let byte = |_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        };
        (0..n).map(byte).collect()
    };
    let outputs = packed(32, 256, 0, &noise(32 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &outputs, [1, 256, 32], [128, 128]);
    let inputs = packed(256, 128, INPUT_SIDE, &noise(128 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &inputs, [128, 256, 1], [128, 128]);
    assert!(TRELLIS_K2.share(&inputs, [1, 128, 256], [0, 64]).is_none());
    assert!(!TRELLIS_K2.shares(&[128, 256], 0, 64));
    // Across the stored side: tiles from each entry.
    let rows = packed(256, 128, 1, &noise(256 * 128 * 2 / 8));
    same_part(&TRELLIS_K2, &rows, [128, 256, 1], [128, 128]);
    let cols = packed(128, 256, INPUT_SIDE, &noise(128 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &cols, [1, 256, 128], [128, 128]);
    // A set: matrices behind one header, cut whole.
    let (a, b) = (noise(128 * 256 * 2 / 8), noise(128 * 256 * 2 / 8));
    let set = packed(256, 128, INPUT_SIDE, &[a, b.clone()].concat());
    let second = packed(256, 128, INPUT_SIDE, &b);
    for i in (0..128 * 256).step_by(97) {
        let got = TRELLIS_K2.unpack(&set, 128 * 256 + i);
        assert_eq!(got, TRELLIS_K2.unpack(&second, i), "element {i}");
    }
    same_part(&TRELLIS_K2, &set, [1, 2, 128 * 256], [1, 1]);
    same_part(&TRELLIS_K2, &set, [2 * 128, 256, 1], [128, 128]);
    let out_set = packed(
        128,
        256,
        1,
        &[noise(128 * 256 * 2 / 8), noise(128 * 256 * 2 / 8)].concat(),
    );
    same_part(&TRELLIS_K2, &out_set, [2, 256, 128], [128, 128]);
}

/// A rotated file splits into codes, `suh` and `svh`.
#[test]
fn rotated_trellis_splits() {
    let codes = packed(128, 128, 1, &vec![0x5A; 128 * 128 * 2 / 8]);
    // Values f16 holds exactly, so the halves round trip.
    let suh: Vec<f32> = (0..128).map(|i| 1.0 + i as f32 / 64.0).collect();
    let svh: Vec<f32> = (0..128).map(|i| 0.5 - i as f32 / 512.0).collect();
    let half = |v: &[f32]| {
        v.iter()
            .flat_map(|x| F16::from(*x).0.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let file = [codes.clone(), half(&suh), half(&svh)].concat();
    let want = rotated(&TRELLIS_K2.unpack_all(&codes), 128, 128, &suh, &svh);
    assert_eq!(TRELLIS_K2_ROT.unpack_all(&file), want);
    let floats = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let parts = TRELLIS_K2_ROT.split(&file).unwrap();
    assert_eq!(parts, [codes, floats(&suh), floats(&svh)]);
    assert!(TRELLIS_K2.split(&file).is_none());
}

/// mul1 weights match a decode written from the format alone.
#[test]
fn mul1_values() {
    let want = [
        (0, -3.5),
        (1, 0.625),
        (0x1234, -0.5625),
        (0x8000, -0.875),
        (0xBEEF, 0.15625),
        (0xFFFF, -0.625),
        (0x7A5C, -0.9375),
        (0x0F0F, 0.6875),
    ];
    for (state, v) in want {
        assert_eq!(Codebook::Mul1.weight(state, false), v, "state {state:#x}");
    }
}

/// pmcgx pairs match the exlr8 loader's table.
#[test]
fn pmcgx_values() {
    let want = [
        (0, [1.875, 1.875]),
        (1, [-0.203_125, 0.3125]),
        (74565, [1.625, -2.25]),
        (524287, [-2.5, 0.005_859_375]),
        (782069, [-1.25, 2.25]),
    ];
    for (state, pair) in want {
        let got = [false, true].map(|odd| Codebook::Pmcgx.weight(state, odd));
        assert_eq!(got, pair, "state {state}");
    }
}

/// A tile of bytes `37j+11` decodes as exlr8's loader does.
#[test]
fn pmcgx_tiles() {
    let at = [
        (0, 0),
        (1, 0),
        (0, 1),
        (8, 0),
        (0, 8),
        (9, 9),
        (15, 15),
        (7, 12),
    ];
    let want: [(u32, [f32; 8]); 3] = [
        (
            2,
            [0.171_875, -2.25, -0.875, -2.25, -2.0, -1.375, -0.3125, 0.5],
        ),
        (
            3,
            [2.0, -1.0, 0.9375, 0.109_375, -0.156_25, -1.0, -1.25, 0.75],
        ),
        (
            4,
            [-0.625, -2.5, 0.085_937_5, -1.5, -1.625, 0.6875, 1.625, 0.75],
        ),
    ];
    for (k, values) in want {
        let tile: Vec<u8> = (0..32 * k as usize).map(|j| (37 * j + 11) as u8).collect();
        let form = form_for(k, Codebook::Pmcgx).unwrap();
        let t = form.unpack_all(&packed(16, 16, 1, &tile));
        let got = at.map(|(row, col)| t[col * 16 + row]);
        assert_eq!(got, values, "K{k}");
    }
}

/// A mixed set decodes as each matrix at its width.
#[test]
fn mixed_widths_match() {
    let widths = [3u32, 2, 4, 2];
    for (book, (inp, out, side)) in
        Codebook::ALL
            .into_iter()
            .zip([(256, 128, 1), (128, 256, INPUT_SIDE), (256, 256, 1)])
    {
        let mix = mixed_for(book).unwrap();
        let (mut set, at) = alloc_mixed(inp, out, side, &widths);
        let mut want = Vec::new();
        for (m, (k, at)) in widths.iter().zip(at).enumerate() {
            let len = inp * out * *k as usize / 8;
            let codes: Vec<u8> = (0..len).map(|i| (i * 31 + m * 7 + i / 251) as u8).collect();
            set[at..at + len].copy_from_slice(&codes);
            let one = form_for(*k, book).unwrap();
            want.extend(one.unpack_all(&packed(inp, out, side, &codes)));
        }
        assert_eq!(mix.unpack_all(&set), want, "{book:?}");
        for i in (0..want.len()).step_by(89) {
            assert_eq!(mix.unpack(&set, i), want[i], "{book:?} element {i}");
        }
        same_part(mix, &set, [1, 4, inp * out], [1, 2]);
        same_part(mix, &set, [4, out, inp], [0, 128]);
        same_part(mix, &set, [4 * out, inp, 1], [0, 128]);
    }
}

#[test]
fn mix_names_parse() {
    let books = [Codebook::Mcg, Codebook::Mul1, Codebook::Pmcgx];
    for (book, codebook) in (0u32..).zip(books) {
        let name = mixed_for(codebook).unwrap().name();
        assert_eq!(parse_mix_form(name), Some(book));
        assert_eq!(parse_form(name), None);
    }
    assert_eq!(parse_mix_form("trellis_k3_pmcgx"), None);
    assert_eq!(parse_mix_form("trellis_mix_rot"), None);
    assert_eq!(parse_form("trellis_k3_pmcgx"), Some((3, 2)));
}

/// Validates mixed-width bounds, widths, offsets, and geometry.
#[test]
fn mixed_fits_validation() {
    let mix = &TRELLIS_MIX_PMCGX;
    let elems = 2 * 128 * 128;
    let (valid, offsets) = alloc_mixed(128, 128, 1, &[2, 4]);
    assert!(mix.fits(elems, &valid));

    let (valid_in, _) = alloc_mixed(128, 256, INPUT_SIDE, &[3, 2, 4]);
    assert!(mix.fits(3 * 128 * 256, &valid_in));

    // Truncated 4-bit matrix still above 2-bit lower bound.
    let mut trunc = valid.clone();
    trunc.truncate(valid.len() - 64);
    assert!(mix.fits_size(elems, trunc.len()));
    assert!(!mix.fits(elems, &trunc));

    // Unsupported widths in table entries.
    for bad_k in [0u32, 1, 5, u32::MAX] {
        let mut buf = valid.clone();
        buf[MIX_TABLE + 8..MIX_TABLE + 12].copy_from_slice(&bad_k.to_le_bytes());
        assert!(!mix.fits(elems, &buf), "accepted width {bad_k}");
    }

    // Offset inside header, misaligned, out-of-bounds, or overflowing.
    let bad_offsets = [0u64, HEAD as u64, offsets[0] as u64 + 4, u64::MAX - 15];
    for off in bad_offsets {
        let mut buf = valid.clone();
        buf[MIX_TABLE..MIX_TABLE + 8].copy_from_slice(&off.to_le_bytes());
        assert!(!mix.fits(elems, &buf), "accepted offset {off}");
    }
    let mut past = valid.clone();
    let near_end = (past.len() - 64) as u64;
    past[MIX_TABLE + MIX_ROW..MIX_TABLE + MIX_ROW + 8].copy_from_slice(&near_end.to_le_bytes());
    assert!(!mix.fits(elems, &past));

    // Bad header geometry and side values.
    for (inp, out, side) in [
        (0u32, 128u32, 1u32),
        (15, 128, 1),
        (32, 32, 1),
        (16, 256, INPUT_SIDE),
        (128, 128, 99),
    ] {
        let mut buf = valid.clone();
        buf[0..4].copy_from_slice(&inp.to_le_bytes());
        buf[4..8].copy_from_slice(&out.to_le_bytes());
        buf[8..12].copy_from_slice(&side.to_le_bytes());
        let target = 2usize.saturating_mul(inp as usize * out as usize);
        assert!(
            !mix.fits(target, &buf),
            "accepted shape ({inp},{out},{side})"
        );
    }

    // Zero or inflated matrix count.
    for bad_count in [0u32, 3, u32::MAX] {
        let mut buf = valid.clone();
        buf[HEAD..HEAD + 4].copy_from_slice(&bad_count.to_le_bytes());
        assert!(!mix.fits(elems, &buf), "accepted count {bad_count}");
    }

    // Shape-level checks reject same-total different-axes headers.
    assert!(mix.fits_shape(&[2, 128, 128], &valid));
    assert!(!mix.fits_shape(&[1, 256, 128], &valid));
    assert!(!mix.fits_shape(&[4, 128, 64], &valid));
    assert!(!mix.fits_shape(&[elems], &valid));
    assert!(mix.fits_shape(&[3, 256, 128], &valid_in));
    assert!(!mix.fits_shape(&[3, 128, 256], &valid_in));
    assert_eq!(mix.count(&valid), Some(elems));
    assert_eq!(mix.count(&trunc), None);
}
