use super::fp4::*;

#[test]
fn e2m1_roundtrip_magnitudes() {
    for (bits, val) in [
        (0x0, 0.0),
        (0x1, 0.5),
        (0x2, 1.0),
        (0x3, 1.5),
        (0x4, 2.0),
        (0x5, 3.0),
        (0x6, 4.0),
        (0x7, 6.0),
    ] {
        assert!((e2m1_to_f32(bits) - val).abs() < 1e-6, "bits={bits}");
        assert_eq!(e2m1_from_f32(val), bits);
        assert_eq!(e2m1_from_f32(-val), bits | 0x8);
    }
}

#[test]
fn fp8_e4m3_encoder_known_table() {
    let table: &[(f32, u8)] = &[
        (0.0, 0x00),
        (1.0 / 512.0, 0x01),
        (1.0 / 64.0, 0x08),
        (0.25, 0x28),
        (0.5, 0x30),
        (1.0, 0x38),
        (1.5, 0x3C),
        (2.0, 0x40),
        (8.0, 0x50),
        (448.0, 0x7E),
    ];
    for &(value, expected_bits) in table {
        let got = fp8_e4m3_from_f32(value);
        assert_eq!(
            got, expected_bits,
            "encode({value}): got 0x{got:02X} expected 0x{expected_bits:02X}"
        );
        assert!(
            (fp8_e4m3_to_f32(got) - value).abs() < 1e-3 * value.abs().max(1e-6),
            "encode/decode roundtrip for {value}: got {}",
            fp8_e4m3_to_f32(got)
        );
    }
}

#[test]
fn fp8_e4m3_known_table() {
    let table: &[(u8, f32)] = &[
        (0x00, 0.0),
        (0x01, 2f32.powi(-9)),
        (0x08, 2f32.powi(-6)),
        (0x38, 1.0),
        (0x3C, 1.5),
        (0x40, 2.0),
        (0x7E, 448.0),
    ];
    for &(bits, expected) in table {
        let got = fp8_e4m3_to_f32(bits);
        assert!(
            (got - expected).abs() < 1e-6,
            "0x{bits:02X}: got {got} expected {expected}"
        );
    }
    assert!(fp8_e4m3_to_f32(0x7F).is_nan());
}

#[test]
fn fp8_e4m3_roundtrip_non_nan() {
    for bits in 0u8..=0x7E {
        if bits == 0x7F {
            continue;
        }
        let decoded = fp8_e4m3_to_f32(bits);
        if decoded.is_nan() || decoded == 0.0 {
            continue;
        }
        let reenc = fp8_e4m3_from_f32(decoded);
        let again = fp8_e4m3_to_f32(reenc);
        assert!(
            (again - decoded).abs() < 0.02 * decoded.abs().max(1e-3),
            "bits=0x{bits:02X} decoded={decoded} reenc=0x{reenc:02X} again={again}"
        );
    }
}

#[test]
fn fp8_scale_half() {
    let bits = fp8_e4m3_from_f32(0.5);
    assert_eq!(bits, 48);
    assert!((fp8_e4m3_to_f32(bits) - 0.5).abs() < 1e-3);
}
