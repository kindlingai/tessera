use crate::norm::add_rms_norm::AddLaneNorm;
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Device, Limits, Platform};

#[test]
fn sum_form_key_order() {
    let base = AddLaneNorm {
        rows: 4096,
        width: 4096,
        eps: 1e-5,
        add3: false,
        weighted: false,
        save_sum: true,
        block: true,
        warps: 8,
        cache_trips: 0,
        ahead: 1,
        a_form: F32_FORM,
        b_form: F32_FORM,
        c_form: F32_FORM,
        w_form: F32_FORM,
        out_form: F32_FORM,
        sum_form: F32_FORM,
    };
    let b = AddLaneNorm {
        sum_form: BF16_FORM,
        ..base
    };
    let dev = || {
        Device::default().on(Platform {
            name: "NVIDIA GB10".into(),
            backend: "cuda".into(),
            units: 48,
            bandwidth: 273e9,
            limits: Limits::assumed(),
            ..Default::default()
        })
    };
    let n = base.rows * base.width;
    let (x, y) = (dev().seconds(&base, n), dev().seconds(&b, n));
    let (ab, ba) = (dev(), dev());
    let got = (
        ab.seconds(&base, n),
        ab.seconds(&b, n),
        ba.seconds(&b, n),
        ba.seconds(&base, n),
    );
    assert_ne!(x, y);
    assert_eq!(got, (x, y, y, x));
}
