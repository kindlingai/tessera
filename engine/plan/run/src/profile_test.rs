use crate::profile::{Profile, Spent};
use tessera_gpukit::device::KernelAttrs;

#[test]
fn rates_round_trip() {
    let mut profile = Profile::default();
    let mm = Spent {
        seconds: 0.5,
        count: 4,
        work: 1e12,
        bytes: 2e11,
        attrs: None,
    };
    profile.spent.insert("mm 32x1024".into(), mm);
    profile.add("engine: plan", 0.25);
    for i in 0..40 {
        profile.add(&format!("small {i}"), 1e-6);
    }
    let text = profile.report(2).to_string();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 42, "{text}");
    assert!(lines[1].contains("2000.0    400.0  mm 32x1024"), "{text}");
    assert!(lines[2].contains("-        -  engine: plan"), "{text}");
    assert!(lines[41].ends_with("other (2 rows)"), "{text}");
    let bytes = profile.to_bytes();
    assert_eq!(Profile::from_bytes(&bytes), Some(profile));
    assert_eq!(Profile::from_bytes(&bytes[..bytes.len() - 1]), None);
}

#[test]
fn attrs_flag_spills() {
    let mut profile = Profile::default();
    let clean = KernelAttrs {
        regs: 48,
        local_bytes: 0,
        shared_bytes: 16384,
        ctas_per_sm: 6,
    };
    let spill = KernelAttrs {
        regs: 128,
        local_bytes: 64,
        shared_bytes: 1536,
        ctas_per_sm: 2,
    };
    profile.spent.insert(
        "gemv 1x1024".into(),
        Spent {
            seconds: 0.2,
            count: 2,
            work: 1e9,
            bytes: 2e9,
            attrs: Some(clean),
        },
    );
    profile.spent.insert(
        "attn 1x1024".into(),
        Spent {
            seconds: 0.1,
            count: 2,
            work: 1e9,
            bytes: 1e9,
            attrs: Some(spill),
        },
    );
    profile.add("gpu: graph", 0.3);
    for i in 0..39 {
        let attrs = (i == 38).then_some(spill);
        profile.spent.insert(
            format!("tail {i}"),
            Spent {
                seconds: (40 - i) as f64 * 1e-6,
                count: 1,
                attrs,
                ..Spent::default()
            },
        );
    }
    let text = profile.report(1).to_string();
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[0].contains("regs   smem   lmem cta  name")
            || lines[0].contains("regs   smem   lmem  cta  name"),
        "{text}"
    );
    assert!(
        lines[1].contains("-      -      -   -  gpu: graph"),
        "{text}"
    );
    assert!(
        lines[2].contains("48    16K      0   6  gemv 1x1024"),
        "{text}"
    );
    assert!(!lines[2].contains("SPILL"), "{text}");
    assert!(
        lines[3].contains("128   1.5K    64B   2  attn 1x1024  SPILL"),
        "{text}"
    );
    assert!(
        lines.last().unwrap().ends_with("other (2 rows)  SPILL"),
        "{text}"
    );
    let bytes = profile.to_bytes();
    assert_eq!(Profile::from_bytes(&bytes), Some(profile));
}
