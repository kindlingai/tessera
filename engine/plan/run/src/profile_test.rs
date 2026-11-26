use crate::profile::{Profile, Spent};

#[test]
fn rates_round_trip() {
    let mut profile = Profile::default();
    let mm = Spent {
        seconds: 0.5,
        count: 4,
        work: 1e12,
        bytes: 2e11,
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
