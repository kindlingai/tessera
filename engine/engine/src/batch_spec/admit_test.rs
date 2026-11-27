use super::admit::{Admit, Ask, Headroom};

#[test]
fn headroom_holds_rows() {
    let mut policy = Headroom::default();
    let ask = |busy, free| Ask {
        busy,
        free,
        prompt: 100,
    };
    assert!(policy.admit(ask(false, 0)));
    assert!(!policy.admit(ask(true, 2147)));
    assert!(policy.admit(ask(true, 2148)));
}
