use super::interleave::{Cooldown, Interleave, Work};

#[test]
fn cooldown_spaces_prefill() {
    let work = |prefilling, least| Work {
        prefilling,
        active: 1,
        least,
    };
    let mut policy = Cooldown::default();
    assert!(policy.prefill(work(1, 9000)));
    policy.fed(2048, work(1, 6952));
    let rounds: Vec<bool> = (0..3).map(|_| policy.prefill(work(1, 6952))).collect();
    assert_eq!(rounds, [false, false, true]);
    policy.fed(8192, work(1, 400));
    assert!(policy.prefill(work(1, 400)), "a near end skips the wait");
    assert!(!policy.prefill(work(0, usize::MAX)));
}
