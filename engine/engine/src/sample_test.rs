use crate::Allowed;
use crate::sample::{Sampler, Sampling, Target, Weights, accept};

/// One pick: the row's weights, then one uniform.
fn pick(s: &mut Sampler, logits: &mut [f32], allowed: Option<&Allowed>) -> u32 {
    let rows = [s.weights(logits, allowed, &[])];
    let units = [s.unit()];
    accept(&[], Target::Drawn(&rows, &units)).1
}

#[test]
fn greedy_matches_argmax() {
    let mut logits = [1.0, 3.0, 3.0, 2.0];
    let mut s = Sampler::new(Sampling::default());
    assert_eq!(pick(&mut s, &mut logits, None), 1);
    let few = Allowed::Few(vec![0, 3]);
    assert_eq!(pick(&mut s, &mut logits, Some(&few)), 3);
    assert_eq!(pick(&mut s, &mut logits, Some(&Allowed::One(0))), 0);
}

#[test]
fn seed_reproduces_stream() {
    let mut logits = [1.0, 1.5, 1.4, 0.8, 1.2];
    let cfg = Sampling {
        temperature: 0.8,
        seed: Some(42),
        ..Sampling::default()
    };
    let mut draw = |c| {
        let mut s = Sampler::new(c);
        let one = Allowed::One(0);
        // A forced pick draws too.
        pick(&mut s, &mut logits, Some(&one));
        let picks = (0..16).map(|_| pick(&mut s, &mut logits, None));
        (picks.collect::<Vec<_>>(), s.unit())
    };
    assert_eq!(draw(cfg), draw(cfg));
    let other = Sampling {
        seed: Some(7),
        ..cfg
    };
    assert_ne!(draw(cfg), draw(other));
}

#[test]
fn filters_restrict_support() {
    let mut logits = [10.0, 9.0, 1.0, 0.0];
    let cfg = Sampling {
        temperature: 1.0,
        top_k: 2,
        seed: Some(1),
        ..Sampling::default()
    };
    let mut s = Sampler::new(cfg);
    for _ in 0..64 {
        assert!(pick(&mut s, &mut logits, None) <= 1);
    }
    let min_p_cfg = Sampling {
        temperature: 1.0,
        min_p: 0.5,
        seed: Some(2),
        ..Sampling::default()
    };
    let mut s = Sampler::new(min_p_cfg);
    for _ in 0..64 {
        assert_eq!(pick(&mut s, &mut [10.0, 8.0, 0.0], None), 0);
    }
    let top_p_cfg = Sampling {
        temperature: 1.0,
        top_p: 0.6,
        seed: Some(3),
        ..Sampling::default()
    };
    let mut s = Sampler::new(top_p_cfg);
    for _ in 0..64 {
        assert_eq!(pick(&mut s, &mut [10.0, 8.0, 0.0], None), 0);
    }
}

#[test]
fn accept_rules() {
    let ids = |d: &[u32], t: &[u32]| accept(d, Target::Ids(t));
    assert_eq!(ids(&[1, 2, 3], &[1, 2, 3, 4]), (3, 4));
    assert_eq!(ids(&[1, 2, 3], &[1, 9, 3, 4]), (1, 9));
    assert_eq!(ids(&[], &[8]), (0, 8));
    let rows = [
        Weights::of(vec![(0, 0.5), (1, 0.4), (2, 0.1)]),
        Weights::of(vec![(0, 0.1), (1, 0.2), (2, 0.7)]),
    ];
    let drawn = |units: &[f32]| accept(&[1], Target::Drawn(&rows, units));
    // The draft holds 0.4 of row 0.
    assert_eq!(drawn(&[0.7, 0.9]), (0, 2));
    assert_eq!(drawn(&[0.3, 0.5]), (1, 2));
    assert_eq!(accept(&[], Target::Drawn(&rows[..1], &[0.6])), (0, 1));
}
