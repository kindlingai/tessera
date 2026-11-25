use crate::guide::{
    Allowed, Edit, Fault, Guide, Move, Prefix, Program, Scoped, Stacked, Tally, Term, edit,
};
use crate::sample::{Sampler, Sampling};

/// Allows a fixed set and edits fixed ids.
struct Fixed(Allowed, Vec<Edit>);

impl Guide for Fixed {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(self.0.clone())
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        match self.0.allows(id) {
            true => Ok(()),
            false => Err("refused".into()),
        }
    }

    fn edits(&mut self) -> Vec<Edit> {
        self.1.clone()
    }
}

fn add(id: u32, add: f32, rep: f32) -> Edit {
    Edit { id, add, rep }
}

fn stack(members: Vec<Box<dyn Guide>>) -> Box<dyn Guide> {
    Stacked::of(members).expect("a guide")
}

#[test]
fn stack_meets_and_combines() {
    let bits = Allowed::Bits(vec![0b1111_0000, u32::MAX]);
    let few = Allowed::Few(vec![2, 5, 7, 40]);
    let mut g = stack(vec![
        Box::new(Fixed(bits, vec![add(5, 1.0, 2.0)])),
        Box::new(Fixed(few, vec![add(5, -3.0, 1.5), add(7, 0.5, 1.0)])),
        Box::new(Fixed(Allowed::Except(vec![7]), Vec::new())),
    ]);
    assert_eq!(g.allowed().unwrap(), Allowed::Few(vec![5, 40]));
    assert_eq!(g.edits(), [add(5, -2.0, 3.0), add(7, 0.5, 1.0)]);
    assert!(g.accept(7).is_err());
    let mut none = stack(vec![
        Box::new(Fixed(Allowed::One(3), Vec::new())),
        Box::new(Fixed(Allowed::Except(vec![3]), Vec::new())),
    ]);
    assert!(none.allowed().is_err());
}

#[test]
fn stack_shares_forced() {
    let mut g = stack(vec![
        Box::new(Prefix(vec![4, 6])),
        Box::new(Fixed(Allowed::Few(vec![4, 6, 9]), Vec::new())),
    ]);
    assert_eq!(g.forced(), [4, 6]);
    assert!(g.allowed().is_ok());
    let mut bad = stack(vec![
        Box::new(Prefix(vec![4])),
        Box::new(Fixed(Allowed::Few(vec![6, 9]), Vec::new())),
    ]);
    assert_eq!(bad.forced(), [4]);
    assert!(!bad.done() && bad.allowed().is_err());
    let mut split = stack(vec![Box::new(Prefix(vec![4])), Box::new(Prefix(vec![5]))]);
    split.forced();
    assert!(split.allowed().is_err());
}

#[test]
fn edits_steer_host_pick() {
    let mut logits = [2.0, 3.0, -1.0, 2.5];
    edit(
        &mut logits,
        &[add(1, 0.0, 2.0), add(2, 0.5, 2.0), add(9, 5.0, 1.0)],
    );
    assert_eq!(logits, [2.0, 1.5, -1.5, 2.5]);
    let mut greedy = Sampler::new(Sampling::default());
    assert_eq!(greedy.pick(&logits, None), 3);
    assert_eq!(greedy.pick(&logits, Some(&Allowed::Few(vec![0, 1]))), 0);
}

fn moves() -> Vec<Move> {
    vec![
        Move {
            from: 0,
            id: 9,
            to: 1,
        },
        Move {
            from: 1,
            id: 8,
            to: 0,
        },
    ]
}

#[test]
fn scopes_move() {
    let program = Program {
        moves: moves(),
        terms: Vec::new(),
    };
    let mut tally = Tally::new(&program, &[9], 1);
    let mut seen = Vec::new();
    for id in [9, 8, 5, 9] {
        tally.step(&program, id);
        seen.push((tally.scope, tally.since));
    }
    assert_eq!(seen, [(1, 1), (0, 0), (0, 1), (1, 0)]);
}

#[test]
fn counts_follow_scope() {
    let terms = vec![
        Term::Count {
            presence: vec![1.0, 0.0],
            frequency: vec![0.5, 2.0],
        },
        Term::Repeat {
            repetition: vec![1.25, 1.0],
        },
    ];
    let mut g = Scoped::new(
        Program {
            moves: moves(),
            terms,
        },
        &[3],
        0,
    );
    g.accept(4).unwrap();
    g.accept(4).unwrap();
    assert_eq!(g.edits(), [add(3, 0.0, 1.25), add(4, -2.0, 1.25)]);
    g.accept(9).unwrap();
    assert_eq!(g.edits(), [add(4, -4.0, 1.0), add(9, -2.0, 1.0)]);
}

#[test]
fn ramp_is_bounded() {
    let ramp = Term::Ramp {
        id: 2,
        scope: 1,
        start: 2,
        slope: 0.5,
        max: 1.0,
    };
    let program = Program {
        moves: moves(),
        terms: vec![ramp],
    };
    let mut tally = Tally::new(&program, &[], 1);
    let mut adds = Vec::new();
    for id in [5, 5, 5, 5, 5, 8] {
        tally.step(&program, id);
        adds.push(program.edits(&tally).first().map(|e| e.add));
    }
    let want = [None, None, Some(0.5), Some(1.0), Some(1.0), None];
    assert_eq!(adds, want);
}
