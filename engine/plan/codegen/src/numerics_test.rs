use crate::ir::{BinOp, Body, Expr, Param, Stmt, Ty};
use crate::numerics::{Precision, Terms, terms};
use std::collections::HashMap;

#[test]
fn operands_round() {
    let (hi, lo) = Precision::HIP.split(1.0 + 1.0 / 1024.0 + 1.0 / 4096.0);
    assert_eq!((hi, lo), (1.0 + 1.0 / 1024.0, 1.0 / 4096.0));
    // f16 overflows past 65504; tf32 keeps f32's range.
    assert!(Precision::HIP.split(70_000.0).0.is_infinite());
    assert!(Precision::CUDA.split(70_000.0).0.is_finite());
    // Below 2^-14 an f16 keeps a fixed step of 2^-24.
    let small = 3.0 * 2f32.powi(-20) + 2f32.powi(-26);
    assert_eq!(Precision::HIP.split(small).0, 3.0 * 2f32.powi(-20));
    assert_eq!(Precision::HIP.operand_error(small), 2f32.powi(-25));
    // Pieces sum back to within the stated bits.
    for x in [0.3f32, -7.25, 1e-3, 12345.678] {
        for p in [Precision::CUDA, Precision::HIP] {
            let (hi, lo) = p.split(x);
            assert!(
                (hi + lo - x).abs() <= p.operand_error(x),
                "{x} under {:?}",
                p.operand
            );
        }
    }
    assert_eq!(Precision::EXACT.split(0.3), (0.3, 0.0));
}

#[test]
fn accumulators_truncate() {
    // 22 bits kept: at 1.0 the accumulator's ulp is 2^-21.
    let one_ulp = 2f32.powi(-21);
    // A truncating accumulator drops the half ulp a round keeps.
    assert_eq!(
        Precision::CUDA.accumulate(1.0, one_ulp * 1.5),
        1.0 + one_ulp
    );
    assert_eq!(
        Precision::METAL.accumulate(1.0, one_ulp * 1.5),
        1.0 + one_ulp * 1.5
    );
    assert_eq!(Precision::CUDA.acc_error(), one_ulp);
}

/// A dot product loop over `k`, then a lane sum.
fn dot_body(k: Expr) -> Body {
    let var = |n: &str| Expr::Var(n.into());
    let load = |b: &str| Expr::Index(Box::new(var(b)), Box::new(var("i")));
    Body {
        entry: "dot".into(),
        params: vec![Param::Scalar {
            name: "k".into(),
            ty: Ty::U32,
        }],
        stmts: vec![
            Stmt::Let {
                name: "s".into(),
                init: Expr::Float(0.0),
            },
            Stmt::For {
                var: "i".into(),
                from: Expr::Int(0),
                to: k,
                step: Expr::Int(1),
                unroll: 0,
                body: vec![Stmt::Assign {
                    target: var("s"),
                    op: Some(BinOp::Add),
                    value: Expr::Bin(BinOp::Mul, Box::new(load("x")), Box::new(load("w"))),
                }],
            },
            Stmt::Assign {
                target: var("s"),
                op: None,
                value: Expr::Call(crate::ir::Intrinsic::Sum32, vec![var("s")]),
            },
        ],
        ret: None,
    }
}

#[test]
fn terms_count_sums() {
    let params = HashMap::from([("k".to_string(), 40.0)]);
    let got = terms(&dot_body(Expr::Var("k".into())), &params).unwrap();
    assert_eq!(
        got,
        Terms {
            tile: 0.0,
            lane: 45.0
        }
    );
    // An unbound size leaves the count unknown.
    assert!(terms(&dot_body(Expr::Var("n".into())), &params).is_none());
    // Each lane term costs one f32 rounding.
    let bound = got.matmul(&Precision::EXACT, &[1.0; 2], &[1.0; 2], 2);
    assert!((bound - 2.0 * (2.0 + 45.0) * 2f32.powi(-24)).abs() < 1e-12);
    // Tile terms cost the operand rounding too.
    let tiled = Terms {
        tile: 2.0,
        lane: 1.0,
    };
    let f32_bound = tiled.matmul(&Precision::METAL, &[1.0; 2], &[1.0; 2], 2);
    let f16_bound = tiled.matmul(&Precision::HIP, &[1.0; 2], &[1.0; 2], 2);
    assert!(f16_bound > 4.0 * f32_bound, "{f16_bound} vs {f32_bound}");
}
