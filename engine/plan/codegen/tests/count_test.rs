use tessera_codegen::form::{BF16_FORM, F32_FORM};
use tessera_codegen::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt, Ty};

/// Repeated global loads deduplicate and derived vars jump.
#[test]
fn trip_dedups_derived() {
    let idx = Expr::cast(Ty::U64, Expr::var("k"));
    let load_x = Expr::index(Expr::var("x"), idx.clone());
    let body = Body {
        entry: "dedup".into(),
        params: vec![
            Param::Buffer {
                name: "x".into(),
                form: F32_FORM,
                writes: false,
            },
            Param::Buffer {
                name: "w".into(),
                form: BF16_FORM,
                writes: false,
            },
            Param::Scalar {
                name: "n".into(),
                ty: Ty::U32,
            },
        ],
        stmts: vec![Stmt::For {
            var: "k".into(),
            from: Expr::lit(0),
            to: Expr::var("n"),
            step: Expr::lit(4),
            unroll: 0,
            body: vec![
                Stmt::Let {
                    name: "off".into(),
                    init: Expr::bin(BinOp::Mul, Expr::var("k"), Expr::lit(3)),
                },
                Stmt::Let {
                    name: "sum".into(),
                    init: Expr::bin(BinOp::Add, load_x.clone(), load_x),
                },
                Stmt::Let {
                    name: "v".into(),
                    init: Expr::Call(
                        Intrinsic::Load4,
                        vec![Expr::var("w"), Expr::cast(Ty::U64, Expr::var("off"))],
                    ),
                },
            ],
        }],
        ret: None,
        helpers: Vec::new(),
    };
    let trip = body.trip();
    assert_eq!(trip.loads, 2.0);
    assert_eq!(trip.width, 6.0);
    // Mean of 0.25 (x[k]) and 1.0 (derived off).
    assert!((trip.fresh - 0.625).abs() < 1e-6, "fresh={}", trip.fresh);
}
