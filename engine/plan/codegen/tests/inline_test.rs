use tessera_codegen::inline::{Callee, Inline, Inliner};
use tessera_codegen::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt, Ty};
use tessera_codegen::kept::{self, PAGED16};

/// In-place inlining renames locals, groups, and large scopes accurately.
#[test]
fn inline_renames_inplace() {
    let mut stmts = vec![
        Stmt::Let {
            name: "self__acc".into(),
            init: Expr::var("k"),
        },
        Stmt::Assign {
            target: Expr::var("mut_s"),
            op: Some(BinOp::Add),
            value: Expr::lit(1),
        },
    ];
    for i in 0..16 {
        stmts.push(Stmt::Let {
            name: format!("t{i}"),
            init: Expr::bin(BinOp::Add, Expr::var("mut_s"), Expr::lit(i as i64)),
        });
    }
    let callee = Body {
        entry: "inner".into(),
        params: vec![
            Param::Scalar {
                name: "k".into(),
                ty: Ty::U32,
            },
            Param::Scalar {
                name: "mut_s".into(),
                ty: Ty::U32,
            },
        ],
        stmts,
        ret: Some(Expr::var("t15")),
        helpers: Vec::new(),
    };
    let mut inliner = Inliner::default();
    let mut pre = Vec::new();
    let ret = inliner.bind(
        &mut pre,
        Inline {
            callee: Callee::Kernel(callee),
            args: vec![Expr::var("outer_k"), Expr::var("outer_s")],
        },
        "blk",
    );
    assert_eq!(ret, Some(Expr::var("t15__1")));
    assert!(matches!(
        &pre[0],
        Stmt::Let { name, .. } if name == "mut_s__1"
    ));
    assert!(matches!(
        &pre[1],
        Stmt::Let { name, init } if name == "blk__acc" && *init == Expr::var("outer_k")
    ));
    assert!(matches!(
        &pre[18],
        Stmt::Let { name, .. } if name == "t15__1"
    ));
}

/// In-place `kept::attach` rewrites `KeptRow` and inserts `kept_table`.
#[test]
fn attach_rewrites_inplace() {
    let body = Body {
        entry: "kv".into(),
        params: vec![
            Param::Buffer {
                name: "x".into(),
                form: tessera_codegen::form::F32_FORM,
                writes: false,
            },
            Param::Scalar {
                name: "s".into(),
                ty: Ty::U32,
            },
            Param::Scalar {
                name: "p".into(),
                ty: Ty::U32,
            },
        ],
        stmts: vec![Stmt::Let {
            name: "row".into(),
            init: Expr::Call(Intrinsic::KeptRow, vec![Expr::var("s"), Expr::var("p")]),
        }],
        ret: None,
        helpers: Vec::new(),
    };
    let attached = kept::attach(body, &PAGED16);
    assert!(attached.params.iter().any(|p| p.name() == kept::TABLE));
    assert!(attached.params.iter().any(|p| p.name() == "kept_pages"));
    let Stmt::Let { init, .. } = &attached.stmts[0] else {
        panic!("expected let");
    };
    assert!(matches!(init, Expr::Bin(BinOp::BitOr, ..)));
}
