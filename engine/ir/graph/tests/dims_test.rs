use tessera_graph::DimName::*;
use tessera_graph::{Dim, DimExpr, DimName, Shape};

/// Inline and spilled monomials match symbolic algebra.
#[test]
fn poly_algebra_spill() {
    let dims = vec![
        Dim::bounded(Seqs, 8),
        Dim::bounded(Rows, 16),
        Dim::bounded(Context, 32),
        Dim::bounded(Picks, 4),
        Dim::bounded(Segments, 2),
        Dim::fixed(Hidden, 64),
    ];
    let expr =
        (DimExpr::Dim(Rows) + DimExpr::Dim(Context)) * (DimExpr::Dim(Rows) + DimExpr::Const(2));
    let p = expr.poly(&dims).unwrap();
    let expected = (DimExpr::Dim(Rows) * DimExpr::Dim(Rows)
        + DimExpr::Dim(Rows) * DimExpr::Dim(Context)
        + DimExpr::Const(2) * DimExpr::Dim(Rows)
        + DimExpr::Const(2) * DimExpr::Dim(Context))
    .poly(&dims)
    .unwrap();
    assert_eq!(p, expected);
    assert!(p.covers(&DimExpr::Dim(Rows).poly(&dims).unwrap()));
    assert!(!p.covers(&DimExpr::Const(1).poly(&dims).unwrap()));
    assert_eq!(DimExpr::Const(0).poly(&dims).unwrap().as_const(), Some(0));
    assert_eq!(DimExpr::Const(42).poly(&dims).unwrap().as_const(), Some(42));
    assert_eq!(p.as_const(), None);
    // Five bounded factors spill past the 4-factor inline monomial.
    let big = Shape::from([Segments, Picks, Context, Rows, Seqs])
        .elements(&dims)
        .unwrap();
    let big_rev = Shape::from([Seqs, Rows, Context, Picks, Segments])
        .elements(&dims)
        .unwrap();
    assert_eq!(big, big_rev);
    let size = |d: DimName| match d {
        Seqs => Some(2),
        Rows => Some(3),
        Context => Some(5),
        Picks => Some(7),
        Segments => Some(11),
        _ => None,
    };
    assert_eq!(big.eval(&size), Some(2 * 3 * 5 * 7 * 11));
    assert_eq!(p.eval(&size), Some((3 + 5) * (3 + 2)));
    assert_eq!(DimName::parse("Rows"), Rows);
    let custom = DimName::parse("CustomAxis");
    assert_eq!(custom.as_str(), "CustomAxis");
}
