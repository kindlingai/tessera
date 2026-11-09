use crate::*;
use DimName::*;

fn dims() -> Vec<Dim> {
    vec![
        Dim {
            name: Rows,
            size: Size::Bounded(64),
        },
        Dim {
            name: Context,
            size: Size::Bounded(64),
        },
        Dim {
            name: Hidden,
            size: Size::Fixed(8),
        },
        Dim {
            name: Heads,
            size: Size::Fixed(2),
        },
        Dim {
            name: HeadDim,
            size: Size::Fixed(4),
        },
    ]
}

#[test]
fn normal_form() {
    let d = dims();
    let p = |e: DimExpr| e.poly(&d).unwrap();
    assert_eq!(p(Hidden * 3), p(DimExpr::from(3) * Hidden));
    assert_eq!(p(Hidden * 3).as_const(), Some(24));
    assert_eq!(p(Rows * Hidden), p(DimExpr::from(8) * Rows));
    assert_ne!(p(Rows.into()), p(Context.into()));
    assert_eq!(p(Rows * Context), p(Context * Rows));
    let at_max = |_: DimName| Some(64);
    assert_eq!(p(Rows * Hidden + 1).eval(&at_max), Some(513));
}

#[test]
fn fixed_by_value() {
    let d = dims();
    let heads = Shape::from([Rows, Heads, HeadDim]);
    assert_eq!(heads.elements(&d), Shape::from([Rows, Hidden]).elements(&d));
    assert_ne!(
        heads.elements(&d),
        Shape::from([Context, Hidden]).elements(&d)
    );
}

#[test]
fn undeclared_dim() {
    assert_eq!(DimExpr::from(Vocab).poly(&dims()), Err(Vocab));
}
