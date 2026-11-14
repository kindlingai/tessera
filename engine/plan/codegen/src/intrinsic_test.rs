use crate::form::F32_FORM;
use crate::ir::{Body, Expr, Intrinsic, Param, Stmt};
use crate::{Dialect, kept};

/// One call of `i`, with a buffer as every argument.
fn call(i: Intrinsic) -> Body {
    let args = vec![Expr::var("x"); i.arity()];
    let stmt = match i {
        Intrinsic::Sum32 => Stmt::Assign {
            target: Expr::var("x"),
            op: None,
            value: Expr::Call(i, args),
        },
        _ => Stmt::Expr(Expr::Call(i, args)),
    };
    Body {
        entry: format!("{i:?}"),
        layout: kept::named("dense"),
        params: vec![Param::Buffer {
            name: "x".into(),
            form: F32_FORM,
            writes: true,
        }],
        stmts: vec![stmt],
        ret: None,
        helpers: Vec::new(),
    }
}

/// Each `f_` name `text` calls.
fn device_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
            continue;
        }
        if c == '(' && word.starts_with("f_") {
            names.push(word.clone());
        }
        word.clear();
    }
    names
}

#[test]
fn every_intrinsic_spelled() {
    for i in Intrinsic::ALL.iter().copied() {
        let text = call(i).text();
        for dialect in [Dialect::Metal, Dialect::Cuda] {
            let rendered = text.render(dialect);
            for name in device_names(&text.body) {
                let define = format!("#define {name}(");
                assert!(
                    rendered.contains(&define),
                    "{i:?}: no `{name}` in {dialect:?}"
                );
            }
        }
    }
}
