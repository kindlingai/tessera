use crate::Dialect;
use crate::form::F32_FORM;
use crate::ir::{Body, Expr, Intrinsic, Param, Stmt};

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
fn metal_narrow_stores() {
    let rendered = call(Intrinsic::Bf16Pair).text().render(Dialect::Metal);
    assert!(rendered.contains("bfloat2((bfloat)lo, (bfloat)hi)"));
    assert!(rendered.contains("*(device bfloat4 *)p = bfloat4(v)"));
    assert!(rendered.contains("as_type<u16>((bfloat)x)"));
    assert!(rendered.contains("0xc407ffffu"));
    assert!(rendered.contains("select(norm, sub"));
}

#[test]
fn every_intrinsic_spelled() {
    // A layout replaces `kept_row` before any text.
    let spelled = Intrinsic::ALL.iter().filter(|i| **i != Intrinsic::KeptRow);
    for i in spelled.copied() {
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
    let e2m1 = call(Intrinsic::E2M1).text().render(Dialect::Metal);
    assert!(e2m1.contains("f_e2m1_lut[16]"));
    let e4m3 = call(Intrinsic::E4M3).text().render(Dialect::Metal);
    assert!(e4m3.contains("as_type<half>"));
    let round = call(Intrinsic::RoundF16).text().render(Dialect::Metal);
    assert!(round.contains("(float)(half)clamp"));
}

#[test]
fn zero4_initializes() {
    let body = Body {
        entry: "zero4".into(),
        params: vec![Param::Buffer {
            name: "out".into(),
            form: F32_FORM,
            writes: true,
        }],
        stmts: vec![
            Stmt::Let {
                name: "o".into(),
                init: Expr::Call(Intrinsic::Zero4, Vec::new()),
            },
            Stmt::Expr(Expr::Call(
                Intrinsic::Store4,
                vec![Expr::var("out"), Expr::Int(0), Expr::var("o")],
            )),
        ],
        ret: None,
        helpers: Vec::new(),
    };
    let text = body.text();
    let cuda = text.render(Dialect::Cuda);
    let metal = text.render(Dialect::Metal);
    let wgsl = body.wgsl().unwrap().text;
    assert!(cuda.contains("#define f_make4(a, b, c, d) make_float4(a, b, c, d)"));
    assert!(metal.contains("#define f_make4(a, b, c, d) float4(a, b, c, d)"));
    for rendered in [&cuda, &metal] {
        assert!(
            rendered.contains("float4 o = f_make4(0.0f, 0.0f, 0.0f, 0.0f);"),
            "{rendered}"
        );
    }
    assert!(wgsl.contains("var o: vec4<f32> = vec4<f32>();"), "{wgsl}");
}
