use crate::Dialect;
use crate::ir::{Body, Param};
use tessera_mem::block::Q4_0;

/// A kernel reading a block form carries its decode functions.
#[test]
fn block_headers_render() {
    let body = Body {
        entry: "read".into(),
        params: vec![Param::Buffer {
            name: "w".into(),
            form: &Q4_0,
            writes: false,
        }],
        stmts: Vec::new(),
        ret: None,
        helpers: Vec::new(),
    };
    let text = body.text();
    assert_eq!(text.headers.len(), 2, "one element, then four");
    let metal = text.render(Dialect::Metal);
    assert!(metal.contains("f_f16("), "the scale is an f16 field");
    assert!(metal.contains("f_fn float form_q4_0(f_const(u8) p, u64 i)"));
    assert!(metal.contains("f_fn float4 form_q4_0_x4(f_const(u8) p, u64 i)"));
    assert!(
        metal.contains("f_field(w, "),
        "codes come out as bit fields"
    );
    assert!(metal.contains("device const u8 *w"));
    for f in tessera_mem::form::FORMS.iter() {
        assert!(
            crate::form::named(f.name()).is_some(),
            "missing Load registration for form {}",
            f.name()
        );
    }
    assert_eq!(
        tessera_mem::form::FORMS.iter().count(),
        crate::form::LOADS.iter().count()
    );
}
