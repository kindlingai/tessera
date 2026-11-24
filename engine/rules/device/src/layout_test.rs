use crate::abi::{self, Body, Layout};
use crate::layout::*;
use crate::{HEADERS, Header};
use std::io::Write;
use std::process::{Command, Stdio};

/// Asserts that the compiler lays out `T` as Rust does.
fn check<T: Layout>(out: &mut String) {
    let Body::Struct(s) = T::DECL.body else {
        panic!("{:?} is no struct", T::DECL);
    };
    let name = s.name;
    let align = align_of::<T>();
    out.push_str(&format!(
        "static_assert(sizeof({name}) == {} && alignof({name}) == {align}, \"{name}\");\n",
        s.size
    ));
    for m in s.members {
        let (field, at) = (m.name, m.offset);
        out.push_str(&format!(
            "static_assert(__builtin_offsetof({name}, {field}) == {at}, \"{name}.{field}\");\n"
        ));
    }
}

fn hand(name: &str) -> &'static str {
    let found: Option<&Header> = HEADERS.iter().find(|h| h.name.ends_with(name));
    (found.unwrap().text)()
}

/// The host compiler reads `abi.h`. Without one, this skips.
#[test]
fn compiler_agrees() {
    let mut text = String::new();
    for cap in HEADERS.iter().filter_map(|h| h.cap) {
        text.push_str(&format!("#define {cap} 1\n"));
    }
    text.push_str(hand("types.h"));
    text.push_str(abi::text());
    text.push_str("namespace tessera {\n");
    check::<Ref>(&mut text);
    check::<Item>(&mut text);
    check::<Form>(&mut text);
    check::<Launch>(&mut text);
    check::<ConventionRef>(&mut text);
    check::<Fault>(&mut text);
    check::<Table>(&mut text);
    check::<Barrier>(&mut text);
    check::<Flag>(&mut text);
    check::<Lane>(&mut text);
    check::<Pool>(&mut text);
    check::<Counters>(&mut text);
    check::<PoolSet>(&mut text);
    check::<Kept>(&mut text);
    check::<KvTables>(&mut text);
    check::<Request>(&mut text);
    check::<Held>(&mut text);
    check::<StateTables>(&mut text);
    check::<LlmStep>(&mut text);
    check::<ImageStep>(&mut text);
    text.push_str("}\n");
    let args = ["-std=c++17", "-fsyntax-only", "-x", "c++", "-"];
    let child = Command::new("c++")
        .args(args)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        println!("no c++ here, skipped");
        return;
    };
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let log = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{log}\n{text}");
}

#[test]
fn form_kinds_hold() {
    for pair in FORM_KINDS.windows(2) {
        assert_ne!(pair[0].1, pair[1].1, "{pair:?} share a code");
    }
    // `trellis.h` reads these nine as one range.
    let trellis = |e: &&abi::Entry| e.0.starts_with("FORM_TRELLIS_");
    let codes: Vec<u32> = FORM_KINDS.iter().filter(trellis).map(|e| e.1).collect();
    assert_eq!(codes, (8..=16).collect::<Vec<u32>>());
    for f in tessera_lower::form::LOADS.iter() {
        if dtype_code(f.name()) == BYTES {
            assert_ne!(form_kind(f.name()), 0, "no form_kind for {}", f.name());
        }
    }
    assert_eq!(form_kind("trellis_k4_rot"), form_kind("trellis_k4"));
}
