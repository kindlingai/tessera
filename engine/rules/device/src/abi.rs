//! Declarations the host and kernels share, printed as kernel text.

use crate::HEADERS;
use std::sync::LazyLock;

/// A type with a name in kernel text.
pub trait Named {
    const C: &'static str;
}

macro_rules! named {
    ($($ty:ident),*) => {$(
        impl Named for $ty {
            const C: &'static str = stringify!($ty);
        }
    )*};
}

named!(u8, u16, u32, u64);

/// A name and its value.
pub type Entry = (&'static str, u32);

/// One struct field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub name: &'static str,
    /// The element type's kernel name.
    pub ty: &'static str,
    /// Each array length, as written.
    pub dims: &'static [&'static str],
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Struct {
    pub name: &'static str,
    /// Zero for the natural alignment.
    pub align: usize,
    pub size: usize,
    pub members: &'static [Member],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    /// `constexpr u32` values.
    Numbers(&'static [Entry]),
    /// A `u32` enum.
    Codes(&'static str, &'static [Entry]),
    Struct(Struct),
}

/// One declaration, under its comment lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decl {
    pub about: &'static [&'static str],
    pub body: Body,
}

impl Decl {
    pub fn entries(&self) -> &'static [Entry] {
        match self.body {
            Body::Numbers(list) | Body::Codes(_, list) => list,
            Body::Struct(_) => &[],
        }
    }

    /// The name of the entry with value `code`.
    pub fn name(&self, code: u32) -> Option<&'static str> {
        let found = self.entries().iter().find(|e| e.1 == code);
        found.map(|e| e.0)
    }
}

/// A struct the host and kernels share.
pub trait Layout: Named {
    const DECL: Decl;
}

fn line(out: &mut String, parts: &[&str]) {
    for part in parts {
        out.push_str(part);
    }
    out.push('\n');
}

fn member(out: &mut String, m: &Member) {
    out.push_str("    ");
    out.push_str(m.ty);
    out.push(' ');
    out.push_str(m.name);
    for dim in m.dims {
        out.push('[');
        out.push_str(dim);
        out.push(']');
    }
    out.push_str(";\n");
}

/// Appends `decl` as kernel text.
pub fn print(out: &mut String, decl: &Decl) {
    for about in decl.about {
        line(out, &["//", about]);
    }
    match decl.body {
        Body::Numbers(list) => {
            for (name, value) in list {
                line(
                    out,
                    &["constexpr u32 ", name, " = ", &value.to_string(), "u;"],
                );
            }
        }
        Body::Codes(name, list) => {
            line(out, &["enum ", name, " : u32 {"]);
            for (name, value) in list {
                line(out, &["    ", name, " = ", &value.to_string(), ","]);
            }
            line(out, &["};"]);
        }
        Body::Struct(s) => {
            let align = s.align.to_string();
            match s.align {
                0 => line(out, &["struct ", s.name, " {"]),
                _ => line(out, &["struct alignas(", &align, ") ", s.name, " {"]),
            }
            for m in s.members {
                member(out, m);
            }
            line(out, &["};"]);
        }
    }
}

/// The header `tessera/1/abi.h`, from every header's declarations.
pub fn text() -> &'static str {
    static TEXT: LazyLock<String> = LazyLock::new(|| {
        let mut out = String::new();
        line(
            &mut out,
            &["// Generated from ", crate::layout::SOURCE, "."],
        );
        line(&mut out, &["\nnamespace tessera {"]);
        for h in &HEADERS {
            let decls = (h.abi)();
            if decls.is_empty() {
                continue;
            }
            if let Some(cap) = h.cap {
                line(&mut out, &["\n#if defined(", cap, ")"]);
            }
            for decl in &decls {
                out.push('\n');
                print(&mut out, decl);
            }
            if h.cap.is_some() {
                line(&mut out, &["\n#endif"]);
            }
        }
        line(&mut out, &["\n}  // namespace tessera"]);
        out
    });
    &TEXT
}
