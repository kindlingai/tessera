//! Functions with the same body in more than one place.

use std::collections::BTreeMap;

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::visit::Visit;
use syn::{Attribute, ImplItem, Item};

use crate::files::Source;
use crate::rust::is_test;
use crate::{LIMITS, Violation};

/// Words a body keeps when it collapses names.
const KEPT: &[&str] = &[
    "as", "break", "const", "continue", "else", "false", "for", "if", "in", "let", "loop", "match",
    "move", "mut", "ref", "return", "self", "Self", "true", "unsafe", "where", "while", "bool",
    "char", "f32", "f64", "i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize",
    "str",
];

/// Names, numbers and strings collapse. Calls and keywords stay.
fn skeleton(tokens: TokenStream, out: &mut Vec<String>) {
    let mut after_dot = false;
    let mut iter = tokens.into_iter().peekable();
    while let Some(t) = iter.next() {
        match t {
            TokenTree::Ident(i) => {
                let name = i.to_string();
                let call = after_dot
                    || matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '!');
                out.push(if call || KEPT.contains(&name.as_str()) {
                    name
                } else {
                    "a".into()
                });
                after_dot = false;
            }
            TokenTree::Literal(_) => {
                out.push("0".into());
                after_dot = false;
            }
            TokenTree::Punct(p) => {
                out.push(p.as_char().to_string());
                after_dot = matches!(p.as_char(), '.' | ':');
            }
            TokenTree::Group(g) => {
                out.push("(".into());
                skeleton(g.stream(), out);
                out.push(")".into());
                after_dot = false;
            }
        }
    }
}

/// Counts statements, nested ones too.
struct Count(usize);

impl<'ast> Visit<'ast> for Count {
    fn visit_stmt(&mut self, s: &'ast syn::Stmt) {
        self.0 += 1;
        syn::visit::visit_stmt(self, s);
    }
}

struct Func {
    path: String,
    stmts: usize,
    line: usize,
    name: String,
    skeleton: Vec<String>,
}

fn skipped(attrs: &[Attribute]) -> bool {
    attrs.iter().any(is_test)
}

fn push(found: &mut Vec<Func>, path: &str, sig: &syn::Signature, body: &syn::Block) {
    let mut skel = Vec::new();
    skeleton(body.to_token_stream(), &mut skel);
    let mut count = Count(0);
    count.visit_block(body);
    found.push(Func {
        path: path.to_string(),
        stmts: count.0,
        line: sig.ident.span().start().line,
        name: sig.ident.to_string(),
        skeleton: skel,
    });
}

fn collect(path: &str, items: &[Item], found: &mut Vec<Func>) {
    for item in items {
        match item {
            Item::Fn(f) if !skipped(&f.attrs) => push(found, path, &f.sig, &f.block),
            Item::Impl(i) if i.trait_.is_none() => {
                for it in &i.items {
                    if let ImplItem::Fn(f) = it
                        && !skipped(&f.attrs)
                    {
                        push(found, path, &f.sig, &f.block);
                    }
                }
            }
            Item::Mod(m) if !skipped(&m.attrs) => {
                let inner = m.content.iter().flat_map(|(_, i)| i.iter());
                collect(path, &inner.cloned().collect::<Vec<_>>(), found);
            }
            _ => {}
        }
    }
}

pub fn check(sources: &[Source], out: &mut Vec<Violation>) {
    let mut found = Vec::new();
    for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
        if let Ok(file) = syn::parse_file(&src.text) {
            collect(&src.path, &file.items, &mut found);
        }
    }
    let mut groups: BTreeMap<&[String], Vec<&Func>> = BTreeMap::new();
    for f in &found {
        if f.stmts >= LIMITS.min_repeat_statements {
            groups.entry(&f.skeleton).or_default().push(f);
        }
    }
    for (_, fns) in groups.iter().filter(|(_, f)| f.len() > 1) {
        let at = |f: &&Func| format!("{}:{} `{}`", f.path, f.line, f.name);
        let others: Vec<String> = fns[1..].iter().map(at).collect();
        out.push(Violation {
            path: fns[0].path.clone(),
            line: fns[0].line,
            rule: "repeated-fn",
            message: format!(
                "`{}` has the same body as {}",
                fns[0].name,
                others.join(", ")
            ),
        });
    }
}
