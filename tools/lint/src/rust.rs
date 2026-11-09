//! Rust function names and the share of inline tests.

use std::path::Path;

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::visit::Visit;
use syn::{Attribute, Item};

use crate::files::Source;
use crate::names::Name;
use crate::share::TestFile;
use crate::{LIMITS, Violation};

struct Names<'a> {
    src: &'a Source,
    found: &'a mut Vec<Name>,
}

impl Names<'_> {
    fn name(&mut self, ident: &syn::Ident) {
        self.found.push(Name {
            path: self.src.path.clone(),
            line: ident.span().start().line,
            name: ident.to_string(),
            what: "fn",
        });
    }
}

// Trait impls and extern blocks take their names from elsewhere.
impl<'ast> Visit<'ast> for Names<'_> {
    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        self.name(&f.sig.ident);
        syn::visit::visit_item_fn(self, f);
    }
    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        for item in &i.items {
            match item {
                syn::ImplItem::Fn(f) if i.trait_.is_none() => self.name(&f.sig.ident),
                _ => {}
            }
        }
        syn::visit::visit_item_impl(self, i);
    }
    fn visit_item_trait(&mut self, t: &'ast syn::ItemTrait) {
        for item in &t.items {
            if let syn::TraitItem::Fn(f) = item {
                self.name(&f.sig.ident);
            }
        }
        syn::visit::visit_item_trait(self, t);
    }
}

/// `#[test]`, `#[tokio::test]` and `#[cfg(test)]` style attributes.
pub fn is_test(attr: &Attribute) -> bool {
    let path = attr.path();
    if path.segments.last().is_some_and(|s| s.ident == "test") {
        return true;
    }
    if !path.is_ident("cfg") {
        return false;
    }
    let Ok(list) = attr.meta.require_list() else {
        return false;
    };
    has_test(list.tokens.clone())
}

fn has_test(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|t| match t {
        TokenTree::Ident(i) => i == "test",
        TokenTree::Group(g) => has_test(g.stream()),
        _ => false,
    })
}

fn attrs_of(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

/// Lines of inline test items, and the first such line.
pub fn test_lines(items: &[Item]) -> (usize, Option<usize>) {
    let (mut total, mut first) = (0, None);
    for item in items {
        let inline_test = attrs_of(item).iter().any(is_test);
        // `#[cfg(test)] mod x;` only names a test file.
        let declares = matches!(item, Item::Mod(m) if m.content.is_none());
        if inline_test && !declares {
            let (lo, hi) = lines_of(item);
            total += hi - lo + 1;
            first = first.or(Some(lo));
        } else if let Item::Mod(m) = item
            && let Some((_, inner)) = &m.content
        {
            let (n, f) = test_lines(inner);
            total += n;
            first = first.or(f);
        }
    }
    (total, first)
}

/// Lines of `#[test]` functions, including those in inline modules.
fn test_fns(items: &[Item]) -> usize {
    items
        .iter()
        .map(|item| match item {
            Item::Fn(f)
                if f.attrs
                    .iter()
                    .any(|a| is_test(a) && !a.path().is_ident("cfg")) =>
            {
                let (lo, hi) = lines_of(item);
                hi - lo + 1
            }
            Item::Mod(m) => m.content.as_ref().map_or(0, |(_, inner)| test_fns(inner)),
            _ => 0,
        })
        .sum()
}

/// Files named `*test.rs` hold tests and fixtures.
pub fn is_test_file(path: &str) -> bool {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.ends_with("test"))
}

pub fn check(
    src: &Source,
    out: &mut Vec<Violation>,
    names: &mut Vec<Name>,
    tests: &mut Vec<TestFile>,
) {
    let file = match syn::parse_file(&src.text) {
        Ok(f) => f,
        Err(e) => {
            let line = e.span().start().line;
            out.push(src.violation(line, "parse", e.to_string()));
            return;
        }
    };
    Names { src, found: names }.visit_file(&file);
    if is_test_file(&src.path) {
        tests.push(TestFile {
            path: src.path.clone(),
            lines: test_fns(&file.items),
        });
        return;
    }
    let whole = file.attrs.iter().any(is_test);
    let (n, first) = if whole {
        (src.lines, Some(1))
    } else {
        test_lines(&file.items)
    };
    let max = LIMITS.max_test_percent;
    if n * 100 > src.lines * max {
        let message = format!(
            "{n} of {} lines are tests (max {max}%): move them to a *_test.rs file",
            src.lines
        );
        out.push(src.violation(first.unwrap_or(1), "inline-tests", message));
    }
}

fn walk(tokens: TokenStream, lo: &mut usize, hi: &mut usize) {
    for t in tokens {
        let (a, b) = match &t {
            TokenTree::Group(g) => (g.span_open().start().line, g.span_close().end().line),
            _ => (t.span().start().line, t.span().end().line),
        };
        *lo = (*lo).min(a);
        *hi = (*hi).max(b);
        if let TokenTree::Group(g) = t {
            walk(g.stream(), lo, hi);
        }
    }
}

/// First and last source line of a syntax node.
pub fn lines_of(node: &impl ToTokens) -> (usize, usize) {
    let (mut lo, mut hi) = (usize::MAX, 0);
    walk(node.to_token_stream(), &mut lo, &mut hi);
    (lo, hi.max(lo))
}
