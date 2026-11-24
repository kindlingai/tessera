//! One file's tokens: macro calls, paths, literals and impls.

use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};

pub struct Call {
    pub name: String,
    pub line: usize,
    pub args: TokenStream,
    /// The `macro_rules` body it sits in, by index into `defs`.
    pub def: Option<usize>,
    /// It sits in a `$( ... )` repetition.
    pub fan: bool,
}

pub struct Def {
    pub name: String,
    pub line: usize,
    pub export: bool,
}

/// A name followed by a `{ ... }` body: struct literals, mostly.
pub struct Body {
    pub name: String,
    pub args: TokenStream,
}

#[derive(Default)]
pub struct Scan {
    pub calls: Vec<Call>,
    pub defs: Vec<Def>,
    pub chains: Vec<(usize, Vec<String>)>,
    pub idents: Vec<(usize, String)>,
    pub lits: Vec<(usize, String)>,
    /// The trait name of each `impl Trait for Type`.
    pub impls: Vec<(usize, String)>,
    pub derives: Vec<(usize, String)>,
    /// Lines of `#[test]` attributes.
    pub tests: Vec<usize>,
    pub bodies: Vec<Body>,
}

#[derive(Clone, Copy, Default)]
struct Ctx {
    def: Option<usize>,
    fan: bool,
}

fn line(t: &TokenTree) -> usize {
    t.span().start().line
}

fn is_punct(t: Option<&TokenTree>, c: char) -> bool {
    matches!(t, Some(TokenTree::Punct(p)) if p.as_char() == c)
}

fn ident_of(t: Option<&TokenTree>) -> Option<String> {
    match t {
        Some(TokenTree::Ident(i)) => Some(i.to_string()),
        _ => None,
    }
}

/// A string literal's text, without quotes.
fn string_of(t: &TokenTree) -> Option<String> {
    let TokenTree::Literal(l) = t else {
        return None;
    };
    let s = l.to_string();
    let s = s
        .strip_prefix('r')
        .map_or(s.as_str(), |r| r.trim_matches('#'));
    Some(s.strip_prefix('"')?.strip_suffix('"')?.to_string())
}

pub fn scan(text: &str) -> Scan {
    let mut out = Scan::default();
    if let Ok(ts) = text.parse::<TokenStream>() {
        walk(ts, Ctx::default(), &mut out);
    }
    out
}

fn walk(ts: TokenStream, ctx: Ctx, out: &mut Scan) {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let (mut chain, mut colons, mut export) = (Vec::<String>::new(), 0, false);
    let mut start = 0;
    let mut i = 0;
    while i < toks.len() {
        let t = &toks[i];
        let l = line(t);
        match t {
            TokenTree::Ident(id) => {
                let s = id.to_string();
                out.idents.push((l, s.clone()));
                if colons != 2 || chain.is_empty() {
                    flush(&mut chain, start, out);
                    start = l;
                }
                chain.push(s.clone());
                colons = 0;
                if s == "impl" {
                    impl_of(&toks[i + 1..], l, out);
                }
                if is_punct(toks.get(i + 1), '!') {
                    let name = ident_of(toks.get(i + 2));
                    if let (true, Some(name), Some(TokenTree::Group(g))) =
                        (s == "macro_rules", name, toks.get(i + 3))
                    {
                        let at = out.defs.len();
                        out.defs.push(Def {
                            name,
                            line: line(&toks[i + 2]),
                            export,
                        });
                        export = false;
                        walk(
                            g.stream(),
                            Ctx {
                                def: Some(at),
                                fan: false,
                            },
                            out,
                        );
                        flush(&mut chain, start, out);
                        i += 4;
                        continue;
                    }
                    if let Some(TokenTree::Group(g)) = toks.get(i + 2) {
                        out.calls.push(Call {
                            name: s,
                            line: l,
                            args: g.stream(),
                            def: ctx.def,
                            fan: ctx.fan,
                        });
                        flush(&mut chain, start, out);
                        walk(g.stream(), ctx, out);
                        i += 3;
                        continue;
                    }
                }
                let item = [
                    "struct", "enum", "union", "trait", "impl", "for", "mod", "fn",
                ];
                let defines =
                    i > 0 && ident_of(toks.get(i - 1)).is_some_and(|p| item.contains(&p.as_str()));
                if let Some(TokenTree::Group(g)) = toks.get(i + 1)
                    && g.delimiter() == Delimiter::Brace
                    && !defines
                {
                    out.bodies.push(Body {
                        name: s,
                        args: g.stream(),
                    });
                }
            }
            TokenTree::Punct(p) if p.as_char() == ':' => {
                colons += 1;
                if p.spacing() == Spacing::Alone && colons == 1 {
                    flush(&mut chain, start, out);
                    colons = 0;
                }
            }
            TokenTree::Group(g) => {
                flush(&mut chain, start, out);
                colons = 0;
                let after_hash = i > 0 && is_punct(toks.get(i - 1), '#');
                if after_hash && g.delimiter() == Delimiter::Bracket {
                    export |= attr_of(g.stream(), l, out);
                }
                let fan = ctx.fan
                    || (ctx.def.is_some()
                        && g.delimiter() == Delimiter::Parenthesis
                        && i > 0
                        && is_punct(toks.get(i - 1), '$'));
                walk(g.stream(), Ctx { def: ctx.def, fan }, out);
                if g.delimiter() == Delimiter::Brace {
                    export = false;
                }
            }
            other => {
                flush(&mut chain, start, out);
                colons = 0;
                if let Some(s) = string_of(other) {
                    out.lits.push((l, s));
                }
                if is_punct(Some(other), ';') {
                    export = false;
                }
            }
        }
        i += 1;
    }
    flush(&mut chain, start, out);
}

fn flush(chain: &mut Vec<String>, line: usize, out: &mut Scan) {
    if chain.len() > 1 {
        out.chains.push((line, std::mem::take(chain)));
    }
    chain.clear();
}

/// Records derives. True for `#[macro_export]`.
fn attr_of(ts: TokenStream, l: usize, out: &mut Scan) -> bool {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let head = ident_of(toks.first());
    if head.as_deref() == Some("derive")
        && let Some(TokenTree::Group(g)) = toks.get(1)
    {
        for t in g.stream() {
            if let TokenTree::Ident(id) = t {
                out.derives.push((l, id.to_string()));
            }
        }
    }
    if head.as_deref() == Some("test") && toks.len() == 1 {
        out.tests.push(l);
    }
    head.as_deref() == Some("macro_export")
}

/// The trait in `impl<..> path::Trait<..> for Type`, if any.
fn impl_of(rest: &[TokenTree], l: usize, out: &mut Scan) {
    let (mut depth, mut last) = (0i32, None);
    for t in rest {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => return,
            TokenTree::Punct(p) if p.as_char() == ';' => return,
            TokenTree::Punct(p) if p.as_char() == '<' => depth += 1,
            TokenTree::Punct(p) if p.as_char() == '>' => depth -= 1,
            TokenTree::Ident(id) if id == "for" && depth == 0 => {
                if let Some(name) = last {
                    out.impls.push((l, name));
                }
                return;
            }
            TokenTree::Ident(id) if depth == 0 => last = Some(id.to_string()),
            _ => {}
        }
    }
}

/// String literals at the top level of `args`.
pub(crate) fn top_lits(args: &TokenStream) -> Vec<String> {
    args.clone()
        .into_iter()
        .filter_map(|t| string_of(&t))
        .collect()
}

pub(crate) fn top_idents(args: &TokenStream) -> Vec<String> {
    args.clone()
        .into_iter()
        .filter_map(|t| ident_of(Some(&t)))
        .collect()
}

/// Every identifier in `args`, groups included.
pub(crate) fn all_idents(args: &TokenStream, out: &mut Vec<String>) {
    for t in args.clone() {
        match &t {
            TokenTree::Ident(id) => out.push(id.to_string()),
            TokenTree::Group(g) => all_idents(&g.stream(), out),
            _ => {}
        }
    }
}

/// The name after `static` or `const`. `$name` for a metavariable.
pub(crate) fn static_name(args: &TokenStream) -> Option<String> {
    let toks: Vec<TokenTree> = args.clone().into_iter().collect();
    let at = toks
        .iter()
        .position(|t| matches!(t, TokenTree::Ident(i) if i == "static" || i == "const"))?;
    let mut rest = toks[at + 1..].iter();
    let mut dollar = "";
    for t in rest.by_ref() {
        match t {
            TokenTree::Punct(p) if p.as_char() == '$' => dollar = "$",
            TokenTree::Ident(i) if i == "ref" || i == "mut" => {}
            TokenTree::Ident(i) => return Some(format!("{dollar}{i}")),
            _ => return None,
        }
    }
    None
}
