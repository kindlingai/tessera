//! Kernel text belongs in the DSL, not in Rust strings.

use std::str::FromStr;

use proc_macro2::{TokenStream, TokenTree};

use crate::Violation;
use crate::files::Source;
use crate::rust::is_test_file;

/// Tokens strings may not hold, and files exempt from this.
#[derive(serde::Deserialize)]
pub struct Config {
    pub tokens: Vec<String>,
    pub allow: Vec<String>,
}

/// Whether a bracket group is `[doc = ".."]`: a comment.
fn is_doc(g: &proc_macro2::Group) -> bool {
    matches!(g.stream().into_iter().next(), Some(TokenTree::Ident(i)) if i == "doc")
}

/// String literals in `tokens`, with their lines. Comments skip.
fn literals(tokens: TokenStream, out: &mut Vec<(usize, String)>) {
    for t in tokens {
        match t {
            TokenTree::Group(g) if is_doc(&g) => {}
            TokenTree::Group(g) => literals(g.stream(), out),
            TokenTree::Literal(l) => {
                let text = l.to_string();
                if text.starts_with('"') || text.starts_with("r\"") || text.starts_with("r#") {
                    out.push((l.span().start().line, text));
                }
            }
            _ => {}
        }
    }
}

pub fn check(src: &Source, out: &mut Vec<Violation>) {
    let rule = &crate::LIMITS.kernel_text;
    if rule.allow.iter().any(|a| src.path.ends_with(a)) || is_test_file(&src.path) {
        return;
    }
    let Ok(tokens) = TokenStream::from_str(&src.text) else {
        return;
    };
    let mut found = Vec::new();
    literals(tokens, &mut found);
    for (line, text) in found {
        if let Some(token) = rule.tokens.iter().find(|t| text.contains(t.as_str())) {
            let message = format!("`{token}` in a string: write the kernel in the DSL");
            out.push(src.violation(line, "kernel-text", message));
        }
    }
}
