//! `#[derive(Kernel)]`: the `Kernel` impl from a `#[launch]` line.

use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Error, Expr, Fields, Ident, Token, parenthesized, token};

/// `key`, `key = expr` or `key(args)`.
struct Entry {
    key: Ident,
    value: Value,
}

enum Value {
    Flag,
    Expr(Expr),
    Call(Punctuated<Expr, Token![,]>),
}

impl Parse for Entry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key = input.parse()?;
        let value = if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            Value::Expr(input.parse()?)
        } else if input.peek(token::Paren) {
            let inside;
            parenthesized!(inside in input);
            Value::Call(inside.parse_terminated(Expr::parse, Token![,])?)
        } else {
            Value::Flag
        };
        Ok(Entry { key, value })
    }
}

/// The entries after the first: one value per key.
struct Line {
    entries: Vec<Entry>,
}

impl Line {
    fn get(&self, key: &str) -> Option<&Value> {
        let found = self.entries.iter().find(|e| e.key == key);
        found.map(|e| &e.value)
    }

    fn expr(&self, key: &str) -> Option<&Expr> {
        match self.get(key)? {
            Value::Expr(e) => Some(e),
            _ => None,
        }
    }

    fn flag(&self, key: &str) -> bool {
        matches!(self.get(key), Some(Value::Flag))
    }
}

/// A field a scalar parameter can name, by its type.
fn scalar(ty: &syn::Type) -> Option<&'static str> {
    let text = quote!(#ty).to_string();
    match text.as_str() {
        "usize" | "u32" | "f32" | "Extent" => Some(if text == "usize" { "count" } else { "" }),
        _ => None,
    }
}

pub fn derive(input: DeriveInput) -> Result<Tokens, Error> {
    let attr = (input.attrs.iter().find(|a| a.path().is_ident("launch"))).ok_or_else(|| {
        Error::new_spanned(&input.ident, "add `#[launch(kernel, threads = ...)]`")
    })?;
    let entries: Punctuated<Entry, Token![,]> =
        attr.parse_args_with(Punctuated::parse_terminated)?;
    let mut entries: Vec<Entry> = entries.into_iter().collect();
    if entries.is_empty() {
        return Err(Error::new_spanned(attr, "name the kernel function first"));
    }
    let first = entries.remove(0);
    let (ir, build): (Ident, Vec<Expr>) = match first.value {
        Value::Flag => (format_ident!("{}_ir", first.key), Vec::new()),
        Value::Call(args) => (
            format_ident!("{}_ir", first.key),
            args.into_iter().collect(),
        ),
        Value::Expr(e) => {
            return Err(Error::new_spanned(
                e,
                "the kernel function, then its build args",
            ));
        }
    };
    let line = Line { entries };
    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(&input.ident, "a struct of sizes"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new_spanned(&input.ident, "named fields"));
    };
    let ty = &input.ident;
    let fn_name = first.key.to_string();
    let name = match line.expr("name") {
        Some(e) => quote!(#e),
        None => quote!(#fn_name),
    };
    let threads = line
        .expr("threads")
        .ok_or_else(|| Error::new_spanned(attr, "give `threads = ...`"))?;
    let mut arms = Vec::new();
    let mut sizes = Vec::new();
    for f in &fields.named {
        let ident = f.ident.as_ref().expect("named");
        let Some(kind) = scalar(&f.ty) else { continue };
        let key = ident.to_string();
        arms.push(quote!(#key => lower::ToArg::arg(&self.#ident),));
        let text = quote!(#f.ty).to_string();
        if kind == "count" {
            sizes.push(quote!(self.#ident));
        } else if text.ends_with("Extent") {
            sizes.push(quote!(self.#ident.most()));
        }
    }
    let sizes = match line.expr("sizes") {
        Some(e) => quote!((#e).to_vec()),
        None => quote!(vec![#(#sizes),*]),
    };
    let overwrites = line.flag("overwrites");
    let flag = |key: &str, method: Tokens| match line.flag(key) {
        true => quote!(fn #method(&self) -> bool { true }),
        false => quote!(),
    };
    let shared = flag("shared", quote!(shared));
    let overwrite = flag("overwrites", quote!(overwrites));
    let work = line
        .expr("work")
        .map(|e| quote!(fn work(&self, len: usize) -> f64 { let _ = len; (#e) as f64 }));
    let needs = line
        .expr("needs")
        .map(|e| quote!(fn needs(&self) -> lower::Needs { #e }));
    let order = line
        .expr("order")
        .map(|e| quote!(fn order(&self) -> graph::Order { #e }));
    let load = line
        .expr("load")
        .map(|e| quote!(fn load(&self, len: usize) -> Option<lower::Load> { #e(self, len) }));
    let accepts = line.expr("accepts").map(|e| {
        quote!(fn accepts(&self, input: usize, form: &dyn lower::Form) -> bool { #e(self, input, form) })
    });
    let ty_name = ty.to_string();
    Ok(quote! {
        #[allow(clippy::all, unused_variables)]
        impl lower::Kernel for #ty {
            fn name(&self) -> &'static str {
                #name
            }
            fn source(&self) -> lower::Source {
                let entry = format!("plan_{}", lower::Kernel::name(self));
                lower::Source::named(#ir(#(#build),*), entry)
            }
            fn launches(&self, step: &lower::Step) -> Vec<lower::Launch> {
                let body = #ir(#(#build),*);
                let scalar = |name: &str| -> lower::Arg {
                    match name {
                        #(#arms)*
                        other => panic!("{} has no field `{other}`", #ty_name),
                    }
                };
                let shape: lower::Shape = (#threads).into();
                vec![lower::Launch::of(&body, step, #overwrites, &scalar, shape)]
            }
            fn sizes(&self) -> Vec<usize> {
                #sizes
            }
            #work
            #needs
            #order
            #load
            #accepts
            #shared
            #overwrite
        }
    })
}
