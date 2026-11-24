//! `#[derive(Weights)]`: one weight declaration per struct field.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenTree};
use quote::{ToTokens, quote};
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Expr, Fields, Ident, Token, parse_macro_input};

/// `#[shape(..)]` in `DimName` terms, or `S0`.. from the parent.
#[proc_macro_derive(Weights, attributes(shape, flatten))]
pub fn derive_weights(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(input, "Weights needs a struct"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(input, "Weights needs named fields"));
    };
    let mut inits = Vec::new();
    for field in &fields.named {
        let name = field.ident.as_ref().expect("named field");
        let ty = &field.ty;
        let mut dims: Vec<Expr> = Vec::new();
        for attr in field.attrs.iter().filter(|a| a.path().is_ident("shape")) {
            let list = attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?;
            dims.extend(list);
        }
        let key = name.to_string();
        let passed = passed(&dims);
        let binds = passed.iter().map(|(ident, i)| {
            quote! { let #ident: ::tessera_trace::DimExpr = given[#i].clone(); }
        });
        // A `#[flatten]` field's fields sit at this struct's path.
        let flat = field.attrs.iter().any(|a| a.path().is_ident("flatten"));
        let (push, pop) = match flat {
            true => (quote! {}, quote! {}),
            false => (
                quote! { path.push(::tessera_trace::PathSeg::Field(#key.into())); },
                quote! { path.pop(); },
            ),
        };
        inits.push(quote! {
            #name: {
                #push
                let shape = {
                    use ::tessera_trace::DimName::*;
                    #(#binds)*
                    [#(::tessera_trace::DimExpr::from(#dims)),*]
                };
                let value = <#ty as ::tessera_trace::Weights>::declare(m, path, &shape);
                #pop
                value
            }
        });
    }
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::tessera_trace::Weights for #ident #ty_generics #where_clause {
            #[allow(unused_imports)]
            #[allow(unused_variables, non_snake_case)]
            fn declare(
                m: &::tessera_trace::Model,
                path: &mut Vec<::tessera_trace::PathSeg>,
                given: &[::tessera_trace::DimExpr],
            ) -> Self {
                Self { #(#inits),* }
            }
        }
    })
}

/// Indices of the `S<i>` names in the shape expressions.
fn passed(dims: &[Expr]) -> Vec<(Ident, usize)> {
    let mut found: Vec<(Ident, usize)> = Vec::new();
    for dim in dims {
        find_passed(dim.to_token_stream(), &mut found);
    }
    found
}

/// Adds each new `S<i>` in `tokens`, also inside groups.
fn find_passed(tokens: proc_macro2::TokenStream, found: &mut Vec<(Ident, usize)>) {
    for tree in tokens {
        let ident = match tree {
            TokenTree::Group(group) => {
                find_passed(group.stream(), found);
                continue;
            }
            TokenTree::Ident(ident) => ident,
            _ => continue,
        };
        let text = ident.to_string();
        let Some(i) = text.strip_prefix('S').and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        if !found.iter().any(|(_, at)| *at == i) {
            found.push((Ident::new(&text, Span::call_site()), i));
        }
    }
}
