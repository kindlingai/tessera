//! `#[derive(Weights)]`: one weight declaration per struct field.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Expr, Fields, Ident, Token, parse_macro_input};

/// `#[shape(..)]` in `DimName` terms, or `S0`.. from the parent.
#[proc_macro_derive(Weights, attributes(shape))]
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
            quote! { let #ident: ::trace::DimExpr = given[#i].clone(); }
        });
        inits.push(quote! {
            #name: {
                path.push(::trace::PathSeg::Field(#key.into()));
                let shape = {
                    use ::trace::DimName::*;
                    #(#binds)*
                    [#(::trace::DimExpr::from(#dims)),*]
                };
                let value = <#ty as ::trace::Weights>::declare(m, path, &shape);
                path.pop();
                value
            }
        });
    }
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::trace::Weights for #ident #ty_generics #where_clause {
            #[allow(unused_imports)]
            #[allow(unused_variables, non_snake_case)]
            fn declare(
                m: &::trace::Model,
                path: &mut Vec<::trace::PathSeg>,
                given: &[::trace::DimExpr],
            ) -> Self {
                Self { #(#inits),* }
            }
        }
    })
}

/// Indices of the `S<i>` names in the shape expressions.
fn passed(dims: &[Expr]) -> Vec<(Ident, usize)> {
    let mut found: Vec<(Ident, usize)> = Vec::new();
    for word in dims.iter().flat_map(|d| d.to_token_stream().into_iter()) {
        let text = word.to_string();
        let Some(i) = text.strip_prefix('S').and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        if !found.iter().any(|(_, at)| *at == i) {
            found.push((Ident::new(&text, Span::call_site()), i));
        }
    }
    found
}
