//! `#[derive(Weights)]`: one weight declaration per struct field.

use proc_macro::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Expr, Fields, Token, parse_macro_input};

/// Fields take `#[shape(..)]` in `DimName` expressions.
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
        inits.push(quote! {
            #name: {
                path.push(::trace::PathSeg::Field(#key.into()));
                let shape = {
                    use ::trace::DimName::*;
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
            fn declare(
                m: &::trace::Model,
                path: &mut Vec<::trace::PathSeg>,
                _shape: &[::trace::DimExpr],
            ) -> Self {
                Self { #(#inits),* }
            }
        }
    })
}
