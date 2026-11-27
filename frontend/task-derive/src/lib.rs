//! `#[derive(Task)]`: flags, schema and tool calls from one struct.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use syn::meta::ParseNestedMeta;
use syn::{Attribute, Data, DeriveInput, Expr, Fields, Ident, Lit, LitInt, LitStr, Meta, Type};

/// `#[task(..)]` on the struct and `#[arg(..)]` on its fields.
#[proc_macro_derive(Task, attributes(task, arg))]
pub fn derive_task(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The text of the `///` lines in `attrs`.
fn doc(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        let Meta::NameValue(pair) = &attr.meta else {
            continue;
        };
        if let (true, Expr::Lit(lit)) = (pair.path.is_ident("doc"), &pair.value)
            && let Lit::Str(text) = &lit.lit
        {
            lines.push(text.value().trim().to_string());
        }
    }
    lines.join(" ")
}

fn key(meta: &ParseNestedMeta) -> String {
    meta.path
        .get_ident()
        .map(Ident::to_string)
        .unwrap_or_default()
}

fn text(meta: &ParseNestedMeta) -> syn::Result<String> {
    Ok(meta.value()?.parse::<LitStr>()?.value())
}

/// The `#[task(..)]` values.
#[derive(Default)]
struct Head {
    name: Option<String>,
    kind: Option<String>,
    example: Option<String>,
}

fn head(input: &DeriveInput) -> syn::Result<Head> {
    let mut out = Head::default();
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("task")) {
        attr.parse_nested_meta(|meta| {
            let slot = match key(&meta).as_str() {
                "name" => &mut out.name,
                "kind" => &mut out.kind,
                "example" => &mut out.example,
                _ => return Err(meta.error("task takes name, kind and example")),
            };
            *slot = Some(text(&meta)?);
            Ok(())
        })?;
    }
    Ok(out)
}

/// One field and its `#[arg(..)]` values.
struct Arg {
    ident: Ident,
    ty: Type,
    about: String,
    /// The `task::Role` variant: `Flag`, `Words` or `Extra`.
    role: &'static str,
    flag: String,
    value: String,
    tool: bool,
    load: bool,
    min: Option<LitInt>,
    max: Option<LitInt>,
    example: Option<String>,
}

fn arg(field: &syn::Field) -> syn::Result<Arg> {
    let ident = field.ident.clone().expect("named field");
    let name = ident.to_string();
    let mut out = Arg {
        flag: format!("--{}", name.replace('_', "-")),
        value: format!("<{name}>"),
        ident,
        ty: field.ty.clone(),
        about: doc(&field.attrs),
        role: "Flag",
        tool: true,
        load: false,
        min: None,
        max: None,
        example: None,
    };
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("arg")) {
        attr.parse_nested_meta(|meta| {
            match key(&meta).as_str() {
                "positional" => out.role = "Words",
                "extra" => out.role = "Extra",
                "cli_only" => out.tool = false,
                "load" => out.load = true,
                "flag" => out.flag = text(&meta)?,
                "value" => out.value = text(&meta)?,
                "min" => out.min = Some(meta.value()?.parse()?),
                "max" => out.max = Some(meta.value()?.parse()?),
                "example" => out.example = Some(example(meta.value()?.parse()?)?),
                _ => return Err(meta.error("unknown arg key")),
            }
            Ok(())
        })?;
    }
    if out.role == "Extra" {
        out.tool = false;
    }
    Ok(out)
}

/// An example literal as text: a string, or an integer.
fn example(lit: Lit) -> syn::Result<String> {
    match lit {
        Lit::Str(s) => Ok(s.value()),
        Lit::Int(i) => Ok(i.base10_digits().into()),
        other => Err(syn::Error::new_spanned(
            other,
            "example takes text or a number",
        )),
    }
}

fn some<T: quote::ToTokens>(value: &Option<T>) -> Tokens {
    match value {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    }
}

/// The `Field` table entry for `a`.
fn field(a: &Arg) -> Tokens {
    let ty = &a.ty;
    let role = Ident::new(a.role, proc_macro2::Span::call_site());
    let (kind, takes) = match a.role {
        "Flag" => (
            quote! { <#ty as task::Given>::KIND },
            quote! { <#ty as task::Given>::TAKES },
        ),
        "Words" => (quote! { <#ty as task::Arg>::KIND }, quote! { false }),
        _ => (quote! { "array" }, quote! { false }),
    };
    let (key, flag, value, about) = (a.ident.to_string(), &a.flag, &a.value, &a.about);
    let tool = a.tool;
    let (min, max, example) = (some(&a.min), some(&a.max), some(&a.example));
    quote! {
        task::Field {
            key: #key, flag: #flag, value: #value, about: #about,
            kind: #kind, role: task::Role::#role, takes: #takes,
            tool: #tool, min: #min, max: #max, example: #example,
        }
    }
}

/// The field initializers for `parse` and `from_json`, and `push`.
fn bodies(args: &[Arg]) -> (Vec<Tokens>, Vec<Tokens>, Vec<Tokens>) {
    let (mut cli, mut json, mut push) = (Vec::new(), Vec::new(), Vec::new());
    for a in args {
        let (ident, key, flag, load) = (&a.ident, a.ident.to_string(), &a.flag, a.load);
        let (from_cli, put) = match a.role {
            "Flag" => (
                quote! { cli.take(#flag)? },
                quote! { out.flag(#flag, &self.#ident, #load); },
            ),
            "Words" => (quote! { cli.words()? }, quote! { out.words(&self.#ident); }),
            _ => (quote! { cli.extra() }, quote! { out.extra(&self.#ident); }),
        };
        let from_json = match (a.tool, a.role) {
            (false, _) => quote! { Default::default() },
            (true, "Words") => quote! { json.need(#key)? },
            _ => quote! { json.take(#key)? },
        };
        cli.push(quote! { #ident: #from_cli });
        json.push(quote! { #ident: #from_json });
        push.push(put);
    }
    (cli, json, push)
}

fn expand(input: &DeriveInput) -> syn::Result<Tokens> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(input, "Task needs a struct"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(input, "Task needs named fields"));
    };
    let mut args = Vec::new();
    for f in &fields.named {
        args.push(arg(f)?);
    }
    let head = head(input)?;
    let ident = &input.ident;
    let name = head
        .name
        .unwrap_or_else(|| ident.to_string().to_lowercase());
    let (kind, example, about) = (some(&head.kind), some(&head.example), doc(&input.attrs));
    let table: Vec<Tokens> = args.iter().map(field).collect();
    let (cli, json, push) = bodies(&args);
    Ok(quote! {
        const _: () = {
            use ::tessera_tasks::task;
            impl task::Task for #ident {
                const NAME: &'static str = #name;
                const KIND: Option<&'static str> = #kind;
                const ABOUT: &'static str = #about;
                const EXAMPLE: Option<&'static str> = #example;
                const FIELDS: &'static [task::Field] = &[#(#table),*];

                #[allow(unused_variables)]
                fn parse(
                    args: &[String],
                    extra: &dyn Fn(&str) -> Option<bool>,
                ) -> Result<Self, task::Usage> {
                    let cli = task::Cli::split(args, Self::FIELDS, extra)?;
                    Ok(Self { #(#cli),* })
                }

                #[allow(unused_variables)]
                fn from_json(value: &task::Value) -> Result<Self, task::Usage> {
                    let json = task::Json::new(value, Self::FIELDS)?;
                    Ok(Self { #(#json),* })
                }

                #[allow(unused_variables)]
                fn push(&self, out: &mut task::Argv) {
                    #(#push)*
                }
            }
        };
    })
}
