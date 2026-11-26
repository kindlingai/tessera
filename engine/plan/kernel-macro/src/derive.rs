//! `#[derive(Kernel)]`: the `Kernel` impl from a `#[launch]` line.

use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    AngleBracketedGenericArguments as Generics, Data, DeriveInput, Error, Expr, Fields,
    GenericArgument, Ident, Token, parenthesized, parse_quote, token,
};

/// Keys that take `= expr`.
const VALUES: [&str; 19] = [
    "name", "threads", "sizes", "work", "needs", "order", "load", "accepts", "emits", "emits_at",
    "guess", "runs_on", "source", "launches", "crate", "variant", "wire", "commit", "shape",
];

/// Keys that stand alone.
const FLAGS: [&str; 3] = ["shared", "overwrites", "unproven"];

/// `key`, `key = expr` or `key(args)`. A kernel may take `::<..>`.
struct Entry {
    key: Ident,
    generics: Option<Generics>,
    value: Value,
}

enum Value {
    Flag,
    Expr(Expr),
    Call(Punctuated<Expr, Token![,]>),
}

impl Parse for Entry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key = Ident::parse_any(input)?;
        let generics = match input.peek(Token![::]) {
            true => Some(input.parse()?),
            false => None,
        };
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
        Ok(Entry {
            key,
            generics,
            value,
        })
    }
}

impl Entry {
    fn is_key(&self) -> bool {
        let key = self.key.to_string();
        VALUES.contains(&key.as_str()) || FLAGS.contains(&key.as_str())
    }

    /// Whether the entry has the shape its key asks for.
    fn check(&self) -> Result<(), Error> {
        let key = self.key.to_string();
        let flag = matches!(self.value, Value::Flag);
        let expr = matches!(self.value, Value::Expr(_));
        if FLAGS.contains(&key.as_str()) && !flag {
            return Err(Error::new_spanned(
                &self.key,
                format!("`{key}` stands alone"),
            ));
        }
        if VALUES.contains(&key.as_str()) && !expr {
            return Err(Error::new_spanned(
                &self.key,
                format!("write `{key} = ...`"),
            ));
        }
        match self.is_key() {
            true => Ok(()),
            false => Err(unknown(&self.key)),
        }
    }
}

fn unknown(key: &Ident) -> Error {
    let keys = [VALUES.as_slice(), FLAGS.as_slice()].concat().join(", ");
    Error::new_spanned(key, format!("unknown key `{key}`, the keys are {keys}"))
}

/// The entries after the kernel function: one value per key.
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

/// A scalar field: a count, extent or plain value.
enum Scalar {
    Count,
    Extent,
    Plain,
}

fn scalar(ty: &syn::Type) -> Option<Scalar> {
    match quote!(#ty).to_string().as_str() {
        "usize" => Some(Scalar::Count),
        "Extent" => Some(Scalar::Extent),
        "u32" | "f32" => Some(Scalar::Plain),
        _ => None,
    }
}

/// A path is called with `args`, else `e` is used.
fn applied(e: &Expr, args: &[Tokens]) -> Tokens {
    match e {
        Expr::Path(_) => quote!(#e(#(#args),*)),
        _ => quote!(#e),
    }
}

/// The kernel function, its const generics and build arguments.
struct KernelFn {
    name: Ident,
    generics: Option<Generics>,
    build: Vec<Expr>,
}

impl KernelFn {
    /// The `_ir` call. A `variant` picks the last const generic.
    fn body(&self, variant: Option<&Expr>) -> Tokens {
        let ir = format_ident!("{}_ir", self.name);
        let build = &self.build;
        let with = |value: bool| {
            let mut g = self.generics.clone().unwrap_or_else(|| parse_quote!(::<>));
            g.colon2_token = Some(Default::default());
            g.args.push(GenericArgument::Const(parse_quote!(#value)));
            g
        };
        match variant {
            None => {
                let generics = &self.generics;
                quote!(#ir #generics (#(#build),*))
            }
            Some(v) => {
                let (yes, no) = (with(true), with(false));
                quote!(match #v {
                    true => #ir #yes (#(#build),*),
                    false => #ir #no (#(#build),*),
                })
            }
        }
    }
}

/// The kernel function, if named first.
fn kernel_fn(entries: &mut Vec<Entry>) -> Result<Option<KernelFn>, Error> {
    if entries.first().is_none_or(Entry::is_key) {
        return Ok(None);
    }
    let first = entries.remove(0);
    let build = match first.value {
        Value::Flag => Vec::new(),
        Value::Call(args) => args.into_iter().collect(),
        Value::Expr(_) => return Err(unknown(&first.key)),
    };
    Ok(Some(KernelFn {
        name: first.key,
        generics: first.generics,
        build,
    }))
}

/// Scalar arguments by field name, and the default sizes.
fn fields_of(input: &DeriveInput, krate: &Tokens) -> Result<(Vec<Tokens>, Vec<Tokens>), Error> {
    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(&input.ident, "a struct of sizes"));
    };
    let (mut arms, mut sizes) = (Vec::new(), Vec::new());
    let Fields::Named(fields) = &data.fields else {
        return Ok((arms, sizes));
    };
    for f in &fields.named {
        let ident = f.ident.as_ref().expect("named");
        let Some(kind) = scalar(&f.ty) else { continue };
        let key = ident.to_string();
        arms.push(quote!(#key => #krate::ToArg::arg(&self.#ident),));
        match kind {
            Scalar::Count => sizes.push(quote!(self.#ident)),
            Scalar::Extent => sizes.push(quote!(self.#ident.most())),
            Scalar::Plain => {}
        }
    }
    Ok((arms, sizes))
}

pub fn derive(input: DeriveInput) -> Result<Tokens, Error> {
    let attr = (input.attrs.iter().find(|a| a.path().is_ident("launch"))).ok_or_else(|| {
        Error::new_spanned(&input.ident, "add `#[launch(kernel, threads = ...)]`")
    })?;
    let entries: Punctuated<Entry, Token![,]> =
        attr.parse_args_with(Punctuated::parse_terminated)?;
    let mut entries: Vec<Entry> = entries.into_iter().collect();
    let kernel = kernel_fn(&mut entries)?;
    for e in &entries {
        e.check()?;
    }
    let line = Line { entries };
    let krate = match line.expr("crate") {
        Some(e) => quote!(#e),
        None => quote!(tessera_lower),
    };
    let ty = &input.ident;
    let (arms, sizes) = fields_of(&input, &krate)?;
    let name = match (line.expr("name"), &kernel) {
        (Some(e), _) => quote!(#e),
        (None, Some(k)) => {
            let name = k.name.to_string();
            quote!(#name)
        }
        (None, None) => return Err(Error::new_spanned(attr, "give `name = ...`")),
    };
    let body = kernel.as_ref().map(|k| k.body(line.expr("variant")));
    let source = source(&line, body.as_ref(), &krate, attr)?;
    let launches = launches(&line, body.as_ref(), &krate, &arms, ty, attr)?;
    let sizes = match line.expr("sizes") {
        Some(e) => quote!((#e).to_vec()),
        None => quote!(vec![#(#sizes),*]),
    };
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
        .map(|e| quote!(fn needs(&self) -> #krate::Needs { #e }));
    let order = line
        .expr("order")
        .map(|e| quote!(fn order(&self) -> tessera_graph::Order { #e }));
    let wire = line
        .expr("wire")
        .map(|e| quote!(fn wire(&self) -> Option<tessera_graph::DType> { Some(#e) }));
    let shape = (line.expr("threads").or(line.expr("shape")))
        .map(|e| quote!(fn shape(&self) -> Option<#krate::Shape> { Some((#e).into()) }));
    let guess = guess(&line, &krate)?;
    let runs_on = line
        .expr("runs_on")
        .map(|e| quote!(fn runs_on(&self) -> #krate::ZoneKind { #e }));
    let commit = line.expr("commit").map(|e| {
        let call = applied(e, &[quote!(self)]);
        quote!(fn commit(&self) -> Option<#krate::KernelRef> { #call })
    });
    // An expression may read `body`, the kernel's tree.
    let load = line.expr("load").map(|e| {
        let call = applied(e, &[quote!(self), quote!(len)]);
        let body = body.as_ref().map(|b| quote!(let body = #b;));
        // The group memory comes from what the kernel needs.
        quote!(fn load(&self, len: usize) -> Option<#krate::Load> {
            #body
            let mut load = #call?;
            let needs = #krate::Kernel::needs(self);
            load.group_bytes = (needs.group_bytes + needs.dynamic_shared) as f64;
            Some(load)
        })
    });
    let accepts = line.expr("accepts").map(|e| {
        let call = applied(e, &[quote!(self), quote!(input), quote!(form)]);
        quote!(fn accepts(&self, input: usize, form: &dyn #krate::form::Load) -> bool { #call })
    });
    let emits = line.expr("emits").map(|e| {
        let call = applied(e, &[quote!(self), quote!(form)]);
        quote!(fn emits(&self, form: &dyn #krate::form::Load) -> bool { #call })
    });
    let emits_at = line.expr("emits_at").map(|e| {
        let call = applied(e, &[quote!(self), quote!(output), quote!(form)]);
        quote!(fn emits_at(&self, output: usize, form: &dyn #krate::form::Load) -> bool { #call })
    });
    Ok(quote! {
        #[allow(clippy::all, unused_variables)]
        impl #krate::Kernel for #ty {
            fn name(&self) -> &'static str {
                #name
            }
            #source
            #launches
            #shape
            fn sizes(&self) -> Vec<usize> {
                #sizes
            }
            #work
            #guess
            #needs
            #order
            #wire
            #load
            #accepts
            #emits
            #emits_at
            #runs_on
            #commit
            #shared
            #overwrite
        }
    })
}

/// `source`, else the kernel function's tree under `plan_<name>`.
fn source(
    line: &Line,
    body: Option<&Tokens>,
    krate: &Tokens,
    attr: &syn::Attribute,
) -> Result<Tokens, Error> {
    let body = match (line.expr("source"), body) {
        (Some(e), _) => applied(e, &[quote!(self)]),
        (None, Some(body)) => {
            quote! {
                let entry = format!("plan_{}", #krate::Kernel::name(self));
                #krate::Source::named(#body, entry)
            }
        }
        (None, None) => {
            let note = "name the kernel function first, or give `source = ...`";
            return Err(Error::new_spanned(attr, note));
        }
    };
    Ok(quote!(fn source(&self) -> #krate::Source { #body }))
}

/// `launches`, else one launch over the kernel function's arguments.
fn launches(
    line: &Line,
    body: Option<&Tokens>,
    krate: &Tokens,
    arms: &[Tokens],
    ty: &Ident,
    attr: &syn::Attribute,
) -> Result<Tokens, Error> {
    let step = quote!(step: &#krate::Step);
    if let Some(e) = line.expr("launches") {
        if let Some(threads) = line.expr("threads") {
            return Err(Error::new_spanned(threads, "`launches` sets the threads"));
        }
        let call = applied(e, &[quote!(self), quote!(step)]);
        return Ok(quote!(fn launches(&self, #step) -> Vec<#krate::Launch> { #call }));
    }
    let Some(body) = body else {
        let note = "name the kernel function first, or give `launches = ...`";
        return Err(Error::new_spanned(attr, note));
    };
    let threads = (line.expr("threads"))
        .ok_or_else(|| Error::new_spanned(attr, "give `threads = ...` or `launches = ...`"))?;
    let overwrites = line.flag("overwrites");
    let ty_name = ty.to_string();
    Ok(quote! {
        fn launches(&self, #step) -> Vec<#krate::Launch> {
            let body = #body;
            let scalar = |name: &str| -> #krate::Arg {
                match name {
                    #(#arms)*
                    other => panic!("{} has no field `{other}`", #ty_name),
                }
            };
            let shape: #krate::Shape = (#threads).into();
            vec![#krate::Launch::of(&body, step, #overwrites, &scalar, shape)]
        }
    })
}

/// `guess = expr`, or `unproven` for the shared placeholder.
fn guess(line: &Line, krate: &Tokens) -> Result<Option<Tokens>, Error> {
    let unproven = line.flag("unproven");
    match (line.expr("guess"), unproven) {
        (Some(e), true) => Err(Error::new_spanned(e, "`unproven` is a guess already")),
        (Some(e), false) => Ok(Some(quote!(fn guess(&self) -> f64 { #e }))),
        (None, true) => Ok(Some(quote!(fn guess(&self) -> f64 { #krate::UNPROVEN }))),
        (None, false) => Ok(None),
    }
}
