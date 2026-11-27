//! `#[kernel]` keeps the function and adds one building its tree.

mod derive;
mod stmt;

use proc_macro::TokenStream;
use proc_macro2::{TokenStream as Tokens, TokenTree};
use quote::{format_ident, quote};
use stmt::{block, build_group, named};
use syn::spanned::Spanned;
use syn::{
    BinOp, Error, Expr, ExprCall, ExprLit, FnArg, GenericArgument, GenericParam, Generics, Ident,
    ItemFn, Lit, LitStr, Pat, PathArguments, ReturnType, Stmt, Type, UnOp, parse_macro_input,
};

type Res = Result<Tokens, Error>;

/// Names that are values when the tree is built.
struct Ctx {
    build: Vec<String>,
}

/// A `Kernel` impl from the struct and its `#[launch]` line.
#[proc_macro_derive(Kernel, attributes(launch))]
pub fn derive_kernel(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);
    match derive::derive(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// A `Group` impl: one kernel parameter per field.
#[proc_macro_derive(Group)]
pub fn derive_group(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);
    match group(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Vector lanes, which `v.x` and the like read.
const LANES: [&str; 4] = ["x", "y", "z", "w"];

/// Each field's parameter, named `{name}__{field}`.
fn group(input: &syn::DeriveInput) -> Res {
    let syn::Data::Struct(data) = &input.data else {
        return Err(Error::new(input.span(), "a struct of named fields"));
    };
    let mut params = Vec::new();
    for f in &data.fields {
        let Some(ident) = &f.ident else {
            return Err(Error::new(f.span(), "name each field"));
        };
        let field = ident.to_string();
        if LANES.contains(&field.as_str()) {
            return Err(Error::new(ident.span(), "x, y, z and w are vector lanes"));
        }
        params.push(param(
            &quote!(tessera_codegen::inline::member(name, #field)),
            &f.ty,
            None,
        )?);
    }
    let ty = &input.ident;
    Ok(quote! {
        impl tessera_codegen::inline::Group for #ty {
            fn params(name: &str) -> Vec<tessera_codegen::ir::Param> {
                #[allow(unused_imports)]
                use tessera_codegen::ir::{Param, Ty};
                vec![#(#params),*]
            }
        }
    })
}

#[proc_macro_attribute]
pub fn kernel(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut f = parse_macro_input!(item as ItemFn);
    let tree = tree(&mut f, None);
    expand(&f, tree)
}

/// A function defined once, named by the format string.
#[proc_macro_attribute]
pub fn helper(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut f = parse_macro_input!(item as ItemFn);
    let name = match attr.is_empty() {
        true => {
            let name = f.sig.ident.to_string();
            quote!(#name.to_string())
        }
        false => {
            let pattern = parse_macro_input!(attr as LitStr);
            quote!(format!(#pattern))
        }
    };
    let tree = tree(&mut f, Some(name));
    expand(&f, tree)
}

/// The function as written, then its tree builders.
fn expand(f: &ItemFn, tree: Res) -> TokenStream {
    match tree {
        Ok(tree) => quote! {
            #[allow(
                dead_code,
                unused_assignments,
                unused_mut,
                unused_variables,
                unused_parens,
                clippy::all
            )]
            #f
            #tree
        }
        .into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Two siblings: the builder, and the call to inline.
fn tree(f: &mut ItemFn, helper: Option<Tokens>) -> Res {
    let name = &f.sig.ident;
    let ir_name = format_ident!("{name}_ir");
    let inline_name = format_ident!("{name}_inline");
    let vis = &f.vis;
    // Type parameters only the Rust signature reads would hang unbound.
    let generics = tree_generics(f);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let turbofish = ty_generics.as_turbofish();
    let entry = name.to_string();
    let (mut params, mut build, mut ctx) = (Vec::new(), Vec::new(), Ctx { build: Vec::new() });
    let (mut takes, mut binds, mut passes) = (Vec::new(), Vec::new(), Vec::new());
    for (n, arg) in f.sig.inputs.iter_mut().enumerate() {
        let FnArg::Typed(p) = arg else {
            return Err(Error::new(arg.span(), "a kernel cannot take self"));
        };
        let Pat::Ident(ident) = &*p.pat else {
            return Err(Error::new(p.pat.span(), "name each parameter"));
        };
        let name = ident.ident.to_string();
        let at_build = p.attrs.iter().any(|a| a.path().is_ident("build"));
        let form = p
            .attrs
            .iter()
            .find(|a| a.path().is_ident("form"))
            .map(|a| a.parse_args::<Ident>())
            .transpose()?;
        p.attrs
            .retain(|a| !a.path().is_ident("build") && !a.path().is_ident("form"));
        let taken = format_ident!("a{n}");
        match at_build {
            true => {
                let (ident, ty) = (&ident.ident, &p.ty);
                build.push(quote!(#ident: #ty));
                ctx.build.push(name);
                takes.push(quote!(#taken: impl tessera_codegen::inline::Pass<#ty>));
                binds.push(quote!(let #ident = #taken.pass();));
                passes.push(quote!(#ident));
            }
            false => {
                takes.push(
                    quote!(#taken: impl tessera_codegen::inline::Pass<tessera_codegen::ir::Expr>),
                );
                if let Some(g) = group_ty(&p.ty) {
                    params.push(quote!(<#g as tessera_codegen::inline::Group>::params(#name)));
                    binds.push(
                        quote!(args.extend(tessera_codegen::inline::fields::<#g>(#taken.pass()));),
                    );
                } else {
                    let one = param(&quote!(#name), &p.ty, form)?;
                    params.push(quote!(vec![#one]));
                    binds.push(quote!(args.push(#taken.pass());));
                }
            }
        }
    }
    let (stmts, ret) = match (&f.sig.output, f.block.stmts.split_last()) {
        (ReturnType::Default, _) => (block(&f.block.stmts, &ctx)?, quote!(None)),
        // A returned group: the caller's `let` names its fields.
        (ReturnType::Type(_, ty), Some((Stmt::Expr(Expr::Struct(lit), None), rest)))
            if group_ty(ty).is_some() =>
        {
            let (rest, built) = (
                block(rest, &ctx)?,
                build_group(&quote!(tessera_codegen::inline::BUILT), lit, &ctx)?,
            );
            (
                quote!({ let mut b = #rest; b.extend(#built); b }),
                quote!(None),
            )
        }
        (ReturnType::Type(..), Some((Stmt::Expr(e, None), rest))) => {
            let value = expr(e, &ctx)?;
            let ret = quote! {{
                let mut pre: Vec<Stmt> = Vec::new();
                let value = #value;
                stmts.extend(pre);
                Some(value)
            }};
            (block(rest, &ctx)?, ret)
        }
        (ReturnType::Type(..), _) => {
            return Err(Error::new(f.sig.output.span(), "end with the value"));
        }
    };
    let body = quote! {
        Body {
            entry: #entry.into(),
            params: {
                let mut params: Vec<Param> = Vec::new();
                #(params.extend(#params);)*
                params
            },
            stmts,
            ret,
            helpers: inl.helpers,
        }
    };
    let (item, built, callee) = match helper {
        None => (quote!(tessera_codegen::ir::Body), body, quote!(Kernel)),
        Some(name) => {
            let ReturnType::Type(_, ty) = &f.sig.output else {
                return Err(Error::new(f.sig.span(), "a helper has a value"));
            };
            let ret = value_ty(ty)?;
            let built = quote!(tessera_codegen::ir::Helper { name: #name, ret: #ret, body: #body });
            (quote!(tessera_codegen::ir::Helper), built, quote!(Helper))
        }
    };
    Ok(quote! {
        #[allow(clippy::all, unused_mut, unused_parens)]
        #vis fn #ir_name #impl_generics (#(#build),*) -> #item #where_clause {
            #[allow(unused_imports)]
            use tessera_codegen::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt, Ty};
            let mut inl = tessera_codegen::inline::Inliner::default();
            let mut stmts: Vec<Stmt> = #stmts;
            let ret: Option<Expr> = #ret;
            #built
        }

        #[allow(clippy::too_many_arguments, unused_mut)]
        #vis fn #inline_name #impl_generics (#(#takes),*) -> tessera_codegen::inline::Inline #where_clause {
            #[allow(unused_imports)]
            use tessera_codegen::inline::Pass;
            let mut args: Vec<tessera_codegen::ir::Expr> = Vec::new();
            #(#binds)*
            tessera_codegen::inline::Inline {
                callee: tessera_codegen::inline::Callee::#callee(#ir_name #turbofish (#(#passes),*)),
                args,
            }
        }
    })
}

/// A helper's type: a scalar, a `F32x4`, or `Buf<U8>` bytes.
fn value_ty(ty: &Type) -> Res {
    let text = quote!(#ty).to_string().replace(' ', "");
    match text.as_str() {
        "F32x4" => Ok(quote!(Ty::F32x4)),
        "Buf<U8>" => Ok(quote!(Ty::Bytes)),
        _ => scalar(ty),
    }
}

/// Every identifier in `tokens`, groups included.
fn idents(tokens: Tokens, out: &mut Vec<String>) {
    for t in tokens {
        match t {
            TokenTree::Ident(i) => out.push(i.to_string()),
            TokenTree::Group(g) => idents(g.stream(), out),
            _ => {}
        }
    }
}

/// Generics the tree reads: consts, `Buf<F>`, types called through.
fn tree_generics(f: &ItemFn) -> Generics {
    let mut used = Vec::new();
    let block = &f.block;
    idents(quote!(#block), &mut used);
    for arg in &f.sig.inputs {
        let FnArg::Typed(p) = arg else { continue };
        if p.attrs.iter().any(|a| a.path().is_ident("form")) {
            continue;
        }
        if let Type::Path(path) = &*p.ty
            && let Some(last) = path.path.segments.last()
            && let Some(t) = first_ty(&last.arguments)
        {
            used.push(quote!(#t).to_string());
        }
    }
    let mut generics = f.sig.generics.clone();
    generics.params = generics
        .params
        .into_iter()
        .filter(|p| match p {
            GenericParam::Type(t) => used.contains(&t.ident.to_string()),
            _ => true,
        })
        .collect();
    generics
}

/// The struct a `Group` parameter names, through any borrow.
fn group_ty(ty: &Type) -> Option<&Type> {
    let ty = deref_ty(ty);
    let Type::Path(path) = ty else {
        return None;
    };
    let head = path.path.segments.last()?.ident.to_string();
    let known = ["Buf", "BufMut", "Shared", "Tiles", "Held"].contains(&head.as_str());
    (!known && scalar(ty).is_err()).then_some(ty)
}

/// `#[form(f)]` on a `Buf<Dyn>` reads build parameter `f`.
fn param(name: &Tokens, ty: &Type, picked: Option<Ident>) -> Res {
    // A reference passes group memory or tiles by name.
    let ty = deref_ty(ty);
    if let Type::Array(a) = ty {
        let ty = scalar(&a.elem)?;
        return Ok(quote!(Param::Array { name: #name.into(), ty: #ty }));
    }
    let Type::Path(path) = ty else {
        return Err(Error::new(ty.span(), "a buffer or scalar type"));
    };
    let last = path.path.segments.last().expect("a type");
    let head = last.ident.to_string();
    let form = |args: &PathArguments| match (&picked, args) {
        (Some(f), _) => Ok(quote!(#f)),
        (None, PathArguments::AngleBracketed(a)) => match first_ty(args) {
            Some(t) => Ok(quote!(<#t as tessera_codegen::prelude::Form>::FORM)),
            None => Err(Error::new(a.span(), "a form, like `F32` or `F`")),
        },
        _ => Err(Error::new(last.span(), "`Buf<F>` names its form")),
    };
    Ok(match head.as_str() {
        "Buf" | "BufMut" => {
            let form = form(&last.arguments)?;
            let writes = head == "BufMut";
            quote!(Param::Buffer { name: #name.into(), form: #form, writes: #writes })
        }
        "Shared" => {
            let Some(t) = first_ty(&last.arguments) else {
                return Err(Error::new(last.span(), "`Shared<T>` names its element"));
            };
            let ty = scalar(t)?;
            quote!(Param::Shared { name: #name.into(), ty: #ty })
        }
        "Tiles" | "Held" => quote!(Param::Tiles { name: #name.into() }),
        _ => {
            let ty = scalar(ty)?;
            quote!(Param::Scalar { name: #name.into(), ty: #ty })
        }
    })
}

/// The first type inside `<..>`.
pub(crate) fn first_ty(args: &PathArguments) -> Option<&Type> {
    let PathArguments::AngleBracketed(a) = args else {
        return None;
    };
    match a.args.first() {
        Some(GenericArgument::Type(t)) => Some(t),
        _ => None,
    }
}

/// The type under any borrow.
fn deref_ty(ty: &Type) -> &Type {
    match ty {
        Type::Reference(r) => &r.elem,
        other => other,
    }
}

/// A scalar type's `Ty`. `usize` indexes locals as `u32`.
fn scalar(ty: &Type) -> Res {
    let text = quote!(#ty).to_string();
    Ok(match text.as_str() {
        "f32" => quote!(Ty::F32),
        "u32" | "usize" => quote!(Ty::U32),
        "u64" => quote!(Ty::U64),
        "bool" => quote!(Ty::Bool),
        _ => return Err(Error::new(ty.span(), "f32, u32, u64 or bool")),
    })
}

fn assign_op(op: &BinOp) -> Option<Tokens> {
    Some(match op {
        BinOp::AddAssign(_) => quote!(BinOp::Add),
        BinOp::SubAssign(_) => quote!(BinOp::Sub),
        BinOp::MulAssign(_) => quote!(BinOp::Mul),
        BinOp::DivAssign(_) => quote!(BinOp::Div),
        _ => return None,
    })
}

fn bin_op(op: &BinOp) -> Option<Tokens> {
    Some(match op {
        BinOp::Add(_) => quote!(BinOp::Add),
        BinOp::Sub(_) => quote!(BinOp::Sub),
        BinOp::Mul(_) => quote!(BinOp::Mul),
        BinOp::Div(_) => quote!(BinOp::Div),
        BinOp::Rem(_) => quote!(BinOp::Rem),
        BinOp::Shl(_) => quote!(BinOp::Shl),
        BinOp::Shr(_) => quote!(BinOp::Shr),
        BinOp::BitAnd(_) => quote!(BinOp::BitAnd),
        BinOp::BitOr(_) => quote!(BinOp::BitOr),
        BinOp::BitXor(_) => quote!(BinOp::BitXor),
        BinOp::Lt(_) => quote!(BinOp::Lt),
        BinOp::Le(_) => quote!(BinOp::Le),
        BinOp::Gt(_) => quote!(BinOp::Gt),
        BinOp::Ge(_) => quote!(BinOp::Ge),
        BinOp::Eq(_) => quote!(BinOp::Eq),
        BinOp::Ne(_) => quote!(BinOp::Ne),
        BinOp::And(_) => quote!(BinOp::And),
        BinOp::Or(_) => quote!(BinOp::Or),
        _ => return None,
    })
}

/// Prelude names this macro reads as statements.
const STATEMENTS: &[&str] = &["barrier", "shared", "local", "atomic_shared", "tiles"];

/// Each intrinsic's kind, source name and variant.
macro_rules! sources {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:tt)*) $(-> $ret:ident)?
        { $($body:tt)* })*) => {
        const INTRINSICS: &[(&str, &str, &str)] =
            &[$((stringify!($kind), stringify!($($name)::+), stringify!($var))),*];
    };
}

mod intrinsic {
    include!("../../codegen/src/intrinsic.rs");
}

intrinsic::table!(sources);

/// The intrinsic a method or path call named `name` reaches.
fn variant(method: bool, name: &str) -> Option<Tokens> {
    for (kind, source, var) in INTRINSICS {
        if (*kind == "method") == method && source.replace(' ', "") == name {
            let var = format_ident!("{var}");
            return Some(quote!(Intrinsic::#var));
        }
    }
    None
}

/// A kernel or helper call: `f_inline(..)`, build values raw.
fn kernel_call(c: &ExprCall, ctx: &Ctx) -> Result<Option<Tokens>, Error> {
    let Expr::Path(p) = &*c.func else {
        return Ok(None);
    };
    let mut path = p.path.clone();
    let many = path.segments.len() > 1;
    let Some(last) = path.segments.last_mut() else {
        return Ok(None);
    };
    let name = last.ident.to_string();
    let whole = quote!(#p).to_string().replace(' ', "");
    if !many && STATEMENTS.contains(&name.as_str()) || variant(false, &whole).is_some() {
        return Ok(None);
    }
    last.ident = format_ident!("{name}_inline");
    let args: Vec<Tokens> = c
        .args
        .iter()
        .map(|a| match is_const(a, ctx) {
            true => Ok(quote!((#a))),
            false => expr(a, ctx),
        })
        .collect::<Result<_, _>>()?;
    Ok(Some(quote!(#path(#(#args),*))))
}

fn is_path(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::Path(p) if p.path.is_ident(name))
}

/// A name that is a value at build time.
fn is_const_name(name: &str, ctx: &Ctx) -> bool {
    let shouting = name
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && name.chars().any(|c| c.is_ascii_uppercase());
    shouting || ctx.build.iter().any(|b| b == name)
}

/// Whether every leaf is a literal or a build-time constant.
fn is_const(e: &Expr, ctx: &Ctx) -> bool {
    match e {
        Expr::Lit(ExprLit { lit, .. }) => {
            matches!(lit, Lit::Int(_) | Lit::Float(_) | Lit::Str(_))
        }
        // `T::N` is an associated constant.
        Expr::Path(p) => match p.path.get_ident() {
            Some(i) => is_const_name(&i.to_string(), ctx),
            None => p
                .path
                .segments
                .last()
                .is_some_and(|s| is_const_name(&s.ident.to_string(), ctx)),
        },
        Expr::Paren(p) => is_const(&p.expr, ctx),
        Expr::Cast(c) => is_const(&c.expr, ctx),
        Expr::Binary(b) => {
            bin_op(&b.op).is_some() && is_const(&b.left, ctx) && is_const(&b.right, ctx)
        }
        _ => false,
    }
}

fn expr(e: &Expr, ctx: &Ctx) -> Res {
    // A build value: `Pass` picks the literal from its type.
    if is_const(e, ctx) && !matches!(e, Expr::Lit(_)) {
        return Ok(quote!(tessera_codegen::inline::Pass::<Expr>::pass(#e)));
    }
    let sub = |e: &Expr| expr(e, ctx);
    match e {
        Expr::Lit(l) => match &l.lit {
            Lit::Int(n) => {
                let n: i64 = n.base10_parse()?;
                Ok(quote!(Expr::Int(#n)))
            }
            Lit::Float(x) => {
                let x: f64 = x.base10_parse()?;
                Ok(quote!(Expr::Float(#x)))
            }
            Lit::Bool(b) => {
                let b = b.value;
                Ok(quote!(Expr::Bool(#b)))
            }
            other => Err(Error::new(other.span(), "a number or bool")),
        },
        Expr::Path(p) => match p.path.get_ident() {
            Some(i) => {
                let name = i.to_string();
                Ok(quote!(Expr::var(#name)))
            }
            None => Err(Error::new(p.span(), "a plain name")),
        },
        Expr::If(i) => {
            let only = |b: &syn::Block| match b.stmts.as_slice() {
                [Stmt::Expr(e, None)] => expr(e, ctx),
                _ => Err(Error::new(b.span(), "one value in each branch")),
            };
            let Some((_, els)) = &i.else_branch else {
                return Err(Error::new(i.span(), "a value needs both branches"));
            };
            let b = match &**els {
                Expr::Block(els) => only(&els.block)?,
                nested @ Expr::If(_) => expr(nested, ctx)?,
                other => return Err(Error::new(other.span(), "a block or `else if`")),
            };
            let (c, a) = (expr(&i.cond, ctx)?, only(&i.then_branch)?);
            Ok(quote!(Expr::select(#c, #a, #b)))
        }
        Expr::Paren(p) => sub(&p.expr),
        // A borrow passes group memory by name.
        Expr::Reference(r) => sub(&r.expr),
        Expr::Unary(u) => {
            let a = sub(&u.expr)?;
            match u.op {
                UnOp::Neg(_) => Ok(quote!(Expr::Neg(Box::new(#a)))),
                UnOp::Not(_) => Ok(quote!(Expr::Not(Box::new(#a)))),
                _ => Err(Error::new(u.span(), "no references in a kernel")),
            }
        }
        Expr::Binary(b) => {
            let Some(op) = bin_op(&b.op) else {
                return Err(Error::new(b.op.span(), "an assignment is a statement"));
            };
            let (l, r) = (sub(&b.left)?, sub(&b.right)?);
            Ok(quote!(Expr::bin(#op, #l, #r)))
        }
        Expr::Cast(c) => {
            let (ty, a) = (scalar(&c.ty)?, sub(&c.expr)?);
            Ok(quote!(Expr::cast(#ty, #a)))
        }
        Expr::Index(i) => {
            let (base, at) = (sub(&i.expr)?, sub(&i.index)?);
            Ok(quote!(Expr::index(#base, #at)))
        }
        Expr::Field(f) => {
            let member = &f.member;
            let name = quote!(#member).to_string();
            let lane = LANES.contains(&name.as_str()).then(|| name.chars().next());
            let Some(Some(h)) = lane else {
                // A group's field is a parameter of its own.
                let name = named(e)?;
                return Ok(quote!(Expr::var(#name)));
            };
            let base = sub(&f.base)?;
            Ok(quote!(Expr::field(#base, #h)))
        }
        Expr::MethodCall(m) => {
            let recv = sub(&m.receiver)?;
            let args: Vec<Tokens> = m.args.iter().map(sub).collect::<Result<_, _>>()?;
            let method = m.method.to_string();
            // Kernel `u32` products wrap.
            if let ("wrapping_mul", [by]) = (method.as_str(), args.as_slice()) {
                return Ok(quote!(Expr::bin(BinOp::Mul, #recv, #by)));
            }
            let Some(intrinsic) = variant(true, &method) else {
                return Err(Error::new(m.method.span(), format!("no `{method}` here")));
            };
            Ok(quote!(Expr::Call(#intrinsic, vec![#recv, #(#args),*])))
        }
        Expr::Call(c) => {
            // Calls in the arguments come first, so `inl` is free.
            if let Some(call) = kernel_call(c, ctx)? {
                return Ok(quote!({
                    let call = #call;
                    inl.expr(&mut pre, call)
                }));
            }
            let Expr::Path(p) = &*c.func else {
                return Err(Error::new(c.func.span(), "call a prelude function"));
            };
            let name = quote!(#p).to_string().replace(' ', "");
            let args: Vec<Tokens> = c.args.iter().map(sub).collect::<Result<_, _>>()?;
            let Some(intrinsic) = variant(false, &name) else {
                return Err(Error::new(p.span(), format!("no `{name}` here")));
            };
            Ok(quote!(Expr::Call(#intrinsic, vec![#(#args),*])))
        }
        other => Err(Error::new(other.span(), "not part of a kernel body")),
    }
}
