//! `#[kernel]` keeps the function and adds one building its tree.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{
    BinOp, Error, Expr, ExprLit, FnArg, GenericArgument, ItemFn, Lit, Pat, PathArguments, Stmt,
    Type, UnOp, parse_macro_input,
};

type Res = Result<Tokens, Error>;

#[proc_macro_attribute]
pub fn kernel(_: TokenStream, item: TokenStream) -> TokenStream {
    let f = parse_macro_input!(item as ItemFn);
    match tree(&f) {
        Ok(tree) => quote! {
            #[allow(dead_code, unused_mut, clippy::all)]
            #f
            #tree
        }
        .into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The sibling function that builds the body.
fn tree(f: &ItemFn) -> Res {
    let name = &f.sig.ident;
    let ir_name = format_ident!("{name}_ir");
    let vis = &f.vis;
    let (impl_generics, _, where_clause) = f.sig.generics.split_for_impl();
    let entry = name.to_string();
    let params: Vec<Tokens> = f.sig.inputs.iter().map(param).collect::<Result<_, _>>()?;
    let stmts = block(&f.block.stmts)?;
    Ok(quote! {
        #vis fn #ir_name #impl_generics () -> codegen::ir::Body #where_clause {
            #[allow(unused_imports)]
            use codegen::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt, Ty};
            Body {
                entry: #entry.into(),
                params: vec![#(#params),*],
                stmts: #stmts,
            }
        }
    })
}

fn param(arg: &FnArg) -> Res {
    let FnArg::Typed(p) = arg else {
        return Err(Error::new(arg.span(), "a kernel takes no self"));
    };
    let Pat::Ident(ident) = &*p.pat else {
        return Err(Error::new(p.pat.span(), "name each parameter"));
    };
    let name = ident.ident.to_string();
    let Type::Path(ty) = &*p.ty else {
        return Err(Error::new(p.ty.span(), "a buffer or scalar type"));
    };
    let last = ty.path.segments.last().expect("a type");
    let head = last.ident.to_string();
    let form = |args: &PathArguments| match args {
        PathArguments::AngleBracketed(a) => match a.args.first() {
            Some(GenericArgument::Type(t)) => Ok(quote!(<#t as codegen::prelude::Form>::DTYPE)),
            _ => Err(Error::new(a.span(), "a form, like `F32` or `F`")),
        },
        _ => Err(Error::new(last.span(), "`Buf<F>` names its form")),
    };
    Ok(match head.as_str() {
        "Buf" => {
            let dtype = form(&last.arguments)?;
            quote!(Param::Buffer { name: #name.into(), dtype: #dtype, writes: false })
        }
        "BufMut" => {
            let dtype = form(&last.arguments)?;
            quote!(Param::Buffer { name: #name.into(), dtype: #dtype, writes: true })
        }
        _ => {
            let ty = scalar(&p.ty)?;
            quote!(Param::Scalar { name: #name.into(), ty: #ty })
        }
    })
}

/// A scalar type's `Ty`.
fn scalar(ty: &Type) -> Res {
    let text = quote!(#ty).to_string();
    Ok(match text.as_str() {
        "f32" => quote!(Ty::F32),
        "u32" => quote!(Ty::U32),
        "u64" => quote!(Ty::U64),
        "bool" => quote!(Ty::Bool),
        _ => return Err(Error::new(ty.span(), "f32, u32, u64 or bool")),
    })
}

fn block(stmts: &[Stmt]) -> Res {
    let stmts: Vec<Tokens> = stmts.iter().map(stmt).collect::<Result<_, _>>()?;
    Ok(quote!(vec![#(#stmts),*]))
}

fn stmt(s: &Stmt) -> Res {
    match s {
        Stmt::Local(local) => {
            let name = match &local.pat {
                Pat::Ident(i) => i.ident.to_string(),
                Pat::Type(t) => match &*t.pat {
                    Pat::Ident(i) => i.ident.to_string(),
                    _ => return Err(Error::new(t.span(), "name the binding")),
                },
                other => return Err(Error::new(other.span(), "name the binding")),
            };
            let Some(init) = &local.init else {
                return Err(Error::new(local.span(), "give the binding a value"));
            };
            declare(&name, &init.expr)
        }
        Stmt::Expr(e, _) => statement(e),
        other => Err(Error::new(other.span(), "not part of a kernel body")),
    }
}

/// A `let`: group memory, a local array, or a value.
fn declare(name: &str, init: &Expr) -> Res {
    if let Expr::Call(call) = init
        && let Expr::Path(p) = &*call.func
        && p.path.segments.len() == 2
        && p.path.segments[0].ident == "Shared"
    {
        let PathArguments::AngleBracketed(a) = &p.path.segments[0].arguments else {
            return Err(Error::new(p.span(), "`Shared::<T, N>::new()`"));
        };
        let mut args = a.args.iter();
        let (Some(GenericArgument::Type(t)), Some(GenericArgument::Const(n))) =
            (args.next(), args.next())
        else {
            return Err(Error::new(a.span(), "`Shared::<T, N>::new()`"));
        };
        let ty = scalar(t)?;
        return Ok(quote!(Stmt::Shared { name: #name.into(), ty: #ty, len: (#n) as usize }));
    }
    if let Expr::Repeat(r) = init {
        let ty = match &*r.expr {
            Expr::Lit(ExprLit {
                lit: Lit::Float(_), ..
            }) => quote!(Ty::F32),
            _ => quote!(Ty::U32),
        };
        let n = &r.len;
        return Ok(quote!(Stmt::Array { name: #name.into(), ty: #ty, len: (#n) as usize }));
    }
    let init = expr(init)?;
    Ok(quote!(Stmt::Let { name: #name.into(), init: #init }))
}

fn statement(e: &Expr) -> Res {
    match e {
        Expr::Assign(a) => {
            let (target, value) = (expr(&a.left)?, expr(&a.right)?);
            Ok(quote!(Stmt::Assign { target: #target, op: None, value: #value }))
        }
        Expr::Binary(b) if assign_op(&b.op).is_some() => {
            let op = assign_op(&b.op).expect("checked");
            let (target, value) = (expr(&b.left)?, expr(&b.right)?);
            Ok(quote!(Stmt::Assign { target: #target, op: Some(#op), value: #value }))
        }
        Expr::ForLoop(f) => {
            let Pat::Ident(var) = &*f.pat else {
                return Err(Error::new(f.pat.span(), "name the loop variable"));
            };
            let var = var.ident.to_string();
            let (from, to, step, unroll) = range(&f.expr)?;
            let body = block(&f.body.stmts)?;
            Ok(quote!(Stmt::For {
                var: #var.into(),
                from: #from,
                to: #to,
                step: #step,
                unroll: #unroll,
                body: #body,
            }))
        }
        Expr::If(i) => {
            let cond = expr(&i.cond)?;
            let then = block(&i.then_branch.stmts)?;
            let els = match &i.else_branch {
                None => quote!(vec![]),
                Some((_, e)) => match &**e {
                    Expr::Block(b) => block(&b.block.stmts)?,
                    nested @ Expr::If(_) => {
                        let s = statement(nested)?;
                        quote!(vec![#s])
                    }
                    other => return Err(Error::new(other.span(), "a block or `else if`")),
                },
            };
            Ok(quote!(Stmt::If { cond: #cond, then: #then, els: #els }))
        }
        Expr::Return(r) if r.expr.is_none() => Ok(quote!(Stmt::Return)),
        Expr::Call(c) if is_path(&c.func, "barrier") => Ok(quote!(Stmt::Barrier)),
        Expr::Block(b) => {
            // A bare block scopes nothing the tree needs.
            let stmts = block(&b.block.stmts)?;
            Ok(quote!(Stmt::If { cond: Expr::Bool(true), then: #stmts, els: vec![] }))
        }
        other => {
            let e = expr(other)?;
            Ok(quote!(Stmt::Expr(#e)))
        }
    }
}

/// `a..b` under any `.by(s)` and `.unroll(n)`.
fn range(e: &Expr) -> Result<(Tokens, Tokens, Tokens, Tokens), Error> {
    let (mut step, mut unroll) = (quote!(Expr::Int(1)), quote!(0usize));
    let mut e = e;
    loop {
        match e {
            Expr::Paren(p) => e = &p.expr,
            Expr::MethodCall(m) if m.method == "by" => {
                step = expr(&m.args[0])?;
                e = &m.receiver;
            }
            Expr::MethodCall(m) if m.method == "unroll" => {
                let n = &m.args[0];
                unroll = quote!((#n) as usize);
                e = &m.receiver;
            }
            Expr::Range(r) => {
                let (Some(from), Some(to)) = (&r.start, &r.end) else {
                    return Err(Error::new(r.span(), "a range with both ends"));
                };
                return Ok((expr(from)?, expr(to)?, step, unroll));
            }
            other => return Err(Error::new(other.span(), "`for v in (a..b).by(s)`")),
        }
    }
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

fn is_path(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::Path(p) if p.path.is_ident(name))
}

/// A name that is a value at build time.
fn is_const_name(name: &str) -> bool {
    name.chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && name.chars().any(|c| c.is_ascii_uppercase())
}

/// Whether every leaf is a literal or a build-time constant.
fn is_const(e: &Expr) -> bool {
    match e {
        Expr::Lit(ExprLit {
            lit: Lit::Int(_), ..
        }) => true,
        Expr::Path(p) => p
            .path
            .get_ident()
            .is_some_and(|i| is_const_name(&i.to_string())),
        Expr::Paren(p) => is_const(&p.expr),
        Expr::Cast(c) => is_const(&c.expr),
        Expr::Binary(b) => bin_op(&b.op).is_some() && is_const(&b.left) && is_const(&b.right),
        _ => false,
    }
}

fn expr(e: &Expr) -> Res {
    if is_const(e) && !matches!(e, Expr::Lit(_)) {
        return Ok(quote!(Expr::Int((#e) as i64)));
    }
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
        Expr::Paren(p) => expr(&p.expr),
        Expr::Unary(u) => {
            let a = expr(&u.expr)?;
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
            let (l, r) = (expr(&b.left)?, expr(&b.right)?);
            Ok(quote!(Expr::bin(#op, #l, #r)))
        }
        Expr::Cast(c) => {
            let (ty, a) = (scalar(&c.ty)?, expr(&c.expr)?);
            Ok(quote!(Expr::cast(#ty, #a)))
        }
        Expr::Index(i) => {
            let (base, at) = (expr(&i.expr)?, expr(&i.index)?);
            Ok(quote!(Expr::index(#base, #at)))
        }
        Expr::Field(f) => {
            let (base, member) = (expr(&f.base)?, &f.member);
            let name = quote!(#member).to_string();
            let Some(h) = name.chars().next().filter(|_| name.len() == 1) else {
                return Err(Error::new(f.member.span(), "`.x`, `.y`, `.z` or `.w`"));
            };
            Ok(quote!(Expr::field(#base, #h)))
        }
        Expr::MethodCall(m) => {
            let recv = expr(&m.receiver)?;
            let args: Vec<Tokens> = m.args.iter().map(expr).collect::<Result<_, _>>()?;
            let intrinsic = match m.method.to_string().as_str() {
                "load4" => quote!(Intrinsic::Load4),
                "store4" => quote!(Intrinsic::Store4),
                other => return Err(Error::new(m.method.span(), format!("no `{other}` here"))),
            };
            Ok(quote!(Expr::Call(#intrinsic, vec![#recv, #(#args),*])))
        }
        Expr::Call(c) => {
            let Expr::Path(p) = &*c.func else {
                return Err(Error::new(c.func.span(), "call a prelude function"));
            };
            let name = quote!(#p).to_string().replace(' ', "");
            let args: Vec<Tokens> = c.args.iter().map(expr).collect::<Result<_, _>>()?;
            let intrinsic = match name.as_str() {
                "gid" => quote!(Intrinsic::Gid),
                "lid" => quote!(Intrinsic::Lid),
                "lane" => quote!(Intrinsic::Lane),
                "shfl_xor" => quote!(Intrinsic::ShflXor),
                "sum32" => quote!(Intrinsic::Sum32),
                "rsqrt" => quote!(Intrinsic::Rsqrt),
                "exp" => quote!(Intrinsic::Exp),
                "tanh" => quote!(Intrinsic::Tanh),
                "F32x4::zero" => quote!(Intrinsic::Zero4),
                other => return Err(Error::new(p.span(), format!("no `{other}` here"))),
            };
            Ok(quote!(Expr::Call(#intrinsic, vec![#(#args),*])))
        }
        other => Err(Error::new(other.span(), "not part of a kernel body")),
    }
}
