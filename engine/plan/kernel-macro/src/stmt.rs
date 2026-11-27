//! Statement lowering: each kernel statement becomes a `Stmt` builder.

use super::*;

pub(super) fn block(stmts: &[Stmt], ctx: &Ctx) -> Res {
    let stmts: Vec<Tokens> = stmts
        .iter()
        .map(|s| stmt(s, ctx))
        .collect::<Result<_, _>>()?;
    Ok(quote!({
        let mut b: Vec<Stmt> = Vec::new();
        #(b.extend(#stmts);)*
        b
    }))
}

/// A statement's list: any inlined calls, then itself.
fn stmt(s: &Stmt, ctx: &Ctx) -> Res {
    if let Stmt::Expr(Expr::Call(c), _) = s
        && let Some(call) = kernel_call(c, ctx)?
    {
        return Ok(with_pre(quote! {
            let call = #call;
            inl.stmt(&mut pre, call);
        }));
    }
    if let Stmt::Local(local) = s
        && let Some(init) = &local.init
        && let Expr::Struct(lit) = &*init.expr
    {
        let name = binding(&local.pat)?;
        return build_group(&quote!(#name), lit, ctx);
    }
    // A bound call gives a value, or a group's fields.
    if let Stmt::Local(local) = s
        && let Some(init) = &local.init
        && let Expr::Call(c) = &*init.expr
        && let Some(call) = kernel_call(c, ctx)?
    {
        let name = binding(&local.pat)?;
        return Ok(with_pre(quote! {
            let call = #call;
            if let Some(init) = inl.bind(&mut pre, call, #name) {
                pre.push(Stmt::Let { name: #name.into(), init });
            }
        }));
    }
    let one = one(s, ctx)?;
    Ok(with_pre(quote! {
        let s = #one;
        pre.push(s);
    }))
}

/// `body` fills `pre`, the list this statement yields.
fn with_pre(body: Tokens) -> Tokens {
    quote!({
        let mut pre: Vec<Stmt> = Vec::new();
        #body
        pre
    })
}

/// Evaluates all method call arguments in kernel expression context.
fn call_args(m: &syn::ExprMethodCall, ctx: &Ctx) -> Result<Vec<Tokens>, Error> {
    m.args.iter().map(|a| expr(a, ctx)).collect()
}

/// A name, or `g.field` of group `g` as `g__field`.
pub(super) fn named(e: &Expr) -> Result<String, Error> {
    match e {
        Expr::Path(p) => match p.path.get_ident() {
            Some(i) => Ok(i.to_string()),
            None => Err(Error::new(p.span(), "a plain name")),
        },
        Expr::Field(f) => {
            let member = &f.member;
            Ok(format!("{}__{}", named(&f.base)?, quote!(#member)))
        }
        Expr::Reference(r) => named(&r.expr),
        Expr::Paren(p) => named(&p.expr),
        other => Err(Error::new(other.span(), "a name")),
    }
}

fn binding(pat: &Pat) -> Result<String, Error> {
    match pat {
        Pat::Ident(i) => Ok(i.ident.to_string()),
        Pat::Type(t) => binding(&t.pat),
        other => Err(Error::new(other.span(), "name the binding")),
    }
}

/// A group built in place: a `let` for each field.
pub(super) fn build_group(name: &Tokens, lit: &syn::ExprStruct, ctx: &Ctx) -> Res {
    if let Some(rest) = &lit.rest {
        return Err(Error::new(rest.span(), "give every field"));
    }
    let mut lets = Vec::new();
    for f in &lit.fields {
        let member = &f.member;
        let field = quote!(#member).to_string();
        let named = quote!(tessera_codegen::inline::member(#name, #field));
        lets.push(declare(&named, &f.expr, ctx)?);
    }
    Ok(quote!(vec![#(#lets),*]))
}

fn one(s: &Stmt, ctx: &Ctx) -> Res {
    match s {
        Stmt::Local(local) => {
            let name = binding(&local.pat)?;
            let Some(init) = &local.init else {
                return Err(Error::new(local.span(), "give the binding a value"));
            };
            declare(&quote!(#name), &init.expr, ctx)
        }
        Stmt::Expr(e, _) => statement(e, ctx),
        other => Err(Error::new(other.span(), "not part of a kernel body")),
    }
}

/// Fixed-size statements: call name, variant and size fields.
const SIZED: &[(&str, &str, &[&str])] = &[
    ("atomic_shared", "AtomicShared", &["len"]),
    ("tiles", "TileAcc", &["rows", "cols"]),
    ("held", "TileHeld", &["rows", "depth"]),
];

/// A `let`: group memory, tiles, an array, or a value.
fn declare(name: &Tokens, init: &Expr, ctx: &Ctx) -> Res {
    if let Expr::Call(call) = init
        && let Expr::Path(p) = &*call.func
        && let Some(last) = p.path.segments.last()
        && matches!(last.ident.to_string().as_str(), "shared" | "local")
    {
        let kind = last.ident.to_string();
        let Some(t) = first_ty(&last.arguments) else {
            return Err(Error::new(p.span(), format!("`{kind}::<T>(len)`")));
        };
        let (ty, len) = (scalar(t)?, &call.args[0]);
        let kind_id = if kind == "local" {
            quote!(Stmt::Array)
        } else {
            quote!(Stmt::Shared)
        };
        return Ok(quote!(#kind_id { name: #name.into(), ty: #ty, len: (#len) as usize }));
    }
    if let Expr::Call(call) = init {
        for (func, variant, fields) in SIZED {
            if !is_path(&call.func, func) {
                continue;
            }
            let variant = format_ident!("{variant}");
            let mut sizes = Vec::new();
            for (field, arg) in fields.iter().zip(&call.args) {
                let field = format_ident!("{field}");
                sizes.push(quote!(#field: (#arg) as usize));
            }
            return Ok(quote!(Stmt::#variant { name: #name.into(), #(#sizes),* }));
        }
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
    let init = expr(init, ctx)?;
    Ok(quote!(Stmt::Let { name: #name.into(), init: #init }))
}

/// `acc.block(..)` or `acc.parts(..)`: one `TileBlock`.
fn tile_block(m: &syn::ExprMethodCall, ctx: &Ctx) -> Res {
    let (acc, args) = (named(&m.receiver)?, call_args(m, ctx)?);
    let (a, a_at, b, b_at, stride, apart, parts, depth);
    if m.method == "parts" {
        let [x, x_at, x_part, _, y, y_at, y_stride, _] = args.as_slice() else {
            return Err(Error::new(
                m.span(),
                "`acc.parts(a, a_at, apart, parts, b, b_at, stride, depth)`",
            ));
        };
        (a, a_at, b, b_at, stride, apart) = (x, x_at, y, y_at, y_stride, x_part.clone());
        let (n, d) = (&m.args[3], &m.args[7]);
        (parts, depth) = (quote!((#n) as usize), quote!((#d) as usize));
    } else {
        let [x, x_at, y, y_at, y_stride, _] = args.as_slice() else {
            return Err(Error::new(
                m.span(),
                "`acc.block(a, a_at, b, b_at, stride, depth)`",
            ));
        };
        (a, a_at, b, b_at, stride, apart) = (x, x_at, y, y_at, y_stride, quote!(Expr::Int(0)));
        let d = &m.args[5];
        (parts, depth) = (quote!(0), quote!((#d) as usize));
    }
    Ok(quote!(Stmt::TileBlock {
        acc: #acc.into(),
        a: #a,
        a_at: #a_at,
        b: #b,
        b_at: #b_at,
        stride: #stride,
        depth: #depth,
        parts: #parts,
        apart: #apart,
    }))
}

fn statement(e: &Expr, ctx: &Ctx) -> Res {
    match e {
        Expr::Assign(a) => {
            let (target, value) = (expr(&a.left, ctx)?, expr(&a.right, ctx)?);
            Ok(quote!(Stmt::Assign { target: #target, op: None, value: #value }))
        }
        Expr::Binary(b) if assign_op(&b.op).is_some() => {
            let op = assign_op(&b.op).expect("checked");
            let (target, value) = (expr(&b.left, ctx)?, expr(&b.right, ctx)?);
            Ok(quote!(Stmt::Assign { target: #target, op: Some(#op), value: #value }))
        }
        Expr::ForLoop(f) => {
            let Pat::Ident(var) = &*f.pat else {
                return Err(Error::new(f.pat.span(), "name the loop variable"));
            };
            let var = var.ident.to_string();
            let (from, to, step, unroll) = range(&f.expr, ctx)?;
            let body = block(&f.body.stmts, ctx)?;
            Ok(quote!(Stmt::For {
                var: #var.into(),
                from: #from,
                to: #to,
                step: #step,
                unroll: #unroll,
                body: #body,
            }))
        }
        Expr::MethodCall(m) if m.method == "block" || m.method == "parts" => tile_block(m, ctx),
        Expr::MethodCall(m) if m.method == "hold" => {
            let (held, args) = (named(&m.receiver)?, call_args(m, ctx)?);
            let [a, a_at, stride] = args.as_slice() else {
                return Err(Error::new(m.span(), "`held.hold(a, a_at, stride)`"));
            };
            Ok(quote!(Stmt::TileHold {
                held: #held.into(),
                a: #a,
                a_at: #a_at,
                stride: #stride,
            }))
        }
        Expr::MethodCall(m) if m.method == "zero" => {
            let acc = named(&m.receiver)?;
            Ok(quote!(Stmt::TileZero { acc: #acc.into() }))
        }
        Expr::MethodCall(m) if m.method == "add" => {
            let (name, at, val) = (
                named(&m.receiver)?,
                expr(&m.args[0], ctx)?,
                expr(&m.args[1], ctx)?,
            );
            Ok(quote!(Stmt::AtomicAdd { name: #name.into(), at: #at, value: #val }))
        }
        Expr::MethodCall(m) if m.method == "scale" => {
            let acc = named(&m.receiver)?;
            let Some(Expr::Closure(f)) = m.args.first() else {
                return Err(Error::new(m.span(), "`acc.scale(|r| ..)`"));
            };
            let by = expr(&f.body, ctx)?;
            Ok(quote!(Stmt::TileScale { acc: #acc.into(), by: #by }))
        }
        Expr::MethodCall(m) if m.method == "each" => {
            let acc = named(&m.receiver)?;
            let Some(Expr::Closure(f)) = m.args.first() else {
                return Err(Error::new(m.span(), "`acc.each(|r, c, v| { .. })`"));
            };
            let names: Vec<String> = f
                .inputs
                .iter()
                .map(|p| match p {
                    Pat::Ident(i) => Ok(i.ident.to_string()),
                    other => Err(Error::new(other.span(), "`|r, c, v|`")),
                })
                .collect::<Result<_, _>>()?;
            if names != ["r", "c", "v"] {
                return Err(Error::new(f.span(), "the closure takes `|r, c, v|`"));
            }
            let body = match &*f.body {
                Expr::Block(b) => block(&b.block.stmts, ctx)?,
                other => return Err(Error::new(other.span(), "a block body")),
            };
            Ok(quote!(Stmt::TileEach { acc: #acc.into(), body: #body }))
        }
        Expr::If(i) if is_const(&i.cond, ctx) => {
            // Settled at build time: the printer keeps one branch.
            let cond = &i.cond;
            let then = block(&i.then_branch.stmts, ctx)?;
            let els = else_branch(i, false, ctx)?;
            Ok(quote!(Stmt::If { cond: Expr::Bool(#cond), then: #then, els: #els }))
        }
        Expr::If(i) => {
            let cond = expr(&i.cond, ctx)?;
            let then = block(&i.then_branch.stmts, ctx)?;
            let els = else_branch(i, true, ctx)?;
            Ok(quote!(Stmt::If { cond: #cond, then: #then, els: #els }))
        }
        Expr::Return(r) if r.expr.is_none() => Ok(quote!(Stmt::Return)),
        Expr::Call(c) if is_path(&c.func, "barrier") => Ok(quote!(Stmt::Barrier)),
        Expr::Block(b) => {
            // A bare block scopes nothing the tree needs.
            let stmts = block(&b.block.stmts, ctx)?;
            Ok(quote!(Stmt::If { cond: Expr::Bool(true), then: #stmts, els: vec![] }))
        }
        other => {
            let e = expr(other, ctx)?;
            Ok(quote!(Stmt::Expr(#e)))
        }
    }
}

/// The `else` list: a block, or `else if` with `chain`.
fn else_branch(i: &syn::ExprIf, chain: bool, ctx: &Ctx) -> Res {
    let Some((_, e)) = &i.else_branch else {
        return Ok(quote!(vec![]));
    };
    match &**e {
        Expr::Block(b) => block(&b.block.stmts, ctx),
        nested @ Expr::If(_) if chain => stmt(&Stmt::Expr(nested.clone(), None), ctx),
        other => {
            let want = if chain {
                "a block or `else if`"
            } else {
                "a block"
            };
            Err(Error::new(other.span(), want))
        }
    }
}

/// `a..b` under any `.by(s)` and `.unroll(n)`.
fn range(e: &Expr, ctx: &Ctx) -> Result<(Tokens, Tokens, Tokens, Tokens), Error> {
    let (mut step, mut unroll) = (quote!(Expr::Int(1)), quote!(0usize));
    let mut e = e;
    loop {
        match e {
            Expr::Paren(p) => e = &p.expr,
            Expr::MethodCall(m) if m.method == "by" => {
                step = expr(&m.args[0], ctx)?;
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
                return Ok((expr(from, ctx)?, expr(to, ctx)?, step, unroll));
            }
            other => return Err(Error::new(other.span(), "`for v in (a..b).by(s)`")),
        }
    }
}
