//! Prints a kernel body as text, in C or WGSL.

use crate::LANES;
use crate::form::{Load, U8_FORM};
use crate::ir::{Expr, Intrinsic, Param, Stmt, Ty, element};
use std::collections::HashMap;

/// Every name in scope and what it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Value(Ty),
    /// Four elements of this form, loaded at once.
    Wide(&'static dyn Load),
    /// A buffer parameter of this form.
    Buffer(&'static dyn Load),
    /// Group memory, or a local array, of this element type.
    Array(Ty),
    /// Tile sums over this many rows and columns.
    Tiles(usize, usize),
    /// A held slab of A: rows and depth.
    Held(usize, usize),
}

/// Prints a body as the dialect-neutral text, or as WGSL.
pub(crate) struct Printer<'a> {
    /// Each name's kind and the name the text gives it.
    scope: Vec<HashMap<String, (Kind, String)>>,
    pub(crate) lines: Vec<String>,
    indent: usize,
    pub(crate) lane: bool,
    pub(crate) params: &'a [Param],
    pub(crate) wgsl: bool,
    /// Set when WGSL has no spelling for a part.
    pub(crate) refused: Option<String>,
    /// Intrinsics called, each once.
    pub(crate) used: Vec<Intrinsic>,
    /// WGSL declarations outside the entry: group memory.
    pub(crate) module: Vec<String>,
}

impl<'a> Printer<'a> {
    pub(crate) fn new(params: &'a [Param], wgsl: bool) -> Self {
        Printer {
            scope: vec![HashMap::new()],
            lines: Vec::new(),
            indent: 1,
            lane: false,
            params,
            wgsl,
            refused: None,
            used: Vec::new(),
            module: Vec::new(),
        }
    }

    /// Keeps the first refusal.
    pub(crate) fn refuse(&mut self, what: String) {
        self.refused.get_or_insert(what);
    }

    pub(crate) fn uses(&mut self, i: Intrinsic) {
        if !self.used.contains(&i) {
            self.used.push(i);
        }
    }

    /// A type's name in the text.
    pub(crate) fn ty_name(&mut self, ty: Ty) -> &'static str {
        if !self.wgsl {
            return ty.c();
        }
        if ty == Ty::Bytes {
            self.refuse("byte pointers".into());
        }
        crate::wgsl::ty(ty)
    }
    fn kind(&self, name: &str) -> Kind {
        for scope in self.scope.iter().rev() {
            if let Some((k, _)) = scope.get(name) {
                return *k;
            }
        }
        panic!("`{name}` is not in scope")
    }

    /// The name the text gives `name`.
    pub(crate) fn c_name(&self, name: &str) -> String {
        let found = self.scope.iter().rev().find_map(|s| s.get(name));
        found.map_or_else(|| name.into(), |(_, c)| c.clone())
    }

    pub(crate) fn bind(&mut self, name: &str, kind: Kind) {
        let text = self.safe(name);
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), (kind, text));
    }

    /// `name`, with a suffix where WGSL reserves it.
    fn safe(&self, name: &str) -> String {
        match self.wgsl && crate::wgsl::reserved(name) {
            true => format!("{name}_"),
            false => name.into(),
        }
    }

    /// Binds a variable. A shadow gets a fresh C name.
    fn declare(&mut self, name: &str, kind: Kind) -> String {
        let taken = |p: &Self, c: &str| {
            let mut all = p.scope.iter().flat_map(|s| s.iter());
            all.any(|(n, (_, rendered))| n == c || rendered == c)
        };
        let mut c = self.safe(name);
        let mut n = 0;
        while taken(self, &c) {
            n += 1;
            c = format!("{name}__{n}");
        }
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), (kind, c.clone()));
        c
    }

    pub(crate) fn line(&mut self, text: String) {
        self.lines
            .push(format!("{}{text}", "    ".repeat(self.indent)));
    }

    /// An expression's type, from the kinds of its names.
    fn ty(&self, e: &Expr) -> Ty {
        match e {
            Expr::Int(_) => Ty::U32,
            Expr::Float(_) => Ty::F32,
            Expr::Bool(_) => Ty::Bool,
            Expr::Var(name) => match self.kind(name) {
                Kind::Value(ty) => ty,
                Kind::Wide(_) => Ty::F32x4,
                Kind::Buffer(form) if form.ty() == "u8" => Ty::Bytes,
                other => panic!("`{name}` is {other:?}, not a value"),
            },
            Expr::Neg(a) | Expr::Not(a) => self.ty(a),
            Expr::Bin(op, a, b) => match op.compares() {
                true => Ty::Bool,
                false => wider(self.ty(a), self.ty(b)),
            },
            Expr::Cast(ty, _) => *ty,
            Expr::Index(base, _) => match &**base {
                Expr::Var(name) => match self.kind(name) {
                    Kind::Array(ty) => ty,
                    Kind::Buffer(form) => element(form),
                    _ => Ty::F32,
                },
                _ => Ty::F32,
            },
            Expr::Field(..) => Ty::F32,
            Expr::Select(c, a, b) => match **c {
                Expr::Bool(false) => self.ty(b),
                _ => self.ty(a),
            },
            Expr::Fn(_, ty, _) => *ty,
            Expr::Call(i, _) => i.ty(),
        }
    }

    /// The form of a buffer parameter.
    pub(crate) fn buffer(&self, e: &Expr) -> &'static dyn Load {
        match e {
            Expr::Var(name) => match self.kind(name) {
                Kind::Buffer(form) => form,
                other => panic!("`{name}` is {other:?}, not a buffer"),
            },
            _ => panic!("a buffer is named directly"),
        }
    }

    /// The form behind a four-wide value from a buffer.
    fn wide(&self, e: &Expr) -> Option<&'static dyn Load> {
        match e {
            Expr::Var(name) => match self.kind(name) {
                Kind::Wide(form) => Some(form),
                _ => None,
            },
            Expr::Call(Intrinsic::Load4, args) => Some(self.buffer(&args[0])),
            _ => None,
        }
    }

    pub(crate) fn expr(&mut self, e: &Expr) -> String {
        match e {
            // Past i32, a bare literal widens the arithmetic.

            // In WGSL, a bare literal can be signed.
            Expr::Int(n) if *n > i64::from(i32::MAX) || self.wgsl => format!("{n}u"),
            Expr::Int(n) => n.to_string(),
            Expr::Float(x) => {
                // WGSL refuses an infinite constant, so `f_bits` makes it.
                if x.is_infinite() {
                    self.uses(Intrinsic::FromBits);
                }
                float(*x)
            }
            Expr::Bool(b) => b.to_string(),
            Expr::Var(name) => self.c_name(name),
            Expr::Neg(a) => format!("(-{})", self.expr(a)),
            // On an integer, `!` flips every bit.
            Expr::Not(a) => match self.ty(a) {
                Ty::Bool => format!("(!{})", self.expr(a)),
                _ => format!("(~{})", self.expr(a)),
            },
            Expr::Bin(op, a, b) => {
                let (a, b) = (self.expr(a), self.expr(b));
                format!("({a} {} {b})", op.c())
            }
            Expr::Cast(ty, a) => {
                let (ty, a) = (self.ty_name(*ty), self.expr(a));
                match self.wgsl {
                    true => format!("{ty}({a})"),
                    false => format!("({ty}){a}"),
                }
            }
            Expr::Index(base, at) => {
                let at = self.expr(at);
                let Expr::Var(name) = &**base else {
                    panic!("an index names what it reads");
                };
                match (self.kind(name), self.wgsl) {
                    (Kind::Buffer(form), true) => form.wgsl_load(&self.c_name(name), &at),
                    (Kind::Buffer(form), false) => form.load(name, &at),
                    _ => format!("{}[{at}]", self.c_name(name)),
                }
            }
            Expr::Field(base, h) => {
                let form = self.wide(base);
                let base = self.expr(base);
                match (form, self.wgsl) {
                    (Some(form), true) => form.wgsl_part(&base, *h),
                    (Some(form), false) => form.part(&base, *h),
                    (None, _) => format!("{base}.{h}"),
                }
            }
            Expr::Call(i, args) => {
                self.uses(*i);
                i.print(self, args)
            }
            // A choice settled at build time leaves one branch.
            Expr::Select(c, a, b) if matches!(**c, Expr::Bool(_)) => {
                let taken = if **c == Expr::Bool(true) { a } else { b };
                self.expr(taken)
            }
            Expr::Select(c, a, b) => {
                let (c, a, b) = (self.expr(c), self.expr(a), self.expr(b));
                match self.wgsl {
                    true => format!("select({b}, {a}, {c})"),
                    false => format!("({c} ? {a} : {b})"),
                }
            }
            Expr::Fn(name, _, args) => {
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                format!("{name}({})", args.join(", "))
            }
        }
    }

    /// Prints a call: `name(args)`.
    pub(crate) fn spell(&mut self, name: &str, args: &[Expr]) -> String {
        let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
        format!("{name}({})", args.join(", "))
    }

    /// A tile block from bf16 pairs, lowest part first.
    fn parts(
        &mut self,
        acc: &str,
        [a, b, stride, apart]: [&str; 4],
        (tm, tn): (&str, &str),
        parts: usize,
        depth: usize,
    ) {
        self.line("#pragma unroll".into());
        self.line(format!("for (u32 kk = 0; kk < {}; kk += 8) {{", depth / 2));
        self.line(format!("    u32 tb[{tn}][2];"));
        self.line("    #pragma unroll".into());
        self.line(format!("    for (u32 j = 0; j < {tn}; j += 2) {{"));
        self.line(format!(
            "        tile_ldm_b(tb[j], tb[j + 1], {b} + (j * TILE_N + (lane / 16) * 8 + lane % 8) * ({stride}) + kk + ((lane / 8) % 2) * 4);"
        ));
        self.line("    }".into());
        self.line("    #pragma unroll".into());
        self.line(format!("    for (int p = {}; p >= 0; p--) {{", parts - 1));
        self.line("        #pragma unroll".into());
        self.line(format!("        for (u32 i = 0; i < {tm}; i++) {{"));
        self.line("            u32 ta[4];".into());
        self.line(format!(
            "            tile_ldm_a(ta, {a} + p * ({apart}) + (i * TILE_M + lane % 8 + ((lane / 8) % 2) * 8) * ({stride}) + kk + (lane / 16) * 4);"
        ));
        self.line("            #pragma unroll".into());
        self.line(format!("            for (u32 j = 0; j < {tn}; j++) {{"));
        self.line(format!(
            "                tile_mma_bf16({acc}_t[i][j], ta, tb[j]);"
        ));
        self.line("            }".into());
        self.line("        }".into());
        self.line("    }".into());
        self.line("}".into());
    }

    fn block(&mut self, stmts: &[Stmt]) {
        self.scope.push(HashMap::new());
        self.indent += 1;
        for s in stmts {
            self.stmt(s);
        }
        self.indent -= 1;
        self.scope.pop();
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) {
        if self.wgsl && tiles(s) {
            return self.refuse("tiles".into());
        }
        match s {
            Stmt::Let { name, init } => {
                let ty = self.ty(init);
                // The initializer reads names bound before this one.
                let wide =
                    |form: &dyn Load, wgsl| if wgsl { form.wgsl_wide() } else { form.wide() };
                let (ty, value, kind) = match (init, self.wide(init)) {
                    // A zeroed four leaves the compiler to fill it.
                    (Expr::Call(Intrinsic::Zero4, _), _) => {
                        (self.ty_name(ty), None, Kind::Value(ty))
                    }
                    (_, Some(form)) => (
                        wide(form, self.wgsl),
                        Some(self.expr(init)),
                        Kind::Wide(form),
                    ),
                    _ if ty == Ty::Bytes => (
                        self.ty_name(ty),
                        Some(self.expr(init)),
                        Kind::Buffer(U8_FORM),
                    ),
                    _ => (self.ty_name(ty), Some(self.expr(init)), Kind::Value(ty)),
                };
                let name = self.declare(name, kind);
                match (value, self.wgsl) {
                    (Some(value), true) => self.line(format!("var {name}: {ty} = {value};")),
                    (None, true) => self.line(format!("var {name}: {ty};")),
                    (Some(value), false) => self.line(format!("{ty} {name} = {value};")),
                    (None, false) => self.line(format!("{ty} {name};")),
                }
            }
            Stmt::Array { name, ty, len } => {
                let name = self.declare(name, Kind::Array(*ty));
                let ty = self.ty_name(*ty);
                match self.wgsl {
                    true => self.line(format!("var {name}: array<{ty}, {len}>;")),
                    false => self.line(format!("{ty} {name}[{len}] = {{}};")),
                }
            }
            Stmt::Shared { name, ty, len } => {
                self.bind(name, Kind::Array(*ty));
                let (name, ty) = (self.c_name(name), self.ty_name(*ty));
                match self.wgsl {
                    true => {
                        (self.module).push(format!("var<workgroup> {name}: array<{ty}, {len}>;"))
                    }
                    false => self.line(format!("f_group({ty}, {name}, {len});")),
                }
            }
            Stmt::Assign {
                target,
                op,
                value: Expr::Call(Intrinsic::Sum32, args),
            } => {
                assert!(op.is_none() && args[0] == *target, "`v = sum32(v)`");
                let v = self.expr(target);
                self.uses(Intrinsic::ShflXor);
                let mut m = LANES / 2;
                while m > 0 {
                    self.line(format!("{v} += f_shfl_xor({v}, {m});"));
                    m /= 2;
                }
            }
            Stmt::Assign { target, op, value } => {
                if let Expr::Index(base, at) = target
                    && let Expr::Var(name) = &**base
                    && let Kind::Buffer(form) = self.kind(name)
                {
                    if self.wgsl && !matches!(form.name(), "f32" | "u32") {
                        self.refuse("a store narrower than a word".into());
                    } else if !self.wgsl && form.narrow() {
                        let at = self.expr(at);
                        let rhs = match op {
                            Some(op) => {
                                let combined = Expr::Bin(
                                    *op,
                                    Box::new(target.clone()),
                                    Box::new(value.clone()),
                                );
                                self.expr(&combined)
                            }
                            None => self.expr(value),
                        };
                        self.line(format!("{};", form.store(name, &at, &rhs)));
                        return;
                    }
                }
                if op.is_none()
                    && let Some(form) = self.wide(value)
                    && let Expr::Var(name) = target
                    && matches!(self.kind(name), Kind::Value(Ty::F32x4))
                {
                    let (t, v) = (self.expr(target), self.expr(value));
                    self.uses(Intrinsic::Make4);
                    let fields = ['x', 'y', 'z', 'w'];
                    let parts = fields.map(|f| match self.wgsl {
                        true => form.wgsl_part("w__", f),
                        false => form.part("w__", f),
                    });
                    let [p0, p1, p2, p3] = parts;
                    if self.wgsl {
                        let ty = form.wgsl_wide();
                        self.line(format!(
                            "{{ let w__: {ty} = {v}; {t} = f_make4({p0}, {p1}, {p2}, {p3}); }}"
                        ));
                    } else {
                        let ty = form.wide();
                        self.line(format!(
                            "{{ {ty} w__ = {v}; {t} = f_make4({p0}, {p1}, {p2}, {p3}); }}"
                        ));
                    }
                    return;
                }
                let (t, v) = (self.expr(target), self.expr(value));
                let op = op.map_or(String::new(), |op| op.c().into());
                self.line(format!("{t} {op}= {v};"));
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                unroll,
                body,
            } => {
                // Loops over constants unroll unless told otherwise.
                let constant =
                    matches!((from, to, step), (Expr::Int(_), Expr::Int(_), Expr::Int(_)));
                // WGSL has no unroll hint.
                match unroll {
                    _ if self.wgsl => {}
                    0 if constant => self.line("#pragma unroll".into()),
                    0 => {}
                    1 => self.line("#pragma unroll".into()),
                    n => self.line(format!("#pragma unroll {n}")),
                }
                let (from, to, step) = (self.expr(from), self.expr(to), self.expr(step));
                self.scope.push(HashMap::new());
                let var = self.declare(var, Kind::Value(Ty::U32));
                let ty = if self.wgsl { "var" } else { "u32" };
                let typed = if self.wgsl { ": u32" } else { "" };
                self.line(format!(
                    "for ({ty} {var}{typed} = {from}; {var} < {to}; {var} += {step}) {{"
                ));
                self.block(body);
                self.scope.pop();
                self.line("}".into());
            }
            // A condition settled at build time leaves one branch.
            Stmt::If {
                cond: Expr::Bool(taken),
                then,
                els,
            } => {
                let branch = if *taken { then } else { els };
                if !branch.is_empty() {
                    self.line("{".into());
                    self.block(branch);
                    self.line("}".into());
                }
            }
            Stmt::If { cond, then, els } => {
                let cond = self.expr(cond);
                self.line(format!("if ({cond}) {{"));
                self.block(then);
                if !els.is_empty() {
                    self.line("} else {".into());
                    self.block(els);
                }
                self.line("}".into());
            }
            Stmt::TileAcc { name, rows, cols } => {
                // The tile macros read `lane`.
                self.lane = true;
                let (tm, tn) = (format!("({rows}) / TILE_M"), format!("({cols}) / TILE_N"));
                self.line(format!("tile_acc {name}_t[{tm}][{tn}];"));
                self.line(format!("float {name}[{tm}][{tn}][TILE_COUNT];"));
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line("        #pragma unroll".into());
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line(format!("            {name}[i][j][e] = 0.0f;"));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
                self.bind(name, Kind::Tiles(*rows, *cols));
            }
            Stmt::TileHeld { name, rows, depth } => {
                self.lane = true;
                let (tm, tk) = (format!("({rows}) / TILE_M"), format!("({depth}) / TILE_K"));
                self.line(format!("tile_held {name}[{tm}][{tk}];"));
                self.bind(name, Kind::Held(*rows, *depth));
            }
            Stmt::TileHold {
                held,
                a,
                a_at,
                stride,
            } => {
                let Kind::Held(rows, depth) = self.kind(held) else {
                    panic!("`{held}` is not a held slab");
                };
                let a = format!("({} + {})", self.expr(a), self.expr(a_at));
                let stride = self.expr(stride);
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!(
                    "    for (u32 kk = 0; kk < {depth}; kk += TILE_K) {{"
                ));
                self.line(format!(
                    "        TILE_HOLD_A({held}[i][kk / TILE_K], {a} + i * TILE_M * ({stride}) + kk, {stride});"
                ));
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileBlock {
                acc,
                a,
                a_at,
                b,
                b_at,
                stride,
                depth,
                parts,
                apart,
            } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                let held = match a {
                    Expr::Var(name) => match self.kind(name) {
                        Kind::Held(r, d) => {
                            assert!(
                                r == rows && d >= *depth && *parts == 0,
                                "`{name}` does not fit"
                            );
                            Some(name.clone())
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let (tm, tn) = (format!("({rows}) / TILE_M"), format!("({cols}) / TILE_N"));
                let a = format!("({} + {})", self.expr(a), self.expr(a_at));
                let b = format!("({} + {})", self.expr(b), self.expr(b_at));
                let stride = self.expr(stride);
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line(format!("        TILE_ZERO({acc}_t[i][j]);"));
                self.line("    }".into());
                self.line("}".into());
                if *parts > 0 {
                    let apart = self.expr(apart);
                    self.parts(acc, [&a, &b, &stride, &apart], (&tm, &tn), *parts, *depth);
                } else {
                    self.line("#pragma unroll".into());
                    self.line(format!("for (u32 kk = 0; kk < {depth}; kk += TILE_K) {{"));
                    self.line(format!("    tile_a ta[{tm}];"));
                    self.line(format!("    tile_b tb[{tn}];"));
                    self.line("    #pragma unroll".into());
                    self.line(format!("    for (u32 i = 0; i < {tm}; i++) {{"));
                    self.line(match &held {
                        Some(h) => format!("        TILE_HELD_A(ta[i], {h}[i][kk / TILE_K]);"),
                        None => format!(
                            "        TILE_LOAD_A(ta[i], {a} + i * TILE_M * ({stride}) + kk, {stride});"
                        ),
                    });
                    self.line("    }".into());
                    self.line("    #pragma unroll".into());
                    self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                    self.line(format!(
                        "        TILE_LOAD_B(tb[j], {b} + j * TILE_N * ({stride}) + kk, {stride});"
                    ));
                    self.line("    }".into());
                    self.line("    #pragma unroll".into());
                    self.line(format!("    for (u32 i = 0; i < {tm}; i++) {{"));
                    self.line("        #pragma unroll".into());
                    self.line(format!("        for (u32 j = 0; j < {tn}; j++) {{"));
                    self.line(format!(
                        "            TILE_MMA({acc}_t[i][j], ta[i], tb[j]);"
                    ));
                    self.line("        }".into());
                    self.line("    }".into());
                    self.line("}".into());
                }
                // The unit's sums truncate, so blocks add up in floats.
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line("        #pragma unroll".into());
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line(format!(
                    "            {acc}[i][j][e] += TILE_VAL({acc}_t[i][j], e);"
                ));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileEach { acc, body } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line(format!(
                    "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
                ));
                self.line("        #pragma unroll".into());
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line("            u32 r = i * TILE_M + TILE_ROW(e);".into());
                self.line("            u32 c = j * TILE_N + TILE_COL(e);".into());
                self.line(format!("            float v = {acc}[i][j][e];"));
                self.scope.push(HashMap::new());
                for (name, ty) in [("r", Ty::U32), ("c", Ty::U32), ("v", Ty::F32)] {
                    self.bind(name, Kind::Value(ty));
                }
                self.indent += 2;
                self.block(body);
                self.indent -= 2;
                self.scope.pop();
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileZero { acc } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line(format!(
                    "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
                ));
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line(format!("            {acc}[i][j][e] = 0.0f;"));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileScale { acc, by } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                self.scope.push(HashMap::new());
                self.bind("r", Kind::Value(Ty::U32));
                let by = self.expr(by);
                self.scope.pop();
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line(format!(
                    "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
                ));
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line("            u32 r = i * TILE_M + TILE_ROW(e);".into());
                self.line(format!("            {acc}[i][j][e] *= {by};"));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::Return => self.line("return;".into()),
            Stmt::Barrier if self.wgsl => self.line("workgroupBarrier();".into()),
            Stmt::Barrier => self.line("f_barrier();".into()),
            // WGSL discards a call's value only by name.
            Stmt::Expr(e @ Expr::Call(i, _)) if self.wgsl && *i != Intrinsic::Store4 => {
                let text = self.expr(e);
                self.line(format!("_ = {text};"));
            }
            Stmt::Expr(e) => {
                let text = self.expr(e);
                self.line(format!("{text};"));
            }
        }
    }
}

/// Whether `s` uses the matrix unit.
fn tiles(s: &Stmt) -> bool {
    use Stmt::*;
    matches!(
        s,
        TileAcc { .. }
            | TileHeld { .. }
            | TileHold { .. }
            | TileBlock { .. }
            | TileEach { .. }
            | TileZero { .. }
            | TileScale { .. }
    )
}

/// The wider of two types in arithmetic.
fn wider(a: Ty, b: Ty) -> Ty {
    match (a, b) {
        (Ty::Bytes, _) | (_, Ty::Bytes) => Ty::Bytes,
        (Ty::F32, _) | (_, Ty::F32) => Ty::F32,
        (Ty::U64, _) | (_, Ty::U64) => Ty::U64,
        (Ty::F32x4, _) | (_, Ty::F32x4) => Ty::F32x4,
        _ => Ty::U32,
    }
}

/// A float literal the dialects accept.
fn float(x: f64) -> String {
    if x.is_infinite() {
        let bits = if x > 0.0 {
            "0x7f800000u"
        } else {
            "0xff800000u"
        };
        return format!("f_bits({bits})");
    }
    let text = format!("{x:?}");
    match text.contains('.') || text.contains('e') {
        true => format!("{text}f"),
        false => format!("{text}.0f"),
    }
}
