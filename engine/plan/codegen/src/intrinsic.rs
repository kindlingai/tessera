// `kernel-macro` and `host` include this file: only the table.

/// Every intrinsic. `table!(m)` hands the rows to `m!`.
macro_rules! table {
    ($then:ident) => {
        // Closures resolve names in the module that expands the table.
        $then! {
            // Device text prints in row order: later rows use earlier.
            /// The thread's index in the grid.
            Gid = fn gid() -> u32 {
                device: none(),
                text: |_, _| "gid".into(),
                wgsl: bare(),
                host: |g, _, _| (0..g.threads).map(|t| Val::U((g.first + t) as u32)).collect(),
                count: |_, _| false,
            }
            /// The thread's index in its group.
            Lid = fn lid() -> u32 {
                device: none(),
                text: |_, _| "lid".into(),
                wgsl: bare(),
                host: |g, _, _| (0..g.threads).map(|t| Val::U(t as u32)).collect(),
                count: |_, _| false,
            }
            /// The thread's lane among the 32 that shuffle together.
            Lane = fn lane() -> u32 {
                device: none(),
                text: |p, _| {
                    p.lane = true;
                    "lane".into()
                },
                wgsl: bare(),
                host: |g, _, _| (0..g.threads).map(|t| Val::U((t % g.lanes) as u32)).collect(),
                count: |_, _| false,
            }
            /// `shfl_xor(value, mask)`: the value in the lane `mask` away.
            ShflXor = fn shfl_xor(f32, u32) -> f32 {
                // HIP leaves out the mask.
                device: spelled(
                    "#define f_shfl_xor(v, m) simd_shuffle_xor(v, m)\n",
                    &hip(
                        "#define f_shfl_xor(v, m) __shfl_xor(v, m)\n",
                        "#define f_shfl_xor(v, m) __shfl_xor_sync(0xffffffffu, v, m)\n",
                    ),
                ),
                text: |p, args| p.spell("f_shfl_xor", args),
                wgsl: call("f_shfl_xor(v: f32, m: u32)", "f32", "subgroupShuffleXor(v, m)"),
                host: |g, args, mask| {
                    let (v, m) = (g.eval(&args[0], mask), g.eval(&args[1], mask));
                    let (n, lanes) = (g.threads, g.lanes);
                    (0..n)
                        .map(|t| {
                            let other = t ^ (m[t].u() as usize);
                            // Lanes shuffle within their own SIMD group.
                            if other / lanes == t / lanes && other < n {
                                v[other]
                            } else {
                                v[t]
                            }
                        })
                        .collect()
                },
                count: |c, _| c.tally(float(1.0), true),
            }
            /// `sum32(value)`: the sum over the 32 lanes, in every lane.
            Sum32 = fn sum32(f32) -> f32 {
                device: none(),
                text: |_, _| panic!("sum32 is a statement: `v = sum32(v)`"),
                wgsl: bare(),
                host: |g, args, mask| {
                    let v = g.eval(&args[0], mask);
                    let (n, lanes) = (g.threads, g.lanes);
                    (0..n)
                        .map(|t| {
                            let group = t / lanes * lanes;
                            let end = (group + lanes).min(n);
                            Val::F((group..end).filter(|s| mask[*s]).map(|s| v[s].f()).sum())
                        })
                        .collect()
                },
                count: |c, _| c.tally(Ops { float: 5.0, int: 5.0, ..Ops::default() }, true),
            }
                        Exp = fn exp(f32) -> f32 {
                device: define("f_exp(x)", "exp(x)", "expf(x)"),
                text: |p, args| p.spell("f_exp", args),
                wgsl: call("f_exp(x: f32)", "f32", "exp(x)"),
                host: |g, args, mask| g.unary(args, mask, f32::exp),
                count: |c, _| c.tally(special(1.0), true),
            }
            Tanh = fn tanh(f32) -> f32 {
                device: define("f_tanh(x)", "tanh(x)", "tanhf(x)"),
                text: |p, args| p.spell("f_tanh", args),
                wgsl: call("f_tanh(x: f32)", "f32", "tanh(x)"),
                host: |g, args, mask| g.unary(args, mask, f32::tanh),
                count: |c, _| c.tally(special(1.0), true),
            }
            Max = fn max(f32, f32) -> f32 {
                device: define("f_max(a, b)", "max(a, b)", "fmaxf(a, b)"),
                text: |p, args| p.spell("f_max", args),
                wgsl: call("f_max(a: f32, b: f32)", "f32", "max(a, b)"),
                host: |g, args, mask| g.pair(args, mask, f32::max),
                count: |c, _| c.tally(float(1.0), true),
            }
            /// `clamp(value, low, high)`.
            Clamp = fn clamp(f32, f32, f32) -> f32 {
                device: define("f_clamp(x, lo, hi)", "clamp(x, lo, hi)", "fminf(fmaxf(x, lo), hi)"),
                text: |p, args| p.spell("f_clamp", args),
                wgsl: call("f_clamp(x: f32, lo: f32, hi: f32)", "f32", "min(max(x, lo), hi)"),
                host: |g, args, mask| {
                    let v = g.eval(&args[0], mask);
                    let (lo, hi) = (g.eval(&args[1], mask), g.eval(&args[2], mask));
                    (0..g.threads)
                        .map(|t| Val::F(v[t].f().clamp(lo[t].f(), hi[t].f())))
                        .collect()
                },
                count: |c, _| c.tally(float(2.0), true),
            }
            Pow = fn pow(f32, f32) -> f32 {
                device: define("f_pow(a, b)", "precise::pow(a, b)", "powf(a, b)"),
                text: |p, args| p.spell("f_pow", args),
                wgsl: call("f_pow(a: f32, b: f32)", "f32", "pow(a, b)"),
                host: |g, args, mask| g.pair(args, mask, f32::powf),
                count: |c, _| c.tally(special(1.0), true),
            }
            Cos = fn cos(f32) -> f32 {
                device: define("f_cos(x)", "precise::cos(x)", "cosf(x)"),
                text: |p, args| p.spell("f_cos", args),
                wgsl: call("f_cos(x: f32)", "f32", "cos(x)"),
                host: |g, args, mask| g.unary(args, mask, f32::cos),
                count: |c, _| c.tally(special(1.0), true),
            }
            Sin = fn sin(f32) -> f32 {
                device: define("f_sin(x)", "precise::sin(x)", "sinf(x)"),
                text: |p, args| p.spell("f_sin", args),
                wgsl: call("f_sin(x: f32)", "f32", "sin(x)"),
                host: |g, args, mask| g.unary(args, mask, f32::sin),
                count: |c, _| c.tally(special(1.0), true),
            }
            /// `f32::from_bits(x)`.
            FromBits = assoc f32::from_bits(u32) -> f32 {
                device: define("f_bits(x)", "as_type<float>(x)", "__uint_as_float(x)"),
                text: |p, args| p.spell("f_bits", args),
                wgsl: call("f_bits(x: u32)", "f32", "bitcast<f32>(x)"),
                host: |g, args, mask| {
                    let v = g.eval(&args[0], mask);
                    v.into_iter().map(|x| Val::F(f32::from_bits(x.u()))).collect()
                },
                count: |_, _| true,
            }
            /// `x.to_bits()`.
            ToBits = method to_bits(self) -> u32 {
                device: define("f_asbits(x)", "as_type<u32>(x)", "__float_as_uint(x)"),
                text: |p, args| p.spell("f_asbits", args),
                wgsl: call("f_asbits(x: f32)", "u32", "bitcast<u32>(x)"),
                host: |g, args, mask| {
                    let v = g.eval(&args[0], mask);
                    v.into_iter().map(|x| Val::U(x.f().to_bits())).collect()
                },
                count: |_, _| false,
            }
            Rsqrt = fn rsqrt(f32) -> f32 {
                device: define("f_rsqrt(x)", "rsqrt(x)", "rsqrtf(x)"),
                text: |p, args| p.spell("f_rsqrt", args),
                wgsl: call("f_rsqrt(x: f32)", "f32", "inverseSqrt(x)"),
                host: |g, args, mask| g.unary(args, mask, |x| 1.0 / x.sqrt()),
                count: |c, _| c.tally(special(1.0), true),
            }
            /// `load4(buffer, index)`: four elements as floats.
            Load4 = method load4(self, u64) -> F32x4 {
                device: none(),
                text: |p, args| {
                    let (form, at) = (p.buffer(&args[0]), p.expr(&args[1]));
                    let Expr::Var(name) = &args[0] else {
                        unreachable!()
                    };
                    match p.wgsl {
                        true => form.wgsl_load4(&p.c_name(name), &at),
                        false => form.load4(name, &at),
                    }
                },
                wgsl: bare(),
                host: |g, args, mask| {
                    let Expr::Var(name) = &args[0] else {
                        panic!("a load names its buffer");
                    };
                    let (arg, at) = (g.arg(name), g.eval(&args[1], mask));
                    (0..g.threads)
                        .map(|t| match mask[t] {
                            true => g.load4(&arg, at[t].l()),
                            false => Val::V([0.0; 4]),
                        })
                        .collect()
                },
                count: |c, args| {
                    let bytes = c.global(&args[0]).map_or(4.0, |f| f.element_bytes());
                    let (at, mut l) = (&args[1], c.loops.pop());
                    // Four elements move the address by one of them.
                    let fresh = match &mut l {
                        None => 1.0,
                        Some(l) => match c.advance(at, l).times(l.step.unwrap_or(f64::NAN)) {
                            Advance::Still => 0.0,
                            Advance::By(u) if u.is_finite() => (u * bytes / LINE).min(1.0),
                            _ => 1.0,
                        },
                    };
                    c.loops.extend(l);
                    let ops = Ops { loads: 1.0, fresh, width: 4.0 * bytes, ..Ops::default() };
                    c.tally(ops, true)
                },
            }
            /// `store4(buffer, index, value)`.
            Store4 = method store4(self, u64, F32x4) {
                device: define("f_f32x4(p)", "(*(device float4 *)(p))", "(*(float4 *)(p))"),
                text: |p, args| {
                    let Expr::Var(name) = &args[0] else {
                        panic!("a store names its buffer");
                    };
                    let (at, v) = (p.expr(&args[1]), p.expr(&args[2]));
                    match p.wgsl {
                        true => store4(&p.c_name(name), &at, &v),
                        false => format!("f_f32x4({name} + {at}) = {v}"),
                    }
                },
                wgsl: bare(),
                host: |g, args, mask| {
                    let Expr::Var(name) = &args[0] else {
                        panic!("a store names its buffer");
                    };
                    let arg = g.arg(name);
                    let (at, v) = (g.eval(&args[1], mask), g.eval(&args[2], mask));
                    for t in (0..g.threads).filter(|t| mask[*t]) {
                        for (i, x) in v[t].v().into_iter().enumerate() {
                            g.store(&arg, at[t].l() + i as u64, Val::F(x));
                        }
                    }
                    vec![Val::B(false); g.threads]
                },
                count: |c, _| c.tally(Ops { stores: 1.0, ..Ops::default() }, false),
            }
            /// A zeroed `float4`.
            Zero4 = assoc F32x4::zero() -> F32x4 {
                device: none(),
                text: |p, _| match p.wgsl {
                    true => "vec4<f32>()".into(),
                    false => "float4(0.0f, 0.0f, 0.0f, 0.0f)".into(),
                },
                wgsl: bare(),
                host: |g, _, _| vec![Val::V([0.0; 4]); g.threads],
                count: |_, _| false,
            }
            /// `F32x4::new(a, b, c, d)`.
            Make4 = assoc F32x4::new(f32, f32, f32, f32) -> F32x4 {
                device: define("f_make4(a, b, c, d)", "float4(a, b, c, d)", "make_float4(a, b, c, d)"),
                text: |p, args| p.spell("f_make4", args),
                wgsl: call("f_make4(a: f32, b: f32, c: f32, d: f32)", "vec4<f32>", "vec4<f32>(a, b, c, d)"),
                host: |g, args, mask| {
                    let parts: Vec<Vec<Val>> = args.iter().map(|a| g.eval(a, mask)).collect();
                    (0..g.threads)
                        .map(|t| Val::V([0, 1, 2, 3].map(|i| parts[i][t].f())))
                        .collect()
                },
                count: |_, _| true,
            }
            /// `kept_row(slot, position)`. `kept::attach` replaces each call.
            KeptRow = fn kept_row(u32, u32) -> u32 {
                device: none(),
                text: |_, _| panic!("kept_row without a layout attached"),
                wgsl: bare(),
                host: |_, _, _| panic!("kept_row without a layout attached"),
                count: |c, _| c.tally(int(2.0), false),
            }
            /// `p.word(offset)`: the u32 at a byte offset.
            Word = method word(self, u32) -> u32 {
                device: define(
                    "f_word(p, o)",
                    "(*(device const u32 *)((device const u8 *)(p) + (o)))",
                    "(*(const u32 *)((const u8 *)(p) + (o)))",
                ),
                text: |p, args| p.spell("f_word", args),
                wgsl: None,
                host: |g, args, mask| g.read(args, mask, 4),
                count: |_, _| false,
            }
            /// `p.half(offset)`: the u16 at a byte offset.
            Half = method half(self, u32) -> u32 {
                device: define(
                    "f_half(p, o)",
                    "(*(device const u16 *)((device const u8 *)(p) + (o)))",
                    "(*(const u16 *)((const u8 *)(p) + (o)))",
                ),
                text: |p, args| p.spell("f_half", args),
                wgsl: None,
                host: |g, args, mask| g.read(args, mask, 2),
                count: |_, _| false,
            }
            /// `w.bits(low, n)`: a bit field as a u32.
            // Bit fields: one extract instruction on each backend.
            Bits = method bits(self, u32, u32) -> u32 {
                device: spelled(
                    "#define f_field(w, l, n) extract_bits((u32)(w), (l), (n))\n",
                    &hip(
                        "#define f_field(w, l, n) (((u32)(w) >> (l)) & ((1u << (n)) - 1u))\n",
                        "f_fn u32 f_field(u32 w, u32 l, u32 n) {
    u32 r;
    asm(\"bfe.u32 %0, %1, %2, %3;\" : \"=r\"(r) : \"r\"(w), \"r\"(l), \"r\"(n));
    return r;
}
",
                    ),
                ),
                text: |p, args| p.spell("f_field", args),
                wgsl: call("f_field(w: u32, l: u32, n: u32)", "u32", "extractBits(w, l, n)"),
                host: |g, args, mask| g.fields(args, mask, |f, _| Val::U(f)),
                count: |c, _| c.tally(int(1.0), false),
            }
            /// `w.signed(low, n)`: a bit field, sign-extended, as a float.
            Signed = method signed(self, u32, u32) -> f32 {
                device: spelled(
                    "#define f_sfield(w, l, n) ((float)extract_bits((int)(w), (l), (n)))\n",
                    &hip(
                        "#define f_sfield(w, l, n) \
                         ((float)(((int)((u32)(w) << (32u - (l) - (n)))) >> (32u - (n))))\n",
                        "f_fn float f_sfield(u32 w, u32 l, u32 n) {
    int r;
    asm(\"bfe.s32 %0, %1, %2, %3;\" : \"=r\"(r) : \"r\"(w), \"r\"(l), \"r\"(n));
    return (float)r;
}
",
                    ),
                ),
                text: |p, args| p.spell("f_sfield", args),
                wgsl: call("f_sfield(w: u32, l: u32, n: u32)", "f32", "f32(extractBits(bitcast<i32>(w), l, n))"),
                host: |g, args, mask| g.fields(args, mask, |f, n| Val::F(sign_extend(f, n) as f32)),
                count: |c, _| c.tally(Ops { int: 1.0, float: 1.0, ..Ops::default() }, true),
            }
            // Tallied by format: each chip prices its conversions.
            /// `w.f16(low)`: 16 bits read as an f16.
            F16 = method f16(self, u32) -> f32 {
                device: spelled(
                    "#define f_f16(w, l) ((float)as_type<half>((u16)f_field(w, l, 16)))\n",
                    &hip(
                        "#define f_f16(w, l) \
                         ((float)__builtin_bit_cast(_Float16, (unsigned short)f_field(w, l, 16)))\n",
                        "f_fn float f_f16(u32 w, u32 l) {
    float r;
    unsigned short h = (unsigned short)f_field(w, l, 16);
    asm(\"{.reg .b16 t; mov.b16 t, %1; cvt.f32.f16 %0, t;}\" : \"=f\"(r) : \"h\"(h));
    return r;
}
",
                    ),
                ),
                text: |p, args| p.spell("f_f16", args),
                wgsl: call("f_f16(w: u32, l: u32)", "f32", "unpack2x16float(w >> l).x"),
                host: |g, args, mask| g.narrow(args, mask, 16, |x| f32::from(F16(x as u16))),
                count: |c, _| c.tally(Ops { f16: 1.0, ..Ops::default() }, true),
            }
            /// `w.e4m3(low)`: 8 bits read as an e4m3.
            E4M3 = method e4m3(self, u32) -> f32 {
                device: {
                    let bits = widen("f_e4m3_bits", 4, 3);
                    let soft = "#define f_e4m3(w, l) f_e4m3_bits(f_field(w, l, 8))\n";
                    // Ada on converts in one instruction.
                    let cuda = format!(
                        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
f_fn float f_e4m3(u32 w, u32 l) {{
    unsigned short b = (unsigned short)f_field(w, l, 8);
    u32 h;
    asm(\"cvt.rn.f16x2.e4m3x2 %0, %1;\" : \"=r\"(h) : \"h\"(b));
    return f_f16(h, 0);
}}
#else
{soft}#endif
"
                    );
                    [format!("{bits}{soft}"), format!("{bits}{cuda}")]
                },
                text: |p, args| p.spell("f_e4m3", args),
                wgsl: None,
                host: |g, args, mask| g.narrow(args, mask, 8, |x| f32::from(E4M3(x as u8))),
                count: |c, _| c.tally(Ops { e4m3: 1.0, ..Ops::default() }, true),
            }
            /// `a.dot(a_at, b, b_at, n)`: the sum of `n` products.
            // Devices spell it over f32 rows. CPU bodies take any.
            Dot = method dot(self, u64, Buf, u64, u32) -> f32 {
                device: both(
                    "f_fn float f_dot_rows(f_const(float) a, f_const(float) b, u32 n) {
    float s = 0.0f;
    for (u32 i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}
#define f_dot(a, i, b, j, n) f_dot_rows((a) + (i), (b) + (j), (n))
",
                ),
                text: |p, args| p.spell("f_dot", args),
                wgsl: None,
                host: |g, args, mask| {
                    let (Expr::Var(a), Expr::Var(b)) = (&args[0], &args[2]) else {
                        panic!("a dot names its buffers");
                    };
                    let (a, b) = (g.arg(a), g.arg(b));
                    let (a_at, b_at) = (g.eval(&args[1], mask), g.eval(&args[3], mask));
                    let n = g.eval(&args[4], mask);
                    (0..g.threads)
                        .map(|t| {
                            let mut sum = 0.0;
                            for i in 0..u64::from(n[t].u()) * u64::from(mask[t]) {
                                let x = g.load(&a, a_at[t].l() + i).f();
                                sum += x * g.load(&b, b_at[t].l() + i).f();
                            }
                            Val::F(sum)
                        })
                        .collect()
                },
                count: |c, _| c.tally(float(2.0), true),
            }
            /// `o.stage(at, n, code, s, b, j)`: one row stage.
            // `code` names a `stage::Row`. `b` and `s` are its arguments.
            Stage = method stage(self, u64, u32, u32, f32, Buf, u64) {
                device: {
                    let rows = |out: &str| format!(
                        "f_fn void f_stage_rows({out} o, f_const(float) b, u32 n, u32 code, float s) {{
    for (u32 i = 0; i < n; i++) {{
        float v = o[i];
        if (code == 0u) {{ v = v + b[i]; }}
        else if (code == 1u) {{ v = v * b[i]; }}
        else if (code == 2u) {{ v = v * s; }}
        else if (code == 3u) {{
            float u = fmin(fmax(0.7978846f * (v + 0.044715f * (v * v * v)), -8.0f), 8.0f);
            v = 0.5f * v * (1.0f + tanh(u));
        }}
        else if (code == 4u) {{ v = v / (1.0f + exp(-v)); }}
        else if (code == 5u) {{ v = 1.0f / (1.0f + exp(-v)); }}
        else {{ v = fmax(v, 0.0f); }}
        o[i] = v;
    }}
}}
#define f_stage(o, a, n, c, s, b, j) f_stage_rows((o) + (a), (b) + (j), n, c, s)
"
                    );
                    [rows("device float *"), rows("float *")]
                },
                text: |p, args| p.spell("f_stage", args),
                wgsl: None,
                host: |g, args, mask| {
                    let (Expr::Var(out), Expr::Var(b)) = (&args[0], &args[5]) else {
                        panic!("a stage names its buffers");
                    };
                    let (out, b) = (g.arg(out), g.arg(b));
                    let v: Vec<Vec<Val>> = [1, 2, 3, 4, 6].iter().map(|i| g.eval(&args[*i], mask)).collect();
                    for t in (0..g.threads).filter(|t| mask[*t]) {
                        let (at, n, b_at) = (v[0][t].l(), u64::from(v[1][t].u()), v[4][t].l());
                        let row = tessera_codegen::stage::Row::from_code(v[2][t].u()).expect("a stage code");
                        let mut line: Vec<f32> = (0..n).map(|i| g.load(&out, at + i).f()).collect();
                        let operand: Vec<f32> = match row.reads() {
                            true => (0..n).map(|i| g.load(&b, b_at + i).f()).collect(),
                            false => Vec::new(),
                        };
                        tessera_codegen::stage::apply(row, &mut line, &operand, v[3][t].f());
                        for (i, x) in line.into_iter().enumerate() {
                            g.store(&out, at + i as u64, Val::F(x));
                        }
                    }
                    vec![Val::B(false); g.threads]
                },
                count: |c, _| c.tally(Ops { stores: 1.0, ..float(8.0) }, false),
            }
            /// `o.mix(at, keep, p, b, j, n)`: `o * keep + p * b`.
            // A zero `keep` drops old values, maybe junk.
            /// `o.attend(..)`: one query's softmax attention into `o`.
            // Keys `from..to`, or the `n` listed at `l`.
            Attend = method attend(self, u64, Buf, Buf, Buf, u64, u32, u32, u32, u32, f32, Buf, u64, u32) {
                device: {
                    let rows = |out: &str| format!(
                        "f_fn void f_attend_row({out} o, f_const(float) q, f_const(float) k, f_const(float) v, u32 stride, u32 from, u32 to, u32 width, float scale, f_const(u32) ids, u32 picks) {{
    u32 n = picks > 0u ? picks : to - from;
    float top = -1.0e30f;
    for (u32 i = 0; i < n; i++) {{
        u32 j = picks > 0u ? ids[i] : from + i;
        if (j < from || j >= to) {{ continue; }}
        float s = 0.0f;
        for (u32 d = 0; d < width; d++) {{ s += q[d] * k[j * stride + d]; }}
        top = fmax(top, s * scale);
    }}
    for (u32 d = 0; d < width; d++) {{ o[d] = 0.0f; }}
    float total = 0.0f;
    for (u32 i = 0; i < n; i++) {{
        u32 j = picks > 0u ? ids[i] : from + i;
        if (j < from || j >= to) {{ continue; }}
        float s = 0.0f;
        for (u32 d = 0; d < width; d++) {{ s += q[d] * k[j * stride + d]; }}
        float p = exp(s * scale - top);
        total += p;
        for (u32 d = 0; d < width; d++) {{ o[d] += p * v[j * stride + d]; }}
    }}
    for (u32 d = 0; d < width; d++) {{ o[d] = o[d] / total; }}
}}
#define f_attend(o, a, q, k, v, ka, st, f, t, w, sc, ids, la, n) f_attend_row((o) + (a), (q) + (a), (k) + (ka), (v) + (ka), st, f, t, w, sc, (ids) + (la), n)
"
                    );
                    [rows("device float *"), rows("float *")]
                },
                text: |p, args| p.spell("f_attend", args),
                wgsl: None,
                host: |g, args, mask| {
                    let names = [0, 2, 3, 4, 11].map(|i| match &args[i] {
                        Expr::Var(name) => g.arg(name),
                        _ => panic!("attention names its buffers"),
                    });
                    let [out, q, k, v, ids] = names;
                    let n: Vec<Vec<Val>> = [1, 5, 6, 7, 8, 9, 10, 12, 13].iter().map(|i| g.eval(&args[*i], mask)).collect();
                    for t in (0..g.threads).filter(|t| mask[*t]) {
                        let (at, k_at, stride) = (n[0][t].l(), n[1][t].l(), u64::from(n[2][t].u()));
                        let (from, to, width) = (n[3][t].u(), n[4][t].u(), u64::from(n[5][t].u()));
                        let (scale, list_at, picks) = (n[6][t].f(), n[7][t].l(), n[8][t].u());
                        let listed = (0..picks).map(|i| g.load(&ids, list_at + u64::from(i)).u());
                        let keys: Vec<u32> = match picks {
                            0 => (from..to).collect(),
                            _ => listed.filter(|j| (from..to).contains(j)).collect(),
                        };
                        let row = |b, at: u64| -> Vec<f32> { (0..width).map(|d| g.load(b, at + d).f()).collect() };
                        let qr = row(&q, at);
                        let mut scores: Vec<f32> = Vec::new();
                        for j in &keys {
                            let kr = row(&k, k_at + u64::from(*j) * stride);
                            scores.push(qr.iter().zip(&kr).map(|(a, b)| a * b).sum::<f32>() * scale);
                        }
                        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let ps: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                        let total: f32 = ps.iter().sum();
                        let mut sum = vec![0.0f32; width as usize];
                        for (j, p) in keys.iter().zip(&ps) {
                            let vr = row(&v, k_at + u64::from(*j) * stride);
                            sum.iter_mut().zip(&vr).for_each(|(s, x)| *s += p / total * x);
                        }
                        for (d, x) in sum.into_iter().enumerate() {
                            g.store(&out, at + d as u64, Val::F(x));
                        }
                    }
                    vec![Val::B(false); g.threads]
                },
                count: |c, _| c.tally(Ops { stores: 1.0, ..float(8.0) }, false),
            }
            Mix = method mix(self, u64, f32, f32, Buf, u64, u32) {
                device: {
                    let rows = |out: &str| format!(
                        "f_fn void f_mix_rows({out} o, f_const(float) b, float keep, float p, u32 n) {{
    for (u32 i = 0; i < n; i++) {{
        o[i] = (keep == 0.0f ? 0.0f : o[i] * keep) + p * b[i];
    }}
}}
#define f_mix(o, i, keep, p, b, j, n) f_mix_rows((o) + (i), (b) + (j), keep, p, n)
"
                    );
                    [rows("device float *"), rows("float *")]
                },
                text: |p, args| p.spell("f_mix", args),
                wgsl: None,
                host: |g, args, mask| {
                    let (Expr::Var(out), Expr::Var(b)) = (&args[0], &args[4]) else {
                        panic!("a mix names its buffers");
                    };
                    let (out, b) = (g.arg(out), g.arg(b));
                    let at = g.eval(&args[1], mask);
                    let (keep, p) = (g.eval(&args[2], mask), g.eval(&args[3], mask));
                    let (b_at, n) = (g.eval(&args[5], mask), g.eval(&args[6], mask));
                    for t in (0..g.threads).filter(|t| mask[*t]) {
                        let (keep, p) = (keep[t].f(), p[t].f());
                        for i in 0..u64::from(n[t].u()) {
                            let old = match keep == 0.0 {
                                true => 0.0,
                                false => g.load(&out, at[t].l() + i).f() * keep,
                            };
                            let add = p * g.load(&b, b_at[t].l() + i).f();
                            g.store(&out, at[t].l() + i, Val::F(old + add));
                        }
                    }
                    vec![Val::B(false); g.threads]
                },
                count: |c, _| c.tally(Ops { stores: 1.0, ..float(3.0) }, false),
            }
            /// `o.dots(..)`: a block of sums, rows of `x` by `w`.
            Dots = method dots(self, u64, u32, Buf, u64, Buf, u64, u32, u32, u32) {
                device: {
                    let rows = |out: &str| format!(
                        "f_fn void f_dots_rows({out} o, u32 stride, f_const(float) x, f_const(float) w, u32 k, u32 rows, u32 cols) {{
    for (u32 r = 0; r < rows; r++) {{
        for (u32 c = 0; c < cols; c++) {{
            float s = 0.0f;
            for (u32 i = 0; i < k; i++) {{
                s += x[r * k + i] * w[c * k + i];
            }}
            o[r * stride + c] = s;
        }}
    }}
}}
#define f_dots(o, a, s, x, i, w, j, k, r, n) f_dots_rows((o) + (a), s, (x) + (i), (w) + (j), k, r, n)
"
                    );
                    [rows("device float *"), rows("float *")]
                },
                text: |p, args| p.spell("f_dots", args),
                wgsl: None,
                host: |g, args, mask| {
                    let (Expr::Var(out), Expr::Var(x), Expr::Var(w)) = (&args[0], &args[3], &args[5])
                    else {
                        panic!("a dots names its buffers");
                    };
                    let (out, x, w) = (g.arg(out), g.arg(x), g.arg(w));
                    let v: Vec<Vec<Val>> = [1, 2, 4, 6, 7, 8, 9].iter().map(|i| g.eval(&args[*i], mask)).collect();
                    for t in (0..g.threads).filter(|t| mask[*t]) {
                        let (at, stride, x_at, w_at) = (v[0][t].l(), u64::from(v[1][t].u()), v[2][t].l(), v[3][t].l());
                        let (k, rows) = (u64::from(v[4][t].u()), u64::from(v[5][t].u()));
                        for r in 0..rows {
                            for c in 0..u64::from(v[6][t].u()) {
                                let mut sum = 0.0;
                                for i in 0..k {
                                    let a = g.load(&x, x_at + r * k + i).f();
                                    sum += a * g.load(&w, w_at + c * k + i).f();
                                }
                                g.store(&out, at + r * stride + c, Val::F(sum));
                            }
                        }
                    }
                    vec![Val::B(false); g.threads]
                },
                count: |c, _| c.tally(Ops { stores: 256.0, ..float(512.0) }, false),
            }
            /// `w.e2m1(low)`: 4 bits read as an e2m1.
            E2M1 = method e2m1(self, u32) -> f32 {
                device: both(&format!(
                    "{}#define f_e2m1(w, l) f_e2m1_bits(f_field(w, l, 4))\n",
                    widen("f_e2m1_bits", 2, 1)
                )),
                text: |p, args| p.spell("f_e2m1", args),
                wgsl: None,
                host: |g, args, mask| g.narrow(args, mask, 4, |x| f32::from(E2M1(x as u8))),
                count: |c, _| c.tally(Ops { e2m1: 1.0, ..Ops::default() }, true),
            }
            /// `bf16_pair(hi, lo)`: both rounded to bf16, nearest even, `hi` high.
            Bf16Pair = fn bf16_pair(f32, f32) -> u32 {
                device: {
                    let soft = pair_bits("as_type<u32>");
                    // Ampere on rounds a pair in one instruction.
                    let cuda = format!(
                        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
f_fn u32 f_pair_bf16(float hi, float lo) {{
    u32 d;
    asm(\"cvt.rn.bf16x2.f32 %0, %1, %2;\" : \"=r\"(d) : \"f\"(hi), \"f\"(lo));
    return d;
}}
#else
{}#endif
",
                        pair_bits("__float_as_uint")
                    );
                    let call = "#define f_bf16_pair(hi, lo) f_pair_bf16(hi, lo)\n";
                    let cuda = hip(&pair_bits("__float_as_uint"), &cuda);
                    spelled(&format!("{soft}{call}"), &format!("{cuda}{call}"))
                },
                text: |p, args| p.spell("f_bf16_pair", args),
                wgsl: None,
                host: |g, args, mask| {
                    let (hi, lo) = (g.eval(&args[0], mask), g.eval(&args[1], mask));
                    let pair = |(h, l): (Val, Val)| Val::U(crate::lockstep::bf16_pair(h.f(), l.f()));
                    hi.into_iter().zip(lo).map(pair).collect()
                },
                count: |c, _| c.tally(int(2.0), false),
            }
        }
    };
}

pub(crate) use table;
