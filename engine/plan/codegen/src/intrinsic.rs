// `kernel-macro` includes this file, so it holds only the table.

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
                host: |g, _, _| (0..g.threads).map(|t| Val::U((g.first + t) as u32)).collect(),
                count: |_, _| false,
            }
            /// The thread's index in its group.
            Lid = fn lid() -> u32 {
                device: none(),
                text: |_, _| "lid".into(),
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
                host: |g, _, _| (0..g.threads).map(|t| Val::U((t % g.lanes) as u32)).collect(),
                count: |_, _| false,
            }
            /// `shfl_xor(value, mask)`: what the lane `mask` away holds.
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
                host: |g, args, mask| g.unary(args, mask, f32::exp),
                count: |c, _| c.tally(special(1.0), true),
            }
            Tanh = fn tanh(f32) -> f32 {
                device: define("f_tanh(x)", "tanh(x)", "tanhf(x)"),
                text: |p, args| p.spell("f_tanh", args),
                host: |g, args, mask| g.unary(args, mask, f32::tanh),
                count: |c, _| c.tally(special(1.0), true),
            }
            Max = fn max(f32, f32) -> f32 {
                device: define("f_max(a, b)", "max(a, b)", "fmaxf(a, b)"),
                text: |p, args| p.spell("f_max", args),
                host: |g, args, mask| g.pair(args, mask, f32::max),
                count: |c, _| c.tally(float(1.0), true),
            }
            /// `clamp(value, low, high)`.
            Clamp = fn clamp(f32, f32, f32) -> f32 {
                device: define("f_clamp(x, lo, hi)", "clamp(x, lo, hi)", "fminf(fmaxf(x, lo), hi)"),
                text: |p, args| p.spell("f_clamp", args),
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
                host: |g, args, mask| g.pair(args, mask, f32::powf),
                count: |c, _| c.tally(special(1.0), true),
            }
            Cos = fn cos(f32) -> f32 {
                device: define("f_cos(x)", "precise::cos(x)", "cosf(x)"),
                text: |p, args| p.spell("f_cos", args),
                host: |g, args, mask| g.unary(args, mask, f32::cos),
                count: |c, _| c.tally(special(1.0), true),
            }
            Sin = fn sin(f32) -> f32 {
                device: define("f_sin(x)", "precise::sin(x)", "sinf(x)"),
                text: |p, args| p.spell("f_sin", args),
                host: |g, args, mask| g.unary(args, mask, f32::sin),
                count: |c, _| c.tally(special(1.0), true),
            }
            /// `f32::from_bits(x)`.
            FromBits = assoc f32::from_bits(u32) -> f32 {
                device: define("f_bits(x)", "as_type<float>(x)", "__uint_as_float(x)"),
                text: |p, args| p.spell("f_bits", args),
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
                host: |g, args, mask| {
                    let v = g.eval(&args[0], mask);
                    v.into_iter().map(|x| Val::U(x.f().to_bits())).collect()
                },
                count: |_, _| false,
            }
            Rsqrt = fn rsqrt(f32) -> f32 {
                device: define("f_rsqrt(x)", "rsqrt(x)", "rsqrtf(x)"),
                text: |p, args| p.spell("f_rsqrt", args),
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
                    form.load4(name, &at)
                },
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
                    format!("f_f32x4({name} + {}) = {}", p.expr(&args[1]), p.expr(&args[2]))
                },
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
                text: |_, _| "float4(0.0f, 0.0f, 0.0f, 0.0f)".into(),
                host: |g, _, _| vec![Val::V([0.0; 4]); g.threads],
                count: |_, _| false,
            }
            /// `F32x4::new(a, b, c, d)`.
            Make4 = assoc F32x4::new(f32, f32, f32, f32) -> F32x4 {
                device: define("f_make4(a, b, c, d)", "float4(a, b, c, d)", "make_float4(a, b, c, d)"),
                text: |p, args| p.spell("f_make4", args),
                host: |g, args, mask| {
                    let parts: Vec<Vec<Val>> = args.iter().map(|a| g.eval(a, mask)).collect();
                    (0..g.threads)
                        .map(|t| Val::V([0, 1, 2, 3].map(|i| parts[i][t].f())))
                        .collect()
                },
                count: |_, _| true,
            }
            /// `kept_row(slot, position)`: the row the layout gives.
            KeptRow = fn kept_row(u32, u32) -> u32 {
                device: none(),
                text: |p, args| {
                    let (slot, position) = (p.expr(&args[0]), p.expr(&args[1]));
                    let layout = p.layout.expect("a kernel reading kept rows names its layout");
                    layout.row(&slot, &position)
                },
                host: |g, args, mask| {
                    let layout = g.layout.expect("a kernel reading kept rows names its layout");
                    let values: Vec<u32> = (layout.params().iter())
                        .map(|p| match g.args.get(*p) {
                            Some(ArgVal::U32(v)) => *v,
                            other => panic!("layout scalar `{p}` is {other:?}"),
                        })
                        .collect();
                    let (s, p) = (g.eval(&args[0], mask), g.eval(&args[1], mask));
                    (0..g.threads)
                        .map(|t| Val::U(layout.at(s[t].u(), p[t].u(), &values)))
                        .collect()
                },
                count: |c, _| c.tally(int(2.0), false),
            }
            /// `p.word(offset)`: the u32 at a byte offset.
            Word = method word(self, u32) -> u32 {
                device: define("f_u32(p)", "(*(device const u32 *)(p))", "(*(const u32 *)(p))"),
                text: |p, args| format!("f_u32(({} + {}))", p.expr(&args[0]), p.expr(&args[1])),
                host: |g, args, mask| g.read(args, mask, 4),
                count: |_, _| false,
            }
            /// `p.half(offset)`: the u16 at a byte offset.
            Half = method half(self, u32) -> u32 {
                device: define("f_u16(p)", "(*(device const u16 *)(p))", "(*(const u16 *)(p))"),
                text: |p, args| format!("f_u16(({} + {}))", p.expr(&args[0]), p.expr(&args[1])),
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
                host: |g, args, mask| g.narrow(args, mask, 8, |x| f32::from(E4M3(x as u8))),
                count: |c, _| c.tally(Ops { e4m3: 1.0, ..Ops::default() }, true),
            }
            /// `w.e2m1(low)`: 4 bits read as an e2m1.
            E2M1 = method e2m1(self, u32) -> f32 {
                device: both(&format!(
                    "{}#define f_e2m1(w, l) f_e2m1_bits(f_field(w, l, 4))\n",
                    widen("f_e2m1_bits", 2, 1)
                )),
                text: |p, args| p.spell("f_e2m1", args),
                host: |g, args, mask| g.narrow(args, mask, 4, |x| f32::from(E2M1(x as u8))),
                count: |c, _| c.tally(Ops { e2m1: 1.0, ..Ops::default() }, true),
            }
        }
    };
}

pub(crate) use table;
