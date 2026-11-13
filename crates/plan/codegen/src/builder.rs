//! A kernel body, line by line, with its trip cost.

use crate::dialect::Text;
use graph::DType;
use sim::Ops;

/// Threads that share sums by shuffles, on every backend.
pub const LANES: usize = 32;

pub struct Gen {
    entry: String,
    buffers: Vec<(String, &'static str, bool)>,
    scalars: Vec<(&'static str, &'static str)>,
    lines: Vec<String>,
    indent: usize,
    /// A loop trip's cost. Code outside loops goes uncounted.
    trip: Ops,
    looping: usize,
    lane: bool,
    names: usize,
}

impl Gen {
    pub fn new(entry: impl Into<String>) -> Self {
        Self {
            entry: entry.into(),
            buffers: Vec::new(),
            scalars: Vec::new(),
            lines: Vec::new(),
            indent: 1,
            trip: Ops::default(),
            looping: 0,
            lane: false,
            names: 0,
        }
    }

    pub fn buffer(&mut self, name: &str, ty: &'static str, writes: bool) {
        self.buffers.push((name.into(), ty, writes));
    }

    pub fn floats(&mut self, name: &str, writes: bool) {
        self.buffer(name, "float", writes);
    }

    pub fn scalar(&mut self, name: &'static str, ty: &'static str) {
        self.scalars.push((name, ty));
    }

    /// A statement at the current depth.
    pub fn line(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.lines
            .push(format!("{}{text}", "    ".repeat(self.indent)));
    }

    /// A fresh variable name.
    pub fn fresh(&mut self, prefix: &str) -> String {
        self.names += 1;
        format!("{prefix}{}", self.names)
    }

    fn count(&mut self, ops: Ops) {
        if self.looping == 0 {
            return;
        }
        let t = &mut self.trip;
        t.float += ops.float;
        t.int += ops.int;
        t.special += ops.special;
        t.loads += ops.loads;
        t.stores += ops.stores;
        t.tile_flops += ops.tile_flops;
        // Loads of one trip share one width and freshness.
        if ops.loads > 0.0 {
            t.width = ops.width;
            t.fresh = ops.fresh;
        }
    }

    /// Threads whose `cond` holds leave at once.
    pub fn guard(&mut self, cond: &str) {
        self.line(format!("if ({cond}) {{"));
        self.line("    return;");
        self.line("}");
    }

    /// This thread's lane among the 32 that shuffle together.
    pub fn lane(&mut self) -> &'static str {
        if !self.lane {
            self.line(format!("u32 lane = gid % {LANES};"));
            self.lane = true;
        }
        "lane"
    }

    /// `open` and a brace, `body` indented inside, then the close.
    pub fn block(&mut self, open: &str, body: impl FnOnce(&mut Gen)) {
        self.line(format!("{open} {{"));
        self.indent += 1;
        body(self);
        self.indent -= 1;
        self.line("}");
    }

    /// `var` from `from` below `to` by `step`, `ahead` trips unrolled.
    pub fn repeat(
        &mut self,
        var: &str,
        from: &str,
        to: &str,
        step: &str,
        ahead: usize,
        body: impl FnOnce(&mut Gen),
    ) {
        if ahead > 1 {
            self.line(format!("#pragma unroll {ahead}"));
        }
        self.line(format!(
            "for (u32 {var} = {from}; {var} < {to}; {var} += {step}) {{"
        ));
        self.indent += 1;
        self.looping += 1;
        self.count(Ops {
            int: 2.0,
            ..Ops::default()
        });
        body(self);
        self.looping -= 1;
        self.indent -= 1;
        self.line("}");
    }

    /// Four consecutive elements at `ptr`, as floats, in one load.
    pub fn four(&mut self, dtype: DType, ptr: &str) -> [String; 4] {
        let v = self.fresh("v");
        let (width, parts) = match dtype {
            DType::Bf16 => {
                self.line(format!("uint2 {v} = f_u32x2({ptr});"));
                let low = |h: &str| format!("f_bits({v}.{h} << 16)");
                let high = |h: &str| format!("f_bits({v}.{h} & 0xffff0000u)");
                (8.0, [low("x"), high("x"), low("y"), high("y")])
            }
            _ => {
                self.line(format!("float4 {v} = f_f32x4({ptr});"));
                (16.0, ["x", "y", "z", "w"].map(|h| format!("{v}.{h}")))
            }
        };
        self.count(Ops {
            loads: 1.0,
            fresh: 1.0,
            width,
            ..Ops::default()
        });
        parts
    }

    /// Writes four floats at `ptr` in one store.
    pub fn store4(&mut self, ptr: &str, value: &str) {
        self.line(format!("f_f32x4({ptr}) = {value};"));
        self.count(Ops {
            stores: 1.0,
            ..Ops::default()
        });
    }

    /// One line of float math, `floats` ops of it.
    pub fn arith(&mut self, line: impl Into<String>, floats: f64) {
        self.line(line);
        self.count(Ops {
            float: floats,
            ..Ops::default()
        });
    }

    pub fn fma(&mut self, acc: &str, a: &str, b: &str) {
        self.line(format!("{acc} += {a} * {b};"));
        self.count(Ops {
            float: 1.0,
            ..Ops::default()
        });
    }

    /// Every lane's `var` becomes the sum over the 32 lanes.
    pub fn sum32(&mut self, var: &str) {
        let mut m = LANES / 2;
        while m > 0 {
            self.line(format!("{var} += f_shfl_xor({var}, {m});"));
            m /= 2;
        }
        self.count(Ops {
            float: 5.0,
            int: 5.0,
            ..Ops::default()
        });
    }

    /// A warp's zeroed block of `rows` by `cols` sums.
    pub fn tile_acc(&mut self, name: &str, rows: usize, cols: usize) {
        let (tm, tn) = (format!("({rows}) / TILE_M"), format!("({cols}) / TILE_N"));
        self.line(format!("tile_acc {name}_t[{tm}][{tn}];"));
        self.line(format!("float {name}[{tm}][{tn}][TILE_COUNT];"));
        self.line("#pragma unroll");
        self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
        self.line("    #pragma unroll");
        self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
        self.line("        #pragma unroll");
        self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {");
        self.line(format!("            {name}[i][j][e] = 0.0f;"));
        self.line("        }");
        self.line("    }");
        self.line("}");
    }

    /// Adds `depth` staged columns of A times B to `acc`.
    pub fn tile_block(&mut self, acc: &str, a: &str, b: &str, stride: &str, dims: [usize; 3]) {
        let [rows, cols, depth] = dims;
        let (tm, tn) = (format!("({rows}) / TILE_M"), format!("({cols}) / TILE_N"));
        self.line("#pragma unroll");
        self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
        self.line("    #pragma unroll");
        self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
        self.line(format!("        TILE_ZERO({acc}_t[i][j]);"));
        self.line("    }");
        self.line("}");
        self.line("#pragma unroll");
        self.line(format!("for (u32 kk = 0; kk < {depth}; kk += TILE_K) {{"));
        self.line(format!("    tile_a ta[{tm}];"));
        self.line(format!("    tile_b tb[{tn}];"));
        self.line("    #pragma unroll");
        self.line(format!("    for (u32 i = 0; i < {tm}; i++) {{"));
        self.line(format!(
            "        TILE_LOAD_A(ta[i], ({a}) + i * TILE_M * ({stride}) + kk, {stride});"
        ));
        self.line("    }");
        self.line("    #pragma unroll");
        self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
        self.line(format!(
            "        TILE_LOAD_B(tb[j], ({b}) + j * TILE_N * ({stride}) + kk, {stride});"
        ));
        self.line("    }");
        self.line("    #pragma unroll");
        self.line(format!("    for (u32 i = 0; i < {tm}; i++) {{"));
        self.line("        #pragma unroll");
        self.line(format!("        for (u32 j = 0; j < {tn}; j++) {{"));
        self.line(format!(
            "            TILE_MMA({acc}_t[i][j], ta[i], tb[j]);"
        ));
        self.line("        }");
        self.line("    }");
        self.line("}");
        // The unit's sums truncate, so blocks add up in floats.
        self.line("#pragma unroll");
        self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
        self.line("    #pragma unroll");
        self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
        self.line("        #pragma unroll");
        self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {");
        self.line(format!(
            "            {acc}[i][j][e] += TILE_VAL({acc}_t[i][j], e);"
        ));
        self.line("        }");
        self.line("    }");
        self.line("}");
        // A lane's share of the block's multiply-adds, as tile flops.
        let flops = (2 * rows * cols * depth) as f64 / LANES as f64;
        self.count(Ops {
            tile_flops: flops,
            float: (rows * cols) as f64 / LANES as f64,
            loads: ((rows + cols) * depth) as f64 / LANES as f64,
            width: 4.0,
            ..Ops::default()
        });
    }

    /// Each sum this lane holds: `r`, `c`, `v`.
    pub fn tile_each(&mut self, acc: &str, rows: usize, cols: usize, body: impl FnOnce(&mut Gen)) {
        self.line("#pragma unroll");
        self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
        self.line(format!(
            "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
        ));
        self.line("        #pragma unroll");
        self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {");
        self.line("            u32 r = i * TILE_M + TILE_ROW(e);");
        self.line("            u32 c = j * TILE_N + TILE_COL(e);");
        self.line(format!("            float v = {acc}[i][j][e];"));
        self.indent += 3;
        body(self);
        self.indent -= 3;
        self.line("        }");
        self.line("    }");
        self.line("}");
    }

    /// `1 / sqrt(expr)`, one special op.
    pub fn rsqrt(&mut self, expr: &str) -> String {
        self.count(Ops {
            special: 1.0,
            ..Ops::default()
        });
        format!("f_rsqrt({expr})")
    }

    pub fn trip(&self) -> Ops {
        self.trip
    }

    pub fn finish(self) -> Text {
        Text {
            entry: self.entry,
            buffers: self.buffers,
            scalars: self.scalars,
            body: self.lines.join("\n") + "\n",
        }
    }
}
