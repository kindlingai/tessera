//! A rotated matmul for few rows, in one launch.

use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_kernels::{BLOCK, spin, spin_inline};
use tessera_lower::form::Load as Stored;
use tessera_lower::sites::Rotated;
use tessera_lower::{Draft, Kernel, Load, Needs, Shape, Source, T, Tree, knob};

/// Its row limit: each thread keeps one sum per row.
const ROWS: usize = 8;

knob!("turned", "max_rows", 1, 16);

/// SIMD groups in a group, a block of `x` each.
const SIMDS: u32 = BLOCK / LANES as u32;

/// Inputs one pass stages: a block per SIMD group.
const SPAN: u32 = SIMDS * BLOCK;

/// Each group owns 128 outputs, a thread each.
#[kernel]
fn turned(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    suh: Buf<F32>,
    svh: Buf<F32>,
    mut out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] rows_cap: usize,
    #[build] form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    let four = lane() * (BLOCK / LANES as u32);
    let col = (gid() / BLOCK) * BLOCK + lid();
    let mut xs = shared::<f32>(rows_cap as u32 * SPAN);
    let mut sums = local::<f32>(rows_cap);
    for row in 0..m as u32 {
        sums[row as usize] = 0.0;
    }
    for k0 in (0..k).by(SPAN) {
        // Each SIMD group turns its block of each row.
        let at = k0 + simd * BLOCK + four;
        for row in 0..m as u32 {
            let to = row * SPAN + simd * BLOCK + four;
            let v = x.load4((row * k + at) as u64);
            let s = suh.load4(at as u64);
            let t = spin(v.x * s.x, v.y * s.y, v.z * s.z, v.w * s.w);
            xs[to] = t.a;
            xs[to + 1] = t.b;
            xs[to + 2] = t.c;
            xs[to + 3] = t.d;
        }
        barrier();
        for i in (0..SPAN).by(4) {
            let c = w.load4(col as u64 * k as u64 + (k0 + i) as u64);
            for row in 0..m as u32 {
                let at = row * SPAN + i;
                sums[row as usize] +=
                    xs[at] * c.x + xs[at + 1] * c.y + xs[at + 2] * c.z + xs[at + 3] * c.w;
            }
        }
        barrier();
    }
    // Each SIMD group turns every fourth row of the sums.
    for row in 0..m as u32 {
        xs[row * BLOCK + lid()] = sums[row as usize];
    }
    barrier();
    let first = (gid() / BLOCK) * BLOCK + four;
    let s = svh.load4(first as u64);
    for row in (simd..m as u32).by(SIMDS) {
        let from = row * BLOCK + four;
        let t = spin(xs[from], xs[from + 1], xs[from + 2], xs[from + 3]);
        let mut o = F32x4::zero();
        o.x = t.a * s.x;
        o.y = t.b * s.y;
        o.z = t.c * s.z;
        o.w = t.d * s.w;
        out.store4((row * n + first) as u64, o);
    }
}

/// `x[m, k]` through a rotated `w[n, k]`, in one launch.
#[derive(Clone, Debug, Kernel)]
#[launch(
    turned(self.m, self.rows_cap, self.form),
    name = self.form.suffixed("turned"),
    threads = Shape::Grid { across: self.n / BLOCK as usize, down: 1, threads: BLOCK as usize },
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = accepts,
    needs = turned_needs(self.rows_cap),
    order = Order::PerConfig,
    source = turned_source,
    load = turned_load,
    load_key = tessera_lower::load_key((self.m, self.n, self.k, self.rows_cap, self.form.name())),
    unproven,
)]
pub struct Turned {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub rows_cap: usize,
    pub form: &'static dyn Stored,
}

fn turned_source(t: &Turned) -> Source {
    let tag = if t.rows_cap == ROWS {
        String::new()
    } else {
        format!("_r{}", t.rows_cap)
    };
    Source::named(
        turned_ir(t.m, t.rows_cap, t.form),
        format!("plan_turned_{}_m{}{tag}", t.form.name(), t.m),
    )
}

fn accepts(t: &Turned, input: usize, form: &dyn Stored) -> bool {
    t.form.taken(input, 1, form)
}

fn turned_needs(rows_cap: usize) -> Needs {
    Needs {
        lanes: LANES,
        group_bytes: 4 * rows_cap * SPAN as usize,
        group_threads: BLOCK as usize,
        ..Needs::default()
    }
}

/// Each group reads `x`, `suh` and its rows of `w`.
fn turned_load(t: &Turned, _: usize) -> Option<Load> {
    let Turned {
        m,
        n,
        k,
        rows_cap,
        form,
    } = *t;
    let groups = (n / BLOCK as usize) as f64;
    Some(Load {
        threads: n,
        group: BLOCK as usize,
        trips: (k / SPAN as usize) as f64,
        trip: turned_ir(m, rows_cap, form).trip(),
        reads: vec![
            4.0 * (m * k) as f64 * groups,
            form.bytes(n * k) as f64,
            4.0 * k as f64 * groups,
            4.0 * n as f64,
        ],
        written: 4.0 * (m * n) as f64,
        flight: 64.0,
        ..Load::default()
    })
}

/// A rotated matmul at `at`.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let r = tree.site::<Rotated>(at).filter(|r| r.block == BLOCK)?;
    let (x, w, suh, svh) = (r.x, r.w, r.suh, r.svh);
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    let m = tree.elements(x) / k;
    let max_rows = tree.knobs.get("turned", "max_rows").unwrap_or(ROWS);
    let block = BLOCK as usize;
    let fits = n % block == 0 && k % SPAN as usize == 0 && m <= max_rows;
    if !fits || tree.elements(suh) != k || tree.elements(svh) != n {
        return None;
    }
    let form = tree.node(w).form;
    let rows_cap = m.max(ROWS);
    let kernel = Turned {
        m,
        n,
        k,
        rows_cap,
        form,
    };
    Some(vec![Draft::new(kernel, [x, w, suh, svh], m * n)])
}

tessera_lower::rule!("turned", propose);
