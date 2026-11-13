//! A matmul for few rows, a warp per column.

use codegen::{Gen, LANES, hash};
use graph::{DType, Order};
use lower::kernels::{MapStage, operands, stage_lines};
use lower::{Arg, Draft, Kernel, Launch, Load, Read, Shape, Site, Source, Step, Stored, T, Tree};
use std::cmp::Ordering;

/// Elements one wide load brings a lane.
const WIDE: usize = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: usize = 4;

/// The most rows this schedule is offered for.
const ROWS: usize = 32;

/// Elementwise stages a sum runs through before it is stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct After {
    pub stages: Vec<MapStage>,
    /// Which of the chain's reads the sum stands in for.
    pub sum: usize,
}

/// `x[m, k]` times `w[n, k]` transposed, one warp per column.
#[derive(Clone, Debug)]
pub struct Lanes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: DType,
    pub after: After,
}

impl Lanes {
    fn others(&self) -> usize {
        match self.after.stages.is_empty() {
            true => 0,
            false => operands(&self.after.stages),
        }
    }

    /// The whole kernel: the loop, the reduce, then the stages.
    fn write(&self) -> Gen {
        let mut g = Gen::new("linear");
        g.floats("x", false);
        g.buffer("w", self.form.ty(), false);
        for a in 0..self.others() {
            g.floats(&format!("a{a}"), false);
        }
        g.floats("out", true);
        g.scalar("n", "u32");
        g.scalar("k", "u32");
        let lane = g.lane();
        g.line(format!("u32 col = gid / {LANES};"));
        g.guard("col >= n");
        g.line("u64 at = (u64)col * k;");
        for r in 0..self.m {
            g.line(format!("float s{r} = 0.0f;"));
        }
        let step = (LANES * WIDE).to_string();
        g.repeat("i", &format!("{lane} * {WIDE}"), "k", &step, AHEAD, |g| {
            let c = g.four(self.form, "w + at + i");
            for r in 0..self.m {
                let x = g.four(DType::F32, &format!("x + {r}u * k + i"));
                for (x, c) in x.iter().zip(&c) {
                    g.fma(&format!("s{r}"), x, c);
                }
            }
        });
        for r in 0..self.m {
            g.sum32(&format!("s{r}"));
        }
        g.guard("lane != 0");
        for r in 0..self.m {
            self.store(&mut g, r);
        }
        g
    }

    /// Row `r`'s sum through the stages, into its place.
    fn store(&self, g: &mut Gen, r: usize) {
        let After { stages, sum } = &self.after;
        let o = format!("((u64){r} * n + col)");
        if stages.is_empty() {
            return g.line(format!("out[{o}] = s{r};"));
        }
        // The chain's reads in order, the sum among them.
        let read = |j: usize| match j.cmp(sum) {
            Ordering::Equal => format!("s{r}"),
            Ordering::Less => format!("a{j}[{o}]"),
            Ordering::Greater => format!("a{}[{o}]", j - 1),
        };
        g.line(format!("{{ float v = {};", read(0)));
        for line in stage_lines(stages, &|j| read(j + 1)).lines() {
            g.line(line.trim_start().to_string());
        }
        g.line(format!("out[{o}] = v; }}"));
    }
}

impl Kernel for Lanes {
    fn name(&self) -> &'static str {
        match self.form {
            DType::Bf16 => "lanes_bf16",
            _ => "lanes",
        }
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// The lanes sum in one fixed tree on every device.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m, self.n, self.k);
        let bytes = self.form.bytes() * WIDE;
        Some(Load {
            threads: n * LANES,
            group: 256,
            trips: (k / (LANES * WIDE)) as f64,
            trip: self.write().trip(),
            reads: self.form.matmul_bytes(m, n, k),
            written: (4 * m * n) as f64,
            flight: (bytes * AHEAD) as f64,
        })
    }

    fn source(&self) -> Source {
        let mut text = self.write().finish();
        text.entry = format!("linear_{}_{:x}", self.form.name(), hash(&text.body));
        Source::Made(text)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.n, self.k].map(Arg::count));
        vec![Launch {
            args,
            shape: Shape::Line(self.n * LANES),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }

    fn accepts(&self, input: usize, form: DType) -> bool {
        self.form.taken(input, 1, form)
    }
}

/// The matmul at `at`, or the one its stages finish.
pub(crate) fn finishing(tree: &Tree, at: T) -> Option<(Site, After, Vec<Read>)> {
    match tree.site(at)? {
        product @ Site::Matmul { .. } => Some((product, After::default(), Vec::new())),
        Site::Map { stages, reads, .. } => {
            let lone = |r: &T| match tree.site(*r) {
                Some(product @ Site::Matmul { .. }) if !tree.root(*r) => Some(product),
                _ => None,
            };
            let sum = reads.iter().position(|r| lone(r).is_some())?;
            let others = reads.iter().filter(|r| **r != reads[sum]);
            let others = others.map(|r| Read::Node(*r)).collect();
            Some((lone(&reads[sum])?, After { stages, sum }, others))
        }
        _ => None,
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, after, others) = finishing(tree, at)?;
    let Site::Matmul {
        x,
        w,
        m,
        n,
        k,
        form,
    } = product
    else {
        return None;
    };
    // Wide loads need whole groups of four along `k`.
    if m > ROWS || !k.is_multiple_of(WIDE) {
        return None;
    }
    let lanes = Lanes {
        m,
        n,
        k,
        form,
        after,
    };
    let reads = [Read::Node(x), Read::Node(w)].into_iter().chain(others);
    Some(vec![Draft::new(lanes, reads, m * n)])
}

lower::rule!("lanes", propose);
