//! The hand-written tiled matmul.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use graph::Order;
use sim::{Load, Ops};

/// `x[m, k]` times `w[n, k]` transposed.
#[derive(Clone, Copy, Debug)]
pub struct Matmul {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub(crate) fn source() -> Source {
    Source::Fixed {
        metal: gemm::METAL,
        cuda: gemm::CUDA,
        entry: gemm::ENTRY,
    }
}

/// A count rounded up to whole 64-wide tiles.
pub(crate) fn tiles(n: usize) -> usize {
    n.div_ceil(64) * 64
}

/// One tiled launch on buffers `a`, `b` and `c`.
pub(crate) fn launch(p: gemm::Problem, buffers: [Arg; 3]) -> Launch {
    let shape = Shape::Grid {
        across: p.n.div_ceil(gemm::BN),
        down: p.m.div_ceil(gemm::BM),
        threads: gemm::THREADS as usize,
    };
    let params = Arg::Bytes(gemm::abi_params(&p));
    let args = Vec::from(buffers).into_iter().chain([params]);
    Launch {
        args: args.collect(),
        shape,
    }
}

impl Kernel for Matmul {
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    /// Each group stages 16 of `k`, then every thread multiplies.
    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m as f64, self.n as f64, self.k as f64);
        let groups = self.m.div_ceil(gemm::BM) * self.n.div_ceil(gemm::BN);
        let threads = gemm::THREADS as usize;
        let per_thread = |outputs: usize| (outputs * gemm::BK / threads) as f64;
        let trip = Ops {
            float: per_thread(gemm::BM * gemm::BN),
            loads: per_thread(gemm::BM + gemm::BN),
            fresh: 1.0 / 16.0,
            ..Ops::default()
        };
        Some(Load {
            threads: groups * threads,
            group: threads,
            trips: k / gemm::BK as f64,
            trip,
            reads: vec![4.0 * n * k, 4.0 * m * k],
            written: 4.0 * m * n,
            flight: 64.0,
        })
    }

    fn source(&self) -> Source {
        source()
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let p = gemm::Problem::linear(self.m, self.n, self.k);
        vec![launch(p, [step.read(0), step.read(1), step.out()])]
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// The tiled kernel pays for whole tiles.
    fn work(&self, _: usize) -> f64 {
        (tiles(self.m) * tiles(self.n) * self.k) as f64
    }
}

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Matmul { x, w, m, n, k } = *site else {
        return None;
    };
    Some(vec![Draft::new(Matmul { m, n, k }, [x, w], m * n)])
}

crate::rule!("matmul", propose);
