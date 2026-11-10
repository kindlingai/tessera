//! Causal attention that walks the keys one at a time.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::kernels::{Join, Map, MapStage};
use crate::plan::Step;
use crate::rule::{Attn, Draft, Read, Site};
use crate::tree::{T, Tree};
use graph::Order;
use ops_model::gqa_attention;

/// Reads scaled queries, and keys joined with values.
#[derive(Clone, Copy, Debug)]
pub struct Attend(pub Attn);

impl Kernel for Attend {
    fn name(&self) -> &'static str {
        "attend"
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn sizes(&self) -> Vec<usize> {
        self.0.sizes()
    }

    fn source(&self) -> Source {
        use gqa_attention::{CUDA, ENTRY, METAL};
        Source::Fixed {
            metal: METAL,
            cuda: CUDA,
            entry: ENTRY,
        }
    }

    /// One launch per sequence, each at its own offset.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let shape = gqa_attention::Fixture {
            q: Vec::new(),
            kv: Vec::new(),
            seq_len: a.queries,
            total_kv: a.keys,
            n_heads: a.heads,
            n_kv_heads: a.kv_heads,
            head_dim: a.width,
            sliding_window: 0,
            kv_cache_len: a.keys - a.queries,
        };
        let (q, kv) = (
            a.queries * a.heads * a.width,
            a.keys * a.kv_heads * a.width * 2,
        );
        let launch = |s: usize| Launch {
            args: vec![
                Arg::Buffer(step.reads[0], s * q),
                Arg::Buffer(step.reads[1], s * kv),
                Arg::Buffer(step.write, s * q),
                Arg::bytes(&shape.params()),
                Arg::pair(a.queries, a.heads),
            ],
            shape: Shape::Rows(a.queries * a.heads),
        };
        (0..a.seqs).map(launch).collect()
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * a.queries * a.keys * a.width) as f64
    }
}

/// Scale the queries, join keys with values, then attend.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Attend {
        scale,
        q,
        k,
        v,
        sizes,
    } = tree.site(at)?
    else {
        return None;
    };
    let side = sizes.kv_heads * sizes.width;
    let out = sizes.seqs * sizes.queries * sizes.heads * sizes.width;
    let join = Join {
        left: side,
        right: side,
    };
    Some(vec![
        Draft::new(Map(vec![MapStage::Scale(scale)]), [q], out),
        Draft::new(join, [k, v], 2 * sizes.seqs * sizes.keys * side),
        Draft::new(Attend(sizes), [Read::Draft(0), Read::Draft(1)], out),
    ])
}

crate::rule!("attend", propose);
