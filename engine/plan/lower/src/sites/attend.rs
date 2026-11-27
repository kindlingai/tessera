use super::Pattern;
use crate::extent::Extent;
use crate::form::{Load, plain, writable};
use crate::kernel::Kernel;
use crate::kernels::Map;
use crate::rule::{Attn, Draft, Read};
use crate::tree::{T, Term, Tree};
use tessera_graph::{DimName, Op};

/// Attention with `scale` on the scores.
#[derive(Clone, Copy, Debug)]
pub struct Attend {
    pub scale: f32,
    pub q: T,
    pub k: T,
    pub v: T,
    pub q_form: &'static dyn Load,
    pub sizes: Attn,
    /// Ragged rows: each query's slot and position.
    pub rows: Option<[T; 2]>,
    /// Packed variable-length rows: `[Rows, 2]` segment bounds.
    pub packed: Option<T>,
}

impl Pattern for Attend {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        let Term::Op(Op::Attend(a)) = &node.term else {
            return None;
        };
        let wide = |t: T| &tree.node(t).shape;
        let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
        // Kernels read one head width for all three.
        if wide(v).last() != wide(q).last() {
            return None;
        }
        if a.packed() {
            let (&[queries, heads, width], &[_, kv_heads, _]) = (&wide(q)[..], &wide(k)[..]) else {
                return None;
            };
            let keys = tree.node(k).extents[0];
            let sizes = Attn {
                seqs: 1,
                queries,
                keys,
                heads,
                kv_heads,
                width,
                context: keys.most(),
                kept: tree.kept,
                segments: 1,
                mask: a.mask,
                k_form: plain(tree.node(k).form),
                v_form: plain(tree.node(v).form),
                out_form: writable(node.form),
            };
            return Some(Self {
                scale: a.scale,
                q,
                k,
                v,
                q_form: plain(tree.node(q).form),
                sizes,
                rows: None,
                packed: Some(node.args[3]),
            });
        }
        // Ragged kernels read every key, not a list.
        if a.sinks || a.rel_bias || a.ragged && a.picked {
            return None;
        }
        if a.ragged {
            let (&[queries, heads, width], &[seqs, _, kv_heads, _]) = (&wide(q)[..], &wide(k)[..])
            else {
                return None;
            };
            // The kept rows keep their layout and slot bound.
            let Term::Op(Op::Append(index)) = tree.node(k).term else {
                return None;
            };
            let (_, kept, context) = tree.kept_layout(index)?;
            if let Term::Op(Op::Append(vi)) = tree.node(v).term {
                let (_, vk, vc) = tree.kept_layout(vi)?;
                if vk != kept || vc != context {
                    return None;
                }
            }
            // Slots the rows span, if the launch says. Else any.
            let segments = tree.sizes.iter().find(|(n, _)| *n == DimName::Segments);
            let segments = segments.map_or(queries, |(_, n)| *n as usize).min(queries);
            let sizes = Attn {
                seqs,
                queries,
                keys: tree.node(k).extents[1],
                heads,
                kv_heads,
                width,
                context,
                kept,
                segments,
                mask: a.mask,
                k_form: plain(tree.node(k).form),
                v_form: plain(tree.node(v).form),
                out_form: writable(node.form),
            };
            return Some(Self {
                scale: a.scale,
                q,
                k,
                v,
                q_form: plain(tree.node(q).form),
                sizes,
                rows: Some([node.args[3], node.args[4]]),
                packed: None,
            });
        }
        if a.picked {
            return None;
        }
        Self::whole(tree, at)
    }
}

impl Attend {
    /// Attention over listed keys, and the node listing those keys.
    pub fn listed(tree: &Tree, at: T) -> Option<(Self, T)> {
        let node = tree.node(at);
        let Term::Op(Op::Attend(a)) = &node.term else {
            return None;
        };
        let site = Self::whole(tree, at).filter(|_| a.picked && !a.ragged)?;
        Some((site, node.args[3]))
    }

    /// Whole sequences, keys and queries ending together.
    fn whole(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        let Term::Op(Op::Attend(a)) = &node.term else {
            return None;
        };
        if a.sinks || a.rel_bias {
            return None;
        }
        let wide = |t: T| &tree.node(t).shape;
        let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
        if wide(v).last() != wide(q).last() {
            return None;
        }
        let (&[seqs, queries, heads, width], &[_, _, kv_heads, _]) = (&wide(q)[..], &wide(k)[..])
        else {
            return None;
        };
        let keys = tree.node(k).extents[1];
        // Whole sequences read the kept rows as they lie, unbroken.
        if tree.kept.run(keys.most()) < keys.most() {
            return None;
        }
        let sizes = Attn {
            seqs,
            queries,
            keys,
            heads,
            kv_heads,
            width,
            context: keys.most(),
            kept: tree.kept,
            segments: 1,
            mask: a.mask,
            k_form: plain(tree.node(k).form),
            v_form: plain(tree.node(v).form),
            out_form: writable(node.form),
        };
        Some(Self {
            scale: a.scale,
            q,
            k,
            v,
            q_form: plain(tree.node(q).form),
            sizes,
            rows: None,
            packed: None,
        })
    }

    /// The output's element count.
    pub fn out(&self) -> usize {
        self.sizes.queries * self.sizes.heads * self.sizes.width
    }

    /// Scales `q`, then runs `kernel` over packed bounds.
    pub fn packed_draft(&self, kernel: impl Kernel + 'static) -> Option<Vec<Draft>> {
        let bounds = self.packed?;
        let out = self.out();
        let scale = Map::scaled(self.scale, self.q_form);
        let reads = [
            Read::Draft(0),
            Read::Node(self.k),
            Read::Node(self.v),
            bounds.into(),
        ];
        Some(vec![
            Draft::new(scale, [self.q], out),
            Draft::new(kernel, reads, out),
        ])
    }

    /// For ragged rows: scale `q`, run `kernel` over kept keys.
    pub fn ragged(&self, kernel: impl Kernel + 'static) -> Option<Vec<Draft>> {
        let [slots, positions] = self.rows?;
        let out = self.out();
        let scale = Map::scaled(self.scale, self.q_form);
        let reads = [
            Read::Draft(0),
            Read::Node(self.k),
            Read::Node(self.v),
            slots.into(),
            positions.into(),
        ];
        Some(vec![
            Draft::new(scale, [self.q], out),
            Draft::new(kernel, reads, out),
        ])
    }
}

/// The gated delta rule, whole or ragged with kept state.
#[derive(Clone, Copy, Debug)]
pub struct Delta {
    pub scale: f32,
    /// Queries, keys, values, log decays and write rates.
    pub reads: [T; 5],
    /// Kept state index and `[slots, positions]` when ragged.
    pub state: Option<(u32, [T; 2])>,
    pub seqs: usize,
    /// Sequence or batch rows, runtime in dynamic plans.
    pub rows: Extent,
    /// Value heads. Each key head serves a contiguous group.
    pub heads: usize,
    pub kv_heads: usize,
    pub key_width: usize,
    pub width: usize,
}

impl Pattern for Delta {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        let Term::Op(Op::Delta(d)) = &node.term else {
            return None;
        };
        if let Some(idx) = d.state {
            let [q_t, k_t, v_t, dec_t, rate_t, slots, positions] = node.args[..] else {
                return None;
            };
            let (q, v) = (tree.node(q_t), tree.node(v_t));
            let (&[_, kv_heads, key_width], &[_, heads, width]) = (&q.shape[..], &v.shape[..])
            else {
                return None;
            };
            let seqs = *tree.fixed_kept(idx)?.first()?;
            return Some(Self {
                scale: d.scale,
                reads: [q_t, k_t, v_t, dec_t, rate_t],
                state: Some((idx, [slots, positions])),
                seqs,
                rows: q.extents[0],
                heads,
                kv_heads,
                key_width,
                width,
            });
        }
        let reads: [T; 5] = node.args[..].try_into().ok()?;
        let (q, v) = (tree.node(reads[0]), tree.node(reads[2]));
        // An extent holds one runtime dimension: the rows.
        if q.extents[0].runtime() {
            return None;
        }
        let rows = q.extents[1];
        let (&[seqs, _, kv_heads, key_width], &[_, _, heads, width]) = (&q.shape[..], &v.shape[..])
        else {
            return None;
        };
        Some(Self {
            scale: d.scale,
            reads,
            state: None,
            seqs,
            rows,
            heads,
            kv_heads,
            key_width,
            width,
        })
    }
}

impl Delta {
    /// Output elements at the most rows.
    pub fn out(&self) -> usize {
        let count = match self.state {
            Some(_) => self.rows.most(),
            None => self.seqs * self.rows.most(),
        };
        count * self.heads * self.width
    }
}
