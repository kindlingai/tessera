//! The builder a description runs against.

use crate::tensor::Tensor;
use crate::weights::{Embed, Linears};
use std::cell::RefCell;
use std::rc::Rc;
use tessera_graph::{
    Attend, Composite, Delta, Dim, DimName, Entry, EntryKind, Graph, Mask, Node, NodeId, Numerics,
    Op, Output, PathSeg, Port, Region, Role, Select, Size, TensorType, Weight, infer, verify,
};

/// Shared by every tensor the description creates.
pub(crate) type Shared = Rc<RefCell<Graph>>;

pub struct Model {
    g: Shared,
}

/// Rows a model keeps between launches.
pub struct Kept(pub(crate) u32);

impl Model {
    pub fn new(dims: Vec<Dim>) -> Self {
        Self {
            g: Rc::new(RefCell::new(Graph::new(dims))),
        }
    }

    /// A fixed dimension's size. Panics on bounded ones.
    pub fn size(&self, name: DimName) -> u64 {
        let g = self.g.borrow();
        match g.dims.iter().find(|d| d.name == name) {
            Some(Dim {
                size: Size::Fixed(n),
                ..
            }) => *n,
            other => panic!("{name:?} has no fixed size: {other:?}"),
        }
    }

    pub fn weight(&self, path: &[PathSeg], ty: TensorType) -> Tensor {
        let index = {
            let mut g = self.g.borrow_mut();
            g.weights.push(Weight {
                path: path.to_vec(),
                ty,
            });
            g.weights.len() as u32 - 1
        };
        push(&self.g, Op::Weight(index), &[])
    }

    /// Declares rows kept between launches, shaped `[Seqs, Kept, ..]`.
    pub fn kept(&self, ty: TensorType) -> Kept {
        let mut g = self.g.borrow_mut();
        g.kept.push(ty);
        Kept(g.kept.len() as u32 - 1)
    }

    /// Adds an entry and returns its input ports as tensors.
    pub fn entry<const N: usize>(&self, kind: EntryKind, ports: [Port; N]) -> [Tensor; N] {
        let entry = {
            let mut g = self.g.borrow_mut();
            g.entries.push(Entry {
                kind,
                inputs: ports.to_vec(),
                outputs: Vec::new(),
            });
            g.entries.len() as u32 - 1
        };
        std::array::from_fn(|i| {
            let port = i as u32;
            push(&self.g, Op::Input { entry, port }, &[])
        })
    }

    pub fn embed(&self, ids: &Tensor, table: &Embed) -> Tensor {
        push(&self.g, Op::Select(Select::Gather), &[&table.0, ids])
    }

    /// One `[.., Out, In]` matrix per index.
    pub fn pick(&self, set: &Linears, ids: &Tensor) -> Tensor {
        push(&self.g, Op::Select(Select::Gather), &[&set.0, ids])
    }

    /// With `keys`, each query reads only its listed keys.
    pub fn attend(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        keys: Option<&Tensor>,
    ) -> Tensor {
        self.attend_with(mask, scale, qkv, keys, None)
    }

    /// Whole-sequence attention with per-head sink logits `[Heads]`.
    pub fn attend_sinks(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        keys: Option<&Tensor>,
        sinks: &Tensor,
    ) -> Tensor {
        self.attend_with(mask, scale, qkv, keys, Some(sinks))
    }

    /// Whole-sequence attention with optional sinks and relative bias.
    pub fn attend_biased(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        keys: Option<&Tensor>,
        [sinks, rel_bias]: [Option<&Tensor>; 2],
    ) -> Tensor {
        self.attend_all(mask, scale, qkv, None, [keys, sinks, rel_bias])
    }

    fn attend_with(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        keys: Option<&Tensor>,
        sinks: Option<&Tensor>,
    ) -> Tensor {
        self.attend_all(mask, scale, qkv, None, [keys, sinks, None])
    }

    fn attend_all(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        by: Option<[&Tensor; 2]>,
        [keys, sinks, rel_bias]: [Option<&Tensor>; 3],
    ) -> Tensor {
        let op = Op::Attend(Attend {
            mask,
            scale,
            picked: keys.is_some(),
            ragged: by.is_some(),
            sinks: sinks.is_some(),
            rel_bias: rel_bias.is_some(),
        });
        let args: Vec<&Tensor> = qkv
            .into_iter()
            .chain(by.into_iter().flatten())
            .chain(keys)
            .chain(sinks)
            .chain(rel_bias)
            .collect();
        push(&self.g, op, &args)
    }

    /// Query rows by slot and position, or their listed keys.
    pub fn attend_rows(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        by: [&Tensor; 2],
        keys: Option<&Tensor>,
    ) -> Tensor {
        self.attend_rows_with(mask, scale, qkv, by, keys, None)
    }

    /// Ragged attention with per-head sink logits `[Heads]`.
    pub fn attend_rows_sinks(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        by: [&Tensor; 2],
        keys: Option<&Tensor>,
        sinks: &Tensor,
    ) -> Tensor {
        self.attend_rows_with(mask, scale, qkv, by, keys, Some(sinks))
    }

    /// Ragged attention with optional sinks and relative bias.
    pub fn attend_rows_biased(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        by: [&Tensor; 2],
        keys: Option<&Tensor>,
        [sinks, rel_bias]: [Option<&Tensor>; 2],
    ) -> Tensor {
        self.attend_all(mask, scale, qkv, Some(by), [keys, sinks, rel_bias])
    }

    fn attend_rows_with(
        &self,
        mask: Mask,
        scale: f32,
        qkv: [&Tensor; 3],
        by: [&Tensor; 2],
        keys: Option<&Tensor>,
        sinks: Option<&Tensor>,
    ) -> Tensor {
        self.attend_all(mask, scale, qkv, Some(by), [keys, sinks, None])
    }

    /// The gated delta rule, with log decays and write rates.
    pub fn delta(
        &self,
        scale: f32,
        [q, k, v]: [&Tensor; 3],
        [decay, rate]: [&Tensor; 2],
    ) -> Tensor {
        let op = Op::Delta(Delta { scale, state: None });
        push(&self.g, op, &[q, k, v, decay, rate])
    }

    /// Ragged gated delta rule updating kept state per slot.
    pub fn delta_rows(
        &self,
        scale: f32,
        state: &Kept,
        [q, k, v]: [&Tensor; 3],
        [decay, rate]: [&Tensor; 2],
        [slots, positions]: [&Tensor; 2],
    ) -> Tensor {
        let op = Op::Delta(Delta {
            scale,
            state: Some(state.0),
        });
        push(&self.g, op, &[q, k, v, decay, rate, slots, positions])
    }

    /// Marks the nodes `body` traces as one composite region.
    pub fn composite<const N: usize>(
        &self,
        kind: Composite,
        body: impl FnOnce() -> [Tensor; N],
    ) -> [Tensor; N] {
        let start = NodeId(self.g.borrow().nodes.len() as u32);
        let outs = body();
        let mut g = self.g.borrow_mut();
        let end = NodeId(g.nodes.len() as u32);
        assert!(start < end, "{kind:?} traced zero nodes");
        let outputs = outs.iter().map(|t| t.id).collect::<Vec<_>>();
        let within = outputs.iter().all(|&o| start <= o && o < end);
        assert!(within, "{kind:?} must return values it computed");
        let inputs = g.reads(start, end);
        g.regions.push(Region {
            kind,
            start,
            end,
            inputs,
            outputs,
        });
        drop(g);
        outs
    }

    /// Sets the last entry's outputs and returns the verified graph.
    pub fn finish<const N: usize>(&self, outputs: [(Role, &Tensor); N]) -> Graph {
        self.finish_many(&outputs)
    }

    /// Sets a slice of entry outputs and verifies the graph.
    pub fn finish_many(&self, outputs: &[(Role, &Tensor)]) -> Graph {
        let mut shared = self.g.borrow_mut();
        let entry = shared.entries.last_mut().expect("finish needs an entry");
        for &(role, t) in outputs {
            entry.outputs.push(Output { role, node: t.id });
        }
        let g = shared.clone();
        if let Err(e) = verify(&g) {
            panic!("traced graph fails verification: {e}");
        }
        g
    }
}

/// Appends a node with the strongest numerics. Typing errors panic.
pub(crate) fn push(g: &Shared, op: Op, args: &[&Tensor]) -> Tensor {
    let tys: Vec<&TensorType> = args.iter().map(|t| &t.ty).collect();
    let ty = match infer(&g.borrow(), &op, &tys) {
        Ok(ty) => ty,
        Err(e) => panic!("{op:?}: {e}"),
    };
    let id = {
        let mut graph = g.borrow_mut();
        graph.nodes.push(Node {
            numerics: op.reduces().then(Numerics::default),
            op,
            args: args.iter().map(|t| t.id).collect(),
            ty: ty.clone(),
        });
        NodeId(graph.nodes.len() as u32 - 1)
    };
    Tensor {
        id,
        ty,
        g: g.clone(),
    }
}
