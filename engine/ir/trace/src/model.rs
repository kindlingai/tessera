//! The builder a description runs against.

use crate::tensor::Tensor;
use crate::weights::{Embed, Linears};
use graph::{
    Attend, Composite, Dim, DimName, Entry, EntryKind, Graph, Mask, Node, NodeId, Numerics, Op,
    Output, PathSeg, Port, Region, Role, Select, Size, TensorType, Weight, infer, verify,
};
use std::cell::RefCell;
use std::rc::Rc;

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

    /// A fixed dimension's size; panics on bounded ones.
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
        [q, k, v]: [&Tensor; 3],
        keys: Option<&Tensor>,
    ) -> Tensor {
        let picked = keys.is_some();
        let op = Op::Attend(Attend {
            mask,
            scale,
            picked,
        });
        let args: Vec<&Tensor> = [q, k, v].into_iter().chain(keys).collect();
        push(&self.g, op, &args)
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
        assert!(start < end, "{kind:?} traced no nodes");
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
        let mut shared = self.g.borrow_mut();
        let entry = shared.entries.last_mut().expect("finish needs an entry");
        for (role, t) in outputs {
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
