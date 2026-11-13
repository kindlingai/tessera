//! Writes the chosen covers as ops over values.

use crate::Error;
use crate::form::Stored;
use crate::kernel::{Kernel, KernelRef};
use crate::kernels::{Move, Scatter, Widen};
use crate::rule::Draft;
use crate::select::cover::EDGE;
use crate::select::{Home, Op, Selected, Val, Value};
use crate::tree::{T, Tree, unsupported};
use graph::{DType, DimExpr, NodeId};
use std::collections::HashMap;

/// The ops written, and the values they use.
#[derive(Default)]
pub(crate) struct Emit {
    pub(crate) out: Selected,
    /// Results that later equal launches may reuse.
    shared: HashMap<(String, Vec<Val>), Val>,
    /// Each value's copy in another zone, once made.
    moved: HashMap<(Val, u32), Val>,
    /// Each narrow value as floats, once made.
    widened: HashMap<Val, Val>,
}

impl Emit {
    pub(crate) fn value(&mut self, len: usize, home: Home, zone: u32, dtype: DType) -> Val {
        self.out.values.push(Value {
            len,
            home,
            zone,
            dtype,
        });
        Val(self.out.values.len() as u32 - 1)
    }

    fn len(&self, v: Val) -> usize {
        self.out.values[v.0 as usize].len
    }

    /// The value in `zone`, moved there if it lies elsewhere.
    pub(crate) fn bring(&mut self, v: Val, zone: u32) -> Val {
        if self.out.values[v.0 as usize].zone == zone {
            return v;
        }
        if let Some(copy) = self.moved.get(&(v, zone)) {
            return *copy;
        }
        let (len, dtype) = (self.len(v), self.out.values[v.0 as usize].dtype);
        let write = self.value(len, Home::Temp, zone, dtype);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Move { len }),
            rule: "move",
            reads: vec![v],
            write,
            covers: Vec::new(),
            zone,
        });
        self.moved.insert((v, zone), write);
        write
    }

    /// The value as floats, unless `k` reads it as stored.
    fn widen(&mut self, v: Val, input: usize, k: &dyn Kernel) -> Val {
        let Value {
            len, zone, dtype, ..
        } = self.out.values[v.0 as usize];
        if !dtype.narrow() || k.accepts(input, dtype) {
            return v;
        }
        if let Some(wide) = self.widened.get(&v) {
            return *wide;
        }
        let write = self.value(len, Home::Temp, zone, DType::F32);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Widen { len, form: dtype }),
            rule: "widen",
            reads: vec![v],
            write,
            covers: Vec::new(),
            zone,
        });
        self.widened.insert(v, write);
        write
    }

    /// Adds an op writing a new value, or shares one.
    pub(crate) fn op(
        &mut self,
        d: Draft,
        rule: &'static str,
        reads: Vec<Val>,
        covers: &[NodeId],
        zone: u32,
        dtype: DType,
    ) -> Val {
        let mut taken = Vec::with_capacity(reads.len());
        for (i, v) in reads.into_iter().enumerate() {
            let v = self.bring(v, zone);
            taken.push(self.widen(v, i, &*d.kernel));
        }
        let reads = taken;
        // Printing kernels is slow, so shared ones alone are keyed.
        let key = d
            .kernel
            .shared()
            .then(|| (format!("{:?}", d.kernel), reads.clone()));
        if let Some(same) = key.as_ref().and_then(|k| self.shared.get(k)) {
            return *same;
        }
        let write = self.value(d.len, Home::Temp, zone, dtype);
        if let Some(key) = key {
            self.shared.insert(key, write);
        }
        self.out.ops.push(Op {
            kernel: d.kernel,
            rule,
            reads,
            write,
            covers: covers.to_vec(),
            zone,
        });
        write
    }

    /// Writes new rows into kept rows by slot and position.
    pub(crate) fn append(
        &mut self,
        tree: &Tree,
        t: T,
        index: u32,
        args: [Val; 3],
    ) -> Result<Val, Error> {
        let (n, all) = (tree.node(t), &tree.node(t).shape);
        let DimExpr::Dim(axis) = &tree.g.kept[index as usize].shape.0[1] else {
            return unsupported("kept rows along a computed axis");
        };
        // A slot holds the axis's bound, whatever this launch sees.
        let most = tree.g.dims.iter().find(|d| d.name == *axis);
        let stride = most.map_or(0, |d| d.size.max() as usize);
        let width: usize = all[2..].iter().product();
        let room = all[0] * stride * width;
        let len: usize = all.iter().product();
        let write = self.value(len, Home::Kept { index, room }, EDGE, DType::F32);
        let [rows, slots, positions] = args.map(|v| self.bring(v, EDGE));
        let kernel = Scatter {
            rows: self.len(rows) / width,
            width,
            stride,
        };
        self.out.ops.push(Op {
            kernel: KernelRef::new(kernel),
            rule: "append",
            reads: vec![rows, slots, positions],
            write,
            covers: vec![n.from],
            zone: EDGE,
        });
        Ok(write)
    }
}
