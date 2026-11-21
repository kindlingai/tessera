//! Writes the chosen covers as ops over values.

use crate::Error;
use crate::Unsupported;
use crate::form::{self, Load};
use crate::kept;
use crate::kernel::{Kernel, KernelRef};
use crate::kernels::{Move, Narrow, Scatter, Widen};
use crate::rule::Draft;
use crate::select::cover::EDGE;
use crate::select::{Home, Op, Selected, Val, Value};
use crate::tree::{T, Tree};
use std::collections::HashMap;
use tessera_graph::NodeId;

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
    /// Each kept index's rows, once made.
    kept: HashMap<u32, Val>,
}

impl Emit {
    pub(crate) fn value(
        &mut self,
        len: usize,
        home: Home,
        zone: u32,
        form: &'static dyn Load,
    ) -> Val {
        self.out.values.push(Value {
            len,
            home,
            zone,
            form,
            share: None,
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
        let (len, form) = (self.len(v), self.out.values[v.0 as usize].form);
        let write = self.value(len, Home::Temp, zone, form);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Move),
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
        let form = self.out.values[v.0 as usize].form;
        if !form.narrow() || k.accepts(input, form) {
            return v;
        }
        self.widen_val(v)
    }

    /// `v` widened to f32, cached once per value.
    fn widen_val(&mut self, v: Val) -> Val {
        let Value {
            len, zone, form, ..
        } = self.out.values[v.0 as usize];
        if !form.narrow() {
            return v;
        }
        if let Some(wide) = self.widened.get(&v) {
            return *wide;
        }
        let write = self.value(len, Home::Temp, zone, form::F32_FORM);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Widen { len, form }),
            rule: "widen",
            reads: vec![v],
            write,
            covers: Vec::new(),
            zone,
        });
        self.widened.insert(v, write);
        write
    }

    /// `v` narrowed from f32 into `form`.
    fn narrow(&mut self, v: Val, form: &'static dyn Load) -> Val {
        let Value { len, zone, .. } = self.out.values[v.0 as usize];
        let write = self.value(len, Home::Temp, zone, form);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Narrow { len, form }),
            rule: "narrow",
            reads: vec![v],
            write,
            covers: Vec::new(),
            zone,
        });
        write
    }

    /// `v` converted to `want` if its form differs.
    pub(crate) fn as_form(&mut self, v: Val, want: &'static dyn Load) -> Val {
        let have = self.out.values[v.0 as usize].form;
        if have.name() == want.name() {
            return v;
        }
        let wide = self.widen_val(v);
        match want.narrow() {
            true => self.narrow(wide, want),
            false => wide,
        }
    }

    /// Adds an op writing a new value, or shares one.
    pub(crate) fn op(
        &mut self,
        d: Draft,
        rule: &'static str,
        reads: Vec<Val>,
        covers: &[NodeId],
        zone: u32,
        form: &'static dyn Load,
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
        let same_in = !d.kernel.overwrites()
            || reads
                .first()
                .is_some_and(|r| self.out.values[r.0 as usize].form.name() == form.name());
        let native = (!form.narrow() || d.kernel.emits(form)) && same_in;
        let out_form = if native { form } else { form::F32_FORM };
        let write = self.value(d.len, Home::Temp, zone, out_form);
        self.out.ops.push(Op {
            kernel: d.kernel,
            rule,
            reads,
            write,
            covers: covers.to_vec(),
            zone,
        });
        let result = match native {
            true => write,
            false => self.narrow(write, form),
        };
        if let Some(key) = key {
            self.shared.insert(key, result);
        }
        result
    }

    /// The kept rows of `index`, made once.
    pub(crate) fn kept(&mut self, tree: &Tree, index: u32) -> Result<Val, Error> {
        if let Some(val) = self.kept.get(&index) {
            return Ok(*val);
        }
        let (t, context) = tree.kept_rows(index).ok_or(Unsupported::KeptRows)?;
        let node = tree.node(t);
        let all = &node.shape;
        let width: usize = all[2..].iter().product();
        let room = tree.kept.rows(all[0], context) * width;
        let entries = kept::entries(tree.kept, all[0], context);
        self.out.table = self.out.table.max(entries);
        let len: usize = all.iter().product();
        let val = self.value(len, Home::Kept { index, room }, EDGE, node.form);
        self.kept.insert(index, val);
        Ok(val)
    }

    /// Writes new rows into kept rows by slot and position.
    pub(crate) fn append(
        &mut self,
        tree: &Tree,
        t: T,
        index: u32,
        args: [Val; 3],
    ) -> Result<Val, Error> {
        let write = self.kept(tree, index)?;
        let kernel = Scatter::at(tree, t, index).ok_or(Unsupported::KeptRows)?;
        let [rows, slots, positions] = args.map(|v| self.bring(v, EDGE));
        let rows = self.widen(rows, 0, &kernel);
        self.out.ops.push(Op {
            kernel: KernelRef::new(kernel),
            rule: "append",
            reads: vec![rows, slots, positions],
            write,
            covers: vec![tree.node(t).from],
            zone: EDGE,
        });
        Ok(write)
    }
}
