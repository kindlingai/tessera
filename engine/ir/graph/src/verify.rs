//! Whole-graph checks: references, order and recorded types.

use crate::dims::{Dim, DimExpr, Poly, Size};
use crate::infer::{ErrorKind, TypeError, elements, infer, need, poly, same_shape};
use crate::json::VERSION;
use crate::node::{Graph, NodeId, Op};
use crate::numerics::{Numerics, sums_ranks};
use crate::regions;
use crate::types::{DType, Placement, TensorType};
use std::fmt;

/// A failed check, with the node it concerns.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub struct VerifyError {
    pub node: Option<u32>,
    pub error: TypeError,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node {
            Some(n) => write!(f, "node {n}: {}", self.error),
            None => write!(f, "graph: {}", self.error),
        }
    }
}

/// Checks every node against the typing rules.
pub fn verify(g: &Graph) -> Result<(), VerifyError> {
    let top = |error| VerifyError { node: None, error };
    need(g.version == VERSION, ErrorKind::Reference, || {
        format!("format version {} (want {VERSION})", g.version)
    })
    .map_err(top)?;
    for (i, d) in g.dims.iter().enumerate() {
        need(
            !g.dims[..i].iter().any(|o| o.name == d.name),
            ErrorKind::Reference,
            || format!("dimension {:?} declared twice", d.name),
        )
        .map_err(top)?;
    }
    for w in &g.weights {
        let fixed = elements(&g.dims, &w.ty.shape)
            .map_err(top)?
            .as_const()
            .is_some();
        need(fixed, ErrorKind::Shape, || {
            format!("weight {:?} uses a bounded dimension", w.path)
        })
        .map_err(top)?;
    }
    if !g.kept.is_empty() {
        let owners = owners(g);
        check_kept(g, &owners).map_err(top)?;
    }
    let ports = g
        .entries
        .iter()
        .flat_map(|e| e.inputs.iter().map(|p| &p.ty));
    for ty in g.weights.iter().map(|w| &w.ty).chain(&g.kept).chain(ports) {
        declared(&g.dims, ty).map_err(top)?;
    }
    let mut args = Vec::with_capacity(8);
    for (i, n) in g.nodes.iter().enumerate() {
        let at = |error| VerifyError {
            node: Some(i as u32),
            error,
        };
        args.clear();
        for a in &n.args {
            need((a.0 as usize) < i, ErrorKind::Reference, || {
                format!("argument {} is not an earlier node", a.0)
            })
            .map_err(at)?;
            args.push(&g.nodes[a.0 as usize].ty);
        }
        let ty = infer(g, &n.op, &args).map_err(at)?;
        agree(&g.dims, &ty, &n.ty).map_err(at)?;
        numerics(&n.op, n.numerics.as_ref()).map_err(at)?;
    }
    for o in g.entries.iter().flat_map(|e| &e.outputs) {
        need(
            (o.node.0 as usize) < g.nodes.len(),
            ErrorKind::Reference,
            || format!("output names missing node {}", o.node.0),
        )
        .map_err(top)?;
    }
    regions::check(g).map_err(top)
}

/// The entry whose inputs each node reads, if any.
fn owners(g: &Graph) -> Vec<Option<u32>> {
    let mut out: Vec<Option<u32>> = Vec::with_capacity(g.nodes.len());
    for n in &g.nodes {
        let own = match n.op {
            Op::Input { entry, .. } => Some(entry),
            _ => n
                .args
                .iter()
                .find_map(|a| out.get(a.0 as usize).copied().flatten()),
        };
        out.push(own);
    }
    out
}

type AppendKey = (Option<u32>, Option<NodeId>, Option<NodeId>);

#[derive(Clone, Default)]
struct KeptUse {
    first_append: Option<AppendKey>,
    more_appends: Vec<AppendKey>,
    first_state: Option<Option<u32>>,
    more_states: Vec<Option<u32>>,
    dup_append: bool,
    dup_state: bool,
}

fn record_key<T: Copy + PartialEq>(
    first: &mut Option<T>,
    more: &mut Vec<T>,
    dup: &mut bool,
    key: T,
) {
    match *first {
        None => *first = Some(key),
        Some(prev) if prev == key || more.contains(&key) => *dup = true,
        Some(_) => more.push(key),
    }
}

/// Checks all kept slots in one pass over nodes.
fn check_kept(g: &Graph, owners: &[Option<u32>]) -> Result<(), TypeError> {
    let mut uses = vec![KeptUse::default(); g.kept.len()];
    for (n, own) in g.nodes.iter().zip(owners) {
        if let Op::Append(idx) = n.op {
            if let Some(u) = uses.get_mut(idx as usize) {
                let key = (*own, n.args.get(1).copied(), n.args.get(2).copied());
                record_key(
                    &mut u.first_append,
                    &mut u.more_appends,
                    &mut u.dup_append,
                    key,
                );
            }
        } else if let Some(idx) = n.op.facts().state
            && let Some(u) = uses.get_mut(idx as usize)
        {
            record_key(
                &mut u.first_state,
                &mut u.more_states,
                &mut u.dup_state,
                *own,
            );
        }
    }
    for (i, (k, u)) in g.kept.iter().zip(&uses).enumerate() {
        need(!u.dup_append, ErrorKind::Reference, || {
            format!("kept rows {i} are appended to twice")
        })?;
        need(!u.dup_state, ErrorKind::Reference, || {
            format!("kept state {i} is updated twice")
        })?;
        need(
            u.first_append.is_none() || u.first_state.is_none(),
            ErrorKind::Reference,
            || format!("kept {i} mixes appends and state"),
        )?;
        if u.first_state.is_some() {
            let fixed = k.shape.0.len() >= 2 && fixed_tail(&g.dims, &k.shape.0[1..])?;
            need(fixed, ErrorKind::Shape, || {
                format!("kept state {} must have fixed tail", k.shape)
            })?;
            continue;
        }
        let grows = match k.shape.0.get(1) {
            Some(DimExpr::Dim(name)) => g.dims.iter().find(|d| d.name == *name),
            _ => None,
        };
        let bounded = grows.is_some_and(|d| matches!(d.size, Size::Bounded(_)));
        need(bounded, ErrorKind::Shape, || {
            format!("kept rows {} need a bounded second axis", k.shape)
        })?;
    }
    Ok(())
}

fn fixed_tail(dims: &[Dim], tail: &[DimExpr]) -> Result<bool, TypeError> {
    let mut acc = Poly::constant(1);
    for e in tail {
        acc = acc.product(&poly(dims, e)?);
    }
    Ok(acc.as_const().is_some())
}

fn numerics(op: &Op, n: Option<&Numerics>) -> Result<(), TypeError> {
    need(n.is_some() == op.reduces(), ErrorKind::Numerics, || {
        let want = if op.reduces() { "needs" } else { "takes no" };
        format!("{op:?} {want} numerics")
    })?;
    let acc = n.map(|n| n.accumulate);
    need(
        acc.is_none_or(|a| matches!(a, DType::F32 | DType::Bf16 | DType::F16)),
        ErrorKind::Numerics,
        || format!("accumulator {acc:?} is not F32, Bf16 or F16"),
    )?;
    let Some(Numerics {
        accumulate,
        wire: Some(wire),
        ..
    }) = n.copied()
    else {
        return Ok(());
    };
    need(sums_ranks(op), ErrorKind::Numerics, || {
        format!("{op:?} sums on one rank but has a wire dtype")
    })?;
    let carried = matches!(wire, DType::F32 | DType::Bf16 | DType::F16 | DType::Fp8)
        && wire.bytes() <= accumulate.bytes();
    need(carried, ErrorKind::Numerics, || {
        format!("wire {wire:?} must be F32, Bf16, F16 or Fp8 and no wider than {accumulate:?}")
    })
}

fn declared(dims: &[Dim], t: &TensorType) -> Result<(), TypeError> {
    elements(dims, &t.shape)?;
    if let Placement::Split { axis } = t.placement {
        need((axis as usize) < t.rank(), ErrorKind::Placement, || {
            format!("split axis {axis} is outside {}", t.shape)
        })?;
    }
    Ok(())
}

fn agree(dims: &[Dim], inferred: &TensorType, recorded: &TensorType) -> Result<(), TypeError> {
    let shape_ok =
        inferred.shape == recorded.shape || same_shape(dims, &inferred.shape, &recorded.shape)?;
    need(shape_ok, ErrorKind::Shape, || {
        format!(
            "recorded {} but the op gives {}",
            recorded.shape, inferred.shape
        )
    })?;
    need(inferred.dtype == recorded.dtype, ErrorKind::DType, || {
        format!(
            "recorded {:?} but the op gives {:?}",
            recorded.dtype, inferred.dtype
        )
    })?;
    need(
        inferred.placement == recorded.placement,
        ErrorKind::Placement,
        || {
            format!(
                "recorded {:?} but the op gives {:?}",
                recorded.placement, inferred.placement
            )
        },
    )
}
