//! Whole-graph checks: references, order and recorded types.

use crate::dims::Dim;
use crate::infer::{ErrorKind, TypeError, elements, infer, need, same_shape};
use crate::json::VERSION;
use crate::node::Graph;
use crate::types::{Placement, TensorType};
use std::fmt;

/// A failed check, with the node it concerns.
#[derive(Clone, Debug, PartialEq, Eq)]
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

impl std::error::Error for VerifyError {}

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
    let ports = g
        .entries
        .iter()
        .flat_map(|e| e.inputs.iter().map(|p| &p.ty));
    for ty in g.weights.iter().map(|w| &w.ty).chain(ports) {
        declared(&g.dims, ty).map_err(top)?;
    }
    for (i, n) in g.nodes.iter().enumerate() {
        let at = |error| VerifyError {
            node: Some(i as u32),
            error,
        };
        let mut args = Vec::with_capacity(n.args.len());
        for a in &n.args {
            need((a.0 as usize) < i, ErrorKind::Reference, || {
                format!("argument {} is not an earlier node", a.0)
            })
            .map_err(at)?;
            args.push(&g.nodes[a.0 as usize].ty);
        }
        let ty = infer(g, &n.op, &args).map_err(at)?;
        agree(&g.dims, &ty, &n.ty).map_err(at)?;
    }
    for o in g.entries.iter().flat_map(|e| &e.outputs) {
        need(
            (o.node.0 as usize) < g.nodes.len(),
            ErrorKind::Reference,
            || format!("output names missing node {}", o.node.0),
        )
        .map_err(top)?;
    }
    Ok(())
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
    need(
        same_shape(dims, &inferred.shape, &recorded.shape)?,
        ErrorKind::Shape,
        || {
            format!(
                "recorded {} but the op gives {}",
                recorded.shape, inferred.shape
            )
        },
    )?;
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
