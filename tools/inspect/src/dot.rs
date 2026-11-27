//! A small Graphviz builder, and the graph drawn with it.

use crate::model::Model;
use std::fmt::Write;
use tessera_graph::{Graph, Op, Placement, Stage, TensorType, View};

type Attrs = Vec<(&'static str, String)>;

pub enum Stmt {
    Node {
        id: String,
        attrs: Attrs,
    },
    Edge {
        from: String,
        to: String,
    },
    Cluster {
        id: String,
        label: String,
        body: Vec<Stmt>,
    },
}

/// Graphviz text quotes. Line breaks become `\n`.
fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn render(stmts: &[Stmt], depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    for s in stmts {
        match s {
            Stmt::Node { id, attrs } => {
                let attrs: Vec<String> = attrs
                    .iter()
                    .map(|(k, v)| format!("{k}={}", quote(v)))
                    .collect();
                let _ = writeln!(out, "{pad}{} [{}];", quote(id), attrs.join(", "));
            }
            Stmt::Edge { from, to } => {
                let _ = writeln!(out, "{pad}{} -> {};", quote(from), quote(to));
            }
            Stmt::Cluster { id, label, body } => {
                let _ = writeln!(out, "{pad}subgraph {} {{", quote(&format!("cluster_{id}")));
                let _ = writeln!(out, "{pad}  label={}; style=rounded;", quote(label));
                render(body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
            }
        }
    }
}

/// The whole digraph, top to bottom.
pub fn digraph(name: &str, stmts: &[Stmt]) -> String {
    let mut out = format!("digraph {}", quote(name));
    out.push_str(" {");
    out.push('\n');
    out.push_str("  rankdir=TB;\n  node [fontname=\"Helvetica\", fontsize=10];\n");
    render(stmts, 1, &mut out);
    out.push_str("}\n");
    out
}

/// A stage's name without its fields.
fn stage(s: &Stage) -> String {
    let text = format!("{s:?}");
    let end = text.find([' ', '(']).unwrap_or(text.len());
    text[..end].to_string()
}

/// A weight's path, joined with dots.
pub fn path(g: &Graph, w: u32) -> String {
    let segs = g.weights[w as usize].path.iter().map(|s| match s {
        tessera_graph::PathSeg::Field(f) => f.clone(),
        tessera_graph::PathSeg::Index(i) => i.to_string(),
    });
    segs.collect::<Vec<_>>().join(".")
}

/// One line naming what a node does.
pub fn op_label(g: &Graph, op: &Op) -> String {
    match op {
        Op::Input { port, .. } => format!("input {:?}", g.entries[0].inputs[*port as usize].role),
        Op::Weight(w) => path(g, *w),
        Op::Row(r) => r.0.iter().map(stage).collect::<Vec<_>>().join(" · "),
        Op::View(View::Reshape(s)) => format!("reshape {s}"),
        Op::View(v) => format!("{v:?}").to_lowercase(),
        Op::Attend(a) => format!("attend {:?}", a.mask),
        Op::Append(k) => format!("append kept {k}"),
        other => format!("{other:?}").to_lowercase(),
    }
}

fn type_label(ty: &TensorType) -> String {
    let place = match ty.placement {
        Placement::Replicated => String::new(),
        p => format!(" {p:?}"),
    };
    format!("{} {:?}{place}", ty.shape, ty.dtype)
}

fn fill(ty: &TensorType, added: bool) -> &'static str {
    match (added, ty.placement) {
        (true, _) => "#f4a6a6",
        (_, Placement::Split { .. }) => "#a8d0f0",
        (_, Placement::Partial) => "#f6c88a",
        (_, Placement::Routed) => "#d4b3e8",
        (_, Placement::Owned { .. }) => "#b8e0b0",
        (_, Placement::Replicated) => "#ffffff",
    }
}

/// One graph node. Sharded nodes keep their source id.
fn node(model: &Model, sharded: bool, i: usize) -> Stmt {
    let g = if sharded {
        &model.sharded
    } else {
        &model.source
    };
    let n = &g.nodes[i];
    let (src, added) = match sharded {
        true => (model.origin[i].0 as usize, model.added[i]),
        false => (i, false),
    };
    let ty = match &n.op {
        Op::Weight(w) => &g.weights[*w as usize].ty,
        _ => &n.ty,
    };
    let tag = match added {
        true => format!("sum of #{src}"),
        false => format!("#{src}"),
    };
    let shape = match &n.op {
        Op::Weight(_) => "note",
        Op::Input { .. } => "ellipse",
        _ if added => "doubleoctagon",
        _ => "box",
    };
    let label = format!("{tag} {}\n{}", op_label(g, &n.op), type_label(ty));
    Stmt::Node {
        id: format!("n{i}"),
        attrs: vec![
            ("label", label),
            ("shape", shape.into()),
            ("style", "filled".into()),
            ("fillcolor", fill(ty, added).into()),
            ("URL", format!("lowering.html#n{src}")),
        ],
    }
}

/// The source graph, or its sharded form, as Graphviz text.
pub fn draw(model: &Model, sharded: bool) -> String {
    let g = if sharded {
        &model.sharded
    } else {
        &model.source
    };
    let mut stmts = Vec::new();
    let mut inside = vec![false; g.nodes.len()];
    for (k, r) in g.regions.iter().enumerate() {
        let (start, end) = (r.start.0 as usize, r.end.0 as usize);
        let body = (start..end).map(|i| node(model, sharded, i)).collect();
        inside[start..end].iter_mut().for_each(|x| *x = true);
        stmts.push(Stmt::Cluster {
            id: k.to_string(),
            label: format!("{:?}", r.kind),
            body,
        });
    }
    for i in (0..g.nodes.len()).filter(|i| !inside[*i]) {
        stmts.push(node(model, sharded, i));
    }
    for (i, n) in g.nodes.iter().enumerate() {
        for a in &n.args {
            stmts.push(Stmt::Edge {
                from: format!("n{}", a.0),
                to: format!("n{i}"),
            });
        }
    }
    for (k, o) in g.entries[0].outputs.iter().enumerate() {
        stmts.push(Stmt::Node {
            id: format!("out{k}"),
            attrs: vec![
                ("label", format!("{:?}", o.role)),
                ("shape", "ellipse".into()),
            ],
        });
        stmts.push(Stmt::Edge {
            from: format!("n{}", o.node.0),
            to: format!("out{k}"),
        });
    }
    digraph(&model.name, &stmts)
}
