//! The lowering page: each stage and its effect per node.

use crate::dot::{op_label, path};
use crate::model::Model;
use crate::stages::{Pass, RankView};
use tessera_graph::{Op, Placement};
use tessera_html::{El, Part, el, table, text};

pub const STYLE: &str = include_str!("page.css");

fn kb(bytes: usize) -> String {
    format!("{:.1} KB", bytes as f64 / 1024.0)
}

fn us(seconds: f64) -> String {
    format!("{:.1} µs", seconds * 1e6)
}

/// A link to a source node's row.
fn node_link(id: u32) -> Part {
    text("a", format!("#{id}"))
        .attr("href", format!("#n{id}"))
        .into()
}

fn links(ids: &[u32]) -> Part {
    let mut span = el("span");
    for id in ids {
        span = span.kid(node_link(*id)).text(" ");
    }
    span.into()
}

pub fn head(title: &str) -> El {
    el("head")
        .kid(el("meta").attr("charset", "utf-8"))
        .kid(text("title", title))
        .kid(el("style").kid(Part::Raw(STYLE.into())))
}

fn ranks(pass: &Pass) -> El {
    let total: u32 = pass.ranks.iter().map(|r| r.part).sum();
    let rows = pass.ranks.iter().enumerate().map(|(i, r)| {
        let held: usize = r.held.iter().map(|h| h.bytes).sum();
        let time: f64 = r.steps.iter().map(|s| s.seconds).sum();
        let sums = r.steps.iter().filter(|s| s.sum).count();
        vec![
            format!("{i}").into(),
            r.name.clone().into(),
            format!("{} {}", r.backend, r.arch).into(),
            format!("{} of {total}", r.part).into(),
            format!("{}", r.steps.len()).into(),
            format!("{sums}").into(),
            kb(held).into(),
            kb(r.peak).into(),
            us(time).into(),
        ]
    });
    let head = [
        "rank", "machine", "backend", "part", "steps", "sums", "weights", "peak", "estimate",
    ];
    table(&head, rows)
}

fn stages(model: &Model, pass: &Pass) -> El {
    let (g, s) = (&model.source, &model.sharded);
    let added = model.added.iter().filter(|a| **a).count();
    let split = |p: fn(&Placement) -> bool| s.weights.iter().filter(|w| p(&w.ty.placement)).count();
    let mut items = vec![
        format!(
            "Front end: {} nodes, {} weights, {} regions.",
            g.nodes.len(),
            g.weights.len(),
            g.regions.len()
        ),
        format!(
            "Shard: {added} sums added, {} weights split by rows or columns, {} kept whole.",
            split(|p| matches!(p, Placement::Split { .. })),
            split(|p| *p == Placement::Replicated)
        ),
        format!(
            "Launch sizes: {} new rows at context {}.",
            pass.rows, pass.context
        ),
    ];
    for r in &pass.ranks {
        items.push(format!(
            "{}: the tree gives each node its local shape. Select picks {} launches ({} rules contested). Schedule keeps {} live. Assign gives {} of buffers and {} steps.",
            r.name,
            r.ops,
            r.contested.len(),
            kb(r.peak),
            kb(r.buffers),
            r.steps.len()
        ));
    }
    items.push(format!(
        "Kernels: {} distinct texts over all ranks.",
        pass.texts.len()
    ));
    items.push(
        "Estimates come from the cost model. With no timed samples, a fixed launch cost leads."
            .into(),
    );
    el("ol").kids(items.into_iter().map(|i| text("li", i)))
}

fn share(r: &RankView, w: u32) -> String {
    let Some(h) = r.held.iter().find(|h| h.weight == w) else {
        return "unused".into();
    };
    match h.share {
        Some([axis, first, count, whole]) => format!(
            "axis {axis}: {first}..{} of {whole}, {}",
            first + count,
            kb(h.bytes)
        ),
        None => format!("whole, {}", kb(h.bytes)),
    }
}

fn weights(model: &Model, pass: &Pass) -> El {
    let mut head = vec!["weight", "type"];
    head.extend(pass.ranks.iter().map(|r| r.name.as_str()));
    let rows = model.sharded.weights.iter().enumerate().map(|(w, weight)| {
        let mut row: Vec<Part> = vec![
            path(&model.sharded, w as u32).into(),
            format!("{} {:?}", weight.ty.shape, weight.ty.placement).into(),
        ];
        row.extend(pass.ranks.iter().map(|r| share(r, w as u32).into()));
        row
    });
    table(&head, rows)
}

/// Steps on `r` that cover source node `id`.
fn covering(r: &RankView, id: u32) -> String {
    let mut out = Vec::new();
    for (i, s) in r.steps.iter().enumerate() {
        if s.covers.contains(&id) {
            out.push(format!("s{i} {}", s.kernel));
        }
    }
    out.join(", ")
}

fn nodes(model: &Model, pass: &Pass) -> El {
    let mut head = vec!["node", "op", "type", "sharded"];
    for r in &pass.ranks {
        head.push(&r.name);
    }
    let mut rows = Vec::new();
    for (i, n) in model.source.nodes.iter().enumerate() {
        let id = i as u32;
        let mine: Vec<usize> = (0..model.origin.len())
            .filter(|s| model.origin[*s].0 == id)
            .collect();
        let main = mine.iter().copied().find(|s| !model.added[*s]);
        let placed = |s: usize| format!("{:?}", model.sharded.nodes[s].ty.placement);
        let mut sharded: Vec<String> = main.map(placed).into_iter().collect();
        if mine.iter().any(|s| model.added[*s]) {
            sharded.push("then summed over ranks".into());
        }
        let weight = matches!(n.op, Op::Weight(_));
        let mut row: Vec<Part> = vec![
            text("a", format!("#{i}"))
                .attr("id", format!("n{i}"))
                .into(),
            op_label(&model.source, &n.op).into(),
            format!("{} {:?}", n.ty.shape, n.ty.dtype).into(),
            sharded.join(", ").into(),
        ];
        for r in &pass.ranks {
            let shape = main.and_then(|s| r.shapes[s].clone());
            let cell = match (shape, weight) {
                (_, true) => String::new(),
                (Some(shape), false) => format!("{shape:?} {}", covering(r, id)),
                (None, false) => "not needed".into(),
            };
            row.push(cell.into());
        }
        rows.push(row);
    }
    table(&head, rows)
}

fn steps(r: &RankView) -> El {
    let rows = r.steps.iter().enumerate().map(|(i, s)| {
        let kernel = text("a", s.kernel).attr("href", format!("#k{}", s.text));
        let class = if s.sum { "sum" } else { "" };
        vec![
            text("span", format!("s{i}")).attr("class", class).into(),
            kernel.into(),
            s.rule.into(),
            format!("{}", s.zone).into(),
            links(&s.covers),
            kb(s.bytes).into(),
            us(s.seconds).into(),
        ]
    });
    let head = [
        "step", "kernel", "rule", "zone", "covers", "writes", "estimate",
    ];
    let summary = format!("{} on {}: {} steps", r.name, r.backend, r.steps.len());
    el("details")
        .kid(text("summary", summary))
        .kid(table(&head, rows))
}

fn texts(pass: &Pass) -> impl Iterator<Item = El> + '_ {
    pass.texts.iter().enumerate().map(|(i, t)| {
        el("details")
            .attr("id", format!("k{i}"))
            .kid(text("summary", format!("{} as {}", t.kernel, t.dialect)))
            .kid(text("pre", t.text.clone()))
    })
}

fn section(title: &str, body: El) -> El {
    el("section").kid(text("h2", title)).kid(body)
}

/// The whole page for one pass.
pub fn page(model: &Model, pass: &Pass) -> String {
    let title = format!("{}: lowering, {}", model.name, pass.label);
    let nav = el("p")
        .kid(text("a", "source graph").attr("href", "graph.dot"))
        .text(" · ")
        .kid(text("a", "sharded graph").attr("href", "sharded.dot"))
        .text(" · ")
        .kid(text("a", "tokens moving").attr("href", "pump.html"));
    let body = el("body")
        .kid(text("h1", title.clone()))
        .kid(nav)
        .kid(section("Ranks", ranks(pass)))
        .kid(section("Stages", stages(model, pass)))
        .kid(section("Weights per rank", weights(model, pass)))
        .kid(section("Nodes through each stage", nodes(model, pass)))
        .kid(
            el("section")
                .kid(text("h2", "Steps per rank"))
                .kids(pass.ranks.iter().map(steps)),
        )
        .kid(
            el("section")
                .kid(text("h2", "Kernel texts"))
                .kids(texts(pass)),
        );
    el("html").kid(head(&title)).kid(body).page()
}
