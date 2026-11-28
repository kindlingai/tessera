//! Per-way zone fitness checks for mixed CPU/GPU rules.

use tessera_graph::DimName::Rows;
use tessera_graph::{
    DType, Dim, Entry, EntryKind, Graph, Node, NodeId, Op, Output, Placement, Port, Role, Row,
    Shape, Stage, TensorType,
};
use tessera_lower::kernels::{Map, MapStage};
use tessera_lower::rule::ProposeFn;
use tessera_lower::{
    ByBackend, Cluster, Device, Draft, Kernel, Launch, Needs, Rule, Source, Step, T, Term, Tree,
    lower,
};

#[derive(Clone, Copy, Debug)]
struct FixedGpu;

impl Kernel for FixedGpu {
    fn name(&self) -> &'static str {
        "fixed_gpu_only"
    }

    fn source(&self) -> Source {
        Source::Fixed {
            text: ByBackend::default(),
            entry: "fixed_gpu_only",
        }
    }

    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }

    fn needs(&self) -> Needs {
        Needs::text()
    }
}

fn propose_mixed(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if !matches!(node.term, Term::Stage(Stage::Softmax)) {
        return None;
    }
    let len = tree.elements(at);
    let x = node.args[0];
    Some(vec![
        Draft::new(Map::plain(vec![MapStage::Scale(1.0)]), [x], len).choice(0),
        Draft::new(FixedGpu, [x], len).choice(1),
    ])
}

fn softmax_graph() -> Graph {
    let mut g = Graph::new(vec![Dim::bounded(Rows, 8)]);
    let ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Rows.into(), 16u64.into()]),
        placement: Placement::Replicated,
    };
    let x = NodeId(0);
    g.nodes.push(Node {
        op: Op::Input { entry: 0, port: 0 },
        args: vec![],
        ty: ty.clone(),
        numerics: None,
    });
    let out = NodeId(1);
    g.nodes.push(Node {
        op: Op::Row(Row(vec![Stage::Softmax])),
        args: vec![x],
        ty: ty.clone(),
        numerics: None,
    });
    g.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs: vec![Port {
            role: Role::Tokens,
            ty: ty.clone(),
        }],
        outputs: vec![Output {
            role: Role::Logits,
            node: out,
        }],
    });
    g
}

#[test]
fn cpu_keeps_made() {
    let g = softmax_graph();
    let rule = Rule::made("mixed_ways", "way_zone_test", propose_mixed as ProposeFn);
    let cpu = Device::default().zoned(Cluster::cpu("cpu")).adding(rule);
    let plan = lower(&g, 0, &[(Rows, 4)], &cpu).unwrap();
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].kernel.name(), "map");
    let favored = lower(&g, 0, &[(Rows, 4)], &cpu.favoring("mixed_ways")).unwrap();
    assert_eq!(favored.steps[0].kernel.name(), "map");
}
