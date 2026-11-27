//! How selection picks among rules: cost, favor and limits.

use tessera_graph::DimName::{self, Hidden, Rows, Seqs};
use tessera_graph::{Act, DType, Dim, Graph, Stage, TensorType};
use tessera_lower::kernels::Map;
use tessera_lower::{
    Device, Draft, Kernel, Launch, Limits, Needs, Plan, Platform, Propose, Rule, Shape, Step, T,
    Tree, explain, lower, select,
};
use tessera_lower::{Term, sites};
use tessera_run::testing::host;
use tessera_trace::{EntryKind, Model, Port, Role};

/// A map stand-in with its own cost.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "cheap",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = 1e-9,
)]
struct Cheap {
    len: usize,
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "dear",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = 1e-8,
)]
struct Dear {
    len: usize,
}

/// The cheapest, but only for 32 lanes.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "lanes",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    needs = Needs::lanes(),
    guess = 1e-12,
)]
struct Lanes {
    len: usize,
}

fn cheap(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Cheap { len }, reads, len)])
}

fn dear(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Dear { len }, reads, len)])
}

fn lanes(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Lanes { len }, reads, len)])
}

/// A map stand-in whose cost its rule sets.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "priced",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = self.guess,
)]
struct Priced {
    len: usize,
    guess: f64,
}

/// The cheapest, with a grid that grows with its length.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "tall",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = tall_grid,
    work = self.len,
    guess = 1e-12,
)]
struct Tall {
    len: usize,
}

fn tall_grid(k: &Tall, step: &Step) -> Vec<Launch> {
    let (across, down) = (1, k.len * 400);
    let shape = Shape::Grid {
        across,
        down,
        threads: 256,
    };
    vec![Launch::plain(step, &[], shape)]
}

/// A fast rule for `Tall`.
struct Towers;

impl Propose for Towers {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let sites::Chain { reads, len, .. } = tree.site(at)?;
        Some(vec![Draft::new(Tall { len }, reads, len)])
    }

    fn fast(&self) -> bool {
        true
    }
}

/// A rule made at run time, its price as data.
struct Price(f64);

impl Propose for Price {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let sites::Chain { reads, len, .. } = tree.site(at)?;
        let guess = self.0;
        Some(vec![Draft::new(Priced { len, guess }, reads, len)])
    }
}

tessera_lower::rule!("cheap", cheap);
tessera_lower::rule!("dear", dear);
tessera_lower::rule!("lanes", lanes);

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Rows, 3)];

/// One chain over rows.
fn chain() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 16),
    ]);
    let ty = TensorType::new([Seqs, Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x, y] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role,
                ty: ty.clone(),
            },
            Port { role, ty },
        ],
    );
    m.finish([(Role::Logits, &x.add(&y).act(Act::Silu))])
}

fn plan(device: &Device) -> Plan {
    lower(&chain(), 0, &SIZES, device).unwrap()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

/// A device where only `cheap` and `dear` apply.
fn pair() -> Device {
    Device::guess(&host()).without("map").without("lanes")
}

#[test]
fn rules_compete() {
    let picked = select(&chain(), 0, &SIZES, &pair()).unwrap();
    assert!(picked.contested.contains("cheap") && picked.contested.contains("dear"));
}

#[test]
fn cost_picks_rule() {
    assert_eq!(names(&plan(&pair())), ["cheap"]);
    // Samples outrank the guesses.
    let (slow, quick) = (plan(&pair()), plan(&pair().favoring("dear")));
    let mut measured = pair();
    measured.fit([(&slow.steps[0], 1e-3), (&quick.steps[0], 1e-6)]);
    let fitted = plan(&measured);
    assert_eq!(names(&fitted), ["dear"]);
    assert!(measured.cost(&fitted).seconds < measured.cost(&slow).seconds);
    // A favored rule wins whatever it costs.
    assert_eq!(names(&plan(&measured.favoring("cheap"))), ["cheap"]);
}

#[test]
fn limits_bar_kernels() {
    let device = Device::guess(&host()).without("map");
    assert_eq!(names(&plan(&device)), ["lanes"]);
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let wide = device.on(Platform::assumed(wide));
    assert_eq!(names(&plan(&wide)), ["cheap"]);
}

#[test]
fn made_rules_compete() {
    let made = |guess| Rule::made("priced", module_path!(), Price(guess));
    assert_eq!(names(&plan(&pair().adding(made(1e-12)))), ["priced"]);
    assert_eq!(names(&plan(&pair().adding(made(1e-6)))), ["cheap"]);
    let recipe = tessera_lower::Recipe {
        rules: vec![made(1e-6)],
        favor: Some("priced".into()),
        require_favored: true,
        ..Default::default()
    };
    assert_eq!(names(&plan(&pair().with(&recipe))), ["priced"]);
    // Requiring an unused favored rule fails lowering.
    let missing = pair().favoring("unused_rule").requiring_favored(true);
    let err = lower(&chain(), 0, &SIZES, &missing).unwrap_err();
    assert!(matches!(err, tessera_lower::Error::FavoredUnused(r) if r == "unused_rule"));
}

#[test]
fn grids_bar_kernels() {
    let cuda = Platform {
        backend: "cuda".into(),
        ..Platform::host(Limits::assumed())
    };
    let device = Device::default().without("map").without("lanes").on(cuda);
    let device = device.adding(Rule::made("tall", module_path!(), Towers));
    assert_eq!(names(&plan(&device)), ["tall"]);
    // 512 elements need a grid 204800 high.
    let big = [(Seqs, 4), (Rows, 8)];
    assert_eq!(
        names(&lower(&chain(), 0, &big, &device).unwrap()),
        ["cheap"]
    );
    let report = explain(&chain(), 0, &big, &device).unwrap().format(10);
    assert!(
        report.contains("grid height 204800 exceeds 65535"),
        "{report}"
    );
}

/// Matches only the inner add stage of a chain.
struct Inner;

impl Propose for Inner {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        if tree.node(at).term != Term::Stage(Stage::Add) {
            return None;
        }
        let len = tree.elements(at);
        let reads = tree.node(at).args.clone();
        Some(vec![Draft::new(Priced { len, guess: 1e-6 }, reads, len)])
    }
}

/// Matches only the outer silu stage without swallowing add.
struct Outer;

impl Propose for Outer {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        if tree.node(at).term != Term::Stage(Stage::Act(Act::Silu)) {
            return None;
        }
        let len = tree.elements(at);
        let reads = tree.node(at).args.clone();
        Some(vec![Draft::new(Dear { len }, reads, len)])
    }
}

#[test]
fn swallowed_fallback_works() {
    let inner = Rule::made("inner", module_path!(), Inner);
    let only_fused = pair().adding(inner).favoring("inner");
    assert_eq!(names(&plan(&only_fused)), ["cheap"]);

    // When a non-swallowing cover exists, the favored rule still wins.
    let outer = Rule::made("outer", module_path!(), Outer);
    let split = pair().adding(inner).adding(outer).favoring("inner");
    assert_eq!(names(&plan(&split)), ["priced", "dear"]);
}

#[derive(Debug)]
struct TagKernel(&'static str);

impl Kernel for TagKernel {
    fn name(&self) -> &'static str {
        "fast_mm"
    }
    fn sizes(&self) -> Vec<usize> {
        vec![8, 640]
    }
    fn source(&self) -> tessera_lower::Source {
        tessera_lower::Source::Fixed {
            text: tessera_lower::ByBackend::from_fn(|_| ""),
            entry: self.0,
        }
    }
    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }
}

#[test]
fn key_and_ranks_attrs() {
    use tessera_gpukit::device::KernelAttrs;
    use tessera_lower::{KernelRef, Slot};
    use tessera_run::{Profile, Ranks, Spent};
    let mk = |entry, rule, site: Option<&str>| Step {
        kernel: KernelRef::new(TagKernel(entry)),
        rule,
        site: site.map(str::to_string),
        reads: Vec::new(),
        write: Slot::floats(0, 8),
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: None,
        index_table: None,
    };
    let k0 = Profile::key(&mk("mm_r8", "fast", Some("matmul [8]")));
    let k1 = Profile::key(&mk("mm_r16", "fast", Some("matmul [8]")));
    let k2 = Profile::key(&mk("mm_r8", "alt", Some("matmul [8]")));
    let k3 = Profile::key(&mk("mm_r8", "fast", Some("matmul [16]")));
    assert_eq!(k0.split(' ').next(), Some("fast_mm"));
    assert!(k0 != k1 && k0 != k2 && k0 != k3);
    assert!(k0.contains("mm_r8 | fast | matmul [8]"), "{k0}");
    let mut r0 = Profile::default();
    let mut r1 = Profile::default();
    let spill = KernelAttrs {
        regs: 96,
        local_bytes: 32,
        shared_bytes: 4096,
        ctas_per_sm: 3,
    };
    let s = |sec, attrs| Spent {
        seconds: sec,
        count: 4,
        work: 4e8,
        bytes: 8e8,
        attrs,
    };
    r0.spent.insert(k0.clone(), s(0.2, None));
    r1.spent.insert(k0.clone(), s(0.2, Some(spill)));
    let text = Ranks {
        launches: 2,
        profiles: vec![Some(r0), Some(r1)],
    }
    .to_string();
    assert!(
        text.contains("count    us each      G/s     GB/s regs   smem   lmem cta"),
        "{text}"
    );
    assert!(
        text.contains("2.0    50000.0      2.0      4.0   96     4K    32B   3"),
        "{text}"
    );
    assert!(text.contains(&format!("  {k0}  SPILL")), "{text}");
}

struct SameMadeTag(&'static str);

impl std::fmt::Debug for SameMadeTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("same_made_tag")
    }
}

impl Kernel for SameMadeTag {
    fn name(&self) -> &'static str {
        "made_mm"
    }
    fn sizes(&self) -> Vec<usize> {
        vec![8, 640]
    }
    fn source(&self) -> tessera_lower::Source {
        let tessera_lower::Source::Made(mut body) = Map::plain(Vec::new()).source() else {
            unreachable!();
        };
        body.entry = self.0.into();
        tessera_lower::Source::Made(body)
    }
    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }
}

#[test]
fn sites_separate_layers() {
    use tessera_graph::{
        Composite, Entry, Node, NodeId, Op, Output, Placement, Region, Shape, Weight,
    };
    use tessera_lower::{KernelRef, Slot, kernel_identity};
    use tessera_run::Profile;

    struct MmRule;
    impl Propose for MmRule {
        fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
            let n = tree.node(at);
            matches!(n.term, Term::Op(Op::Matmul)).then(|| {
                vec![Draft::new(
                    TagKernel("mm_entry"),
                    n.args.clone(),
                    tree.elements(at),
                )]
            })
        }
    }

    let mut g = Graph::new(vec![Dim::bounded(Rows, 8)]);
    let ty = |dims: Vec<tessera_graph::DimExpr>| TensorType {
        dtype: DType::F32,
        shape: Shape(dims),
        placement: Placement::Replicated,
    };
    let x_ty = ty(vec![Rows.into(), 16u64.into()]);
    let w_ty = ty(vec![16u64.into(), 16u64.into()]);
    g.weights.push(Weight {
        path: vec![],
        ty: w_ty.clone(),
    });
    g.weights.push(Weight {
        path: vec![],
        ty: w_ty.clone(),
    });
    let mut push = |op: Op, args: Vec<NodeId>, t: TensorType| -> NodeId {
        let id = NodeId(g.nodes.len() as u32);
        g.nodes.push(Node {
            op,
            args,
            ty: t,
            numerics: None,
        });
        id
    };
    let x = push(Op::Input { entry: 0, port: 0 }, vec![], x_ty.clone());
    let w0 = push(Op::Weight(0), vec![], w_ty.clone());
    let mm0 = push(Op::Matmul, vec![x, w0], x_ty.clone());
    g.regions.push(Region {
        kind: Composite::Layer,
        start: w0,
        end: NodeId(mm0.0 + 1),
        inputs: vec![x],
        outputs: vec![mm0],
    });
    let w1 = push(Op::Weight(1), vec![], w_ty);
    let mm1 = push(Op::Matmul, vec![mm0, w1], x_ty.clone());
    g.regions.push(Region {
        kind: Composite::Layer,
        start: w1,
        end: NodeId(mm1.0 + 1),
        inputs: vec![mm0],
        outputs: vec![mm1],
    });
    g.entries.push(Entry {
        kind: EntryKind::Step,
        inputs: vec![Port {
            role: Role::Tokens,
            ty: x_ty,
        }],
        outputs: vec![Output {
            role: Role::Logits,
            node: mm1,
        }],
    });
    let rule = Rule::made("mm_rule", module_path!(), MmRule);
    let plan = lower(&g, 0, &[(Rows, 1)], &Device::default().adding(rule)).unwrap();
    let s0 = plan.steps.iter().find(|s| s.covers.contains(&mm0)).unwrap();
    let s1 = plan.steps.iter().find(|s| s.covers.contains(&mm1)).unwrap();
    assert_eq!(s0.kernel.name(), s1.kernel.name());
    assert_eq!(s0.kernel.sizes(), s1.kernel.sizes());
    assert_eq!(s0.rule, s1.rule);
    assert_eq!(kernel_identity(&*s0.kernel), kernel_identity(&*s1.kernel));
    let site0 = s0.site.as_deref().unwrap();
    let site1 = s1.site.as_deref().unwrap();
    assert!(site0.contains("layer.0"), "{site0}");
    assert!(site1.contains("layer.1"), "{site1}");
    assert_ne!(Profile::key(s0), Profile::key(s1));

    // Identical-Debug Source::Made kernels still separate by entry.
    let mk = |entry| Step {
        kernel: KernelRef::new(SameMadeTag(entry)),
        rule: "r",
        site: Some("layer.0@n1".into()),
        reads: Vec::new(),
        write: Slot::floats(0, 8),
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: None,
        index_table: None,
    };
    let ka = Profile::key(&mk("made_a"));
    let kb = Profile::key(&mk("made_b"));
    assert_ne!(ka, kb);
    assert!(ka.contains("made_a") && kb.contains("made_b"));
}
