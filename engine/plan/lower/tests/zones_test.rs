//! Plans over a GPU zone and a host zone.

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

use gpukit::testing::scratch;
use graph::DimName::{Context, Rows, Seqs};
use lower::kernels::Map;
use lower::sites;
use lower::{
    Cluster, Device, Draft, Error, Kernel, Launch, Link, Plan, Source, Step, T, Tree, Zone,
    ZoneKind, lower, open,
};
use models::qwen3::{Config, stepper};
use settings::read as settings;
use sim::Tag;

/// Picks each row's largest entry on the host.
#[derive(Clone, Copy, Debug)]
struct HostTop {
    rows: usize,
}

impl Kernel for HostTop {
    fn name(&self) -> &'static str {
        "host_top"
    }

    fn source(&self) -> Source {
        Map::plain(Vec::new()).source()
    }

    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }

    fn work(&self, _: usize) -> f64 {
        self.rows as f64
    }

    fn guess(&self) -> f64 {
        0.0
    }

    fn runs_on(&self) -> ZoneKind {
        ZoneKind::Cpu
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top { x, rows, k: 1, .. } = tree.site(at)? else {
        return None;
    };
    Some(vec![Draft::new(HostTop { rows }, [x], rows)])
}

lower::rule!("host_top", propose);

/// A GPU and a host, linked both ways at `latency`.
fn two(latency: f64) -> Cluster {
    let mut cluster = Cluster::one("gpu");
    let host = cluster.add(Zone {
        name: "host".into(),
        kind: ZoneKind::Cpu,
        parent: None,
    });
    for (from, to) in [(0, host), (host, 0)] {
        cluster.links.push(Link {
            from,
            to,
            latency,
            rate: 1e12,
            tag: Tag::Assumed,
        });
    }
    cluster
}

fn step(device: &Device) -> Plan {
    let sizes = [(Seqs, 1), (Rows, 2), (Context, 5)];
    lower(&stepper(&Config::toy(4, 16)), 0, &sizes, device).unwrap()
}

fn named<'a>(plan: &'a Plan, kernel: &'a str) -> impl Iterator<Item = &'a Step> {
    plan.steps.iter().filter(move |s| s.kernel.name() == kernel)
}

#[test]
fn cheap_links_place_work() {
    let device = Device::guess(&settings()).zoned(two(1e-6));
    let plan = step(&device);
    assert!(plan.zoned());
    let [top] = &named(&plan, "host_top").collect::<Vec<_>>()[..] else {
        panic!("one top on the host");
    };
    assert_eq!((top.zone, plan.zone(top.write.buffer)), (1, 1));
    assert!(named(&plan, "argmax").next().is_none());
    // The logits move over, and the ids move back.
    let moves: Vec<&Step> = named(&plan, "move").collect();
    assert_eq!(moves.len(), 2);
    for m in &moves {
        assert_ne!(plan.zone(m.reads[0].buffer), m.zone);
        assert_eq!(plan.zone(m.write.buffer), m.zone);
        assert!(m.launches().is_empty());
    }
    assert_eq!(
        (moves[0].write, moves[1].reads[0]),
        (top.reads[0], top.write)
    );
    for o in &plan.outputs {
        assert_eq!(
            plan.zone(o.slot.buffer),
            0,
            "outputs lie where the runner reads"
        );
    }
    let cost = device.cost(&plan);
    assert!(cost.seconds.is_finite() && cost.seconds > 0.0);
    assert!(matches!(
        open(plan, &settings()),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn slow_links_stay() {
    let home = step(&Device::guess(&settings()));
    assert!(!home.zoned() && named(&home, "argmax").count() == 1);
    assert_eq!(step(&Device::guess(&settings()).zoned(two(1.0))), home);
    // Unlinked zones cannot be reached at all.
    let mut apart = two(0.0);
    apart.links.clear();
    assert_eq!(step(&Device::guess(&settings()).zoned(apart)), home);
    assert_eq!(home.zones, vec![0; home.buffers.len()]);
}

#[test]
fn clusters_are_stored() {
    let device = Device::guess(&settings()).zoned(two(2e-6));
    let file = scratch("tessera-lower-zones-test.json");
    device.save(&file).unwrap();
    let loaded = Device::load(&file).unwrap();
    std::fs::remove_file(&file).unwrap();
    assert!(loaded == device && loaded.zones() == 2);
    assert_eq!(loaded.transfer(0, 1, 1_000_000), Some(2e-6 + 1e-6));
    assert_eq!(loaded.transfer(1, 1, 1_000_000), Some(0.0));
}
