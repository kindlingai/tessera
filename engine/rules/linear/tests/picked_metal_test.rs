//! Which launch takes picked trellis matmuls on each backend.

// The other matmul rules come from here.
use tessera_kernels as _;
use tessera_linear as _;

use tessera_gpukit::platform::{Asked, Platform};
use tessera_graph::DimName::{Experts, Ffn, Hidden, Rows};
use tessera_lower::form::named;
use tessera_lower::{Device, Pins, Plan, Source, lower};
use tessera_run::testing::host;
use tessera_trace::{
    DType, Dim, Embed, EntryKind, Graph, Linears, Model, Port, Role, TensorType, Vector, Weights,
};

const RULE: &str = "picked_trellis_metal";
const MIXED: &str = "tm_pickmix_pmcgx_t";
const GENERAL: &str = "tm_picked_gen";

#[derive(Weights)]
struct Set {
    #[shape(Experts, Hidden, Ffn)]
    codes: Linears,
    #[shape(Experts, Hidden)]
    su: Embed,
    #[shape(Experts, Ffn)]
    sv: Embed,
    #[shape(Ffn)]
    signs: Vector,
}

/// Rows on picked experts. `post` scales rotated rows by signs.
fn picked(rotated: bool, post: bool) -> Graph {
    let sizes = [(Hidden, 256), (Ffn, 128), (Experts, 4)];
    let mut dims = vec![Dim::bounded(Rows, 64)];
    dims.extend(sizes.map(|(name, n)| Dim::fixed(name, n)));
    let m = Model::new(dims);
    let set: Set = tessera_trace::declare(&m);
    let role = Role::Logits;
    let x = TensorType::new([Rows, Hidden], DType::F32);
    let ids = TensorType::new([Rows], DType::U32);
    let ports = [Port { role, ty: x }, Port { role, ty: ids }];
    let [x, ids] = m.entry(EntryKind::Forward, ports);
    let codes = m.pick(&set.codes, &ids);
    let mut out = match rotated {
        true => {
            let turned = x.mul(&m.embed(&ids, &set.su)).hadamard(128);
            let product = turned.matmul_rows(&codes).hadamard(128);
            product.mul(&m.embed(&ids, &set.sv))
        }
        false => x.matmul_rows(&codes),
    };
    if post {
        out = out.mul_vector(&set.signs);
    }
    m.finish([(Role::Logits, &out)])
}

fn platform(backend: &'static str, arch: &str) -> Platform {
    Platform::new(Asked {
        backend,
        arch: arch.into(),
        name: arch.into(),
        driver: String::new(),
        limits: tessera_gpukit::Limits::assumed(),
        units: 0,
        bandwidth: 0.0,
        integrated: true,
        memory: 0,
    })
}

fn plan(g: &Graph, form: &str, on: Platform) -> Plan {
    let pins = Pins::default().pin("codes", named(form).unwrap());
    let device = Device::guess(&host()).on(on).pinned(pins);
    lower(g, 0, &[(Rows, 8)], &device).unwrap()
}

/// The entry of each launch the Metal rule took.
fn entries(plan: &Plan) -> Vec<&'static str> {
    let taken = plan.steps.iter().filter(|s| s.rule == RULE);
    let entry = |s: &tessera_lower::Step| match s.kernel.source() {
        Source::Fixed { entry, .. } => entry,
        Source::Made(_) => panic!("a fixed kernel"),
    };
    taken.map(entry).collect()
}

fn apple() -> Platform {
    platform("metal", "apple9")
}

#[test]
fn mixed_takes_fast_entry() {
    for post in [false, true] {
        let plan = plan(&picked(true, post), "trellis_mix_pmcgx", apple());
        assert_eq!(entries(&plan), [MIXED], "post {post}");
        // The signs ride in the same launch.
        assert_eq!(plan.steps.len(), 1, "post {post}");
        let args = plan.steps[0].kernel.launches(&plan.steps[0]).remove(0).args;
        assert_eq!(args[13].u32(), Some(0));
        assert_eq!(args[16].u32(), Some(u32::from(post)));
    }
}

#[test]
fn uniform_takes_general_entry() {
    for rotated in [false, true] {
        let plan = plan(&picked(rotated, false), "trellis_k3_pmcgx", apple());
        assert_eq!(entries(&plan), [GENERAL], "rotated {rotated}");
        let args = plan.steps[0].kernel.launches(&plan.steps[0]).remove(0).args;
        assert_eq!(args[13].u32(), Some(3));
        assert_eq!(args[15].u32(), Some(u32::from(rotated)));
    }
    let plain = plan(&picked(false, false), "trellis_mix_pmcgx", apple());
    assert_eq!(entries(&plain), [GENERAL]);
}

#[test]
fn other_backends_skip() {
    for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx1151")] {
        for form in ["trellis_mix_pmcgx", "trellis_k3_pmcgx"] {
            let plan = plan(&picked(true, true), form, platform(backend, arch));
            assert!(entries(&plan).is_empty(), "{backend} {form}");
        }
    }
}
