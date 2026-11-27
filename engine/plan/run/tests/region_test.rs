//! Rules that cover a whole traced region.

use tessera_graph::DimName::{self, Context, Hidden, Rows, Seqs};
use tessera_graph::{Act, Composite, DType, Dim, Graph, NodeId, TensorType};
use tessera_lower::kernels::Map;
use tessera_lower::{Device, Draft, Kernel, Plan, Propose, Read, Rule, T, Tree, lower, sites};
use tessera_run::testing::host;
use tessera_testkit::graphs;
use tessera_trace::{EntryKind, Model, Port, Role};

/// A launch whose cost its rule sets.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "block",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = self.guess,
)]
struct Block {
    len: usize,
    guess: f64,
}

/// Two launches over a region: the first feeds the second.
struct Whole(f64);

impl Propose for Whole {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let r: sites::Region = tree.site(at)?;
        let (len, guess) = (r.len, self.0);
        let first = Draft::new(Block { len, guess }, r.reads(), len);
        let reads = [Read::Draft(0)].into_iter().chain(r.reads());
        Some(vec![first, Draft::new(Block { len, guess }, reads, len)])
    }

    fn regions(&self) -> bool {
        true
    }
}

const SIZES: [(DimName, u64); 1] = [(Rows, 3)];
const STEP: [(DimName, u64); 3] = [(Seqs, 1), (Rows, 2), (Context, 5)];
const BOUNDED: [Dim; 3] = [
    Dim::bounded(Seqs, 4),
    Dim::bounded(Rows, 16),
    Dim::bounded(Context, 16),
];

/// A region reads `a = x + y` twice. Then an add.
fn graph() -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 8), Dim::fixed(Hidden, 16)]);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let port = || Port {
        role,
        ty: ty.clone(),
    };
    let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
    let [inner] = m.composite(Composite::Mlp, || {
        let a = x.add(&y);
        [a.act(Act::Silu).mul(&a)]
    });
    m.finish([(Role::Logits, &inner.add(&x))])
}

fn plan(g: &Graph, sizes: &[(DimName, u64)], guess: f64) -> Plan {
    let rule = Rule::made("whole", module_path!(), Whole(guess));
    lower(g, 0, sizes, &Device::guess(&host()).adding(rule)).unwrap()
}

fn rules(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.rule).collect()
}

#[test]
fn region_rules_cover() {
    let g = graph();
    let region = &g.regions[0];
    let within = |n: &NodeId| region.start <= *n && *n < region.end;
    let cheap = plan(&g, &SIZES, 1e-15);
    assert_eq!(rules(&cheap), ["whole", "whole", "map"]);
    // The second launch reads the first. No other enters.
    assert_eq!(cheap.steps[1].reads[0], cheap.steps[0].write);
    assert!(cheap.steps[2].covers.iter().all(|n| !within(n)));
    let dear = plan(&g, &SIZES, 1.0);
    assert!(dear.steps.iter().all(|s| s.rule != "whole"));
}

#[test]
fn regions_write_kept() {
    let plan = plan(&graphs::appended(&BOUNDED, 16), &STEP, 1e-15);
    // The region writes K and V. No append runs.
    assert_eq!(rules(&plan), ["whole", "whole"]);
    assert_eq!(plan.kept.len(), 2);
    for (_, slot) in &plan.kept {
        assert!(plan.steps[0].reads.contains(slot));
    }
}

struct MemFile(Vec<u8>);

impl tessera_lower::form::Filed for MemFile {
    fn bytes(&self) -> usize {
        self.0.len()
    }

    fn read(&self, into: &mut [u8], _: tessera_engine_settings::Reads) -> std::io::Result<()> {
        into.copy_from_slice(&self.0);
        Ok(())
    }
}

fn mix_plan(
    form: &'static dyn tessera_lower::form::Load,
    shape: [usize; 3],
    slots: usize,
    share: Option<tessera_lower::Share>,
) -> Plan {
    let elems = shape[0] * shape[1] * shape[2];
    let bytes = form.bytes(elems);
    let packed = [shape[0] as u32, shape[1] as u32, shape[2] as u32, 0];
    let weights = (0..slots)
        .map(|i| tessera_lower::Loaded {
            weight: 0,
            slot: tessera_lower::Slot {
                buffer: 0,
                offset: i * bytes,
                len: elems,
                form,
                dims: 3,
                shape: packed,
            },
            share,
            part: None,
        })
        .collect();
    Plan {
        id: Plan::fresh(),
        buffers: vec![bytes * slots],
        zones: vec![0],
        weights,
        ..Plan::default()
    }
}

#[test]
fn filed_validates_mixed_trellis() {
    use std::sync::Arc;
    use tessera_lower::form::{Filed, Weight, named};
    use tessera_mem::trellis::{MIX_ROW, MIX_TABLE, alloc_mixed};
    use tessera_run::{Error, open};

    let form = named("trellis_mix_pmcgx").expect("mixed form");
    let shape = [2, 128, 128];
    let elems = shape[0] * shape[1] * shape[2];
    let (mut valid, offsets) = alloc_mixed(128, 128, 1, &[2, 4]);
    valid[offsets[0]..offsets[0] + 16].fill(0x5A);

    let mut trunc = valid.clone();
    trunc.truncate(valid.len() - 64);
    assert!(form.fits_size(elems, trunc.len()));

    let mut bad_width = valid.clone();
    bad_width[MIX_TABLE + 8..MIX_TABLE + 12].copy_from_slice(&5u32.to_le_bytes());

    let mut bad_offset = valid.clone();
    let misaligned = (offsets[0] + 4) as u64;
    bad_offset[MIX_TABLE..MIX_TABLE + 8].copy_from_slice(&misaligned.to_le_bytes());

    let mut past_end = valid.clone();
    let near_end = (past_end.len() - 64) as u64;
    past_end[MIX_TABLE + MIX_ROW..MIX_TABLE + MIX_ROW + 8].copy_from_slice(&near_end.to_le_bytes());

    // Same-total mixed buffers with different axes.
    let (wrong_count, _) = alloc_mixed(128, 256, 1, &[3]);
    let (wrong_axes, _) = alloc_mixed(256, 128, 1, &[2, 4]);
    let shape_256 = [2, 128, 256];

    for slots in [1, 2] {
        let plan = mix_plan(form, shape, slots, None);
        let s0 = plan.weights[0].slot;
        let s1 = (slots == 2).then(|| plan.weights[1].slot);
        let mut runner = open(plan, &host()).unwrap();
        assert!(runner.held_mut().mapped_slot(s0).is_some());

        let file: Arc<dyn Filed> = Arc::new(MemFile(valid.clone()));
        runner
            .load_weight(0, &Weight::Packed(form, valid.clone()))
            .unwrap();
        runner.load_weight(0, &Weight::Filed(form, file)).unwrap();
        let prev0 = runner.held().bytes(s0).unwrap();
        let prev1 = s1.map(|s| runner.held().bytes(s).unwrap());

        for bad in [&trunc, &bad_width, &bad_offset, &past_end, &wrong_count] {
            let err_p = runner
                .load_weight(0, &Weight::Packed(form, bad.clone()))
                .unwrap_err();
            let file: Arc<dyn Filed> = Arc::new(MemFile(bad.clone()));
            let err_f = runner
                .load_weight(0, &Weight::Filed(form, file))
                .unwrap_err();
            assert!(matches!(err_p, Error::WeightSize { weight: 0, .. }));
            assert!(matches!(err_f, Error::WeightSize { weight: 0, .. }));
            // Rejected reloads preserve slot 0 and every later slot.
            assert_eq!(runner.held().bytes(s0).unwrap(), prev0);
            if let (Some(slot), Some(prev)) = (s1, prev1.as_ref()) {
                assert_eq!(&runner.held().bytes(slot).unwrap(), prev);
            }
        }
    }

    // Swapped [out, inp] with same element total is rejected.
    let mut r2 = open(mix_plan(form, shape_256, 1, None), &host()).unwrap();
    let (ok_256, _) = alloc_mixed(256, 128, 1, &[2, 4]);
    let (swapped, _) = alloc_mixed(128, 256, 1, &[2, 4]);
    assert_eq!(wrong_axes.len(), ok_256.len());
    r2.load_weight(0, &Weight::Packed(form, ok_256)).unwrap();
    assert!(matches!(
        r2.load_weight(0, &Weight::Packed(form, swapped.clone())),
        Err(Error::WeightSize { weight: 0, .. })
    ));
    assert!(matches!(
        r2.load_packed(0, &swapped),
        Err(Error::WeightSize { weight: 0, .. })
    ));
    let f_swap: Arc<dyn Filed> = Arc::new(MemFile(swapped));
    assert!(matches!(
        r2.load_weight(0, &Weight::Filed(form, f_swap)),
        Err(Error::WeightSize { weight: 0, .. })
    ));
}

#[test]
fn sharded_validates_mixed_trellis() {
    use std::sync::Arc;
    use tessera_lower::Share;
    use tessera_lower::form::{F32_FORM, Filed, Weight, named};
    use tessera_mem::trellis::alloc_mixed;
    use tessera_run::{Error, open};

    let form = named("trellis_mix_pmcgx").expect("mixed form");
    let share = Share {
        axis: 0,
        first: 2,
        count: 2,
        whole: 4,
        inner: 128 * 128,
    };
    let (mut whole, offsets) = alloc_mixed(128, 128, 1, &[3, 2, 4, 2]);
    for (m, &off) in offsets.iter().enumerate() {
        whole[off..off + 32].fill(0x11 * (m as u8 + 1));
    }
    let expect = form
        .share(&whole, [1, 4, 128 * 128], [2, 2])
        .expect("direct slice");
    assert_eq!(share.of(form, &whole).as_deref(), Some(expect.as_slice()));

    // Valid 4x128x128 set shards through packed and filed loads.
    let plan = mix_plan(form, [2, 128, 128], 1, Some(share));
    let s0 = plan.weights[0].slot;
    let mut runner = open(plan, &host()).unwrap();
    runner
        .load_weight(0, &Weight::Packed(form, whole.clone()))
        .unwrap();
    let loaded = runner.held().bytes(s0).unwrap();
    assert_eq!(&loaded[..expect.len()], expect.as_slice());

    // Pre-cut rank-local mixed buffer also loads directly.
    runner
        .load_weight(0, &Weight::Packed(form, expect.clone()))
        .unwrap();
    let whole_file: Arc<dyn Filed> = Arc::new(MemFile(whole.clone()));
    runner
        .load_weight(0, &Weight::Filed(form, whole_file))
        .unwrap();
    let prev = runner.held().bytes(s0).unwrap();
    assert_eq!(&prev[..expect.len()], expect.as_slice());

    // Invalid and wrong-shape mixed sets return WeightSize.
    let mut trunc_whole = whole.clone();
    trunc_whole.truncate(whole.len() - 64);
    let (wrong_whole, _) = alloc_mixed(128, 256, 1, &[3, 2]);
    let (wrong_local, _) = alloc_mixed(128, 256, 1, &[3]);
    for bad in [&trunc_whole, &wrong_whole, &wrong_local] {
        assert!(matches!(
            runner.load_weight(0, &Weight::Packed(form, bad.clone())),
            Err(Error::WeightSize { weight: 0, .. })
        ));
        let bad_file: Arc<dyn Filed> = Arc::new(MemFile(bad.clone()));
        assert!(matches!(
            runner.load_weight(0, &Weight::Filed(form, bad_file)),
            Err(Error::WeightSize { weight: 0, .. })
        ));
        assert_eq!(runner.held().bytes(s0).unwrap(), prev);
    }

    // Widening sharded path rejects invalid input before unpack_all.
    let wide = mix_plan(F32_FORM, [2, 128, 128], 1, Some(share));
    let mut r_wide = open(wide, &host()).unwrap();
    assert!(matches!(
        r_wide.load_weight(0, &Weight::Packed(form, trunc_whole)),
        Err(Error::WeightSize { weight: 0, .. })
    ));
}
