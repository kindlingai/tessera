//! Modules from text: a matmul, regions, and bad modules.

// The linked rules the module's rules compete with.
use sensai_kernels as _;
use sensai_linear as _;

use sensai_graph::DimName::{Context, Ffn, Hidden, Rows, Seqs};
use sensai_graph::{Act, Composite};
use sensai_graph::{Parts, shard};
use sensai_interp::{Store, Tensor, run_kept};
use sensai_lower::kept::{self, Layout};
use sensai_lower::{
    Arg, Cache, Cluster, Device, Extent, Limits, Plan, Platform, Rule, Shape, Source, lower,
};
use sensai_run::testing::{host, runs};
use sensai_run::{Loopback, Run};
use sensai_testkit::{Noise, graphs, max_abs_diff};
use sensai_trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, TensorType, Weights};
use serde_json::{Value, json};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// `a = x w`, then `silu(a) * a` as a region.
fn graph(region: bool) -> Graph {
    sized(region, 64, 32)
}

/// A model of `hidden` by `ffn`, with weights and rows.
fn traced<W: Weights>(hidden: u64, ffn: u64) -> (Model, W, sensai_trace::Tensor) {
    let m = Model::new(vec![
        Dim::bounded(Rows, 16),
        Dim::fixed(Hidden, hidden),
        Dim::fixed(Ffn, ffn),
    ]);
    let w: W = sensai_trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    (m, w, x)
}

fn sized(region: bool, hidden: u64, ffn: u64) -> Graph {
    let (m, w, x) = traced::<Layer>(hidden, ffn);
    if !region {
        return m.finish([(Role::Logits, &x.matmul(&w.w))]);
    }
    let [y] = m.composite(Composite::Mlp, || {
        let a = x.matmul(&w.w);
        [a.act(Act::Silu).mul(&a)]
    });
    m.finish([(Role::Logits, &y)])
}

const GEMM_METAL: &str = "#include <metal_stdlib>
using namespace metal;
kernel void gemm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]],
    uint2 t [[thread_position_in_threadgroup]]) {
  uint col = g.x * 32 + t.x, row = g.y;
  if (row >= m || col >= n) return;
  float s = 0;
  for (uint i = 0; i < k; i++) s += x[row * k + i] * w[col * k + i];
  y[row * n + col] = s;
}";

const GEMM_CUDA: &str = "extern \"C\" __global__ void gemm(const float* x, const float* w,
    float* y, unsigned m, unsigned n, unsigned k) {
  unsigned col = blockIdx.x * 32 + threadIdx.x, row = blockIdx.y;
  if (row >= m || col >= n) return;
  float s = 0;
  for (unsigned i = 0; i < k; i++) s += x[row * k + i] * w[col * k + i];
  y[row * n + col] = s;
}";

const ACT_METAL: &str = "#include <metal_stdlib>
using namespace metal;
kernel void act(device const float* a [[buffer(0)]], device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float v = a[i];
  y[i] = v / (1.0f + exp(-v)) * v;
}";

const ACT_CUDA: &str = "extern \"C\" __global__ void act(const float* a, float* y, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = a[i];
  y[i] = v / (1.0f + expf(-v)) * v;
}";

/// The version these modules speak.
const ABI: u32 = 5;

/// `kernels`, and one rule at `site`.
fn declare(abi: u32, site: &str, kernels: Value) -> Value {
    json!({
        "abi": abi,
        "kernels": kernels,
        "rules": [{"name": format!("wasm_{site}"), "site": site}],
    })
}

/// The matmul kernels, and one rule at `site`.
fn manifest(abi: u32, site: &str) -> Value {
    let kernels = json!([
        {"name": "wasm_gemm", "entry": "gemm", "metal": GEMM_METAL, "cuda": GEMM_CUDA},
        {"name": "wasm_act", "entry": "act", "metal": ACT_METAL, "cuda": ACT_CUDA},
    ]);
    declare(abi, site, kernels)
}

/// `x w` over 4 rows. Rows are read at launch.
fn gemm(reads: Value, n: usize, k: usize) -> Value {
    json!({
        "kernel": "wasm_gemm", "reads": reads, "len": 4 * n,
        "counts": [{"dim": "Rows"}, n, k],
        "shape": {"grid": {"across": n.div_ceil(32), "down": 4, "threads": 32}},
        "work": 4 * n * k, "guess": 1e-15,
    })
}

/// The plain `x w` of `graph`.
fn matmul() -> Value {
    gemm(json!([0, 1]), 32, 64)
}

/// `silu(a) * a` over 4 rows of `n`.
fn act(a: Value, n: usize) -> Value {
    json!({
        "kernel": "wasm_act", "reads": [a], "len": 4 * n,
        "counts": [{"dim": "Rows", "mul": n}], "shape": {"line": 4 * n},
        "work": 4 * n, "guess": 1e-15,
    })
}

/// A WAT string literal holding `text`.
fn quoted(text: &str) -> String {
    let text = text.replace('\\', "\\\\").replace('"', "\\\"");
    text.replace('\n', "\\n")
}

/// A module whose `manifest` and `propose` return fixed JSON.
fn module(manifest: &Value, offer: &Value, body: &str) -> Vec<u8> {
    let (manifest, offer) = (manifest.to_string(), offer.to_string());
    let packed = |at: u64, text: &str| (at << 32) | text.len() as u64;
    let wat = format!(
        r#"(module
  (memory (export "memory") 1)
  (data (i32.const 0) "{}")
  (data (i32.const 16384) "{}")
  (func (export "alloc") (param i32) (result i32) (i32.const 32768))
  (func (export "manifest") (result i64) {body} (i64.const {}))
  (func (export "propose") (param i32 i32) (result i64) (i64.const {})))"#,
        quoted(&manifest),
        quoted(&offer),
        packed(0, &manifest),
        packed(16384, &offer),
    );
    wat::parse_str(wat).unwrap()
}

fn load(site: &str, offer: Value) -> &'static Rule {
    sensai_wasm::load(&module(&manifest(ABI, site), &offer, "")).unwrap()[0]
}

/// `g` with `rule` on each GPU, against the interpreter.
fn on_gpus(g: &Graph, rule: &'static Rule) {
    let mut draws = Noise(5);
    let w = Tensor::f32(vec![32, 64], (0..32 * 64).map(|_| draws.draw()).collect());
    let x = Tensor::f32(vec![4, 64], (0..4 * 64).map(|_| draws.draw()).collect());
    let want =
        sensai_interp::run(g, std::slice::from_ref(&w), 0, std::slice::from_ref(&x)).unwrap();
    // Only a GPU runs fixed text.
    for settings in runs().iter().filter(|s| s.gpu().is_some()) {
        let device = Device::guess(settings).adding(rule);
        let plan = lower(g, 0, &[(Rows, 4)], &device).unwrap();
        let mut runner = sensai_run::open(plan, settings).unwrap();
        runner.load(0, w.view()).unwrap();
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-4, "{}: off by {off:e}", runner.device());
    }
}

/// A device with a compiler, so lowering runs on CPU.
fn compiling() -> Device {
    let mut device = Device::default();
    device.platform = Platform::assumed(Limits::assumed());
    device
}

fn rules(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.rule).collect()
}

#[test]
fn module_matmul_wins() {
    let rule = load("matmul", json!([matmul()]));
    let plan = lower(&graph(false), 0, &[(Rows, 4)], &compiling().adding(rule)).unwrap();
    assert_eq!(rules(&plan), ["wasm_matmul"]);
    // The host lacks a compiler.
    let host = Device::guess(&host()).adding(rule);
    let plain = lower(&graph(false), 0, &[(Rows, 4)], &host).unwrap();
    assert!(!rules(&plain).contains(&"wasm_matmul"));
    let step = &plan.steps[0];
    let Source::Fixed { entry, cuda, .. } = step.kernel.source() else {
        panic!("module kernels are fixed text");
    };
    assert!(entry == "gemm" && cuda.contains("extern \"C\""));
    // Rows are fixed here, so the count is plain.
    let launch = &step.kernel.launches(step)[0];
    let counts: Vec<u32> = launch.args[3..].iter().filter_map(Arg::u32).collect();
    assert_eq!(counts, [4, 32, 64]);
    let grid = Shape::Grid {
        across: 1,
        down: 4,
        threads: 32,
    };
    assert_eq!(launch.shape, grid);
    on_gpus(&graph(false), rule);
}

#[test]
fn cuda_alone_opens() {
    let kernels = json!([{"name": "wasm_gemm", "entry": "gemm", "cuda": GEMM_CUDA}]);
    let bytes = module(&declare(ABI, "matmul", kernels), &json!([matmul()]), "");
    let rule = sensai_wasm::load(&bytes).unwrap()[0];
    let plan = lower(&graph(false), 0, &[(Rows, 4)], &compiling().adding(rule)).unwrap();
    let Source::Fixed { metal, cuda, .. } = plan.steps[0].kernel.source() else {
        panic!("module kernels are fixed text");
    };
    assert!(metal.is_empty() && cuda == GEMM_CUDA);
}

#[test]
fn module_region_wins() {
    let rule = load("mlp", json!([matmul(), act(json!({"draft": 0}), 32)]));
    let g = graph(true);
    let device = compiling().adding(rule);
    let mut cache = Cache::default();
    let (plan, _) = cache.upto(&g, 0, &[(Rows, 4)], &[Rows], &device).unwrap();
    assert_eq!(rules(plan), ["wasm_mlp", "wasm_mlp"]);
    assert_eq!(plan.steps[1].reads, [plan.steps[0].write]);
    // Rows read at launch stay sizes until then.
    let act = &plan.steps[1];
    let size = &act.kernel.launches(act)[0].args[2];
    assert!(
        matches!(size, Arg::Size(Extent::Dim { mul: 32, .. })),
        "{size:?}"
    );
    on_gpus(&g, rule);
}

/// A layer returns `silu(a) * a`, then `a = x w`.
fn layer() -> Graph {
    let (m, w, x) = traced::<Layer>(64, 32);
    let [y, a] = m.composite(Composite::Layer, || {
        let a = x.matmul(&w.w);
        let [y] = m.composite(Composite::Mlp, || [a.act(Act::Silu).mul(&a)]);
        [y, a]
    });
    m.finish([(Role::Logits, &y.add(&a))])
}

#[test]
fn module_layer_wins() {
    let g = layer();
    let mut offers = [matmul(), act(json!({"draft": 0}), 32)];
    (offers[0]["writes"], offers[1]["writes"]) = (json!(1), json!(0));
    let rule = load("layer", json!(offers));
    let plan = lower(&g, 0, &[(Rows, 4)], &compiling().adding(rule)).unwrap();
    assert_eq!(&rules(&plan)[..2], ["wasm_layer", "wasm_layer"]);
    // The add after it reads both outputs.
    let [a, y] = [0, 1].map(|i| plan.steps[i].write);
    assert!(plan.steps[2..].iter().any(|s| s.reads == [y, a]));
    on_gpus(&g, rule);
}

const ADD_METAL: &str = "#include <metal_stdlib>
using namespace metal;
kernel void add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = a[i] + b[i];
}";

const ADD_CUDA: &str = "extern \"C\" __global__ void add(const float* a, const float* b,
    float* y, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}";

#[derive(Weights)]
struct Split {
    #[shape(Hidden, Ffn)]
    up: Linear,
    #[shape(Ffn, Hidden)]
    down: Linear,
}

/// A residual MLP layer, to split over two ranks.
fn split_layer() -> Graph {
    let (m, w, x) = traced::<Split>(64, 32);
    let [y] = m.composite(Composite::Layer, || {
        let a = x.matmul(&w.up);
        [x.add(&a.act(Act::Silu).mul(&a).matmul(&w.down))]
    });
    m.finish([(Role::Logits, &y)])
}

/// Each rank's half of the layer, summed by the engine.
fn halves() -> &'static Rule {
    let kernels = json!([
        {"name": "wasm_gemm", "entry": "gemm", "metal": GEMM_METAL, "cuda": GEMM_CUDA},
        {"name": "wasm_act", "entry": "act", "metal": ACT_METAL, "cuda": ACT_CUDA},
        {"name": "wasm_add", "entry": "add", "metal": ADD_METAL, "cuda": ADD_CUDA},
    ]);
    let add = json!({
        "kernel": "wasm_add", "reads": [0, {"draft": 3}], "len": 256,
        "counts": [{"dim": "Rows", "mul": 64}], "shape": {"line": 256},
        "work": 256, "guess": 1e-15,
    });
    let up = gemm(json!([0, 1]), 16, 64);
    let act = act(json!({"draft": 0}), 16);
    let down = gemm(json!([{"draft": 1}, 2]), 64, 16);
    let offers = json!([up, act, down, {"all_reduce": 2}, add]);
    let bytes = module(&declare(ABI, "layer", kernels), &offers, "");
    sensai_wasm::load(&bytes).unwrap()[0]
}

/// Both ranks' plans of `g`, with `rule`.
fn ranks(g: &Graph, device: Device, rule: &'static Rule) -> [Plan; 2] {
    let device = device.adding(rule).zoned(Cluster::two("a", "b"));
    let device = device.ranked(vec![0, 1]);
    [0, 1].map(|r| lower(g, 0, &[(Rows, 4)], &device.clone().as_rank(r)).unwrap())
}

/// Noise of `shape`.
fn drawn(draws: &mut Noise, shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::f32(shape, (0..len).map(|_| draws.draw()).collect())
}

#[test]
fn module_sums_shards() {
    let g = split_layer();
    let sharded = shard(&g, &Parts::even(2)).unwrap().graph;
    let rule = halves();
    for plan in ranks(&sharded, compiling(), rule) {
        assert!(rules(&plan).iter().all(|r| *r == "wasm_layer"));
        assert_eq!(plan.steps[3].kernel.name(), "all_reduce");
    }
    let mut draws = Noise(7);
    let [up, down, x] = [vec![32, 64], vec![64, 32], vec![4, 64]].map(|s| drawn(&mut draws, s));
    let weights = [up, down];
    let want = sensai_interp::run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
    // Only a GPU runs fixed text.
    for settings in runs().iter().filter(|s| s.gpu().is_some()) {
        let plans = ranks(&sharded, Device::guess(settings), rule);
        let mut ranks = Loopback::open(&plans, settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            ranks.load(at as u32, w.view()).unwrap();
        }
        let got = ranks.run(&[x.view()]).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-4, "{:?}: off by {off:e}", settings.gpu());
    }
}

#[test]
fn module_reads_parts() {
    let whole = json!({
        "kernel": "wasm_gemm", "len": 512,
        "reads": [0, {"part": [1, 0]}, {"part": [1, 1]}, {"part": [1, 2]}],
        "shape": {"line": 512}, "work": 512, "guess": 1e-15,
    });
    let rule = load("mlp", json!([whole]));
    let rotated = sensai_lower::form::named("trellis_k4_rot").unwrap();
    let device = Device::default()
        .adding(rule)
        .pinned(sensai_lower::Pins::all(rotated));
    let plan = lower(&sized(true, 128, 128), 0, &[(Rows, 4)], &device).unwrap();
    assert_eq!(rules(&plan), ["wasm_mlp"]);
    // Codes, `suh`, then `svh`, in the offer's order.
    let parts: Vec<_> = (0..3).map(|p| plan.weights[p].slot).collect();
    assert_eq!(
        plan.weights.iter().map(|w| w.part).collect::<Vec<_>>(),
        [Some(0), Some(1), Some(2)]
    );
    assert_eq!(plan.steps[0].reads[1..], parts);
}

#[test]
fn bad_modules_fail() {
    let offer = json!([matmul()]);
    let newer = module(&manifest(ABI + 1, "matmul"), &offer, "");
    assert!(matches!(
        sensai_wasm::load(&newer),
        Err(sensai_wasm::Error::Abi(6))
    ));
    // An endless loop runs out of fuel.
    let spin = module(&manifest(ABI, "matmul"), &offer, "(loop (br 0))");
    assert!(matches!(
        sensai_wasm::load(&spin),
        Err(sensai_wasm::Error::Wasm(_))
    ));
    let imports = r#"(module (import "env" "f" (func)) (memory (export "memory") 1))"#;
    let imports = wat::parse_str(imports).unwrap();
    assert!(matches!(
        sensai_wasm::load(&imports),
        Err(sensai_wasm::Error::Wasm(_))
    ));
}

/// An attention region appends K and V and sums them.
fn kept_graph() -> Graph {
    let bounded = [
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 16),
        Dim::bounded(Context, 16),
    ];
    graphs::appended(&bounded, 16)
}

/// Writes K and V rows at their slots and positions.
const PUT: &str = "KERNEL put(BUF(const float*, k, 0), BUF(const float*, v, 1),
    BUF(const uint*, slots, 2), BUF(const uint*, positions, 3), BUF(float*, keys, 4),
    BUF(float*, values, 5), BUF(float*, done, 6), ARG(rows, 7), ARG(width, 8), TAIL(9) INDEX) {
  if (i >= rows * width) return;
  uint r = i / width, j = i % width, at = ROW(slots[r], positions[r]);
  keys[at * width + j] = k[i];
  values[at * width + j] = v[i];
}";

/// Sums K and V for every slot and position.
const SUM: &str = "KERNEL sum(BUF(const float*, keys, 0), BUF(const float*, values, 1),
    BUF(float*, out, 2), ARG(seqs, 3), ARG(ctx, 4), ARG(width, 5), TAIL(6) INDEX) {
  if (i >= seqs * ctx * width) return;
  uint j = i % width, p = (i / width) % ctx, s = i / (width * ctx), at = ROW(s, p);
  out[i] = keys[at * width + j] + values[at * width + j];
}";

/// Metal and CUDA spellings of the macros the kernels use.
fn spelled(paged: bool) -> [String; 2] {
    let row = match paged {
        true => "#define ROW(s, p) ((table[(s) * kept + ((p) >> 4)] << 4) | ((p) & 15u))",
        false => "#define ROW(s, p) ((s) * kept + (p))",
    };
    let tail = match paged {
        true => "#define TAIL(b) BUF(const uint*, table, b), ARG(kept, b + 1)",
        false => "#define TAIL(b) ARG(kept, b)",
    };
    let metal = "#include <metal_stdlib>\nusing namespace metal;
#define KERNEL kernel void
#define BUF(t, n, b) device t n [[buffer(b)]]
#define ARG(n, b) constant uint& n [[buffer(b)]]
#define INDEX , uint i [[thread_position_in_grid]]
#define START";
    let cuda = "typedef unsigned uint;
#define KERNEL extern \"C\" __global__ void
#define BUF(t, n, b) t n
#define ARG(n, b) uint n
#define INDEX
#define START uint i = blockIdx.x * blockDim.x + threadIdx.x;";
    [metal, cuda].map(|head| format!("{head}\n{tail}\n{row}\n"))
}

fn kept_kernels(paged: bool) -> Value {
    let [metal, cuda] = spelled(paged);
    let start = |text: &str| text.replacen(") {\n", ") {\n  START\n", 1);
    json!([
        {"name": "wasm_put", "entry": "put", "metal": format!("{metal}{}", start(PUT)),
         "cuda": format!("{cuda}{}", start(PUT))},
        {"name": "wasm_sum", "entry": "sum", "metal": format!("{metal}{}", start(SUM)),
         "cuda": format!("{cuda}{}", start(SUM))},
    ])
}

/// Inputs 0, 3, 1 and 2: K, V, slots, positions.
fn put() -> Value {
    json!({
        "kernel": "wasm_put", "reads": [0, 3, 1, 2, {"kept": 0}, {"kept": 1}], "len": 1,
        "counts": [{"dim": "Rows"}, 16], "shape": {"line": 256},
        "work": 64, "guess": 1e-15,
    })
}

/// K plus V over `ctx` positions of all four slots.
fn sum(ctx: usize) -> Value {
    let len = 4 * ctx * 16;
    json!({
        "kernel": "wasm_sum", "reads": [{"kept": 0}, {"kept": 1}], "len": len,
        "counts": [{"dim": "Seqs"}, {"dim": "Context"}, 16], "shape": {"line": len},
        "work": len, "guess": 1e-15,
    })
}

/// The rule for `layout`, its offers sized for `ctx` positions.
fn attention(layout: &dyn Layout, ctx: usize) -> &'static Rule {
    let kernels = kept_kernels(layout.page().is_some());
    let offers = json!([put(), sum(ctx)]);
    let bytes = module(&declare(ABI, "attention", kernels), &offers, "");
    sensai_wasm::load(&bytes).unwrap()[0]
}

/// The plan for one launch of `ctx` positions.
fn kept_plan(device: Device, layout: &'static dyn Layout, ctx: usize) -> Plan {
    let mut device = device.adding(attention(layout, ctx));
    device.kept = layout;
    let sizes = [(Seqs, 4), (Rows, 4), (Context, ctx as u64)];
    lower(&kept_graph(), 0, &sizes, &device).unwrap()
}

#[test]
fn module_attention_writes_kept() {
    // Each launch fills two positions of both slots, in order.
    let launches: [[(u32, u32); 4]; 2] = [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 2), (0, 3), (1, 2), (1, 3)],
    ];
    let layouts: [&'static dyn Layout; 2] = [kept::DENSE, &kept::PAGED16];
    for layout in layouts {
        let plan = kept_plan(compiling(), layout, 4);
        assert_eq!(rules(&plan), ["wasm_attention", "wasm_attention"]);
        assert_eq!(plan.kept.len(), 2);
        // Only a GPU runs fixed text.
        for settings in runs().iter().filter(|s| s.gpu().is_some()) {
            let (mut store, mut draws) = (Store::default(), Noise(9));
            let mut runner: Option<Box<dyn Run>> = None;
            for (i, rows) in launches.iter().enumerate() {
                let plan = kept_plan(Device::guess(settings), layout, 2 * i + 2);
                let table = plan.table;
                let r = match &mut runner {
                    Some(r) => {
                        r.replan(&plan).unwrap();
                        r
                    }
                    None => runner.insert(sensai_run::open(plan, settings).unwrap()),
                };
                // Reversed pages: every row lies away from its dense place.
                if let Some(table) = table {
                    let pages = (0..table.len as u32).rev().flat_map(u32::to_le_bytes);
                    r.held_mut()
                        .upload(table, &pages.collect::<Vec<u8>>())
                        .unwrap();
                }
                let noise = |draws: &mut Noise| (0..64).map(|_| draws.draw()).collect();
                let inputs = [
                    Tensor::f32(vec![4, 16], noise(&mut draws)),
                    Tensor::f32(vec![4, 16], noise(&mut draws)),
                    Tensor::u32(vec![4], rows.iter().map(|r| r.0).collect()),
                    Tensor::u32(vec![4], rows.iter().map(|r| r.1).collect()),
                ];
                let want = run_kept(&kept_graph(), &[], &mut store, 0, &inputs).unwrap();
                let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
                let got = r.run(&views).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    off < 1e-6,
                    "{layout:?} launch {i} on {}: off by {off:e}",
                    r.device()
                );
            }
        }
    }
}

#[test]
fn module_speaks_guest() {
    use sensai_guest::site::{self, Count, Kind, Manifest, Offer, Place};
    let manifest = Manifest {
        abi: sensai_guest::ABI,
        kernels: vec![site::kernel("wasm_gemm", "gemm", GEMM_CUDA)],
        rules: vec![site::Rule {
            name: "wasm_matmul",
            site: Kind::Matmul,
        }],
        graph: false,
    };
    let rows = Count::Dim {
        dim: "Rows".into(),
        mul: 1,
    };
    let offer = Offer {
        kernel: "wasm_gemm",
        reads: vec![Place::Read(0), Place::Read(1)],
        len: 128,
        counts: vec![rows, Count::Fixed(32), Count::Fixed(64)],
        shape: site::Shape::Line(128),
        work: 8192.0,
        guess: 1e-15,
        writes: None,
    };
    let bytes = module(&json!(manifest), &json!([offer]), "");
    let rule = sensai_wasm::load(&bytes).unwrap()[0];
    let device = Device::default().adding(rule);
    let plan = lower(&graph(false), 0, &[(Rows, 4)], &device).unwrap();
    assert_eq!(rules(&plan), ["wasm_matmul"]);
}
