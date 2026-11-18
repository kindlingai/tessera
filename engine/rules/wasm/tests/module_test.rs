//! Modules from text: a matmul, a region, and bad modules.

// The linked rules the module's rules compete with.
use sensai_kernels as _;
use sensai_linear as _;

use sensai_graph::DimName::{Ffn, Hidden, Rows};
use sensai_graph::{Act, Composite};
use sensai_interp::Tensor;
use sensai_lower::{Arg, Cache, Device, Extent, Plan, Rule, Shape, Source, lower};
use sensai_run::testing::{host, runs};
use sensai_testkit::{Noise, max_abs_diff};
use sensai_trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, TensorType, Weights};
use serde_json::{Value, json};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// `a = x w`, then `silu(a) * a` as a region.
fn graph(region: bool) -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Rows, 16),
        Dim::fixed(Hidden, 64),
        Dim::fixed(Ffn, 32),
    ]);
    let w: Layer = sensai_trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
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

/// The manifest: both kernels, and one rule at `site`.
fn manifest(abi: u32, site: &str) -> Value {
    json!({
        "abi": abi,
        "kernels": [
            {"name": "wasm_gemm", "entry": "gemm", "metal": GEMM_METAL, "cuda": GEMM_CUDA},
            {"name": "wasm_act", "entry": "act", "metal": ACT_METAL, "cuda": ACT_CUDA},
        ],
        "rules": [{"name": format!("wasm_{site}"), "site": site}],
    })
}

/// `x w`, with its rows read at launch.
fn gemm() -> Value {
    json!({
        "kernel": "wasm_gemm", "reads": [0, 1], "len": 128,
        "counts": [{"dim": "Rows"}, 32, 64],
        "shape": {"grid": {"across": 1, "down": 4, "threads": 32}},
        "work": 8192, "guess": 1e-15,
    })
}

/// `silu(a) * a` over the first launch's output.
fn act() -> Value {
    json!({
        "kernel": "wasm_act", "reads": [{"draft": 0}], "len": 128,
        "counts": [{"dim": "Rows", "mul": 32}], "shape": {"line": 128},
        "work": 128, "guess": 1e-15,
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
    sensai_wasm::load(&module(&manifest(2, site), &offer, "")).unwrap()[0]
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

fn rules(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.rule).collect()
}

#[test]
fn module_matmul_wins() {
    let rule = load("matmul", json!([gemm()]));
    let device = Device::guess(&host()).adding(rule);
    let plan = lower(&graph(false), 0, &[(Rows, 4)], &device).unwrap();
    assert_eq!(rules(&plan), ["wasm_matmul"]);
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
fn module_region_wins() {
    let rule = load("mlp", json!([gemm(), act()]));
    let g = graph(true);
    let device = Device::guess(&host()).adding(rule);
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

#[test]
fn bad_modules_fail() {
    let offer = json!([gemm()]);
    let newer = module(&manifest(3, "matmul"), &offer, "");
    assert!(matches!(
        sensai_wasm::load(&newer),
        Err(sensai_wasm::Error::Abi(3))
    ));
    // An endless loop runs out of fuel.
    let spin = module(&manifest(2, "matmul"), &offer, "(loop (br 0))");
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
