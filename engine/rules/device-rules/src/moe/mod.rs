//! CUDA Tensor-Core picked trellis matmul for MoE experts.

use crate::flatten_cuda;
use std::sync::OnceLock;
use tessera_device::{CUDA_LACKS, cuda_rule};
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites::Coded;
use tessera_lower::{Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree};
use tessera_mem::trellis::parse_form;

const KERNEL: &str = include_str!("trellis.cu");
const BLOCK: usize = 128;
const THREADS: usize = 256;
const SMEM: usize = 4096 * 4;

tessera_lower::knob!("picked_trellis", "chunk", 1, 4);
tessera_lower::knob!("picked_trellis", "min_blocks", 1, 16);
tessera_lower::knob!("picked_trellis", "gen_min_blocks", 1, 16);

fn cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_cuda("moe_trellis.kernel", KERNEL))
}

fn entry_name(k: &PickedTrellis) -> &'static str {
    if k.site.turned && k.bits == 3 && k.book == 2 {
        "ptr_k3_pmcgx_t"
    } else {
        "ptr_gen"
    }
}

fn trellis_source(k: &PickedTrellis) -> Source {
    let entry = entry_name(k);
    let mut defs = Vec::new();
    if let Some(c) = k.chunk {
        defs.push(("PTR_CHUNK", c as usize));
    }
    if entry == "ptr_k3_pmcgx_t" {
        if let Some(mb) = k.min_blocks {
            defs.push(("PTR_K3_MIN_BLOCKS", mb as usize));
        }
    } else if let Some(mb) = k.gen_min_blocks {
        defs.push(("PTR_GEN_MIN_BLOCKS", mb as usize));
    }
    tessera_device::cuda_source_with(cuda_source(), entry, defs)
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.site.form.suffixed(if self.site.turned { "picked_trellis_t" } else { "picked_trellis" }),
    sizes = [self.site.rows, self.site.n, self.site.k],
    work = self.site.rows * self.site.n * self.site.k,
    accepts = self.site.accepts(input, form),
    emits = emits,
    needs = needs(),
    order = Order::PerConfig,
    source = trellis_source(self),
    launches = launches,
    load = load,
    load_key = tessera_lower::load_key((
        self.site.rows,
        self.site.n,
        self.site.k,
        self.site.repeat,
        self.site.turned,
        self.site.post,
        self.site.form.name(),
    )),
    unproven,
)]
struct PickedTrellis {
    site: Coded,
    bits: u32,
    book: u32,
    chunk: Option<u32>,
    min_blocks: Option<u32>,
    gen_min_blocks: Option<u32>,
}

fn emits(_: &PickedTrellis, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: SMEM,
        group_threads: THREADS,
        compiler: true,
        bf16_tile: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn load(k: &PickedTrellis, _: usize) -> Option<Load> {
    Some(k.site.load(BLOCK, THREADS))
}

fn launches(k: &PickedTrellis, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.site.n / BLOCK,
        down: k.site.rows,
        threads: THREADS,
    };
    let args = k.site.args(step, k.bits, k.book);
    vec![Launch { args, shape }]
}

/// Matches rotated or plain picked trellis MoE matmuls.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, reads) = Coded::find(tree, at, BLOCK)?;
    let (bits, book) = parse_form(site.form.name())?;
    let kernel = PickedTrellis {
        site,
        bits,
        book,
        chunk: tree.knobs.get("picked_trellis", "chunk").map(|v| v as u32),
        min_blocks: tree
            .knobs
            .get("picked_trellis", "min_blocks")
            .map(|v| v as u32),
        gen_min_blocks: tree
            .knobs
            .get("picked_trellis", "gen_min_blocks")
            .map(|v| v as u32),
    };
    Some(vec![Draft::new(kernel, reads, site.rows * site.n)])
}

cuda_rule!("picked_trellis", propose);
