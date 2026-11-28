//! Single-launch in-place Metal RoPE kernel.

use tessera_graph::{DType, Pairs, Stage};
use tessera_lower::Backend;
use tessera_lower::form::{Load as Stored, writable};
use tessera_lower::sites;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree,
};

const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("metal.metal");
const THREADS: usize = 128;

fn fmt_idx(form: &dyn Stored) -> Option<usize> {
    match form.name() {
        "f32" => Some(0),
        "bf16" => Some(1),
        "f16" => Some(2),
        _ => None,
    }
}

const ENTRIES: [[&str; 2]; 3] = [
    ["rope_mtl_f32_half", "rope_mtl_f32_adj"],
    ["rope_mtl_bf16_half", "rope_mtl_bf16_adj"],
    ["rope_mtl_f16_half", "rope_mtl_f16_adj"],
];

fn heads_per_group(positions: usize, heads: usize) -> usize {
    if positions >= 16 {
        heads.max(1)
    } else {
        let target_groups = 16usize.div_ceil(positions.max(1));
        heads.div_ceil(target_groups).max(1)
    }
}

/// Fused in-place Metal RoPE computing angles once per position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.adjacent {
        "rope_metal_adj"
    } else {
        "rope_metal"
    }),
    sizes = [self.positions, self.heads, self.width, self.turned],
    work = self.positions * self.heads * self.turned.max(1),
    needs = metal_needs(),
    accepts = metal_accepts,
    emits = metal_emits,
    source = metal_source(self),
    launches = metal_launches,
    load = metal_load,
    overwrites,
    unproven,
)]
struct RopeMetal {
    positions: usize,
    heads: usize,
    width: usize,
    turned: usize,
    theta: f32,
    adjacent: bool,
    form: &'static dyn Stored,
}

fn metal_needs() -> Needs {
    Needs {
        lanes: 32,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn metal_accepts(k: &RopeMetal, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.form.name(),
        1 => form.dtype() == DType::U32,
        _ => false,
    }
}

fn metal_emits(k: &RopeMetal, form: &dyn Stored) -> bool {
    form.name() == k.form.name()
}

fn metal_source(k: &RopeMetal) -> Source {
    let fi = fmt_idx(k.form).expect("verified form");
    Source::Fixed {
        text: ByBackend([SOURCE, "", ""]),
        entry: ENTRIES[fi][usize::from(k.adjacent)],
    }
}

fn metal_load(k: &RopeMetal, _: usize) -> Option<Load> {
    let hpg = heads_per_group(k.positions, k.heads);
    let across = k.heads.div_ceil(hpg);
    let active = k.positions * k.heads * (2 * k.turned);
    let bytes = k.form.bytes(active) as f64;
    Some(Load {
        threads: across * k.positions * THREADS,
        group: THREADS,
        trips: (hpg * k.turned).div_ceil(THREADS) as f64,
        reads: vec![bytes, (4 * k.positions) as f64],
        written: bytes,
        flight: 128.0,
        ..Load::default()
    })
}

fn metal_launches(k: &RopeMetal, step: &Step) -> Vec<Launch> {
    let hpg = heads_per_group(k.positions, k.heads);
    let shape = Shape::Grid {
        across: k.heads.div_ceil(hpg),
        down: k.positions,
        threads: THREADS,
    };
    let args = vec![
        step.out(),
        step.read(1),
        Arg::count(k.positions),
        Arg::count(k.heads),
        Arg::count(k.width),
        Arg::count(k.turned),
        Arg::count(hpg),
        Arg::value(&k.theta),
    ];
    vec![Launch { args, shape }]
}

/// Returns a fused Metal RoPE draft when `at` fits.
pub(super) fn draft(tree: &Tree, at: T) -> Option<Draft> {
    let sites::Stage {
        stage: Stage::Rope {
            theta,
            pairs,
            turned,
        },
        x,
        operand: Some((pos, positions)),
        len,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    if positions == 0 || width == 0 || !width.is_multiple_of(2) {
        return None;
    }
    let out_form = writable(tree.node(at).form);
    if fmt_idx(out_form).is_none() || tree.node(x).form.name() != out_form.name() {
        return None;
    }
    let heads = len / (positions * width);
    if heads == 0 {
        return None;
    }
    let kernel = RopeMetal {
        positions,
        heads,
        width,
        turned: turned.map_or(width / 2, |t| (t as usize).min(width / 2)),
        theta,
        adjacent: pairs == Pairs::Adjacent,
        form: out_form,
    };
    Some(Draft::new(kernel, [x, pos], len).choice(1))
}
