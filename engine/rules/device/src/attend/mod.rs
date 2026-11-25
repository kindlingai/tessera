//! Tensor-Core grouped-Q split-K MLA attention rule for CUDA.

use crate::linear::{CUDA_LACKS, f32_only};
use crate::{Backend, Options, flatten};
use std::sync::OnceLock;
use tessera_graph::{Op, Order};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Extent, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term,
    Tree,
};

const KERNEL: &str = include_str!("mla.cu");
const WIDTH: usize = 576;
const PART_STRIDE: usize = WIDTH + 4;
const SMEM_BYTES: usize = 42_240;

fn mla_entry(entry: &'static str) -> Source {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    let text = *CUDA.get_or_init(|| {
        let options = Options {
            backend: Backend::Cuda,
            remote_kv: false,
            marks: false,
        };
        let flat = flatten("mla.kernel", KERNEL, &|_| None, &options)
            .expect("mla kernel flattens")
            .text;
        Box::leak(flat.into_boxed_str())
    });
    Source::Fixed {
        text: ByBackend::from_fn(|_| text),
        entry,
    }
}

fn entry_for(kv: &str) -> Option<&'static str> {
    match kv {
        "fp8" => Some("tessera_mla_attend_fp8"),
        "bf16" => Some("tessera_mla_attend_bf16"),
        "f32" => Some("tessera_mla_attend_f32"),
        _ => None,
    }
}

fn mla_splits(rows: usize, hgroups: usize) -> usize {
    let groups = rows * hgroups;
    if groups <= 4 {
        16
    } else if groups <= 8 {
        8
    } else if groups <= 16 {
        4
    } else {
        2
    }
}

/// Grouped-Q split-K MLA attention over dense KV.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_mla_attend",
    sizes = [self.rows, self.heads, self.keys.most(), self.picks, self.splits],
    work = self.rows * self.heads * self.picks * WIDTH,
    accepts = attend_accepts,
    emits = f32_only,
    needs = attend_needs(),
    order = Order::PerConfig,
    source = mla_entry(self.entry),
    launches = attend_launches,
    load = attend_load,
    unproven,
)]
struct MlaAttend {
    rows: usize,
    heads: usize,
    keys: Extent,
    context: usize,
    picks: usize,
    from_list: bool,
    scale: f32,
    window: (usize, usize),
    splits: usize,
    kv_form: &'static dyn Stored,
    entry: &'static str,
}

fn attend_accepts(k: &MlaAttend, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == "f32",
        1 => form.name() == k.kv_form.name(),
        _ => form.name() == "u32",
    }
}

fn attend_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: SMEM_BYTES,
        group_threads: 128,
        compiler: true,
        lacks: CUDA_LACKS,
        bf16_tile: true,
        ..Needs::default()
    }
}

fn attend_load(k: &MlaAttend, _: usize) -> Option<Load> {
    let hgroups = k.heads / 16;
    let groups = k.rows * hgroups * k.splits;
    let seen = k.picks.min(k.keys.most()).max(1);
    let q_bytes = (4 * k.rows * k.heads * WIDTH) as f64;
    let kv_bytes = k.kv_form.bytes(k.rows * seen * WIDTH) as f64;
    let out_bytes = (4 * k.rows * k.heads * k.splits * PART_STRIDE) as f64;
    Some(Load {
        threads: groups * 128,
        group: 128,
        trips: seen.div_ceil(16 * k.splits) as f64,
        reads: vec![kv_bytes, q_bytes],
        written: out_bytes,
        flight: 1024.0,
        ..Load::default()
    })
}

fn attend_launches(k: &MlaAttend, step: &Step) -> Vec<Launch> {
    let hgroups = k.heads / 16;
    let shape = Shape::Grid {
        across: k.splits,
        down: k.rows * hgroups,
        threads: 128,
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.read(4),
        step.out(),
        Arg::value(&k.scale),
        Arg::count(k.rows),
        Arg::size(k.keys),
        Arg::count(k.context),
        Arg::count(k.heads),
        Arg::count(k.picks),
        Arg::count(usize::from(k.from_list)),
        Arg::count(k.window.0),
        Arg::count(k.window.1),
        Arg::count(k.splits),
    ];
    vec![Launch { args, shape }]
}

/// Combines split-K MLA partials into final outputs.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_mla_reduce",
    sizes = [self.lists, self.splits],
    work = self.lists * self.splits * WIDTH,
    accepts = red_accepts,
    emits = f32_only,
    needs = red_needs(),
    order = Order::PerConfig,
    source = mla_entry("tessera_mla_reduce"),
    launches = red_launches,
    load = red_load,
    unproven,
)]
struct MlaReduce {
    lists: usize,
    splits: usize,
}

fn red_accepts(_: &MlaReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn red_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: 4 * 32 * 3 + 4,
        group_threads: 160,
        compiler: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn red_load(r: &MlaReduce, _: usize) -> Option<Load> {
    Some(Load {
        threads: r.lists * 160,
        group: 160,
        trips: r.splits as f64,
        reads: vec![(4 * r.lists * r.splits * PART_STRIDE) as f64],
        written: (4 * r.lists * WIDTH) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn red_launches(r: &MlaReduce, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: r.lists,
        down: 1,
        threads: 160,
    };
    let args = vec![
        step.read(0),
        step.out(),
        Arg::count(r.lists),
        Arg::count(r.splits),
    ];
    vec![Launch { args, shape }]
}

fn propose_mla(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    if !a.ragged || node.form.name() != "f32" {
        return None;
    }
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    if k != v || tree.node(q).form.name() != "f32" {
        return None;
    }
    let kv_form = tree.node(k).form;
    let entry = entry_for(kv_form.name())?;
    let (&[rows, heads, width], &[_, most, kv_heads, _]) =
        (&tree.node(q).shape[..], &tree.node(k).shape[..])
    else {
        return None;
    };
    if width != WIDTH || kv_heads != 1 || heads < 16 || !heads.is_multiple_of(16) || rows > 32 {
        return None;
    }
    let context = match &tree.node(k).term {
        Term::Op(Op::Append(ki)) => {
            if tree.kept.name() != "dense" {
                return None;
            }
            tree.kept_rows(*ki)?.1
        }
        _ => most,
    };
    let [slots, positions] = [node.args[3], node.args[4]];
    let (ids, picks) = match node.args.get(5) {
        Some(&ids) => (ids, *tree.node(ids).shape.last()?),
        None => (positions, most),
    };
    let splits = mla_splits(rows, heads / 16);
    let attend = MlaAttend {
        rows,
        heads,
        keys: tree.node(k).extents[1],
        context,
        picks,
        from_list: a.picked,
        scale: a.scale,
        window: a.mask.window(most),
        splits,
        kv_form,
        entry,
    };
    let lists = rows * heads;
    let reduce = MlaReduce { lists, splits };
    Some(vec![
        Draft::new(
            attend,
            [q, k, ids, slots, positions],
            lists * splits * PART_STRIDE,
        ),
        Draft::new(reduce, [Read::Draft(0)], lists * WIDTH),
    ])
}

tessera_lower::rule!("fast_mla_attend", propose_mla);
