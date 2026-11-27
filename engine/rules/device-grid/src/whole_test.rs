use crate::whole::{
    RefillCache, TableBufs, fill_pages, format_summary, pack_into, plan, prepare_source,
};
use tessera_device::layout::{
    ARENAS, BF16, BYTES, F32, FORM, Form, INPUT, Item, KV, Kept, KvTables, NONE, OUTPUT, Ref,
    SCRATCH, Table, form_kind,
};
use tessera_device::proxy::Op;
use tessera_device::resident::{Plan, USER, build_control};
use tessera_gpukit::device::KernelStats;
use tessera_lower::{ByBackend, Collective, Kernel, KernelRef, Launch, Resident, Source, Step};
use tessera_run::{FormMeta, Item as Placed, ItemKind, KeptDesc, Layout as GridLayout, marks};

#[test]
fn prepares_resident_sources() {
    let plain = "extern \"C\" __global__ void step() {}";
    let marked = "extern \"C\" __global__ void step() { TS_MARK(); }";
    assert_eq!(prepare_source(plain, None, false), (None, false));
    assert_eq!(prepare_source(plain, None, true), (None, false));
    let (off, on) = prepare_source(marked, None, false);
    assert!(!on && off.unwrap().starts_with(marks::OFF));
    let (active, on) = prepare_source(marked, None, true);
    assert!(on && active.unwrap().starts_with(marks::ON));
    let (edited, on) = prepare_source(plain, Some(marked), true);
    assert!(on && edited.unwrap().starts_with(marks::ON));
    let (edited_off, on) = prepare_source(plain, Some(plain), false);
    assert!(!on && edited_off.unwrap().starts_with(marks::OFF));
}

#[test]
fn formats_telemetry_lines() {
    let stats = KernelStats {
        regs: 112,
        shared_bytes: 16384,
        local_bytes: 64,
        max_threads: 512,
    };
    let with = format_summary("whole", 20, 256, 2, Some(stats));
    assert!(with.contains("grid=20 threads=256 blocks/sm=2"));
    assert!(with.contains("regs=112 shared_bytes=16384 local_bytes=64 SPILL"));
    let clean = format_summary(
        "whole",
        20,
        256,
        2,
        Some(KernelStats {
            local_bytes: 0,
            ..stats
        }),
    );
    assert!(!clean.contains("SPILL"));
    let without = format_summary("whole", 20, 256, 2, None);
    assert!(!without.contains("regs="));
}

fn read_at<T: Copy>(raw: &[u8], offset: u64) -> T {
    let at = offset as usize;
    unsafe { std::ptr::read_unaligned(raw[at..at + size_of::<T>()].as_ptr().cast()) }
}

fn placed(at: u64, kind: ItemKind, elem: &'static str, shape: [u32; 4]) -> Placed {
    let dims = shape.iter().take_while(|&&d| d > 0).count() as u32;
    Placed {
        at,
        bytes: 64,
        kind,
        elem,
        dims,
        shape,
        share: [0; 4],
        form: None,
    }
}

fn sample_kept(pages: Option<u64>, slots: u32, context: u32, interleaved: bool) -> KeptDesc {
    let page_rows = if pages.is_some() {
        16
    } else if interleaved {
        1
    } else {
        context
    };
    KeptDesc {
        base: 0x7000,
        pages,
        len: None,
        slots,
        context,
        pages_per_slot: context.div_ceil(page_rows),
        page_rows,
        row_bytes: 256,
        elem: "f32",
        interleaved,
    }
}

fn sample_items() -> [Placed; 4] {
    let mut rot = placed(0x2000, ItemKind::Input, "trellis_k4", [64, 128, 0, 0]);
    rot.share = [0, 0, 64, 128];
    rot.form = Some(FormMeta {
        name: "trellis_k4",
        rows: 64,
        cols: 128,
        suh: Some(0x3000),
        svh: Some(0x4000),
    });
    let mut mxfp = placed(0x5000, ItemKind::Input, "mxfp8", [32, 64, 0, 0]);
    mxfp.form = Some(FormMeta {
        name: "mxfp8",
        rows: 32,
        cols: 64,
        suh: None,
        svh: None,
    });
    [
        placed(0x1000, ItemKind::Input, "bf16", [4, 128, 0, 0]),
        rot,
        mxfp,
        placed(0x6000, ItemKind::Output, "f32", [4, 256, 0, 0]),
    ]
}

fn synth_pages(kept: &[KeptDesc]) -> Vec<u32> {
    let mut out = Vec::new();
    fill_pages(kept, &mut out);
    out
}

fn pack_bufs(items: &[Placed], scratch: Option<(u64, usize)>, kept: &[KeptDesc]) -> TableBufs {
    let mut bufs = TableBufs::default();
    fill_pages(kept, &mut bufs.synth);
    pack_into(items, scratch, kept, Ref::NONE, &mut bufs).unwrap();
    bufs
}

#[test]
fn packs_table_records() {
    let dense = sample_kept(None, 2, 16, false);
    let bufs = pack_bufs(&sample_items(), Some((0x9000, 512)), &[dense]);
    assert_eq!(bufs.synth, [0, 1]);
    let plan = Plan {
        pools: Vec::new(),
        lanes: Vec::new(),
        items: bufs.items,
        forms: bufs.forms,
        kept: bufs.kept,
        pages: bufs.synth,
        arenas: vec![0],
        threads: 256,
        shared: 0,
        share: 1,
    };
    let (raw, places) = build_control(&plan, &[], [0, 1, 1], [0; ARENAS]);
    let table: Table = read_at(&raw, 0);
    assert_eq!((table.items_len, table.forms_len), (5, 2));
    assert_eq!(
        (table.now.offset, table.status.offset),
        (places.launch, places.status)
    );
    assert_eq!(places.status - places.launch, 256);
    let i0: Item = read_at(&raw, table.items.offset);
    let i1: Item = read_at(&raw, table.items.offset + size_of::<Item>() as u64);
    let i3: Item = read_at(&raw, table.items.offset + 3 * size_of::<Item>() as u64);
    let i4: Item = read_at(&raw, table.items.offset + 4 * size_of::<Item>() as u64);
    assert_eq!(
        (i0.kind, i0.elem, i0.dims, i0.shape),
        (INPUT, BF16, 2, [4, 128, 0, 0])
    );
    assert_eq!(
        (i1.elem, i1.form, i1.forms, i1.share),
        (FORM, 0, 1, [0, 0, 64, 128])
    );
    assert_eq!((i3.kind, i3.elem, i3.form), (OUTPUT, F32, NONE));
    assert_eq!(
        (i4.kind, i4.elem, i4.shape),
        (SCRATCH, BYTES, [512, 0, 0, 0])
    );
    let f0: Form = read_at(&raw, table.forms.offset);
    let f1: Form = read_at(&raw, table.forms.offset + size_of::<Form>() as u64);
    assert_eq!(
        (f0.codes.offset, f0.suh.offset, f0.svh.offset),
        (0x2000, 0x3000, 0x4000)
    );
    assert_eq!(
        (f0.kind, f0.rows, f0.cols),
        (form_kind("trellis_k4"), 64, 128)
    );
    assert_eq!(
        (f1.kind, f1.suh.arena, f1.svh.arena),
        (form_kind("mxfp8"), NONE, NONE)
    );
    let conv = table.conventions[KV as usize];
    assert_eq!(conv.bytes as usize, size_of::<KvTables>());
    let kv: KvTables = read_at(&raw, conv.at.offset);
    let k0: Kept = read_at(&raw, kv.tables.offset);
    assert_eq!((kv.count, k0.base.arena, k0.base.offset), (1, USER, 0x7000));
    assert_eq!((k0.pages.arena, k0.pages.offset), (0, places.pages));
    assert_eq!(
        (k0.pages_per_slot, k0.page_rows, k0.row_bytes, k0.elem),
        (1, 16, 256, F32)
    );
}

#[test]
fn kv_page_modes() {
    let paged = sample_kept(Some(0xA000), 2, 32, false);
    let bufs = pack_bufs(&[], None, &[paged]);
    assert!(bufs.synth.is_empty());
    assert_eq!(
        (bufs.kept[0].pages.arena, bufs.kept[0].pages.offset),
        (USER, 0xA000)
    );
    let inter = sample_kept(None, 2, 3, true);
    assert_eq!(synth_pages(&[inter]), [0, 2, 4, 1, 3, 5]);
}

#[derive(Debug)]
struct Stub(Vec<Collective>);

impl Kernel for Stub {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn source(&self) -> Source {
        Source::Fixed {
            text: ByBackend::default(),
            entry: "stub",
        }
    }

    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }

    fn resident(&self) -> Option<Resident> {
        Some(Resident {
            threads: 64,
            shared: 0,
            scratch: 0,
            lanes: self.0.clone(),
        })
    }
}

#[test]
fn plans_collective_lanes() {
    let kernel = KernelRef::new(Stub(vec![Collective::Sum, Collective::SumBf16]));
    let layout = GridLayout {
        items: 1,
        forms: 0,
        kept: 0,
        pages: 0,
    };
    let planned = plan(&kernel, layout, 2, 1).unwrap();
    let got: Vec<(Op, u32)> = planned.lanes.iter().map(|l| (l.op, l.elem)).collect();
    assert_eq!(got, [(Op::Sum, F32), (Op::Sum, BF16)]);
    let sums = crate::sums::plan(4);
    let lanes: Vec<(Op, u32)> = sums.lanes.iter().map(|l| (l.op, l.elem)).collect();
    assert_eq!(lanes, [(Op::Sum, BF16), (Op::Gather, F32), (Op::Sum, F32)]);
    assert_eq!(sums.pools[0].send_bytes as usize, crate::sums::RELAY);
}

fn sample_bufs() -> TableBufs {
    let dense = sample_kept(None, 2, 16, false);
    pack_bufs(&sample_items(), Some((0x9000, 512)), &[dense])
}

#[test]
fn caches_refill_records() {
    let mut cache = RefillCache::default();
    let mut first = sample_bufs();
    assert!(cache.needs_refill(&first));
    cache.store(&mut first);
    assert!(!cache.needs_refill(&sample_bufs()));

    let mut diff_item = sample_bufs();
    diff_item.items[0].at.offset += 64;
    assert!(cache.needs_refill(&diff_item));

    let mut diff_form = sample_bufs();
    diff_form.forms[0].suh.offset += 64;
    assert!(cache.needs_refill(&diff_form));

    let mut diff_kept = sample_bufs();
    diff_kept.kept[0].base.offset += 64;
    assert!(cache.needs_refill(&diff_kept));

    let mut diff_pages = sample_bufs();
    diff_pages.synth[1] = 2;
    assert!(cache.needs_refill(&diff_pages));
    cache.store(&mut diff_pages);
    let mut updated = sample_bufs();
    updated.synth[1] = 2;
    assert!(!cache.needs_refill(&updated));
}

#[test]
fn sum_kernel_flattens() {
    let options = tessera_device::Options::plain(tessera_device::Backend::Cuda);
    let sum = tessera_device::flatten("sum.cu", include_str!("sum.cu"), &|_| None, &options);
    let text = sum.unwrap().text;
    assert!(text.contains("stage_bf16") && text.contains("cvt.rn.bf16x2.f32"));
}
