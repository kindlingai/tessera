use crate::layout::{
    ARENAS, BF16, BYTES, F32, FORM, Form, INPUT, Item, KV, Kept, KvTables, NONE, OUTPUT, Ref,
    SCRATCH, Table, form_kind,
};
use crate::resident::{Plan, USER, build_control};
use crate::whole::{format_summary, pack_records, prepare_source, synth_pages};
use tessera_gpukit::device::KernelStats;
use tessera_run::{FormMeta, Item as Placed, ItemKind, KeptDesc, marks};

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
    assert!(with.contains("regs=112 shared_bytes=16384 local_bytes=64"));
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

#[test]
fn packs_table_records() {
    let dense = sample_kept(None, 2, 16, false);
    let pages = synth_pages(&[dense]);
    assert_eq!(pages, [0, 1]);
    let (items, forms, kept) =
        pack_records(&sample_items(), Some((0x9000, 512)), &[dense], Ref::NONE).unwrap();
    let plan = Plan {
        pools: Vec::new(),
        lanes: Vec::new(),
        items,
        forms,
        kept,
        pages,
        arenas: vec![0],
        threads: 256,
        shared: 0,
        share: 1,
    };
    let (raw, places) = build_control(&plan, &[], [0, 1, 1], [0; ARENAS]);
    let table: Table = read_at(&raw, 0);
    assert_eq!((table.items_len, table.forms_len), (5, 2));
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
    assert!(synth_pages(&[paged]).is_empty());
    let (_, _, kept) = pack_records(&[], None, &[paged], Ref::NONE).unwrap();
    assert_eq!((kept[0].pages.arena, kept[0].pages.offset), (USER, 0xA000));
    let inter = sample_kept(None, 2, 3, true);
    assert_eq!(synth_pages(&[inter]), [0, 2, 4, 1, 3, 5]);
}
