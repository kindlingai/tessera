use crate::Row;
use crate::kept::Kept;
use crate::lanes::Lanes;

fn row(slot: u32, pos: u32) -> Row {
    Row {
        slot,
        token: pos,
        position: pos,
    }
}

#[test]
fn grow_rows_matches_single() {
    let (slots, context, pages, page) = (4, 128, 32, 16);
    let mut one = Kept::paged(slots, context, pages, page);
    let mut batched = Kept::paged(slots, context, pages, page);
    let mut lanes_one = Lanes::new(page, slots * (context / page), slots, pages);
    let mut lanes_batched = Lanes::new(page, slots * (context / page), slots, pages);
    let (s0, _) = one.admit(&[1, 2], 0).unwrap();
    let (s1, _) = one.admit(&[3, 4], 0).unwrap();
    assert_eq!(batched.admit(&[1, 2], 0).unwrap().0, s0);
    assert_eq!(batched.admit(&[3, 4], 0).unwrap().0, s1);
    let mut rows: Vec<Row> = (0..40).map(|p| row(s0, p)).collect();
    rows.extend((0..20).map(|p| row(s1, p)));
    rows.push(row(s0, 40));
    for r in &rows {
        if let Some(ps) = one.grow(r.slot, r.position as usize + 1).unwrap() {
            lanes_one.set(r.slot, ps);
        }
    }
    batched.grow_rows(&rows, &mut lanes_batched).unwrap();
    assert_eq!(lanes_batched.pending(), lanes_one.pending());
    assert_eq!(batched.totals(), one.totals());
}

#[test]
fn tier_demote_reuse() {
    let (slots, context, page) = (2, 128, 16);
    let (kv_pages, idx_pages) = (4, 16);
    let mut lanes = Lanes::new_split(page, slots * (context / page), slots, kv_pages, idx_pages);
    lanes.enable_tier(idx_pages, Box::new(tessera_kv::Aged::default()));
    assert!(lanes.tiered());
    assert_eq!(lanes.pages(), idx_pages);
    let mut kept = Kept::paged(slots, context, lanes.pages(), page);
    let (mut demotes, mut promotes) = (Vec::new(), Vec::new());
    let (s0, _) = kept.admit(&[1; 80], 0).unwrap();
    let rows0: Vec<Row> = (0..80).map(|p| row(s0, p)).collect();
    // Five pages exceed four VRAM pages, demoting older pages.
    kept.grow_rows_tiered(&rows0[..64], &mut lanes, &mut |up, p| {
        if up {
            promotes.extend_from_slice(p);
        } else {
            demotes.extend_from_slice(p);
        }
        Ok(true)
    })
    .unwrap();
    assert!(demotes.is_empty());
    kept.grow_rows_tiered(&rows0[64..80], &mut lanes, &mut |up, p| {
        if up {
            promotes.extend_from_slice(p);
        } else {
            demotes.extend_from_slice(p);
        }
        Ok(true)
    })
    .unwrap();
    assert!(!demotes.is_empty());
    assert!(promotes.is_empty());
    let kv_tbl = lanes.pending()[0].1.to_vec();
    let host_bit = 1u32 << 30;
    assert!(kv_tbl[..5].iter().any(|&e| (e & host_bit) != 0));
    assert_eq!(kv_tbl[4] & host_bit, 0, "written tail page stays in VRAM");

    // Prefix reuse keeps host pages in host RAM.
    kept.finish(s0, (0..80).collect());
    let prompt: Vec<u32> = (0..80).chain([99]).collect();
    let (s1, skip) = kept.admit(&prompt, 80).unwrap();
    assert_eq!(skip, 80);
    let before_promotes = promotes.len();
    kept.grow_rows_tiered(&[row(s1, 80)], &mut lanes, &mut |up, p| {
        if up {
            promotes.extend_from_slice(p);
        } else {
            demotes.extend_from_slice(p);
        }
        Ok(true)
    })
    .unwrap();
    assert_eq!(promotes.len(), before_promotes);
}
