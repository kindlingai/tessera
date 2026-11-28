use tessera_mem::{DType, Layout, Tensor};

fn ref_offsets(layout: &Layout) -> Vec<usize> {
    let rank = layout.shape.len();
    (0..layout.len())
        .map(|mut flat| {
            let mut at = 0;
            for d in (0..rank).rev() {
                at += (flat % layout.shape[d]) * layout.strides[d];
                flat /= layout.shape[d];
            }
            at
        })
        .collect()
}

#[test]
fn contiguous_edges() {
    assert!(Layout::contiguous(vec![]).is_contiguous());
    assert!(Layout::contiguous(vec![5]).is_contiguous());
    assert!(Layout::contiguous(vec![2, 0, 3]).is_contiguous());
    assert!(Layout::contiguous(vec![1, 4]).is_contiguous());
    assert!(
        !(Layout {
            shape: vec![1, 4],
            strides: vec![1, 1],
        })
        .is_contiguous()
    );
    assert!(
        !(Layout {
            shape: vec![2, 3],
            strides: vec![3],
        })
        .is_contiguous()
    );
    assert!(
        !(Layout {
            shape: vec![2, 0, 3],
            strides: vec![1, 3, 1],
        })
        .is_contiguous()
    );
    assert!(Layout::contiguous(vec![2, 0, 3]).is_empty());
    assert!(!Layout::contiguous(vec![]).is_empty());
}

#[test]
fn offsets_match_reference() {
    let cases = vec![
        Layout::contiguous(vec![]),
        Layout::contiguous(vec![2, 0, 3]),
        Layout {
            shape: vec![5],
            strides: vec![3],
        },
        Layout::contiguous(vec![2, 3]).permute(&[1, 0]),
        Layout {
            shape: vec![2, 1, 3, 2],
            strides: vec![100, 999, 0, 7],
        },
        // Rank 6 exercises heap-backed odometer coordinates.
        Layout::contiguous(vec![2, 3, 2, 2, 2, 3]).permute(&[2, 0, 4, 1, 5, 3]),
    ];
    for layout in cases {
        let want = ref_offsets(&layout);
        // Exercise Iterator::next via a plain for-loop.
        let mut via_next = Vec::new();
        for v in layout.offsets() {
            via_next.push(v);
        }
        assert_eq!(via_next, want);
        // Exercise Iterator::fold via fold accumulation.
        let via_fold = layout.offsets().fold(Vec::new(), |mut acc, v| {
            acc.push(v);
            acc
        });
        assert_eq!(via_fold, want);
    }
}

#[test]
fn packed_chunked_views() {
    let base = Layout::contiguous(vec![2, 3, 4]);
    let data: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    let t = Tensor::f32(vec![2, 3, 4], data);
    // Permuted outer axes with contiguous inner row.
    let p = t.view().permute(&[1, 0, 2]);
    let want: Vec<f32> = ref_offsets(&base.permute(&[1, 0, 2]))
        .into_iter()
        .map(|i| i as f32)
        .collect();
    assert_eq!(p.to_f32().unwrap(), want);
    // Sliced along innermost axis, contiguous sub-rows.
    let s = t.view().slice(2, 1, 2);
    let (sl, skip) = base.slice(2, 1, 2);
    let want_s: Vec<f32> = ref_offsets(&sl)
        .into_iter()
        .map(|i| (i + skip) as f32)
        .collect();
    assert_eq!(s.to_f32().unwrap(), want_s);
    // Permuted inner axis: non-contiguous unit chunks.
    let q = t.view().permute(&[2, 0, 1]);
    let want_q: Vec<f32> = ref_offsets(&base.permute(&[2, 0, 1]))
        .into_iter()
        .map(|i| i as f32)
        .collect();
    assert_eq!(q.to_f32().unwrap(), want_q);
    // Empty strided slice returns empty packed bytes.
    let z = Tensor::zeros(vec![2, 3, 4], DType::F32);
    assert!(z.view().slice(1, 0, 0).packed().unwrap().is_empty());
}
