use crate::{DType, Error, Layout, Tensor, copy};

#[test]
fn layouts_walk() {
    let l = Layout::contiguous(vec![2, 3]);
    assert_eq!((l.strides.clone(), l.len(), l.span()), (vec![3, 1], 6, 6));
    assert!(l.is_contiguous());
    let t = l.permute(&[1, 0]);
    assert_eq!(
        (t.shape.clone(), t.strides.clone()),
        (vec![3, 2], vec![1, 3])
    );
    assert!(!t.is_contiguous());
    assert_eq!(t.offsets().collect::<Vec<_>>(), [0, 3, 1, 4, 2, 5]);
    let (s, skip) = l.slice(1, 1, 2);
    assert_eq!((s.shape.clone(), skip, s.span()), (vec![2, 2], 1, 5));
}

#[test]
fn views_read() {
    let t = Tensor::f32(vec![2, 3], (0..6).map(|i| i as f32).collect());
    let v = t.view();
    assert_eq!(v.as_f32().unwrap(), [0., 1., 2., 3., 4., 5.]);
    let swapped = v.permute(&[1, 0]);
    assert!(swapped.as_f32().is_err(), "strided views are not in place");
    assert_eq!(swapped.to_f32().unwrap(), [0., 3., 1., 4., 2., 5.]);
    let sliced = v.slice(1, 1, 2);
    assert_eq!(sliced.to_f32().unwrap(), [1., 2., 4., 5.]);
    assert_eq!(sliced.to_tensor().unwrap().shape, [2, 2]);
    assert_eq!(v.reshape(vec![6]).unwrap().shape(), [6]);
    assert!(v.reshape(vec![5]).is_err());
    assert!(matches!(v.as_u32(), Err(Error::Mismatch(_))));
    let ids = Tensor::u32(vec![2], vec![7, 9]);
    assert_eq!(ids.indices().unwrap(), [7, 9]);
    assert_eq!(ids.cast(DType::F32).unwrap().floats().unwrap(), [7.0, 9.0]);
}

#[test]
fn copies_same_type() {
    let src = Tensor::f32(vec![3], vec![1.0, 2.0, 3.5]);
    let mut wide = Tensor::zeros(vec![3], DType::F32);
    copy(src.view(), &mut wide.view_mut()).unwrap();
    assert_eq!(wide.floats().unwrap(), [1.0, 2.0, 3.5]);
    let mut narrow = Tensor::zeros(vec![3], DType::Bf16);
    assert!(copy(src.view(), &mut narrow.view_mut()).is_err());
    let short = Tensor::zeros(vec![2], DType::F32);
    assert!(copy(short.view(), &mut wide.view_mut()).is_err());
    wide.view_mut().as_f32_mut().unwrap()[0] = 9.0;
    assert_eq!(wide.clone(), wide);
    assert_ne!(wide, src);
    assert!(format!("{wide:?}").contains("9.0"));
}
