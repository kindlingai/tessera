use crate::cpu_test::{model, port};
use crate::fold::frontier;
use crate::*;
use sensai_trace::DType::F32;
use sensai_trace::DimName::Hidden;
use sensai_trace::{EntryKind, PathSeg, Role, TensorType};

#[test]
fn folds_weights() {
    let m = model();
    let [x] = m.entry(EntryKind::Forward, [port([Hidden], F32)]);
    let path = [PathSeg::Field("w".into())];
    let w = m.weight(&path, TensorType::new([Hidden], F32));
    let g = m.finish([(Role::Logits, &x.add(&w.sin().scale(2.0)))]);
    let weights = [Tensor::f32(vec![4], vec![0.0, 0.5, 1.0, -2.0])];
    let input = [Tensor::f32(vec![4], vec![1.0, 2.0, 3.0, 4.0])];
    let (folded, all) = fold(&g, &weights).unwrap();
    assert_eq!(all.len(), 2, "one value joins the weights");
    let want = run(&g, &weights, 0, &input).unwrap();
    let got = run(&folded, &all, 0, &input).unwrap();
    assert_eq!(got[0].floats().unwrap(), want[0].floats().unwrap());
    // No weight-only work is left for a plan.
    assert!(frontier(&folded).is_empty());
}
