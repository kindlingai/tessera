//! A matrix between two rotations equals the rotated matrix.

use checkpoint::exlr8::rotated;
use interp::{Tensor, run};
use models::Rotated;
use models::eg2::{Config, program};
use std::collections::BTreeMap;
use testkit::{Noise, cosine_f32, weights_with};
use trace::{Graph, Linear, PathSeg};

/// A weight path as `attention.0.q.w`.
pub(super) fn text(path: &[PathSeg]) -> String {
    let seg = |p: &PathSeg| match p {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    path.iter().map(seg).collect::<Vec<_>>().join(".")
}

fn shape(g: &Graph, w: &graph::Weight) -> Vec<usize> {
    let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
    w.ty.shape.0.iter().map(axis).collect()
}

/// Random weights for every path, scales near one.
fn random(g: &Graph, noise: &mut Noise) -> BTreeMap<String, Tensor> {
    let scales = |w: &graph::Weight| {
        let path = text(&w.path);
        match path.ends_with(".suh") || path.ends_with(".svh") {
            true => (1.0, 0.1),
            false => (0.0, 1.0),
        }
    };
    let tensors = weights_with(g, noise, scales);
    let paths = g.weights.iter().map(|w| text(&w.path));
    paths.zip(tensors).collect()
}

#[test]
fn rotations_fold_into_weights() {
    let mut c = Config::release(1, 16);
    c.stack.layers = 2;
    c.stack.vocab = 64;
    let turned = program::<Rotated>(&c);
    let plain = program::<Linear>(&c);
    let mut noise = Noise(5);
    let sides = random(&turned, &mut noise);
    let folded: Vec<Tensor> = (plain.weights.iter())
        .map(|w| {
            let path = text(&w.path);
            let Some(t) = sides.get(&format!("{path}.w")) else {
                return sides[&path].clone();
            };
            let (suh, svh) = (
                &sides[&format!("{path}.suh")],
                &sides[&format!("{path}.svh")],
            );
            let [out, inp] = shape(&plain, w)[..] else {
                panic!("a matrix");
            };
            let f = |t: &Tensor| t.floats().unwrap().to_vec();
            Tensor::f32(vec![out, inp], rotated(&f(t), inp, out, &f(suh), &f(svh)))
        })
        .collect();
    let turned_weights: Vec<Tensor> = turned
        .weights
        .iter()
        .map(|w| sides[&text(&w.path)].clone())
        .collect();
    let tokens = Tensor::u32(vec![1, 6], vec![3, 9, 27, 1, 14, 60]);
    let positions = Tensor::u32(vec![1, 6], (0..6).collect());
    let inputs = [tokens, positions];
    let a = run(&turned, &turned_weights, 0, &inputs).unwrap();
    let b = run(&plain, &folded, 0, &inputs).unwrap();
    let cos = cosine_f32(a[0].floats().unwrap(), b[0].floats().unwrap());
    assert!(cos > 0.99999, "rotated in the graph differs: cos {cos}");
}
