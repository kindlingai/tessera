use crate::Noise;
use tessera_graph::{DType, DimExpr, DimName, Graph, TensorType, Weight};
use tessera_mem::Tensor;

/// The shape of `ty`, given `sizes`.
fn shape(g: &Graph, ty: &TensorType, sizes: &[(DimName, u64)]) -> Option<Vec<usize>> {
    let size = |name| sizes.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
    let mut out = Vec::new();
    for e in &ty.shape.0 {
        out.push(e.poly(&g.dims).ok()?.eval(&size)? as usize);
    }
    Some(out)
}

/// Mask words, then edits with sorted ids below 8.
fn edits(shape: &[usize], words: usize, noise: &mut Noise) -> Tensor {
    let k = (shape[shape.len() - 1] - words) / 3;
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let mut out = Vec::with_capacity(rows * (words + 3 * k));
    for _ in 0..rows {
        for _ in 0..words {
            out.push(((noise.draw() + 1.0) * 4.0) as u32);
        }
        let first = ((noise.draw() + 1.0) * 2.0) as u32;
        out.extend((0..k as u32).map(|j| first + 2 * j));
        for _ in 0..k {
            out.push(noise.draw().to_bits());
        }
        for _ in 0..k {
            out.push((1.5 + noise.draw() / 2.0).to_bits());
        }
    }
    Tensor::u32(shape.to_vec(), out)
}

/// Noise, small ids or edits for `ty`. `None` if unsized.
pub fn fill(
    g: &Graph,
    ty: &TensorType,
    sizes: &[(DimName, u64)],
    noise: &mut Noise,
) -> Option<Tensor> {
    let shape = shape(g, ty, sizes)?;
    if let Some(DimExpr::Add(mask, rest)) = ty.shape.0.last()
        && **rest == DimExpr::Dim(DimName::EditWords)
    {
        let size = |name| sizes.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
        let words = mask.poly(&g.dims).ok()?.eval(&size)? as usize;
        return Some(edits(&shape, words, noise));
    }
    let mut values = (0..shape.iter().product()).map(|_| noise.draw());
    Some(match ty.dtype {
        DType::U32 => Tensor::u32(shape, values.map(|v| ((v + 1.0) * 4.0) as u32).collect()),
        _ => Tensor::f32(shape, values.by_ref().collect()),
    })
}

/// Every weight as noise, in graph order.
pub fn weights(g: &Graph, noise: &mut Noise) -> Vec<Tensor> {
    weights_with(g, noise, |_| (0.0, 1.0))
}

/// Weight noise. `spread` gives offset and gain: `offset + gain * noise`.
pub fn weights_with(
    g: &Graph,
    noise: &mut Noise,
    spread: impl Fn(&Weight) -> (f32, f32),
) -> Vec<Tensor> {
    let mut out = Vec::new();
    for w in &g.weights {
        let shape = shape(g, &w.ty, &[]).expect("a weight with a fixed shape");
        let (offset, gain) = spread(w);
        let values = (0..shape.iter().product()).map(|_| offset + gain * noise.draw());
        out.push(Tensor::f32(shape, values.collect()));
    }
    out
}
