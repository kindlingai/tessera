//! Generic Pins-driven load-time weight repacking for streamed bundles.

use crate::names::Rule;
use crate::{Failure, Stream, numbered};
use tessera_checkpoint::Offer;
use tessera_engine::form::{Load, named};
use tessera_engine::{DimExpr, Graph, PathSeg, Pins, Reads, Weight};

/// Target form and 2D matrix alignment for load-time repacking.
#[derive(Clone, Copy)]
pub(crate) struct Repack {
    pub form: &'static dyn Load,
    /// Required `[rows, cols]` divisibility for 2D weights.
    pub align: [usize; 2],
}

impl Repack {
    /// Repack into `form` using its own storage block alignment.
    #[cfg(test)]
    pub(crate) fn into(form: &'static dyn Load) -> Self {
        Self {
            form,
            align: [1, form.align()],
        }
    }

    /// Repack into MXFP8 with `fast_mm` tile alignment.
    pub(crate) fn mxfp8() -> Option<Self> {
        Some(Self {
            form: named("mxfp8")?,
            align: [16, 512],
        })
    }

    /// Whether a 2D `[rows, cols]` matrix satisfies this policy.
    pub(crate) fn fits(self, [rows, cols]: [usize; 2]) -> bool {
        self.form.packs()
            && rows.is_multiple_of(self.align[0].max(1))
            && cols.is_multiple_of(self.align[1].max(1))
    }
}

/// Fixed 2D `[rows, cols]` shape of a weight, if static.
pub(crate) fn fixed_2d(graph: &Graph, shape: &[DimExpr]) -> Option<[usize; 2]> {
    if shape.len() != 2 {
        return None;
    }
    let val = |d: &DimExpr| d.value(&graph.dims, &|_| None).map(|n| n as usize);
    Some([val(&shape[0])?, val(&shape[1])?])
}

/// Resolves checkpoint pins, upgrading selected float weights to `target`.
pub(crate) fn pins(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
    target: Option<Repack>,
    select: impl Fn(&[PathSeg]) -> bool,
) -> Result<Pins, Failure> {
    let slices = offer.slices();
    let claims = crate::names::resolve(graph, &slices, rules)?;
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(claims.len());
    for (c, w) in claims.iter().zip(&graph.weights) {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let name = slice.stored.name();
        let upgrade = target.filter(|t| {
            matches!(name, "bf16" | "f32" | "f16")
                && name != t.form.name()
                && c.parts.is_empty()
                && c.rows.is_none()
                && select(&w.path)
                && fixed_2d(graph, &w.ty.shape.0).is_some_and(|dims| t.fits(dims))
        });
        match upgrade {
            Some(t) => forms.push(t.form),
            None => forms.push(crate::form(slice.stored, !c.parts.is_empty())?),
        }
    }
    Ok(crate::gguf::pins(graph, &forms, layers))
}

/// Streamed weights from any `Offer`, repacking weights to match `pins`.
pub(crate) struct Streamed {
    pub(crate) graph: Graph,
    pub(crate) offer: Box<dyn Offer>,
    pub(crate) rules: Vec<Rule>,
    pub(crate) pins: Pins,
}

impl Stream for Streamed {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<(), Failure>,
    ) -> Result<(), Failure> {
        stream(
            &self.graph,
            &*self.offer,
            &self.rules,
            &self.pins,
            shares,
            take,
        )
    }
}

/// Streams weights from `offer`, repacking any weight whose pin differs.
pub(crate) fn stream(
    graph: &Graph,
    offer: &dyn Offer,
    rules: &[Rule],
    pins: &Pins,
    shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
    take: &mut dyn FnMut(u32, Weight) -> Result<(), Failure>,
) -> Result<(), Failure> {
    numbered(graph, offer, &[rules], shares, &mut |at, w| {
        let weight = &graph.weights[at as usize];
        let want = pins.form(&weight.path);
        let dims = fixed_2d(graph, &weight.ty.shape.0);
        let repacked = repack_weight(w, dims, shares(at), want)?;
        take(at, repacked)
    })
}

/// Source view for row-parallel unpacking into f32 batches.
#[derive(Clone, Copy)]
enum Source<'a> {
    Bf16(&'a [u8]),
    Floats(&'a [f32]),
    Form(&'static dyn Load, &'a [u8]),
}

impl Source<'_> {
    /// Copies `count` elements from `row` starting at `col` into `dst`.
    fn row(&self, row: usize, col: usize, stride: usize, dst: &mut [f32]) {
        let first = row * stride + col;
        match *self {
            Source::Bf16(bytes) => {
                let raw = &bytes[first * 2..(first + dst.len()) * 2];
                for (o, &pair) in dst.iter_mut().zip(raw.as_chunks::<2>().0) {
                    *o = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                }
            }
            Source::Floats(vals) => {
                dst.copy_from_slice(&vals[first..first + dst.len()]);
            }
            Source::Form(form, bytes) => {
                for (i, o) in dst.iter_mut().enumerate() {
                    *o = form.unpack(bytes, first + i);
                }
            }
        }
    }
}

/// Repacks a 2D weight slice into `target` across CPU threads.
pub(crate) fn repack_weight(
    w: Weight,
    dims: Option<[usize; 2]>,
    share: Option<(u32, usize, usize)>,
    target: &'static dyn Load,
) -> Result<Weight, Failure> {
    if !target.packs() {
        return Ok(w);
    }
    let same = match &w {
        Weight::Floats(_) => target.name() == "f32",
        Weight::Packed(src, _) | Weight::Filed(src, _) => src.name() == target.name(),
    };
    if same {
        return Ok(w);
    }
    let held = match w.held(Reads::default())? {
        Some(loaded) => loaded,
        None => w,
    };
    let Some([rows, cols]) = dims else {
        return Ok(held);
    };
    let (r0, r_cnt, c0, c_cnt) = match share {
        Some((0, first, count)) => (first, count, 0, cols),
        Some((1, first, count)) => (0, rows, first, count),
        _ => (0, rows, 0, cols),
    };
    let (src, len_elems) = match &held {
        Weight::Packed(form, bytes) if form.name() == "bf16" => {
            (Source::Bf16(bytes), bytes.len() / 2)
        }
        Weight::Packed(form, bytes) => (Source::Form(*form, bytes), form.elements(bytes.len())),
        Weight::Floats(t) => {
            let vals = t.floats().map_err(|e| e.to_string())?;
            (Source::Floats(vals), vals.len())
        }
        Weight::Filed(..) => unreachable!("filed weights materialized above"),
    };
    // Pre-sliced shares already hold only `[r_cnt, c_cnt]`.
    let (r0, c0, stride) = if share.is_some() && len_elems == r_cnt * c_cnt {
        (0, 0, c_cnt)
    } else {
        (r0, c0, cols)
    };
    Ok(Weight::Packed(
        target,
        repack_2d(src, stride, [r0, r_cnt, c0, c_cnt], target),
    ))
}

/// Packs a `[r_cnt, c_cnt]` window into `target` in parallel.
fn repack_2d(
    src: Source<'_>,
    stride: usize,
    [r0, r_cnt, c0, c_cnt]: [usize; 4],
    target: &'static dyn Load,
) -> Vec<u8> {
    if r_cnt == 0 || c_cnt == 0 {
        return Vec::new();
    }
    if target.bytes(0) != 0 || !c_cnt.is_multiple_of(target.block()) {
        let mut all = vec![0.0f32; r_cnt * c_cnt];
        for r in 0..r_cnt {
            src.row(r0 + r, c0, stride, &mut all[r * c_cnt..(r + 1) * c_cnt]);
        }
        return target.pack(&all);
    }
    let row_out = target.bytes(c_cnt);
    let mut out = vec![0u8; r_cnt * row_out];
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(r_cnt);
    let per = r_cnt.div_ceil(threads);
    std::thread::scope(|scope| {
        for (ti, chunk) in out.chunks_mut(per * row_out).enumerate() {
            let start_r = r0 + ti * per;
            let count_r = chunk.len() / row_out;
            scope.spawn(move || {
                let batch = (65_536 / c_cnt.max(1)).clamp(1, count_r);
                let mut buf = vec![0.0f32; batch * c_cnt];
                let mut done = 0;
                while done < count_r {
                    let take = batch.min(count_r - done);
                    let floats = &mut buf[..take * c_cnt];
                    for dr in 0..take {
                        let dst = &mut floats[dr * c_cnt..(dr + 1) * c_cnt];
                        src.row(start_r + done + dr, c0, stride, dst);
                    }
                    let packed = target.pack(floats);
                    chunk[done * row_out..(done + take) * row_out].copy_from_slice(&packed);
                    done += take;
                }
            });
        }
    });
    out
}
