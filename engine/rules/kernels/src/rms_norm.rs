//! RMS norm over rows, generated.

use crate::per_row;
use tessera_codegen::prelude::*;
use tessera_lower::form::{F32_FORM, Load as Stored, plain};
use tessera_lower::{Draft, Kernel, T, Tree, sites};

/// One thread per row. Summing in order keeps batches alike.
#[kernel]
fn rms_norm<const WEIGHTED: bool>(
    x: Buf<F32>,
    #[form(w_form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    eps: f32,
    #[build] w_form: &'static dyn Stored,
) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += x[at + i as u64] * x[at + i as u64];
    }
    let inv = rsqrt(sum / width as f32 + eps);
    for i in 0..width {
        let gain = if WEIGHTED { w[i as u64] } else { 1.0 };
        out[at + i as u64] = x[at + i as u64] * inv * gain;
    }
}

/// One thread scales each row by its root mean square.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    rms_norm(self.w_form),
    variant = self.weighted,
    name = self.w_form.suffixed(match self.weighted {
        true => "rms_norm",
        false => "rms",
    }),
    threads = self.rows,
    accepts = rms_accepts,
    // The gains are read once per row.
    load = Some(per_row(
        self.rows,
        self.width,
        body.trip(),
        self.w_form.bytes(self.width) as f64 * f64::from(self.weighted),
    )),
    load_key = tessera_lower::load_key((self.rows, self.width, self.eps.to_bits(), self.weighted)),
)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
    pub w_form: &'static dyn Stored,
}

fn rms_accepts(k: &RmsNorm, input: usize, form: &dyn Stored) -> bool {
    if k.weighted {
        k.w_form.taken(input, 1, form)
    } else {
        !form.narrow()
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    let w_form = site
        .operand
        .map_or(F32_FORM, |(o, _)| plain(tree.node(o).form));
    let kernel = RmsNorm {
        rows: site.len / site.width,
        width: site.width,
        eps,
        weighted,
        w_form,
    };
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

tessera_lower::rule!("rms_norm", propose);
