//! RMS norm, a thread per row in small groups.

use tessera_codegen::ir::{Body, Expr, Intrinsic, Stmt, Ty};
use tessera_codegen::prelude::*;
use tessera_codegen::stage::Row;
use tessera_lower::{Draft, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};
use tessera_lower::{GUESS, source};

/// The squares in one sum, then each element scaled.
#[kernel]
fn cpu_rms_norm(x: Buf<F32>, w: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32, eps: f32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let inv = rsqrt(x.dot(at, x, at, width) / width as f32 + eps);
    out.mix(at, 0.0, inv, x, at, width);
}

fn norm_body(weighted: bool) -> Body {
    let mut body = cpu_rms_norm_ir();
    if weighted {
        let args = vec![
            Expr::var("out"),
            Expr::var("at"),
            Expr::var("width"),
            Expr::lit(Row::Mul.code()),
            Expr::Float(1.0),
            Expr::var("w"),
            Expr::cast(Ty::U64, Expr::lit(0)),
        ];
        body.stmts
            .push(Stmt::Expr(Expr::Call(Intrinsic::Stage, args)));
    }
    body
}

/// Rows scaled by their root mean square and gains.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = "cpu_rms_norm",
    sizes = [self.rows, self.width],
    work = self.rows * self.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = source("cpu_rms_norm", norm_body(self.weighted)),
    launches = norm_launches,
)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

fn norm_launches(k: &RmsNorm, step: &Step) -> Vec<Launch> {
    // One thread per row splits small batches across workers.
    let shape = Shape::Grid {
        across: k.rows,
        down: 1,
        threads: 1,
    };
    let mut launch = Launch::plain(step, &[k.rows, k.width], shape);
    launch.args.push(tessera_lower::Arg::value(&k.eps));
    vec![launch]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    let kernel = RmsNorm {
        rows: site.len / site.width,
        width: site.width,
        eps,
        weighted,
    };
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

tessera_lower::rule!("cpu_rms_norm", propose);
