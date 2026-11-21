//! Whole-plan speed over shapes many models share, for comparisons.

use crate::grid::{bounded, fill, opened};
use std::time::Instant;
use tessera_engine::Error;
use tessera_graph::DimName::{self, Context, Rows, Seqs};
use tessera_graph::Graph;
use tessera_lower::{Device, Settings, Tensor, View, lower};
use tessera_testkit::Noise;
use tessera_testkit::graphs::{attention, linear};

/// Rows, depth and width of each matmul timed.
const GEMMS: [[u64; 3]; 20] = [
    [1, 512, 2048],
    [4, 512, 2048],
    [16, 512, 2048],
    [200, 512, 2048],
    [2048, 512, 2048],
    [1, 2048, 512],
    [4, 2048, 512],
    [16, 2048, 512],
    [200, 2048, 512],
    [2048, 2048, 512],
    [1, 4096, 4096],
    [4, 4096, 4096],
    [16, 4096, 4096],
    [200, 4096, 4096],
    [2048, 4096, 4096],
    // A width past every whole block.
    [1, 512, 1000],
    [4, 512, 1000],
    [16, 512, 1000],
    [200, 512, 1000],
    [2048, 512, 1000],
];

/// Heads, key heads and head width of each attention timed.
const HEADS: [[u64; 3]; 3] = [[8, 8, 64], [16, 4, 128], [4, 2, 256]];
/// Queries and keys of each attention timed.
const SPANS: [[u64; 2]; 4] = [[1, 256], [1, 4096], [200, 200], [512, 2048]];

/// Timing stops after this many seconds of one shape.
const BUDGET: f64 = 0.3;
const MOST: usize = 20;

/// One shape's best time, and the floating-point work it does.
pub struct Mark {
    pub name: String,
    pub seconds: f64,
    pub flops: f64,
}

/// `g` at `sizes`, planned as this machine would, timed whole.
fn timed(g: &Graph, sizes: &[(DimName, u64)], settings: &Settings) -> Result<f64, Error> {
    let mut noise = Noise(7);
    let plan = lower(g, 0, sizes, &Device::guess(settings))?;
    let mut runner = opened(g, plan, &mut noise, settings)?;
    // Stored samples choose kernels, as a model run does.
    let device =
        Device::stored(runner.platform(), settings).unwrap_or_else(|_| Device::guess(settings));
    let plan = lower(g, 0, sizes, &device)?;
    runner.replan(&plan)?;
    let sizes = sizes.to_vec();
    let ports = g.entries[0].inputs.iter();
    let fed: Result<Vec<Tensor>, Error> =
        ports.map(|p| fill(g, &p.ty, &sizes, &mut noise)).collect();
    let fed = fed?;
    let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
    runner.feed(&feeds)?;
    let steps = 0..plan.steps.len();
    runner.submit(steps.clone())?;
    let (mut best, mut spent) = (f64::INFINITY, 0.0);
    for _ in 0..MOST {
        let start = Instant::now();
        let clocked = runner.clocked(steps.clone())?;
        let seconds = clocked.unwrap_or_else(|| start.elapsed().as_secs_f64());
        best = best.min(seconds);
        spent += seconds;
        if spent > BUDGET {
            break;
        }
    }
    Ok(best)
}

/// Every shape, timed on the backend `settings` pick.
pub fn bench(settings: &Settings) -> Result<Vec<Mark>, Error> {
    let mut marks = Vec::new();
    let one = bounded(&[Rows]);
    for [m, k, n] in GEMMS {
        let seconds = timed(&linear(&one, k, n), &[(Rows, m)], settings)?;
        let name = format!("gemm m={m} k={k} n={n}");
        let flops = 2.0 * (m * k * n) as f64;
        marks.push(Mark {
            name,
            seconds,
            flops,
        });
    }
    let three = bounded(&[Seqs, Rows, Context]);
    for heads in HEADS {
        let g = attention(&three, heads);
        for [queries, keys] in SPANS {
            let sizes = [(Seqs, 1), (Rows, queries), (Context, keys)];
            let seconds = timed(&g, &sizes, settings)?;
            let [h, kv, width] = heads;
            let name = format!("attend h={h}/{kv}x{width} q={queries} k={keys}");
            let flops = 4.0 * (h * queries * keys * width) as f64;
            marks.push(Mark {
                name,
                seconds,
                flops,
            });
        }
    }
    Ok(marks)
}

/// Each mark as a name and seconds, a line each.
pub fn text(marks: &[Mark]) -> String {
    let line = |m: &Mark| format!("{}\t{:.6e}\n", m.name, m.seconds);
    marks.iter().map(line).collect()
}

/// Marks slower than `base` by more than `slack`.
pub fn slower(marks: &[Mark], base: &str, slack: f64) -> Vec<String> {
    let mut out = Vec::new();
    for line in base.lines() {
        let Some((name, seconds)) = line.split_once('\t') else {
            continue;
        };
        let Ok(was) = seconds.trim().parse::<f64>() else {
            continue;
        };
        let now = marks.iter().find(|m| m.name == name);
        if let Some(m) = now.filter(|m| m.seconds > was * (1.0 + slack)) {
            out.push(format!("{name}: {:.3e} s, was {was:.3e} s", m.seconds));
        }
    }
    out
}

/// Shapes timed before and after a calibration, to see drift.
const SENTINELS: [[u64; 3]; 2] = [[16, 4096, 4096], [1, 4096, 4096]];

/// Each sentinel shape's best whole-plan time.
pub fn sentinels(settings: &Settings) -> Result<Vec<f64>, Error> {
    let one = bounded(&[Rows]);
    let mut seconds = Vec::new();
    for [m, k, n] in SENTINELS {
        seconds.push(timed(&linear(&one, k, n), &[(Rows, m)], settings)?);
    }
    Ok(seconds)
}

/// Sentinel sets timed again until two agree, so clocks settle.
pub fn settled(settings: &Settings) -> Result<Vec<f64>, Error> {
    let mut last = sentinels(settings)?;
    for _ in 0..SETTLE {
        let now = sentinels(settings)?;
        if drift(&last, &now) < AGREE {
            return Ok(now);
        }
        last = now;
    }
    Ok(last)
}

/// Most sentinel sets `settled` times, and the agreement it needs.
const SETTLE: usize = 20;
const AGREE: f64 = 0.02;

/// The largest change between two sentinel runs, as a share.
pub fn drift(first: &[f64], last: &[f64]) -> f64 {
    let change = first.iter().zip(last).map(|(a, b)| (b / a - 1.0).abs());
    change.fold(0.0, f64::max)
}
