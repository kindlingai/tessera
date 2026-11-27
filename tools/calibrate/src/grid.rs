//! Small graphs that reach every rule, at many sizes.

use crate::{BURST, TRIES, choices, measure_with, pending, time_plans};
use tessera_engine::{Error, trellis};
use tessera_graph::DimName::{self, *};
use tessera_lower::form::{BF16_FORM, Load};
use tessera_lower::{Device, Pins, Settings, Step, Tensor, View, lower};
use tessera_run::{Run, open};
use tessera_testkit::Noise;
use tessera_testkit::graphs::{
    attention, blocks, chain, clipped, delta, indexed, indexed_rows, indexer, layer_norm, linear,
    mla_prep, neighbours, packed_attention, picked, residual_norm, rotation, routed, rows, table,
    turned,
};
use tessera_trace::{Dim, Graph, TensorType};

/// A timed plan holds at most this many elements.
const ROOM: usize = 750_000_000;

type Sizes = Vec<(DimName, u64)>;

/// The grid times every rule at these sizes.
pub struct Grid {
    /// Row counts, for everything that works on rows.
    pub rows: Vec<u64>,
    /// Hidden and inner widths.
    pub widths: Vec<u64>,
    pub vocabs: Vec<u64>,
    /// Heads, key heads and head width.
    pub heads: Vec<[u64; 3]>,
    /// Queries and keys of one attention.
    pub spans: Vec<[u64; 2]>,
    pub timing: Timing,
}

/// How long and how often the grid times each step.
pub struct Timing {
    /// Weight bytes one chain run reads, past current caches.
    pub streamed: u64,
    /// Timed tries per step.
    pub tries: usize,
    /// Launches back to back in one try, at most.
    pub burst: usize,
}

impl Timing {
    /// Enough tries and bursts for steady samples.
    const FULL: Self = Self {
        streamed: 128 << 20,
        tries: TRIES,
        burst: BURST,
    };
}

/// A graph, with each size to time it at.
struct Case {
    g: Graph,
    sizes: Vec<Sizes>,
    /// Weight forms, the same pins a checkpoint would set.
    pins: Pins,
    /// Whether it is a chain that times weights cold.
    cold: bool,
}

/// Every name bounded far past any size the grid uses.
pub(crate) fn bounded(names: &[DimName]) -> Vec<Dim> {
    names.iter().map(|n| Dim::bounded(*n, 1 << 20)).collect()
}

/// Noise for `ty` as one row. Ids stay below eight.
pub(crate) fn fill(
    g: &Graph,
    ty: &TensorType,
    sizes: &Sizes,
    noise: &mut Noise,
) -> Result<Tensor, Error> {
    let unsized_shape = || {
        Error::from(tessera_lower::Error::from(tessera_lower::Size::Shape(
            ty.shape.clone(),
        )))
    };
    tessera_testkit::fill(g, ty, sizes, noise).ok_or_else(unsized_shape)
}

impl Case {
    /// Times this case at each of its sizes that fits.
    fn run(
        &self,
        device: &mut Device,
        noise: &mut Noise,
        settings: &Settings,
        (tries, burst): (usize, usize),
    ) -> Result<Vec<(Step, f64)>, Error> {
        let mut runner: Option<Box<dyn Run>> = None;
        let mut added = Vec::new();
        let given = std::mem::replace(&mut device.pins, self.pins.clone());
        let guess = Device::guess(settings).pinned(self.pins.clone());
        for sizes in &self.sizes {
            let plan = lower(&self.g, 0, sizes, &guess)?;
            if plan.buffers.iter().sum::<usize>() > ROOM {
                continue;
            }
            // Skip opening a runner when every competing plan is known.
            let pre = match device.name.is_empty() {
                true => None,
                false => Some(pending((&self.g, 0), sizes, device, settings, self.cold)?),
            };
            if pre.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            let runner = match &mut runner {
                Some(runner) => runner,
                None => {
                    let first = pre.as_ref().map_or(plan, |p| p[0].clone());
                    runner.insert(opened(&self.g, first, noise, settings)?)
                }
            };
            let ports = self.g.entries[0].inputs.iter();
            let fed: Result<Vec<Tensor>, Error> =
                ports.map(|p| fill(&self.g, &p.ty, sizes, noise)).collect();
            let fed = fed?;
            let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
            let timed = match pre {
                Some(plans) if device.platform == runner.platform() => time_plans(
                    &mut **runner,
                    &plans,
                    &feeds,
                    device,
                    (self.cold, tries, burst),
                )?,
                _ => measure_with(
                    &mut **runner,
                    (&self.g, 0),
                    sizes,
                    &feeds,
                    device,
                    (settings, self.cold),
                    (tries, burst),
                )?,
            };
            added.extend(timed);
        }
        device.pins = given;
        Ok(added)
    }
}

/// A runner for `plan` with noise for weights.
pub(crate) fn opened(
    g: &Graph,
    plan: tessera_lower::Plan,
    noise: &mut Noise,
    settings: &Settings,
) -> Result<Box<dyn Run>, Error> {
    let mut runner = open(plan, settings)?;
    for (at, w) in g.weights.iter().enumerate() {
        let at = at as u32;
        let data = fill(g, &w.ty, &Vec::new(), noise)?;
        // Releases write trellis codes. Here they are noise.
        let loaded = runner.plan().weights.iter().find(|l| l.weight == at);
        match loaded.map(|l| l.slot.form).filter(|f| !f.packs()) {
            Some(form) => runner.load_packed(at, &codes(form, &data, noise))?,
            None => runner.load(at, data.view())?,
        }
    }
    Ok(runner)
}

/// Noise trellis codes for a weight shaped as `data`.
pub(crate) fn codes(form: &dyn Load, data: &Tensor, noise: &mut Noise) -> Vec<u8> {
    let [inp, out] = [1, 2].map(|end| data.shape[data.shape.len() - end]);
    let side = trellis::INPUT_SIDE;
    let head = trellis::packed(inp, out, side, &[]).len();
    let len = form.bytes(data.len()) - head;
    let codes: Vec<u8> = (0..len)
        .map(|_| (noise.draw() * 127.0) as i8 as u8)
        .collect();
    trellis::packed(inp, out, side, &codes)
}

impl Grid {
    /// Sizes from one decoded row to a long prompt.
    pub fn full() -> Self {
        Self {
            rows: vec![1, 4, 16, 64, 256, 1024],
            widths: vec![256, 1024, 4096],
            vocabs: vec![32_768, 131_072],
            heads: vec![[16, 8, 128], [8, 8, 64]],
            spans: vec![
                [1, 16],
                [1, 128],
                [1, 1024],
                [16, 16],
                [128, 128],
                [512, 512],
            ],
            timing: Timing::FULL,
        }
    }

    /// Sizes where a host may beat its GPU.
    pub fn host() -> Self {
        Self {
            rows: vec![1, 4, 16, 64],
            widths: vec![256, 1024, 2048],
            vocabs: vec![32_768],
            heads: vec![[16, 8, 64]],
            spans: vec![[1, 16], [1, 128], [16, 16], [64, 64]],
            timing: Timing::FULL,
        }
    }

    /// Enough to reach every rule, with short chains.
    pub fn tiny() -> Self {
        Self {
            rows: vec![1, 3],
            widths: vec![16],
            vocabs: vec![64],
            heads: vec![[4, 2, 16]],
            spans: vec![[1, 5], [4, 4]],
            timing: Timing {
                streamed: 4 << 10,
                tries: 1,
                burst: 1,
            },
        }
    }

    fn cases(&self) -> Vec<Case> {
        let pinned = |g: Graph, pins: Pins| {
            let sizes = self.rows.iter().map(|r| vec![(Rows, *r)]).collect();
            Case {
                g,
                sizes,
                pins,
                cold: false,
            }
        };
        let by_rows = |g: Graph| pinned(g, Pins::default());
        let (one, three) = (bounded(&[Rows]), bounded(&[Seqs, Rows, Context]));
        let mut cases = Vec::new();
        for &k in &self.widths {
            // Checkpoints mostly store bf16 weights.
            for &n in self.widths.iter().chain(&self.vocabs) {
                cases.push(by_rows(linear(&one, k, n)));
                cases.push(pinned(linear(&one, k, n), Pins::all(BF16_FORM)));
            }
        }
        // Chains read more than any cache, for cold samples.
        for (i, &k) in self.widths.iter().enumerate() {
            for &n in &self.widths[i..] {
                for (pins, bytes) in [(Pins::default(), 4), (Pins::all(BF16_FORM), 2)] {
                    let weight = k * n * bytes;
                    // A weight this large is cold in its own sample.
                    if 4 * weight > self.timing.streamed {
                        continue;
                    }
                    let pairs = (self.timing.streamed / (2 * weight)).clamp(1, 32);
                    let mut case = pinned(chain(&one, k, n, pairs), pins);
                    case.cold = true;
                    cases.push(case);
                }
            }
        }
        cases.push(by_rows(rotation(&one)));
        cases.push(by_rows(turned(&one)));
        cases.push(by_rows(indexed_rows(&one, [2, 16, 2])));
        let mla_sizes = self
            .rows
            .iter()
            .map(|r| vec![(Rows, *r), (Context, 8)])
            .collect();
        cases.push(Case {
            g: mla_prep(&one, [4, 16, 16]),
            sizes: mla_sizes,
            pins: Pins::default(),
            cold: false,
        });
        cases.push(pinned(blocks(&one, [2, 16, 16]), Pins::all(BF16_FORM)));
        cases.extend(
            self.widths
                .iter()
                .map(|&k| by_rows(routed(&one, [8, k, k]))),
        );
        // Trellis experts, for picked trellis kernels.
        let codes = Pins::default().pin("set", &trellis::TRELLIS_K3_PMCGX);
        cases.push(pinned(picked(&one, [4, 128, 128]), codes));
        for &width in self.widths.iter().chain(&self.vocabs) {
            cases.push(by_rows(rows(&one, width)));
            cases.push(by_rows(neighbours(&one, [16, 9, width])));
            cases.extend((self.vocabs.iter()).map(|v| by_rows(table(&one, *v, width))));
        }
        for &width in &self.widths {
            cases.push(by_rows(residual_norm(&one, width)));
            cases.push(by_rows(clipped(&one, width)));
            cases.push(by_rows(layer_norm(&one, width)));
        }
        for &heads in &self.heads {
            cases.push(by_rows(packed_attention(&one, heads)));
            let span =
                |[queries, keys]: &[u64; 2]| vec![(Seqs, 1), (Rows, *queries), (Context, *keys)];
            let sizes = self.spans.iter().map(span).collect();
            let g = attention(&three, heads);
            let pins = Pins::default();
            cases.push(Case {
                g,
                sizes,
                pins,
                cold: false,
            });
            // Picked keys, two kept, and the delta rule: whole sequences.
            let span = |[_, keys]: &[u64; 2]| vec![(Seqs, 1), (Context, *keys)];
            let sizes: Vec<Sizes> = self.spans.iter().map(span).collect();
            for g in [
                indexed(&three, [heads[0], heads[2], 2]),
                delta(&three, heads),
                indexer(&three, [heads[0], heads[2]]),
            ] {
                let (sizes, pins) = (sizes.clone(), Pins::default());
                cases.push(Case {
                    g,
                    sizes,
                    pins,
                    cold: false,
                });
            }
        }
        cases
    }

    /// Each plan `run` times from `guess`, per competing rule.
    pub fn plans(&self, guess: &dyn Fn() -> Device) -> Result<Vec<tessera_lower::Plan>, Error> {
        let mut plans = Vec::new();
        for case in self.cases() {
            let guess = || guess().pinned(case.pins.clone());
            for sizes in &case.sizes {
                for choice in choices((&case.g, 0), sizes, &guess)? {
                    plans.push(lower(&case.g, 0, sizes, &choice)?);
                }
            }
        }
        Ok(plans)
    }

    /// Times every case on this GPU. Returns the launches timed.
    pub fn run(&self, device: &mut Device, settings: &Settings) -> Result<Vec<(Step, f64)>, Error> {
        let mut noise = Noise(17);
        let mut added = Vec::new();
        for case in self.cases() {
            added.extend(case.run(
                device,
                &mut noise,
                settings,
                (self.timing.tries, self.timing.burst),
            )?);
        }
        Ok(added)
    }
}
