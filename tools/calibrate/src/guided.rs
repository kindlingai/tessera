//! Guided decode steps: full head, mask and sparse head.

use std::time::Instant;
use tessera_engine::form::BF16_FORM;
use tessera_engine::{Allowed, Engine, Error, FEW, Pins, Recipe, Row, Settings, Weight};
use tessera_models::decoder::{self, Config, Swiglu};
use tessera_testkit::{Noise, weights_with};
use tessera_trace::Norm;

/// Rows each slot keeps.
const CONTEXT: u64 = 256;
const PROMPT: usize = 16;
/// Steps timed in each pass.
const STEPS: usize = 32;
/// Timed passes, after one that compiles.
const PASSES: usize = 8;
/// Candidate counts timed, up to `FEW`.
const KS: [usize; 6] = [1, 8, 32, 128, 512, 4096];
/// Sequences stepped side by side.
const PICKS: [u64; 2] = [1, 16];

/// A model's name, and its sizes for sequences and context.
type Shape = (&'static str, fn(u64, u64) -> Config);

const SHAPES: [Shape; 2] = [
    ("qwen3-0.6b", Config::qwen3),
    ("gemma3-270m", Config::gemma3),
];

/// Choices for each pick.
#[derive(Clone, Copy)]
enum Case {
    /// A graph with no mask port.
    Plain,
    /// The masked graph, with no guide.
    Open,
    /// Half the vocabulary, through the full head.
    Bits,
    /// `k` ids, through the sparse head.
    Few(usize),
}

impl Case {
    fn label(self) -> String {
        match self {
            Case::Plain => "unmasked graph".into(),
            Case::Open => "masked, no guide".into(),
            Case::Bits => "bits mask".into(),
            Case::Few(k) => format!("few, k={k}"),
        }
    }

    fn allowed(self, vocab: u64) -> Option<Allowed> {
        match self {
            Case::Plain | Case::Open => None,
            Case::Bits => Some(Allowed::Bits(vec![
                0x5555_5555;
                vocab.div_ceil(32) as usize
            ])),
            Case::Few(k) => {
                let stride = vocab as usize / k;
                Some(Allowed::Few((0..k).map(|i| (i * stride) as u32).collect()))
            }
        }
    }
}

/// An engine for `c`, weights loaded.
fn opened(c: Config, w: &[Weight], recipe: &Recipe, settings: &Settings) -> Result<Engine, Error> {
    let g = decoder::stepper::<Norm, Swiglu>(&c);
    let mut engine = Engine::open(g, CONTEXT as usize, recipe, settings)?;
    engine.load(w)?;
    Ok(engine)
}

/// Seconds per step over one prompt and `STEPS` steps.
fn pass(engine: &mut Engine, slots: usize, allowed: &Option<Allowed>) -> Result<f64, Error> {
    let row = |slot: usize, token: u32, position: usize| Row {
        slot: slot as u32,
        token,
        position: position as u32,
    };
    let rows: Vec<Row> = (0..slots)
        .flat_map(|s| (0..PROMPT).map(move |p| row(s, p as u32 + 1, p)))
        .collect();
    let mut last = engine.last(&rows)?;
    let start = Instant::now();
    for i in 0..STEPS {
        let rows: Vec<Row> = (0..slots).map(|s| row(s, last[s], PROMPT + i)).collect();
        last = match allowed {
            None => engine.last(&rows)?,
            Some(a) => engine.last_guided(&rows, &vec![a.clone(); slots])?,
        };
    }
    Ok(start.elapsed().as_secs_f64() / STEPS as f64)
}

/// Times each case on both shapes, a line each.
pub fn report(settings: &Settings, favor: Option<String>) -> Result<(), Error> {
    let recipe = Recipe {
        pins: Pins::all(BF16_FORM),
        favor,
        ..Recipe::default()
    };
    for (name, shape) in SHAPES {
        let g = decoder::stepper::<Norm, Swiglu>(&shape(1, CONTEXT));
        // Small weights keep every value finite.
        let floats = weights_with(&g, &mut Noise(17), |_| (0.0, 0.02));
        let w: Vec<Weight> = floats.into_iter().map(Weight::Floats).collect();
        let vocab = shape(1, CONTEXT).stack.vocab;
        for picks in PICKS {
            let mut cases = vec![Case::Plain, Case::Open, Case::Bits];
            cases.extend(KS.iter().filter(|&&k| k <= FEW).map(|&k| Case::Few(k)));
            let mut c = shape(picks, CONTEXT);
            c.stack.steps = tessera_engine::Steps::Masked;
            let mut engines = [
                opened(shape(picks, CONTEXT), &w, &recipe, settings)?,
                opened(c, &w, &recipe, settings)?,
            ];
            let slots = picks as usize;
            let mut times = vec![Vec::new(); cases.len()];
            // Cases take turns, so load from elsewhere hits each alike.
            for round in 0..=PASSES {
                for (at, case) in cases.iter().enumerate() {
                    let engine = &mut engines[usize::from(!matches!(case, Case::Plain))];
                    let seconds = pass(engine, slots, &case.allowed(vocab))?;
                    if round > 0 {
                        times[at].push(seconds);
                    }
                }
            }
            println!(
                "{name}, {picks} picks, on {}, best and median:",
                engines[0].device()
            );
            for (case, mut t) in cases.into_iter().zip(times) {
                t.sort_by(f64::total_cmp);
                let (best, median) = (t[0] * 1e3, t[t.len() / 2] * 1e3);
                println!("  {:18} {best:8.3} {median:8.3} ms", case.label());
            }
        }
    }
    Ok(())
}
