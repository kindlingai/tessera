//! FLUX.2 klein: a prompt to a picture.

use crate::Error;
use crate::api::{GenerateImage, Image, Sizes, Tokens};
use crate::features::Features;
use crate::flow::Flow;
use crate::pictures::Pictures;
use crate::worker::Worker;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;
use tessera_bundle::Overrides;
use tessera_bundle::tokenize::hf::Hf;
use tessera_bundle::vae;
use tessera_engine::Settings;
use tessera_engine::progress::{Progress, Unit};

/// Channels per latent token.
const LATENT: usize = tessera_engine::flux2::LATENT as usize;

/// Diffusers' fitted shift for `tokens` rows over `steps`.
fn mu(tokens: usize, steps: usize) -> f64 {
    let (a1, b1, a2, b2) = (8.738_095_24e-5, 1.898_333_33, 0.000_169_27, 0.456_666_66);
    let n = tokens as f64;
    if n > 4300.0 {
        return a2 * n + b2;
    }
    let (m200, m10) = (a2 * n + b2, a1 * n + b1);
    let a = (m200 - m10) / 190.0;
    a * steps as f64 + (m200 - 200.0 * a)
}

/// Times from 1 to 0, shifted toward noise.
pub(crate) fn sigmas(tokens: usize, steps: usize) -> Vec<f32> {
    let shift = mu(tokens, steps).exp();
    let mut out: Vec<f32> = (0..steps)
        .map(|i| 1.0 - i as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1).max(1) as f64)
        .map(|s| (shift / (shift + (1.0 / s - 1.0))) as f32)
        .collect();
    out.push(0.0);
    out
}

/// Standard normal values from a seed, by Box-Muller.
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let one = |u: &mut dyn FnMut() -> f64| {
        let (r, a) = ((-2.0 * u().ln()).sqrt(), std::f64::consts::TAU * u());
        (r * a.cos()) as f32
    };
    (0..n).map(|_| one(&mut uniform)).collect()
}

/// The recipe's `[config.pipeline]` table.
#[derive(Clone)]
struct Pipeline {
    /// The text model's layers the transformer reads.
    layers: Vec<u64>,
    /// Text rows each prompt pads to.
    rows: usize,
    steps: usize,
}

impl Pipeline {
    fn read(config: &Overrides) -> Result<Self, Error> {
        let read = config.read("pipeline", None, |k| Pipeline {
            layers: k.need(&["text_layers"]),
            rows: k.need(&["text_rows"]),
            steps: k.need(&["steps"]),
        });
        read.map_err(|e| format!("klein needs a recipe with [config.pipeline]:\n{e}").into())
    }
}

/// Device memory ceiling below which klein runs stages sequentially.
const OFFLOAD_LIMIT: u64 = 32 * 1024 * 1024 * 1024;

/// The models, loaded once or per stage on small devices.
pub struct Klein {
    /// The diffusers repository, in Hugging Face's layout.
    dir: PathBuf,
    settings: Settings,
    config: Overrides,
    pipeline: Pipeline,
    tokenizer: Hf,
    features: Option<Features>,
    flow: Option<Flow>,
    pictures: HashMap<usize, Pictures>,
    /// Drop earlier stages between encode, flow, and decode.
    offload: bool,
    /// Latent tokens a side, at most.
    largest: usize,
    /// Pixels a side per latent token.
    patch: usize,
    pub steps: usize,
}

impl Klein {
    fn open_features(&self, progress: &Progress) -> Result<Features, Error> {
        let words = [self.dir.join("text_encoder"), self.dir.join("tokenizer")];
        let encoding = progress.step("text encoder", Unit::Items, None);
        let features = Features::open(
            words.each_ref().map(|d| d.as_path()),
            &self.settings,
            &self.pipeline.layers,
            [1, self.pipeline.rows],
            &self.config.under("text_encoder"),
            &encoding.progress(),
        )?;
        encoding.finish();
        Ok(features)
    }

    fn open_flow(&self, progress: &Progress) -> Result<Flow, Error> {
        let sizes = [1, self.pipeline.rows, self.largest * self.largest];
        let flowing = progress.step("transformer", Unit::Items, None);
        let flow = Flow::open(
            &self.dir,
            &self.settings,
            sizes,
            &self.config,
            &flowing.progress(),
        )?;
        flowing.finish();
        Ok(flow)
    }

    /// The repository in `dir`, up to `largest` pixels a side.
    pub fn open(
        dir: PathBuf,
        settings: Settings,
        largest: usize,
        config: Overrides,
        progress: &Progress,
    ) -> Result<Self, Error> {
        let patch = vae::config(&dir, [1, 1], &config)?.scale() as usize;
        let largest = largest / patch;
        let pipeline = Pipeline::read(&config)?;
        let tokenizer = Hf::open(&dir.join("tokenizer/tokenizer.json"))?;
        let counting = tessera_engine::sizer::counting();
        let platform = tessera_engine::sizer::platform(&settings);
        let offload = platform.integrated && platform.memory <= OFFLOAD_LIMIT;
        if offload && !settings.weight_streaming {
            let memory = platform.memory;
            let why = format!(
                "klein keeps one stage loaded on a device with {memory} bytes. \
                 --allow-weight-streaming loads each stage per picture"
            );
            return Err(why.into());
        }
        let steps = pipeline.steps;
        let mut klein = Self {
            dir,
            settings,
            config,
            pipeline,
            tokenizer,
            features: None,
            flow: None,
            pictures: HashMap::new(),
            offload,
            largest,
            patch,
            steps,
        };
        if offload {
            if counting {
                drop(klein.open_features(progress)?);
                drop(klein.open_flow(progress)?);
                drop(Pictures::open(
                    &klein.dir,
                    &klein.settings,
                    [largest, 1],
                    &klein.config,
                )?);
            }
        } else {
            klein.features = Some(klein.open_features(progress)?);
            klein.flow = Some(klein.open_flow(progress)?);
            if counting {
                let _ = Pictures::open(&klein.dir, &klein.settings, [largest, 1], &klein.config)?;
            }
        }
        Ok(klein)
    }

    /// The text model's tokenizer.
    pub fn tokenizer(&self) -> &Hf {
        &self.tokenizer
    }

    /// A picture `patch * side` pixels a side, from `noise` latents.
    pub fn paint(&mut self, prompt: &str, side: usize, noise: &[f32]) -> Result<Image, Error> {
        let rows = side * side;
        if side > self.largest || noise.len() != rows * LATENT {
            return Err(format!("{} noise values for {rows} tokens", noise.len()).into());
        }
        let quiet = Progress::default();
        if self.offload {
            self.flow = None;
            self.pictures.clear();
        }
        let text = match &mut self.features {
            Some(f) => f.encode(&[prompt])?.remove(0),
            None => {
                let mut f = self.open_features(&quiet)?;
                let text = f.encode(&[prompt])?.remove(0);
                if !self.offload {
                    self.features = Some(f);
                }
                text
            }
        };
        if self.offload {
            self.features = None;
        }
        let mut x = noise.to_vec();
        let flow = match &mut self.flow {
            Some(f) => f,
            None => self.flow.insert(self.open_flow(&quiet)?),
        };
        for pair in sigmas(rows, self.steps).windows(2) {
            let v = flow.velocity(&x, &text, side, pair[0])?;
            x.iter_mut()
                .zip(&v)
                .for_each(|(x, v)| *x += (pair[1] - pair[0]) * v);
        }
        if self.offload {
            self.flow = None;
        }
        let pictures = match self.pictures.entry(side) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(Pictures::open(
                &self.dir,
                &self.settings,
                [side, 1],
                &self.config,
            )?),
        };
        let image = pictures.decode(&x)?.remove(0);
        if self.offload {
            self.pictures.clear();
        }
        Ok(image)
    }
}

impl Worker<Klein> {
    /// Opens klein on a new thread.
    pub fn open(
        dir: PathBuf,
        settings: Settings,
        largest: usize,
        config: Overrides,
        progress: Progress,
    ) -> Result<Self, Error> {
        Worker::spawn(move || {
            let klein = Klein::open(dir, settings, largest, config, &progress)?;
            eprintln!("klein loaded");
            Ok(klein)
        })
    }
}

impl GenerateImage for Worker<Klein> {
    fn sizes(&self) -> Sizes {
        let (largest, patch) = self
            .run(|k| (k.largest * k.patch, k.patch))
            .unwrap_or((1, 1));
        Sizes {
            step: patch,
            largest,
            default: largest.min(512),
        }
    }

    fn generate(
        &self,
        prompt: &str,
        n: usize,
        seed: u32,
        size: usize,
    ) -> Result<Vec<Image>, Error> {
        let prompt = prompt.to_string();
        self.run(move |k| {
            let side = size / k.patch;
            let one = |i: usize| noise(u64::from(seed) + i as u64, side * side * LATENT);
            (0..n).map(|i| k.paint(&prompt, side, &one(i))).collect()
        })?
    }
}

impl Tokens for Worker<Klein> {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let text = text.to_string();
        self.run(move |k| k.tokenizer().encode_special(&text))?
    }

    fn decode(&self, ids: &[u32]) -> String {
        let ids = ids.to_vec();
        self.run(move |k| k.tokenizer().decode(&ids))
            .unwrap_or_default()
    }
}
