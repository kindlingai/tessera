//! Prompts to 16x16 sprites by masked diffusion.

use crate::Error;
use crate::api::{GenerateImage, Image, Sizes, Tokens};
use crate::worker::Worker;
use sensai_bundle::sprite::{denoiser, text_tower};
use sensai_bundle::tokenize::hf::Hf;
use sensai_engine::wasm::Module;
use sensai_engine::{Model, Recipe, Settings, Tensor};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GRID: usize = 16;
const PIXELS: usize = GRID * GRID;
const PALETTE: usize = 512;
const MASK: u32 = PALETTE as u32;
const TEXT: usize = 24;
const WIDTH: usize = 512;
/// Pixels this sure skip the neighbour check.
const CERTAIN: f32 = 0.9;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub steps: usize,
    pub guidance: f32,
    /// Gumbel noise on commit order, fading to zero.
    pub choice: f32,
    pub seed: u32,
    /// Argmax colors and no noise.
    pub greedy: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            steps: 16,
            guidance: 3.0,
            choice: 2.0,
            seed: 0,
            greedy: false,
        }
    }
}

/// mulberry32, seeded like the sprite model's web demo.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64 / 4_294_967_296.0) as f32
    }
}

pub struct Painter {
    clip: Model,
    tokenizer: Hf,
    denoiser: Model,
    /// RGB per palette id. Id 0 is transparent.
    pub palette: Vec<[u8; 3]>,
    /// The most canvases in one launch.
    most: usize,
}

/// Text states `[TEXT, WIDTH]` and the pooled `[WIDTH]` row.
pub struct Prompt {
    pub ids: Vec<u32>,
    pub states: Vec<f32>,
    pub pooled: Vec<f32>,
}

/// One canvas's state between steps.
struct Canvas {
    tokens: Vec<u32>,
    masked: Vec<bool>,
    /// Last step's guided probabilities, `[PIXELS, PALETTE]`.
    draft: Vec<f32>,
}

impl Painter {
    /// `dir/sprite.safetensors`, CLIP in `clip` or `dir/clip`.
    pub fn open(
        dir: &Path,
        clip: Option<&Path>,
        module: Option<&Module>,
        settings: &Settings,
        most: usize,
    ) -> Result<Self, Error> {
        let clip_dir = clip.map_or_else(|| dir.join("clip"), Path::to_path_buf);
        let c = text_tower(&clip_dir, 1, TEXT as u64)?;
        let mut clip = Model::open(c.graph, &Recipe::default(), settings)?;
        clip.load(&c.weights)?;
        let d = denoiser(dir, 2 * most as u64, module)?;
        let mut denoiser = Model::open(d.graph, &Recipe::default(), settings)?;
        denoiser.load(&d.weights)?;
        Ok(Self {
            clip,
            tokenizer: c.tokenizer,
            denoiser,
            palette: d.palette,
            most,
        })
    }

    pub fn device(&self) -> String {
        self.denoiser.device()
    }

    /// CLIP's view of `text`, padded with `<|endoftext|>`.
    pub fn encode(&mut self, text: &str) -> Result<Prompt, Error> {
        let mut ids = self.tokenizer.encode_special(text)?;
        let eos = *ids.last().ok_or("the tokenizer left out the end token")?;
        if ids.len() > TEXT {
            ids.truncate(TEXT - 1);
            ids.push(eos);
        }
        // CLIP pools the first end token.
        let end = ids.len() - 1;
        ids.resize(TEXT, eos);
        let positions = (0..TEXT as u32).collect();
        let inputs = [
            Tensor::u32(vec![1, TEXT], ids.clone()),
            Tensor::u32(vec![1, TEXT], positions),
        ];
        let states = self.clip.run(&inputs)?[0].floats()?.to_vec();
        let pooled = states[end * WIDTH..][..WIDTH].to_vec();
        Ok(Prompt {
            ids,
            states,
            pooled,
        })
    }

    /// Denoiser logits for each canvas, conditioned then not.
    fn predict(&mut self, p: &Prompt, canvases: &[Canvas]) -> Result<Vec<f32>, Error> {
        let n = canvases.len();
        let rows = 2 * n;
        let (mut tokens, mut counts, mut draft) = (vec![], vec![], vec![]);
        for _ in 0..2 {
            for c in canvases {
                tokens.extend_from_slice(&c.tokens);
                counts.push(c.masked.iter().filter(|&&m| m).count() as u32);
                for (row, &m) in c.draft.chunks(PALETTE).zip(&c.masked) {
                    match m {
                        true => draft.extend_from_slice(row),
                        false => draft.extend(std::iter::repeat_n(0.0, PALETTE)),
                    }
                }
            }
        }
        let repeat = |v: &[f32]| v.repeat(rows);
        let guide = (0..rows).flat_map(|r| if r < n { [1.0, 0.0] } else { [0.0, 1.0] });
        let inputs = [
            Tensor::u32(vec![rows, PIXELS], tokens),
            Tensor::u32(vec![rows], counts),
            Tensor::f32(vec![rows, TEXT, WIDTH], repeat(&p.states)),
            Tensor::f32(vec![rows, WIDTH], repeat(&p.pooled)),
            Tensor::f32(vec![rows, 2], guide.collect()),
            Tensor::f32(vec![rows, PIXELS, PALETTE], draft),
        ];
        Ok(self.denoiser.run(&inputs)?[0].floats()?.to_vec())
    }

    /// `n` sprites, row-major palette ids. Launches take seeds in turn.
    pub fn paint(&mut self, text: &str, n: usize, o: &Options) -> Result<Vec<Vec<u32>>, Error> {
        let p = self.encode(text)?;
        let mut out = Vec::with_capacity(n);
        let mut seed = o.seed;
        while out.len() < n {
            let count = (n - out.len()).min(self.most);
            out.extend(self.sample(&p, count, &Options { seed, ..*o })?);
            seed = seed.wrapping_add(1);
        }
        Ok(out)
    }

    /// The sampling loop for an encoded prompt.
    pub fn sample(&mut self, p: &Prompt, n: usize, o: &Options) -> Result<Vec<Vec<u32>>, Error> {
        let mut rng = Rng(o.seed);
        let blank = || Canvas {
            tokens: vec![MASK; PIXELS],
            masked: vec![true; PIXELS],
            draft: vec![0.0; PIXELS * PALETTE],
        };
        let mut canvases: Vec<Canvas> = (0..n).map(|_| blank()).collect();
        for k in 0..o.steps {
            let logits = self.predict(p, &canvases)?;
            let size = PIXELS * PALETTE;
            for (b, c) in canvases.iter_mut().enumerate() {
                let cond = &logits[b * size..][..size];
                let free = &logits[(b + n) * size..][..size];
                step(c, [cond, free], (k, o), &mut rng);
            }
        }
        Ok(canvases.into_iter().map(|c| c.tokens).collect())
    }

    /// One sprite's picture. Id 0 is transparent.
    pub fn image(&self, tokens: &[u32]) -> Image {
        let mut rgba = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            let [r, g, b] = self
                .palette
                .get(t as usize)
                .copied()
                .unwrap_or([255, 0, 255]);
            let a = if t == 0 { 0 } else { 255 };
            rgba.extend_from_slice(&[r, g, b, a]);
        }
        Image {
            width: GRID,
            height: GRID,
            rgba,
        }
    }
}

impl Worker<Painter> {
    /// Opens the painter in `dir` on a new thread.
    pub fn open(
        dir: PathBuf,
        clip: Option<PathBuf>,
        module: Option<Arc<Module>>,
        settings: Settings,
        most: usize,
    ) -> Result<Self, Error> {
        Worker::spawn(move || {
            let module = module.as_deref();
            let painter = Painter::open(&dir, clip.as_deref(), module, &settings, most)?;
            eprintln!("{}: painter loaded", painter.device());
            Ok(painter)
        })
    }
}

impl GenerateImage for Worker<Painter> {
    /// Sprites are painted 16 a side, then enlarged.
    fn sizes(&self) -> Sizes {
        Sizes {
            step: GRID,
            largest: 1024,
            default: 256,
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
        self.run(move |p| {
            let options = Options {
                seed,
                ..Options::default()
            };
            let sprites = p.paint(&prompt, n, &options)?;
            let scaled = |t: &Vec<u32>| p.image(t).scaled(size / GRID);
            Ok(sprites.iter().map(scaled).collect())
        })?
    }
}

impl Tokens for Worker<Painter> {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let text = text.to_string();
        self.run(move |p| p.tokenizer.encode_special(&text))?
    }

    fn decode(&self, ids: &[u32]) -> String {
        let ids = ids.to_vec();
        let decoded = self.run(move |p| p.tokenizer.decode(&ids));
        decoded.unwrap_or_default()
    }
}

/// Guided probabilities, then a pick and its probability per pixel.
fn choose(
    c: &mut Canvas,
    [cond, free]: [&[f32]; 2],
    o: &Options,
    rng: &mut Rng,
) -> Vec<(u32, f32)> {
    let mut picks = Vec::with_capacity(PIXELS);
    for i in 0..PIXELS {
        let row = &mut c.draft[i * PALETTE..][..PALETTE];
        let (lc, lu) = (
            &cond[i * PALETTE..][..PALETTE],
            &free[i * PALETTE..][..PALETTE],
        );
        let mut max = f32::NEG_INFINITY;
        for j in 0..PALETTE {
            row[j] = lu[j] + o.guidance * (lc[j] - lu[j]);
            max = max.max(row[j]);
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let mut pick = PALETTE - 1;
        match o.greedy {
            true => {
                let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1));
                pick = best.map_or(0, |(j, _)| j);
            }
            false => {
                let mut u = rng.next() * sum;
                for (j, &v) in row.iter().enumerate() {
                    u -= v;
                    if u <= 0.0 {
                        pick = j;
                        break;
                    }
                }
            }
        }
        row.iter_mut().for_each(|v| *v /= sum);
        picks.push((pick as u32, row[pick]));
    }
    picks
}

/// Whether masked pixel `i` may commit before its neighbours.
fn eligible(c: &Canvas, picks: &[(u32, f32)], score: &[f32], i: usize) -> bool {
    if picks[i].1 >= CERTAIN {
        return true;
    }
    let (r, col) = ((i / GRID) as isize, (i % GRID) as isize);
    let mut best = f32::NEG_INFINITY;
    for dr in -1..=1 {
        for dc in -1..=1 {
            let (rr, cc) = (r + dr, col + dc);
            if rr < 0 || cc < 0 || rr >= GRID as isize || cc >= GRID as isize {
                continue;
            }
            let j = (rr as usize) * GRID + cc as usize;
            if c.masked[j] && picks[j].1 < CERTAIN {
                best = best.max(score[j]);
            }
        }
    }
    score[i] >= best
}

/// One decoding step on one canvas.
fn step(c: &mut Canvas, logits: [&[f32]; 2], (k, o): (usize, &Options), rng: &mut Rng) {
    let frac = (k + 1) as f32 / o.steps as f32;
    let picks = choose(c, logits, o, rng);
    let masked = c.masked.iter().filter(|&&m| m).count();
    let left = match k + 1 == o.steps {
        true => 0,
        false => {
            (PIXELS as f64 * (std::f64::consts::FRAC_PI_2 * frac as f64).cos()).floor() as usize
        }
    };
    let commit = masked.saturating_sub(left).max(1).min(masked);
    let noise = if o.greedy {
        0.0
    } else {
        o.choice * (1.0 - frac)
    };
    let mut score = Vec::with_capacity(PIXELS);
    for &(_, p) in &picks {
        let u = rng.next().clamp(1e-10, 1.0 - 1e-10);
        let gumbel = -(-u.ln()).ln();
        score.push(p.max(1e-12).ln() + noise * gumbel);
    }
    let mut order: Vec<(f32, usize)> = Vec::new();
    for i in (0..PIXELS).filter(|&i| c.masked[i]) {
        let bonus = if eligible(c, &picks, &score, i) {
            1e4
        } else {
            0.0
        };
        order.push((score[i] + bonus, i));
    }
    order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, i) in order.iter().take(commit) {
        c.tokens[i] = picks[i].0;
        c.masked[i] = false;
    }
}
