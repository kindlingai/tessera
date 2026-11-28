//! Texts and multimodal inputs to unit-length vectors.

use crate::Error;
use crate::api::{Embed, EmbedInput, EmbedOptions, Embedded, Tokens, finish_vec};
use crate::vision::{self, EncodedImage, ImageSpan};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use tessera_bundle::Overrides;
use tessera_bundle::encoder::{self, Shape};
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::progress::{Progress, Unit};
use tessera_engine::{DType, Model, Recipe, Settings, Tensor};

/// Capacity bounds derived from the loaded model's plan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub seqs: usize,
    pub rows: usize,
    pub merged: usize,
}

/// Shared immutable preprocessing state safe across HTTP handler threads.
struct Prep {
    tokenizer: Hf,
    shape: Shape,
    has_vision: bool,
    pos_grid: usize,
    min_pixels: usize,
    max_pixels: usize,
    image_token: u32,
    max_merged: usize,
    hidden: usize,
}

impl Prep {
    fn limits(&self) -> Limits {
        Limits {
            seqs: self.shape.seqs,
            rows: self.shape.rows,
            merged: self.max_merged,
        }
    }

    /// Tokenizes `text` with special tokens, truncating to `max_tokens`.
    fn tokens(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = self.tokenizer.encode_special(text)?;
        let most = self.shape.max_tokens;
        if ids.len() > most {
            let eos = ids[ids.len() - 1];
            ids.truncate(most - 1);
            ids.push(eos);
        }
        Ok(ids)
    }

    /// Preprocesses texts and images off the GPU worker thread.
    fn prepare(&self, inputs: &[EmbedInput], options: EmbedOptions) -> Result<Prepared, Error> {
        if let Some(dim) = options.dimensions {
            if dim == 0 {
                return Err("dimensions must be positive".into());
            }
            if self.hidden > 0 && dim > self.hidden {
                return Err(format!("dimensions {dim} exceeds hidden size {}", self.hidden).into());
            }
        }
        let mut ids = Vec::with_capacity(inputs.len());
        let mut spans = Vec::with_capacity(inputs.len());
        let mut imgs = Vec::with_capacity(inputs.len());
        let mut tokens = 0usize;
        let mut merged = 0usize;
        for (at, item) in inputs.iter().enumerate() {
            let (seq, sp, im) = match item {
                EmbedInput::Text(t) => (self.tokens(t)?, Vec::new(), Vec::new()),
                EmbedInput::Tokens(toks) => (toks.clone(), Vec::new(), Vec::new()),
                EmbedInput::Messages {
                    messages,
                    add_generation_prompt,
                } => {
                    let (prompt, raw_imgs) = vision::format_chat(messages, *add_generation_prompt);
                    if !raw_imgs.is_empty() && !self.has_vision {
                        return Err("this model has no vision tower".into());
                    }
                    let mut encoded = Vec::with_capacity(raw_imgs.len());
                    for bytes in &raw_imgs {
                        let rgb = vision::decode_image(bytes)?;
                        encoded.push(vision::encode_image(
                            &rgb,
                            self.min_pixels,
                            self.max_pixels,
                            self.pos_grid,
                        )?);
                    }
                    let base = self.tokens(&prompt)?;
                    let (expanded, seq_spans) =
                        vision::expand_images(&base, self.image_token, &encoded)?;
                    (expanded, seq_spans, encoded)
                }
            };
            if seq.is_empty() || seq.len() > self.shape.max_tokens {
                return Err(format!("text {at} has {} tokens", seq.len()).into());
            }
            let seq_merged: usize = im.iter().map(EncodedImage::tokens).sum();
            if self.has_vision && seq_merged > self.max_merged {
                return Err(format!("item {at} has {seq_merged} visual tokens").into());
            }
            tokens += seq.len();
            merged += seq_merged;
            ids.push(seq);
            spans.push(sp);
            imgs.push(im);
        }
        Ok(Prepared {
            ids,
            spans,
            imgs,
            tokens,
            merged,
        })
    }
}

/// Preprocessed embedding request ready for GPU execution.
struct Prepared {
    ids: Vec<Vec<u32>>,
    spans: Vec<Vec<ImageSpan>>,
    imgs: Vec<Vec<EncodedImage>>,
    tokens: usize,
    merged: usize,
}

/// Vision encoder state for multimodal packed causal models.
struct VisionRunner {
    model: Model,
    deepstack: usize,
    max_merged: usize,
    hidden: usize,
}

pub struct Embedder {
    model: Model,
    vision: Option<VisionRunner>,
    prep: Arc<Prep>,
    /// Texts of unlike length share launches, padded.
    padded: bool,
}

/// Texts that share a launch: shortest first, or equal unpadded.
fn groups(ids: &[Vec<u32>], padded: bool, shape: Shape) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| ids[i].len());
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let len = ids[i].len();
        let joins = out.last().is_some_and(|g| {
            let fits = match padded {
                true => (g.len() + 1) * len <= shape.rows,
                false => ids[g[0]].len() == len,
            };
            g.len() < shape.seqs && fits
        });
        match out.last_mut() {
            Some(g) if joins => g.push(i),
            _ => out.push(vec![i]),
        }
    }
    out
}

/// Each row scaled to unit length.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

impl Embedder {
    /// Loads the model in `dir` where `settings` say.
    pub fn open(
        dir: &Path,
        settings: &Settings,
        config: &Overrides,
        progress: &Progress,
    ) -> Result<Self, Error> {
        // GPUs lack a fast kernel for listed keys.
        let padded = settings.gpu().is_none();
        let reading = progress.step("read model", Unit::Items, None);
        let (shape, e) = encoder::open(dir, padded, config)?;
        reading.finish();
        let recipe = Recipe {
            pins: e.pins,
            ..Recipe::default()
        };
        let model = Model::open_loaded(e.graph, &recipe, settings, &e.weights, progress)?;
        let (prep, vision) = match e.vision {
            Some(v) => {
                let v_recipe = Recipe {
                    pins: v.pins,
                    ..Recipe::default()
                };
                let v_model =
                    Model::open_loaded(v.graph, &v_recipe, settings, &v.weights, progress)?;
                let prep = Prep {
                    tokenizer: e.tokenizer,
                    shape,
                    has_vision: true,
                    pos_grid: v.pos_grid,
                    min_pixels: v.min_pixels,
                    max_pixels: v.max_pixels,
                    image_token: v.image_token,
                    max_merged: v.max_merged,
                    hidden: e.hidden,
                };
                let runner = VisionRunner {
                    model: v_model,
                    deepstack: v.deepstack,
                    max_merged: v.max_merged,
                    hidden: e.hidden,
                };
                (Arc::new(prep), Some(runner))
            }
            None => (
                Arc::new(Prep {
                    tokenizer: e.tokenizer,
                    shape,
                    has_vision: false,
                    pos_grid: 48,
                    min_pixels: vision::MIN_PIXELS,
                    max_pixels: vision::MAX_PIXELS,
                    image_token: vision::IMAGE_TOKEN,
                    max_merged: 0,
                    hidden: e.hidden,
                }),
                None,
            ),
        };
        Ok(Self {
            model,
            vision,
            prep,
            padded,
        })
    }

    /// The backend and hardware the model runs on.
    pub fn device(&self) -> String {
        self.model.device()
    }

    /// Ids of `text` between `<bos>` and `<eos>`, cut to fit.
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.prep.tokens(text)
    }

    /// Embeds text, token ids, or multimodal items with `options`.
    pub fn embed_inputs(
        &mut self,
        inputs: &[EmbedInput],
        options: EmbedOptions,
    ) -> Result<Embedded, Error> {
        let prep = self.prep.prepare(inputs, options)?;
        let mut done = self.run_coalesced(&[(&prep, options)])?;
        Ok(done.pop().unwrap())
    }

    /// Executes one or more preprocessed requests in a coalesced batch.
    fn run_coalesced(
        &mut self,
        batch: &[(&Prepared, EmbedOptions)],
    ) -> Result<Vec<Embedded>, Error> {
        let total_seqs: usize = batch.iter().map(|(p, _)| p.ids.len()).sum();
        if self.vision.is_some() {
            let mut seqs = Vec::with_capacity(total_seqs);
            let mut spans = Vec::with_capacity(total_seqs);
            let mut imgs = Vec::with_capacity(total_seqs);
            let mut opts = Vec::with_capacity(total_seqs);
            for &(p, opt) in batch {
                for i in 0..p.ids.len() {
                    seqs.push(p.ids[i].as_slice());
                    spans.push(p.spans[i].as_slice());
                    imgs.push(p.imgs[i].as_slice());
                    opts.push(opt);
                }
            }
            let flat = self.run_packed(&seqs, &spans, &imgs, &opts)?;
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(batch.len());
            for &(p, _) in batch {
                let n = p.ids.len();
                out.push(Embedded {
                    vectors: flat[cursor..cursor + n].to_vec(),
                    tokens: p.tokens,
                });
                cursor += n;
            }
            return Ok(out);
        }
        let mut all_ids = Vec::with_capacity(total_seqs);
        for &(p, _) in batch {
            all_ids.extend_from_slice(&p.ids);
        }
        let raw = self.embed_ids(&all_ids)?;
        let mut iter = raw.into_iter();
        let mut out = Vec::with_capacity(batch.len());
        for &(p, opt) in batch {
            let mut vectors = Vec::with_capacity(p.ids.len());
            for _ in 0..p.ids.len() {
                vectors.push(finish_vec(
                    iter.next().unwrap(),
                    opt.dimensions,
                    opt.normalize,
                )?);
            }
            out.push(Embedded {
                vectors,
                tokens: p.tokens,
            });
        }
        Ok(out)
    }

    /// Runs packed causal text and optional vision launches.
    fn run_packed(
        &mut self,
        ids: &[&[u32]],
        spans: &[&[ImageSpan]],
        imgs: &[&[EncodedImage]],
        opts: &[EmbedOptions],
    ) -> Result<Vec<Vec<f32>>, Error> {
        let Some(vis) = self.vision.as_ref() else {
            return Err("packed runner requires a multimodal embedder".into());
        };
        let (max_merged, deepstack, hidden) = (vis.max_merged, vis.deepstack, vis.hidden);
        let shape = self.prep.shape;
        let mut out = Vec::with_capacity(ids.len());
        let mut start = 0usize;
        while start < ids.len() {
            let mut end = start;
            let mut rows = 0usize;
            let mut merged = 0usize;
            while end < ids.len() {
                let r = ids[end].len();
                let m: usize = imgs[end].iter().map(EncodedImage::tokens).sum();
                if end > start
                    && (end - start >= shape.seqs
                        || rows + r > shape.rows
                        || merged + m > max_merged)
                {
                    break;
                }
                rows += r;
                merged += m;
                end += 1;
            }
            let mut chunk_imgs: Vec<&EncodedImage> = Vec::new();
            for &seq_imgs in &imgs[start..end] {
                for im in seq_imgs {
                    chunk_imgs.push(im);
                }
            }
            let vis_outs: Vec<Tensor> = if chunk_imgs.is_empty() {
                (0..=deepstack)
                    .map(|_| Tensor::zeros(vec![1, hidden], DType::F32))
                    .collect()
            } else {
                let vis_in = vision::pack_vision(&chunk_imgs);
                let raw = self.vision.as_mut().unwrap().model.run(&vis_in)?;
                let mut reshaped = Vec::with_capacity(raw.len());
                for t in raw {
                    reshaped.push(t.reshaped(vec![merged, hidden]));
                }
                reshaped
            };
            let mut text_in = vision::pack_text(&ids[start..end], &spans[start..end]);
            text_in.extend(vis_outs);
            let got = self.model.run(&text_in)?;
            let floats = got[0].floats()?;
            for (row, &opt) in floats.chunks(hidden).zip(&opts[start..end]) {
                out.push(finish_vec(row.to_vec(), opt.dimensions, opt.normalize)?);
            }
            start = end;
        }
        Ok(out)
    }

    /// The same for ids, a few texts per launch.
    pub fn embed_ids(&mut self, ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Error> {
        let shape = self.prep.shape;
        for (at, t) in ids.iter().enumerate() {
            if t.is_empty() || t.len() > shape.max_tokens {
                return Err(format!("text {at} has {} tokens", t.len()).into());
            }
        }
        if self.vision.is_some() {
            let seqs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
            let spans = vec![&[][..]; ids.len()];
            let imgs = vec![&[][..]; ids.len()];
            let opts = vec![EmbedOptions::default(); ids.len()];
            return self.run_packed(&seqs, &spans, &imgs, &opts);
        }
        let mut out = vec![Vec::new(); ids.len()];
        for group in groups(ids, self.padded, shape) {
            let got = self.launch(ids, &group)?;
            for (i, v) in group.into_iter().zip(got) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    /// One launch of `group`, padded to its longest text.
    fn launch(&mut self, ids: &[Vec<u32>], group: &[usize]) -> Result<Vec<Vec<f32>>, Error> {
        let (n, len) = (
            group.len(),
            group.iter().map(|&i| ids[i].len()).max().unwrap_or(1),
        );
        let mut tokens = vec![0; n * len];
        for (row, &i) in tokens.chunks_mut(len).zip(group) {
            row[..ids[i].len()].copy_from_slice(&ids[i]);
        }
        let positions = (0..n).flat_map(|_| 0..len as u32).collect();
        let shape = vec![n, len];
        let mut inputs = vec![
            Tensor::u32(shape.clone(), tokens),
            Tensor::u32(shape, positions),
        ];
        if self.padded {
            // Each row lists its own text's keys. `len` lists none.
            let mut lists = Vec::with_capacity(n * len * len);
            for &i in group {
                let own = ids[i].len() as u32;
                let list = (0..len as u32).map(|j| if j < own { j } else { len as u32 });
                let list: Vec<u32> = list.collect();
                (0..len).for_each(|_| lists.extend_from_slice(&list));
            }
            inputs.push(Tensor::u32(vec![n, len, len], lists));
        }
        let got = self.model.run(&inputs)?;
        let floats = got[0].floats()?;
        let per = floats.len() / n;
        let mut out = Vec::with_capacity(n);
        for (&i, block) in group.iter().zip(floats.chunks(per)) {
            if !self.padded {
                out.push(unit(block.to_vec()));
                continue;
            }
            // The text's own rows. The unit scale divides.
            let width = per / len;
            let mut sum = vec![0.0f32; width];
            for row in block.chunks(width).take(ids[i].len()) {
                sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
            }
            out.push(unit(sum));
        }
        Ok(out)
    }
}

/// One queued request waiting for coalesced worker execution.
struct Job<P, R> {
    seqs: usize,
    rows: usize,
    merged: usize,
    payload: P,
    scope: Option<tessera_engine::check::Scope>,
    reply: mpsc::SyncSender<Result<R, Error>>,
}

struct QueueState<P, R> {
    pending: VecDeque<Job<P, R>>,
    closed: bool,
}

/// Drains one coalesced wave of queued jobs within `limits`.
fn drain_batch<P, R>(pending: &mut VecDeque<Job<P, R>>, limits: Limits) -> Vec<Job<P, R>> {
    let Some(first) = pending.pop_front() else {
        return Vec::new();
    };
    let mut seqs = first.seqs;
    let mut rows = first.rows;
    let mut merged = first.merged;
    let scoped = first.scope.is_some();
    let mut batch = vec![first];
    while let Some(next) = pending.front() {
        if scoped || next.scope.is_some() {
            break;
        }
        if seqs + next.seqs > limits.seqs
            || rows + next.rows > limits.rows
            || merged + next.merged > limits.merged
        {
            break;
        }
        let job = pending.pop_front().unwrap();
        seqs += job.seqs;
        rows += job.rows;
        merged += job.merged;
        batch.push(job);
    }
    batch
}

/// Worker thread that dynamically coalesces concurrent requests into batches.
pub(crate) struct BatchWorker<P: Send + 'static, R: Send + 'static> {
    state: Arc<(Mutex<QueueState<P, R>>, Condvar)>,
}

impl<P: Send + 'static, R: Send + 'static> BatchWorker<P, R> {
    /// Spawns a coalescing worker initialized by `open`.
    pub(crate) fn spawn<T: 'static, M: Send + 'static>(
        open: impl FnOnce() -> Result<(T, M, Limits), Error> + Send + 'static,
        mut run: impl FnMut(&mut T, &[P]) -> Result<Vec<R>, Error> + Send + 'static,
    ) -> Result<(Self, M), Error> {
        let state = Arc::new((
            Mutex::new(QueueState::<P, R> {
                pending: VecDeque::new(),
                closed: false,
            }),
            Condvar::new(),
        ));
        let worker_state = state.clone();
        let (ready, opened) = mpsc::channel();
        let check = tessera_engine::check::current_scope();
        let sizer = tessera_engine::sizer::current_scope();
        std::thread::spawn(move || {
            let open = || match sizer {
                Some(scope) => scope.run(open),
                None => open(),
            };
            let result = match check {
                Some(scope) => scope.run(open),
                None => open(),
            };
            let (mut target, meta, limits) = match result {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok(meta));
            let (lock, cvar) = &*worker_state;
            loop {
                let batch = {
                    let mut guard = match lock.lock() {
                        Ok(g) => g,
                        Err(_) => return,
                    };
                    while guard.pending.is_empty() && !guard.closed {
                        guard = match cvar.wait(guard) {
                            Ok(g) => g,
                            Err(_) => return,
                        };
                    }
                    if guard.pending.is_empty() {
                        return;
                    }
                    drain_batch(&mut guard.pending, limits)
                };
                let scope = batch[0].scope.clone();
                let (mut payloads, mut replies) = (
                    Vec::with_capacity(batch.len()),
                    Vec::with_capacity(batch.len()),
                );
                for job in batch {
                    payloads.push(job.payload);
                    replies.push(job.reply);
                }
                let res = match scope {
                    Some(scope) => scope.run(|| run(&mut target, &payloads)),
                    None => run(&mut target, &payloads),
                };
                match res {
                    Ok(results) => {
                        for (reply, item) in replies.into_iter().zip(results) {
                            let _ = reply.send(Ok(item));
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        for reply in replies {
                            let _ = reply.send(Err(msg.clone().into()));
                        }
                    }
                }
            }
        });
        let meta = opened.recv().map_err(|_| "the model thread stopped")??;
        Ok((Self { state }, meta))
    }

    /// Enqueues a preprocessed job and waits for its coalesced result.
    pub(crate) fn submit(
        &self,
        seqs: usize,
        rows: usize,
        merged: usize,
        payload: P,
    ) -> Result<R, Error> {
        let (reply, answered) = mpsc::sync_channel(1);
        let job = Job {
            seqs,
            rows,
            merged,
            payload,
            scope: tessera_engine::check::current_scope(),
            reply,
        };
        let (lock, cvar) = &*self.state;
        {
            let mut guard = lock.lock().map_err(|_| "the model stopped")?;
            if guard.closed {
                return Err("the model stopped".into());
            }
            guard.pending.push_back(job);
            cvar.notify_one();
        }
        answered.recv().map_err(|_| "the model stopped")?
    }

    #[cfg(test)]
    pub(crate) fn pending(&self) -> usize {
        self.state.0.lock().map(|g| g.pending.len()).unwrap_or(0)
    }
}

impl<P: Send + 'static, R: Send + 'static> Drop for BatchWorker<P, R> {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.state;
        if let Ok(mut guard) = lock.lock() {
            guard.closed = true;
            cvar.notify_all();
        }
    }
}

/// Coalescing embedder worker with off-worker preprocessing.
pub struct Threaded {
    prep: Arc<Prep>,
    worker: BatchWorker<(Prepared, EmbedOptions), Embedded>,
}

impl Threaded {
    /// Opens the model in `dir` on a new coalescing thread.
    pub fn open(
        dir: PathBuf,
        settings: Settings,
        config: Overrides,
        progress: Progress,
    ) -> Result<Self, Error> {
        let (worker, prep) = BatchWorker::spawn(
            move || {
                let embedder = Embedder::open(&dir, &settings, &config, &progress)?;
                eprintln!("{}: embedder loaded", embedder.device());
                let prep = embedder.prep.clone();
                let limits = prep.limits();
                Ok((embedder, prep, limits))
            },
            |embedder, batch| {
                let refs: Vec<(&Prepared, EmbedOptions)> =
                    batch.iter().map(|(p, opt)| (p, *opt)).collect();
                embedder.run_coalesced(&refs)
            },
        )?;
        Ok(Self { prep, worker })
    }
}

impl Embed for Threaded {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error> {
        let mut inputs = Vec::with_capacity(texts.len());
        for t in texts {
            inputs.push(EmbedInput::Text(t.clone()));
        }
        self.embed_inputs(&inputs, EmbedOptions::default())
    }

    fn embed_inputs(
        &self,
        inputs: &[EmbedInput],
        options: EmbedOptions,
    ) -> Result<Embedded, Error> {
        let prep = self.prep.prepare(inputs, options)?;
        let (seqs, rows, merged) = (prep.ids.len(), prep.tokens, prep.merged);
        self.worker.submit(seqs, rows, merged, (prep, options))
    }
}

impl Tokens for Threaded {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.prep.tokens(text)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.prep.tokenizer.decode(ids)
    }
}
