//! Calls an HTTP front end makes on a loaded model.

use crate::Error;
use crate::scopes::{Sequence, Thinking};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tessera_bundle::{Message, RenderOptions, ResponseParser};
pub use tessera_engine::Read;
use tessera_engine::{Penalties, Sampling};

/// The served model's name and context size.
#[derive(Clone, Debug)]
pub struct Info {
    pub name: String,
    pub context: usize,
}

/// Text to ids and back, for generation.
pub trait Tokens: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error>;
    fn decode(&self, ids: &[u32]) -> String;
}

/// Messages to one prompt, in the model's chat layout.
pub trait Chat: Send + Sync {
    fn render(&self, messages: &[Message], think: bool) -> Result<String, Error>;

    fn render_with(
        &self,
        messages: &[Message],
        think: bool,
        _reasoning_effort: Option<&str>,
    ) -> Result<String, Error> {
        self.render(messages, think)
    }

    fn render_options(
        &self,
        messages: &[Message],
        options: &RenderOptions<'_>,
    ) -> Result<String, Error> {
        self.render_with(
            messages,
            options.think.unwrap_or(true),
            options.reasoning_effort,
        )
    }

    fn response(&self) -> Option<&ResponseParser> {
        None
    }
}

/// Settings for one answer, beyond the prompt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    pub max_tokens: usize,
    pub min_tokens: usize,
    /// Strings that end the answer, excluded from its text.
    pub stop: Vec<String>,
    /// The answer's required shape. `None` allows any text.
    pub format: Option<Format>,
    /// Keep special tokens in decoded output for response parsing.
    pub raw: bool,
    /// Sampling parameters. Default is greedy.
    pub sampling: Sampling,
    /// Logit penalties for seen tokens. Default is none.
    pub penalties: Penalties,
    /// Changes inside a think block. Default is none.
    pub thinking: Thinking,
    /// Penalties for copies of earlier runs. Default is none.
    pub sequence: Sequence,
    /// Whether the prompt leaves a think block open.
    pub starts_thinking: bool,
    /// Tool-calling requirement for this answer.
    pub tool_choice: ToolChoice,
}

/// Tool-calling policy for one chat answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice {
    #[default]
    Auto,
    Required,
    Function(String),
}

/// A required shape for an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Format {
    /// Any JSON object.
    Object,
    /// JSON that matches this schema.
    Schema(serde_json::Value),
}

/// A request the model cannot answer: why, and the field.
#[derive(Debug)]
pub struct Refused(pub String, pub &'static str);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// Why an answer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}

/// One step of an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// A token, and the text it lets out, maybe none.
    Token { id: u32, text: String },
    /// Buffered text released from an earlier generated token.
    Text { token: usize, text: String },
    /// The last text, the reason, and the token count.
    Done {
        text: String,
        finish: Finish,
        tokens: usize,
        /// Prompt tokens that kept rows covered.
        cached: usize,
    },
}

/// Takes each step. Returns false when it wants no more.
pub type Sink = Box<dyn FnMut(Step) -> bool + Send>;

/// Answers a prompt, a step at a time, into `sink`.
pub trait Generator: Send + Sync {
    fn generate(&self, prompt: Vec<u32>, params: Params, sink: Sink) -> Result<(), Error>;
    fn supports_guides(&self) -> bool {
        true
    }
}

/// Logits after chosen prompt rows, with no decoding.
pub trait Score: Send + Sync {
    /// Each read's logits over its ids. Blocks.
    fn score(&self, prompt: Vec<u32>, reads: Vec<Read>) -> Result<Vec<Vec<f32>>, Error>;
}

/// Vectors for texts, and the tokens they took.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedded {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: usize,
}

/// One multimodal piece inside an embedding message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedPart {
    Text(String),
    Image(Vec<u8>),
}

/// One message inside a multimodal embedding request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbedMessage {
    pub role: String,
    pub parts: Vec<EmbedPart>,
}

/// One item to embed: text, ids, or messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedInput {
    Text(String),
    Tokens(Vec<u32>),
    Messages {
        messages: Vec<EmbedMessage>,
        add_generation_prompt: bool,
    },
}

/// Options for one embedding request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbedOptions {
    pub dimensions: Option<usize>,
    pub normalize: bool,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            dimensions: None,
            normalize: true,
        }
    }
}

/// Applies optional MRL truncation and FP32 L2 normalization.
pub fn finish_vec(
    mut v: Vec<f32>,
    dimensions: Option<usize>,
    normalize: bool,
) -> Result<Vec<f32>, Error> {
    if let Some(d) = dimensions {
        if d == 0 || d > v.len() {
            return Err(format!("dimensions {d} out of range 1..={}", v.len()).into());
        }
        v.truncate(d);
    }
    if normalize {
        let mut sum = 0.0f32;
        for &x in &v {
            sum += x * x;
        }
        let norm = sum.sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
    }
    Ok(v)
}

/// Texts or multimodal items to vectors. Calls can block.
pub trait Embed: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error>;

    fn embed_inputs(
        &self,
        inputs: &[EmbedInput],
        options: EmbedOptions,
    ) -> Result<Embedded, Error> {
        let mut texts = Vec::with_capacity(inputs.len());
        for item in inputs {
            match item {
                EmbedInput::Text(s) => texts.push(s.clone()),
                EmbedInput::Tokens(_) => {
                    return Err("this embedder does not accept token ids".into());
                }
                EmbedInput::Messages { .. } => {
                    return Err("this embedder does not accept multimodal messages".into());
                }
            }
        }
        let mut got = self.embed(&texts)?;
        if options.dimensions.is_some() {
            let mut vectors = Vec::with_capacity(got.vectors.len());
            for v in got.vectors {
                vectors.push(finish_vec(v, options.dimensions, options.normalize)?);
            }
            got.vectors = vectors;
        }
        Ok(got)
    }
}

/// One picture as row-major RGBA bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Image {
    /// Each pixel as a `k` by `k` block.
    pub fn scaled(&self, k: usize) -> Image {
        let width = self.width * k;
        let mut rgba = Vec::with_capacity(self.rgba.len() * k * k);
        for y in 0..self.height * k {
            for x in 0..width {
                let at = ((y / k) * self.width + x / k) * 4;
                rgba.extend_from_slice(&self.rgba[at..at + 4]);
            }
        }
        Image {
            width,
            height: self.height * k,
            rgba,
        }
    }
}

/// The square sizes a painter makes, in pixels a side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sizes {
    /// Sizes are multiples of this.
    pub step: usize,
    pub largest: usize,
    pub default: usize,
}

impl Sizes {
    /// `side` if a painter with these sizes makes it.
    pub fn square(&self, side: usize) -> Result<usize, String> {
        match side > 0 && side <= self.largest && side.is_multiple_of(self.step) {
            true => Ok(side),
            false => Err(format!(
                "`size` must be square, a multiple of {} up to {}",
                self.step, self.largest
            )),
        }
    }
}

/// A seed from the clock, when the request gives none.
pub fn fresh() -> u32 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |d| d.subsec_nanos())
}

/// Prompts to pictures. Calls can block.
pub trait GenerateImage: Send + Sync {
    fn sizes(&self) -> Sizes;
    /// `n` pictures, `size` a side, repeatable from `seed`.
    fn generate(&self, prompt: &str, n: usize, seed: u32, size: usize)
    -> Result<Vec<Image>, Error>;
}

/// A loaded model: each trait it serves.
pub struct Model {
    pub info: Info,
    pub tokens: Arc<dyn Tokens>,
    pub chat: Option<Arc<dyn Chat>>,
    pub generator: Option<Arc<dyn Generator>>,
    pub embed: Option<Arc<dyn Embed>>,
    pub image: Option<Arc<dyn GenerateImage>>,
    pub score: Option<Arc<dyn Score>>,
    /// Unknown request fields refuse the request. Else they warn.
    pub strict: bool,
    /// Counts requests that ask for a profile, when enabled.
    pub asks: Option<tessera_engine::Asks>,
}
