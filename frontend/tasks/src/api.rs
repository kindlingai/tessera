//! What an HTTP front end asks of a loaded model.

use crate::Error;
use sensai_bundle::Message;

/// The served model's name, and the tokens one answer holds.
#[derive(Clone, Debug)]
pub struct Info {
    pub name: String,
    pub context: usize,
}

/// Text to ids and back, as generation sees them.
pub trait Tokens: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error>;
    fn decode(&self, ids: &[u32]) -> String;
}

/// Messages to one prompt, in the model's chat layout.
pub trait Chat: Send + Sync {
    fn render(&self, messages: &[Message], think: bool) -> Result<String, Error>;
}

/// What one answer may do beyond the prompt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    pub max_tokens: usize,
    /// Strings that end the answer, left out of it.
    pub stop: Vec<String>,
    /// The shape the answer must take. None allows any text.
    pub format: Option<Format>,
}

/// A shape an answer must take.
#[derive(Clone, Debug, PartialEq)]
pub enum Format {
    /// Any JSON object.
    Object,
    /// JSON that matches this schema.
    Schema(serde_json::Value),
}

/// A request the model cannot answer as asked.
#[derive(Debug)]
pub struct Refused(pub String);

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
}

/// Vectors for texts, and the tokens they took.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedded {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: usize,
}

/// Texts to unit vectors. Calls can block.
pub trait Embed: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error>;
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

/// Prompts to pictures. Calls can block.
pub trait GenerateImage: Send + Sync {
    fn sizes(&self) -> Sizes;
    /// `n` pictures, `size` a side. A seed repeats them.
    fn generate(&self, prompt: &str, n: usize, seed: u32, size: usize)
    -> Result<Vec<Image>, Error>;
}
