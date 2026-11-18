//! Guides that hold answers to a format.

use crate::Error;
use crate::api::Format;
use sensai_bundle::tokenize::hf::Hf;
use sensai_engine::Guide;
use std::sync::Arc;

/// Makes one guide per answer, for one loaded model.
pub trait Guides: Send + Sync {
    /// A format the model cannot follow gives `Refused`.
    fn guide(&self, format: &Format) -> Result<Box<dyn Guide>, Error>;
}

#[cfg(feature = "guided")]
impl Guides for sensai_grammar::Vocab {
    fn guide(&self, format: &Format) -> Result<Box<dyn Guide>, Error> {
        let made = match format {
            Format::Object => self.object(),
            Format::Schema(schema) => self.schema(schema.clone()),
        };
        match made {
            Ok(matcher) => Ok(Box::new(matcher)),
            Err(e) => Err(Box::new(crate::api::Refused(e.to_string()))),
        }
    }
}

/// Guides for a model's tokenizer, stop ids and mask words.
#[cfg(feature = "guided")]
pub fn guides(tokenizer: &Hf, stop: &[u32], words: usize) -> Result<Arc<dyn Guides>, Error> {
    let json = tokenizer.json()?;
    let vocab = sensai_grammar::Vocab::new(json.as_bytes(), stop, words * 32)?;
    Ok(Arc::new(vocab))
}

/// This build has no guides.
#[cfg(not(feature = "guided"))]
pub fn guides(_: &Hf, _: &[u32], _: usize) -> Result<Arc<dyn Guides>, Error> {
    Err("this build has no guided decoding: enable the `guided` feature".into())
}
