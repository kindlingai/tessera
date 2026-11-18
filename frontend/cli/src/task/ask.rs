//! `task ask`: one prompt, one answer.

use super::Talk;
use crate::source::Stop;
use sensai_bundle::Message;

pub fn run(talk: Talk, words: &[String]) -> Result<(), Stop> {
    let prompt = super::text(words)?;
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(Stop::Usage);
    }
    let user = Message {
        role: "user".into(),
        content: prompt.into(),
    };
    let text = talk.render(&[user]).unwrap_or_else(|_| prompt.into());
    talk.say(&text)?;
    println!();
    Ok(())
}
