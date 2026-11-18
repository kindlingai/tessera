//! `task ask`: one prompt, one answer.

use super::Talk;
use crate::source::Stop;
use sensai_bundle::Message;
use std::io::Read;

pub fn run(talk: Talk, words: &[String]) -> Result<(), Stop> {
    let mut prompt = words.join(" ");
    if words.is_empty() {
        std::io::stdin()
            .read_to_string(&mut prompt)
            .map_err(|e| Stop::Failed(e.to_string()))?;
    }
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
