//! `task chat`: turns on stdin, answers on stdout.

use super::Talk;
use crate::source::Stop;
use std::io::{BufRead, IsTerminal, Write};
use tessera_bundle::Message;

fn message(role: &str, content: &str) -> Message {
    Message::text(role, content)
}

pub fn run(talk: Talk, extra: &[String]) -> Result<(), Stop> {
    if !extra.is_empty() {
        return Err(Stop::Usage);
    }
    let tty = std::io::stdin().is_terminal();
    let mut history = Vec::new();
    let mut lines = std::io::stdin().lock().lines();
    loop {
        if tty {
            eprint!("> ");
            let _ = std::io::stderr().flush();
        }
        let Some(Ok(line)) = lines.next() else { break };
        if line.trim().is_empty() {
            break;
        }
        history.push(message("user", line.trim()));
        let answer = talk.render(&history).and_then(|text| talk.say(&text));
        println!();
        match answer {
            Ok(text) => history.push(message("assistant", text.trim())),
            Err(Stop::Failed(e)) => {
                history.pop();
                eprintln!("tesserax: {e}");
            }
            Err(other) => return Err(other),
        }
    }
    Ok(())
}
