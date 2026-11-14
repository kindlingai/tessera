//! Tag patterns as regexes over `word/TAG ` tokens. See tags.txt.

use fancy_regex::Regex;

const NOUN: &str = "NN|NNS|NNP|NNPS";
const VERB: &str = "VB|VBD|VBG|VBN|VBP|VBZ";
const FINITE: &str = "VBD|VBP|VBZ|MD";
const BE: &str = "is|are|was|were|be|been|being|am|'s|'re";
const HAVE_DO: &str = "has|have|had|having|do|does|did|doing|done";
/// Operations as `by` agents, as in "read by the redo".
const ACTIONS: &str = "read|write|redo|step|call|pass|drop|flush|commit|insert|load|store|update|retry|reset|restart|reload|lookup|free|release|sweep|run|launch|swap|merge|refresh|resize|rewrite|replay";
const ARITH: &str = "multiplied|divided|scaled|shifted|rounded|padded|incremented|decremented|offset|bounded|capped";

fn tags(t: &str) -> String {
    format!(r"[^ ]+/(?:{t}) ")
}

fn words(w: &str) -> String {
    format!(r"(?:{w})/[^ ]+ ")
}

/// One token's regex.
fn class(name: &str) -> Result<String, String> {
    Ok(match name {
        "Noun" => tags(NOUN),
        "NounPlural" => tags("NNS|NNPS"),
        "Pronoun" => tags("PRP"),
        "Det" => tags(r"DT|PRP\$|WP\$"),
        "Adj" => tags("JJ|JJR|JJS"),
        "Adverb" => tags("RB|RBR|RBS"),
        "Num" => tags("CD"),
        "Prep" => tags("IN|TO"),
        "Possessive" => tags("POS"),
        "Hyphen" => tags("HYPH"),
        "Verb" => tags(VERB),
        "Verb-Finite" => tags(FINITE),
        "Verb-Past" => tags("VBD"),
        "Verb-PastParticiple" => tags("VBN"),
        "Verb-Ing" => tags("VBG"),
        "Verb-Base" => tags("VB"),
        "NotFinite" => format!(r"[^ ]+/(?!(?:{FINITE}) )[^ ]+ "),
        "NotVerb" => format!(r"[^ ]+/(?!(?:{VERB}|MD) )[^ ]+ "),
        "NotComma" => r"[^ ]+/(?!, )[^ ]+ ".into(),
        "Word" => r"[^ ]+/[^ ]+ ".into(),
        "Be" => words(BE),
        "ContentVerb" => format!(r"(?!(?:{BE}|{HAVE_DO})/){}", tags(VERB)),
        "Arith" => words(ARITH),
        "Action" => format!(r"(?:{ACTIONS}|[^ /]*(?:ation|ition|ction|sion)s?)/[^ ]+ "),
        // Code spans read as the noun `foo`.
        "Identifier" => r"(?:foo|[^ /]*_[^ /]*)/[^ ]+ ".into(),
        // Nothing after the verb starts its object.
        "ClauseEnd" => {
            let end = tags(r"\.|,|:|MD|VBD|VBP|VBZ");
            format!(
                r"(?:$|{end}|{}(?:$|{}))",
                tags("IN|RP"),
                tags(r"\.|,|:|IN|RP")
            )
        }
        other => return Err(format!("unknown class <{other}>")),
    })
}

fn one(b: &str) -> Result<String, String> {
    if let Some(inner) = b.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        let alts: Result<Vec<String>, String> = inner.split('|').map(class).collect();
        return Ok(format!("(?:{})", alts?.join("|")));
    }
    if let Some(inner) = b.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let alts: Vec<String> = inner
            .split('|')
            .map(|w| fancy_regex::escape(w).into_owned())
            .collect();
        return Ok(words(&alts.join("|")));
    }
    Ok(words(&fancy_regex::escape(b)))
}

fn element(el: &str) -> Result<String, String> {
    let word_end = el
        .find(['*', '+', '?'])
        .filter(|&i| i > 0 && !el.starts_with('('));
    let split = word_end.or_else(|| el.rfind(')').map(|i| i + 1).filter(|&i| i < el.len()));
    let (body, quant) = split.map_or((el, ""), |i| el.split_at(i));
    let piece = match body {
        "^" | "$" => body.to_string(),
        b if b.starts_with('!') => format!("(?!{})", one(&b[1..])?),
        b if b.starts_with('&') => format!("(?={})", one(&b[1..])?),
        b => one(b)?,
    };
    Ok(format!("(?:{piece}){quant}"))
}

/// Elements in order. `[ … ]` groups, with `|` between branches.
fn sequence(els: &[&str]) -> Result<String, String> {
    let mut out = String::new();
    let mut i = 0;
    while i < els.len() {
        if els[i] != "[" {
            out.push_str(&element(els[i])?);
            i += 1;
            continue;
        }
        let (mut depth, mut j) = (0, i);
        loop {
            match els.get(j) {
                Some(&"[") => depth += 1,
                Some(e) if e.starts_with(']') => depth -= 1,
                Some(_) => {}
                None => return Err("unclosed [".into()),
            }
            if depth == 0 {
                break;
            }
            j += 1;
        }
        let branches: Result<Vec<String>, String> =
            els[i + 1..j].split(|e| *e == "|").map(sequence).collect();
        out.push_str(&format!("(?:{}){}", branches?.join("|"), &els[j][1..]));
        i = j + 1;
    }
    Ok(out)
}

/// A pattern as a regex. Matches start at token starts.
pub fn compile(pattern: &str) -> Result<Regex, String> {
    let els: Vec<&str> = pattern.split_whitespace().collect();
    let re = format!("(?<![^ ]){}", sequence(&els)?);
    Regex::new(&re).map_err(|e| format!("{pattern}: {e}"))
}
