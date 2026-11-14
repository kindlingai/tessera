//! Tag patterns as regexes over `word/TAG ` tokens. See tags.txt.

use fancy_regex::Regex;

const VERB: &str = "VB|VBD|VBG|VBN|VBP|VBZ";
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
        "Noun" => tags("NN|NNS|NNP|NNPS"),
        "Det" => tags(r"DT|PRP\$|WP\$"),
        "Adj" => tags("JJ|JJR|JJS"),
        "Adverb" => tags("RB|RBR|RBS"),
        "Prep" => tags("IN|TO"),
        "Possessive" => tags("POS"),
        "Hyphen" => tags("HYPH"),
        "Verb" => tags(VERB),
        "Verb-Finite" => tags("VBD|VBP|VBZ|MD"),
        "Verb-PastParticiple" => tags("VBN"),
        "Word" => r"[^ ]+/[^ ]+ ".into(),
        "Be" => words(BE),
        "ContentVerb" => format!(r"(?!(?:{BE}|{HAVE_DO})/){}", tags(VERB)),
        "Arith" => words(ARITH),
        "Action" => format!(r"(?:{ACTIONS}|[^ /]*(?:ation|ition|ction|sion)s?)/[^ ]+ "),
        // Code spans read as the noun `foo`.
        "Identifier" => r"(?:foo|[^ /]*_[^ /]*)/[^ ]+ ".into(),
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
        "^" => body.to_string(),
        b if b.starts_with('!') => format!("(?!{})", one(&b[1..])?),
        b => one(b)?,
    };
    Ok(format!("(?:{piece}){quant}"))
}

/// A pattern as a regex. Matches start at token starts.
pub fn compile(pattern: &str) -> Result<Regex, String> {
    let body: Result<String, String> = pattern.split_whitespace().map(element).collect();
    let re = format!("(?<![^ ]){}", body?);
    Regex::new(&re).map_err(|e| format!("{pattern}: {e}"))
}
