//! A small HTML element builder that escapes all text.

pub enum Part {
    El(El),
    Text(String),
    /// Trusted text, such as a static script or style.
    Raw(String),
}

pub struct El {
    tag: &'static str,
    attrs: Vec<(&'static str, String)>,
    kids: Vec<Part>,
}

pub fn el(tag: &'static str) -> El {
    El {
        tag,
        attrs: Vec::new(),
        kids: Vec::new(),
    }
}

/// An element holding only text.
pub fn text(tag: &'static str, body: impl Into<String>) -> El {
    el(tag).text(body)
}

fn escape(text: &str, out: &mut String) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
}

/// JSON safe inside a script element.
pub fn json(value: &impl serde::Serialize) -> Part {
    let text = serde_json::to_string(value).expect("plain data serializes");
    Part::Raw(text.replace("</", "<\\/"))
}

impl El {
    pub fn attr(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.attrs.push((key, value.into()));
        self
    }

    pub fn kid(mut self, kid: impl Into<Part>) -> Self {
        self.kids.push(kid.into());
        self
    }

    pub fn kids(mut self, kids: impl IntoIterator<Item = impl Into<Part>>) -> Self {
        self.kids.extend(kids.into_iter().map(Into::into));
        self
    }

    pub fn text(self, body: impl Into<String>) -> Self {
        self.kid(Part::Text(body.into()))
    }

    fn render(&self, out: &mut String) {
        out.push('<');
        out.push_str(self.tag);
        for (k, v) in &self.attrs {
            out.push(' ');
            out.push_str(k);
            out.push_str("=\"");
            escape(v, out);
            out.push('"');
        }
        out.push('>');
        for kid in &self.kids {
            match kid {
                Part::El(e) => e.render(out),
                Part::Text(t) => escape(t, out),
                Part::Raw(r) => out.push_str(r),
            }
        }
        out.push_str("</");
        out.push_str(self.tag);
        out.push('>');
    }

    /// A whole document, with this element as its root.
    pub fn page(&self) -> String {
        let mut out = String::from("<!DOCTYPE html>\n");
        self.render(&mut out);
        out.push('\n');
        out
    }
}

impl From<El> for Part {
    fn from(e: El) -> Part {
        Part::El(e)
    }
}

/// A table with a header row, cells as text.
pub fn table(head: &[&str], rows: impl IntoIterator<Item = Vec<Part>>) -> El {
    let head = el("tr").kids(head.iter().map(|h| text("th", *h)));
    let body = rows.into_iter().map(|cells| {
        let cells = cells.into_iter().map(|c| el("td").kid(c));
        el("tr").kids(cells)
    });
    el("table").kid(head).kids(body)
}

impl From<String> for Part {
    fn from(s: String) -> Part {
        Part::Text(s)
    }
}

impl From<&str> for Part {
    fn from(s: &str) -> Part {
        Part::Text(s.into())
    }
}
