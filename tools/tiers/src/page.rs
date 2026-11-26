//! Draws the board. The element builder escapes all text.

use crate::Config;
use crate::board::{Pull, Tier, column};
use crate::overlap::Kind;
use crate::session::Session;
use tessera_html::{El, Part, el, text};

const STYLE: &str = "body{font:14px system-ui;margin:1rem}\
main{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}\
article{border:1px solid #888;border-radius:4px;padding:.5rem;margin:.5rem 0}\
.tag{border:1px solid #888;border-radius:4px;padding:0 .3rem;margin-right:.3rem}\
.bad{border-color:#c00;color:#c00}li.bad{border:0}\
form{display:inline}ul{margin:.3rem 0;padding-left:1.2rem}";

pub struct View<'a> {
    pub config: &'a Config,
    pub session: Option<&'a Session>,
    /// The session may change tiers.
    pub editor: bool,
    pub pulls: &'a [Pull],
}

fn tag(body: impl Into<String>, bad: bool) -> El {
    let class = if bad { "tag bad" } else { "tag" };
    text("span", body).attr("class", class)
}

fn hidden(name: &'static str, value: impl Into<String>) -> El {
    el("input")
        .attr("type", "hidden")
        .attr("name", name)
        .attr("value", value)
}

fn tags(pull: &Pull) -> El {
    let mut tags = el("p");
    let ci = pull.ci.as_deref().unwrap_or("no checks");
    let red = matches!(ci, "failure" | "error");
    tags = tags.kid(tag(format!("CI {ci}"), red));
    if pull.draft {
        tags = tags.kid(tag("draft", false));
    }
    if pull.waits {
        tags = tags.kid(tag("waits on a human", false));
    }
    if !pull.trusted {
        let note = format!("label set by {}, CI ignores it", pull.setter);
        tags = tags.kid(tag(note, true));
    }
    tags
}

fn overlaps(pull: &Pull) -> El {
    let mut list = el("ul");
    for overlap in &pull.overlaps {
        let verb = match overlap.kind {
            Kind::Conflict => "conflicts with",
            Kind::Shared => "shares files with",
        };
        let files = overlap.files.join(", ");
        let line = format!("{verb} #{}: {files}", overlap.other);
        let class = if overlap.kind == Kind::Conflict {
            "bad"
        } else {
            ""
        };
        list = list.kid(text("li", line).attr("class", class));
    }
    list
}

/// Buttons that move a pull request to another tier.
fn controls(pull: &Pull, session: &Session) -> El {
    let mut form = el("form")
        .attr("method", "post")
        .attr("action", "/tier")
        .kid(hidden("csrf", &session.csrf))
        .kid(hidden("pr", pull.number.to_string()));
    for tier in Tier::ALL {
        let verb = match tier == pull.tier {
            true if pull.trusted => continue,
            true => "confirm",
            false => "to",
        };
        let button = text("button", format!("{verb} {}", tier.name()));
        form = form.kid(button.attr("name", "tier").attr("value", tier.name()));
    }
    form
}

fn card(view: &View, pull: &Pull) -> El {
    let Config { web, repo, .. } = view.config;
    let link = text("a", format!("#{} {}", pull.number, pull.title))
        .attr("href", format!("{web}/{repo}/pull/{}", pull.number));
    let by = format!(" by {}", pull.author);
    let mut card = el("article")
        .kid(el("p").kid(link).text(by))
        .kid(tags(pull))
        .kid(overlaps(pull));
    if let Some(session) = view.session.filter(|_| view.editor) {
        card = card.kid(controls(pull, session));
    }
    card
}

/// The sign-in link, or the login and a sign-out button.
fn account(view: &View) -> El {
    let Some(session) = view.session else {
        return el("p").kid(text("a", "Sign in with GitHub").attr("href", "/login"));
    };
    let rights = if view.editor { "" } else { " (read only)" };
    let out = el("form")
        .attr("method", "post")
        .attr("action", "/logout")
        .kid(hidden("csrf", &session.csrf))
        .kid(text("button", "Sign out"));
    el("p").text(format!("{}{rights} ", session.login)).kid(out)
}

pub(crate) fn board(view: &View) -> String {
    let mut main = el("main");
    for tier in Tier::ALL {
        let mut section = el("section").kid(text("h2", tier.name()));
        for pull in column(view.pulls, tier) {
            section = section.kid(card(view, pull));
        }
        main = main.kid(section);
    }
    let title = format!("{} tiers", view.config.repo);
    let head = el("head")
        .kid(el("meta").attr("charset", "utf-8"))
        .kid(text("title", title.clone()))
        .kid(el("style").kid(Part::Raw(STYLE.into())));
    let body = el("body")
        .kid(text("h1", title))
        .kid(account(view))
        .kid(main);
    el("html").attr("lang", "en").kid(head).kid(body).page()
}
