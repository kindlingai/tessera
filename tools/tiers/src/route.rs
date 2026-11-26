//! Maps requests to pages and tier changes.

use crate::board::Tier;
use crate::github::Client;
use crate::page::{self, View};
use crate::session::{self, Session};
use crate::{Config, Fault, Host};

const STATE: &str = "__Host-state";
/// Seconds to finish a sign-in.
const STATE_LIFE: u64 = 600;
const POLICY: &str = "default-src 'none'; style-src 'unsafe-inline'; \
form-action 'self'; frame-ancestors 'none'; base-uri 'none'";

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    /// The scheme and host the request came to.
    pub origin: String,
    /// The `Cookie` header.
    pub cookie: String,
    /// The `Origin` header.
    pub source: String,
    pub body: String,
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: String,
}

impl Response {
    fn new(status: u16, kind: &str, body: String) -> Response {
        let headers = vec![
            ("content-type", format!("{kind}; charset=utf-8")),
            ("content-security-policy", POLICY.into()),
            ("x-content-type-options", "nosniff".into()),
            ("referrer-policy", "no-referrer".into()),
            ("cache-control", "no-store".into()),
        ];
        Response {
            status,
            headers,
            body,
        }
    }

    fn redirect(to: String) -> Response {
        Response::new(303, "text/plain", String::new()).with("location", to)
    }

    fn with(mut self, key: &'static str, value: String) -> Response {
        self.headers.push((key, value));
        self
    }
}

type Form = Vec<(String, String)>;

fn form(text: &str) -> Form {
    form_urlencoded::parse(text.as_bytes())
        .into_owned()
        .collect()
}

fn field<'a>(form: &'a Form, name: &str) -> &'a str {
    let found = form.iter().find(|(key, _)| key == name);
    found.map(|(_, value)| value.as_str()).unwrap_or_default()
}

struct Site<'a, H> {
    host: &'a H,
    config: &'a Config,
    req: &'a Request,
    session: Option<Session>,
}

impl<H: Host> Site<'_, H> {
    fn client(&self) -> Client<'_, H> {
        Client {
            host: self.host,
            config: self.config,
        }
    }

    fn is_editor(&self, login: &str) -> bool {
        let editors = &self.config.editors;
        editors.iter().any(|e| e.eq_ignore_ascii_case(login))
    }

    async fn home(&self) -> Result<Response, Fault> {
        let client = self.client();
        let install = client.install().await?;
        let pulls = client.pulls(&install).await?;
        let session = self.session.as_ref();
        let view = View {
            config: self.config,
            session,
            editor: session.is_some_and(|s| self.is_editor(&s.login)),
            pulls: &pulls,
        };
        Ok(Response::new(200, "text/html", page::board(&view)))
    }

    fn login(&self) -> Response {
        let state = session::token(self.host.nonce());
        let to = form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", &self.config.client_id)
            .append_pair("state", &state)
            .append_pair("redirect_uri", &format!("{}/callback", self.req.origin))
            .finish();
        let to = format!("{}/login/oauth/authorize?{to}", self.config.web);
        Response::redirect(to).with("set-cookie", session::bake(STATE, &state, STATE_LIFE))
    }

    async fn callback(&self) -> Result<Response, Fault> {
        let query = form(&self.req.query);
        let sent = session::cookie(&self.req.cookie, STATE).unwrap_or_default();
        if sent.is_empty() || !session::same(sent, field(&query, "state")) {
            return Err(Fault::new(403, "The sign-in did not start here."));
        }
        let session = Session {
            login: self.client().login(field(&query, "code")).await?,
            csrf: session::token(self.host.nonce()),
            expires: self.host.now() + session::LIFE,
        };
        let value = session::seal(&self.config.session_key, &session);
        Ok(Response::redirect("/".into())
            .with("set-cookie", session::bake(STATE, "", 0))
            .with(
                "set-cookie",
                session::bake(session::NAME, &value, session::LIFE),
            ))
    }

    /// The session, when the form came from its own page.
    fn guard(&self, form: &Form) -> Result<&Session, Fault> {
        let Some(session) = &self.session else {
            return Err(Fault::new(401, "Sign in first."));
        };
        let ours = self.req.source == self.req.origin;
        if !ours || !session::same(field(form, "csrf"), &session.csrf) {
            return Err(Fault::new(403, "The request did not come from this page."));
        }
        Ok(session)
    }

    async fn retier(&self) -> Result<Response, Fault> {
        let form = form(&self.req.body);
        let session = self.guard(&form)?;
        if !self.is_editor(&session.login) {
            let text = format!("{} may not change tiers.", session.login);
            return Err(Fault::new(403, text));
        }
        let number = field(&form, "pr").parse();
        let (Ok(number), Some(tier)) = (number, Tier::named(field(&form, "tier"))) else {
            return Err(Fault::new(400, "Unknown pull request or tier."));
        };
        let client = self.client();
        let install = client.install().await?;
        client.retier(&install, number, tier).await?;
        Ok(Response::redirect("/".into()))
    }

    fn logout(&self) -> Result<Response, Fault> {
        self.guard(&form(&self.req.body))?;
        let gone = session::bake(session::NAME, "", 0);
        Ok(Response::redirect("/".into()).with("set-cookie", gone))
    }
}

/// Answers one request.
pub async fn serve<H: Host>(host: &H, config: &Config, req: &Request) -> Response {
    let sent = session::cookie(&req.cookie, session::NAME);
    let session = sent.and_then(|c| session::open(&config.session_key, c, host.now()));
    let site = Site {
        host,
        config,
        req,
        session,
    };
    let done = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => site.home().await,
        ("GET", "/login") => Ok(site.login()),
        ("GET", "/callback") => site.callback().await,
        ("POST", "/tier") => site.retier().await,
        ("POST", "/logout") => site.logout(),
        _ => Err(Fault::new(404, "Not found.")),
    };
    done.unwrap_or_else(|fault| Response::new(fault.status, "text/plain", fault.text))
}
