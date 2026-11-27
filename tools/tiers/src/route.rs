//! Maps requests to pages and tier changes.

use crate::board::Tier;
use crate::github::Client;
use crate::page::{self, View};
use crate::session::{self, Session};
use crate::{Config, Fault, Host};
use crate::{cache, limits};

const STATE: &str = "__Host-state";
/// Seconds to finish a sign-in.
const STATE_LIFE: u64 = 600;
const POLICY: &str = "default-src 'none'; style-src 'unsafe-inline'; \
form-action 'self'; frame-ancestors 'none'; base-uri 'none'";

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
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
            // Under `no-referrer` a browser sends `Origin: null` with a form.
            ("referrer-policy", "same-origin".into()),
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

    /// Refuses a session whose user is off the allow list.
    fn editor<'a>(&self, session: &'a Session) -> Result<&'a Session, Fault> {
        if self.config.editors.contains(&session.id) {
            return Ok(session);
        }
        let text = format!("{} may not use this board.", session.login);
        Err(Fault::new(403, text))
    }

    /// A tab. GitHub is called for editors alone.
    async fn tab(&self, path: &str) -> Result<Response, Fault> {
        let Some(session) = &self.session else {
            return Ok(match path {
                "/" => Response::new(200, "text/html", page::welcome(self.config)),
                _ => Response::redirect("/".into()),
            });
        };
        let session = self.editor(session)?;
        let config = self.config;
        let client = self.client();
        let install = client.install().await?;
        let page = match path {
            "/cache" => cache::render(config, session, &cache::read(&client, &install).await?),
            "/limits" => limits::render(config, session, &limits::read(&client, &install).await?),
            _ => {
                let pulls = client.pulls(&install).await?;
                page::board(&View {
                    config,
                    session,
                    pulls: &pulls,
                })
            }
        };
        Ok(Response::new(200, "text/html", page))
    }

    fn login(&self) -> Response {
        let state = session::token(self.host.nonce());
        let to = form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", &self.config.client_id)
            .append_pair("state", &state)
            .append_pair("redirect_uri", &format!("{}/auth", self.config.origin))
            .finish();
        let to = format!("{}/login/oauth/authorize?{to}", self.config.web);
        Response::redirect(to).with("set-cookie", session::bake(STATE, &state, STATE_LIFE))
    }

    async fn auth(&self) -> Result<Response, Fault> {
        let query = form(&self.req.query);
        let sent = session::cookie(&self.req.cookie, STATE).unwrap_or_default();
        if sent.is_empty() || !session::same(sent, field(&query, "state")) {
            return Err(Fault::new(403, "The sign-in did not start here."));
        }
        let (login, id) = self.client().login(field(&query, "code")).await?;
        let session = Session {
            login,
            id,
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
        let ours = self.req.source == self.config.origin;
        if !ours || !session::same(field(form, "csrf"), &session.csrf) {
            return Err(Fault::new(403, "The request did not come from this page."));
        }
        Ok(session)
    }

    async fn retier(&self) -> Result<Response, Fault> {
        let form = form(&self.req.body);
        self.editor(self.guard(&form)?)?;
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
    if config.session_key.len() < session::KEY_MIN {
        let text = format!("The session key needs {} bytes.", session::KEY_MIN);
        return Response::new(500, "text/plain", text);
    }
    let sent = session::cookie(&req.cookie, session::NAME);
    let session = sent.and_then(|c| session::open(&config.session_key, c, host.now()));
    let site = Site {
        host,
        config,
        req,
        session,
    };
    let done = match (req.method.as_str(), req.path.as_str()) {
        ("GET", path @ ("/" | "/cache" | "/limits")) => site.tab(path).await,
        ("GET", "/login") => Ok(site.login()),
        // GitHub sends the browser here after the app is installed.
        ("GET", "/setup") => Ok(Response::redirect("/".into())),
        ("GET", "/auth") => site.auth().await,
        ("POST", "/tier") => site.retier().await,
        ("POST", "/logout") => site.logout(),
        _ => Err(Fault::new(404, "Not found.")),
    };
    done.unwrap_or_else(|fault| Response::new(fault.status, "text/plain", fault.text))
}
