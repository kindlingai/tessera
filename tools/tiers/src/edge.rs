//! The Cloudflare Worker entry point.

use crate::{Call, Config, Fault, Host, Reply, Request, serve};
use rsa::PrivateKey;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs1v15::SigningKey;
use rsa::signature::{SignatureEncoding, Signer};
use sha2::Sha256;
use std::sync::OnceLock;
use worker::{Context, Env, Fetch, Headers, Method, RequestInit, RequestRedirect, event};

fn fault(error: worker::Error) -> Fault {
    Fault::new(502, error.to_string())
}

/// The app key, read once per Worker instance.
static KEY: OnceLock<SigningKey<Sha256>> = OnceLock::new();

struct Edge {
    key: &'static SigningKey<Sha256>,
}

fn bad(name: &str, error: impl std::fmt::Display) -> worker::Error {
    worker::Error::RustError(format!("{name}: {error}"))
}

impl Host for Edge {
    async fn fetch(&self, call: Call) -> Result<Reply, Fault> {
        let headers = Headers::new();
        for (key, value) in &call.headers {
            headers.set(key, value).map_err(fault)?;
        }
        let mut init = RequestInit::new();
        init.with_method(Method::from(call.method.to_string()));
        init.with_headers(headers);
        init.with_body(call.body.map(Into::into));
        // The client follows redirects, so storage never sees a token.
        init.with_redirect(RequestRedirect::Manual);
        let request = worker::Request::new_with_init(&call.url, &init).map_err(fault)?;
        let mut reply = Fetch::Request(request).send().await.map_err(fault)?;
        Ok(Reply {
            status: reply.status_code(),
            location: reply.headers().get("location").map_err(fault)?,
            body: reply.bytes().await.map_err(fault)?,
        })
    }

    fn sign(&self, text: &[u8]) -> Result<Vec<u8>, Fault> {
        Ok(self.key.sign(text).to_vec())
    }

    fn nonce(&self) -> [u8; 16] {
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce).expect("the runtime has a random source");
        nonce
    }

    fn now(&self) -> u64 {
        worker::Date::now().as_millis() / 1000
    }
}

#[event(fetch)]
async fn fetch(
    mut req: worker::Request,
    env: Env,
    _ctx: Context,
) -> worker::Result<worker::Response> {
    let secret = |name: &str| env.secret(name).map(|s| s.to_string());
    let mut editors = Vec::new();
    for id in secret("EDITORS")?.split(',') {
        editors.push(id.trim().parse().map_err(|e| bad("EDITORS", e))?);
    }
    let config = Config {
        repo: env.var("REPO")?.to_string(),
        client_id: secret("CLIENT_ID")?,
        client_secret: secret("CLIENT_SECRET")?,
        session_key: secret("SESSION_KEY")?.into_bytes(),
        editors,
        origin: secret("ORIGIN")?,
        api: "https://api.github.com".into(),
        web: "https://github.com".into(),
    };
    if KEY.get().is_none() {
        let key = PrivateKey::from_pkcs1_pem(&secret("APP_KEY")?);
        let _ = KEY.set(SigningKey::new(key.map_err(|e| bad("APP_KEY", e))?));
    }
    let key = KEY.get().expect("the key was set above");
    let edge = Edge { key };
    let url = req.url()?;
    let header = |name: &str| req.headers().get(name).map(Option::unwrap_or_default);
    let request = Request {
        method: req.method().to_string(),
        path: url.path().into(),
        query: url.query().unwrap_or_default().into(),
        cookie: header("cookie")?,
        source: header("origin")?,
        body: req.text().await?,
    };
    let done = serve(&edge, &config, &request).await;
    let headers = Headers::new();
    for (key, value) in &done.headers {
        headers.append(key, value)?;
    }
    let reply = worker::Response::ok(done.body)?;
    Ok(reply.with_status(done.status).with_headers(headers))
}
