//! The routes against a fake GitHub.

use crate::session::{self, Session};
use crate::{Call, Config, Fault, Host, Reply, Request, Response, serve};
use std::cell::RefCell;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

const KEY: &[u8] = b"test key";
const HERE: &str = "https://tiers.test";

/// #3 was labelled before #2. An agent labelled #4.
const PULLS: &str = r#"{"data":{"repository":{"pullRequests":{
"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[
{"number":2,"title":"<script>alert(1)</script>","isDraft":true,
 "author":{"login":"ann","__typename":"User"},
 "labels":{"nodes":[{"name":"priority"},{"name":"needs-human-review"}]},
 "commits":{"nodes":[{"commit":{"statusCheckRollup":{"state":"FAILURE"}}}]},
 "timelineItems":{"nodes":[{"createdAt":"2026-11-02T00:00:00Z",
  "label":{"name":"priority"},"actor":{"login":"tiers","__typename":"Bot"}}]},
 "comments":{"nodes":[{"author":{"login":"github-actions","__typename":"Bot"},
  "body":"<!-- pr-overlap -->\n- conflict #3: `a/<b>.rs`\n- shared #4: `c.rs`, `d.rs`"}]}},
{"number":3,"title":"first","isDraft":false,"author":null,
 "labels":{"nodes":[{"name":"priority"}]},"commits":{"nodes":[]},
 "timelineItems":{"nodes":[{"createdAt":"2026-11-01T00:00:00Z",
  "label":{"name":"priority"},"actor":{"login":"tiers","__typename":"Bot"}}]},
 "comments":{"nodes":[{"author":{"login":"ann","__typename":"User"},
  "body":"<!-- pr-overlap -->\n- conflict #9: `forged.rs`"}]}},
{"number":4,"title":"agent","isDraft":false,"author":null,
 "labels":{"nodes":[{"name":"landing"}]},"commits":{"nodes":[]},
 "timelineItems":{"nodes":[{"createdAt":"2026-11-03T00:00:00Z",
  "label":{"name":"landing"},"actor":{"login":"mmastrac","__typename":"User"}}]},
 "comments":{"nodes":[]}}]}}}}"#;

/// Records each call as `METHOD url authorization`.
#[derive(Default)]
struct Fake {
    calls: RefCell<Vec<String>>,
}

impl Host for Fake {
    async fn fetch(&self, call: Call) -> Result<Reply, Fault> {
        let auth = call.headers.iter().find(|(key, _)| *key == "authorization");
        let auth = auth.map(|(_, value)| value.as_str()).unwrap_or_default();
        let path = call.url.split_once(".test").unwrap().1;
        let body = match (call.method, path) {
            ("GET", "/repos/o/r/installation") => r#"{"id":7,"app_slug":"tiers"}"#,
            ("POST", "/app/installations/7/access_tokens") => r#"{"token":"app"}"#,
            ("POST", "/graphql") => PULLS,
            ("GET", "/repos/o/r/pulls/5") => r#"{"state":"open","labels":[{"name":"landing"}]}"#,
            ("POST", "/login/oauth/access_token") => r#"{"access_token":"user"}"#,
            ("GET", "/user") => r#"{"login":"Matt"}"#,
            _ => "[]",
        };
        let line = format!("{} {path} {auth}", call.method);
        self.calls.borrow_mut().push(line);
        Ok(Reply {
            status: 200,
            body: body.into(),
        })
    }

    fn sign(&self, _: &[u8]) -> Result<Vec<u8>, Fault> {
        Ok(vec![1])
    }

    fn nonce(&self) -> [u8; 16] {
        [7; 16]
    }

    fn now(&self) -> u64 {
        1000
    }
}

fn wait<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}

fn send(fake: &Fake, request: &Request) -> Response {
    let config = Config {
        repo: "o/r".into(),
        client_id: "id".into(),
        client_secret: "secret".into(),
        session_key: KEY.into(),
        editors: vec!["matt".into()],
        api: "https://api.test".into(),
        web: "https://web.test".into(),
    };
    wait(serve(fake, &config, request))
}

/// A request with a session for `login`.
fn request(method: &str, path: &str, login: &str, body: &str) -> Request {
    let session = Session {
        login: login.into(),
        csrf: "token".into(),
        expires: 2000,
    };
    let (path, query) = path.split_once('?').unwrap_or((path, ""));
    let mut cookie = format!("__Host-state=st; {}=", session::NAME);
    if !login.is_empty() {
        cookie += &session::seal(KEY, &session);
    }
    Request {
        method: method.into(),
        path: path.into(),
        query: query.into(),
        origin: HERE.into(),
        cookie,
        source: HERE.into(),
        body: body.into(),
    }
}

fn ask(fake: &Fake, method: &str, path: &str, login: &str, body: &str) -> Response {
    send(fake, &request(method, path, login, body))
}

#[test]
fn editor_moves_tier() {
    let fake = Fake::default();
    let done = ask(
        &fake,
        "POST",
        "/tier",
        "Matt",
        "pr=5&tier=priority&csrf=token",
    );
    assert_eq!(done.status, 303);
    let calls = fake.calls.borrow();
    assert_eq!(
        calls[3],
        "DELETE /repos/o/r/issues/5/labels/landing Bearer app"
    );
    assert_eq!(calls[4], "POST /repos/o/r/issues/5/labels Bearer app");
}

#[test]
fn refuses_other_callers() {
    let fake = Fake::default();
    let body = "pr=5&tier=priority&csrf=token";
    assert_eq!(ask(&fake, "POST", "/tier", "", body).status, 401);
    assert_eq!(ask(&fake, "POST", "/tier", "ann", body).status, 403);
    let forged = "pr=5&tier=priority&csrf=guess";
    assert_eq!(ask(&fake, "POST", "/tier", "Matt", forged).status, 403);
    let mut foreign = request("POST", "/tier", "Matt", body);
    foreign.source = "https://evil.test".into();
    assert_eq!(send(&fake, &foreign).status, 403);
    assert!(fake.calls.borrow().is_empty());
}

#[test]
fn board_escapes_text() {
    let page = ask(&Fake::default(), "GET", "/", "ann", "").body;
    assert!(page.contains("&lt;script&gt;alert(1)") && !page.contains("<script>"));
    assert!(page.contains("conflicts with #3: a/&lt;b&gt;.rs"));
    assert!(page.contains("shares files with #4: c.rs, d.rs"));
    assert!(page.contains("label set by mmastrac") && !page.contains("forged.rs"));
    assert!(page.find("#3 first").unwrap() < page.find("#2 ").unwrap());
    assert!(!page.contains("action=\"/tier\""));
}

#[test]
fn sign_in_sets_session() {
    let fake = Fake::default();
    assert_eq!(
        ask(&fake, "GET", "/callback?code=c&state=no", "", "").status,
        403
    );
    let done = ask(&fake, "GET", "/callback?code=c&state=st", "", "");
    let cookie = &done.headers.last().unwrap().1;
    assert!(cookie.ends_with("; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=28800"));
    let value = cookie.split_once(';').unwrap().0;
    let value = session::cookie(value, session::NAME).unwrap();
    assert_eq!(session::open(KEY, value, 1000).unwrap().login, "Matt");
    assert!(session::open(b"other", value, 1000).is_none());
}
