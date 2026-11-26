//! The routes against a fake GitHub.

use crate::session::{self, Session};
use crate::{Call, Config, Fault, Host, Reply, Request, Response, serve};
use std::cell::RefCell;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

const KEY: &[u8] = b"a test key of thirty-two bytes..";
const HERE: &str = "https://tiers.test";

/// #3 was labelled first. An agent labelled #4.
const PULLS: &str = r#"{"data":{"repository":{"pullRequests":{
"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[
{"number":2,"title":"<script>alert(1)</script>","isDraft":true,"mergeable":"MERGEABLE",
 "author":{"login":"ann","__typename":"User"},
 "labels":{"nodes":[{"name":"priority"},{"name":"needs-human-review"}]},
 "commits":{"nodes":[{"commit":{"statusCheckRollup":{"state":"FAILURE"}}}]},
 "timelineItems":{"nodes":[{"createdAt":"2026-11-02T00:00:00Z",
  "label":{"name":"priority"},"actor":{"login":"tiers","__typename":"Bot"}}]},
 "comments":{"nodes":[{"author":{"login":"github-actions","__typename":"Bot"},
  "body":"<!-- conflict-notice -->\n### Overlap with other open pull requests\nA conflict line names files that conflict with that PR.\nThe PR that merges second must rebase. A shared line names\nfiles both PRs change that merge cleanly. This notice fails no check.\n\n- conflict #3: `a/<b>.rs`\n- shared #4: `c.rs` `d.rs`\n"}]}},
{"number":3,"title":"first","isDraft":false,"mergeable":"MERGEABLE","author":null,
 "labels":{"nodes":[{"name":"priority"}]},"commits":{"nodes":[]},
 "timelineItems":{"nodes":[{"createdAt":"2026-11-01T00:00:00Z",
  "label":{"name":"priority"},"actor":{"login":"tiers","__typename":"Bot"}}]},
 "comments":{"nodes":[{"author":{"login":"ann","__typename":"User"},
  "body":"<!-- conflict-notice -->\n- conflict #9: `forged.rs`"}]}},
{"number":4,"title":"agent","isDraft":false,"mergeable":"CONFLICTING","author":null,
 "labels":{"nodes":[{"name":"landing"}]},"commits":{"nodes":[]},
 "timelineItems":{"nodes":[{"createdAt":"2026-11-03T00:00:00Z",
  "label":{"name":"landing"},"actor":{"login":"mmastrac","__typename":"User"}}]},
 "comments":{"nodes":[]}}]}}}}"#;

/// Usage at the default limit of 10 GiB.
const USAGE: &str = r#"{"active_caches_size_in_bytes":10737418240,"active_caches_count":3}"#;
const CACHES: &str = r#"{"actions_caches":[
{"ref":"refs/pull/7/merge","size_in_bytes":1073741824,"last_accessed_at":"2026-11-02"},
{"ref":"refs/heads/main","size_in_bytes":1073741824,"last_accessed_at":"2026-11-01"},
{"ref":"refs/heads/main","size_in_bytes":1073741824,"last_accessed_at":"2026-11-03"}]}"#;

/// Entries from `stickler lint --json` run on tessera.
const LINT: &str = r#"{"tool":"lint","title":"Code shape","rules":{},"findings":[],
"skipped":null,"measures":[
{"rule":"file-lines","path":"engine/models/src/dflash.rs","value":1000,"max":1000},
{"rule":"crate-lines","path":"engine/engine/Cargo.toml","value":12723,"max":15000},
{"rule":"crate-files","path":"engine/gpu/dl/Cargo.toml","value":1,"max":25}]}"#;
const RUNS: &str = r#"{"workflow_runs":[{"id":1,"head_sha":"aaaaaaaaaaaa"},
{"id":2,"head_sha":"bbbbbbbbbbbb"}]}"#;

/// An artifact zip: one deflated entry, with a descriptor.
fn zip(name: &str, data: &[u8]) -> Vec<u8> {
    let packed = miniz_oxide::deflate::compress_to_vec(data, 6);
    let n = name.len() as u32;
    let mut out = Vec::new();
    let mut put = |fields: &[(u32, usize)], tail: &[u8]| {
        for &(v, width) in fields {
            let bytes = v.to_le_bytes();
            out.extend((0..width).map(|i| bytes.get(i).copied().unwrap_or(0)));
        }
        out.extend_from_slice(tail);
    };
    let local = [
        (0x0403_4b50, 4),
        (20, 2),
        (8, 2),
        (8, 2),
        (0, 16),
        (n, 2),
        (0, 2),
    ];
    put(&local, name.as_bytes());
    put(&[], &packed);
    let central = (30 + n as usize + packed.len()) as u32;
    let size = packed.len() as u32;
    let head = [
        (0x0201_4b50, 4),
        (20, 2),
        (20, 2),
        (8, 2),
        (8, 2),
        (0, 8),
        (size, 4),
    ];
    put(&head, &[]);
    put(
        &[(data.len() as u32, 4), (n, 2), (0, 12), (0, 4)],
        name.as_bytes(),
    );
    let count = [
        (0x0605_4b50, 4),
        (0, 4),
        (1, 2),
        (1, 2),
        (46 + n, 4),
        (central, 4),
        (0, 2),
    ];
    put(&count, &[]);
    out
}

/// Records each call as `METHOD path authorization body`.
#[derive(Default)]
struct Fake {
    calls: RefCell<Vec<String>>,
}

impl Host for Fake {
    async fn fetch(&self, call: Call) -> Result<Reply, Fault> {
        let auth = call.headers.iter().find(|(key, _)| *key == "authorization");
        let auth = auth.map(|(_, value)| value.as_str()).unwrap_or_default();
        let path = call.url.split_once(".test").unwrap().1;
        let zip =
            (call.url == "https://store.test/blob").then(|| zip("lint.json", LINT.as_bytes()));
        let body = match (call.method, path) {
            ("GET", "/repos/o/r/installation") => r#"{"id":7,"app_slug":"tiers"}"#,
            ("POST", "/app/installations/7/access_tokens") => r#"{"token":"app"}"#,
            ("POST", "/graphql") => PULLS,
            ("GET", "/repos/o/r/pulls/5") => r#"{"state":"open","labels":[{"name":"landing"}]}"#,
            ("POST", "/login/oauth/access_token") => r#"{"access_token":"user"}"#,
            ("GET", "/user") => r#"{"login":"Matt","id":7}"#,
            ("GET", "/repos/o/r/actions/cache/usage") => USAGE,
            ("GET", "/repos/o/r/actions/caches?per_page=100&page=1") => CACHES,
            ("GET", path) if path.contains("/runs?branch=main&event=push") => RUNS,
            ("GET", "/repos/o/r/actions/runs/2/artifacts?name=checks-lint") => {
                r#"{"artifacts":[{"id":5}]}"#
            }
            _ => "[]",
        };
        let sent = call.body.unwrap_or_default();
        let line = format!("{} {path} {auth} {sent}", call.method);
        self.calls.borrow_mut().push(line);
        // The landing label is gone when the delete arrives.
        let moved = path.ends_with("/artifacts/5/zip");
        let status = match (call.method, path.ends_with("/storage-limit")) {
            _ if moved => 302,
            ("DELETE", _) => 404,
            (_, true) => 402,
            _ => 200,
        };
        Ok(Reply {
            status,
            location: moved.then(|| "https://store.test/blob".into()),
            body: zip.unwrap_or_else(|| body.into()),
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
        editors: vec![7],
        origin: HERE.into(),
        api: "https://api.test".into(),
        web: "https://web.test".into(),
    };
    wait(serve(fake, &config, request))
}

/// A request with a session for `login`.
fn request(method: &str, path: &str, login: &str, body: &str) -> Request {
    let session = Session {
        login: login.into(),
        id: if login == "Matt" { 7 } else { 9 },
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
    let page = ask(&fake, "GET", "/", "Matt", "").body;
    assert!(page.contains(r#"<button name="tier" value="priority">to priority</button>"#));
    fake.calls.borrow_mut().clear();
    let form = "pr=5&tier=priority&csrf=token";
    let done = ask(&fake, "POST", "/tier", "Matt", form);
    assert_eq!(done.status, 303);
    let calls = fake.calls.borrow();
    let labels = "/repos/o/r/issues/5/labels";
    let added = format!(r#"POST {labels} Bearer app {{"labels":["priority"]}}"#);
    assert_eq!(
        calls[3..],
        [added, format!("DELETE {labels}/landing Bearer app ")]
    );
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
    assert_eq!(ask(&fake, "GET", "/", "ann", "").status, 403);
    assert!(ask(&fake, "GET", "/", "", "").body.contains("/login"));
    assert!(fake.calls.borrow().is_empty());
}

#[test]
fn board_escapes_text() {
    let page = ask(&Fake::default(), "GET", "/", "Matt", "").body;
    assert!(page.contains("&lt;script&gt;alert(1)") && !page.contains("<script>"));
    assert!(page.contains("conflicts with #3: a/&lt;b&gt;.rs"));
    assert!(page.contains("shares files with #4: c.rs, d.rs"));
    assert!(page.contains("label set by mmastrac") && !page.contains("forged.rs"));
    assert!(page.contains("no overlaps") && page.contains("overlaps not checked"));
    assert!(page.find("#3 first").unwrap() < page.find("#2 ").unwrap());
}

#[test]
fn sign_in_sets_session() {
    let fake = Fake::default();
    let refused = ask(&fake, "GET", "/callback?code=c&state=no", "", "");
    assert_eq!(refused.status, 403);
    let done = ask(&fake, "GET", "/callback?code=c&state=st", "", "");
    let cookie = &done.headers.last().unwrap().1;
    assert!(cookie.ends_with("; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=28800"));
    let value = cookie.split_once(';').unwrap().0;
    let value = session::cookie(value, session::NAME).unwrap();
    assert_eq!(session::open(KEY, value, 1000).unwrap().id, 7);
    assert!(session::open(b"other", value, 1000).is_none());
}

#[test]
fn tabs_show_usage() {
    let fake = Fake::default();
    let cache = ask(&fake, "GET", "/cache", "Matt", "").body;
    let main = "<td>refs/heads/main</td><td>2</td><td>2.00 GiB</td><td>2026-11-03</td>";
    assert!(cache.contains("10.00 GiB of 10 GiB, GitHub default (100%)") && cache.contains(main));
    let limits = ask(&fake, "GET", "/limits", "Matt", "").body;
    let file = r#"dflash.rs</td><td>1000 of 1000</td><td><span class="bad">100%"#;
    assert!(limits.contains(">bbbbbbbb</a>. 2 of 3 measures") && limits.contains(file));
    let last = fake.calls.borrow().last().cloned().unwrap();
    assert!(last.starts_with("GET /blob  "));
}
