//! GitHub calls, made as the app or during sign-in.

use crate::board::{Pull, Tier};
use crate::overlap;
use crate::{Call, Config, Fault, Host};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;
use serde_json::{Value, json};

const AGENT: &str = "tessera-tiers";
const WAITS: &str = "needs-human-review";
/// Seconds an app token lasts. GitHub allows 600.
const JWT_LIFE: u64 = 540;

/// Label events arrive oldest first.
const QUERY: &str = "query($owner:String!,$name:String!,$after:String){\
repository(owner:$owner,name:$name){\
pullRequests(states:OPEN,first:50,after:$after){\
pageInfo{hasNextPage endCursor}\
nodes{number title isDraft author{login __typename}\
labels(first:100){nodes{name}}\
commits(last:1){nodes{commit{statusCheckRollup{state}}}}\
timelineItems(itemTypes:[LABELED_EVENT],last:100){nodes{\
...on LabeledEvent{createdAt label{name} actor{login __typename}}}}\
comments(last:100){nodes{author{login __typename} body}}}}}}";

#[derive(Deserialize)]
struct Nodes<T> {
    nodes: Vec<T>,
}

#[derive(Deserialize)]
struct Named {
    name: String,
}

#[derive(Deserialize)]
struct Actor {
    login: String,
    #[serde(rename = "__typename")]
    kind: String,
}

impl Actor {
    /// True for the bot user of the app `slug`.
    fn is_app(&self, slug: &str) -> bool {
        self.kind == "Bot" && self.login.trim_end_matches("[bot]") == slug
    }
}

#[derive(Deserialize)]
struct Tip {
    commit: Commit,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Commit {
    status_check_rollup: Option<Rollup>,
}

#[derive(Deserialize)]
struct Rollup {
    state: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Labeled {
    created_at: String,
    label: Named,
    actor: Option<Actor>,
}

#[derive(Deserialize)]
struct Comment {
    author: Option<Actor>,
    body: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Node {
    number: u64,
    title: String,
    is_draft: bool,
    author: Option<Actor>,
    labels: Nodes<Named>,
    commits: Nodes<Tip>,
    timeline_items: Nodes<Labeled>,
    comments: Nodes<Comment>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Page {
    page_info: PageInfo,
    nodes: Vec<Node>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageInfo {
    has_next_page: bool,
    end_cursor: Option<String>,
}

fn pull(node: Node, slug: &str) -> Pull {
    let has = |name: &str| node.labels.nodes.iter().any(|l| l.name == name);
    let mut tier = Tier::Untagged;
    for t in [Tier::Landing, Tier::Priority] {
        if t.label().is_some_and(has) {
            tier = t;
        }
    }
    let mut pull = Pull {
        number: node.number,
        title: node.title,
        author: node.author.map(|a| a.login).unwrap_or_default(),
        draft: node.is_draft,
        waits: has(WAITS),
        ci: None,
        tier,
        since: String::new(),
        setter: String::new(),
        trusted: tier == Tier::Untagged,
        overlaps: Vec::new(),
    };
    for tip in node.commits.nodes {
        pull.ci = tip
            .commit
            .status_check_rollup
            .map(|r| r.state.to_lowercase());
    }
    for event in node.timeline_items.nodes {
        if Some(event.label.name.as_str()) == tier.label() {
            pull.since = event.created_at;
            pull.trusted = event.actor.as_ref().is_some_and(|a| a.is_app(slug));
            pull.setter = event.actor.map(|a| a.login).unwrap_or_default();
        }
    }
    for comment in node.comments.nodes {
        let ours = comment.author.is_some_and(|a| a.is_app(overlap::AUTHOR));
        if ours && comment.body.starts_with(overlap::MARKER) {
            pull.overlaps = overlap::read(&comment.body);
        }
    }
    pull
}

/// The app's access to the repository.
pub struct Install {
    token: String,
    slug: String,
}

pub struct Client<'a, H> {
    pub host: &'a H,
    pub config: &'a Config,
}

impl<H: Host> Client<'_, H> {
    async fn send(
        &self,
        method: &'static str,
        url: String,
        auth: Option<&str>,
        body: Option<(&'static str, String)>,
    ) -> Result<Value, Fault> {
        let mut headers = vec![
            ("accept", "application/vnd.github+json".to_string()),
            ("user-agent", AGENT.to_string()),
        ];
        if let Some(auth) = auth {
            headers.push(("authorization", format!("Bearer {auth}")));
        }
        let body = body.map(|(kind, text)| {
            headers.push(("content-type", kind.to_string()));
            text
        });
        let what = format!("{method} {url}");
        let call = Call {
            method,
            url,
            headers,
            body,
        };
        let reply = self.host.fetch(call).await?;
        if !(200..300).contains(&reply.status) {
            let text = format!("GitHub answered {} for {what}", reply.status);
            return Err(Fault::new(502, text));
        }
        if reply.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&reply.body)
            .map_err(|e| Fault::new(502, format!("GitHub sent bad JSON for {what}: {e}")))
    }

    async fn api(
        &self,
        method: &'static str,
        path: &str,
        auth: &str,
        body: Option<Value>,
    ) -> Result<Value, Fault> {
        let url = format!("{}{path}", self.config.api);
        let body = body.map(|b| ("application/json", b.to_string()));
        self.send(method, url, Some(auth), body).await
    }

    /// A token that proves this is the app.
    fn jwt(&self) -> Result<String, Fault> {
        let now = self.host.now();
        let claims = json!({
            "iat": now - 60,
            "exp": now + JWT_LIFE,
            "iss": self.config.client_id,
        });
        let head = URL_SAFE_NO_PAD.encode(r#"{"alg":"RS256","typ":"JWT"}"#);
        let text = format!("{head}.{}", URL_SAFE_NO_PAD.encode(claims.to_string()));
        let tag = self.host.sign(text.as_bytes())?;
        Ok(format!("{text}.{}", URL_SAFE_NO_PAD.encode(tag)))
    }

    /// An installation token for this repository alone.
    pub async fn install(&self) -> Result<Install, Fault> {
        let jwt = self.jwt()?;
        let repo = &self.config.repo;
        let path = format!("/repos/{repo}/installation");
        let found = self.api("GET", &path, &jwt, None).await?;
        let name = repo.rsplit('/').next();
        let path = format!("/app/installations/{}/access_tokens", found["id"]);
        let body = json!({ "repositories": [name] });
        let made = self.api("POST", &path, &jwt, Some(body)).await?;
        match (made["token"].as_str(), found["app_slug"].as_str()) {
            (Some(token), Some(slug)) => Ok(Install {
                token: token.into(),
                slug: slug.into(),
            }),
            _ => Err(Fault::new(
                502,
                "The GitHub reply lacks an installation token",
            )),
        }
    }

    /// Every open pull request.
    pub async fn pulls(&self, install: &Install) -> Result<Vec<Pull>, Fault> {
        let (owner, name) = self.config.repo.split_once('/').unwrap_or_default();
        let mut pulls = Vec::new();
        let mut after = None;
        loop {
            let variables = json!({ "owner": owner, "name": name, "after": after });
            let body = json!({ "query": QUERY, "variables": variables });
            let mut reply = self
                .api("POST", "/graphql", &install.token, Some(body))
                .await?;
            let page = reply["data"]["repository"]["pullRequests"].take();
            let page: Page = serde_json::from_value(page).map_err(|e| {
                Fault::new(502, format!("The GitHub reply lacks pull requests: {e}"))
            })?;
            for node in page.nodes {
                pulls.push(pull(node, &install.slug));
            }
            if !page.page_info.has_next_page {
                return Ok(pulls);
            }
            after = page.page_info.end_cursor;
        }
    }

    /// Sets the label again. The app becomes its setter.
    pub async fn retier(&self, install: &Install, number: u64, tier: Tier) -> Result<(), Fault> {
        let repo = &self.config.repo;
        let token = &install.token;
        let path = format!("/repos/{repo}/pulls/{number}");
        let found = self.api("GET", &path, token, None).await?;
        if found["state"] != "open" {
            return Err(Fault::new(409, format!("#{number} is not open")));
        }
        let labels = format!("/repos/{repo}/issues/{number}/labels");
        for label in found["labels"].as_array().into_iter().flatten() {
            let name = label["name"].as_str().unwrap_or_default();
            if Tier::named(name).is_some_and(|t| t.label().is_some()) {
                self.api("DELETE", &format!("{labels}/{name}"), token, None)
                    .await?;
            }
        }
        if let Some(label) = tier.label() {
            let body = json!({ "labels": [label] });
            self.api("POST", &labels, token, Some(body)).await?;
        }
        Ok(())
    }

    /// The login behind an OAuth `code`.
    pub async fn login(&self, code: &str) -> Result<String, Fault> {
        let form = form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", &self.config.client_id)
            .append_pair("client_secret", &self.config.client_secret)
            .append_pair("code", code)
            .finish();
        let url = format!("{}/login/oauth/access_token", self.config.web);
        let body = Some(("application/x-www-form-urlencoded", form));
        let made = self.send("POST", url, None, body).await?;
        let Some(token) = made["access_token"].as_str() else {
            return Err(Fault::new(403, "GitHub refused the sign-in"));
        };
        let user = self.api("GET", "/user", token, None).await?;
        match user["login"].as_str() {
            Some(login) => Ok(login.into()),
            None => Err(Fault::new(502, "The GitHub reply lacks a login")),
        }
    }
}
