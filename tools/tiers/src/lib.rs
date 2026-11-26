//! A board that sets pull request tiers as an app.

mod board;
#[cfg(target_arch = "wasm32")]
mod edge;
mod github;
mod overlap;
mod page;
mod route;
mod session;

#[cfg(test)]
mod route_test;

pub use route::{Request, Response, serve};

pub struct Config {
    /// The repository as `owner/name`.
    pub repo: String,
    pub client_id: String,
    pub client_secret: String,
    pub session_key: Vec<u8>,
    /// GitHub user ids that may use the board.
    pub editors: Vec<u64>,
    /// The scheme and host that serve the board.
    pub origin: String,
    pub api: String,
    pub web: String,
}

/// An outbound HTTP request.
pub struct Call {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(&'static str, String)>,
    pub body: Option<String>,
}

pub struct Reply {
    pub status: u16,
    pub body: String,
}

/// A failure, with the HTTP status to answer.
#[derive(Debug)]
pub struct Fault {
    pub status: u16,
    pub text: String,
}

impl Fault {
    pub fn new(status: u16, text: impl Into<String>) -> Fault {
        let text = text.into();
        Fault { status, text }
    }
}

/// The runtime under the board.
pub trait Host {
    fn fetch(&self, call: Call) -> impl Future<Output = Result<Reply, Fault>>;
    /// An RS256 signature made with the app key.
    fn sign(&self, text: &[u8]) -> Result<Vec<u8>, Fault>;
    fn nonce(&self) -> [u8; 16];
    /// Seconds since the Unix epoch.
    fn now(&self) -> u64;
}
