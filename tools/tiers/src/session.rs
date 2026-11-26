//! The signed session cookie and its CSRF token.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

pub const NAME: &str = "__Host-session";
/// Seconds a sign-in lasts.
pub(crate) const LIFE: u64 = 8 * 60 * 60;

/// HMAC-SHA256 gains nothing from a shorter key.
pub(crate) const KEY_MIN: usize = 32;

pub struct Session {
    pub login: String,
    /// The GitHub user id. A login can change owner.
    pub id: u64,
    /// Each form of this session carries this token.
    pub csrf: String,
    pub expires: u64,
}

fn mac(key: &[u8], text: &str) -> Hmac<Sha256> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(text.as_bytes());
    mac
}

/// The cookie value for a session.
pub(crate) fn seal(key: &[u8], session: &Session) -> String {
    let Session {
        login,
        id,
        csrf,
        expires,
    } = session;
    let text = URL_SAFE_NO_PAD.encode(format!("{login}\n{id}\n{csrf}\n{expires}"));
    let tag = URL_SAFE_NO_PAD.encode(mac(key, &text).finalize().into_bytes());
    format!("{text}.{tag}")
}

/// The session in a cookie value, when signed and current.
pub fn open(key: &[u8], value: &str, now: u64) -> Option<Session> {
    let (text, tag) = value.split_once('.')?;
    let tag = URL_SAFE_NO_PAD.decode(tag).ok()?;
    mac(key, text).verify_slice(&tag).ok()?;
    let text = String::from_utf8(URL_SAFE_NO_PAD.decode(text).ok()?).ok()?;
    let mut parts = text.split('\n');
    let session = Session {
        login: parts.next()?.into(),
        id: parts.next()?.parse().ok()?,
        csrf: parts.next()?.into(),
        expires: parts.next()?.parse().ok()?,
    };
    (session.expires > now).then_some(session)
}

/// A random value as text.
pub fn token(nonce: [u8; 16]) -> String {
    URL_SAFE_NO_PAD.encode(nonce)
}

/// Compares in time that depends on length alone.
pub fn same(a: &str, b: &str) -> bool {
    let mut differ = a.len() ^ b.len();
    for (x, y) in a.bytes().zip(b.bytes()) {
        differ |= usize::from(x ^ y);
    }
    differ == 0
}

/// One cookie's value from a `Cookie` header.
pub fn cookie<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    for pair in header.split(';') {
        if let Some((key, value)) = pair.trim().split_once('=')
            && key == name
        {
            return Some(value);
        }
    }
    None
}

/// A `Set-Cookie` value. A life of zero clears the cookie.
pub(crate) fn bake(name: &str, value: &str, life: u64) -> String {
    format!("{name}={value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age={life}")
}
