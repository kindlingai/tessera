//! Display-carveout memory client for `spark-attic` `atticd`.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use tessera_engine_settings::{Attic, Settings};

/// Minimum slice unit that `atticd` lends.
pub(crate) const SLICE: usize = 2 << 20;
const KEY_MAX: usize = 127;
const KEY_FILE_MAX: usize = 4096;
#[cfg(target_os = "linux")]
const REPLY_MAX: usize = 4096;
const TIMEOUT_MS: i32 = 5000;

static CONFIG: Mutex<Option<Attic>> = Mutex::new(None);

/// Sets process-wide display-carveout lender settings.
pub fn configure(settings: &Settings) {
    *CONFIG.lock().unwrap_or_else(|e| e.into_inner()) = settings.attic.clone();
}

/// Active display-carveout settings, if enabled.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn current() -> Option<Attic> {
    CONFIG.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Free 2 MiB-aligned bytes `atticd` can lend now.
pub fn available(settings: &Settings) -> u64 {
    let Some(cfg) = &settings.attic else {
        return 0;
    };
    let Ok(key) = read_key(&cfg.key_file) else {
        return 0;
    };
    let Ok(mut conn) = wire::Client::open(&cfg.socket, key, TIMEOUT_MS) else {
        return 0;
    };
    let Ok(info) = conn.info(TIMEOUT_MS) else {
        return 0;
    };
    (info.free / SLICE as u64) * SLICE as u64
}

/// Planned split between carveout prefix and device tail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
pub(crate) struct SpanPlan {
    pub total: usize,
    pub carve: usize,
    pub tail: usize,
}

/// Least common multiple of two positive alignments.
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
pub(crate) fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    let (mut x, mut y) = (a, b);
    while y != 0 {
        let r = x % y;
        x = y;
        y = r;
    }
    (a / x).saturating_mul(b)
}

/// Splits `bytes` into a carveout prefix and device tail.
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
pub(crate) fn plan_span(bytes: usize, gran: usize, avail: u64) -> SpanPlan {
    if bytes == 0 || gran == 0 {
        return SpanPlan::default();
    }
    let Some(total) = bytes.checked_add(gran - 1).map(|n| (n / gran) * gran) else {
        return SpanPlan::default();
    };
    let usable = (avail / gran as u64) * gran as u64;
    let carve = usize::try_from(usable).map_or(total, |u| u.min(total));
    SpanPlan {
        total,
        carve,
        tail: total - carve,
    }
}

/// Validates a wire key: printable ASCII without quote or backslash.
pub(crate) fn key_ok(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= KEY_MAX
        && key
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b'"' && b != b'\\')
}

/// Reads and validates a key file up to 4 KiB.
pub(crate) fn read_key(path: &Path) -> Result<String, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("key file {}: {e}", path.display()))?;
    let mut raw = Vec::new();
    file.take(KEY_FILE_MAX as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|e| format!("key file {}: {e}", path.display()))?;
    if raw.len() > KEY_FILE_MAX {
        return Err(format!("key file {} exceeds 4 KiB", path.display()));
    }
    let text = std::str::from_utf8(&raw)
        .map_err(|_| format!("key file {} is not UTF-8", path.display()))?;
    let key = text.trim_matches([' ', '\t', '\r', '\n']);
    if !key_ok(key) {
        return Err(format!("key file {} holds an invalid key", path.display()));
    }
    Ok(key.to_string())
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn trim_ws(s: &str) -> &str {
    s.trim_start_matches([' ', '\t', '\r', '\n'])
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_str(s: &str) -> Option<(String, &str)> {
    let bytes = s.as_bytes();
    if bytes.first().copied() != Some(b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = 1usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == b'"' {
            return Some((out, &s[i + 1..]));
        }
        if ch == b'\\' {
            i += 1;
            let esc = *bytes.get(i)?;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                _ => out.push(esc as char),
            }
            i += 1;
            continue;
        }
        let c = s[i..].chars().next()?;
        out.push(c);
        i += c.len_utf8();
    }
    None
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn skip_val(s: &str) -> Option<&str> {
    let s = trim_ws(s);
    if s.starts_with('"') {
        return parse_str(s).map(|(_, rest)| rest);
    }
    let end = s.find([',', '}', ' ', '\t', '\r', '\n']).unwrap_or(s.len());
    (end > 0).then_some(&s[end..])
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn json_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let mut s = trim_ws(body).strip_prefix('{')?;
    loop {
        s = trim_ws(s);
        if s.starts_with('}') {
            return None;
        }
        let (name, rest) = parse_str(s)?;
        let rest = trim_ws(rest).strip_prefix(':')?;
        let val = trim_ws(rest);
        if name == key {
            return Some(val);
        }
        let after = trim_ws(skip_val(val)?);
        s = after.strip_prefix(',')?;
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn json_delim(rest: &str) -> bool {
    matches!(trim_ws(rest).as_bytes().first(), Some(b',' | b'}'))
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn json_bool(body: &str, key: &str) -> Option<bool> {
    let val = json_value(body, key)?;
    if let Some(rest) = val.strip_prefix("true")
        && json_delim(rest)
    {
        return Some(true);
    }
    if let Some(rest) = val.strip_prefix("false")
        && json_delim(rest)
    {
        return Some(false);
    }
    None
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn json_u64(body: &str, key: &str) -> Option<u64> {
    let val = json_value(body, key)?;
    let digits = val.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || (digits > 1 && val.as_bytes()[0] == b'0') {
        return None;
    }
    if !json_delim(&val[digits..]) {
        return None;
    }
    val[..digits].parse::<u64>().ok()
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn json_str(body: &str, key: &str) -> Option<String> {
    let val = json_value(body, key)?;
    let (text, rest) = parse_str(val)?;
    json_delim(rest).then_some(text)
}

/// Capacity summary from an `atticd` `info` reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Info {
    #[allow(dead_code)]
    pub size: u64,
    pub free: u64,
    #[allow(dead_code)]
    pub largest: u64,
}

/// Outcome of an `alloc` request to `atticd`.
#[cfg_attr(
    not(all(target_os = "linux", any(feature = "cuda", test))),
    allow(dead_code)
)]
pub(crate) enum AllocReply {
    Granted { fd: i32 },
    Refused(String),
    Broken(String),
}

#[cfg(target_os = "linux")]
mod wire {
    use super::{AllocReply, Info, REPLY_MAX, SLICE, json_bool, json_str, json_u64, key_ok};
    use std::ffi::{c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::time::Instant;

    pub(crate) fn close_fd(fd: i32) {
        if fd >= 0 {
            unsafe { libc::close(fd) };
        }
    }

    fn set_cloexec(fd: i32) {
        if fd < 0 {
            return;
        }
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags >= 0 {
            unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) };
        }
    }

    fn last_err() -> std::io::Error {
        std::io::Error::last_os_error()
    }

    fn wait_fd(fd: i32, events: i16, timeout_ms: i32) -> Result<(), String> {
        let start = Instant::now();
        loop {
            let spent = start.elapsed().as_millis().min(timeout_ms as u128) as i32;
            let left = (timeout_ms - spent).max(0);
            let mut pfd = libc::pollfd {
                fd,
                events,
                revents: 0,
            };
            let rc = unsafe { libc::poll(&mut pfd, 1, left) };
            if rc < 0 {
                let err = last_err();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return Err(format!("poll failed: {err}"));
            }
            if rc == 0 {
                return Err("timed out waiting for atticd".into());
            }
            return Ok(());
        }
    }

    /// Open `SOCK_SEQPACKET` connection to `atticd`.
    pub(crate) struct Client {
        fd: i32,
        broken: bool,
        key: String,
    }

    impl Drop for Client {
        fn drop(&mut self) {
            self.close();
        }
    }

    impl Client {
        pub(crate) fn open(path: &Path, key: String, timeout_ms: i32) -> Result<Self, String> {
            if !key_ok(&key) {
                return Err("invalid attic key".into());
            }
            let raw = path.as_os_str().as_bytes();
            let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
            addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
            if raw.is_empty() || raw.len() >= addr.sun_path.len() || raw.contains(&0) {
                return Err(format!("invalid socket path {}", path.display()));
            }
            for (dst, &src) in addr.sun_path.iter_mut().zip(raw.iter()) {
                *dst = src as libc::c_char;
            }
            let mut fd = unsafe {
                libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0)
            };
            if fd < 0 && last_err().raw_os_error() == Some(libc::EINVAL) {
                fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET, 0) };
            }
            if fd < 0 {
                return Err(format!("socket failed: {}", last_err()));
            }
            set_cloexec(fd);
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
            {
                let err = last_err();
                close_fd(fd);
                return Err(format!("fcntl failed: {err}"));
            }
            let len = size_of::<libc::sockaddr_un>() as libc::socklen_t;
            let rc = unsafe { libc::connect(fd, (&raw const addr).cast(), len) };
            if rc < 0 {
                let err = last_err();
                let code = err.raw_os_error();
                if code != Some(libc::EINPROGRESS) && code != Some(libc::EAGAIN) {
                    close_fd(fd);
                    return Err(format!("connect {}: {err}", path.display()));
                }
                if let Err(why) = wait_fd(fd, libc::POLLOUT, timeout_ms) {
                    close_fd(fd);
                    return Err(why);
                }
                let mut so_err: c_int = 0;
                let mut slen = size_of::<c_int>() as libc::socklen_t;
                let got = unsafe {
                    libc::getsockopt(
                        fd,
                        libc::SOL_SOCKET,
                        libc::SO_ERROR,
                        (&raw mut so_err).cast(),
                        &mut slen,
                    )
                };
                if got < 0 || so_err != 0 {
                    let err = if got < 0 {
                        last_err()
                    } else {
                        std::io::Error::from_raw_os_error(so_err)
                    };
                    close_fd(fd);
                    return Err(format!("connect {}: {err}", path.display()));
                }
            }
            if unsafe { libc::fcntl(fd, libc::F_SETFL, flags) } < 0 {
                let err = last_err();
                close_fd(fd);
                return Err(format!("fcntl restore failed: {err}"));
            }
            Ok(Self {
                fd,
                broken: false,
                key,
            })
        }

        pub(crate) fn close(&mut self) {
            if self.fd >= 0 {
                close_fd(self.fd);
                self.fd = -1;
            }
            self.broken = false;
        }

        #[cfg_attr(
            not(any(feature = "cuda", all(test, target_os = "linux"))),
            allow(dead_code)
        )]
        pub(crate) fn is_broken(&self) -> bool {
            self.broken
        }

        fn exchange(&mut self, req: &str, timeout_ms: i32) -> Result<(String, i32), String> {
            if self.fd < 0 || self.broken {
                return Err("attic connection is closed".into());
            }
            loop {
                let sent = unsafe {
                    libc::send(self.fd, req.as_ptr().cast(), req.len(), libc::MSG_NOSIGNAL)
                };
                if sent < 0 && last_err().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                if sent != req.len() as isize {
                    self.broken = true;
                    return Err(format!("send failed: {}", last_err()));
                }
                break;
            }
            if let Err(why) = wait_fd(self.fd, libc::POLLIN, timeout_ms) {
                self.broken = true;
                return Err(why);
            }
            let mut buf = [0u8; REPLY_MAX];
            let mut ctrl = [0u64; 16];
            let mut iov = libc::iovec {
                iov_base: buf.as_mut_ptr().cast(),
                iov_len: buf.len(),
            };
            let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = ctrl.as_mut_ptr().cast::<c_void>();
            let cspace = unsafe { libc::CMSG_SPACE((size_of::<c_int>() * 4) as u32) } as usize;
            msg.msg_controllen = cspace.min(size_of_val(&ctrl)) as _;
            let n = loop {
                let rc = unsafe { libc::recvmsg(self.fd, &mut msg, libc::MSG_CMSG_CLOEXEC) };
                if rc < 0 && last_err().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                break rc;
            };
            let mut got_fd: i32 = -1;
            unsafe {
                let mut c = libc::CMSG_FIRSTHDR(&msg);
                while !c.is_null() {
                    let hdr = &*c;
                    if hdr.cmsg_level == libc::SOL_SOCKET && hdr.cmsg_type == libc::SCM_RIGHTS {
                        let base = libc::CMSG_LEN(0) as usize;
                        let clen = hdr.cmsg_len as usize;
                        if clen >= base {
                            let count = (clen - base) / size_of::<c_int>();
                            let data = libc::CMSG_DATA(c);
                            for idx in 0..count {
                                let mut f: c_int = -1;
                                std::ptr::copy_nonoverlapping(
                                    data.add(idx * size_of::<c_int>()),
                                    (&raw mut f).cast::<u8>(),
                                    size_of::<c_int>(),
                                );
                                set_cloexec(f);
                                if got_fd < 0 {
                                    got_fd = f;
                                } else {
                                    close_fd(f);
                                }
                            }
                        }
                    }
                    c = libc::CMSG_NXTHDR(&msg, c);
                }
            }
            let trunc = msg.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC);
            if n <= 0 || trunc != 0 {
                close_fd(got_fd);
                self.broken = true;
                return Err("atticd closed or truncated reply".into());
            }
            let Ok(text) = std::str::from_utf8(&buf[..n as usize]) else {
                close_fd(got_fd);
                self.broken = true;
                return Err("atticd reply is not UTF-8".into());
            };
            Ok((text.to_string(), got_fd))
        }

        pub(crate) fn info(&mut self, timeout_ms: i32) -> Result<Info, String> {
            let req = format!(r#"{{"key":"{}","op":"info"}}"#, self.key);
            let (reply, fd) = self.exchange(&req, timeout_ms)?;
            close_fd(fd);
            if json_bool(&reply, "ok") == Some(false) {
                let reason = json_str(&reply, "error").unwrap_or_else(|| "refused".into());
                return Err(format!("atticd refused: {reason}"));
            }
            if let Some(reason) = json_str(&reply, "error") {
                return Err(format!("atticd refused: {reason}"));
            }
            let (Some(size), Some(free)) = (json_u64(&reply, "size"), json_u64(&reply, "free"))
            else {
                self.broken = true;
                return Err("atticd info reply missing size or free".into());
            };
            let largest = json_u64(&reply, "largest").unwrap_or(free);
            Ok(Info {
                size,
                free,
                largest,
            })
        }

        #[cfg_attr(
            not(any(feature = "cuda", all(test, target_os = "linux"))),
            allow(dead_code)
        )]
        pub(crate) fn alloc(&mut self, bytes: usize, timeout_ms: i32) -> AllocReply {
            if bytes < SLICE || !bytes.is_multiple_of(SLICE) {
                return AllocReply::Refused("size not aligned to 2 MiB".into());
            }
            let req = format!(r#"{{"key":"{}","op":"alloc","size":{bytes}}}"#, self.key);
            let (reply, fd) = match self.exchange(&req, timeout_ms) {
                Ok(pair) => pair,
                Err(why) => return AllocReply::Broken(why),
            };
            let Some(ok) = json_bool(&reply, "ok") else {
                close_fd(fd);
                self.broken = true;
                return AllocReply::Broken("atticd alloc reply missing ok".into());
            };
            if !ok {
                close_fd(fd);
                let reason = json_str(&reply, "error").unwrap_or_else(|| "refused".into());
                return AllocReply::Refused(format!("atticd refused: {reason}"));
            }
            if fd < 0 || json_u64(&reply, "size") != Some(bytes as u64) {
                close_fd(fd);
                self.broken = true;
                return AllocReply::Broken("atticd alloc reply missing fd or size".into());
            }
            AllocReply::Granted { fd }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod wire {
    use super::{AllocReply, Info};
    use std::path::Path;

    #[cfg_attr(
        not(any(feature = "cuda", all(test, target_os = "linux"))),
        allow(dead_code)
    )]
    pub(crate) fn close_fd(_fd: i32) {}

    pub(crate) struct Client;

    impl Client {
        pub(crate) fn open(_path: &Path, _key: String, _timeout_ms: i32) -> Result<Self, String> {
            Err("spark-attic requires Linux".into())
        }

        #[cfg_attr(
            not(any(feature = "cuda", all(test, target_os = "linux"))),
            allow(dead_code)
        )]
        pub(crate) fn is_broken(&self) -> bool {
            true
        }

        pub(crate) fn info(&mut self, _timeout_ms: i32) -> Result<Info, String> {
            Err("spark-attic requires Linux".into())
        }

        #[cfg_attr(
            not(any(feature = "cuda", all(test, target_os = "linux"))),
            allow(dead_code)
        )]
        pub(crate) fn alloc(&mut self, _bytes: usize, _timeout_ms: i32) -> AllocReply {
            AllocReply::Broken("spark-attic requires Linux".into())
        }
    }
}

/// Virtual memory operations for mapping lent carveout slices.
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
pub(crate) trait Vmm {
    fn granularity(&self) -> Option<usize>;
    fn import_fd(&self, fd: i32) -> Result<u64, String>;
    fn create(&self, size: usize) -> Result<u64, String>;
    fn release(&self, handle: u64);
    fn reserve(&self, size: usize, align: usize) -> Result<u64, String>;
    fn free_va(&self, va: u64, size: usize);
    fn map(&self, va: u64, size: usize, handle: u64) -> Result<(), String>;
    fn unmap(&self, va: u64, size: usize);
    fn set_access(&self, va: u64, size: usize) -> Result<(), String>;
    fn sync(&self);
}

/// One live virtual-memory span backed by carveout and tail.
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
struct Span {
    total: usize,
    carve: usize,
    parts: Vec<usize>,
}

/// Maps one carveout descriptor plus optional tail into a span.
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
fn map_span<V: Vmm>(vmm: &V, fd: i32, plan: SpanPlan, gran: usize) -> Result<(u64, Span), String> {
    let imported = vmm.import_fd(fd);
    wire::close_fd(fd);
    let hc = imported?;
    let va = match vmm.reserve(plan.total, gran) {
        Ok(va) => va,
        Err(why) => {
            vmm.release(hc);
            return Err(why);
        }
    };
    let mut parts: Vec<usize> = Vec::new();
    let map_carve = vmm.map(va, plan.carve, hc);
    vmm.release(hc);
    if let Err(why) = map_carve {
        vmm.free_va(va, plan.total);
        return Err(why);
    }
    parts.push(plan.carve);
    let mut off = 0usize;
    while off < plan.tail {
        let mut chunk = plan.tail - off;
        let ht = loop {
            match vmm.create(chunk) {
                Ok(h) => break Ok(h),
                Err(_) if chunk > gran => {
                    chunk = ((chunk / 2) / gran).max(1) * gran;
                }
                Err(why) => break Err(why),
            }
        };
        let ht = match ht {
            Ok(h) => h,
            Err(why) => {
                unmap_parts(vmm, va, &parts, plan.total);
                return Err(why);
            }
        };
        let mapped = vmm.map(va + (plan.carve + off) as u64, chunk, ht);
        vmm.release(ht);
        if let Err(why) = mapped {
            unmap_parts(vmm, va, &parts, plan.total);
            return Err(why);
        }
        parts.push(chunk);
        off += chunk;
    }
    if vmm.set_access(va, plan.total).is_err() {
        let mut at = va;
        for &len in &parts {
            if let Err(why) = vmm.set_access(at, len) {
                unmap_parts(vmm, va, &parts, plan.total);
                return Err(why);
            }
            at += len as u64;
        }
    }
    Ok((
        va,
        Span {
            total: plan.total,
            carve: plan.carve,
            parts,
        },
    ))
}

#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
fn unmap_parts<V: Vmm>(vmm: &V, va: u64, parts: &[usize], total: usize) {
    let mut at = va;
    for &len in parts {
        vmm.unmap(at, len);
        at += len as u64;
    }
    vmm.free_va(va, total);
}

/// Manages one `atticd` connection and its live VMM spans.
#[derive(Default)]
#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
pub(crate) struct Session {
    conn: Option<wire::Client>,
    last_cfg: Option<Attic>,
    probed: bool,
    unavailable: bool,
    warned: bool,
    rem: u64,
    borrowed: u64,
    spans: HashMap<u64, Span>,
}

#[cfg_attr(
    not(any(feature = "cuda", all(test, target_os = "linux"))),
    allow(dead_code)
)]
impl Session {
    fn warn_once(&mut self, why: &str) {
        if !self.warned {
            self.warned = true;
            eprintln!("tessera: attic: {why}; using ordinary device memory");
        }
    }

    fn reset(&mut self) {
        self.conn = None;
        self.probed = false;
        self.unavailable = false;
        self.warned = false;
        self.rem = 0;
    }

    fn fail_open(&mut self, why: &str) {
        self.warn_once(why);
        self.unavailable = true;
        if self.spans.is_empty() {
            self.conn = None;
        }
    }

    /// Borrows carveout memory for `bytes`, or returns `None`.
    pub(crate) fn alloc<V: Vmm>(
        &mut self,
        cfg: Option<&Attic>,
        vmm: &V,
        bytes: usize,
    ) -> Option<u64> {
        let cfg = cfg?;
        if self.spans.is_empty() && self.last_cfg.as_ref() != Some(cfg) {
            self.reset();
            self.last_cfg = Some(cfg.clone());
        }
        if self.unavailable || bytes == 0 {
            return None;
        }
        let Some(gmin) = vmm.granularity().filter(|&g| g > 0) else {
            self.fail_open("CUDA VMM unavailable");
            return None;
        };
        let gran = lcm(SLICE, gmin);
        if !self.probed {
            self.probed = true;
            let key = match read_key(&cfg.key_file) {
                Ok(key) => key,
                Err(why) => {
                    self.fail_open(&why);
                    return None;
                }
            };
            let mut conn = match wire::Client::open(&cfg.socket, key, TIMEOUT_MS) {
                Ok(conn) => conn,
                Err(why) => {
                    self.fail_open(&why);
                    return None;
                }
            };
            let info = match conn.info(TIMEOUT_MS) {
                Ok(info) => info,
                Err(why) => {
                    self.fail_open(&why);
                    return None;
                }
            };
            self.rem = (info.free / gran as u64) * gran as u64;
            if self.rem < gran as u64 {
                self.fail_open("no free 2 MiB carveout slice");
                return None;
            }
            self.conn = Some(conn);
        }
        if self.rem < gran as u64 {
            return None;
        }
        let mut plan = plan_span(bytes, gran, self.rem);
        if plan.carve < gran {
            return None;
        }
        let fd = match self.lease(&mut plan, bytes, gran) {
            Ok(fd) => fd,
            Err(why) => {
                self.fail_open(&why);
                return None;
            }
        };
        self.rem = self.rem.saturating_sub(plan.carve as u64);
        match map_span(vmm, fd, plan, gran) {
            Ok((va, span)) => {
                self.borrowed += plan.carve as u64;
                self.spans.insert(va, span);
                eprintln!(
                    "tessera: attic: borrowed {} MiB display carveout (arena {} MiB)",
                    plan.carve >> 20,
                    plan.total >> 20
                );
                Some(va)
            }
            Err(why) => {
                self.fail_open(&why);
                None
            }
        }
    }

    /// Requests one slice from `atticd` and retries once after refusal.
    fn lease(&mut self, plan: &mut SpanPlan, bytes: usize, gran: usize) -> Result<i32, String> {
        let Some(conn) = self.conn.as_mut() else {
            return Err("attic connection closed".into());
        };
        match conn.alloc(plan.carve, TIMEOUT_MS) {
            AllocReply::Granted { fd } => Ok(fd),
            AllocReply::Broken(why) => Err(why),
            AllocReply::Refused(why) => {
                if conn.is_broken() {
                    return Err(why);
                }
                let info = conn.info(TIMEOUT_MS)?;
                self.rem = (info.free / gran as u64) * gran as u64;
                *plan = plan_span(bytes, gran, self.rem);
                if plan.carve < gran {
                    return Err(why);
                }
                match conn.alloc(plan.carve, TIMEOUT_MS) {
                    AllocReply::Granted { fd } => Ok(fd),
                    AllocReply::Refused(err) | AllocReply::Broken(err) => Err(err),
                }
            }
        }
    }

    /// Unmaps `ptr` if borrowed. Closes `atticd` after the last span.
    pub(crate) fn free<V: Vmm>(&mut self, vmm: &V, ptr: u64) -> bool {
        let Some(span) = self.spans.remove(&ptr) else {
            return false;
        };
        vmm.sync();
        unmap_parts(vmm, ptr, &span.parts, span.total);
        self.borrowed = self.borrowed.saturating_sub(span.carve as u64);
        if self.spans.is_empty() {
            self.reset();
        }
        true
    }

    /// Unmaps every live span and closes the connection.
    pub(crate) fn release_all<V: Vmm>(&mut self, vmm: &V) {
        if !self.spans.is_empty() {
            vmm.sync();
            for (va, span) in self.spans.drain() {
                unmap_parts(vmm, va, &span.parts, span.total);
            }
            self.borrowed = 0;
        }
        self.reset();
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn borrowed(&self) -> u64 {
        self.borrowed
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn has_conn(&self) -> bool {
        self.conn.is_some()
    }
}
