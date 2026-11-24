//! Loopback TCP helpers with bounded timeouts for tests.

use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

/// Timeout for loopback TCP handshakes and test streams.
pub const TCP_TIMEOUT: Duration = Duration::from_secs(5);

/// Binds a free loopback port.
pub fn tcp_bind() -> io::Result<TcpListener> {
    TcpListener::bind("127.0.0.1:0")
}

/// Applies read and write timeouts to a test stream.
pub fn tcp_timeouts(stream: &TcpStream, wait: Duration) -> io::Result<()> {
    stream.set_read_timeout(Some(wait))?;
    stream.set_write_timeout(Some(wait))
}

/// Connects to `addr` with bounded connect and stream timeouts.
pub fn tcp_connect(addr: SocketAddr) -> io::Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&addr, TCP_TIMEOUT)?;
    stream.set_nodelay(true)?;
    tcp_timeouts(&stream, TCP_TIMEOUT)?;
    Ok(stream)
}

/// Accepts one stream on `listener` before `TCP_TIMEOUT`.
fn tcp_accept(listener: &TcpListener) -> io::Result<TcpStream> {
    listener.set_nonblocking(true)?;
    let deadline = Instant::now() + TCP_TIMEOUT;
    let accepted = loop {
        match listener.accept() {
            Ok((stream, _)) => break Ok(stream),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break Err(ErrorKind::TimedOut.into());
                }
                thread::sleep(Duration::from_millis(2).min(left));
            }
            Err(e) => break Err(e),
        }
    };
    let _ = listener.set_nonblocking(false);
    let stream = accepted?;
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    tcp_timeouts(&stream, TCP_TIMEOUT)?;
    Ok(stream)
}

/// Both ends of one loopback stream, with bounded timeouts.
pub fn tcp_pair() -> io::Result<(TcpStream, TcpStream)> {
    let listener = tcp_bind()?;
    let addr = listener.local_addr()?;
    let caller = tcp_connect(addr)?;
    let answer = tcp_accept(&listener)?;
    Ok((caller, answer))
}
