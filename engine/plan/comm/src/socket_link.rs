//! A `Link` over TCP: puts as framed writes.

use crate::Error;
use crate::link::{Link, Put, Span, fits};
use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A frame: region, offset, length as u64s, then bytes.
const FRAME: usize = 24;
/// How long `reset` waits for sends to drain.
const DRAIN: Duration = Duration::from_secs(10);

/// One peer's stream, bytes still to write, and bytes read.
struct Peer {
    stream: TcpStream,
    out: VecDeque<u8>,
    /// Tags, by bytes written when each is done.
    tags: VecDeque<(u64, u64)>,
    written: u64,
    queued: u64,
    input: Vec<u8>,
    /// The peer closed its end. Frames it sent still land.
    closed: bool,
}

/// Regions this rank owns, written by peers through `poll`.
pub(crate) struct SocketLink {
    rank: u32,
    peers: Vec<Option<Peer>>,
    /// Each region as u64 words, so flags are aligned.
    regions: Vec<Vec<u64>>,
    sizes: Vec<usize>,
    done: Vec<u64>,
}

fn io(e: std::io::Error) -> Error {
    Error::Io(e)
}

impl SocketLink {
    /// Rank `rank` over `peers`, one stream per other rank.
    pub(crate) fn over(
        rank: u32,
        peers: Vec<Option<TcpStream>>,
        regions: &[usize],
    ) -> Result<Self, Error> {
        let missing = |(r, p): (usize, &Option<TcpStream>)| p.is_none() != (r == rank as usize);
        if rank as usize >= peers.len() || peers.iter().enumerate().any(missing) {
            return Err(Error::Fit(format!(
                "rank {rank} of {} streams",
                peers.len()
            )));
        }
        let mut wired = Vec::with_capacity(peers.len());
        for stream in peers {
            wired.push(match stream {
                None => None,
                Some(stream) => {
                    stream.set_nonblocking(true).map_err(io)?;
                    stream.set_nodelay(true).map_err(io)?;
                    Some(Peer {
                        stream,
                        out: VecDeque::new(),
                        tags: VecDeque::new(),
                        written: 0,
                        queued: 0,
                        input: Vec::new(),
                        closed: false,
                    })
                }
            });
        }
        Ok(SocketLink {
            rank,
            peers: wired,
            regions: regions.iter().map(|&n| vec![0u64; n.div_ceil(8)]).collect(),
            sizes: regions.to_vec(),
            done: Vec::new(),
        })
    }

    fn bytes(&mut self, region: usize) -> &mut [u8] {
        let words = &mut self.regions[region];
        let len = words.len() * 8;
        // Plain bytes over owned u64 words.
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, len) }
    }

    fn peer(&mut self, to: u32) -> Result<&mut Peer, Error> {
        let peer = self.peers.get_mut(to as usize).and_then(Option::as_mut);
        peer.ok_or_else(|| Error::Fit(format!("no stream to rank {to}")))
    }

    /// Queues one frame: `bytes` for `at` at rank `to`.
    fn frame(&mut self, to: u32, at: Span, bytes: &[u8]) -> Result<(), Error> {
        let peer = self.peer(to)?;
        for word in [at.region as u64, at.at as u64, bytes.len() as u64] {
            peer.out.extend(word.to_le_bytes());
        }
        peer.out.extend(bytes);
        peer.queued += (FRAME + bytes.len()) as u64;
        Ok(())
    }

    fn finish(&mut self, to: u32, tag: u64) -> Result<(), Error> {
        let peer = self.peer(to)?;
        let at = peer.queued;
        peer.tags.push_back((at, tag));
        Ok(())
    }

    /// Writes what each stream takes now, and notes finished tags.
    fn flush(&mut self) -> Result<(), Error> {
        for peer in self.peers.iter_mut().flatten() {
            while !peer.out.is_empty() {
                let (head, _) = peer.out.as_slices();
                match peer.stream.write(head) {
                    Ok(0) => return Err(Error::Link("a peer closed its stream".into())),
                    Ok(n) => {
                        peer.out.drain(..n);
                        peer.written += n as u64;
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) => return Err(io(e)),
                }
            }
            while peer.tags.front().is_some_and(|&(at, _)| at <= peer.written) {
                let (_, tag) = peer.tags.pop_front().expect("a tag");
                self.done.push(tag);
            }
        }
        Ok(())
    }

    /// Reads what each stream holds, and applies whole frames.
    fn receive(&mut self) -> Result<(), Error> {
        let mut frames = Vec::new();
        for peer in self.peers.iter_mut().flatten() {
            let mut chunk = [0u8; 1 << 16];
            while !peer.closed {
                match peer.stream.read(&mut chunk) {
                    Ok(0) => peer.closed = true,
                    Ok(n) => peer.input.extend_from_slice(&chunk[..n]),
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) => return Err(io(e)),
                }
            }
            let mut at = 0;
            while peer.input.len() - at >= FRAME {
                let word = |i: usize| {
                    let start = at + 8 * i;
                    u64::from_le_bytes(peer.input[start..start + 8].try_into().expect("8 bytes"))
                };
                let (region, offset, len) = (word(0) as usize, word(1) as usize, word(2) as usize);
                if peer.input.len() - at - FRAME < len {
                    break;
                }
                let body = peer.input[at + FRAME..at + FRAME + len].to_vec();
                frames.push((Span::new(region, offset, len), body));
                at += FRAME + len;
            }
            peer.input.drain(..at);
        }
        for (span, body) in frames {
            self.apply(span, &body)?;
        }
        Ok(())
    }

    /// One frame. Whole aligned words land as release stores.
    fn apply(&mut self, at: Span, body: &[u8]) -> Result<(), Error> {
        fits(&self.sizes, at, 1)?;
        if body.len() == 8 && at.at.is_multiple_of(8) {
            let value = u64::from_le_bytes(body.try_into().expect("8 bytes"));
            let word = &mut self.regions[at.region][at.at / 8];
            // An aligned word other threads may read.
            unsafe { AtomicU64::from_ptr(word as *mut u64) }.store(value, Ordering::Release);
            return Ok(());
        }
        self.bytes(at.region)[at.at..at.at + body.len()].copy_from_slice(body);
        Ok(())
    }
}

impl Link for SocketLink {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.peers.len() as u32
    }

    fn put(&mut self, put: Put) -> Result<(), Error> {
        if put.from.len != put.into.len || put.flag.len != 16 || !put.flag.at.is_multiple_of(8) {
            return Err(Error::Fit(format!("{put:?} has spans that differ")));
        }
        let bytes = self.read(put.from)?;
        self.frame(put.to, put.into, &bytes)?;
        let count = Span::new(put.flag.region, put.flag.at + 8, 8);
        self.frame(put.to, count, &put.words[1].to_le_bytes())?;
        let ordinal = Span::new(put.flag.region, put.flag.at, 8);
        self.frame(put.to, ordinal, &put.words[0].to_le_bytes())?;
        self.finish(put.to, put.tag)?;
        self.flush()
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        let at = Span::new(at.region, at.at, 8);
        self.frame(to, at, &value.to_le_bytes())?;
        self.finish(to, tag)?;
        self.flush()
    }

    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        self.flush()?;
        self.receive()?;
        done.append(&mut self.done);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), Error> {
        let start = Instant::now();
        while self.peers.iter().flatten().any(|p| !p.out.is_empty()) {
            if start.elapsed() > DRAIN {
                return Err(Error::Link("sends did not drain".into()));
            }
            self.flush()?;
            std::thread::yield_now();
        }
        self.done.clear();
        Ok(())
    }

    fn memory(&self, region: usize) -> Option<(*mut u8, usize)> {
        let words = self.regions.get(region)?;
        Some((words.as_ptr() as *mut u8, self.sizes[region]))
    }

    fn load(&self, at: Span) -> Result<u64, Error> {
        fits(&self.sizes, Span::new(at.region, at.at, 8), 8)?;
        let word = &self.regions[at.region][at.at / 8] as *const u64 as *mut u64;
        // An aligned word `apply` stores atomically.
        Ok(unsafe { AtomicU64::from_ptr(word) }.load(Ordering::Acquire))
    }

    fn read(&self, at: Span) -> Result<Vec<u8>, Error> {
        fits(&self.sizes, at, 1)?;
        let words = &self.regions[at.region];
        let bytes =
            unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 8) };
        Ok(bytes[at.at..at.at + at.len].to_vec())
    }

    fn write(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error> {
        fits(&self.sizes, Span::new(at.region, at.at, bytes.len()), 1)?;
        self.bytes(at.region)[at.at..at.at + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}
