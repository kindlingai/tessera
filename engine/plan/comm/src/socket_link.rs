//! A `Link` over TCP: puts as framed writes.

use crate::Error;
use crate::link::{Link, Put, Span, fits};
use crate::verbs::Region;
use std::collections::VecDeque;
use std::io::{ErrorKind, IoSlice, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A frame: region, offset, length as u64s, then bytes.
const FRAME: usize = 24;
/// A control frame's region word. Its offset says which.
const CONTROL: u64 = u64::MAX;
const MARK: u64 = 0;
const ACK: u64 = 1;
/// How long `reset` waits for each peer's ack.
const DRAIN: Duration = Duration::from_secs(10);

/// One peer's stream, bytes still to write, and bytes read.
struct Peer {
    stream: TcpStream,
    out: VecDeque<u8>,
    input: Vec<u8>,
    /// The peer closed its end. Frames it sent still land.
    closed: bool,
    /// Acks to this rank's marks.
    acks: u64,
}

/// Regions this rank owns, written by peers through `poll`.
pub(crate) struct SocketLink {
    rank: u32,
    peers: Vec<Option<Peer>>,
    regions: Vec<Region>,
    sizes: Vec<usize>,
    done: Vec<u64>,
    recv_buf: Box<[u8]>,
}

// Each rank owns its regions and streams.
unsafe impl Send for SocketLink {}

/// A reset or broken stream means the peer is gone.
fn io(e: std::io::Error) -> Error {
    match e.kind() {
        ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted | ErrorKind::BrokenPipe => {
            Error::Link(format!("a peer dropped its stream: {e}"))
        }
        _ => Error::Io(e),
    }
}

fn write_words(dst: &mut [u8], words: &[u64]) {
    for (slot, w) in dst.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *slot = w.to_le_bytes();
    }
}

fn queue(out: &mut VecDeque<u8>, words: [u64; 3], bytes: &[u8]) {
    let mut hdr = [0u8; FRAME];
    write_words(&mut hdr, &words);
    out.reserve(FRAME + bytes.len());
    out.extend(hdr);
    out.extend(bytes);
}

/// Appends `slices` past `skip` bytes to `out`.
fn queue_rest(out: &mut VecDeque<u8>, slices: &[&[u8]], mut skip: usize) {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    if skip >= total {
        return;
    }
    out.reserve(total - skip);
    for &s in slices {
        if skip >= s.len() {
            skip -= s.len();
        } else {
            out.extend(&s[skip..]);
            skip = 0;
        }
    }
}

/// Applies full frames from `buf` and counts marks.
fn parse_frames(
    rank: usize,
    buf: &[u8],
    regions: &[Region],
    sizes: &[usize],
    peer_acks: &mut u64,
) -> Result<(usize, usize), Error> {
    let mut at = 0;
    let mut marks = 0;
    while buf.len() - at >= FRAME {
        let word = |i: usize| {
            let start = at + 8 * i;
            u64::from_le_bytes(buf[start..start + 8].try_into().expect("8 bytes"))
        };
        let (region, offset, len) = (word(0), word(1), word(2));
        if region == CONTROL {
            match offset {
                MARK => marks += 1,
                ACK => *peer_acks += 1,
                _ => return Err(Error::Link(format!("rank {rank} sent control {offset}"))),
            }
            at += FRAME;
            continue;
        }
        let span = Span::new(region as usize, offset as usize, len as usize);
        fits(sizes, span, 1).map_err(|e| Error::Link(format!("rank {rank}: {e}")))?;
        if buf.len() - at - FRAME < span.len {
            break;
        }
        apply(&regions[span.region], span, &buf[at + FRAME..][..span.len]);
        at += FRAME + span.len;
    }
    Ok((at, marks))
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
                        input: Vec::new(),
                        closed: false,
                        acks: 0,
                    })
                }
            });
        }
        let mut owned = Vec::with_capacity(regions.len());
        for &len in regions {
            owned.push(Region::new(len, None)?);
        }
        Ok(SocketLink {
            rank,
            peers: wired,
            regions: owned,
            sizes: regions.to_vec(),
            done: Vec::new(),
            recv_buf: vec![0u8; 1 << 16].into_boxed_slice(),
        })
    }

    fn peer(&mut self, to: u32) -> Result<&mut Peer, Error> {
        let peer = self.peers.get_mut(to as usize).and_then(Option::as_mut);
        peer.ok_or_else(|| Error::Fit(format!("no stream to rank {to}")))
    }

    /// Queues one frame: `bytes` for `at` at rank `to`.
    fn frame(&mut self, to: u32, at: Span, bytes: &[u8]) -> Result<(), Error> {
        let words = [at.region as u64, at.at as u64, bytes.len() as u64];
        queue(&mut self.peer(to)?.out, words, bytes);
        Ok(())
    }

    /// Writes what each stream accepts now.
    fn flush(&mut self) -> Result<(), Error> {
        for peer in self.peers.iter_mut().flatten() {
            while !peer.out.is_empty() {
                let (head, tail) = peer.out.as_slices();
                let written = if tail.is_empty() {
                    peer.stream.write(head)
                } else {
                    peer.stream
                        .write_vectored(&[IoSlice::new(head), IoSlice::new(tail)])
                };
                match written {
                    Ok(0) => return Err(Error::Link("a peer closed its stream".into())),
                    Ok(n) => drop(peer.out.drain(..n)),
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) => return Err(io(e)),
                }
            }
        }
        Ok(())
    }

    /// Reads each stream, and applies whole frames in order.
    fn receive(&mut self) -> Result<(), Error> {
        let (regions, sizes, chunk) = (&self.regions, &self.sizes, &mut self.recv_buf[..]);
        for (r, peer) in self.peers.iter_mut().enumerate() {
            let Some(peer) = peer else { continue };
            let mut marks = 0;
            while !peer.closed {
                match peer.stream.read(chunk) {
                    Ok(0) => peer.closed = true,
                    Ok(n) if peer.input.is_empty() => {
                        let (at, m) = parse_frames(r, &chunk[..n], regions, sizes, &mut peer.acks)?;
                        marks += m;
                        if at < n {
                            peer.input.extend_from_slice(&chunk[at..n]);
                        }
                    }
                    Ok(n) => {
                        peer.input.extend_from_slice(&chunk[..n]);
                        let (at, m) = parse_frames(r, &peer.input, regions, sizes, &mut peer.acks)?;
                        marks += m;
                        if at == peer.input.len() {
                            peer.input.clear();
                        } else if at > 0 {
                            peer.input.drain(..at);
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) => return Err(io(e)),
                }
            }
            for _ in 0..marks {
                queue(&mut peer.out, [CONTROL, ACK, 0], &[]);
            }
            if peer.closed && !peer.input.is_empty() {
                return Err(Error::Link(format!("rank {r} closed in a frame")));
            }
        }
        Ok(())
    }
}

/// One frame. Whole aligned words land as release stores.
fn apply(region: &Region, at: Span, body: &[u8]) {
    let to = region.at(at.at);
    if body.len() == 8 && at.at.is_multiple_of(8) {
        let value = u64::from_le_bytes(body.try_into().expect("8 bytes"));
        // An aligned word other threads may read.
        unsafe { AtomicU64::from_ptr(to as *mut u64) }.store(value, Ordering::Release);
        return;
    }
    unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), to, body.len()) };
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
        fits(&self.sizes, put.from, 1)?;
        let from = self.regions[put.from.region].at(put.from.at);
        let bytes = unsafe { std::slice::from_raw_parts(from, put.from.len) };
        let mut hdr = [0u8; FRAME];
        write_words(
            &mut hdr,
            &[
                put.into.region as u64,
                put.into.at as u64,
                bytes.len() as u64,
            ],
        );
        let mut flags = [0u8; 64];
        write_words(
            &mut flags,
            &[
                put.flag.region as u64,
                (put.flag.at + 8) as u64,
                8,
                put.words[1],
                put.flag.region as u64,
                put.flag.at as u64,
                8,
                put.words[0],
            ],
        );
        let peer = self
            .peers
            .get_mut(put.to as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Fit(format!("no stream to rank {}", put.to)))?;
        let slices: [&[u8]; 3] = [&hdr, bytes, &flags];
        if peer.out.is_empty() {
            let io_slices = [
                IoSlice::new(&hdr),
                IoSlice::new(bytes),
                IoSlice::new(&flags),
            ];
            match peer.stream.write_vectored(&io_slices) {
                Ok(0) => return Err(Error::Link("a peer closed its stream".into())),
                Ok(n) => queue_rest(&mut peer.out, &slices, n),
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    queue_rest(&mut peer.out, &slices, 0)
                }
                Err(e) => return Err(io(e)),
            }
        } else {
            queue_rest(&mut peer.out, &slices, 0);
        }
        self.done.push(put.tag);
        self.flush()
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        let at = Span::new(at.region, at.at, 8);
        self.frame(to, at, &value.to_le_bytes())?;
        self.done.push(tag);
        self.flush()
    }

    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        self.flush()?;
        self.receive()?;
        done.append(&mut self.done);
        Ok(())
    }

    /// Marks each stream, then waits for every ack.
    fn reset(&mut self) -> Result<(), Error> {
        let mut want = Vec::with_capacity(self.peers.len());
        for peer in self.peers.iter_mut() {
            want.push(peer.as_mut().map(|p| {
                queue(&mut p.out, [CONTROL, MARK, 0], &[]);
                p.acks + 1
            }));
        }
        let start = Instant::now();
        let acked =
            |p: &Option<Peer>, w: &Option<u64>| p.as_ref().is_none_or(|p| Some(p.acks) >= *w);
        while !self.peers.iter().zip(&want).all(|(p, w)| acked(p, w)) {
            if start.elapsed() > DRAIN {
                return Err(Error::Link("a peer did not ack the reset".into()));
            }
            self.flush()?;
            self.receive()?;
            for (p, w) in self.peers.iter().zip(&want) {
                if let (Some(p), Some(w)) = (p, w)
                    && p.closed
                    && p.acks < *w
                {
                    return Err(Error::Link("a peer closed before acking reset".into()));
                }
            }
            std::thread::yield_now();
        }
        self.flush()?;
        self.done.clear();
        Ok(())
    }

    fn memory(&self, region: usize) -> Option<(*mut u8, Option<u64>, usize)> {
        let region = self.regions.get(region)?;
        Some((region.base, None, region.len))
    }

    fn load(&self, at: Span) -> Result<u64, Error> {
        fits(&self.sizes, Span::new(at.region, at.at, 8), 8)?;
        let word = self.regions[at.region].at(at.at) as *mut u64;
        // An aligned word `apply` stores atomically.
        Ok(unsafe { AtomicU64::from_ptr(word) }.load(Ordering::Acquire))
    }

    fn read(&self, at: Span) -> Result<Vec<u8>, Error> {
        fits(&self.sizes, at, 1)?;
        let from = self.regions[at.region].at(at.at);
        Ok(unsafe { std::slice::from_raw_parts(from, at.len) }.to_vec())
    }

    fn write(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error> {
        let at = Span::new(at.region, at.at, bytes.len());
        fits(&self.sizes, at, 1)?;
        apply(&self.regions[at.region], at, bytes);
        Ok(())
    }
}
