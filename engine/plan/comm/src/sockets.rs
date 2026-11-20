//! TCP between every pair of ranks, over streams already connected.

use crate::{Error, Transport};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::thread;

/// A stream to every other rank.
pub struct Sockets {
    rank: u32,
    /// By rank. This rank's own entry is empty.
    peers: Vec<Option<TcpStream>>,
}

impl Sockets {
    /// Rank `rank` over `peers`, one stream per other rank.
    pub fn over(rank: u32, peers: Vec<Option<TcpStream>>) -> Result<Self, Error> {
        let missing = |(r, p): (usize, &Option<TcpStream>)| p.is_none() != (r == rank as usize);
        if (rank as usize) >= peers.len() || peers.iter().enumerate().any(missing) {
            return Err(Error::Fit(format!(
                "rank {rank} of {} streams",
                peers.len()
            )));
        }
        Ok(Self { rank, peers })
    }
}

/// A length, then the bytes.
fn send(stream: &mut TcpStream, bytes: &[u8]) -> io::Result<()> {
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)
}

fn receive(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 8];
    stream.read_exact(&mut len)?;
    let mut bytes = vec![0u8; u64::from_le_bytes(len) as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

impl Transport for Sockets {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.peers.len() as u32
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        let mut out: Vec<Vec<u8>> = vec![Vec::new(); self.peers.len()];
        out[self.rank as usize] = mine.to_vec();
        // Writers run aside, so pairs never wait on each other.
        let mut writers: Vec<TcpStream> = Vec::new();
        for peer in self.peers.iter().flatten() {
            writers.push(peer.try_clone()?);
        }
        thread::scope(|s| -> Result<(), Error> {
            let writing = s.spawn(move || -> io::Result<()> {
                for w in &mut writers {
                    send(w, mine)?;
                }
                Ok(())
            });
            for (r, peer) in self.peers.iter_mut().enumerate() {
                if let Some(peer) = peer {
                    out[r] = receive(peer)?;
                }
            }
            writing.join().expect("the writer does not panic")?;
            Ok(())
        })?;
        Ok(out)
    }
}
