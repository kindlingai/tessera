//! One served model over several ranks.

use sensai_engine::comm::{Fabric, Transport};
use sensai_engine::{Engine, Graph, Place, Recipe, Settings};
use sensai_frontend_comm::{join, meet, ports};
use std::net::SocketAddr;

/// The flags for `Ranks::flag`.
pub const FLAGS: &str = "[--rank <n> --peers <ip:port>,... [--parts <a>,<b>,...] \
[--verbs <dev>[:<gid>],... | --ring <prev>,.../<next>,...]]";

/// A rank's sums transport, then its control transport.
pub(crate) type Links = [Box<dyn Transport>; 2];

/// This rank among its peers. No peers: one engine alone.
#[derive(Default)]
pub struct Ranks {
    rank: u32,
    /// Each rank's address, in rank order.
    peers: Vec<SocketAddr>,
    /// Each rank's share of split weights. Empty: even shares.
    parts: Vec<u32>,
    /// RDMA ports for the sums, after the ranks meet.
    fabric: Option<Fabric>,
}

impl Ranks {
    /// Takes `flag value`. `None` if either is bad.
    pub(crate) fn flag(&mut self, flag: &str, value: &str) -> Option<()> {
        match flag {
            "--rank" => self.rank = value.parse().ok()?,
            "--peers" => {
                let peers = value.split(',').map(|a| a.parse().ok());
                self.peers = peers.collect::<Option<_>>()?;
            }
            "--parts" => {
                let parts = value.split(',').map(|p| p.parse().ok());
                self.parts = parts.collect::<Option<_>>()?;
            }
            "--verbs" => self.fabric = Some(Fabric::Mesh(ports(value).ok()?)),
            "--ring" => {
                let (prev, next) = value.split_once('/')?;
                let (prev, next) = (ports(prev).ok()?, ports(next).ok()?);
                self.fabric = Some(Fabric::Ring { prev, next });
            }
            _ => return None,
        }
        Some(())
    }

    /// This rank runs rank 0's launches and serves nothing.
    pub(crate) fn follows(&self) -> bool {
        self.rank != 0
    }

    /// Joins the peers, if any, before a long load.
    pub(crate) fn join(&self, settings: &Settings) -> Result<Option<Links>, String> {
        if self.peers.is_empty() && self.rank == 0 {
            return Ok(None);
        }
        if self.rank as usize >= self.peers.len() {
            return Err(format!("--rank {} needs --peers with it", self.rank));
        }
        let met = meet(self.rank, &self.peers, None).map_err(|e| e.to_string())?;
        let links = join(Box::new(met), self.fabric.as_ref(), settings);
        links.map(Some).map_err(|e| e.to_string())
    }

    /// The engine, over `links` if the rank joined peers.
    pub(crate) fn open(
        &self,
        links: Option<Links>,
        graph: Graph,
        context: usize,
        (recipe, settings): (&Recipe, &Settings),
    ) -> Result<Engine, String> {
        let opened = match links {
            None => Engine::open(graph, context, recipe, settings),
            Some([sums, control]) => {
                let gpu = settings.gpu().is_some();
                let place = Place::mesh(self.peers.len(), self.parts.clone(), gpu);
                Engine::open_ranked(graph, context, (recipe, settings), place, sums, control)
            }
        };
        opened.map_err(|e| e.to_string())
    }
}
