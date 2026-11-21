//! One served model over several ranks.

use std::net::SocketAddr;
use tessera_engine::comm::{Fabric, Transport};
use tessera_engine::sums::Sum;
use tessera_engine::{Engine, Graph, Place, Recipe, Settings};
use tessera_frontend_comm::{grid, join, meet, ports};

/// The flags for `Ranks::flag`.
pub const FLAGS: &str = "[--rank <n> --peers <ip:port>,... [--parts <a>,<b>,...] \
[--verbs <dev>[:<gid>],... | --ring <prev>,.../<next>,...] [--sums host|grid]]";

/// Sums and control transports, and a grid for sums.
pub(crate) type Links = ([Box<dyn Transport>; 2], Option<Box<dyn Sum>>);

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
    /// The sums run in a persistent grid over the fabric.
    grid: bool,
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
            "--sums" => self.grid = value == "grid",
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

    /// One engine, with no peers.
    pub(crate) fn alone(&self) -> bool {
        self.peers.is_empty() && self.rank == 0
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
        let mut met: Box<dyn Transport> =
            Box::new(meet(self.rank, &self.peers, None).map_err(|e| e.to_string())?);
        if let (true, Some(fabric)) = (self.grid, &self.fabric) {
            let sums = grid(&mut *met, fabric, settings).map_err(|e| e.to_string())?;
            let both = join(met, None, settings).map_err(|e| e.to_string())?;
            return Ok(Some((both, Some(sums))));
        }
        let links = join(met, self.fabric.as_ref(), settings);
        links.map(|l| Some((l, None))).map_err(|e| e.to_string())
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
            Some(([sums, control], sums_grid)) => {
                let gpu = settings.gpu().is_some();
                let place = Place::mesh(self.peers.len(), self.parts.clone(), gpu);
                let place = place.with_grid(sums_grid);
                Engine::open_ranked(graph, context, (recipe, settings), place, sums, control)
            }
        };
        opened.map_err(|e| e.to_string())
    }
}
