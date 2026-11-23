//! One served model over several ranks.

use std::net::SocketAddr;
use tessera_engine::comm::{Fabric, Transport};
use tessera_engine::sums::Sum;
use tessera_engine::{Engine, Graph, Grid, Place, Recipe, Settings};
use tessera_frontend_comm::{grid, grid_flag, join, meet, ports, resident};

/// The flags for `Ranks::flag`.
pub const FLAGS: &str = "[--rank <n> --peers <ip:port>,... [--parts <a>,<b>,...] \
[--verbs <dev>[:<gid>],... | --ring <prev>,.../<next>,...] [--sums host|grid|whole] [--moe ep|tp]]";

/// Sums and control transports, and any grid.
pub(crate) type Links = ([Box<dyn Transport>; 2], Grids);

/// A grid for the sums, or for resident steps.
pub(crate) enum Grids {
    None,
    Sums(Box<dyn Sum>),
    Steps(Box<dyn Grid>),
}

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
    /// The recipe's resident grid runs each step, sums inside.
    whole: bool,
    /// Split wide routed experts along inner width instead of experts.
    moe_tp: bool,
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
            "--sums" if value == "whole" => self.whole = true,
            "--sums" => self.grid = grid_flag(value)?,
            "--moe" => match value {
                "ep" => self.moe_tp = false,
                "tp" => self.moe_tp = true,
                _ => return None,
            },
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

    /// Whether routed experts split along inner width.
    pub(crate) fn moe_tp(&self) -> bool {
        self.moe_tp
    }

    /// Each rank's share of split weights, from `--parts`.
    pub(crate) fn parts(&self) -> &[u32] {
        &self.parts
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
        let grids = match (self.grid || self.whole, &self.fabric) {
            (true, None) => return Err("--sums grid or whole needs --verbs or --ring".into()),
            (true, Some(fabric)) if self.whole => {
                resident(&mut *met, fabric, settings).map(Grids::Steps)
            }
            (true, Some(fabric)) => grid(&mut *met, fabric, settings).map(Grids::Sums),
            (false, _) => {
                let links = join(met, self.fabric.as_ref());
                return links
                    .map(|l| Some((l, Grids::None)))
                    .map_err(|e| e.to_string());
            }
        };
        let grids = grids.map_err(|e| e.to_string())?;
        let both = join(met, None).map_err(|e| e.to_string())?;
        Ok(Some((both, grids)))
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
            Some(([sums, control], grids)) => {
                let gpu = settings.gpu().is_some();
                let place = Place::mesh(self.peers.len(), self.parts.clone(), gpu);
                let (sums_grid, steps) = match grids {
                    Grids::None => (None, None),
                    Grids::Sums(s) => (Some(s), None),
                    Grids::Steps(g) => (None, Some(g)),
                };
                let place = place.with_grid(sums_grid).with_moe_tp(self.moe_tp);
                let opened =
                    Engine::open_ranked(graph, context, (recipe, settings), place, sums, control);
                opened.and_then(|mut e| {
                    steps.map(|g| e.persist(g)).transpose()?;
                    Ok(e)
                })
            }
        };
        opened.map_err(|e| e.to_string())
    }
}
