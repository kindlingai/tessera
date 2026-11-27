//! One served model over several ranks.

use std::net::{SocketAddr, TcpListener};
use tessera_engine::DType;
use tessera_engine::comm::{Error as CommError, Fabric, Transport};
use tessera_engine::sums::Sum;
use tessera_engine::{Engine, Graph, Grid, Knobs, Place, Recipe, Settings};
use tessera_frontend_comm::{
    Sums, arx_sums, grid, join, meet, p2p, ports, resident, sockets, sum_wire_flag, sums_flag,
};

/// The flags for `Ranks::flag`.
pub const FLAGS: &str = "[--rank <n> --peers <ip:port>,... [--parts <a>,<b>,...] \
[--verbs <dev>[:<gid>],... | --ring <prev>,.../<next>,...] [--sums host|grid|arx|p2p|whole] \
[--sum-wire f32|bf16|f16|fp8] [--moe ep|tp]]";

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
    /// How sums run across ranks.
    sums: Sums,
    /// The dtype unpinned sums' parts cross as. `None`: the transport's.
    sum_wire: Option<DType>,
    /// The recipe's resident grid runs each step, sums inside.
    whole: bool,
    /// Split wide routed experts along inner width instead of experts.
    moe_tp: bool,
    /// A bound listener for this rank. `None`: bind at join.
    listener: Option<TcpListener>,
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
            "--sums" => self.sums = sums_flag(value)?,
            "--sum-wire" => self.sum_wire = Some(sum_wire_flag(value)?),
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

    /// Joins on `listener` instead of binding this rank's address.
    #[cfg(test)]
    pub(crate) fn listen(&mut self, listener: TcpListener) {
        self.listener = Some(listener);
    }

    /// One engine, with no peers.
    pub(crate) fn alone(&self) -> bool {
        self.peers.is_empty() && self.rank == 0
    }

    /// This rank runs rank 0's launches and serves nothing.
    pub(crate) fn follows(&self) -> bool {
        self.rank != 0 && !tessera_engine::sizer::counting()
    }

    /// This rank's zero-based index.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Whether routed experts split along inner width.
    pub fn moe_tp(&self) -> bool {
        self.moe_tp
    }

    /// The dtype unpinned sums' parts cross as, from `--sum-wire`.
    pub fn sum_wire(&self) -> Option<DType> {
        self.sum_wire
    }

    /// Each rank's share of split weights, from `--parts`.
    pub fn parts(&self) -> &[u32] {
        &self.parts
    }

    /// Joins the peers, if any, before a long load.
    pub(crate) fn join(&self, settings: &Settings, knobs: &Knobs) -> Result<Option<Links>, String> {
        if self.peers.is_empty() && self.rank == 0 {
            return Ok(None);
        }
        if self.rank as usize >= self.peers.len() {
            return Err(format!("--rank {} needs --peers with it", self.rank));
        }
        if tessera_engine::sizer::counting() {
            let ranks = self.peers.len() as u32;
            let dry = || {
                Box::new(DryRank {
                    rank: self.rank,
                    ranks,
                }) as Box<dyn Transport>
            };
            return Ok(Some(([dry(), dry()], Grids::None)));
        }
        let met = match &self.listener {
            Some(l) => (l.try_clone().map_err(|e| e.to_string()))
                .and_then(|l| sockets(self.rank, l, &self.peers).map_err(|e| e.to_string())),
            None => meet(self.rank, &self.peers, None).map_err(|e| e.to_string()),
        };
        let mut met: Box<dyn Transport> = Box::new(met?);
        let grids = match (self.whole, self.sums, &self.fabric) {
            (true, _, None) | (false, Sums::Grid | Sums::Arx, None) => {
                return Err("--sums grid, arx or whole needs --verbs or --ring".into());
            }
            (true, _, Some(fabric)) => resident(&mut *met, fabric, settings).map(Grids::Steps),
            (false, Sums::P2p, _) => p2p(&mut *met, settings, knobs).map(Grids::Sums),
            (false, Sums::Grid, Some(fabric)) => {
                grid(&mut *met, fabric, settings, knobs).map(Grids::Sums)
            }
            (false, Sums::Arx, Some(fabric)) => {
                arx_sums(&mut *met, fabric, settings, knobs).map(Grids::Sums)
            }
            (false, Sums::Host, _) => {
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
                let place = (place.with_grid(sums_grid))
                    .with_moe_tp(self.moe_tp)
                    .with_wire(self.sum_wire);
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

/// Offline transport used when sizing a ranked layout.
struct DryRank {
    rank: u32,
    ranks: u32,
}

impl Transport for DryRank {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.ranks
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        Ok(vec![mine.to_vec(); self.ranks as usize])
    }
}
