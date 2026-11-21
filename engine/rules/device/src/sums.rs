//! A plan's sums through a persistent grid and its proxy.

use crate::proxy::Op;
use crate::resident::{Error, LanePlan, Plan, PoolPlan, Resident};
use crate::{Options, flatten};
use std::time::Duration;
use tessera_engine_comm::Link;
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::Device;
use tessera_gpukit::launch::Launch;

pub use tessera_run::Sum;

const KERNEL: &str = include_str!("sum.cu");
const ENTRY: &str = "tessera_sum";
/// Floats per round. A longer sum takes several.
const ROOM: u32 = 64 << 10;
const THREADS: u32 = 256;
/// Sums need few blocks, so grids share the device.
const SHARE: u32 = 8;
/// Ranks may still load when the first sum starts.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The most bytes in one relayed put.
pub const RELAY: usize = ROOM as usize * 4;

fn plan(ranks: u32) -> Plan {
    Plan {
        pools: vec![PoolPlan {
            depth: 2,
            one_way: false,
            group: (0..ranks).collect(),
            send_bytes: RELAY as u32,
            recv_bytes: RELAY as u32,
        }],
        lanes: vec![LanePlan {
            op: Op::Sum,
            elem: crate::layout::F32,
            pool: 0,
            block: 0,
        }],
        items: Vec::new(),
        arenas: Vec::new(),
        threads: THREADS,
        share: SHARE,
    }
}

/// The regions a link must hold for `ranks`.
pub fn regions(ranks: u32) -> Vec<usize> {
    plan(ranks).regions().to_vec()
}

struct Grid<D: Device, L>(Resident<D, L>);

impl<D, L> Sum for Grid<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn sum(&mut self, from: u64, to: u64, len: usize) -> Result<(), tessera_run::Error> {
        let n = u32::try_from(len).map_err(|_| tessera_run::Error::SumTooLong(len))?;
        let [a, b] = [from as u32, (from >> 32) as u32];
        let [c, d] = [to as u32, (to >> 32) as u32];
        let launched = self.0.launch(&[n, ROOM, a, b, c, d], TIMEOUT);
        launched.map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }
}

/// Sums over `link` in a grid on `ctx`.
pub fn on<D, L>(ctx: &D, link: L) -> Result<Box<dyn Sum>, Error>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    let options = Options {
        backend: crate::resident::backend::<D>()?,
        remote_kv: false,
        marks: false,
    };
    let text = flatten("sum.cu", KERNEL, &|_| None, &options)?.text;
    let plan = plan(link.ranks());
    let resident = Resident::open(ctx, &text, ENTRY, link, &plan)?;
    Ok(Box::new(Grid(resident)))
}

/// Sums over `link` on the GPU `gpu` names.
pub fn open<L: Link + 'static>(gpu: Option<Gpu>, link: L) -> Result<Box<dyn Sum>, Error> {
    match gpu {
        #[cfg(feature = "cuda")]
        Some(Gpu::Cuda) => on(tessera_gpukit::cuda::cached_context()?, link),
        #[cfg(feature = "rocm")]
        Some(Gpu::Rocm) => on(tessera_gpukit::rocm::cached_context()?, link),
        _ => {
            drop(link);
            Err(Error::Plan(
                "persistent grids need a CUDA or ROCm GPU".into(),
            ))
        }
    }
}
