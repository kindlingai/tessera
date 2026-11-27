//! Replays a plan on the host, each tree in lockstep.

use crate::Error;
use crate::run::{Base, Run, line};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;
use std::time::Instant;
use tessera_codegen::Precision;
use tessera_codegen::ir::{Body, Param, Ty};
use tessera_engine_settings::Memory;
use tessera_gpukit::{Limits, Platform};
use tessera_host::exec::{self, ArgVal, Grid};
use tessera_host::lockstep::THREADS;
use tessera_jit::{Buf, Ctx, Entry};
use tessera_lower::{Arg, Extent, Launch, Plan, Shape, Source};
use tessera_mem::{Buffer, Host};

pub(crate) struct Runner {
    base: Base,
    /// Each step's tree. Hand-written kernels have none here.
    bodies: Vec<Option<Body>>,
    /// Each step's tree as host code, where it compiled.
    compiled: Vec<Option<Entry>>,
    /// Compiled trees by their kernel's print, across plans.
    known: HashMap<String, Option<Entry>>,
    lanes: usize,
    precision: Precision,
    /// Whether trees compile. If not, each runs in lockstep.
    compiles: bool,
}

impl Runner {
    pub fn new(
        plan: Plan,
        lanes: usize,
        precision: Precision,
        memory: Memory,
        compiles: bool,
    ) -> Result<Self, Error> {
        let mut runner = Self {
            base: Base::new(memory, &Platform::host(Limits::host(lanes))),
            bodies: Vec::new(),
            compiled: Vec::new(),
            known: HashMap::new(),
            lanes,
            precision,
            compiles,
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    /// One launch of `step`'s tree over the host.
    fn launch(&mut self, step: usize, launch: &Launch) -> Result<(), Error> {
        let Some(body) = &self.bodies[step] else {
            return Err(Error::Handwritten(self.base.plan.steps[step].kernel.name()));
        };
        // Every distinct slot slice once, however many arguments use it.
        let mut slices: Vec<(u32, usize, usize)> = Vec::new();
        let mut args = Vec::with_capacity(launch.args.len());
        for (arg, param) in launch.args.iter().zip(&body.params) {
            args.push(match (arg, param) {
                (Arg::Buffer(slot, Extent::Fixed(skip)), Param::Buffer { form, .. }) => {
                    let key = (slot.buffer, slot.start(), self.base.held.span(*slot));
                    let at = match slices.iter().position(|s| *s == key) {
                        Some(at) => at,
                        None => {
                            slices.push(key);
                            slices.len() - 1
                        }
                    };
                    ArgVal::Buf {
                        at,
                        skip: *skip,
                        form: *form,
                    }
                }
                // The body says what each value is.
                (Arg::Value(_), Param::Scalar { ty: Ty::F32, .. }) => {
                    ArgVal::F32(arg.f32().unwrap())
                }
                (Arg::Value(_), Param::Scalar { ty: Ty::U32, .. }) => {
                    ArgVal::U32(arg.u32().unwrap())
                }
                (other, param) => {
                    let (arg, param) = (Box::new(other.clone()), Box::new(param.clone()));
                    return Err(Error::HostArg { arg, param });
                }
            });
        }
        let grid = match launch.shape {
            Shape::Line(count) => {
                let (groups, threads) = line(count);
                Grid { groups, threads }
            }
            Shape::Grid {
                across,
                down,
                threads,
            } => Grid {
                groups: across * down,
                threads,
            },
            Shape::Runtime(_) => unreachable!("bound before running"),
        };
        let mut mapped = self.base.held.mapped_mut(&slices);
        if let Some(entry) = self.compiled[step]
            && grid.threads <= THREADS
        {
            let step = &self.base.plan.steps[step];
            let work = step.kernel.work(step.write.len);
            return compiled(
                entry,
                &args,
                &mut mapped,
                grid,
                work,
                self.lanes,
                &self.precision,
            );
        }
        exec::run(body, &args, &mut mapped, grid, self.lanes, self.precision);
        Ok(())
    }
}

impl Run for Runner {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        let plan = self.base.held.reuse_weights(plan, |_, _| None);
        let plan = plan.as_ref();
        if !self.base.fresh(plan)? {
            return Ok(());
        }
        self.base.held.bind(plan, |zone, bytes, _| {
            Ok(Buffer::new(zone, Host::new(bytes)))
        })?;
        self.bodies = plan
            .steps
            .iter()
            .map(|step| match step.kernel.source() {
                Source::Made(body) => Some(body),
                Source::Fixed { .. } => None,
            })
            .collect();
        // A tree that does not compile runs in lockstep instead.
        let mut compiled = Vec::with_capacity(plan.steps.len());
        for body in &self.bodies {
            let Some(body) = body.as_ref().filter(|_| self.compiles) else {
                compiled.push(None);
                continue;
            };
            // Bodies share an entry name when their text matches.
            let key = body.entry.clone();
            let entry = match self.known.get(&key) {
                Some(entry) => *entry,
                None => {
                    let entry = tessera_jit::compile(body).ok();
                    self.known.insert(key, entry);
                    entry
                }
            };
            compiled.push(entry);
        }
        self.compiled = compiled;
        self.base.adopt(plan);
        Ok(())
    }

    fn stage_weights(&mut self, plan: &Plan) -> Result<(), Error> {
        self.base.held.stage_weights(plan, |zone, bytes, _| {
            Ok(Buffer::new(zone, Host::new(bytes)))
        })?;
        self.base.record_extra_weights(plan);
        let mut base = self.base.plan.clone();
        base.id = Plan::fresh();
        self.replan(&base)
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        for i in steps {
            let start = Instant::now();
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
            self.note_step(i, start.elapsed().as_secs_f64());
        }
        Ok(())
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        let start = Instant::now();
        for _ in 0..times {
            self.submit(step..step + 1)?;
        }
        Ok(start.elapsed().as_secs_f64() / times as f64)
    }

    fn device(&self) -> String {
        let tiles = match self.precision {
            Precision::EXACT => "",
            Precision::METAL => ", metal tiles",
            Precision::CUDA => ", cuda tiles",
            _ => ", hip tiles",
        };
        let (compiled, made) = self.compiled();
        format!(
            "cpu {} lanes{tiles}, {compiled} of {made} compiled",
            self.lanes
        )
    }

    fn compiled(&self) -> (usize, usize) {
        let made = self.bodies.iter().flatten().count();
        (self.compiled.iter().flatten().count(), made)
    }

    fn platform(&self) -> Platform {
        Platform::host(Limits::host(self.lanes))
    }

    fn precision(&self) -> Precision {
        self.precision
    }
}

/// Multiply-adds a launch is worth spreading over cores.
const SPREAD: f64 = 65536.0;

/// Workers with room for a body's lane arrays.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(32 << 20)
            .build()
            .expect("a thread pool")
    })
}

/// Group ranges over the cores, or one for small `work`.
fn ranges(groups: usize, work: f64) -> Vec<(usize, usize)> {
    let cores = match work >= SPREAD {
        true => pool().current_num_threads(),
        false => 1,
    };
    let workers = cores.min(groups).max(1);
    let per = groups.div_ceil(workers);
    (0..workers)
        .map(|w| (w * per, ((w + 1) * per).min(groups)))
        .filter(|(lo, hi)| lo < hi)
        .collect()
}

/// A context the worker threads share. Groups write disjoint rows.
struct Shared(Ctx);

// SAFETY: every field points at memory that outlives the launch.
unsafe impl Sync for Shared {}

/// One launch of `entry` over `grid`, spread over the pool.
fn compiled(
    entry: Entry,
    args: &[ArgVal],
    mapped: &mut [&mut [u8]],
    grid: Grid,
    work: f64,
    lanes: usize,
    precision: &Precision,
) -> Result<(), Error> {
    let bufs: Vec<Buf> = mapped
        .iter_mut()
        .map(|b| Buf {
            ptr: b.as_mut_ptr(),
            len: b.len(),
        })
        .collect();
    let args: Vec<tessera_jit::Arg> = args
        .iter()
        .map(|a| match *a {
            ArgVal::Buf { at, skip, .. } => tessera_jit::Arg {
                buf: at,
                skip,
                bits: 0,
            },
            ArgVal::U32(v) => tessera_jit::Arg {
                bits: v,
                ..tessera_jit::Arg::default()
            },
            ArgVal::F32(v) => tessera_jit::Arg {
                bits: v.to_bits(),
                ..tessera_jit::Arg::default()
            },
        })
        .collect();
    let shared = Shared(Ctx {
        bufs: bufs.as_ptr(),
        args: args.as_ptr(),
        threads: grid.threads,
        lanes,
        precision,
    });
    let (ranges, shared) = (ranges(grid.groups, work), &shared);
    let run = |(lo, hi): (usize, usize)| {
        // SAFETY: the entry was compiled for this body's parameters.
        unsafe { entry(&shared.0, lo, hi) }
    };
    match ranges.len() {
        1 => run(ranges[0]),
        _ => pool().scope(|scope| {
            for range in &ranges {
                scope.spawn(move |_| run(*range));
            }
        }),
    }
    match tessera_jit::failure() {
        Some(why) => Err(Error::Jit(why)),
        None => Ok(()),
    }
}
