//! Replays a plan on the host, each tree in lockstep.

use crate::Error;
use crate::abi::{Arg, Launch, Shape, Source};
use crate::plan::Plan;
use crate::run::{Held, Run};
use codegen::Precision;
use codegen::exec::{self, ArgVal, Grid};
use codegen::ir::{Body, Param};
use gpukit::Limits;
use graph::DimName;
use mem::{Buffer, Host};
use std::collections::HashMap;
use std::ops::Range;
use std::time::Instant;

pub struct Runner {
    plan: Plan,
    held: Held,
    /// Each step's tree. Hand-written kernels have none here.
    bodies: Vec<Option<Body>>,
    /// Runtime dimensions, as last bound.
    bound: HashMap<DimName, u32>,
    lanes: usize,
    precision: Precision,
}

impl Runner {
    pub fn new(plan: Plan, lanes: usize, precision: Precision) -> Result<Self, Error> {
        let mut runner = Self {
            plan: Plan::default(),
            held: Held::default(),
            bodies: Vec::new(),
            bound: HashMap::new(),
            lanes,
            precision,
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    /// Each launch of a step, its runtime sizes filled in.
    fn launches(&self, step: usize) -> Result<Vec<Launch>, Error> {
        let size = |name| self.bound.get(&name).copied();
        let bound = |l: Launch| l.bound(&size).ok_or(Error::Unbound);
        self.plan.steps[step]
            .launches()
            .into_iter()
            .map(bound)
            .collect()
    }

    /// One launch of `step`'s tree over the host.
    fn launch(&mut self, step: usize, launch: &Launch) -> Result<(), Error> {
        let Some(body) = &self.bodies[step] else {
            let what = self.plan.steps[step].kernel.name();
            return Err(Error::Unsupported(format!("{what} is hand-written")));
        };
        // Every buffer once, however many arguments point into it.
        let mut buffers: Vec<u32> = Vec::new();
        let mut args = Vec::with_capacity(launch.args.len());
        for (arg, param) in launch.args.iter().zip(&body.params) {
            args.push(match (arg, param) {
                (Arg::Buffer(slot, skip), Param::Buffer { form, .. }) => {
                    let at = match buffers.iter().position(|b| *b == slot.buffer) {
                        Some(at) => at,
                        None => {
                            buffers.push(slot.buffer);
                            buffers.len() - 1
                        }
                    };
                    let skip = form.elements(slot.start()) + skip;
                    ArgVal::Buf {
                        at,
                        skip,
                        form: *form,
                    }
                }
                (Arg::U32(v), _) => ArgVal::U32(*v),
                (Arg::F32(v), _) => ArgVal::F32(*v),
                (other, _) => {
                    return Err(Error::Unsupported(format!("{other:?} on the host")));
                }
            });
        }
        let grid = match launch.shape {
            Shape::Line(count) => {
                let threads = count.clamp(1, 256);
                Grid {
                    groups: count.div_ceil(threads),
                    threads,
                }
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
        let body = body.clone();
        let mut mapped = self.held.mapped_mut(&buffers);
        exec::run(&body, &args, &mut mapped, grid, self.lanes, self.precision);
        Ok(())
    }
}

/// A host group's limits, `lanes` to a SIMD group.
pub fn limits(lanes: usize) -> Limits {
    Limits {
        lanes,
        group_bytes: 64 * 1024,
        group_threads: 1024,
        compiler: false,
    }
}

impl Run for Runner {
    fn plan(&self) -> &Plan {
        &self.plan
    }

    fn held(&self) -> &Held {
        &self.held
    }

    fn held_mut(&mut self) -> &mut Held {
        &mut self.held
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        if plan.id != 0 && plan.id == self.plan.id {
            return Ok(());
        }
        if plan.zoned() {
            return Err(Error::Unsupported("a plan over several zones".into()));
        }
        self.held.bind(plan, |zone, bytes| {
            Ok(Buffer::new(zone, Host::new(bytes.max(4))))
        })?;
        self.bodies = plan
            .steps
            .iter()
            .map(|step| match step.kernel.source() {
                Source::Made(body) => Some(body),
                Source::Fixed { .. } => None,
            })
            .collect();
        self.plan = plan.clone();
        Ok(())
    }

    fn bind(&mut self, name: DimName, value: u32) {
        self.bound.insert(name, value);
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        for i in steps {
            for launch in self.launches(i)? {
                self.launch(i, &launch)?;
            }
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
        format!("cpu {} lanes{tiles}", self.lanes)
    }

    fn limits(&self) -> Limits {
        limits(self.lanes)
    }

    fn precision(&self) -> Precision {
        self.precision
    }
}
