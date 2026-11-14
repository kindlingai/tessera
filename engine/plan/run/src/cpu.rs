//! Replays a plan on the host, each tree in lockstep.

use crate::run::{Base, Run, line};
use codegen::Precision;
use codegen::exec::{self, ArgVal, Grid};
use codegen::ir::{Body, Param, Ty};
use gpukit::{Limits, Platform};
use lower::{Arg, Error, Extent, Launch, Plan, Shape, Source};
use mem::{Buffer, Host};
use std::ops::Range;
use std::time::Instant;

pub(crate) struct Runner {
    base: Base,
    /// Each step's tree. Hand-written kernels have none here.
    bodies: Vec<Option<Body>>,
    lanes: usize,
    precision: Precision,
}

impl Runner {
    pub fn new(plan: Plan, lanes: usize, precision: Precision) -> Result<Self, Error> {
        let mut runner = Self {
            base: Base::default(),
            bodies: Vec::new(),
            lanes,
            precision,
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    /// One launch of `step`'s tree over the host.
    fn launch(&mut self, step: usize, launch: &Launch) -> Result<(), Error> {
        let Some(body) = &self.bodies[step] else {
            let what = self.base.plan.steps[step].kernel.name();
            return Err(Error::Unsupported(format!("{what} is hand-written")));
        };
        // Every buffer once, however many arguments point into it.
        let mut buffers: Vec<u32> = Vec::new();
        let mut args = Vec::with_capacity(launch.args.len());
        for (arg, param) in launch.args.iter().zip(&body.params) {
            args.push(match (arg, param) {
                (Arg::Buffer(slot, Extent::Fixed(skip)), Param::Buffer { form, .. }) => {
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
                // The body says what each value is.
                (Arg::Value(_), Param::Scalar { ty: Ty::F32, .. }) => {
                    ArgVal::F32(arg.f32().unwrap())
                }
                (Arg::Value(_), Param::Scalar { ty: Ty::U32, .. }) => {
                    ArgVal::U32(arg.u32().unwrap())
                }
                (other, param) => {
                    let what = format!("{other:?} as {param:?} on the host");
                    return Err(Error::Unsupported(what));
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
        let body = body.clone();
        let mut mapped = self.base.held.mapped_mut(&buffers);
        exec::run(&body, &args, &mut mapped, grid, self.lanes, self.precision);
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
        if !self.base.fresh(plan)? {
            return Ok(());
        }
        self.base.held.bind(plan, |zone, bytes| {
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
        self.base.plan = plan.clone();
        Ok(())
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        for i in steps {
            for launch in self.base.launches(i)? {
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

    fn platform(&self) -> Platform {
        Platform::host(Limits::host(self.lanes))
    }

    fn precision(&self) -> Precision {
        self.precision
    }
}
