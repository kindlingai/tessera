//! Recorded Metal dispatch graphs with pre-resolved buffers.

use crate::Error;
use crate::launch::{Bound, KernelArgs};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLSize};
use std::ops::Range;
use std::sync::Arc;

pub(super) type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(super) type Pipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A pre-resolved buffer binding for one kernel parameter slot.
pub(super) struct BufArg {
    pub(super) buf: Buf,
    pub(super) offset: usize,
    pub(super) index: u16,
}

/// An inline constant slice inside [`GraphData::bytes`].
pub(super) struct ByteArg {
    pub(super) start: u32,
    pub(super) len: u16,
    pub(super) index: u16,
}

/// One recorded command inside a Metal graph.
pub(super) enum Op {
    Concurrent(bool),
    Barrier,
    Dispatch {
        pipeline: Pipe,
        grid: MTLSize,
        block: MTLSize,
        shared: u32,
        bufs: Range<u32>,
        bytes: Range<u32>,
    },
}

/// Immutable recorded Metal launches shared with command buffers.
pub(super) struct GraphData {
    pub(super) owner: usize,
    pub(super) initial_concurrent: bool,
    pub(super) last_dispatch: Option<usize>,
    pub(super) ops: Vec<Op>,
    pub(super) bufs: Vec<BufArg>,
    pub(super) bytes: Vec<u8>,
    pub(super) byte_args: Vec<ByteArg>,
}

// SAFETY: Metal buffers and pipelines are immutable and thread-safe.
unsafe impl Send for GraphData {}
unsafe impl Sync for GraphData {}

/// A captured sequence of Metal dispatches ready for fast replay.
#[derive(Clone)]
pub struct Graph(pub(super) Arc<GraphData>);

/// In-progress Metal graph capture on one thread.
pub(super) struct Recording {
    data: GraphData,
    concurrent: bool,
}

impl Recording {
    pub(super) fn new(owner: usize, initial_concurrent: bool) -> Self {
        Self {
            data: GraphData {
                owner,
                initial_concurrent,
                last_dispatch: None,
                ops: Vec::new(),
                bufs: Vec::new(),
                bytes: Vec::new(),
                byte_args: Vec::new(),
            },
            concurrent: initial_concurrent,
        }
    }

    pub(super) fn owner(&self) -> usize {
        self.data.owner
    }

    pub(super) fn set_concurrent(&mut self, on: bool) {
        if self.concurrent != on {
            if !on {
                self.barrier();
            }
            self.concurrent = on;
            self.data.ops.push(Op::Concurrent(on));
        }
    }

    pub(super) fn barrier(&mut self) {
        if self.concurrent && !matches!(self.data.ops.last(), Some(Op::Barrier)) {
            self.data.ops.push(Op::Barrier);
        }
    }

    pub(super) fn dispatch(
        &mut self,
        pipeline: Pipe,
        grid: MTLSize,
        block: MTLSize,
        shared: u32,
        args: &KernelArgs,
        mut resolve: impl FnMut(u64) -> Result<(Buf, usize), Error>,
    ) -> Result<(), Error> {
        let bufs_len = self.data.bufs.len();
        let bytes_len = self.data.bytes.len();
        let byte_args_len = self.data.byte_args.len();
        let res = (|| -> Result<(Range<u32>, Range<u32>), Error> {
            let bufs_start = u32::try_from(bufs_len)
                .map_err(|_| Error::Metal("graph buffer binding overflow".into()))?;
            let bytes_start = u32::try_from(byte_args_len)
                .map_err(|_| Error::Metal("graph byte binding overflow".into()))?;
            for (index, arg) in args.bound().enumerate() {
                let index = u16::try_from(index)
                    .map_err(|_| Error::Metal("kernel argument index overflow".into()))?;
                match arg {
                    Bound::Pointer(ptr) => {
                        let (buf, offset) = resolve(ptr)?;
                        self.data.bufs.push(BufArg { buf, offset, index });
                    }
                    Bound::Bytes(slice) => {
                        let start = u32::try_from(self.data.bytes.len())
                            .map_err(|_| Error::Metal("graph inline bytes overflow".into()))?;
                        let len = u16::try_from(slice.len())
                            .map_err(|_| Error::Metal("kernel byte arg exceeds 64 KiB".into()))?;
                        self.data.bytes.extend_from_slice(slice);
                        self.data.byte_args.push(ByteArg { start, len, index });
                    }
                }
            }
            let bufs_end = u32::try_from(self.data.bufs.len())
                .map_err(|_| Error::Metal("graph buffer binding overflow".into()))?;
            let bytes_end = u32::try_from(self.data.byte_args.len())
                .map_err(|_| Error::Metal("graph byte binding overflow".into()))?;
            Ok((bufs_start..bufs_end, bytes_start..bytes_end))
        })();
        let (bufs, bytes) = match res {
            Ok(ranges) => ranges,
            Err(e) => {
                self.data.bufs.truncate(bufs_len);
                self.data.bytes.truncate(bytes_len);
                self.data.byte_args.truncate(byte_args_len);
                return Err(e);
            }
        };
        self.data.last_dispatch = Some(self.data.ops.len());
        self.data.ops.push(Op::Dispatch {
            pipeline,
            grid,
            block,
            shared,
            bufs,
            bytes,
        });
        Ok(())
    }

    pub(super) fn finish(self) -> Graph {
        Graph(Arc::new(self.data))
    }
}
