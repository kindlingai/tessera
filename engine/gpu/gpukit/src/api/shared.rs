//! Shared `Api` method bodies for CUDA and ROCm backends.

/// The `Api` methods both backends share. Use inside `impl Api`.
macro_rules! shared_api {
    (
        context: $ctx:ty,
        error: $error:path,
        check: $check:path,
        synchronize: ($sync:ident, $sync_label:literal),
        begin_capture: ($begin:ident, $begin_label:literal, $mode:expr),
        end_capture: ($end:ident, $end_label:literal),
        replay: ($replay:ident, $replay_label:literal),
        unload: ($module:ident, $exec:ident, $graph:ident, $stream:ident),
        events: ($event:ident, $record:ident, $wait_ev:ident, $elapsed:ident, $unevent:ident),
        global: $global:ident,
        load: ($load:ident, $load_label:literal),
        function: ($func:ident, $func_label:literal),
        alloc: ($alloc:ident, $alloc_label:literal),
        free: $free:ident,
        free_memory: ($info:ident, $info_label:literal),
        host_alloc: (
            $host:ident, $host_label:literal, $flags:expr,
            $map:ident, $map_label:literal
        ),
        host_free: $host_free:ident,
        register: ($reg:ident, $reg_label:literal, $reg_flags:expr, $unreg:ident),
        ipc: (
            $ipc_get:ident, $ipc_get_label:literal,
            $ipc_open:ident, $ipc_open_label:literal, $ipc_flags:expr,
            $ipc_close:ident
        ),
        occupancy: ($occ:ident, $occ_label:literal),
        func_attr: ($func_attr:ident, $func_attr_label:literal),
        zero: ($zero:ident, $zero_label:literal) $(,)?
    ) => {
        fn error(msg: String) -> $crate::Error {
            $error(msg)
        }

        fn describe(ctx: &$ctx) -> $crate::Platform {
            ctx.platform()
        }

        fn open() -> Result<$ctx, $crate::Error> {
            <$ctx>::new($crate::api::ContextConfig {
                device_ordinal: $crate::device::ordinal(),
            })
        }

        fn caches() -> &'static $crate::device::Caches<$ctx> {
            static CACHES: $crate::device::Caches<$ctx> = $crate::device::Caches::new();
            &CACHES
        }

        fn capturing() -> &'static ::std::thread::LocalKey<::std::cell::Cell<$crate::api::Handle>> {
            ::std::thread_local! {
                static CAPTURING: ::std::cell::Cell<$crate::api::Handle> =
                    const { ::std::cell::Cell::new(::std::ptr::null_mut()) };
            }
            &CAPTURING
        }

        fn active_lane() -> &'static ::std::thread::LocalKey<::std::cell::Cell<u8>> {
            ::std::thread_local! {
                static LANE: ::std::cell::Cell<u8> = const { ::std::cell::Cell::new(0) };
            }
            &LANE
        }

        fn check(&self, code: ::std::ffi::c_int, what: &str) -> Result<(), $crate::Error> {
            $check(self, code, what)
        }

        fn wait(&self, stream: $crate::api::Handle) -> Result<(), $crate::Error> {
            let code = unsafe { (self.$sync)(stream) };
            self.check(code, $sync_label)
        }

        fn begin_capture(&self, stream: $crate::api::Handle) -> Result<(), $crate::Error> {
            let code = unsafe { (self.$begin)(stream, $mode) };
            self.check(code, $begin_label)
        }

        fn end_capture(
            &self,
            stream: $crate::api::Handle,
            graph: &mut $crate::api::Handle,
        ) -> Result<(), $crate::Error> {
            let code = unsafe { (self.$end)(stream, graph) };
            self.check(code, $end_label)
        }

        fn replay(&self, exec: $crate::api::Handle) -> Result<(), $crate::Error> {
            let code = unsafe { (self.$replay)(exec, ::std::ptr::null_mut()) };
            self.check(code, $replay_label)
        }

        fn destroy(&self, kind: $crate::api::Kind, handle: $crate::api::Handle) {
            let free = match kind {
                $crate::api::Kind::Module => self.$module,
                $crate::api::Kind::Exec => self.$exec,
                $crate::api::Kind::Graph => self.$graph,
                $crate::api::Kind::Stream => self.$stream,
                $crate::api::Kind::Event => self.$unevent,
            };
            unsafe { free(handle) };
        }

        fn load(&self, image: &[u8]) -> Result<$crate::api::Handle, $crate::Error> {
            self.out($load_label, |m| unsafe {
                (self.$load)(m, image.as_ptr().cast())
            })
        }

        fn function(
            &self,
            module: $crate::api::Handle,
            name: &::std::ffi::CStr,
        ) -> Result<$crate::api::Handle, $crate::Error> {
            self.out($func_label, |f| unsafe {
                (self.$func)(f, module, name.as_ptr())
            })
        }

        fn alloc(&self, size: usize) -> Result<u64, $crate::Error> {
            let mut ptr = 0;
            self.check(unsafe { (self.$alloc)(&mut ptr, size) }, $alloc_label)?;
            Ok(ptr)
        }

        fn free(&self, ptr: u64) {
            unsafe { (self.$free)(ptr) };
        }

        fn free_memory(&self) -> Result<u64, $crate::Error> {
            let (mut free, mut total) = (0, 0);
            self.check(unsafe { (self.$info)(&mut free, &mut total) }, $info_label)?;
            Ok(free as u64)
        }

        fn host_alloc(&self, size: usize) -> Result<(u64, u64), $crate::Error> {
            let mut host = ::std::ptr::null_mut();
            let code = unsafe { (self.$host)(&mut host, size, $flags) };
            self.check(code, $host_label)?;
            let mut device = 0;
            let code = unsafe { (self.$map)(&mut device, host, 0) };
            self.check(code, $map_label)?;
            Ok((host as u64, device))
        }

        fn host_free(&self, host: u64) {
            unsafe { (self.$host_free)(host as *mut ::std::ffi::c_void) };
        }

        fn zero(&self, ptr: u64, size: usize) -> Result<(), $crate::Error> {
            self.check(unsafe { (self.$zero)(ptr, 0, size) }, $zero_label)
        }

        fn register(
            &self,
            host: *mut ::std::ffi::c_void,
            size: usize,
        ) -> Result<u64, $crate::Error> {
            let code = unsafe { (self.$reg)(host, size, $reg_flags) };
            self.check(code, $reg_label)?;
            let mut device = 0;
            let code = unsafe { (self.$map)(&mut device, host, 0) };
            if let Err(e) = self.check(code, $map_label) {
                self.unregister(host);
                return Err(e);
            }
            Ok(device)
        }

        fn unregister(&self, host: *mut ::std::ffi::c_void) {
            unsafe { (self.$unreg)(host) };
        }

        fn ipc_export(&self, ptr: u64) -> Result<$crate::device::IpcHandle, $crate::Error> {
            let mut handle = $crate::device::IpcHandle([0; 64]);
            self.check(unsafe { (self.$ipc_get)(&mut handle, ptr) }, $ipc_get_label)?;
            Ok(handle)
        }

        fn ipc_open(&self, handle: $crate::device::IpcHandle) -> Result<u64, $crate::Error> {
            let mut ptr = 0;
            let code = unsafe { (self.$ipc_open)(&mut ptr, handle, $ipc_flags) };
            self.check(code, $ipc_open_label)?;
            Ok(ptr)
        }

        fn ipc_close(&self, ptr: u64) {
            unsafe { (self.$ipc_close)(ptr) };
        }

        fn occupancy(
            &self,
            kernel: $crate::api::Handle,
            threads: u32,
            shared: u32,
        ) -> Result<u32, $crate::Error> {
            let blocks = self.int($occ_label, |n| unsafe {
                (self.$occ)(n, kernel, threads as ::std::ffi::c_int, shared as usize)
            })?;
            Ok(blocks.max(0) as u32)
        }

        fn stats(&self, kernel: $crate::api::Handle) -> Option<$crate::device::KernelStats> {
            let attr = |which: ::std::ffi::c_int| {
                self.int($func_attr_label, |v| unsafe {
                    (self.$func_attr)(v, which, kernel)
                })
                .ok()
                .map(|v| v.max(0) as u32)
            };
            // CUDA and HIP share function attribute codes.
            Some($crate::device::KernelStats {
                regs: attr(4)?,
                shared_bytes: attr(1)?,
                local_bytes: attr(3)?,
                max_threads: attr(0)?,
            })
        }

        fn event(&self) -> Result<$crate::api::Handle, $crate::Error> {
            let event = self.out(stringify!($event), |e| unsafe { (self.$event)(e, 0) })?;
            let code = unsafe { (self.$record)(event, ::std::ptr::null_mut()) };
            self.check(code, stringify!($record))?;
            Ok(event)
        }

        fn sync_event(&self) -> Result<$crate::api::Handle, $crate::Error> {
            self.out(stringify!($event), |e| unsafe { (self.$event)(e, 2) })
        }

        fn record_event(
            &self,
            event: $crate::api::Handle,
            stream: $crate::api::Handle,
        ) -> Result<(), $crate::Error> {
            let code = unsafe { (self.$record)(event, stream) };
            self.check(code, stringify!($record))
        }

        fn wait_event(
            &self,
            stream: $crate::api::Handle,
            event: $crate::api::Handle,
        ) -> Result<(), $crate::Error> {
            let code = unsafe { (self.$wait_ev)(stream, event, 0) };
            self.check(code, stringify!($wait_ev))
        }

        fn global(&self, module: $crate::api::Handle, name: &::std::ffi::CStr) -> Option<u64> {
            let (mut at, mut size) = (0, 0);
            let code = unsafe { (self.$global)(&mut at, &mut size, module, name.as_ptr()) };
            self.check(code, stringify!($global)).ok().map(|_| at)
        }

        fn elapsed(
            &self,
            from: $crate::api::Handle,
            to: $crate::api::Handle,
        ) -> Result<f32, $crate::Error> {
            let mut ms = 0.0;
            let code = unsafe { (self.$elapsed)(&mut ms, from, to) };
            self.check(code, stringify!($elapsed))?;
            Ok(ms)
        }
    };
}
pub(crate) use shared_api;

use super::{Api, Context, Graph, Kernel};
use crate::Error;
use crate::backend::Backend;
use crate::device::{Caches, Device, IpcHandle, KernelStats};
use crate::platform::Platform;
use std::cell::Cell;

impl<A: Api> Device for Context<A> {
    type Kernel = Kernel<A>;
    type Graph = Graph<A>;

    const BACKEND: Backend = A::BACKEND;

    fn open() -> Result<Self, Error> {
        A::open()
    }

    fn error(msg: String) -> Error {
        A::error(msg)
    }

    fn caches() -> &'static Caches<Self> {
        A::caches()
    }

    fn name(&self) -> String {
        self.0.name.clone()
    }

    fn platform(&self) -> Platform {
        A::describe(self)
    }

    fn finish(&self) -> Result<Option<f64>, Error> {
        self.synchronize()?;
        Ok(None)
    }

    fn stamp(&self) -> Result<bool, Error> {
        Context::stamp(self).map(|_| true)
    }

    fn stamps(&self) -> Result<Vec<f64>, Error> {
        Context::stamps(self)
    }

    fn gate(&self) -> Result<bool, Error> {
        Context::gate(self).map(|_| true)
    }

    fn global(&self, kernel: &Kernel<A>, name: &str) -> Result<Option<u64>, Error> {
        Context::global(self, kernel, name)
    }

    fn ticks(&self) -> f64 {
        A::ticks(self)
    }

    fn release(&self) {
        Context::release(self);
    }

    fn programmatic(&self, on: bool) {
        A::programmatic(on);
    }

    fn pin(&self, size: usize) -> Result<(u64, u64), Error> {
        Context::pin(self, size)
    }

    fn unpin(&self, host: u64) {
        Context::unpin(self, host);
    }

    fn unified(&self) -> bool {
        let platform = A::describe(self);
        platform.integrated && platform.row.is_some_and(|r| r.pinned)
    }

    fn numa_node(&self) -> Option<u32> {
        self.0.numa_node
    }

    fn bind_numa(&self) {
        Context::bind_numa(self);
    }

    fn register(&self, host: *mut u8, size: usize) -> Result<u64, Error> {
        Context::register(self, host, size)
    }

    fn unregister(&self, host: *mut u8) {
        Context::unregister(self, host);
    }

    fn ipc_export(&self, ptr: u64) -> Result<IpcHandle, Error> {
        Context::ipc_export(self, ptr)
    }

    fn ipc_open(&self, handle: IpcHandle) -> Result<u64, Error> {
        Context::ipc_open(self, handle)
    }

    fn ipc_close(&self, ptr: u64) {
        Context::ipc_close(self, ptr);
    }

    fn resident(&self, kernel: &Kernel<A>, threads: u32, shared: u32) -> Result<u32, Error> {
        Context::resident(self, kernel, threads, shared)
    }

    fn stats(&self, kernel: &Kernel<A>) -> Option<KernelStats> {
        Context::stats(self, kernel)
    }

    fn graphs(&self) -> bool {
        true
    }

    fn has_lanes(&self) -> bool {
        true
    }

    fn set_lane(&self, lane: u8) {
        Context::set_lane(self, lane);
    }

    fn active_lane(&self) -> u8 {
        Context::active_lane(self)
    }

    fn record_lane(&self, lane: u8, slot: usize) -> Result<(), Error> {
        Context::record_lane(self, lane, slot)
    }

    fn wait_lane(&self, lane: u8, slot: usize) -> Result<(), Error> {
        Context::wait_lane(self, lane, slot)
    }

    fn capturing(&self) -> bool {
        !A::capturing().with(Cell::get).is_null()
    }

    fn capture<E: From<Error>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<Graph<A>, E> {
        Context::capture(self, record)
    }

    fn replay(&self, graph: &Graph<A>) -> Result<(), Error> {
        Context::replay(self, graph)
    }

    fn alloc(&self, size: usize) -> Result<u64, Error> {
        self.locked(|a| a.alloc(size))
    }

    fn alloc_kept(&self, size: usize) -> Result<u64, Error> {
        self.locked(|a| a.alloc_kept(&self.0.own, self.0.device, size))
    }

    fn free_memory(&self) -> Option<u64> {
        // Integrated devices count reclaimable host pages as used.
        if A::describe(self).integrated {
            return None;
        }
        self.locked(|a| a.free_memory()).ok()
    }

    fn free(&self, ptr: u64) {
        let _hold = self.hold();
        // At process exit the context may already be gone.
        if self.set_current().is_ok() {
            self.0.api.free_ptr(&self.0.own, self.0.device, ptr);
        }
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        self.locked(|a| a.upload(ptr, bytes))
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        self.locked(|a| a.download(ptr, bytes))
    }

    fn upload_batch(&self, spans: &[(u64, &[u8])]) -> Result<(), Error> {
        if spans.is_empty() {
            return Ok(());
        }
        self.locked(|a| {
            for &(ptr, bytes) in spans {
                a.upload_async(ptr, bytes)?;
            }
            a.synchronize()
        })
    }

    fn download_batch(&self, spans: &mut [(u64, &mut [u8])]) -> Result<(), Error> {
        if spans.is_empty() {
            return Ok(());
        }
        self.locked(|a| {
            for (ptr, bytes) in spans {
                a.download_async(*ptr, bytes)?;
            }
            a.synchronize()
        })
    }

    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), Error> {
        if size == 0 || dst == src {
            return Ok(());
        }
        let len = size as u64;
        if src < dst + len && dst < src + len {
            let mut tmp = vec![0u8; size];
            self.download(src, &mut tmp)?;
            return self.upload(dst, &tmp);
        }
        self.locked(|a| a.copy_d2d(dst, src, size))
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        self.locked(|a| a.zero(ptr, size))
    }

    fn compile(&self, source: &str, entry: &str) -> Result<Kernel<A>, Error> {
        self.compile_kernel(source, entry)
    }
}
