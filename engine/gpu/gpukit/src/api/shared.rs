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
        events: ($event:ident, $record:ident, $elapsed:ident, $unevent:ident),
        global: $global:ident,
        load: ($load:ident, $load_label:literal),
        function: ($func:ident, $func_label:literal),
        alloc: ($alloc:ident, $alloc_label:literal),
        free: $free:ident,
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

        fn event(&self) -> Result<$crate::api::Handle, $crate::Error> {
            let event = self.out(stringify!($event), |e| unsafe { (self.$event)(e, 0) })?;
            let code = unsafe { (self.$record)(event, ::std::ptr::null_mut()) };
            self.check(code, stringify!($record))?;
            Ok(event)
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
