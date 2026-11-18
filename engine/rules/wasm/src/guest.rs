//! One module instance: no imports, fuel per call.

use crate::Error;
use std::sync::Mutex;
use wasmi::{Config, Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

/// Instructions one call may run.
pub(crate) const FUEL: u64 = 1 << 26;
/// Instructions building a whole graph may run.
pub(crate) const BUILD_FUEL: u64 = 1 << 36;

struct Live {
    store: Store<()>,
    memory: Memory,
    alloc: TypedFunc<i32, i32>,
    instance: Instance,
}

/// A module, one call at a time.
pub(crate) struct Guest(Mutex<Live>);

impl Guest {
    /// Instantiates `bytes`. Any import refuses the module.
    pub fn new(bytes: &[u8]) -> Result<Self, Error> {
        let mut config = Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, bytes).map_err(Error::wasm)?;
        let mut store = Store::new(&engine, ());
        store.set_fuel(FUEL).map_err(Error::wasm)?;
        let linker = Linker::<()>::new(&engine);
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(Error::wasm)?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or(Error::Missing("memory"))?;
        let alloc = instance
            .get_typed_func(&store, "alloc")
            .map_err(|_| Error::Missing("alloc"))?;
        let live = Live {
            store,
            memory,
            alloc,
            instance,
        };
        Ok(Guest(Mutex::new(live)))
    }

    /// Calls `name` with `input` in guest memory, or with nothing.
    pub fn call(
        &self,
        name: &'static str,
        input: Option<&[u8]>,
        fuel: u64,
    ) -> Result<Vec<u8>, Error> {
        let mut live = self.0.lock().map_err(|_| Error::Poisoned)?;
        let live = &mut *live;
        live.store.set_fuel(fuel).map_err(Error::wasm)?;
        let packed = match input {
            None => {
                let f = live.instance.get_typed_func::<(), i64>(&live.store, name);
                let f = f.map_err(|_| Error::Missing(name))?;
                f.call(&mut live.store, ()).map_err(Error::wasm)?
            }
            Some(bytes) => {
                let len = bytes.len() as i32;
                let at = live.alloc.call(&mut live.store, len).map_err(Error::wasm)?;
                let memory = live.memory;
                memory
                    .write(&mut live.store, at as u32 as usize, bytes)
                    .map_err(Error::wasm)?;
                let f = live
                    .instance
                    .get_typed_func::<(i32, i32), i64>(&live.store, name);
                let f = f.map_err(|_| Error::Missing(name))?;
                f.call(&mut live.store, (at, len)).map_err(Error::wasm)?
            }
        };
        let (at, len) = ((packed as u64 >> 32) as usize, packed as u32 as usize);
        let mut out = vec![0; len];
        live.memory
            .read(&live.store, at, &mut out)
            .map_err(Error::wasm)?;
        Ok(out)
    }
}
