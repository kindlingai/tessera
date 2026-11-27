use crate::Error;
use crate::device::Runner;
use crate::grid::{Grid, Item, KeptDesc, Layout};
use crate::overrides::{LaunchOverride, Overrides, apply_launch, set_flattener};
use crate::run::{Outcome, Run};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tessera_engine_settings::{ByBackend, Settings};
use tessera_gpukit::backend::Backend;
use tessera_gpukit::device::{Caches, Device, KernelStats};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_gpukit::platform::Platform;
use tessera_gpukit::{Error as GpuError, Limits};
use tessera_lower::{
    Kernel, KernelRef, Launch as Call, Needs, Plan, Resident, Shape, Slot, Source, Step,
};

fn check(files: &mut Overrides<(String, Vec<String>)>) -> Option<Outcome> {
    let compile = |text: &str, flags: &[&str]| match text.contains("bad") {
        true => Err(Error::Zones),
        false => Ok((
            text.to_string(),
            flags.iter().map(|s| (*s).to_string()).collect(),
        )),
    };
    files
        .check("k", false, || "planned".to_string(), compile)
        .map(|r| r.outcome)
}

#[test]
fn files_replace_kernels() {
    let dir = tessera_testkit::scratch(&format!("overrides-basic-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut files = Overrides::new(dir.clone(), Backend::Cuda);
    assert_eq!(check(&mut files), None);
    assert!(files.write("k", "planned").unwrap().is_some());
    assert_eq!(check(&mut files), None);
    // Each write changes the length, whatever the clock's grain.
    std::fs::write(files.path("k"), "edited").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    assert_eq!(files.text("k"), Some("edited"));
    std::fs::write(files.path("k"), "bad!!").unwrap();
    assert!(matches!(check(&mut files), Some(Outcome::Failed(_))));
    assert_eq!(files.text("k"), Some("edited"));
    std::fs::remove_file(files.path("k")).unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Planned));
    assert_eq!(files.kernel("k"), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn headers_expand_watch() {
    let dir = tessera_testkit::scratch(&format!("overrides-hdr-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut files = Overrides::new(dir.clone(), Backend::Cuda);
    // Missing header fails, then creating it reloads automatically.
    std::fs::write(
        files.path("k"),
        "#include \"h.cuh\"\n#include \"h.cuh\"\nbody\n",
    )
    .unwrap();
    assert!(matches!(check(&mut files), Some(Outcome::Failed(_))));
    std::fs::write(dir.join("h.cuh"), "#pragma once\nint x = 1;\n").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    let txt = files.text("k").unwrap();
    assert_eq!(txt.matches("int x = 1;").count(), 1);
    assert_eq!(check(&mut files), None);
    // Editing only the header reloads the kernel.
    std::fs::write(dir.join("h.cuh"), "#pragma once\nint x = 22;\n").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    assert!(files.text("k").unwrap().contains("int x = 22;"));
    // Mutually including #pragma once headers terminate cleanly.
    std::fs::write(
        dir.join("m1.cuh"),
        "#pragma once\n#include \"m2.cuh\"\nint m1 = 1;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("m2.cuh"),
        "#pragma once\n#include \"m1.cuh\"\nint m2 = 2;\n",
    )
    .unwrap();
    std::fs::write(files.path("k"), "#include \"m1.cuh\"\nmain_body\n").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    let mutual = files.text("k").unwrap();
    assert_eq!(mutual.matches("int m1 = 1;").count(), 1);
    assert_eq!(mutual.matches("int m2 = 2;").count(), 1);
    // Escapes and unguarded cycles fail while keeping the last kernel.
    std::fs::write(files.path("k"), "#include \"../escape.cuh\"\n").unwrap();
    assert!(matches!(check(&mut files), Some(Outcome::Failed(_))));
    std::fs::write(dir.join("a.cuh"), "#include \"b.cuh\"\n").unwrap();
    std::fs::write(dir.join("b.cuh"), "#include \"a.cuh\"\n").unwrap();
    std::fs::write(files.path("k"), "#include \"a.cuh\"\n").unwrap();
    assert!(matches!(check(&mut files), Some(Outcome::Failed(_))));
    assert!(files.text("k").unwrap().contains("int m2 = 2;"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sidecar_and_flattener_work() {
    let dir = tessera_testkit::scratch(&format!("overrides-cfg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut files = Overrides::new(dir.clone(), Backend::Cuda);
    let cfg = "threads = 128\ndynamic_shared = 4096\nnvrtc_flags = [\"--maxrregcount=64\"]\n";
    std::fs::write(files.launch_path("k"), cfg).unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    assert_eq!(files.kernel("k").unwrap().1, vec!["--maxrregcount=64"]);
    let want = LaunchOverride {
        threads: Some(128),
        dynamic_shared: Some(4096),
        nvrtc_flags: vec!["--maxrregcount=64".into()],
    };
    assert_eq!(files.launch("k"), Some(&want));
    let (sh, sm) = apply_launch(Shape::Line(300), 0, Some(&want));
    assert_eq!(
        sh,
        Shape::Grid {
            across: 3,
            down: 1,
            threads: 128
        }
    );
    assert_eq!(sm, 4096);
    std::fs::write(files.launch_path("k"), "threads = 0\n").unwrap();
    assert!(matches!(check(&mut files), Some(Outcome::Failed(_))));
    assert_eq!(files.launch("k"), Some(&want));
    std::fs::remove_file(files.launch_path("k")).unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Planned));
    set_flattener(|_, _, src, load| {
        Ok(format!(
            "FLAT:{}:{}",
            load("p.cuh").unwrap_or_default(),
            src
        ))
    });
    std::fs::write(dir.join("p.cuh"), "HDR").unwrap();
    std::fs::write(files.path("k"), "#include <tessera/1/core>\nbody").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    assert!(files.text("k").unwrap().starts_with("FLAT:HDR:"));
    // Transitive <tessera/...> import inside a quoted local header flattens too.
    std::fs::write(
        dir.join("wrap.cuh"),
        "#pragma once\n#include <tessera/1/core>\nint wrap_v = 7;\n",
    )
    .unwrap();
    std::fs::write(files.path("k"), "#include \"wrap.cuh\"\nuse_wrap\n").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    let trans = files.text("k").unwrap();
    assert!(trans.starts_with("FLAT:HDR:"));
    assert!(trans.contains("#include <tessera/1/core>"));
    assert!(trans.contains("int wrap_v = 7;"));
    assert!(trans.contains("use_wrap"));
    // Editing the transitive header reloads and reflattens automatically.
    std::fs::write(
        dir.join("wrap.cuh"),
        "#pragma once\n#include <tessera/1/core>\nint wrap_v = 99;\n",
    )
    .unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    assert!(files.text("k").unwrap().contains("int wrap_v = 99;"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[derive(Default)]
struct DispatchState {
    next_ptr: u64,
    bufs: HashMap<u64, Vec<u8>>,
    compiled: Vec<(String, Vec<String>)>,
    attrs_asked: Vec<(u32, u32)>,
    launched: Vec<(u32, u32, u32, u32)>,
}

#[derive(Clone)]
struct DispatchDev(Arc<Mutex<DispatchState>>);

#[derive(Clone)]
struct DispatchKernel;

impl Device for DispatchDev {
    const BACKEND: Backend = Backend::Cuda;
    type Kernel = DispatchKernel;
    type Graph = ();

    fn compile(&self, _source: &str, entry: &str) -> Result<DispatchKernel, GpuError> {
        self.0
            .lock()
            .unwrap()
            .compiled
            .push((entry.into(), Vec::new()));
        Ok(DispatchKernel)
    }
    fn compile_with(
        &self,
        _source: &str,
        entry: &str,
        flags: &[&str],
    ) -> Result<DispatchKernel, GpuError> {
        self.0.lock().unwrap().compiled.push((
            entry.into(),
            flags.iter().map(|s| (*s).to_string()).collect(),
        ));
        Ok(DispatchKernel)
    }
    fn resident(&self, _: &DispatchKernel, threads: u32, shared: u32) -> Result<u32, GpuError> {
        self.0.lock().unwrap().attrs_asked.push((threads, shared));
        Ok(2)
    }
    fn stats(&self, _: &DispatchKernel) -> Option<KernelStats> {
        Some(KernelStats {
            regs: 32,
            shared_bytes: 0,
            local_bytes: 0,
            max_threads: 512,
        })
    }
    fn open() -> Result<Self, GpuError> {
        Ok(Self(Arc::new(Mutex::new(DispatchState::default()))))
    }
    fn error(msg: String) -> GpuError {
        GpuError::Cuda(msg)
    }
    fn caches() -> &'static Caches<Self> {
        static CACHE: Caches<DispatchDev> = Caches::new();
        &CACHE
    }
    fn name(&self) -> String {
        "dispatch".into()
    }
    fn platform(&self) -> Platform {
        Platform::assumed(Limits::assumed())
    }
    fn graphs(&self) -> bool {
        false
    }
    fn finish(&self) -> Result<Option<f64>, GpuError> {
        Ok(None)
    }
    fn alloc(&self, size: usize) -> Result<u64, GpuError> {
        let mut s = self.0.lock().unwrap();
        let ptr = s.next_ptr.max(0x1000);
        s.next_ptr = ptr + size.max(64) as u64;
        s.bufs.insert(ptr, vec![0; size]);
        Ok(ptr)
    }
    fn free(&self, ptr: u64) {
        self.0.lock().unwrap().bufs.remove(&ptr);
    }
    fn zero(&self, ptr: u64, len: usize) -> Result<(), GpuError> {
        self.0.lock().unwrap().bufs.get_mut(&ptr).unwrap()[..len].fill(0);
        Ok(())
    }
    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), GpuError> {
        self.0.lock().unwrap().bufs.get_mut(&ptr).unwrap()[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), GpuError> {
        bytes.copy_from_slice(&self.0.lock().unwrap().bufs[&ptr][..bytes.len()]);
        Ok(())
    }
    fn capture<E: From<GpuError>>(&self, record: impl FnOnce() -> Result<(), E>) -> Result<(), E> {
        record()
    }
    fn replay(&self, _: &()) -> Result<(), GpuError> {
        Ok(())
    }
}

impl Launch for DispatchDev {
    type Kernel = DispatchKernel;

    fn launch(
        &self,
        _: &DispatchKernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        _: &mut KernelArgs,
    ) -> Result<(), GpuError> {
        self.0
            .lock()
            .unwrap()
            .launched
            .push((grid.0, grid.1, block.0, shared));
        Ok(())
    }
}

#[derive(Debug)]
struct TestKernel {
    entry: &'static str,
    body: &'static str,
    resident: Option<Resident>,
}

impl Kernel for TestKernel {
    fn name(&self) -> &'static str {
        self.entry
    }
    fn source(&self) -> Source {
        Source::Fixed {
            text: ByBackend::from_fn(|_| self.body),
            entry: self.entry,
        }
    }
    fn launches(&self, step: &Step) -> Vec<Call> {
        vec![Call {
            args: vec![step.read(0), step.out()],
            shape: Shape::Grid {
                across: 2,
                down: 1,
                threads: 32,
            },
        }]
    }
    fn needs(&self) -> Needs {
        Needs::text()
    }
    fn resident(&self) -> Option<Resident> {
        self.resident.clone()
    }
}

type ReplacedLog = Arc<Mutex<Vec<(String, Option<String>)>>>;

struct MockGrid {
    launches: Arc<Mutex<usize>>,
    replaced: ReplacedLog,
}

impl Grid for MockGrid {
    fn open(&mut self, _: &KernelRef, _: Layout) -> Result<(), Error> {
        Ok(())
    }
    fn launch(
        &mut self,
        _: &KernelRef,
        _: &[Item],
        _: &[KeptDesc],
        _: &[u32],
    ) -> Result<(), Error> {
        *self.launches.lock().unwrap() += 1;
        Ok(())
    }
    fn replace(&mut self, entry: &str, source: Option<String>) {
        self.replaced
            .lock()
            .unwrap()
            .push((entry.to_string(), source));
    }
}

fn single_step_with(entry: &'static str, body: &'static str, resident: Option<Resident>) -> Plan {
    let slot_in = Slot::floats(0, 16);
    let slot_out = Slot::floats(1, 16);
    Plan {
        id: Plan::fresh(),
        zones: vec![0, 0],
        buffers: vec![64, 64],
        inputs: vec![slot_in],
        steps: vec![Step {
            zone: 0,
            reads: vec![slot_in],
            write: slot_out,
            also: Vec::new(),
            covers: Vec::new(),
            kernel: KernelRef::new(TestKernel {
                entry,
                body,
                resident,
            }),
            rule: "test",
            site: None,
            lane: 0,
            table: None,
            index_table: None,
        }],
        ..Plan::default()
    }
}

fn single_step_plan(entry: &'static str, resident: Option<Resident>) -> Plan {
    single_step_with(entry, "__global__ void k() {}", resident)
}

#[test]
fn launch_override_dispatch_resident() {
    let dir = tessera_testkit::scratch(&format!("overrides-dispatch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let settings = Settings {
        kernels: Some(dir.clone()),
        ..Settings::default()
    };

    // Non-resident launch.toml updates compile flags, attrs, and actual grid dispatch.
    let plan = single_step_plan("disp_k", None);
    let mut runner = Runner::<DispatchDev>::new(plan, &settings).unwrap();
    let state = tessera_gpukit::device::cached_context::<DispatchDev>()
        .unwrap()
        .0
        .clone();
    state.lock().unwrap().launched.clear();
    runner.submit(0..1).unwrap();
    assert_eq!(state.lock().unwrap().launched, vec![(2, 1, 32, 0)]);

    std::fs::write(
        dir.join("disp_k.launch.toml"),
        "threads = 64\ndynamic_shared = 2048\nnvrtc_flags = [\"-DMY_OPT=1\"]\n",
    )
    .unwrap();
    state.lock().unwrap().compiled.clear();
    state.lock().unwrap().attrs_asked.clear();
    state.lock().unwrap().launched.clear();
    let reloads = runner.refresh().unwrap();
    assert_eq!(reloads.len(), 1);
    assert_eq!(reloads[0].outcome, Outcome::Loaded);
    assert!(
        state
            .lock()
            .unwrap()
            .compiled
            .contains(&("disp_k".into(), vec!["-DMY_OPT=1".into()]))
    );
    assert_eq!(state.lock().unwrap().attrs_asked.last(), Some(&(64, 2048)));
    runner.submit(0..1).unwrap();
    assert_eq!(state.lock().unwrap().launched, vec![(2, 1, 64, 2048)]);

    // Resident step rejects launch.toml explicitly and dispatches via Grid.
    let rdir = dir.join("res");
    std::fs::create_dir_all(&rdir).unwrap();
    let rsettings = Settings {
        kernels: Some(rdir.clone()),
        ..Settings::default()
    };
    let rplan = single_step_plan(
        "res_k",
        Some(Resident {
            threads: 128,
            shared: 1024,
            scratch: 0,
            lanes: Vec::new(),
        }),
    );
    let mut rrunner = Runner::<DispatchDev>::new(rplan, &rsettings).unwrap();
    let launches = Arc::new(Mutex::new(0usize));
    let replaced = Arc::new(Mutex::new(Vec::new()));
    rrunner
        .persist(Box::new(MockGrid {
            launches: launches.clone(),
            replaced: replaced.clone(),
        }))
        .unwrap();
    std::fs::write(rdir.join("res_k.launch.toml"), "threads = 64\n").unwrap();
    let rreloads = rrunner.refresh().unwrap();
    assert_eq!(rreloads.len(), 1);
    match &rreloads[0].outcome {
        Outcome::Failed(why) => assert!(why.contains("resident"), "{why}"),
        other => panic!("expected resident launch override rejection, got {other:?}"),
    }
    rrunner.submit(0..1).unwrap();
    assert_eq!(*launches.lock().unwrap(), 1);

    // Removing launch.toml and editing source reloads resident kernel cleanly.
    std::fs::remove_file(rdir.join("res_k.launch.toml")).unwrap();
    std::fs::write(
        rdir.join("res_k.cu"),
        "__global__ void res_k() { /* v2 */ }",
    )
    .unwrap();
    let ok = rrunner.refresh().unwrap();
    assert_eq!(ok.len(), 1);
    assert_eq!(ok[0].outcome, Outcome::Loaded);
    assert!(
        replaced
            .lock()
            .unwrap()
            .iter()
            .any(|(e, s)| e == "res_k" && s.as_deref().is_some_and(|t| t.contains("v2")))
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cache_key_hash() {
    let mut cache = crate::kernels::Kernels::<usize>::default();
    let mut count = 0usize;
    let step = |body| single_step_with("same", body, None).steps.pop().unwrap();
    let mut compile = |body| {
        cache
            .get(
                &step(body),
                Backend::Cuda,
                |_, _| {
                    count += 1;
                    Ok(count)
                },
                |_, _| unreachable!(),
            )
            .unwrap()
            .0
    };
    assert_eq!((compile("a"), compile("b"), compile("a")), (1, 2, 1));
}
