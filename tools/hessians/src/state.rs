//! Resume state, BF16 hidden-state I/O, and background writer.

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use tessera_checkpoint::hessian::HessianMeta;
use tessera_graph::Bf16;

/// Persistent progress file (`calib-state.json`) for `--resume`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct CalibState {
    pub done: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_id: Option<String>,
}

impl CalibState {
    /// Loads existing state on resume or starts empty.
    pub fn load(path: &Path, resume: bool) -> Result<Self, String> {
        if !resume || !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Checks and stamps `weights_id` and `hessian_id` against mixing runs.
    pub fn sync_meta(&mut self, meta: &HessianMeta, path: &Path) -> Result<(), String> {
        if !self.done.is_empty() && self.weights_id.is_none() {
            return Err(format!(
                "{}: completed layers lack weights_id",
                path.display()
            ));
        }
        if let Some(prev) = &self.weights_id
            && prev != &meta.weights_id
        {
            return Err(format!(
                "{}: weights_id mismatch ({prev} vs {})",
                path.display(),
                meta.weights_id
            ));
        }
        if let Some(prev) = &self.hessian_id
            && prev != &meta.hessian_id
        {
            return Err(format!(
                "{}: hessian_id mismatch ({prev} vs {})",
                path.display(),
                meta.hessian_id
            ));
        }
        self.weights_id = Some(meta.weights_id.clone());
        self.hessian_id = Some(meta.hessian_id.clone());
        Ok(())
    }

    /// Atomically persists `calib-state.json`.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(&tmp, text).map_err(|e| e.to_string())?;
        fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}

/// Path of layer `layer`'s BF16 hidden-state binary file.
pub(crate) fn hidden_path(work: &Path, layer: usize) -> PathBuf {
    work.join(format!("h-{layer:03}.bin"))
}

/// Chunked BF16 hidden-state file (`[N, S, H]` packed as `u16`).
pub(crate) struct HiddenFile {
    file: File,
    row_elems: usize,
}

impl HiddenFile {
    /// Creates and sizes a hidden-state file for `rows` sequences.
    pub fn create(path: &Path, rows: usize, seq_len: usize, hidden: usize) -> Result<Self, String> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let row_elems = seq_len * hidden;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        file.set_len((rows * row_elems * 2) as u64)
            .map_err(|e| e.to_string())?;
        Ok(Self { file, row_elems })
    }

    /// Opens an existing hidden-state file for reading.
    pub fn open(path: &Path, seq_len: usize, hidden: usize) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            file,
            row_elems: seq_len * hidden,
        })
    }

    /// Reads rows `[start..start+count)` widened from BF16 to `f32`.
    pub fn read_rows(&mut self, start: usize, count: usize) -> Result<Vec<f32>, String> {
        let elems = count * self.row_elems;
        let mut raw = vec![0u8; elems * 2];
        self.file
            .seek(SeekFrom::Start((start * self.row_elems * 2) as u64))
            .map_err(|e| e.to_string())?;
        self.file.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(elems);
        for &c in raw.as_chunks::<2>().0 {
            out.push(f32::from(Bf16(u16::from_le_bytes(c))));
        }
        Ok(out)
    }

    /// Writes already-packed BF16 bytes for `[start..start+count)`.
    pub fn write_raw(&mut self, start: usize, count: usize, raw: &[u8]) -> Result<(), String> {
        assert_eq!(raw.len(), count * self.row_elems * 2);
        self.file
            .seek(SeekFrom::Start((start * self.row_elems * 2) as u64))
            .map_err(|e| e.to_string())?;
        self.file.write_all(raw).map_err(|e| e.to_string())
    }
}

/// Double-buffered background reader overlapping hidden-state disk reads.
pub(crate) struct HiddenReader {
    rx: Receiver<Result<Vec<f32>, String>>,
    handle: Option<JoinHandle<()>>,
}

impl HiddenReader {
    /// Starts prefetching `batches` from `path` with depth-2 buffering.
    pub fn spawn(
        path: PathBuf,
        seq_len: usize,
        hidden: usize,
        batches: Vec<(usize, usize)>,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Result<Vec<f32>, String>>(2);
        let handle = thread::spawn(move || {
            let mut file = match HiddenFile::open(&path, seq_len, hidden) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            for (s0, s1) in batches {
                let res = file.read_rows(s0, s1 - s0);
                if tx.send(res).is_err() {
                    return;
                }
            }
        });
        Self {
            rx,
            handle: Some(handle),
        }
    }

    pub fn next_batch(&self) -> Result<Vec<f32>, String> {
        self.rx
            .recv()
            .map_err(|_| "prefetch reader exited early".to_string())?
    }

    pub fn finish(mut self) {
        drop(self.rx);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn join_worker<T>(
    tx: &mut Option<SyncSender<T>>,
    handle: &mut Option<JoinHandle<Result<(), String>>>,
    label: &str,
) -> Result<(), String> {
    tx.take();
    if let Some(h) = handle.take() {
        h.join().map_err(|_| format!("{label} panicked"))??;
    }
    Ok(())
}

/// Double-buffered background writer for BF16 hidden-state rows.
pub(crate) struct HiddenWriter {
    tx: Option<SyncSender<(usize, usize, Vec<u8>)>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl HiddenWriter {
    /// Creates `path` and spawns a depth-2 writer thread.
    pub fn spawn(
        path: PathBuf,
        rows: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        let mut file = HiddenFile::create(&path, rows, seq_len, hidden)?;
        let (tx, rx) = mpsc::sync_channel::<(usize, usize, Vec<u8>)>(2);
        let handle = thread::spawn(move || {
            for (start, count, raw) in rx {
                file.write_raw(start, count, &raw)?;
            }
            Ok(())
        });
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    pub fn send_raw(&self, start: usize, count: usize, raw: Vec<u8>) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or("hidden writer closed")?
            .send((start, count, raw))
            .map_err(|_| "hidden writer exited".to_string())
    }

    pub fn send_f32(&self, start: usize, count: usize, vals: &[f32]) -> Result<(), String> {
        let mut raw = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            raw.extend_from_slice(&Bf16::from(v).0.to_le_bytes());
        }
        self.send_raw(start, count, raw)
    }

    pub fn finish(mut self) -> Result<(), String> {
        join_worker(&mut self.tx, &mut self.handle, "hidden writer")
    }
}

#[cfg(feature = "cuda")]
/// Pinned host staging buffer for DMA hidden-state transfers.
pub(crate) struct PinnedBuf {
    ctx: &'static tessera_gpukit::cuda::Context,
    host: u64,
    len: usize,
}

#[cfg(feature = "cuda")]
impl PinnedBuf {
    pub fn new(ctx: &'static tessera_gpukit::cuda::Context, len: usize) -> Result<Self, String> {
        let (host, _) = ctx.pin(len.max(1)).map_err(|e| format!("pin: {e}"))?;
        Ok(Self { ctx, host, len })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host as *mut u8, self.len) }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host as *const u8, self.len) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedBuf {
    fn drop(&mut self) {
        self.ctx.unpin(self.host);
    }
}

type WriteTask = Box<dyn FnOnce() -> Result<(), String> + Send>;

/// Worker pool overlapping file packing and writes with compute.
pub(crate) struct AsyncWriter {
    tx: Option<SyncSender<WriteTask>>,
    handles: Vec<JoinHandle<Result<(), String>>>,
}

impl AsyncWriter {
    pub fn new(depth: usize) -> Self {
        Self::with_workers(depth, 2)
    }

    pub fn with_workers(depth: usize, workers: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel::<WriteTask>(depth.max(1));
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let mut handles = Vec::with_capacity(workers.max(1));
        for _ in 0..workers.max(1) {
            let rx = std::sync::Arc::clone(&rx);
            handles.push(thread::spawn(move || {
                loop {
                    let job = {
                        let guard = rx.lock().map_err(|_| "writer lock poisoned".to_string())?;
                        guard.recv()
                    };
                    match job {
                        Ok(task) => task()?,
                        Err(_) => return Ok(()),
                    }
                }
            }));
        }
        Self {
            tx: Some(tx),
            handles,
        }
    }

    pub fn submit(
        &self,
        job: impl FnOnce() -> Result<(), String> + Send + 'static,
    ) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or("writer closed")?
            .send(Box::new(job))
            .map_err(|_| "writer thread exited".to_string())
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.tx.take();
        let mut first_err = None;
        for handle in self.handles.drain(..) {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some("writer thread panicked".into());
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
