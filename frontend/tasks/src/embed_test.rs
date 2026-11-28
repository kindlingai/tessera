use super::api::{EmbedOptions, finish_vec};
use super::embed::{BatchWorker, Embedder, Limits};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use tessera_bundle::Overrides;
use tessera_engine::{Pick, Settings};
use tessera_testkit::cosine_f32;

/// Each eg2 case through the exlr8 release, against eg2cpu.
#[test]
fn eg2_cases() {
    let dir = tessera_testkit::reference("eg2");
    let Some(dir) = dir.filter(|d| d.join("exlr8/tokenizer.json").is_file()) else {
        return eprintln!("no eg2 reference with its exlr8 release: skipped");
    };
    let mut settings = Settings::default();
    // The host takes minutes on the long case.
    settings.backend = tessera_engine::gpu().map_or(settings.backend, Pick::Gpu);
    let mut embedder = Embedder::open(
        &dir.join("exlr8"),
        &settings,
        &Overrides::default(),
        &Default::default(),
    )
    .unwrap();
    for entry in std::fs::read_dir(dir.join("cases")).unwrap() {
        let case = entry.unwrap().path();
        let ids = std::fs::read_to_string(case.join("tokens.txt")).unwrap();
        let ids: Vec<u32> = ids.split_whitespace().map(|t| t.parse().unwrap()).collect();
        if let Ok(text) = std::fs::read_to_string(case.join("text.txt")) {
            assert_eq!(embedder.tokens(&text).unwrap(), ids, "{case:?}");
        }
        let bytes = std::fs::read(case.join("embedding.f32")).unwrap();
        let want = bytes.as_chunks().0.iter().map(|b| f32::from_le_bytes(*b));
        let got = embedder.embed_ids(&[ids]).unwrap();
        let cos = cosine_f32(&got[0], &want.collect::<Vec<_>>());
        eprintln!("{case:?} on {}: cos {cos:.5}", embedder.device());
        // The golden test's floor for the exlr8 release.
        assert!(cos > 0.98, "{case:?}: cos {cos}");
    }
}

#[test]
fn coalesces_concurrent_requests() {
    let limits = Limits {
        seqs: 4,
        rows: 10,
        merged: 2,
    };
    let (entered_tx, entered_rx) = mpsc::channel();
    let gate = Arc::new(Barrier::new(2));
    let gate_w = gate.clone();
    let batches = Arc::new(AtomicUsize::new(0));
    let max_batch = Arc::new(AtomicUsize::new(0));
    let batches_w = batches.clone();
    let max_w = max_batch.clone();
    let (worker, ()) = BatchWorker::<(u32, usize, usize, EmbedOptions), Vec<f32>>::spawn(
        move || Ok((false, (), limits)),
        move |blocked, batch| {
            let n = batch.len();
            let rows: usize = batch.iter().map(|x| x.1).sum();
            let merged: usize = batch.iter().map(|x| x.2).sum();
            assert!(n <= limits.seqs && rows <= limits.rows && merged <= limits.merged);
            batches_w.fetch_add(1, Ordering::SeqCst);
            max_w.fetch_max(n, Ordering::SeqCst);
            if !*blocked {
                *blocked = true;
                let _ = entered_tx.send(());
                gate_w.wait();
            }
            let mut out = Vec::with_capacity(n);
            for &(id, _, _, opts) in batch {
                let raw = vec![id as f32 + 1.0, 2.0, 3.0, 4.0];
                out.push(finish_vec(raw, opts.dimensions, opts.normalize)?);
            }
            Ok(out)
        },
    )
    .unwrap();
    let worker = Arc::new(worker);
    let first = {
        let w = worker.clone();
        std::thread::spawn(move || {
            w.submit(1, 3, 0, (0, 3, 0, EmbedOptions::default()))
                .unwrap()
        })
    };
    entered_rx.recv().unwrap();
    let mut handles = Vec::new();
    for i in 1u32..8 {
        let w = worker.clone();
        let opts = EmbedOptions {
            dimensions: if i.is_multiple_of(2) { Some(2) } else { None },
            normalize: !i.is_multiple_of(3),
        };
        let merged = usize::from(i == 3 || i == 4 || i == 5);
        handles.push(std::thread::spawn(move || {
            (
                i,
                opts,
                w.submit(1, 3, merged, (i, 3, merged, opts)).unwrap(),
            )
        }));
    }
    while worker.pending() < 7 {
        std::thread::yield_now();
    }
    gate.wait();
    assert_eq!(
        first.join().unwrap(),
        finish_vec(vec![1.0, 2.0, 3.0, 4.0], None, true).unwrap()
    );
    for h in handles {
        let (i, opts, got) = h.join().unwrap();
        let want = finish_vec(
            vec![i as f32 + 1.0, 2.0, 3.0, 4.0],
            opts.dimensions,
            opts.normalize,
        )
        .unwrap();
        assert_eq!(got, want);
    }
    assert!(batches.load(Ordering::SeqCst) <= 4);
    assert_eq!(max_batch.load(Ordering::SeqCst), 3);
}
