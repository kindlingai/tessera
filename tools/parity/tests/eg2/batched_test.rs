//! Padded texts with listed keys match each text alone.

use tessera_interp::{Tensor, run};
use tessera_lower::{Device, Pick, Settings, lower};
use tessera_models::eg2::{batched, program};
use tessera_parity::{interpreted, loaded};
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, cosine_f32, weights_with};
use tessera_trace::DimName::{Context, Seqs};
use tessera_trace::Linear;

/// Five and eight tokens, padded to eight, on every runner.
#[test]
fn padded_texts() {
    let c = tessera_models::eg2::small();
    let (g, alone) = (batched::<Linear>(&c), program::<Linear>(&c));
    let w = weights_with(&g, &mut Noise(5), |_| (0.0, 0.2));
    let texts = [vec![2, 9, 4, 7, 1], vec![2, 3, 5, 8, 6, 11, 10, 1]];
    let len = 8;
    let (mut tokens, mut lists) = (vec![0; 2 * len], Vec::new());
    for (row, text) in tokens.chunks_mut(len).zip(&texts) {
        row[..text.len()].copy_from_slice(text);
        let own = |j: u32| {
            if (j as usize) < text.len() {
                j
            } else {
                len as u32
            }
        };
        (0..len).for_each(|_| lists.extend((0..len as u32).map(own)));
    }
    let positions = (0..2).flat_map(|_| 0..len as u32).collect();
    let inputs = [
        Tensor::u32(vec![2, len], tokens),
        Tensor::u32(vec![2, len], positions),
        Tensor::u32(vec![2, len, len], lists),
    ];
    let mut outs = vec![run(&g, &w, 0, &inputs).unwrap().remove(0)];
    let sizes = [(Seqs, 2), (Context, len as u64)];
    let cpu = Settings {
        backend: Pick::Cpu,
        ..host()
    };
    let feeds: Vec<_> = inputs.iter().map(Tensor::view).collect();
    for s in runs().into_iter().chain([cpu]) {
        let plan = lower(&g, 0, &sizes, &Device::guess(&s)).unwrap();
        outs.push(loaded(plan, &g, &w, &s).run(&feeds).unwrap().remove(0));
    }
    for (t, text) in texts.iter().enumerate() {
        let want = interpreted(&alone, &w, 1, text);
        for got in &outs {
            let block = &got.floats().unwrap()[t * len * want.len()..][..len * want.len()];
            let mut mean = vec![0.0; want.len()];
            for row in block.chunks(want.len()).take(text.len()) {
                mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
            }
            let cos = cosine_f32(&mean, &want);
            assert!(cos > 0.9999, "text {t}: cos {cos}");
        }
    }
}
