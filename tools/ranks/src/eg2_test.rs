//! The EG2 exlr8 release over two ranks on one box.

use super::{device, model};
use sensai_graph::DimName::{Context, Seqs};
use sensai_graph::shard;
use sensai_interp::Tensor;
use sensai_lower::lower;
use sensai_run::Loopback;
use sensai_run::testing::runs;
use sensai_testkit::cosine_f32;

/// Each rank loads its part of the packed weights.
#[test]
fn eg2_run() {
    let dir = sensai_testkit::reference("eg2");
    let Some(dir) = dir.filter(|d| d.join("exlr8/tokenizer.json").is_file()) else {
        return eprintln!("no eg2 reference with its exlr8 release: skipped");
    };
    let case = dir.join("cases/words");
    let text = std::fs::read_to_string(case.join("tokens.txt")).unwrap();
    let ids: Vec<u32> = text
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();
    let n = ids.len();
    let (g, weights, _) = model("eg2", &dir, n as u64).unwrap();
    let sharded = shard(&g, &sensai_graph::Parts::even(2)).unwrap().graph;
    let settings = runs().pop().unwrap();
    let device = device(&settings, Default::default(), 2, settings.gpu().is_none());
    let sizes = [(Seqs, 1), (Context, n as u64)];
    let plans = [0, 1].map(|r| lower(&sharded, 0, &sizes, &device.clone().as_rank(r)).unwrap());
    let mut ranks = Loopback::open(&plans, &settings).unwrap();
    weights.each(|at, w| ranks.load_weight(at, w)).unwrap();
    let tokens = Tensor::u32(vec![1, n], ids);
    let positions = Tensor::u32(vec![1, n], (0..n as u32).collect());
    let got = ranks.run(&[tokens.view(), positions.view()]).unwrap();
    let bytes = std::fs::read(case.join("embedding.f32")).unwrap();
    let want: Vec<f32> = bytes
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let cos = cosine_f32(got[0].floats().unwrap(), &want);
    eprintln!(
        "eg2 over two ranks on {}: cos {cos:.5}",
        ranks.rank(0).device()
    );
    // The golden test's floor for the exlr8 release.
    assert!(cos > 0.98, "cos {cos}");
}
