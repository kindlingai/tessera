use crate::calib::{round_bf16, sort_cpu, syrk_dense_cpu, syrk_tiles_cpu};
use tessera_testkit::Noise;

#[test]
fn calib_syrk_accuracy() {
    let mut rng = Noise(42);
    let (rows, top_k, experts, dim, tile) = (12usize, 2usize, 4usize, 8usize, 4usize);
    let x: Vec<f32> = (0..rows * dim).map(|_| round_bf16(rng.draw())).collect();
    let ids: Vec<u32> = (0..rows * top_k)
        .map(|i| ((i * 3 + 1) % experts) as u32)
        .collect();
    let (offsets, perm, counts) = sort_cpu(&ids, experts, 0);
    assert_eq!(counts.iter().sum::<i64>() as usize, rows * top_k);

    let tiles = (dim / tile) * (dim / tile + 1) / 2;
    let mut acc1 = vec![0.0f64; experts * tiles * tile * tile];
    let mut acc2 = vec![0.0f64; experts * tiles * tile * tile];
    syrk_tiles_cpu(&x, &perm, &offsets, dim, tile, top_k, &mut acc1);
    syrk_tiles_cpu(&x, &perm, &offsets, dim, tile, top_k, &mut acc2);
    assert_eq!(acc1, acc2);

    // Check packed tiles against float64 reference.
    for e in 0..experts {
        let mut ref64 = vec![0.0f64; dim * dim];
        for (pick, &eid) in ids.iter().enumerate() {
            if eid as usize != e {
                continue;
            }
            let r = pick / top_k;
            for i in 0..dim {
                for j in 0..dim {
                    ref64[i * dim + j] += f64::from(x[r * dim + i]) * f64::from(x[r * dim + j]);
                }
            }
        }
        let mut tidx = 0usize;
        for br in 0..(dim / tile) {
            for bc in br..(dim / tile) {
                let base = (e * tiles + tidx) * tile * tile;
                for lr in 0..tile {
                    for lc in 0..tile {
                        let got = acc1[base + lr * tile + lc];
                        let want = ref64[(br * tile + lr) * dim + (bc * tile + lc)];
                        assert!(
                            (got - want).abs() < 1e-4,
                            "e={e} ({br},{bc}) {got} vs {want}"
                        );
                    }
                }
                tidx += 1;
            }
        }
    }

    let mut dense = vec![0.0f64; dim * dim];
    syrk_dense_cpu(&x, rows, dim, &mut dense);
    for i in 0..dim {
        for j in 0..dim {
            assert_eq!(dense[i * dim + j], dense[j * dim + i]);
        }
    }
}
