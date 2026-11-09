//! Kernels from the ops crates, with CPU fallbacks.

use crate::cpu::Cpu;
use crate::run::Kernels;
use crate::tensor::{Error, Tensor};
use graph::{Act, Attend, Mask, Stage};
use ops_math::{gelu, rms_norm, softmax, vec_add, vec_scale};
use ops_model::{gather_rows, gqa_attention};

/// Whether this host can dispatch GPU kernels.
pub fn gpu_ready() -> bool {
    ops::testing::gpu_available()
}

/// `fallbacks` counts ops that ran on the CPU instead.
#[derive(Default)]
pub struct Gpu {
    pub fallbacks: usize,
}

fn gpu_err(e: ops::Error) -> Error {
    Error::Gpu(e.to_string())
}

impl Kernels for Gpu {
    fn matmul(&mut self, x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
        let (m, k) = x.rows();
        let n = w.shape[0];
        let problem = gemm::Problem {
            m,
            n,
            k,
            lda: k,
            ldb: k,
            ldc: n,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: true,
        };
        let call = gemm::Call {
            a: x.floats()?.to_vec(),
            b: w.floats()?.to_vec(),
            c: vec![0.0; m * n],
            problem,
        };
        let mut shape = x.shape.clone();
        *shape.last_mut().expect("verified rank") = n;
        Ok(Tensor::f32(shape, gemm::gpu(&call).map_err(gpu_err)?))
    }

    fn attend(&mut self, a: &Attend, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        let [seqs, ctx, heads, d] = q.shape[..] else {
            return Err(Error::Mismatch("attention wants rank 4".into()));
        };
        if a.mask != Mask::Causal {
            self.fallbacks += 1;
            return Cpu.attend(a, q, k, v);
        }
        let (keys, kv_heads) = (k.shape[1], k.shape[2]);
        let past = crate::cpu::past_keys(a, ctx, keys)?;
        // The kernel expects queries already scaled.
        let scaled = vec_scale::Fixture {
            x: q.floats()?.to_vec(),
            scale: a.scale,
        };
        let qs = vec_scale::gpu(&scaled).map_err(gpu_err)?;
        let (ks, vs) = (k.floats()?, v.floats()?);
        let (q_len, kv_len) = (ctx * heads * d, kv_heads * d);
        let mut out = Vec::with_capacity(qs.len());
        for s in 0..seqs {
            let mut kv = Vec::with_capacity(2 * keys * kv_len);
            for t in s * keys..(s + 1) * keys {
                kv.extend_from_slice(&ks[t * kv_len..(t + 1) * kv_len]);
                kv.extend_from_slice(&vs[t * kv_len..(t + 1) * kv_len]);
            }
            let fix = gqa_attention::Fixture {
                q: qs[s * q_len..(s + 1) * q_len].to_vec(),
                kv,
                seq_len: ctx,
                total_kv: keys,
                n_heads: heads,
                n_kv_heads: kv_heads,
                head_dim: d,
                sliding_window: 0,
                kv_cache_len: past,
            };
            out.extend(gqa_attention::gpu(&fix).map_err(gpu_err)?);
        }
        Ok(Tensor::f32(q.shape.clone(), out))
    }

    fn stage(&mut self, s: &Stage, x: &Tensor, operand: Option<&Tensor>) -> Result<Tensor, Error> {
        let xs = x.floats()?.to_vec();
        let (rows, width) = x.rows();
        let out = match (s, operand) {
            (Stage::Add, Some(o)) => {
                let o = o.floats()?;
                let addend = o.iter().copied().cycle().take(xs.len()).collect();
                vec_add::gpu(&vec_add::Fixture { x: xs, addend })
            }
            (Stage::Scale(scale), _) => vec_scale::gpu(&vec_scale::Fixture {
                x: xs,
                scale: *scale,
            }),
            (Stage::RmsNorm { eps }, Some(w)) => rms_norm::gpu(&rms_norm::Fixture {
                x: xs,
                weight: w.floats()?.to_vec(),
                rows,
                hidden: width,
                eps: *eps,
            }),
            (Stage::Act(Act::GeluTanh), _) => gelu::gpu(&gelu::Fixture { x: xs }),
            (Stage::Softmax, _) => softmax::gpu(&softmax::Fixture {
                logits: xs,
                rows,
                cols: width,
            }),
            // One rank holds the whole sum already.
            (Stage::AllReduce, _) => Ok(xs),
            _ => {
                self.fallbacks += 1;
                return Cpu.stage(s, x, operand);
            }
        };
        Ok(Tensor::f32(x.shape.clone(), out.map_err(gpu_err)?))
    }

    fn gather(&mut self, table: &Tensor, ids: &Tensor) -> Result<Tensor, Error> {
        let rows = table.shape[0];
        let indices = ids.indices()?;
        if let Some(bad) = indices.iter().find(|&&i| i >= rows) {
            return Err(Error::Mismatch(format!("index {bad} past {rows} rows")));
        }
        let fix = gather_rows::Fixture {
            src: table.floats()?.to_vec(),
            indices: indices.iter().map(|&i| i as u32).collect(),
            hidden: table.data.len() / rows.max(1),
        };
        let mut shape = ids.shape.clone();
        shape.extend_from_slice(&table.shape[1..]);
        Ok(Tensor::f32(shape, gather_rows::gpu(&fix).map_err(gpu_err)?))
    }
}
