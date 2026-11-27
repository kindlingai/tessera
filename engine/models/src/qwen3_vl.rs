//! Qwen3-VL-Embedding: vision tower with DeepStack and text embedding backbone.

use crate::keys::{At, Keys};
use crate::shared::{Dense, Swiglu, part};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::hf::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, DimName, Embed, EntryKind, Graph, Linear, Model, Norm,
    PathSeg, Port, Role, Shape, Tensor, TensorType, Vector, Weights, each,
};

/// Merged visual tokens per vision launch.
pub const MERGED: DimName = DimName::named("Merged");
/// Visual tokens injected into one text launch.
pub const VIS: DimName = DimName::named("Vis");

const PATCH_DIM: DimName = DimName::named("PatchDim");
const MERGE_AREA: DimName = DimName::named("MergeArea");
const POS_GRID: DimName = DimName::named("PosGrid");
const MERGE_DIM: DimName = DimName::named("MergeDim");
const OUT_HIDDEN: DimName = DimName::named("OutHidden");
const QKV_DIM: DimName = DimName::named("QkvDim");

/// Vision tower configuration for Qwen3-VL.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub depth: u64,
    pub hidden: u64,
    pub intermediate: u64,
    pub heads: u64,
    pub out_hidden: u64,
    pub patch_size: u64,
    pub temporal_patch_size: u64,
    pub in_channels: u64,
    pub merge_size: u64,
    pub pos_embed_count: u64,
    pub deepstack_indexes: Vec<u64>,
    pub eps: f32,
    pub rope_theta: f32,
    pub max_merged: u64,
}

impl VisionConfig {
    /// Elements per flattened spatio-temporal patch.
    pub fn patch_dim(&self) -> u64 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Width of each attention head.
    pub fn head_dim(&self) -> u64 {
        self.hidden / self.heads
    }

    /// Patches merged into one visual token.
    pub fn merge_area(&self) -> u64 {
        self.merge_size * self.merge_size
    }

    /// Side length of the learned positional grid.
    pub fn pos_grid(&self) -> u64 {
        (self.pos_embed_count as f64).sqrt().round() as u64
    }

    /// Dimensions declared by the vision graph.
    pub fn dims(&self) -> Vec<Dim> {
        let head_dim = self.head_dim();
        let merge_area = self.merge_area();
        assert!(
            self.hidden.is_multiple_of(self.heads) && head_dim.is_multiple_of(4),
            "vision head_dim must divide hidden and be a multiple of 4"
        );
        vec![
            Dim::bounded(MERGED, self.max_merged),
            Dim::fixed(PATCH_DIM, self.patch_dim()),
            Dim::fixed(MERGE_AREA, merge_area),
            Dim::fixed(POS_GRID, self.pos_embed_count),
            Dim::fixed(Hidden, self.hidden),
            Dim::fixed(Ffn, self.intermediate),
            Dim::fixed(Heads, self.heads),
            Dim::fixed(HeadDim, head_dim),
            Dim::fixed(Layers, self.depth),
            Dim::fixed(MERGE_DIM, merge_area * self.hidden),
            Dim::fixed(OUT_HIDDEN, self.out_hidden),
            Dim::fixed(QKV_DIM, 3 * self.hidden),
        ]
    }
}

/// Text embedding backbone configuration for Qwen3-VL.
#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub vocab: u64,
    pub hidden: u64,
    pub layers: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    pub eps: f32,
    pub theta: f32,
    pub mrope_section: [u32; 3],
    pub deepstack_layers: usize,
    pub max_rows: u64,
    pub max_picks: u64,
    pub max_vis: u64,
}

impl TextConfig {
    /// Dimensions declared by the text graph.
    pub fn dims(&self) -> Vec<Dim> {
        vec![
            Dim::bounded(Rows, self.max_rows),
            Dim::bounded(Picks, self.max_picks),
            Dim::bounded(VIS, self.max_vis),
            Dim::fixed(Vocab, self.vocab),
            Dim::fixed(Hidden, self.hidden),
            Dim::fixed(Layers, self.layers),
            Dim::fixed(Heads, self.heads),
            Dim::fixed(KvHeads, self.kv_heads),
            Dim::fixed(HeadDim, self.head_dim),
            Dim::fixed(Ffn, self.ffn),
        ]
    }
}

/// Combined vision and text configuration for Qwen3-VL-Embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub vision: VisionConfig,
    pub text: TextConfig,
}

impl Config {
    /// Reads HF `Qwen3VLConfig` JSON keys with launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_rows: u64, max_merged: u64) -> Self {
        let text = At::text_or_top();
        k.fix(&text, HIDDEN_ACT.name, json!("silu"));
        k.fix(&text, ATTENTION_BIAS.name, json!(false));
        let vocab: u64 = k.read(&text, VOCAB);
        let hidden: u64 = k.read(&text, HIDDEN);
        let layers: u64 = k.read(&text, LAYERS);
        let heads: u64 = k.read(&text, HEADS);
        let kv_heads: u64 = k.read(&text, KV_HEADS);
        let head_dim: u64 = k.read_or(&text, HEAD_DIM, hidden / heads.max(1));
        let ffn: u64 = k.read(&text, FFN);
        let eps: f32 = k.read(&text, EPS);
        let rope_at = text
            .clone()
            .or(text.under(ROPE_PARAMETERS))
            .or(text.under(ROPE_SCALING));
        let theta: f32 = k.read_or(&rope_at, ROPE_THETA, 5_000_000.0);
        let sec_at = text.under(ROPE_SCALING).or(text.under(ROPE_PARAMETERS));
        let mrope_section: [u32; 3] = k.read_or(&sec_at, "mrope_section", [24, 20, 20]);

        let vis = At::top().under(VISION_CONFIG);
        k.fix(&vis, HIDDEN_ACT.name, json!("gelu_pytorch_tanh"));
        let deepstack_indexes: Vec<u64> = k.read_or(&vis, "deepstack_visual_indexes", Vec::new());
        let vision = VisionConfig {
            depth: k.read(&vis, "depth"),
            hidden: k.read(&vis, HIDDEN),
            intermediate: k.read(&vis, FFN),
            heads: k.read(&vis, "num_heads"),
            out_hidden: k.read_or(&vis, "out_hidden_size", hidden),
            patch_size: k.read_or(&vis, "patch_size", 16),
            temporal_patch_size: k.read_or(&vis, "temporal_patch_size", 2),
            in_channels: k.read_or(&vis, "in_channels", 3),
            merge_size: k.read_or(&vis, "spatial_merge_size", 2),
            pos_embed_count: k.read_or(&vis, "num_position_embeddings", 2304),
            deepstack_indexes,
            eps: 1e-6,
            rope_theta: 10_000.0,
            max_merged,
        };
        let text_cfg = TextConfig {
            vocab,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            ffn,
            eps,
            theta,
            mrope_section,
            deepstack_layers: vision.deepstack_indexes.len(),
            max_rows,
            max_picks: max_seqs,
            max_vis: max_merged,
        };
        Self {
            vision,
            text: text_cfg,
        }
    }

    fn preset(
        vis: (u64, u64, u64, Vec<u64>),
        text: (u64, u64, u64, u64),
        bounds: [u64; 3],
    ) -> Self {
        let [max_seqs, max_rows, max_merged] = bounds;
        let (v_depth, v_hidden, v_inter, deepstack_indexes) = vis;
        let (hidden, layers, heads, ffn) = text;
        Self {
            vision: VisionConfig {
                depth: v_depth,
                hidden: v_hidden,
                intermediate: v_inter,
                heads: 16,
                out_hidden: hidden,
                patch_size: 16,
                temporal_patch_size: 2,
                in_channels: 3,
                merge_size: 2,
                pos_embed_count: 2304,
                deepstack_indexes,
                eps: 1e-6,
                rope_theta: 10_000.0,
                max_merged,
            },
            text: TextConfig {
                vocab: 151_936,
                hidden,
                layers,
                heads,
                kv_heads: 8,
                head_dim: 128,
                ffn,
                eps: 1e-6,
                theta: 5_000_000.0,
                mrope_section: [24, 20, 20],
                deepstack_layers: 3,
                max_rows,
                max_picks: max_seqs,
                max_vis: max_merged,
            },
        }
    }

    /// Qwen3-VL-Embedding-2B preset sizes.
    pub fn qwen3_vl_2b(max_seqs: u64, max_rows: u64, max_merged: u64) -> Self {
        Self::preset(
            (24, 1024, 4096, vec![5, 11, 17]),
            (2048, 28, 16, 6144),
            [max_seqs, max_rows, max_merged],
        )
    }

    /// Qwen3-VL-Embedding-8B preset sizes.
    pub fn qwen3_vl_8b(max_seqs: u64, max_rows: u64, max_merged: u64) -> Self {
        Self::preset(
            (27, 1152, 4304, vec![8, 16, 24]),
            (4096, 36, 32, 12_288),
            [max_seqs, max_rows, max_merged],
        )
    }

    /// Small test configuration covering vision, DeepStack, and text.
    pub fn small() -> Self {
        Self {
            vision: VisionConfig {
                depth: 4,
                hidden: 64,
                intermediate: 128,
                heads: 4,
                out_hidden: 64,
                patch_size: 4,
                temporal_patch_size: 2,
                in_channels: 3,
                merge_size: 2,
                pos_embed_count: 16,
                deepstack_indexes: vec![0, 2, 3],
                eps: 1e-6,
                rope_theta: 10_000.0,
                max_merged: 16,
            },
            text: TextConfig {
                vocab: 64,
                hidden: 64,
                layers: 4,
                heads: 4,
                kv_heads: 2,
                head_dim: 16,
                ffn: 128,
                eps: 1e-6,
                theta: 5_000_000.0,
                mrope_section: [4, 2, 2],
                deepstack_layers: 3,
                max_rows: 32,
                max_picks: 4,
                max_vis: 16,
            },
        }
    }
}

/// One ViT transformer block.
#[derive(Weights)]
struct VisBlock<D: Dense> {
    #[shape(Hidden)]
    norm1: Norm,
    #[shape(Hidden)]
    norm1_bias: Vector,
    #[shape(Hidden, QKV_DIM)]
    qkv: D,
    #[shape(QKV_DIM)]
    qkv_bias: Vector,
    #[shape(Hidden, Hidden)]
    proj: D,
    #[shape(Hidden)]
    proj_bias: Vector,
    #[shape(Hidden)]
    norm2: Norm,
    #[shape(Hidden)]
    norm2_bias: Vector,
    #[shape(Hidden, Ffn)]
    fc1: D,
    #[shape(Ffn)]
    fc1_bias: Vector,
    #[shape(Ffn, Hidden)]
    fc2: D,
    #[shape(Hidden)]
    fc2_bias: Vector,
}

/// Spatial 2x2 patch merger, with norm width `S0`.
#[derive(Weights)]
struct Merger<D: Dense> {
    #[shape(S0)]
    norm: Norm,
    #[shape(S0)]
    norm_bias: Vector,
    #[shape(MERGE_DIM, MERGE_DIM)]
    fc1: D,
    #[shape(MERGE_DIM)]
    fc1_bias: Vector,
    #[shape(MERGE_DIM, OUT_HIDDEN)]
    fc2: D,
    #[shape(OUT_HIDDEN)]
    fc2_bias: Vector,
}

/// Vision tower weights.
struct Vision<D: Dense> {
    patch_proj: Linear,
    patch_bias: Vector,
    pos_embed: Embed,
    blocks: Vec<VisBlock<D>>,
    merger: Merger<D>,
    deepstack: Vec<Merger<D>>,
}

impl<D: Dense> Vision<D> {
    fn declare(m: &Model, deepstack_count: usize) -> Self {
        let mut path = Vec::new();
        let patch_proj = part(
            m,
            &mut path,
            "patch_proj",
            &[PATCH_DIM.into(), Hidden.into()],
        );
        let patch_bias = part(m, &mut path, "patch_bias", &[Hidden.into()]);
        let pos_embed = part(m, &mut path, "pos_embed", &[POS_GRID.into(), Hidden.into()]);
        let blocks = each(m, "blocks", Layers, |_| Vec::new());
        let merger = part(m, &mut path, "merger", &[Hidden.into()]);
        let mut deepstack = Vec::with_capacity(deepstack_count);
        path.push(PathSeg::Field("deepstack".into()));
        for i in 0..deepstack_count {
            path.push(PathSeg::Index(i as u32));
            deepstack.push(Merger::declare(m, &mut path, &[MERGE_DIM.into()]));
            path.pop();
        }
        path.pop();
        Self {
            patch_proj,
            patch_bias,
            pos_embed,
            blocks,
            merger,
            deepstack,
        }
    }
}

/// Standard affine LayerNorm along the last axis.
fn layer_norm(x: &Tensor, w: &Norm, b: &Vector, eps: f32) -> Tensor {
    x.center().rms_norm(w, eps).add_vector(b)
}

/// Applies the main or DeepStack spatial 2x2 merger.
fn apply_merger<D: Dense>(x: &Tensor, merger: &Merger<D>, postshuffle: bool, eps: f32) -> Tensor {
    let h = if postshuffle {
        let flat = x.reshape(vec![MERGED.into(), MERGE_DIM.into()]);
        layer_norm(&flat, &merger.norm, &merger.norm_bias, eps)
    } else {
        layer_norm(x, &merger.norm, &merger.norm_bias, eps)
            .reshape(vec![MERGED.into(), MERGE_DIM.into()])
    };
    let inner = merger
        .fc1
        .apply(&h)
        .add_vector(&merger.fc1_bias)
        .act(Act::Gelu);
    merger.fc2.apply(&inner).add_vector(&merger.fc2_bias)
}

/// Builds the Qwen3-VL vision tower graph.
pub fn vision_program<D: Dense>(c: &VisionConfig) -> Graph {
    let m = Model::new(c.dims());
    let w: Vision<D> = Vision::declare(&m, c.deepstack_indexes.len());
    let f32_port = |shape: Vec<DimExpr>| Port {
        role: Role::Tokens,
        ty: TensorType::new(Shape(shape), DType::F32),
    };
    let u32_port = |role, shape: Vec<DimExpr>| Port {
        role,
        ty: TensorType::new(Shape(shape), DType::U32),
    };
    let [patches, pos_ids, pos_weights, rot_pos, bounds] = m.entry(
        EntryKind::Forward,
        [
            f32_port(vec![MERGED.into(), MERGE_AREA.into(), PATCH_DIM.into()]),
            u32_port(
                Role::Positions,
                vec![MERGED.into(), MERGE_AREA.into(), 4u64.into()],
            ),
            f32_port(vec![MERGED.into(), MERGE_AREA.into(), 4u64.into()]),
            u32_port(
                Role::Positions,
                vec![MERGED.into(), MERGE_AREA.into(), 2u64.into()],
            ),
            u32_port(
                Role::Positions,
                vec![MERGED.into(), MERGE_AREA.into(), 2u64.into()],
            ),
        ],
    );
    let p: DimExpr = DimExpr::Dim(MERGED) * DimExpr::Dim(MERGE_AREA);
    let mut x = patches
        .reshape(vec![p.clone(), PATCH_DIM.into()])
        .matmul(&w.patch_proj)
        .add_vector(&w.patch_bias);
    let ids = pos_ids.reshape(vec![p.clone(), 4u64.into()]);
    let weights = pos_weights.reshape(vec![p.clone(), 4u64.into()]);
    let pe = m.embed(&ids, &w.pos_embed).mix(1, &weights);
    x = x.add(&pe);
    let rot = rot_pos.reshape(vec![p.clone(), 2u64.into()]);
    let bnds = bounds.reshape(vec![p.clone(), 2u64.into()]);
    let rf = (c.head_dim() / 4) as u32;
    let sections = [rf, rf, 0];
    let scale = (c.head_dim() as f32).powf(-0.5);
    let mut ds_outs: Vec<Option<Tensor>> = vec![None; c.deepstack_indexes.len()];
    let head_shape = vec![p.clone(), Heads.into(), HeadDim.into()];
    let flat_shape = vec![p.clone(), Hidden.into()];
    let hidden_dim: DimExpr = Hidden.into();
    let two_hidden: DimExpr = DimExpr::Const(2) * hidden_dim.clone();
    for (i, blk) in w.blocks.iter().enumerate() {
        let [attn] = m.composite(Composite::Attention, || {
            let h = layer_norm(&x, &blk.norm1, &blk.norm1_bias, c.eps);
            let qkv = blk.qkv.apply(&h).add_vector(&blk.qkv_bias);
            let q = qkv
                .slice(1, 0u64, hidden_dim.clone())
                .reshape(head_shape.clone())
                .mrope(&rot, c.rope_theta, sections, false);
            let k = qkv
                .slice(1, hidden_dim.clone(), hidden_dim.clone())
                .reshape(head_shape.clone())
                .mrope(&rot, c.rope_theta, sections, false);
            let v = qkv
                .slice(1, two_hidden.clone(), hidden_dim.clone())
                .reshape(head_shape.clone());
            let o = m
                .attend_packed(false, scale, [&q, &k, &v], &bnds)
                .reshape(flat_shape.clone());
            [blk.proj.apply(&o).add_vector(&blk.proj_bias)]
        });
        x = x.add(&attn);
        let [mlp] = m.composite(Composite::Mlp, || {
            let h = layer_norm(&x, &blk.norm2, &blk.norm2_bias, c.eps);
            let inner = blk
                .fc1
                .apply(&h)
                .add_vector(&blk.fc1_bias)
                .act(Act::GeluTanh);
            [blk.fc2.apply(&inner).add_vector(&blk.fc2_bias)]
        });
        x = x.add(&mlp);
        if let Some(slot) = c.deepstack_indexes.iter().position(|&idx| idx == i as u64) {
            ds_outs[slot] = Some(apply_merger(&x, &w.deepstack[slot], true, c.eps));
        }
    }
    let main = apply_merger(&x, &w.merger, false, c.eps);
    let mut outputs = vec![(Role::Logits, &main)];
    let ds_refs: Vec<Tensor> = ds_outs
        .into_iter()
        .map(|t| t.expect("deepstack index in range"))
        .collect();
    for ds in &ds_refs {
        outputs.push((Role::Taps, ds));
    }
    m.finish_many(&outputs)
}

/// One text transformer layer with QK-RMSNorm and SwiGLU.
#[derive(Weights)]
struct TextLayer<D: Dense> {
    #[shape(Hidden)]
    ln: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    q: D,
    #[shape(Hidden, KvHeads * HeadDim)]
    k: D,
    #[shape(Hidden, KvHeads * HeadDim)]
    v: D,
    #[shape(HeadDim)]
    q_norm: Norm,
    #[shape(HeadDim)]
    k_norm: Norm,
    #[shape(Heads * HeadDim, Hidden)]
    o: D,
    #[shape(Hidden)]
    post: Norm,
    mlp: Swiglu<D>,
}

/// Text embedding backbone weights.
#[derive(Weights)]
struct Text<D: Dense> {
    #[shape(Vocab, Hidden)]
    embed: Embed,
    #[shape(Layers)]
    layers: Vec<TextLayer<D>>,
    #[shape(Hidden)]
    norm: Norm,
}

/// Builds the Qwen3-VL packed text embedding backbone graph.
pub fn text_program<D: Dense>(c: &TextConfig) -> Graph {
    let m = Model::new(c.dims());
    let w: Text<D> = tessera_trace::declare(&m);
    let mut ports = vec![
        Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows], DType::U32),
        },
        Port {
            role: Role::Positions,
            ty: TensorType::new(Shape(vec![Rows.into(), 3u64.into()]), DType::U32),
        },
        Port {
            role: Role::Positions,
            ty: TensorType::new(Shape(vec![Rows.into(), 2u64.into()]), DType::U32),
        },
        Port {
            role: Role::Picks,
            ty: TensorType::new([Picks], DType::U32),
        },
        Port {
            role: Role::Positions,
            ty: TensorType::new([Rows], DType::U32),
        },
        Port {
            role: Role::Tokens,
            ty: TensorType::new([VIS, Hidden], DType::F32),
        },
    ];
    for _ in 0..c.deepstack_layers {
        ports.push(Port {
            role: Role::Tokens,
            ty: TensorType::new([VIS, Hidden], DType::F32),
        });
    }
    let ins = m.entry_many(EntryKind::Forward, &ports);
    let (tokens, positions, bounds, picks, vis_idx, vis_main) =
        (&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], &ins[5]);
    let vis_ds = &ins[6..];
    let mut x = m.embed(tokens, &w.embed).add(&vis_main.gather(vis_idx));
    let scale = (c.head_dim as f32).powf(-0.5);
    let q_shape = vec![Rows.into(), Heads.into(), HeadDim.into()];
    let kv_shape = vec![Rows.into(), KvHeads.into(), HeadDim.into()];
    let flat_shape = vec![Rows.into(), DimExpr::Dim(Heads) * DimExpr::Dim(HeadDim)];
    for (l, layer) in w.layers.iter().enumerate() {
        let [attn] = m.composite(Composite::Attention, || {
            let h = x.rms_norm(&layer.ln, c.eps);
            let q = layer
                .q
                .apply(&h)
                .reshape(q_shape.clone())
                .rms_norm(&layer.q_norm, c.eps)
                .mrope(positions, c.theta, c.mrope_section, true);
            let k = layer
                .k
                .apply(&h)
                .reshape(kv_shape.clone())
                .rms_norm(&layer.k_norm, c.eps)
                .mrope(positions, c.theta, c.mrope_section, true);
            let v = layer.v.apply(&h).reshape(kv_shape.clone());
            let o = m
                .attend_packed(true, scale, [&q, &k, &v], bounds)
                .reshape(flat_shape.clone());
            [layer.o.apply(&o)]
        });
        x = x.add(&attn);
        let [mlp] = m.composite(Composite::Mlp, || {
            [layer.mlp.eval(&x.rms_norm(&layer.post, c.eps))]
        });
        x = x.add(&mlp);
        if l < vis_ds.len() {
            x = x.add(&vis_ds[l].gather(vis_idx));
        }
    }
    let out = x.gather(picks).rms_norm(&w.norm, c.eps);
    m.finish([(Role::Logits, &out)])
}

crate::preset::preset!(
    "qwen3_vl_vision",
    small: || vision_program::<Linear>(&Config::small().vision),
    forward: |n| vision_program::<Linear>(&Config::qwen3_vl_2b(1, 2048, n).vision),
);

crate::preset::preset!(
    "qwen3_vl_text",
    small: || text_program::<Linear>(&Config::small().text),
    forward: |n| text_program::<Linear>(&Config::qwen3_vl_2b(1, n, 1024).text),
);
