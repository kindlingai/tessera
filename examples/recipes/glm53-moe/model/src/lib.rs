//! GLM-5.3's routed block as fused CUDA kernels.

use tessera_guest::site::{Found, Kind, Offer, Place, Shape, Site, Weights, fixed, same, trellis};

const NAME: &str = "glm53_moe";
const ROUTE: &str = "glm53_route";
const FC1: &str = "glm53_fc1";
const FC2: &str = "glm53_fc2";
const FC1_TURN: &str = "glm53_fc1_turn";
const FC1_PF: &str = "glm53_fc1_pf";
const FC2_PF: &str = "glm53_fc2_pf";
const FINISH: &str = "glm53_finish";
const MX8_TILE: usize = 272;

/// The routed weights, by field.
const ROUTED: [&str; 13] = [
    "router",
    "router_bias",
    "signs_in",
    "gate_su",
    "gate",
    "gate_sv",
    "up_su",
    "up",
    "up_sv",
    "down_su",
    "down",
    "down_sv",
    "signs_out",
];
const SHARED: [&str; 3] = ["shared.gate", "shared.up", "shared.down"];
/// Expert codes: plain trellis, split by expert.
const CODES: [&str; 3] = ["gate", "up", "down"];
/// Weights a split of the inner width cuts.
const INNER: [&str; 6] = ["gate", "gate_sv", "up", "up_sv", "down_su", "down"];

/// Past these, the linked rules serve better or `moe.cu` cannot.
const MOST_ROWS: usize = 2048;
const MOST_EXPERTS: usize = 2048;
const MAX_SLOTS: usize = 640;
const PICKS: usize = 8;

/// Experts each row takes, by expert count.
fn picks(experts: usize) -> Option<u32> {
    match experts {
        256 => Some(8),
        32 => Some(2),
        _ => None,
    }
}

/// Largest divisor of `blocks` up to `want`.
fn split_dim(blocks: usize, want: usize) -> usize {
    let mut d = want.min(blocks);
    while d > 1 {
        if blocks.is_multiple_of(d) {
            return d;
        }
        d -= 1;
    }
    1
}

/// One width and mcg or mul1 codebook for all `codes`.
fn width(codes: &[Found]) -> Option<[u32; 2]> {
    let t = same(codes.iter().map(|c| trellis(&c.form)))??;
    (!t.rotated && t.book < 2).then_some([t.k, t.book])
}

/// Experts this rank holds: first and count.
fn held(w: &Weights, experts: usize, ffn: usize) -> Option<[usize; 2]> {
    let share = |field: &str| w.get(field)?.share;
    let whole_but =
        |split: &[&str]| (ROUTED.iter()).all(|f| split.contains(f) || share(f).is_none());
    if whole_but(&CODES) {
        return match same(CODES.map(share))? {
            None => Some([0, experts]),
            Some([0, first, count, whole]) if whole == experts => Some([first, count]),
            Some(_) => None,
        };
    }
    let Some([1, first, count, whole]) = share("gate") else {
        return None;
    };
    let same = |field: &str, axis: usize| share(field) == Some([axis, first, count, whole]);
    let cut = INNER[..5].iter().all(|f| same(f, 1)) && same("down", 2);
    (whole_but(&INNER) && count == ffn && cut).then_some([0, experts])
}

/// Route+plan, fc1, fc2, then finish over a routed block.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let r = site.region(Kind::Moe)?;
    let [len] = r.outputs[..] else {
        return None;
    };
    let x = r.inputs.first()?;
    let [rows, hidden] = *x.shape.as_slice() else {
        return None;
    };
    let w = Weights::new(r.inputs);
    let routed = w.all(ROUTED)?;
    let shared = w.all(SHARED)?;
    let [
        router,
        bias,
        signs_in,
        gate_su,
        gate,
        gate_sv,
        up_su,
        up,
        up_sv,
        down_su,
        down,
        down_sv,
        signs_out,
    ] = routed;
    let [shared_gate, shared_up, shared_down] = shared;
    let plain = routed.iter().all(|i| i.parts.is_none());
    let turned = shared.iter().all(|i| i.parts.is_some());
    let fits = r.inputs.len() == 1 + ROUTED.len() + SHARED.len()
        && w.len() == ROUTED.len() + SHARED.len()
        && r.sums.is_empty()
        && x.weight.is_none()
        && x.dims.iter().all(Option::is_none)
        && r.kept.is_empty();
    if !(plain && turned && fits) {
        return None;
    }
    let [experts, ffn] = *gate_sv.shape.as_slice() else {
        return None;
    };
    let [first, count] = held(&w, experts, ffn)?;
    let [inner, wide] = shared_gate.parts?;
    let max_rows = r.knob(NAME, "max_rows").unwrap_or(MOST_ROWS);
    let max_slots = r.knob(NAME, "max_slots").unwrap_or(MAX_SLOTS);
    let sized = router.shape == [experts, hidden]
        && inner == hidden
        && shared_up.parts == shared_gate.parts
        && shared_down.parts == Some([wide, hidden])
        && rows > 0
        && rows <= max_rows
        && experts <= MOST_EXPERTS
        && hidden % 128 == 0
        && ffn % 128 == 0
        && wide % 128 == 0
        && wide <= ffn;
    if !sized {
        return None;
    }
    // Floats are f32, but the router may be bf16.
    let mut floats = ROUTED.iter().zip(&routed);
    let floats = floats.all(|(f, i)| *f == "router" || CODES.contains(f) || i.form == "f32");
    if x.form != "f32" || !floats {
        return None;
    }
    let bf16 = match router.form.as_str() {
        "f32" => 0,
        "bf16" => 1,
        _ => return None,
    };
    let [k1, book1] = width(&[gate, up])?;
    let [k2, book2] = width(&[down])?;
    let [sh_k1, sh_book1] = width(&[shared_gate, shared_up])?;
    let [sh_k2, sh_book2] = width(&[shared_down])?;
    let picks = picks(experts)? as usize;
    let [plan, gpart, z] = [0, 1, 2].map(|draft| Place::Draft { draft });
    let chunks = ffn / 128;
    let ibs = hidden / 128;
    let prefill_rows = r.knob(NAME, "prefill").unwrap_or(32);
    let prefill = rows > prefill_rows;
    let slot_cap = match r.knob(NAME, "slot_cap") {
        Some(v) => {
            let valid = if prefill {
                matches!(v, 8 | 32 | 64)
            } else {
                v == 8
            };
            if !valid {
                return None;
            }
            v
        }
        None => {
            if prefill {
                64
            } else {
                8
            }
        }
    };
    let z_chunks = if prefill { 0 } else { chunks };
    let ks1_want = r.knob(NAME, "ks1").unwrap_or(if rows == 1 { 8 } else { 4 });
    let ns2_want = r.knob(NAME, "ns2").unwrap_or(if rows == 1 { 8 } else { 3 });
    let ks1 = split_dim(ibs, ks1_want);
    let ns2 = split_dim(ibs, ns2_want);
    let n_sh = rows.div_ceil(slot_cap);
    let assigns = (rows * picks).min(count * rows);
    let active = count.min(assigns);
    let max_routed = (count * n_sh).min(active + (assigns - active) / slot_cap);
    if max_routed > max_slots {
        return None;
    }
    let total_slots = n_sh + max_routed;
    let max_col = rows.min(8);
    let pf64_rows = r.knob(NAME, "route_pf64_rows").unwrap_or(512);
    let (route_x, route_y, rks) = if rows == 1 && bf16 == 1 && experts % 2 == 0 && hidden % 8 == 0 {
        (experts / 2, 1, 1)
    } else if rows >= 64 && bf16 == 1 && experts == 256 && hidden % 64 == 0 {
        (1, rows.div_ceil(32), 1)
    } else if rows >= 64 && experts % 64 == 0 && hidden % 32 == 0 {
        let rm = if rows >= pf64_rows { 64 } else { 32 };
        let nb = experts / 64;
        let rtiles = rows.div_ceil(rm);
        let want = r
            .knob(NAME, "rks")
            .unwrap_or_else(|| 192usize.div_ceil(nb * rtiles).clamp(1, 8));
        let rks = split_dim(hidden / 32, want);
        (nb * rks, rtiles, rks)
    } else if rows > 1 && experts % 16 == 0 && hidden % 256 == 0 {
        let nb = experts / 16;
        let rtiles = rows.div_ceil(16);
        let want = r
            .knob(NAME, "rks")
            .unwrap_or_else(|| 128usize.div_ceil(nb * rtiles).max(1));
        let rks = split_dim(hidden / 256, want);
        (nb * rks, rtiles, rks)
    } else {
        (experts.div_ceil(2), 1, 1)
    };
    let mut shared_defs = Vec::new();
    if max_rows != MOST_ROWS {
        shared_defs.push(("MAX_ROWS".into(), max_rows.to_string()));
    }
    if max_slots != MAX_SLOTS {
        shared_defs.push(("MAX_SLOTS".into(), max_slots.to_string()));
    }
    if r.knob(NAME, "slot_cap").is_some() || r.knob(NAME, "prefill").is_some() {
        shared_defs.push(("GLM53_SLOT_CAP".into(), format!("{slot_cap}u")));
    }
    if let Some(mx8) = r.knob(NAME, "use_mx8") {
        shared_defs.push(("GLM53_USE_MX8".into(), mx8.to_string()));
    }
    if r.knob(NAME, "route_pf64_rows").is_some() {
        shared_defs.push(("ROUTE_PF64_MIN_ROWS".into(), format!("{pf64_rows}u")));
    }
    let defs_with = |extra: Option<(&str, usize)>| {
        let mut defs = shared_defs.clone();
        if let Some((k, v)) = extra {
            defs.push((k.into(), v.to_string()));
        }
        defs
    };
    let plan_words = 1 + max_slots * 50 + max_rows * PICKS * 2;
    let scores_off = (plan_words + 3) & !3;
    let route = Offer {
        kernel: ROUTE,
        reads: vec![Place::Read(0), router.read(), bias.read()],
        len: scores_off + rows * experts * rks,
        counts: fixed(&[rows, hidden, experts, bf16, picks, first, count]),
        shape: Shape::grid(route_x, route_y, 256),
        work: (rows * experts * hidden) as f64,
        defines: defs_with(
            r.knob(NAME, "route_min_blocks")
                .map(|v| ("ROUTE_MIN_BLOCKS", v)),
        ),
        ..Offer::default()
    };
    let codes1 = [k1, sh_k1, book1, sh_book1].map(|b| b as usize);
    let codes2 = [k2, sh_k2, book2, sh_book2].map(|b| b as usize);
    let mut steps = vec![route];
    let z_src = if prefill {
        let [x_turned, act_out, z_pf] = [1, 2, 3].map(|draft| Place::Draft { draft });
        let r_turn = vec![
            Place::Read(0),
            plan,
            signs_in.read(),
            gate_su.read(),
            up_su.read(),
            shared_gate.part(1),
            shared_up.part(1),
        ];
        let c1 = [&[rows, hidden, ffn, picks][..], &codes1, &[first, wide]].concat();
        let c2 = [&[rows, hidden, ffn, picks][..], &codes2, &[first, wide]].concat();
        let mut r1 = vec![Place::Read(0), plan, x_turned];
        r1.extend([signs_in, gate_su, gate, gate_sv, up_su, up, up_sv, down_su].map(|f| f.read()));
        r1.extend(
            [shared_gate, shared_up]
                .map(|f| [f.part(0), f.part(1), f.part(2)])
                .concat(),
        );
        r1.push(shared_down.part(1));
        steps.extend([
            grid_offer(
                FC1_TURN,
                r_turn,
                total_slots * ibs * 16 * MX8_TILE,
                &[rows, hidden, first],
                (ibs, total_slots),
                total_slots * 64 * hidden,
                defs_with(
                    r.knob(NAME, "turn_min_blocks")
                        .map(|v| ("FC1_TURN_MIN_BLOCKS", v)),
                ),
            ),
            grid_offer(
                FC1_PF,
                r1,
                total_slots * chunks * 4096,
                &c1,
                (chunks, total_slots),
                total_slots * 2 * hidden * ffn,
                defs_with(
                    r.knob(NAME, "pf1_min_blocks")
                        .map(|v| ("FC1_PF_MIN_BLOCKS", v)),
                ),
            ),
            grid_offer(
                FC2_PF,
                vec![
                    plan,
                    act_out,
                    down.read(),
                    shared_down.part(0),
                    down_sv.read(),
                    signs_out.read(),
                    shared_down.part(2),
                ],
                rows * (picks + 1) * hidden / 2,
                &c2,
                (ibs, total_slots),
                total_slots * hidden * ffn,
                defs_with(
                    r.knob(NAME, "pf2_min_blocks")
                        .map(|v| ("FC2_PF_MIN_BLOCKS", v)),
                ),
            ),
        ]);
        z_pf
    } else {
        let c1 = [
            &[rows, hidden, ffn, picks][..],
            &codes1,
            &[first, wide, ks1],
        ]
        .concat();
        let c2 = [
            &[rows, hidden, ffn, picks][..],
            &codes2,
            &[first, wide, ks1, ns2],
        ]
        .concat();
        let mut r1 = vec![Place::Read(0), plan];
        r1.extend([signs_in, gate_su, gate, up_su, up].map(|f| f.read()));
        r1.extend(
            [shared_gate, shared_up]
                .map(|f| [f.part(0), f.part(1)])
                .concat(),
        );
        let mut r2 = vec![plan, gpart];
        r2.extend([gate_sv, up_sv, down_su, down].map(|f| f.read()));
        r2.extend([shared_gate.part(2), shared_up.part(2)]);
        r2.extend([shared_down.part(0), shared_down.part(1)]);
        steps.extend([
            grid_offer(
                FC1,
                r1,
                total_slots * chunks * 2 * ks1 * max_col * 128,
                &c1,
                (chunks * 2 * ks1, total_slots),
                total_slots * 2 * hidden * ffn,
                defs_with(
                    r.knob(NAME, "fc1_min_blocks")
                        .map(|v| ("FC1_MIN_BLOCKS", v)),
                ),
            ),
            grid_offer(
                FC2,
                r2,
                rows * (picks + 1) * chunks * hidden / 2,
                &c2,
                (chunks * ns2, total_slots),
                total_slots * hidden * ffn,
                defs_with(
                    r.knob(NAME, "fc2_min_blocks")
                        .map(|v| ("FC2_MIN_BLOCKS", v)),
                ),
            ),
        ]);
        z
    };
    let finish_shape = if prefill {
        Shape::grid(hidden / 128, rows.div_ceil(8), 256)
    } else {
        Shape::grid(hidden / 128, rows, (picks + 1) * 32)
    };
    steps.push(Offer {
        kernel: FINISH,
        reads: vec![
            plan,
            down_sv.read(),
            signs_out.read(),
            shared_down.part(2),
            z_src,
        ],
        len,
        counts: fixed(&[rows, hidden, ffn, picks, wide, z_chunks]),
        shape: finish_shape,
        work: len as f64,
        defines: defs_with(
            r.knob(NAME, "finish_min_blocks")
                .map(|v| ("FINISH_MIN_BLOCKS", v)),
        ),
        ..Offer::default()
    });
    Some(steps)
}

fn grid_offer(
    kernel: &'static str,
    reads: Vec<Place>,
    len: usize,
    counts: &[usize],
    (x, y): (usize, usize),
    work: usize,
    defines: Vec<(String, String)>,
) -> Offer {
    Offer {
        kernel,
        reads,
        len,
        counts: fixed(counts),
        shape: Shape::grid(x, y, 256),
        work: work as f64,
        guess: 1e-15,
        defines,
        ..Offer::default()
    }
}

tessera_guest::exports! {
    rules: { NAME => Kind::Moe },
    kernels: { "moe.cu": [ROUTE, FC1, FC2, FC1_TURN, FC1_PF, FC2_PF, FINISH] },
    files: ["mx8.cuh", "route.cuh"],
    knobs: [
        (NAME, "prefill", 0, 2048),
        (NAME, "slot_cap", 8, 64),
        (NAME, "ks1", 1, 16),
        (NAME, "ns2", 1, 16),
        (NAME, "rks", 1, 32),
        (NAME, "route_pf64_rows", 64, 2048),
        (NAME, "max_rows", 1, 2048),
        (NAME, "max_slots", 1, 2048),
        (NAME, "use_mx8", 0, 1),
        (NAME, "route_min_blocks", 1, 16),
        (NAME, "fc1_min_blocks", 1, 16),
        (NAME, "fc2_min_blocks", 1, 16),
        (NAME, "turn_min_blocks", 1, 16),
        (NAME, "pf1_min_blocks", 1, 16),
        (NAME, "pf2_min_blocks", 1, 16),
        (NAME, "finish_min_blocks", 1, 16),
    ],
    propose: offer,
}

#[cfg(test)]
mod offer_test;
