// Native Metal coalesced argmax and small-k top-k kernels.

#include <metal_stdlib>
using namespace metal;

static inline uint arg_key(float v) {
    uint b = as_type<uint>(v);
    uint mag = b & 0x7FFFFFFFu;
    bool valid = mag <= 0x7F800000u && b != 0xFF800000u;
    uint ord = (mag == 0u) ? 0x80000000u : ((b & 0x80000000u) ? ~b : (b ^ 0x80000000u));
    return valid ? ord : 0u;
}

static inline bool arg_better(uint ka, uint ia, uint kb, uint ib) {
    return ka > kb || (ka != 0u && ka == kb && ia < ib);
}

static inline void arg_update(
    thread uint& bk,
    thread uint& bi,
    uint ek,
    uint ei) {
    if (arg_better(ek, ei, bk, bi)) {
        bk = ek;
        bi = ei;
    }
}

static inline void warp_reduce_max(thread uint& bk, thread uint& bi) {
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        uint ok = simd_shuffle_down(bk, off);
        uint oi = simd_shuffle_down(bi, off);
        arg_update(bk, bi, ok, oi);
    }
}

template <uint WARPS>
static inline void block_reduce_max(
    threadgroup uint* sm_k,
    threadgroup uint* sm_i,
    uint tid,
    thread uint& bk,
    thread uint& bi) {
    uint wid = tid >> 5u;
    uint lane = tid & 31u;
    warp_reduce_max(bk, bi);
    if (lane == 0u) {
        sm_k[wid] = bk;
        sm_i[wid] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0u) {
        bk = (lane < WARPS) ? sm_k[lane] : 0u;
        bi = (lane < WARPS) ? sm_i[lane] : 0xFFFFFFFFu;
        warp_reduce_max(bk, bi);
    }
}

static inline void scan_range_max(
    const device float* row_x,
    uint from,
    uint to,
    uint tid,
    uint stride,
    thread uint& bk,
    thread uint& bi) {
    uint vec_end = from + ((to - from) & ~3u);
    for (uint i = from + tid * 4u; i < vec_end; i += stride * 4u) {
        float4 v = float4(*reinterpret_cast<const device packed_float4*>(row_x + i));
        arg_update(bk, bi, arg_key(v.x), i);
        arg_update(bk, bi, arg_key(v.y), i + 1u);
        arg_update(bk, bi, arg_key(v.z), i + 2u);
        arg_update(bk, bi, arg_key(v.w), i + 3u);
    }
    if (tid == 0u) {
        for (uint i = vec_end; i < to; ++i) {
            arg_update(bk, bi, arg_key(row_x[i]), i);
        }
    }
}

// Single-launch coalesced argmax: one 256-thread group per row.
kernel void argmax_row_mtl(
    const device float* x [[buffer(0)]], device uint* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& width [[buffer(3)]],
    constant uint& first [[buffer(4)]], constant uint& whole [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= rows) {
        return;
    }
    threadgroup uint sm_k[8];
    threadgroup uint sm_i[8];
    const device float* row_x = x + ulong(tgid) * ulong(width);
    uint bk = 0u;
    uint bi = width;
    scan_range_max(row_x, 0u, width, tid, 256u, bk, bi);
    block_reduce_max<8u>(sm_k, sm_i, tid, bk, bi);
    if (tid == 0u) {
        out[tgid] = (bk != 0u) ? (first + bi) : whole;
    }
}

// Stage 1 split-width coalesced peak finder per chunk.
kernel void argmax_part_mtl(
    const device float* x [[buffer(0)]], device uint* peaks [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& width [[buffer(3)]],
    constant uint& parts [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint p = tgid.x;
    uint row = tgid.y;
    if (row >= rows || p >= parts) {
        return;
    }
    threadgroup uint sm_k[8];
    threadgroup uint sm_i[8];
    uint chunk = ((width + parts - 1u) / parts + 3u) & ~3u;
    uint from = p * chunk;
    uint to = min(from + chunk, width);
    uint bk = 0u;
    uint bi = width;
    if (from < width) {
        const device float* row_x = x + ulong(row) * ulong(width);
        scan_range_max(row_x, from, to, tid, 256u, bk, bi);
    }
    block_reduce_max<8u>(sm_k, sm_i, tid, bk, bi);
    if (tid == 0u) {
        peaks[ulong(row) * ulong(parts) + p] = (bk != 0u) ? bi : width;
    }
}

// Stage 2 argmax reduction across split-width chunk peaks.
kernel void argmax_fin_mtl(
    const device float* x [[buffer(0)]], const device uint* peaks [[buffer(1)]],
    device uint* out [[buffer(2)]], constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]], constant uint& parts [[buffer(5)]],
    constant uint& first [[buffer(6)]], constant uint& whole [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= rows) {
        return;
    }
    threadgroup uint sm_k[2];
    threadgroup uint sm_i[2];
    const device float* row_x = x + ulong(tgid) * ulong(width);
    const device uint* row_p = peaks + ulong(tgid) * ulong(parts);
    uint bk = 0u;
    uint bi = width;
    for (uint p = tid; p < parts; p += 64u) {
        uint idx = row_p[p];
        if (idx < width) {
            arg_update(bk, bi, arg_key(row_x[idx]), idx);
        }
    }
    block_reduce_max<2u>(sm_k, sm_i, tid, bk, bi);
    if (tid == 0u) {
        out[tgid] = (bk != 0u) ? (first + bi) : whole;
    }
}

// Single-launch top-k for widths up to 2048 in shared memory.
kernel void topk_small_mtl(
    const device float* x [[buffer(0)]], device uint* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& width [[buffer(3)]],
    constant uint& k [[buffer(4)]], constant uint& first [[buffer(5)]],
    constant uint& whole [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= rows) {
        return;
    }
    threadgroup uint keys[2048];
    threadgroup uint sm_k[4];
    threadgroup uint sm_i[4];
    const device float* row_x = x + ulong(tgid) * ulong(width);
    device uint* row_out = out + ulong(tgid) * ulong(k);
    for (uint i = tid; i < width; i += 128u) {
        keys[i] = arg_key(row_x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = 0u; p < k; ++p) {
        uint bk = 0u;
        uint bi = width;
        for (uint i = tid; i < width; i += 128u) {
            arg_update(bk, bi, keys[i], i);
        }
        block_reduce_max<4u>(sm_k, sm_i, tid, bk, bi);
        if (tid == 0u) {
            if (bk != 0u) {
                row_out[p] = first + bi;
                keys[bi] = 0u;
            } else {
                row_out[p] = whole;
            }
        }
        if (p + 1u < k) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Stage 2 lazy chunk-pop top-k from stage 1 peaks.
kernel void topk_select_mtl(
    const device float* x [[buffer(0)]], const device uint* peaks [[buffer(1)]],
    device uint* out [[buffer(2)]], constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]], constant uint& parts [[buffer(5)]],
    constant uint& k [[buffer(6)]], constant uint& first [[buffer(7)]],
    constant uint& whole [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= rows) {
        return;
    }
    threadgroup uint p_key[256];
    threadgroup uint p_idx[256];
    threadgroup uint sm_k[8];
    threadgroup uint sm_i[8];
    threadgroup uint win_state[3];

    const device float* row_x = x + ulong(tgid) * ulong(width);
    const device uint* row_p = peaks + ulong(tgid) * ulong(parts);
    device uint* row_out = out + ulong(tgid) * ulong(k);
    uint chunk = ((width + parts - 1u) / parts + 3u) & ~3u;

    if (tid < parts) {
        uint idx = row_p[tid];
        p_idx[tid] = idx;
        p_key[tid] = (idx < width) ? arg_key(row_x[idx]) : 0u;
    } else {
        p_idx[tid] = width;
        p_key[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0u; step < k; ++step) {
        uint bk = p_key[tid];
        uint bi = p_idx[tid];
        block_reduce_max<8u>(sm_k, sm_i, tid, bk, bi);
        if (tid == 0u) {
            if (bk != 0u) {
                row_out[step] = first + bi;
                win_state[0] = bk;
                win_state[1] = bi;
                win_state[2] = bi / chunk;
            } else {
                row_out[step] = whole;
                win_state[0] = 0u;
                win_state[1] = width;
                win_state[2] = parts;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint cap_k = win_state[0];
        if (cap_k == 0u || step + 1u == k) {
            if (cap_k == 0u) {
                for (uint rest = step + 1u + tid; rest < k; rest += 256u) {
                    row_out[rest] = whole;
                }
                return;
            }
            return;
        }
        uint cap_i = win_state[1];
        uint win_c = win_state[2];
        uint from = win_c * chunk;
        uint to = min(from + chunk, width);
        uint nk = 0u;
        uint ni = width;
        uint vec_end = from + ((to - from) & ~3u);
        for (uint i = from + tid * 4u; i < vec_end; i += 1024u) {
            float4 v = float4(*reinterpret_cast<const device packed_float4*>(row_x + i));
            uint ks[4] = {arg_key(v.x), arg_key(v.y), arg_key(v.z), arg_key(v.w)};
            for (uint j = 0u; j < 4u; ++j) {
                uint ek = ks[j];
                uint ei = i + j;
                if (ek != 0u && (ek < cap_k || (ek == cap_k && ei > cap_i))) {
                    arg_update(nk, ni, ek, ei);
                }
            }
        }
        if (tid == 0u) {
            for (uint i = vec_end; i < to; ++i) {
                uint ek = arg_key(row_x[i]);
                if (ek != 0u && (ek < cap_k || (ek == cap_k && i > cap_i))) {
                    arg_update(nk, ni, ek, i);
                }
            }
        }
        block_reduce_max<8u>(sm_k, sm_i, tid, nk, ni);
        if (tid == 0u) {
            p_key[win_c] = nk;
            p_idx[win_c] = (nk != 0u) ? ni : width;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
