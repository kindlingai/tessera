static inline float nb_load_f32(device const uchar* base, ulong idx, uint fmt) {
    if (fmt == 0u) {
        return reinterpret_cast<device const float*>(base)[idx];
    }
    if (fmt == 1u) {
        ushort bits = reinterpret_cast<device const ushort*>(base)[idx];
        return as_type<float>(uint(bits) << 16);
    }
    if (fmt == 2u) {
        return float(reinterpret_cast<device const half*>(base)[idx]);
    }
    return FP8_LUT[base[idx]];
}

static inline void nb_store_f32(device uchar* base, ulong idx, float val, uint fmt) {
    if (fmt == 0u) {
        reinterpret_cast<device float*>(base)[idx] = val;
    } else if (fmt == 1u) {
        uint bits = as_type<uint>(val);
        uint lsb = (bits >> 16) & 1u;
        uint rounded = bits + 0x7FFFu + lsb;
        ushort bf = ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            ? ushort((bits >> 16) | 0x0040u)
            : ushort(rounded >> 16);
        reinterpret_cast<device ushort*>(base)[idx] = bf;
    } else {
        reinterpret_cast<device half*>(base)[idx] = half(val);
    }
}

static inline float4 nb_load4_f32(device const uchar* base, ulong idx, uint fmt) {
    if (fmt == 0u) {
        return *reinterpret_cast<device const packed_float4*>(base + idx * 4ul);
    }
    if (fmt == 1u) {
        ushort4 u = *reinterpret_cast<device const packed_ushort4*>(base + idx * 2ul);
        return float4(
            as_type<float>(uint(u.x) << 16),
            as_type<float>(uint(u.y) << 16),
            as_type<float>(uint(u.z) << 16),
            as_type<float>(uint(u.w) << 16)
        );
    }
    if (fmt == 2u) {
        return float4(*reinterpret_cast<device const packed_half4*>(base + idx * 2ul));
    }
    uchar4 b = *reinterpret_cast<device const packed_uchar4*>(base + idx);
    return float4(FP8_LUT[b.x], FP8_LUT[b.y], FP8_LUT[b.z], FP8_LUT[b.w]);
}

static inline void nb_mma_step(
    threadgroup float (&As)[32][36],
    threadgroup float (&Bs)[32][36],
    thread simdgroup_float8x8& c00,
    thread simdgroup_float8x8& c01,
    thread simdgroup_float8x8& c10,
    thread simdgroup_float8x8& c11,
    uint sg_r,
    uint sg_c
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0u; kk < 32u; kk += 8u) {
        simdgroup_float8x8 ma0, ma1, mb0, mb1;
        simdgroup_load(ma0, &As[sg_r][kk], 36u);
        simdgroup_load(ma1, &As[sg_r + 8u][kk], 36u);
        simdgroup_load(mb0, &Bs[sg_c][kk], 36u, 0ul, true);
        simdgroup_load(mb1, &Bs[sg_c + 8u][kk], 36u, 0ul, true);
        simdgroup_multiply_accumulate(c00, ma0, mb0, c00);
        simdgroup_multiply_accumulate(c01, ma0, mb1, c01);
        simdgroup_multiply_accumulate(c10, ma1, mb0, c10);
        simdgroup_multiply_accumulate(c11, ma1, mb1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void nb_epilogue(
    threadgroup float (&As)[32][36],
    simdgroup_float8x8 c00,
    simdgroup_float8x8 c01,
    simdgroup_float8x8 c10,
    simdgroup_float8x8 c11,
    device uchar* out,
    uint tid,
    uint sg_r,
    uint sg_c,
    uint row0,
    uint col0,
    uint total_rows,
    uint cout,
    uint o_fmt
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(c00, &As[sg_r][sg_c], 36u);
    simdgroup_store(c01, &As[sg_r][sg_c + 8u], 36u);
    simdgroup_store(c10, &As[sg_r + 8u][sg_c], 36u);
    simdgroup_store(c11, &As[sg_r + 8u][sg_c + 8u], 36u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r_base = tid >> 2;
    uint c_base = (tid & 3u) * 8u;
    uint row = row0 + r_base;
    if (row < total_rows) {
        ulong row_off = ulong(row) * ulong(cout);
        for (uint j = 0u; j < 8u; ++j) {
            uint co = col0 + c_base + j;
            if (co < cout) {
                nb_store_f32(out, row_off + ulong(co), As[r_base][c_base + j], o_fmt);
            }
        }
    }
}

template<bool ALIGNED>
static inline void nb_mma_run(
    threadgroup float (&As)[32][36],
    threadgroup float (&Bs)[32][36],
    device const uchar* x,
    device const uint* ids,
    device const uchar* w,
    device uchar* out,
    uint total_rows,
    uint pixels,
    uint cin,
    uint cout,
    uint flags,
    uint2 tgid,
    uint tid,
    uint sg_id
) {
    uint x_fmt = flags & 3u;
    uint w_fmt = (flags >> 2) & 3u;
    uint o_fmt = (flags >> 4) & 3u;

    uint col0 = tgid.x * 32u;
    uint row0 = tgid.y * 32u;
    uint sg_r = (sg_id >> 1) * 16u;
    uint sg_c = (sg_id & 1u) * 16u;
    uint k_total = 9u * cin;

    simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c11 = simdgroup_float8x8(0.0f);

    uint ld_row = tid >> 3;
    uint ld_k = (tid & 7u) << 2;
    uint row_a0 = row0 + ld_row;
    uint row_a1 = row0 + ld_row + 16u;
    uint p0 = (row_a0 < total_rows) ? (row_a0 % pixels) : 0u;
    uint p1 = (row_a1 < total_rows) ? (row_a1 % pixels) : 0u;
    ulong first0 = ulong(row_a0 - p0) * ulong(cin);
    ulong first1 = ulong(row_a1 - p1) * ulong(cin);
    uint co0 = col0 + ld_row;
    uint co1 = col0 + ld_row + 16u;
    ulong w_row0 = ulong(co0) * ulong(k_total);
    ulong w_row1 = ulong(co1) * ulong(k_total);

    if (ALIGNED) {
        for (uint t = 0u; t < 9u; ++t) {
            uint id0 = (row_a0 < total_rows) ? ids[ulong(p0) * 9ul + ulong(t)] : pixels;
            uint id1 = (row_a1 < total_rows) ? ids[ulong(p1) * 9ul + ulong(t)] : pixels;
            ulong tap_off = ulong(t) * ulong(cin);
            for (uint c0 = 0u; c0 < cin; c0 += 32u) {
                ulong c_idx = ulong(c0 + ld_k);
                float4 va0 = (id0 < pixels)
                    ? nb_load4_f32(x, first0 + ulong(id0) * ulong(cin) + c_idx, x_fmt)
                    : float4(0.0f);
                float4 va1 = (id1 < pixels)
                    ? nb_load4_f32(x, first1 + ulong(id1) * ulong(cin) + c_idx, x_fmt)
                    : float4(0.0f);
                *reinterpret_cast<threadgroup float4*>(&As[ld_row][ld_k]) = va0;
                *reinterpret_cast<threadgroup float4*>(&As[ld_row + 16u][ld_k]) = va1;

                float4 vb0 = (co0 < cout)
                    ? nb_load4_f32(w, w_row0 + tap_off + c_idx, w_fmt)
                    : float4(0.0f);
                float4 vb1 = (co1 < cout)
                    ? nb_load4_f32(w, w_row1 + tap_off + c_idx, w_fmt)
                    : float4(0.0f);
                *reinterpret_cast<threadgroup float4*>(&Bs[ld_row][ld_k]) = vb0;
                *reinterpret_cast<threadgroup float4*>(&Bs[ld_row + 16u][ld_k]) = vb1;

                nb_mma_step(As, Bs, c00, c01, c10, c11, sg_r, sg_c);
            }
        }
    } else {
        for (uint k0 = 0u; k0 < k_total; k0 += 32u) {
            for (uint j = 0u; j < 4u; ++j) {
                uint k = k0 + ld_k + j;
                float va0 = 0.0f;
                float va1 = 0.0f;
                float vb0 = 0.0f;
                float vb1 = 0.0f;
                if (k < k_total) {
                    uint t = k / cin;
                    uint c = k - t * cin;
                    if (row_a0 < total_rows) {
                        uint id0 = ids[ulong(p0) * 9ul + ulong(t)];
                        if (id0 < pixels) {
                            va0 = nb_load_f32(x, first0 + ulong(id0) * ulong(cin) + ulong(c), x_fmt);
                        }
                    }
                    if (row_a1 < total_rows) {
                        uint id1 = ids[ulong(p1) * 9ul + ulong(t)];
                        if (id1 < pixels) {
                            va1 = nb_load_f32(x, first1 + ulong(id1) * ulong(cin) + ulong(c), x_fmt);
                        }
                    }
                    if (co0 < cout) {
                        vb0 = nb_load_f32(w, w_row0 + ulong(k), w_fmt);
                    }
                    if (co1 < cout) {
                        vb1 = nb_load_f32(w, w_row1 + ulong(k), w_fmt);
                    }
                }
                As[ld_row][ld_k + j] = va0;
                As[ld_row + 16u][ld_k + j] = va1;
                Bs[ld_row][ld_k + j] = vb0;
                Bs[ld_row + 16u][ld_k + j] = vb1;
            }
            nb_mma_step(As, Bs, c00, c01, c10, c11, sg_r, sg_c);
        }
    }

    nb_epilogue(As, c00, c01, c10, c11, out, tid, sg_r, sg_c, row0, col0, total_rows, cout, o_fmt);
}

/// Aligned 3x3 MMA convolution for `cin % 32 == 0`.
kernel void nb_mma_aligned(
    device const uchar* x [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device uchar* out [[buffer(3)]],
    constant uint& total_rows [[buffer(4)]], constant uint& pixels [[buffer(5)]],
    constant uint& cin [[buffer(6)]], constant uint& cout [[buffer(7)]],
    constant uint& flags [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float As[32][36];
    threadgroup float Bs[32][36];
    nb_mma_run<true>(As, Bs, x, ids, w, out, total_rows, pixels, cin, cout, flags, tgid, tid, sg_id);
}

/// General implicit-GEMM 3x3 convolution for arbitrary `cin`.
kernel void nb_mma_general(
    device const uchar* x [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device uchar* out [[buffer(3)]],
    constant uint& total_rows [[buffer(4)]], constant uint& pixels [[buffer(5)]],
    constant uint& cin [[buffer(6)]], constant uint& cout [[buffer(7)]],
    constant uint& flags [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float As[32][36];
    threadgroup float Bs[32][36];
    nb_mma_run<false>(As, Bs, x, ids, w, out, total_rows, pixels, cin, cout, flags, tgid, tid, sg_id);
}
