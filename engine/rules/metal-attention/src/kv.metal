#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

enum KvFmt : uint {
    KV_F32 = 0u,
    KV_BF16 = 1u,
    KV_F16 = 2u,
    KV_FP8 = 3u,
};

static inline uint kv_kept_row(
    uint slot,
    uint pos,
    uint kept_mode,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    device const uint *page_table) {
    if (kept_mode == 0u) {
        return slot * kept_param + pos;
    }
    if (kept_mode == 2u) {
        return slot * kept_param + (pos % kept_param);
    }
    uint pg = page_table[slot * kept_param + (pos >> page_shift)];
    return (pg << page_shift) | (pos & page_mask);
}

// Packed uint2 E4M3 to half4 decode.
static inline half4 kv_fp8_h4(uchar4 raw) {
    uint2 u = uint2(uint(raw.x) | (uint(raw.y) << 16u), uint(raw.z) | (uint(raw.w) << 16u));
    uint2 s = (u & 0x00800080u) << 8u;
    uint2 m = u & 0x007F007Fu;
    half4 norm = as_type<half4>(s | ((m << 7u) + 0x20002000u));
    half4 sub = as_type<half4>(s | 0x40004000u | m) - as_type<half4>(s | 0x40004000u);
    return select(norm, sub, abs(norm) < half4(0.015625h));
}

template <uint FMT>
static inline float4 kv_load4_f32(device const uchar *base, ulong elem_off) {
    if constexpr (FMT == KV_F32) {
        return *(device const float4 *)(base + (elem_off << 2u));
    } else if constexpr (FMT == KV_BF16) {
        ushort4 raw = *(device const ushort4 *)(base + (elem_off << 1u));
        return as_type<float4>(uint4(raw) << 16u);
    } else if constexpr (FMT == KV_F16) {
        return float4(*(device const half4 *)(base + (elem_off << 1u)));
    } else {
        return float4(kv_fp8_h4(*(device const uchar4 *)(base + elem_off)));
    }
}

template <uint FMT>
static inline half4 kv_load4_h4(device const uchar *base, ulong elem_off) {
    if constexpr (FMT == KV_F32) {
        return half4(*(device const float4 *)(base + (elem_off << 2u)));
    } else if constexpr (FMT == KV_BF16) {
        ushort4 raw = *(device const ushort4 *)(base + (elem_off << 1u));
        return half4(as_type<float4>(uint4(raw) << 16u));
    } else if constexpr (FMT == KV_F16) {
        return *(device const half4 *)(base + (elem_off << 1u));
    } else {
        return kv_fp8_h4(*(device const uchar4 *)(base + elem_off));
    }
}

static inline float4 kv_load4_plain(device const uchar *base, ulong elem_off, uint fmt) {
    if (fmt == KV_F32) {
        return *(device const float4 *)(base + (elem_off << 2u));
    }
    if (fmt == KV_BF16) {
        ushort4 raw = *(device const ushort4 *)(base + (elem_off << 1u));
        return as_type<float4>(uint4(raw) << 16u);
    }
    return float4(*(device const half4 *)(base + (elem_off << 1u)));
}

static inline ushort kv_f32_to_bf16(float x) {
    uint u = as_type<uint>(x);
    uint rnd = 0x7FFFu + ((u >> 16u) & 1u);
    return ushort((u + rnd) >> 16u);
}

static inline void kv_store1(device uchar *base, ulong elem_off, float val, uint out_fmt) {
    if (out_fmt == KV_F32) {
        *(device float *)(base + (elem_off << 2u)) = val;
    } else if (out_fmt == KV_BF16) {
        *(device ushort *)(base + (elem_off << 1u)) = kv_f32_to_bf16(val);
    } else {
        *(device half *)(base + (elem_off << 1u)) = half(val);
    }
}
