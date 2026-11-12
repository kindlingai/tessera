#include <metal_stdlib>
using namespace metal;

inline float widen(float v) { return v; }
inline float widen(ushort v) { return as_type<float>((uint)v << 16); }

/// One threadgroup per gathered row.
template <typename T>
kernel void gather_rows_t(
    device float *out [[buffer(0)]],
    device const T *src [[buffer(1)]],
    device const uint *indices [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint num_indices = dims.x;
    uint hidden = dims.y;
    uint t = tgp.y;
    if (t >= num_indices) {
        return;
    }
    uint row = indices[t];
    const device T *s = src + (ulong)row * hidden;
    device float *d = out + (ulong)t * hidden;
    for (uint i = lid; i < hidden; i += 256u) {
        d[i] = widen(s[i]);
    }
}

template [[host_name("gather_rows")]] kernel void gather_rows_t<float>(
    device float *, device const float *, device const uint *, constant uint2 &, uint3, uint);
template [[host_name("gather_rows_bf16")]] kernel void gather_rows_t<ushort>(
    device float *, device const ushort *, device const uint *, constant uint2 &, uint3, uint);
