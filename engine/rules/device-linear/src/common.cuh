using namespace tessera;

#if defined(TESSERA_CUDA)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void dl_cp16(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
}
static __device__ __forceinline__ void dl_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
static __device__ __forceinline__ void dl_cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
static __device__ __forceinline__ void dl_cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
#else
static __device__ __forceinline__ void dl_cp16(void* smem, const void* gmem, bool pred) {
    *reinterpret_cast<uint4*>(smem) = pred ? *reinterpret_cast<const uint4*>(gmem) : make_uint4(0u, 0u, 0u, 0u);
}
static __device__ __forceinline__ void dl_cp_commit() {}
static __device__ __forceinline__ void dl_cp_wait1() {}
static __device__ __forceinline__ void dl_cp_wait0() {}
#endif

static __device__ __forceinline__ float dl_silu(float v) { return v / (1.0f + expf(-v)); }
static __device__ __forceinline__ float dl_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
static __device__ __forceinline__ float dl_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}
static __device__ __forceinline__ float dl_load_b1(const u8* p, u32 col, bool bf16) {
    return bf16 ? __uint_as_float(u32(reinterpret_cast<const u16*>(p)[col]) << 16u)
                : reinterpret_cast<const float*>(p)[col];
}
static __device__ __forceinline__ void dl_load_b4(const u8* p, u32 col, bool bf16, float b[4]) {
    if (bf16) {
        uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(p) + col);
        b[0] = __uint_as_float(w.x << 16u); b[1] = __uint_as_float(w.x & 0xFFFF0000u);
        b[2] = __uint_as_float(w.y << 16u); b[3] = __uint_as_float(w.y & 0xFFFF0000u);
    } else {
        trellis::load4(reinterpret_cast<const float*>(p) + col, b);
    }
}

template <bool VIT = true>
static __device__ __forceinline__ float dl_epilogue(float v, const trellis::o_t* other, u64 off,
                                                    u32 col, u32 mode, const u8* bias_p = nullptr) {
    u32 base = VIT ? (mode & 15u) : mode;
    if (base == 1u) return v + trellis::load_o(other, off);
    if (base == 2u) return v + trellis::load_o(other, col);
    if (base == 3u) return dl_silu(v) * trellis::load_o(other, off);
    if (base == 4u) return v * trellis::load_o(other, off);
    if (base == 5u) return dl_silu(v);
    if constexpr (VIT) {
        if (base == 6u) return dl_gelu_tanh(v + trellis::load_o(other, col));
        if (base == 7u) return dl_gelu(v + trellis::load_o(other, col));
        if (base == 8u) {
            return v + dl_load_b1(bias_p, col, (mode & 16u) != 0u) + trellis::load_o(other, off);
        }
    }
    return v;
}

static __device__ __forceinline__ void dl_apply_epi4(const trellis::o_t* other, u64 off,
                                                     const float b[4], float v[4], u32 mode) {
    u32 base = mode & 15u;
    if (base == 1u || base == 3u || base == 4u || base == 8u) {
        float o[4];
        trellis::load_o4(other + off, o);
        if (base == 1u) { v[0] += o[0]; v[1] += o[1]; v[2] += o[2]; v[3] += o[3]; }
        else if (base == 8u) { v[0] += b[0] + o[0]; v[1] += b[1] + o[1]; v[2] += b[2] + o[2]; v[3] += b[3] + o[3]; }
        else if (base == 3u) { v[0] = dl_silu(v[0]) * o[0]; v[1] = dl_silu(v[1]) * o[1]; v[2] = dl_silu(v[2]) * o[2]; v[3] = dl_silu(v[3]) * o[3]; }
        else { v[0] *= o[0]; v[1] *= o[1]; v[2] *= o[2]; v[3] *= o[3]; }
    } else if (base == 2u) { v[0] += b[0]; v[1] += b[1]; v[2] += b[2]; v[3] += b[3]; }
    else if (base == 5u) { v[0] = dl_silu(v[0]); v[1] = dl_silu(v[1]); v[2] = dl_silu(v[2]); v[3] = dl_silu(v[3]); }
    else if (base == 6u) { v[0] = dl_gelu_tanh(v[0] + b[0]); v[1] = dl_gelu_tanh(v[1] + b[1]); v[2] = dl_gelu_tanh(v[2] + b[2]); v[3] = dl_gelu_tanh(v[3] + b[3]); }
    else if (base == 7u) { v[0] = dl_gelu(v[0] + b[0]); v[1] = dl_gelu(v[1] + b[1]); v[2] = dl_gelu(v[2] + b[2]); v[3] = dl_gelu(v[3] + b[3]); }
}

static __device__ __forceinline__ void dl_swizzle(u32 nb, u32 mb, u32 ob_in, u32 by_in, u32& bx,
                                                  u32& by, u32 sw = 8u) {
    u32 lin = by_in * nb + ob_in, grp_sz = sw * mb;
    u32 gid = lin / grp_sz, rem = lin - gid * grp_sz;
    u32 first_n = gid * sw, width = min(sw, nb - first_n);
    by = rem / width;
    bx = first_n + (rem - by * width);
}

static __device__ __forceinline__ void dl_mma_bf16(float d[4], const u32 a[4], const u32 b[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)d; (void)a; (void)b;
#endif
}

#if defined(X_BF16)
static __device__ __forceinline__ void dl_stage_bf16_x4(const u16* xp, bool ok, u32* dst_h) {
    *reinterpret_cast<uint2*>(dst_h) = ok ? *reinterpret_cast<const uint2*>(xp) : make_uint2(0u, 0u);
}
#else
static __device__ __forceinline__ void dl_stage_bf16_x4(const float* xp, bool ok, u32* dst_h,
                                                        u32* dst_l) {
    uint2 hi = make_uint2(0u, 0u), lo = make_uint2(0u, 0u);
    if (ok) {
        float4 xv = *reinterpret_cast<const float4*>(xp);
        u32 h0 = trellis::f32x2_to_bf16x2(xv.x, xv.y);
        u32 h1 = trellis::f32x2_to_bf16x2(xv.z, xv.w);
        hi = make_uint2(h0, h1);
        lo = make_uint2(
            trellis::f32x2_to_bf16x2(xv.x - __uint_as_float(h0 << 16u),
                                     xv.y - __uint_as_float(h0 & 0xFFFF0000u)),
            trellis::f32x2_to_bf16x2(xv.z - __uint_as_float(h1 << 16u),
                                     xv.w - __uint_as_float(h1 & 0xFFFF0000u)));
    }
    *reinterpret_cast<uint2*>(dst_h) = hi;
    *reinterpret_cast<uint2*>(dst_l) = lo;
}
#endif

#endif
