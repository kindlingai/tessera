// Primitives for CUDA (sm_70 on) and HIP. Not on Metal.

#ifndef TESSERA_BACKEND_H
#define TESSERA_BACKEND_H

namespace tessera {
namespace prim {

#if defined(TESSERA_CUDA)

TESSERA_FN u64 load_acquire_sys(const u64* p) {
    u64 v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

TESSERA_FN void store_release_sys(u64* p, u64 v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

TESSERA_FN void store_relaxed_sys(u64* p, u64 v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

TESSERA_FN u64 load_acquire_gpu64(const u64* p) {
    u64 v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

TESSERA_FN void store_release_gpu64(u64* p, u64 v) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

TESSERA_FN u32 load_acquire_gpu(const u32* p) {
    u32 v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

TESSERA_FN void store_relaxed_gpu(u32* p, u32 v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

TESSERA_FN u32 fetch_add_gpu(u32* p, u32 v) {
    u32 old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Loads past every cache: data another agent wrote.
TESSERA_FN float load_fresh(const float* p) {
    return __ldcv(p);
}

TESSERA_FN u16 load_fresh(const u16* p) {
    return __ldcv(p);
}

TESSERA_FN u32 load_fresh(const u32* p) {
    return __ldcv(p);
}

TESSERA_FN float4 load_fresh4(const float* p) {
    float4 v;
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
    return v;
}

TESSERA_FN uint2 load_fresh4(const u16* p) {
    uint2 v;
    asm volatile("ld.global.cv.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

TESSERA_FN void pause() {
    __nanosleep(32);
}

// Nanoseconds on a clock all SMs share.
TESSERA_FN u64 clock_ticks() {
    u64 v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}

#elif defined(TESSERA_HIP)

#define TESSERA_LOAD(name, type, order, scope)                               \
    TESSERA_FN type name(const type* p) {                                    \
        return __hip_atomic_load(p, __ATOMIC_##order, __HIP_MEMORY_SCOPE_##scope); \
    }

#define TESSERA_STORE(name, type, order, scope)                                \
    TESSERA_FN void name(type* p, type v) {                                    \
        __hip_atomic_store(p, v, __ATOMIC_##order, __HIP_MEMORY_SCOPE_##scope); \
    }

TESSERA_LOAD(load_acquire_sys, u64, ACQUIRE, SYSTEM)
TESSERA_STORE(store_release_sys, u64, RELEASE, SYSTEM)
TESSERA_STORE(store_relaxed_sys, u64, RELAXED, SYSTEM)
TESSERA_LOAD(load_acquire_gpu64, u64, ACQUIRE, AGENT)
TESSERA_STORE(store_release_gpu64, u64, RELEASE, AGENT)
TESSERA_LOAD(load_acquire_gpu, u32, ACQUIRE, AGENT)
TESSERA_STORE(store_relaxed_gpu, u32, RELAXED, AGENT)
TESSERA_LOAD(load_fresh, u32, RELAXED, SYSTEM)
TESSERA_LOAD(load_fresh, u16, RELAXED, SYSTEM)

TESSERA_FN u32 fetch_add_gpu(u32* p, u32 v) {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

TESSERA_FN float load_fresh(const float* p) {
    return __uint_as_float(load_fresh(reinterpret_cast<const u32*>(p)));
}

TESSERA_FN float4 load_fresh4(const float* p) {
    return make_float4(load_fresh(p), load_fresh(p + 1), load_fresh(p + 2), load_fresh(p + 3));
}

TESSERA_FN uint2 load_fresh4(const u16* p) {
    u32 lo = u32(load_fresh(p)) | (u32(load_fresh(p + 1)) << 16);
    u32 hi = u32(load_fresh(p + 2)) | (u32(load_fresh(p + 3)) << 16);
    return make_uint2(lo, hi);
}

// Sleeps the whole wave. Use it in leader spins only.
TESSERA_FN void pause() {
    __builtin_amdgcn_s_sleep(1);
}

// Wall clock ticks, at hipDeviceAttributeWallClockRate kHz.
TESSERA_FN u64 clock_ticks() {
    return wall_clock64();
}

#endif

TESSERA_FN u32 cas(u32* p, u32 want, u32 put) {
    return atomicCAS(p, want, put);
}

TESSERA_FN void fence_sys() {
    __threadfence_system();
}

TESSERA_FN float from_bits(u32 bits) {
    return __uint_as_float(bits);
}

// Packs `lo` then `hi` as BF16, ties to even.
TESSERA_FN u32 to_bf16x2(float lo, float hi) {
#if defined(TESSERA_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    u32 d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
#else
    u32 out = 0;
    float v[2] = {lo, hi};
    for (u32 k = 0; k < 2; k++) {
        u32 b = __float_as_uint(v[k]);
        u32 h = (b & 0x7fffffffu) > 0x7f800000u ? (b >> 16) | 0x0040u
                                                 : (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
        out |= (h & 0xffffu) << (16 * k);
    }
    return out;
#endif
}

TESSERA_FN u16 to_bf16(float x) {
    return static_cast<u16>(to_bf16x2(x, 0.0f));
}

// Widens one BF16 value to FP32.
TESSERA_FN float from_bf16(u16 h) {
    return from_bits(u32(h) << 16);
}

// `x` rounded to BF16 and back.
TESSERA_FN float round_bf16(float x) {
    return from_bf16(to_bf16(x));
}

// `x` within the finite FP16 range. NaN stays NaN.
TESSERA_FN float clamp_fp16(float x) {
    return x > 65504.0f ? 65504.0f : (x < -65504.0f ? -65504.0f : x);
}

// Packs `lo` then `hi` as FP16, clamped to finite range.
TESSERA_FN u32 to_fp16x2(float lo, float hi) {
    lo = clamp_fp16(lo);
    hi = clamp_fp16(hi);
#if defined(TESSERA_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    u32 d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
#else
    u32 out = 0;
    float v[2] = {lo, hi};
    for (u32 k = 0; k < 2; k++) {
        u32 b = __float_as_uint(v[k]);
        u32 sign = (b >> 16) & 0x8000u;
        int exp = int((b >> 23) & 0xFFu) - 112;
        u32 mant = b & 0x7FFFFFu;
        u32 h;
        if (exp > 30) {
            // After the clamp, only NaN is this large.
            h = sign | 0x7E00u;
        } else if (exp <= 0) {
            if (exp < -10) {
                h = sign;
            } else {
                mant |= 0x800000u;
                u32 sh = u32(14 - exp);
                u32 rnd = (1u << (sh - 1u)) - 1u + ((mant >> sh) & 1u);
                h = sign | ((mant + rnd) >> sh);
            }
        } else {
            u32 rnd = 0xFFFu + ((mant >> 13) & 1u);
            mant += rnd;
            if (mant & 0x800000u) {
                mant = 0u;
                exp += 1;
            }
            h = sign | (u32(exp) << 10) | (mant >> 13);
        }
        out |= (h & 0xFFFFu) << (16 * k);
    }
    return out;
#endif
}

TESSERA_FN u16 to_fp16(float x) {
    return static_cast<u16>(to_fp16x2(x, 0.0f));
}

// Widens one FP16 value to FP32.
TESSERA_FN float from_fp16(u16 h) {
#if defined(TESSERA_CUDA)
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
#else
    u32 sign = u32(h & 0x8000u) << 16;
    u32 exp = (h >> 10) & 0x1Fu;
    u32 mant = h & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) return from_bits(sign);
        float f = float(mant) * from_bits(0x33800000u);
        return sign ? -f : f;
    }
    if (exp == 31u) return from_bits(sign | 0x7F800000u | (mant << 13));
    return from_bits(sign | ((exp + 112u) << 23) | (mant << 13));
#endif
}

// `x` rounded to FP16 and back.
TESSERA_FN float round_fp16(float x) {
    return from_fp16(to_fp16(x));
}

// Packs one float as OCP E4M3, saturating at 448.
TESSERA_FN u8 to_e4m3(float x) {
    u32 bits = __float_as_uint(x);
    u32 sign = (bits >> 24) & 0x80u;
    u32 mag = bits & 0x7FFFFFFFu;
    if (mag > 0x7F800000u) return static_cast<u8>(sign | 0x7Fu);
    if (mag >= 0x43E00000u) return static_cast<u8>(sign | 0x7Eu);
    if (mag < 0x3C800000u) {
        float s = (from_bits(mag) * 512.0f + 8388608.0f) - 8388608.0f;
        return static_cast<u8>(sign | static_cast<u32>(s));
    }
    u32 nudge = 0x7FFFFu + ((mag >> 20) & 1u);
    u32 r = (mag + nudge) & ~0xFFFFFu;
    if (r >= 0x43E00000u) return static_cast<u8>(sign | 0x7Eu);
    return static_cast<u8>(sign | (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u));
}

// Widens one OCP E4M3 byte to FP32.
TESSERA_FN float from_e4m3(u8 b) {
    u32 sign = u32(b & 0x80u) << 24;
    u32 exp = (b >> 3) & 0xFu;
    u32 man = b & 7u;
    if (exp == 0u) {
        if (man == 0u) return from_bits(sign);
        float f = float(man) * (1.0f / 512.0f);
        return sign ? -f : f;
    }
    if (exp == 15u && man == 7u) return from_bits(sign | 0x7FC00000u);
    return from_bits(sign | ((exp + 120u) << 23) | (man << 20));
}

// Packs four floats into four OCP E4M3 bytes.
TESSERA_FN u32 to_e4m3x4(float4 v) {
#if defined(TESSERA_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    u16 lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(v.y), "f"(v.x));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(v.w), "f"(v.z));
    return u32(lo) | (u32(hi) << 16);
#else
    return u32(to_e4m3(v.x)) | (u32(to_e4m3(v.y)) << 8) | (u32(to_e4m3(v.z)) << 16) |
           (u32(to_e4m3(v.w)) << 24);
#endif
}

// Widens four OCP E4M3 bytes into four floats.
TESSERA_FN float4 from_e4m3x4(u32 p) {
#if defined(TESSERA_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    u32 h0, h1;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(static_cast<u16>(p)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(static_cast<u16>(p >> 16)));
    return make_float4(from_fp16(static_cast<u16>(h0)), from_fp16(static_cast<u16>(h0 >> 16)),
                       from_fp16(static_cast<u16>(h1)), from_fp16(static_cast<u16>(h1 >> 16)));
#else
    return make_float4(from_e4m3(static_cast<u8>(p)), from_e4m3(static_cast<u8>(p >> 8)),
                       from_e4m3(static_cast<u8>(p >> 16)), from_e4m3(static_cast<u8>(p >> 24)));
#endif
}

struct Fp8Blk {
    uint4 q;
    u16 s;
};

TESSERA_FN float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

TESSERA_FN float4 scale4(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

TESSERA_FN float max4_abs(float4 v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

// Rounds `amax / 448` up to normal BF16 without clipping.
TESSERA_FN u16 scale_bf16(float amax) {
    if (amax == 0.0f) return 0u;
    if (!(amax > 0.0f)) return 0x3F80u;
    u32 bits = __float_as_uint(amax / 448.0f);
    if (bits >= 0x7F7F0000u) return 0x7F7Fu;
    u32 up = (bits + 0xFFFFu) >> 16;
    return static_cast<u16>(up < 0x0080u ? 0x0080u : up);
}

// Packs sixteen floats as OCP E4M3 with one BF16 scale.
TESSERA_FN Fp8Blk pack_fp8_blk(float4 v0, float4 v1, float4 v2, float4 v3) {
    float amax = fmaxf(fmaxf(max4_abs(v0), max4_abs(v1)), fmaxf(max4_abs(v2), max4_abs(v3)));
    u16 s = scale_bf16(amax);
    float inv = (s == 0u) ? 0.0f : (1.0f / from_bf16(s));
    uint4 q = make_uint4(to_e4m3x4(scale4(v0, inv)), to_e4m3x4(scale4(v1, inv)),
                         to_e4m3x4(scale4(v2, inv)), to_e4m3x4(scale4(v3, inv)));
    return {q, s};
}

// Unpacks sixteen OCP E4M3 values scaled by one BF16 scale.
TESSERA_FN void unpack_fp8_blk(Fp8Blk blk, float4& v0, float4& v1, float4& v2, float4& v3) {
    float s = from_bf16(blk.s);
    v0 = scale4(from_e4m3x4(blk.q.x), s);
    v1 = scale4(from_e4m3x4(blk.q.y), s);
    v2 = scale4(from_e4m3x4(blk.q.z), s);
    v3 = scale4(from_e4m3x4(blk.q.w), s);
}

TESSERA_FN void block_sync() {
    __syncthreads();
}

TESSERA_FN bool leader_thread() {
    return (threadIdx.x | threadIdx.y | threadIdx.z) == 0;
}

TESSERA_FN bool leader_block() {
    return (blockIdx.x | blockIdx.y | blockIdx.z) == 0;
}

TESSERA_FN bool leader() {
    return leader_thread() && leader_block();
}

TESSERA_FN u32 blocks() {
    return gridDim.x * gridDim.y * gridDim.z;
}

}  // namespace prim
}  // namespace tessera

#endif  // TESSERA_BACKEND_H
