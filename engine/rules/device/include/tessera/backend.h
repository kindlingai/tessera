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
