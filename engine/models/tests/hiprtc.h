// hiprtc names that clang's HIP wrapper does not declare.
#pragma once
typedef unsigned int uint;
#define __forceinline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))
#define __shared__ __attribute__((shared))

static __device__ inline float __uint_as_float(uint x) { return __builtin_bit_cast(float, x); }
static __device__ inline uint __float_as_uint(float x) { return __builtin_bit_cast(uint, x); }

#define TS_ARGS2 x, y
#define TS_ARGS4 x, y, z, w
#define TS_VEC(t, n, ...)                                             \
    typedef t t##n __attribute__((ext_vector_type(n)));               \
    static __device__ inline t##n make_##t##n(__VA_ARGS__) {          \
        return (t##n){TS_ARGS##n};                                     \
    }
TS_VEC(float, 2, float x, float y)
TS_VEC(float, 4, float x, float y, float z, float w)
TS_VEC(uint, 2, uint x, uint y)
TS_VEC(uint, 4, uint x, uint y, uint z, uint w)
TS_VEC(int, 2, int x, int y)
TS_VEC(int, 4, int x, int y, int z, int w)

struct ts_dim3 {
    uint x, y, z;
};
#define TS_DIM3(f) (ts_dim3{f##_x(), f##_y(), f##_z()})
#define threadIdx TS_DIM3(__builtin_amdgcn_workitem_id)
#define blockIdx TS_DIM3(__builtin_amdgcn_workgroup_id)
#define blockDim TS_DIM3(__builtin_amdgcn_workgroup_size)

static __device__ inline void __syncthreads() { __builtin_amdgcn_s_barrier(); }

template <class T>
static __device__ inline T __shfl_xor(T v, int m) {
    int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    int moved = __builtin_amdgcn_ds_bpermute((lane ^ m) << 2, __builtin_bit_cast(int, v));
    return __builtin_bit_cast(T, moved);
}
