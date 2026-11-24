// Helpers for a whole step in one persistent grid.

#ifndef TESSERA_CAP_WHOLE
#error "import <tessera/1/whole>"
#endif

#ifndef TESSERA_WHOLE_H
#define TESSERA_WHOLE_H

namespace tessera {

// The launch values, in order, as the fields of `S`.
template <typename S>
TESSERA_FN S values(const table* t) {
    static_assert(sizeof(S) % 4 == 0, "values are 32-bit words");
    S s;
    u32* w = reinterpret_cast<u32*>(&s);
    for (u32 i = 0; i < sizeof(S) / 4; i++) w[i] = value(t, i);
    return s;
}

// The kernel's scratch, its last item.
template <typename T>
TESSERA_FN T* scratch(const table* t) {
    return at<T>(t, t->items_len - 1);
}

// Every thread of the grid, as one line.
struct line {
    u32 at;
    u32 step;
};

TESSERA_FN line threads() {
    return {blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x};
}

// A grid barrier after this thread's writes.
TESSERA_FN bool sync(const table* t) {
    __threadfence();
    return grid_sync(t);
}

#if defined(TESSERA_CUDA)

// Warps of 32 lanes. Every lane gets the sum.
TESSERA_FN float warp_sum(float v) {
#pragma unroll
    for (u32 off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

// Block sum, fixed order. `red` holds one float per warp.
TESSERA_FN float block_sum(float v, float* red) {
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0u) red[warp] = v;
    __syncthreads();
    u32 nw = blockDim.x >> 5;
    float sum = 0.0f;
    for (u32 w = 0; w < nw; w++) sum += red[w];
    __syncthreads();
    return sum;
}

#elif defined(TESSERA_HIP)

// Waves of `warpSize` lanes. Every lane gets the sum.
TESSERA_FN float warp_sum(float v) {
    for (u32 off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

// Block sum, fixed order. `red` holds one float per wave.
TESSERA_FN float block_sum(float v, float* red) {
    u32 lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    v = warp_sum(v);
    if (lane == 0u) red[wave] = v;
    __syncthreads();
    u32 nw = blockDim.x / warpSize;
    float sum = 0.0f;
    for (u32 w = 0; w < nw; w++) sum += red[w];
    __syncthreads();
    return sum;
}

#endif

}  // namespace tessera

#endif  // TESSERA_WHOLE_H
