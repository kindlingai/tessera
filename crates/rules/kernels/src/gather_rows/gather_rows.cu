// Gather rows by index. Mirrors gather_rows.metal.

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(unsigned short v) { return __uint_as_float((unsigned)v << 16); }

template <typename T>
__device__ __forceinline__ void gather_rows_t(
    float *out,
    const T *src,
    const unsigned *indices,
    unsigned num_indices,
    unsigned hidden
) {
    unsigned t = blockIdx.x;
    if (t >= num_indices) {
        return;
    }
    unsigned row = indices[t];
    const T *s = src + (size_t)row * hidden;
    float *d = out + (size_t)t * hidden;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        d[i] = widen(s[i]);
    }
}

extern "C" __global__ void gather_rows(
    float *out, const float *src, const unsigned *indices, unsigned num_indices, unsigned hidden
) {
    gather_rows_t<float>(out, src, indices, num_indices, hidden);
}

extern "C" __global__ void gather_rows_bf16(
    float *out, const unsigned short *src, const unsigned *indices, unsigned num_indices, unsigned hidden
) {
    gather_rows_t<unsigned short>(out, src, indices, num_indices, hidden);
}
