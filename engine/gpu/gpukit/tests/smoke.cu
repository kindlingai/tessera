extern "C" __global__ void saxpy(float *y, const float *x, float a, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

extern "C" __global__ void high_shared(float *out, unsigned index) {
    extern __shared__ float buf[];
    if (threadIdx.x == 0) {
        buf[index] = 42.0f;
        *out = buf[index];
    }
}
