// probe_kernel.cu — minimal kernels for toolchain/profiler validation.
//   vecadd   : light, memory-bound (low register pressure)
//   reg_heavy: high register pressure (stresses the P_occ path)
// Built with MetaX `cucc` (CUDA-compatible driver) or NVIDIA `nvcc`.
#include <cstdio>

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void reg_heavy(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) acc[k] = in[(i + k) % n] * (k + 1);
#pragma unroll
    for (int k = 0; k < 32; ++k) acc[k] = acc[k] * acc[(k + 7) % 32] + acc[(k + 13) % 32];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) s += acc[k];
    if (i < n) out[i] = s;
}

int main() {
    const int n = 1 << 20;
    size_t sz = n * sizeof(float);
    float *a, *b, *c;
    cudaMalloc(&a, sz); cudaMalloc(&b, sz); cudaMalloc(&c, sz);
    dim3 block(256), grid((n + 255) / 256);
    vecadd<<<grid, block>>>(a, b, c, n);
    reg_heavy<<<grid, block>>>(a, c, n);
    cudaError_t e = cudaDeviceSynchronize();
    printf("kernels launched, sync=%s\n", cudaGetErrorString(e));
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
