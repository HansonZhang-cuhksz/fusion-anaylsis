// layout.cu -- raw-CUDA P_layout study: transpose->relu with a shared-tile bank-conflict knob.
// Shared by (a) nvcc -Xptxas -v  for single-compile static registers, and
//           (b) torch load_inline for the runnable kernels.
#include <cstdio>

// fused transpose + relu epilogue; PAD toggles the shared-tile bank conflict (0 -> 32-way conflict).
template <int PAD>
__global__ void txpose_relu(const float* __restrict__ x, float* __restrict__ y,
                            int R, int C, float s, float b) {
  __shared__ float tile[32][32 + PAD];
  int bx = blockIdx.x * 32, by = blockIdx.y * 32;
  int r = by + threadIdx.y, c = bx + threadIdx.x;
  if (r < R && c < C) tile[threadIdx.y][threadIdx.x] = x[r * (long)C + c];
  __syncthreads();
  int cr = bx + threadIdx.y, rr = by + threadIdx.x;                 // transposed indices
  if (cr < C && rr < R) {
    float v = tile[threadIdx.x][threadIdx.y];                       // PAD=0 -> bank conflict
    y[cr * (long)R + rr] = fmaxf(s * v + b, 0.f);
  }
}

// clean (padded, conflict-free) transpose only -> writes X^T
__global__ void txpose_only(const float* __restrict__ x, float* __restrict__ xt, int R, int C) {
  __shared__ float tile[32][33];
  int bx = blockIdx.x * 32, by = blockIdx.y * 32;
  int r = by + threadIdx.y, c = bx + threadIdx.x;
  if (r < R && c < C) tile[threadIdx.y][threadIdx.x] = x[r * (long)C + c];
  __syncthreads();
  int cr = bx + threadIdx.y, rr = by + threadIdx.x;
  if (cr < C && rr < R) xt[cr * (long)R + rr] = tile[threadIdx.x][threadIdx.y];
}

__global__ void relu_ep(const float* __restrict__ xt, float* __restrict__ y, long n, float s, float b) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(s * xt[i] + b, 0.f);
}

// force instantiation so ptxas -v reports both variants
template __global__ void txpose_relu<0>(const float*, float*, int, int, float, float);
template __global__ void txpose_relu<1>(const float*, float*, int, int, float, float);
