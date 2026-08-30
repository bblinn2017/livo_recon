#include "livo_recon/cuda/lio_cuda.h"

namespace livo_recon {

// One thread per residual.
// Accumulates the 6×6 H^T*W*H matrix and 6×1 H^T*W*r vector using shared-memory
// atomics within each block, then reduces block results into global memory.
__global__ void accumulateHtWH_kernel(
    const float3* __restrict__ normals,
    const float3* __restrict__ crosses,
    const float*  __restrict__ sigma_squared,
    const float*  __restrict__ rs,
    double* __restrict__ d_HtH,
    double* __restrict__ d_Htz,
    int n)
{
  __shared__ double s_HtH[36];
  __shared__ double s_Htz[6];

  if (threadIdx.x < 36) s_HtH[threadIdx.x] = 0.0;
  if (threadIdx.x <  6) s_Htz[threadIdx.x] = 0.0;
  __syncthreads();

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    const double w  = 1.0 / static_cast<double>(sigma_squared[i]);
    const double wr = w  * static_cast<double>(rs[i]);

    const double hr[3] = { crosses[i].x, crosses[i].y, crosses[i].z };
    const double hp[3] = { normals[i].x, normals[i].y, normals[i].z };

    // Four 3×3 outer-product blocks filling the symmetric 6×6 HtH
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        atomicAdd(&s_HtH[(a    ) * 6 + (b    )], w * hr[a] * hr[b]);  // [0:3, 0:3]
        atomicAdd(&s_HtH[(a    ) * 6 + (b + 3)], w * hr[a] * hp[b]);  // [0:3, 3:6]
        atomicAdd(&s_HtH[(a + 3) * 6 + (b    )], w * hp[a] * hr[b]);  // [3:6, 0:3]
        atomicAdd(&s_HtH[(a + 3) * 6 + (b + 3)], w * hp[a] * hp[b]);  // [3:6, 3:6]
      }
      atomicAdd(&s_Htz[a    ], wr * hr[a]);
      atomicAdd(&s_Htz[a + 3], wr * hp[a]);
    }
  }
  __syncthreads();

  if (threadIdx.x < 36) atomicAdd(&d_HtH[threadIdx.x], s_HtH[threadIdx.x]);
  if (threadIdx.x <  6) atomicAdd(&d_Htz[threadIdx.x], s_Htz[threadIdx.x]);
}

// ─── Buffer lifecycle ─────────────────────────────────────────────────────────

void LioCudaBuffers::init()
{
  cudaStreamCreate(&stream);
  cudaMalloc(&d_HtH, 36 * sizeof(double));
  cudaMalloc(&d_Htz,  6 * sizeof(double));
}

void LioCudaBuffers::ensure(int n)
{
  if (!stream) init();
  if (n <= capacity) return;

  if (h_normals) { cudaFreeHost(h_normals); h_normals = nullptr; }
  if (h_crosses) { cudaFreeHost(h_crosses); h_crosses = nullptr; }
  if (h_sigma_squared)  { cudaFreeHost(h_sigma_squared);  h_sigma_squared  = nullptr; }
  if (h_rs)      { cudaFreeHost(h_rs);      h_rs      = nullptr; }
  if (d_normals) { cudaFree(d_normals);     d_normals = nullptr; }
  if (d_crosses) { cudaFree(d_crosses);     d_crosses = nullptr; }
  if (d_sigma_squared)  { cudaFree(d_sigma_squared);      d_sigma_squared  = nullptr; }
  if (d_rs)      { cudaFree(d_rs);          d_rs      = nullptr; }

  capacity = n;
  cudaMallocHost(&h_normals, n * sizeof(float3));
  cudaMallocHost(&h_crosses, n * sizeof(float3));
  cudaMallocHost(&h_sigma_squared,  n * sizeof(float));
  cudaMallocHost(&h_rs,      n * sizeof(float));
  cudaMalloc(&d_normals, n * sizeof(float3));
  cudaMalloc(&d_crosses, n * sizeof(float3));
  cudaMalloc(&d_sigma_squared,  n * sizeof(float));
  cudaMalloc(&d_rs,      n * sizeof(float));
}

void LioCudaBuffers::destroy()
{
  if (h_normals) { cudaFreeHost(h_normals); h_normals = nullptr; }
  if (h_crosses) { cudaFreeHost(h_crosses); h_crosses = nullptr; }
  if (h_sigma_squared)  { cudaFreeHost(h_sigma_squared);  h_sigma_squared  = nullptr; }
  if (h_rs)      { cudaFreeHost(h_rs);      h_rs      = nullptr; }
  if (d_normals) { cudaFree(d_normals);     d_normals = nullptr; }
  if (d_crosses) { cudaFree(d_crosses);     d_crosses = nullptr; }
  if (d_sigma_squared)  { cudaFree(d_sigma_squared);      d_sigma_squared  = nullptr; }
  if (d_rs)      { cudaFree(d_rs);          d_rs      = nullptr; }
  if (d_HtH)    { cudaFree(d_HtH);         d_HtH     = nullptr; }
  if (d_Htz)    { cudaFree(d_Htz);          d_Htz     = nullptr; }
  if (stream)   { cudaStreamDestroy(stream); stream   = nullptr; }
  capacity = 0;
}

// ─── Host-side launcher ───────────────────────────────────────────────────────

void runLioAccumulation(LioCudaBuffers& buf, int n,
                        double* HtH_out, double* Htz_out)
{
  cudaMemsetAsync(buf.d_HtH, 0, 36 * sizeof(double), buf.stream);
  cudaMemsetAsync(buf.d_Htz, 0,  6 * sizeof(double), buf.stream);

  cudaMemcpyAsync(buf.d_normals, buf.h_normals, n * sizeof(float3), cudaMemcpyHostToDevice, buf.stream);
  cudaMemcpyAsync(buf.d_crosses, buf.h_crosses, n * sizeof(float3), cudaMemcpyHostToDevice, buf.stream);
  cudaMemcpyAsync(buf.d_sigma_squared,  buf.h_sigma_squared,  n * sizeof(float),  cudaMemcpyHostToDevice, buf.stream);
  cudaMemcpyAsync(buf.d_rs,      buf.h_rs,      n * sizeof(float),  cudaMemcpyHostToDevice, buf.stream);

  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  accumulateHtWH_kernel<<<blocks, kThreads, 0, buf.stream>>>(
      buf.d_normals, buf.d_crosses, buf.d_sigma_squared, buf.d_rs,
      buf.d_HtH, buf.d_Htz, n);

  cudaMemcpyAsync(HtH_out, buf.d_HtH, 36 * sizeof(double), cudaMemcpyDeviceToHost, buf.stream);
  cudaMemcpyAsync(Htz_out, buf.d_Htz,  6 * sizeof(double), cudaMemcpyDeviceToHost, buf.stream);
  cudaStreamSynchronize(buf.stream);
}

}  // namespace livo_recon
