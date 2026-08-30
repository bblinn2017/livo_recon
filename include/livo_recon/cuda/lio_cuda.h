#pragma once
#include <cuda_runtime.h>

namespace livo_recon {

struct LioCudaBuffers
{
  // Pinned host arrays (enable zero-copy async transfer)
  float3* h_normals  = nullptr;  // plane normals
  float3* h_crosses  = nullptr;  // pt × normal (point_cross_normal)
  float*  h_sigma_squared = nullptr;  // measurement variances
  float*  h_rs       = nullptr;  // signed plane distances

  // Device mirrors
  float3* d_normals  = nullptr;
  float3* d_crosses  = nullptr;
  float*  d_sigma_squared = nullptr;
  float*  d_rs       = nullptr;

  // Per-frame output accumulators (device); zeroed each call by runLioAccumulation
  double* d_HtH      = nullptr;  // 6×6, row-major
  double* d_Htz      = nullptr;  // 6×1

  int          capacity = 0;
  cudaStream_t stream   = nullptr;

  // init() creates the stream and the fixed-size output buffers.
  // ensure(n) grows the per-residual data buffers; no-op if capacity >= n.
  // destroy() frees everything.
  void init();
  void ensure(int n);
  void destroy();
  ~LioCudaBuffers() { destroy(); }
};

// Caller must populate h_normals/h_crosses/h_sigma_squared/h_rs[0..n) before calling.
// Transfers data to device, runs H^T*W*H reduction kernel, copies results back.
// HtH_out and Htz_out must be caller-allocated arrays of size 36 and 6 respectively.
void runLioAccumulation(LioCudaBuffers& buf, int n,
                        double* HtH_out,
                        double* Htz_out);

}  // namespace livo_recon
