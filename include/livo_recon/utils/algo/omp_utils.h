#pragma once

#include <algorithm>
#include <cstdlib>
#include <omp.h>

namespace livo_recon
{

// Per-frame OMP loops here handle a few thousand to ~10k-20k items (LiDAR
// residuals, voxel points, VIO correspondences) -- on a high-core-count
// machine, naively parallelizing across omp_get_max_threads() threads makes
// fork/join/scheduling overhead dominate the actual (small, sub-millisecond
// per thread) work, and any per-thread storage costs scale with thread
// count too. 16 is a rough sweet spot for the item counts seen so far, not
// a tuned constant -- revisit if per-frame item counts grow much further.
// See lio_processing.cpp's solveSystem() woodbury-correction comment for
// the regression this was found fixing.
inline int cappedOmpThreads()
{
  return std::min(omp_get_max_threads(), 16);
}

// History (24-55): see docs/livo_recon_changelog.md#include-livo_recon-utils-algo-omp_utils.h-24
inline void pinOmpThreadsForDeterminism()
{
  const char* env = std::getenv("OMP_NUM_THREADS");
  const int n = (env && *env) ? std::atoi(env) : 16;
  omp_set_num_threads(n > 0 ? n : 16);
}

}  // namespace livo_recon
