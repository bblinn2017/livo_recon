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

// Call ONCE, as the very first statement in main() -- before ros::init(),
// before constructing anything that might touch LibTorch/AOTInductor (the
// CoTracker backend's model loading, even in cache-replay mode where the
// model is loaded but never invoked) -- to fix a confirmed run-to-run
// non-determinism (2026-08-16): with OMP_NUM_THREADS left unset,
// omp_get_max_threads() (and therefore cappedOmpThreads(), and therefore
// the actual thread count a "#pragma omp parallel for" runs with) was
// observed to vary BETWEEN otherwise-identical runs of the same
// offline+cache-replay config on the same host -- most likely LibTorch's
// own OpenMP thread-pool initialization racing with or overriding the
// effective global thread count before this codebase's own parallel
// regions ever run.
//
// Two independent fixes are combined here, deliberately:
// 1. VioAccumulator::accumulate() and accumulateLioResiduals() (2026-08-16)
//    were changed to reduce HtH/Htz/err in fixed POINT/residual-index order
//    (not per-thread-partial-sum order) -- this makes their own numeric
//    result independent of thread count entirely, confirmed: reruns at
//    OMP_NUM_THREADS=4 vs 16 now produce byte-identical HtH/Htz for the
//    same frame. This is the real fix for the two call sites that matter
//    most (the ones feeding the EKF solve).
// 2. This function additionally PINS the thread count itself, as defense
//    in depth for any other OMP-parallel region in this codebase that
//    hasn't been (or in the future isn't) given the same point-indexed
//    treatment -- an unset OMP_NUM_THREADS otherwise leaves the actual
//    thread count to whatever LibTorch/the OS picked at that moment, which
//    is its own source of nondeterminism even where the reduction order no
//    longer matters (e.g. wall-clock-timing-sensitive code, if any exists).
//    Respects OMP_NUM_THREADS if the environment explicitly sets it (so a
//    deliberate override, e.g. for a benchmark or this fix's own
//    verification testing, still works); defaults to 16 (matching
//    cappedOmpThreads()'s own cap) otherwise.
inline void pinOmpThreadsForDeterminism()
{
  const char* env = std::getenv("OMP_NUM_THREADS");
  const int n = (env && *env) ? std::atoi(env) : 16;
  omp_set_num_threads(n > 0 ? n : 16);
}

}  // namespace livo_recon
