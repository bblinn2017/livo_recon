#pragma once
// R74 (F-410): the joint IMU prior of the coupled estimator (a 54 or 114 dimension covariance that is built and inverted) is needed only to SOLVE.
// A scan with no LiDAR residual has no H^T H and no H^T z; building and inverting the prior then adds nothing and can throw when the prior is
// numerically singular (the three R72 aborts, scans 1131, 1302, 2401, each after a run of scans with zero residuals). This helper is the one rule,
// ROS-free so tests/test_joint_knot_lazy_prior.cpp can check it: build only when there is at least one residual and it is not built yet.
// Returns true when the prior is available after the call.
#include <cstddef>

namespace livo_recon
{
template <class BuildFn>
inline bool buildPriorIfResiduals(std::size_t n_residuals, bool already_built, BuildFn&& build)
{
  if (already_built) return true;
  if (n_residuals == 0) return false;
  build();
  return true;
}
}  // namespace livo_recon
