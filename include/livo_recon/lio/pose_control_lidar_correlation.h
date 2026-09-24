#pragma once

#include "livo_recon/lio/pose_control_lidar_factor.h"
#include "livo_recon/lio/residual_redundancy.h"  // reuses ResidualRedundancyOptions/Stats -- generic, not decoupled-specific

// ============================================================================
// process-prior-REFORMULATION phase, items 24/25: correlated-noise (Woodbury)
// correction for pose_control's LiDAR information, adapted from
// residual_redundancy.h's exact algebra (group residuals sharing a plane's
// plane_var_term, Sherman-Morrison-marginalize their joint contribution)
// but operating on PoseControlLidarRecord's REDUCED z-space Jacobian rows
// instead of the decoupled arm's own state-space Jacobian. Off by default
// (mode=="off" reproduces the naive per-point A += w*J*J^T sum exactly, to
// the ULP -- addPoseControlLidarFactor's own per-point accumulation is
// UNCHANGED regardless; this function only ever SUBTRACTS a correction on
// top when mode != "off").
//
// Sigma_group = D + rho*plane_var_term*ones(k,k), D_ii = sigma_i^2 -
// rho*plane_var_term (independent-only variance). By Sherman-Morrison,
// Sigma_group^-1 = D^-1 - c*(D^-1 1)(D^-1 1)^T, c = rho*pv/(1+rho*pv*sum(1/D_ii)),
// so the group's correct joint information is
//   sum_i w_i J_i J_i^T - c*(sum_i w_i J_i)(sum_i w_i J_i)^T
// i.e. a RANK-ONE DOWNDATE of the naive sum already in A/b.
// ============================================================================

namespace livo_recon
{

// Groups `records` by plane_id; for every group with >= 2 members, computes
// and SUBTRACTS the Woodbury downdate from A/b (which must already contain
// the naive per-point sum these records came from -- call this AFTER
// addPoseControlLidarFactor(..., &records)). Returns the same kind of
// engagement stats residual_redundancy.h's own function does. mode=="off"
// computes stats but does not mutate A/b (matches residual_redundancy.h's
// own "log beside, do not drive" convention).
ResidualRedundancyStats applyPoseControlLidarCorrelationCorrection(
    const std::vector<PoseControlLidarRecord>& records,
    const ResidualRedundancyOptions& opts,
    Eigen::MatrixXd& A, Eigen::VectorXd& b);

}  // namespace livo_recon
