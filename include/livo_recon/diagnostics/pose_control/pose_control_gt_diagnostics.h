#pragma once

#include "livo_recon/utils/algo/math.h"
#include <Eigen/Dense>

// ============================================================================
// 2026-09-24 pre-real-data-campaign instrumentation, items 7/24: ground-
// truth error and covariance-calibration (NEES) diagnostics for the
// pose-control estimator.
//
// SCOPE / WHY THIS IS NOT WIRED TO A LIVE GT FEED THIS PASS. The only
// existing ground-truth channel in this codebase is
// DataQueues::popGt()/CbkProc::gtCallback() -> EvoProc's gt_buffer_ (see
// evo_processing.h) -- a CONSUMING queue that EvoProc itself drains to
// produce the estimator's own authoritative ATE/RPE scoring. Reading
// (popping) from that same queue inside LioProcCoupled's diagnostic code
// would risk starving EvoProc of samples it needs for real scoring --
// exactly the kind of change item 45 asks NOT to make without being able
// to verify it end-to-end, and pose-control real-data runs (the only way
// to verify a live GT-queue integration doesn't perturb the ATE pipeline)
// are explicitly forbidden this phase. A non-consuming "peek" API does not
// currently exist on DataQueues.
//
// So: per this phase's own explicit instruction ("if real data is required
// only to evaluate the final estimator behavior: implement and statically/
// synthetically validate the machinery now; defer ONLY the real-data
// execution"), this file provides the CALCULATION machinery --
// computeGtError() and computeNees() -- as pure, generic, synthetically-
// tested functions, ready to be called the moment a non-destructive GT
// peek exists and pose-control real-data experiments are authorized. The
// call site in lio_coupled.cpp's post-loop block documents exactly where
// this would plug in.
// ============================================================================

namespace livo_recon
{

struct PoseControlGtError
{
  V3D e_p = V3D::Zero();   // p_estimate - p_gt
  V3D e_R = V3D::Zero();   // Log(R_gt^T * R_estimate) -- body-frame rotation error
  V3D e_v = V3D::Zero();   // v_estimate - v_gt
};

// e_R uses the SAME Log(R_gt^T*R_est) convention already used by the x1/
// knot IMU-chain-vs-final comparisons elsewhere in lio_coupled.cpp (see
// x1_covariance/knot_prior_posterior rows) -- consistent sign/frame
// convention throughout this file's diagnostics.
PoseControlGtError computeGtError(
    const V3D& p_est, const M3D& R_est, const V3D& v_est,
    const V3D& p_gt, const M3D& R_gt, const V3D& v_gt);

struct PoseControlNees
{
  double nees_p = 0.0;    // e_p^T * pinv(P_pp) * e_p / 3   (per-DOF normalized, chi-square/3 under correct calibration)
  double nees_R = 0.0;
  double nees_v = 0.0;
  double nees_rpv = 0.0;  // joint 9-dim NEES using the FULL 9x9 P (with cross-covariance), not the sum of the three marginals
};

// P_9 is the full 9x9 [R,p,v] covariance block (NOT block-diagonal --
// includes R/p/v cross-covariance, per item 24's own "do not equate
// covariance contraction with calibration" -- an accurate joint NEES needs
// the true joint covariance, not three independent marginal ones).
// generalPseudoInverse-style rel_thresh, matching this file's convention
// elsewhere, rather than a plain .inverse() that could fail on a
// near-singular block.
PoseControlNees computeNees(
    const PoseControlGtError& err, const Eigen::Matrix<double, 9, 9>& P_9, double rel_thresh = 1e-9);

}  // namespace livo_recon
