#pragma once

#include <Eigen/Dense>

// ============================================================================
// Information-form Schur-complement marginalization -- the mechanism that
// reconciles two requirements from the 2026-09-22 follow-up that look like
// they contradict each other but don't:
//   "the head mean never changes" (no head GN variable at all -- see
//   solveHeadControlPoints() in pose_control_spline.h)
// and
//   "incoming head covariance P0 must still influence the posterior
//   uncertainty of the tail... do NOT turn 'fixed mean' into 'zero
//   uncertainty'."
//
// Marginal COVARIANCE of a jointly-Gaussian posterior is a property of the
// JOINT INFORMATION MATRIX's block structure alone -- it does not depend on
// the gradient/RHS (which determines the MEAN). So the head can be held at
// a fixed mean throughout the GN mean-solve (no head columns ever appear in
// the system that's actually solved for delta_c/delta_sT) while STILL
// getting the mathematically correct marginal covariance for the free
// variables that accounts for head uncertainty, via:
//
//   1. Build the JOINT information matrix over [head(9); free(n_free)]:
//        A_hh = Omega0 (=pinv(P0)) + (head-block Hessian contribution from
//               every factor touching the head, e.g. segment 0's process
//               factor via its Jxj columns restricted to cp[0..2]/
//               cp_phi[0..2] -- these columns are NEVER part of the actual
//               GN solve, only collected here for the covariance step)
//        A_hf = the head-free CROSS block from those same factors
//        A_ff = the ordinary free-variable information matrix already
//               built and used for the mean solve (LiDAR + process factor
//               + tail-state prior, restricted to free columns only)
//   2. Schur-complement OUT the head block:
//        P_free = (A_ff - A_hf^T * A_hh^-1 * A_hf)^-1
//      This IS the marginal covariance of the free variables in the joint
//      posterior -- exactly as if the head had been a free variable with
//      prior Omega0 and then marginalized, which is the textbook-correct
//      way to propagate a fixed-but-uncertain quantity's uncertainty
//      forward without ever letting its own mean move.
// ============================================================================

namespace livo_recon
{

// Returns false (leaves P_free untouched) if A_hh is not invertible via
// LDLT or the resulting Schur complement is not PD/invertible.
bool schurComplementFreeCovariance(
    const Eigen::MatrixXd& A_hh, const Eigen::MatrixXd& A_hf,
    const Eigen::MatrixXd& A_ff, Eigen::MatrixXd& P_free);

}  // namespace livo_recon
