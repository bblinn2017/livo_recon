#pragma once

#include "livo_recon/utils/algo/math.h"

#include <Eigen/Dense>
#include <array>
#include <string>

namespace livo_recon
{

// One phase's (post_imu or post_lio) full covariance-consistency evaluation
// against the fixed stationary reference. All quantities are computed from
// the estimator's right/body-frame perturbation convention -- see
// computeInitConsistency()'s own derivation comment.
struct InitConsistencyResult
{
  // Errors
  V3D eR = V3D::Zero(), ep = V3D::Zero(), ev = V3D::Zero();
  double eR_norm = 0.0, ep_norm = 0.0, ev_norm = 0.0;
  V3D tilt = V3D::Zero();
  double tilt_norm = 0.0;
  double yaw_error = 0.0;
  double exact_gravity_tilt = 0.0;

  // Covariance summaries
  double Prr_trace = 0.0; std::array<double, 3> Prr_eig{};
  double Ppp_trace = 0.0; std::array<double, 3> Ppp_eig{};
  double Pvv_trace = 0.0; std::array<double, 3> Pvv_eig{};
  double Prp_fro = 0.0, Prv_fro = 0.0, Ppv_fro = 0.0;
  std::array<double, 2> tilt2d_eig{};
  double yaw_variance = 0.0;
  double Prpv_min_eig = 0.0, Prpv_max_eig = 0.0, Prpv_condition = 0.0;

  // Consistency (NEES) values, via symmetric LDLT -- never an explicit
  // inverse. `*_valid` is false (and the corresponding epsilon left at 0)
  // whenever the relevant covariance block is not numerically positive
  // definite; the block's own eigenvalues above remain populated either way
  // so a non-PD block is diagnosable, not silently hidden.
  double eps_R = 0.0, eps_p = 0.0, eps_v = 0.0, eps_tilt = 0.0, eps_yaw = 0.0, eps_RPV = 0.0;
  bool rr_solve_valid = false, pp_solve_valid = false, vv_solve_valid = false;
  bool tilt_solve_valid = false, rpv_solve_valid = false;
  double min_ldlt_pivot = 0.0;  // worst (smallest signed) pivot across the 5 solves

  // 68%/95%/99% chi-square bound flags (true = statistic falls at/under the
  // bound for its DOF), one triple per statistic. false whenever the
  // corresponding solve is invalid.
  std::array<bool, 3> chi2_R{}, chi2_p{}, chi2_v{}, chi2_tilt{}, chi2_yaw{}, chi2_RPV{};
};

// Computes every quantity in InitConsistencyResult for one phase.
// `P_RPV` must be the 9x9 block of the state covariance in [rot,pos,vel]
// order (StateGroup::idxR()/idxP()/idxV() are contiguous, so this is simply
// state.cov().block<9,9>(StateGroup::idxR(), StateGroup::idxR())).
InitConsistencyResult computeInitConsistency(
    const M3D& R_ref, const V3D& p_ref, const V3D& v_ref,
    const M3D& R, const V3D& p, const V3D& v,
    const Eigen::MatrixXd& P_RPV);

// Writes exactly two rows (phase="post_imu", phase="post_lio") to
// initialization_consistency_all_scans.csv for one scan. `post_lio_valid`
// distinguishes "the LIO stage ran and produced a solved post_covariance
// state" (any_solved) from "no usable residuals, no solve exists" (the
// no_residuals stop reason) -- in the latter case the post_lio row is still
// written (from the same, unmodified, propagated state) but
// `completed_iterations=0` and the row's own values are identical to
// post_imu's, never described as a converged update.
void writeInitializationConsistencyDiagnostics(
    const std::string& run_id, int scan_id, double t_abs,
    const std::string& p0_mode, const Eigen::MatrixXd& P0_actual,
    double configured_init_pos, double configured_init_vel,
    double configured_init_rot_tilt, double configured_init_rot_yaw,
    double configured_init_gravity, double configured_init_bg,
    double configured_init_ba,
    int residual_count, int completed_iterations,
    const M3D& R_ref, const V3D& p_ref, const V3D& v_ref,
    const M3D& R_post_imu, const V3D& p_post_imu, const V3D& v_post_imu,
    const Eigen::MatrixXd& P_RPV_post_imu,
    const M3D& R_post_lio, const V3D& p_post_lio, const V3D& v_post_lio,
    const Eigen::MatrixXd& P_RPV_post_lio);

}  // namespace livo_recon
