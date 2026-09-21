#pragma once

#include "livo_recon/lio/spline.h"

// ============================================================================
// CQ-82 Phase 2, Artifacts 1/3: the pose-basis coefficient-block builder.
//
// STANDALONE BY DESIGN. This is not yet wired into estimateCoupledCorrection()
// -- the shared dispatcher's state persistence (coupled_c_acc_/coupled_c_gyr_
// and their many raw_imu-specific touch points: bg-projection, delta
// application, effective-config reporting) has raw_imu-specific meaning
// throughout, and safely branching a genuinely different basis through ~600
// lines of that function without risking Phase 1's byte-identical guarantee
// is a separate, larger integration task. This file gives CQ-82's own
// Artifact 3 (item 6's correctness gate) a real, testable implementation of
// the pose basis's own normal-equations construction, verifiable in complete
// isolation (see scripts/test/test_pose_spline_cblock.cpp) -- zero risk to
// any shipped behavior, since nothing here is called from the live node.
//
// LiDAR term (linear in c_p, chain rule in c_phi via Jr(phi)):
//   r_i = n_i^T ( R(t_i) q_i + p(t_i) ) + d_i
//   dr_i/dc_p[j]   = b_j(t_i) * n_i^T
//   dr_i/dc_phi[j] = ( dr_i/dtheta_i ) * Jr(phi_i) * b_j(t_i)
//   dr_i/dtheta_i, the rotation-jacobian row, reuses the SAME convention this
//   codebase's own point_time residual already uses (raw_body_point.cross(
//   R(t)^T * normal) -- verified right-multiplicative, CQ-71's own
//   delta_phi0->dtheta ratio of exactly 1.000000) -- matched, not re-derived.
//
// IMU-as-measurement-factor (per raw IMU sample k, NOT a prior):
//   r_a,k     = R(t_k)^T ( pddot(t_k) - gravity ) + bias_acc - acc_k
//   r_omega,k = omega_spline(t_k) + bias_gyr - gyr_k
//   weighted by pose_imu_weight_{acc,gyr}. The acceleration term is LINEAR
//   in c_p (pddot(t) = sum_j ddb_j(t) cp_p[j]) and computed analytically;
//   every rotation-dependent piece (R(t_k)'s own dependence on c_phi for the
//   acc term, and the whole omega term's dependence on both c_p [none] and
//   c_phi) is differentiated by CENTRAL FINITE DIFFERENCE on a scratch
//   ScanSpline copy -- deliberately, to avoid hand-deriving a second-order
//   tangent-chart chain rule (phi(t) through Jr, through R(t)'s own use in
//   pddot's frame rotation) under real time pressure; FD is an established,
//   already-used pattern in this codebase's own test suite (see
//   test_coupled_phix_full18.cpp) for exactly this kind of cross-check.
//
// Smoothness (second-difference, separately weighted -- "ONE SCALAR FOR BOTH
// WOULD BE THE SAME UNITS ERROR IN A DIFFERENT PLACE" per the card):
//   pose_curvature_weight_pos on c_p's D^T D block, pose_curvature_weight_rot
//   on c_phi's, D the same [+1,-2,+1] second-difference operator
//   buildImuCorrectionSystem()'s own Curv already uses. No natural sigma
//   normalization exists for a position/attitude control point the way
//   sigma_a/sigma_g exist for accel/gyro corrections, so this term is NOT
//   sigma-scaled (unlike the raw_imu basis's smoothness_weight_{acc,gyr}) --
//   the raw weight IS the precision. Both default 0.0 -- md5-inert (this
//   whole file is unreachable from any shipped path regardless).
//
// Dimensionality: c = [c_p (3*n_c); c_phi (3*n_c)], matching the card's own
// "18+6*n_c... only the meaning of the 6*n_c block changes" instruction --
// this builder returns exactly that 6*n_c-square system's own A/b, the
// c-block only (the 18-dim s-block/Pi_ss prior is unchanged from the
// raw_imu path and out of scope for this standalone builder).
// ============================================================================

namespace livo_recon
{

// One LiDAR point-to-plane observation against the pose basis, reduced to
// what buildPoseSplineCBlock() needs. raw_body_point is q_i (the body-frame
// point BEFORE any deskew warp), matching hcol_reldiff's own
// raw_body_point.cross(...) convention.
struct PoseSplineLidarObs
{
  double t = 0.0;
  V3D    raw_body_point = V3D::Zero();
  V3D    normal = V3D::Zero();
  double r = 0.0;        // current signed point-to-plane residual, at c=0
  double sigma2 = 1.0;
};

// One raw IMU sample, body frame, for the IMU-as-measurement-factor term.
struct PoseSplineImuObs
{
  double t = 0.0;
  V3D    acc = V3D::Zero();
  V3D    gyr = V3D::Zero();
};

struct PoseSplineCBlockBuild
{
  Eigen::MatrixXd A;              // 6*n_c x 6*n_c: [c_p(3n_c); c_phi(3n_c)]
  Eigen::VectorXd b;
  int    n_lidar = 0;
  int    n_imu   = 0;
};

// Builds the pose-basis coefficient-block normal equations for a correction
// c ON TOP OF `spline`'s CURRENT control points (spline is read-only here --
// never mutated; scratch copies are used internally for the FD Jacobian).
// gravity/bias_acc/bias_gyr are FIXED inputs (read from state_ in the live
// path; the s-block, not this c-block, owns their correction). A small fixed
// Tikhonov term (TIKHONOV_EPS below) is always added to the diagonal so A is
// PD even when every weight is 0 -- required for item 6's own correctness
// gate (LiDAR AND IMU terms both disabled) to have a well-posed solve at all.
PoseSplineCBlockBuild buildPoseSplineCBlock(
    const ScanSpline& spline,
    const std::vector<PoseSplineLidarObs>& lidar_obs,
    const std::vector<PoseSplineImuObs>& imu_obs,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double pose_imu_weight_acc, double pose_imu_weight_gyr,
    double pose_curvature_weight_pos, double pose_curvature_weight_rot);

// The fixed regularizer floor -- see buildPoseSplineCBlock()'s own doc
// comment. Small enough to be negligible whenever any real term is active,
// large enough to keep A invertible when every weight is 0 (item 6's gate).
inline constexpr double POSE_SPLINE_TIKHONOV_EPS = 1e-6;

}  // namespace livo_recon
