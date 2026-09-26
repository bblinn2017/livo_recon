#pragma once

#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"

// ============================================================================
// TARGETED VALIDATION PHASE instrumentation (2026-09-24).
//
// Pure diagnostics -- never mutates production prior/covariance state, and
// nothing here is wired into the mean solve's A/b. Two families of
// quantities:
//
// (1) evaluatePoseControlPhysicalSample() (item 8): p(t)/R(t)/v(t)/a(t)/
//     omega(t) and their Jacobians w.r.t. the reduced control state eta,
//     independent of LiDAR -- the basis for the parameterization/
//     covariance/information-invariance tests (items 9-13).
//
// (2) The raw IMU-vs-spline residual/Jacobian/information diagnostics
//     (items 18/19/32, carried over from the prior real-data campaign) --
//     e_acc(t) = R(t)^T*(accAt(t)-gravity) + bias_acc - a_measured(t)
// e_gyr(t) = omegaBodyAt(t) + bias_gyr - omega_measured(t)
// -- the SAME convention pose_control_adaptive_q.h/.cpp already uses (see
// its own header comment); this file's per-sample loop is a second,
// independent evaluation of the SAME formula (not a second implementation
// of a DIFFERENT formula) so the residuals reported here are directly
// comparable, sample for sample, to the adaptive-Q residual statistics.
// Jacobian primitives are the exact ones already used/verified by
// computePoseControlImuResidualStateJacobian() (dAccDcp/dOmegaDcphi/
// dThetaDcphi, chained through hns.Z) plus three additional constant
// blocks (d/d(delta_ba)=I, d/d(delta_g)=-R^T for e_acc; d/d(delta_bg)=I
// for e_gyr) matching pose_control_adaptive_q.cpp's own documented bias/
// gravity Jacobian convention exactly (J_ba=I, J_g=-R^T, J_bg=I).
// ============================================================================

namespace livo_recon
{

// Diagnostics-only physical correction geometry used by the architecture
// comparison. These helpers contain no estimator state and use independently
// derived Euclidean/SO(3) formulas so the same math can be unit-tested and
// reused by offline analysis.
struct PoseControlCorrectionMetrics
{
  double correction_norm = 0.0;
  double gt_distance_before = 0.0;
  double gt_distance_after = 0.0;
  double gt_error_reduction = 0.0;
  double gt_cosine = 0.0;
  double gt_parallel = 0.0;
  double gt_perpendicular = 0.0;
  bool gt_direction_valid = false;
};

PoseControlCorrectionMetrics computePoseControlCorrectionMetrics(
    const V3D& p_before, const V3D& p_after, const V3D& p_gt,
    double eps = 1e-12);

// Solve a diagnostic quadratic factor step x = pinv(A) b. The input matrix
// is symmetrized before the eigendecomposition. This is for analysis only; it
// never modifies production matrices or the estimator state.
struct PoseControlFactorStep
{
  Eigen::VectorXd delta;
  double lambda_min = 0.0;
  double lambda_max = 0.0;
  int effective_rank = 0;
};

PoseControlFactorStep solvePoseControlFactorStep(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
    double rel_threshold = 1e-12);


// ============================================================================
// Item 8: physical trajectory sample + Jacobians w.r.t. the reduced control
// state eta (dEta-wide, the SAME coordinate the mean solve's A/b uses).
// Independent of LiDAR -- constructed purely from the spline + head-
// nullspace basis. p/v/a/omega share ONE spline.jacobianAt(t) evaluation
// (item 34: no duplicated computation).
// ============================================================================
struct PoseControlPhysicalSample
{
  double t = 0.0;
  V3D p = V3D::Zero(), v = V3D::Zero(), a = V3D::Zero(), omega = V3D::Zero();
  M3D R = M3D::Identity();
  // Each dQ_deta is 3 x hns.freeDim(). dtheta_deta is d(local body-frame
  // rotation perturbation at t)/d(eta) -- the attitude analogue of the
  // others (distinct from domega_deta, which is the angular-RATE
  // sensitivity); attitude covariance at t is J_theta*P_eta*J_theta^T.
  Eigen::MatrixXd dp_deta, dv_deta, da_deta, domega_deta, dtheta_deta;

  // 2026-09-25 covariance-reformulation fix (item 3 of the campaign-2
  // audit): each dQ_dhead is 3x9, w.r.t. the fixed-head STATE x0=
  // [dtheta0(3),dp0(3),dv0(3)] used elsewhere in this file's own
  // head_block/M_full machinery (see pose_control_imu_prior_builder.cpp's
  // Jhead and lio_coupled.cpp's J_h_tail). The mean solve holds x0 at its
  // fixed value (conditional mean, eta is the only free variable), but x0
  // itself is UNCERTAIN in the joint prior Sigma_full=[x0;eta;sT] -- any
  // physical quantity y(t) that depends on x0 therefore needs THIS
  // Jacobian too when mapping Sigma_full (not just Sigma_eta) into
  // physical covariance, or the propagated uncertainty silently omits the
  // head's own contribution (most visibly wrong at t==t0, where dQ_deta is
  // exactly zero by construction -- the head is fixed there -- so omitting
  // dQ_dhead there wrongly reports P_y(t0)=0 regardless of how uncertain
  // x0 actually is).
  //
  // Translation (p/v/a) depends on p0/v0 but NOT theta0 (the position and
  // attitude splines are independently parameterized); rotation (theta)
  // depends on theta0 but NOT p0/v0; angular velocity omega(t) is exactly
  // INVARIANT to theta0 (a global left-rotation of the whole attitude
  // trajectory does not change its own body-frame rate) and has no p0/v0
  // dependence either -- domega_dhead is therefore always identically
  // zero, kept as an explicit zero matrix (not omitted) so callers never
  // have to special-case its absence.
  Eigen::MatrixXd dp_dhead, dv_dhead, da_dhead, domega_dhead, dtheta_dhead;
};

PoseControlPhysicalSample evaluatePoseControlPhysicalSample(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity);

// Item 3: full physical covariance P_y(t) = J_full * Sigma_full_9plus *
// J_full^T, where J_full = [dQ_dhead (3x9), dQ_deta (3xdEta)] and
// Sigma_full_9plus is the LEADING (9+dEta) x (9+dEta) block of the joint
// [x0;eta;sT] covariance (x0/eta only -- sT's contribution to a purely
// kinematic quantity like p/v/a/theta/omega is zero, since these physical
// quantities have no direct bg/ba/g dependence; a quantity that DOES
// depend on bias/gravity, e.g. the IMU residual itself, needs its own
// wider Jacobian and is handled separately, not through this helper).
// dQ_deta/dQ_dhead must already be the two blocks for the SAME physical
// quantity Q (e.g. both position, or both velocity).
Eigen::Matrix3d poseControlPhysicalCovariance(
    const Eigen::MatrixXd& dQ_dhead, const Eigen::MatrixXd& dQ_deta,
    const Eigen::MatrixXd& Sigma_head_eta);

// One IMU sample's full residual breakdown against the converged spline --
// mirrors computePoseControlImuResidual()'s per-sample loop body but
// returns every intermediate quantity (items 6-9/30's required CSV fields)
// instead of only the reduced ra/rw used for adaptive-Q statistics.
struct ImuSplineResidualSample
{
  double t = 0.0;
  V3D a_meas = V3D::Zero();
  V3D a_spline_body = V3D::Zero();   // R(t)^T*(accAt(t)-gravity), i.e. the specific-force prediction BEFORE +bias_acc
  V3D e_acc = V3D::Zero();           // a_spline_body + bias_acc - a_meas
  V3D omega_meas = V3D::Zero();
  V3D omega_spline_body = V3D::Zero();
  V3D e_gyr = V3D::Zero();           // omega_spline_body + bias_gyr - omega_meas
};

// Threaded (thread-local disjoint-index writes, no merge race -- item 27),
// mirrors computePoseControlImuResidual()'s own windowing/threading
// pattern exactly (spline.t0()/t1() bracket, schedule(static)).
std::vector<ImuSplineResidualSample> computePoseControlImuSplineResidualSamples(
    const PoseControlSpline& spline, const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity);

// Full measurement Jacobians d(e_acc)/dz, d(e_gyr)/dz at time t, where
// z = [eta (dEta); delta_bg (3, if present); delta_ba (3, if present);
// delta_g (3, if present)] -- the SAME z ordering/offsets the mean solve's
// own A/b (lio_coupled.cpp's estimateCoupledPoseControlSpline) uses.
// dimZ/off_bg/off_ba/off_g are the caller's own PoseControlFreeLayout-
// derived offsets (dEta + layout.col{BG,BA,G}() - layout.dimCFree()), -1
// if that block doesn't exist in z, passed in directly rather than
// re-derived here to guarantee identical indexing with the caller's A/b.
void computePoseControlImuMeasurementJacobianZ(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity,
    int dimZ, int off_bg, int off_ba, int off_g,
    Eigen::MatrixXd& H_acc, Eigen::MatrixXd& H_gyr);

// Batch information-diagnostic accumulation (item 13): treats H_acc/H_gyr
// (evaluated ONCE at a representative time -- the residual window's
// midpoint, the SAME simplification computePoseControlImuResidualStateJacobian
// already documents for the trajectory-state term) as constant across the
// n_acc/n_gyr in-window samples, so
//   Lambda_acc = n_acc * H_acc^T R_acc^-1 H_acc,
//   b_acc      = n_acc * H_acc^T R_acc^-1 * mean_e_acc
// (and likewise for gyro), Lambda_imu_meas = Lambda_acc + Lambda_gyr,
// b_imu_meas = b_acc + b_gyr. R_acc/R_gyr are the NOMINAL (non-adaptive)
// per-axis sensor variances (state_->varAcc()/varGyr()) -- item 11's
// "C_sensor" role, deliberately NOT the adaptive-Q-corrected variance,
// since this is meant to represent the raw sensor noise spec, not a
// process-consistency estimate.
struct ImuMeasurementInformation
{
  Eigen::MatrixXd Lambda_acc, Lambda_gyr, Lambda_imu_meas;
  Eigen::VectorXd b_acc, b_gyr, b_imu_meas;
  double trace_lambda_imu_meas = 0.0, min_eig = 0.0, max_eig = 0.0, condition = 0.0;
  int effective_rank = 0;
};

ImuMeasurementInformation computePoseControlImuMeasurementInformation(
    const Eigen::MatrixXd& H_acc, const Eigen::MatrixXd& H_gyr,
    const V3D& R_acc_diag, const V3D& R_gyr_diag,
    int n_acc, int n_gyr, const V3D& mean_e_acc, const V3D& mean_e_gyr);

// Item 11's residual-to-Q accounting, computed WITHOUT mutating the input
// stats (unlike pose_control_adaptive_q.cpp's apply*Correction functions,
// which subtract in place for production consumption) -- this returns
// every intermediate term so the diagnostic CSV can report exactly what
// is/isn't explained, before and after PSD projection.
struct ResidualToQAccounting
{
  double C_empirical_acc = 0.0, C_pred_state_acc = 0.0, C_sensor_acc = 0.0;
  double C_extra_acc = 0.0, C_extra_acc_psd = 0.0;
  double C_empirical_gyr = 0.0, C_pred_state_gyr = 0.0, C_sensor_gyr = 0.0;
  double C_extra_gyr = 0.0, C_extra_gyr_psd = 0.0;
};

ResidualToQAccounting computePoseControlResidualToQAccounting(
    double cov_acc_empirical, double cov_gyr_empirical,
    const M3D& R_rep, const Eigen::Matrix3d& P_ba, const Eigen::Matrix3d& P_bg,
    const Eigen::Matrix3d& P_g, const Eigen::Matrix3d& P_ba_g_cross,
    const Eigen::MatrixXd& J_acc_eta, const Eigen::MatrixXd& J_gyr_eta,
    const Eigen::MatrixXd& P_eta, double R_acc_sensor, double R_gyr_sensor);

}  // namespace livo_recon
