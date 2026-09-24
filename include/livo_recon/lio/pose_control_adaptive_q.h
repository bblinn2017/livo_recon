#pragma once

#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"

// ============================================================================
// 2026-09-23: wires AdaptiveQ (adaptive_q.h/.cpp) into the pose-control
// estimator using PHYSICAL IMU-domain residuals, not r_position+r_velocity.
//
// e_acc(t) = R(t)^T * (a_spline_predicted(t) - gravity) + bias_acc - a_measured(t)
// e_gyr(t) = omega_spline_predicted_body(t) + bias_gyr - omega_measured(t)
//
// i.e. exactly computeSplineImuResidual()'s (spline.h) own convention,
// re-derived here for PoseControlSpline (accAt/omegaBodyAt/rotAt) instead of
// ScanSpline -- both reduce through the SAME reduceImuResidualSamples()
// (spline.h) so there is one authoritative statistics reduction, not two.
// This was PREVIOUSLY UNWIRED for pose_control entirely: AdaptiveQ::update()
// was never called on this path (only varAcc()/varGyr() were read, always
// falling through to the nominal since active() can never become true
// without a prior update() call) -- confirmed via a full repo grep before
// this file existed. That is the concrete bug this file fixes.
//
// CAUSALITY (item 17): computePoseControlImuResidual() must be called AFTER
// this scan's GN loop has converged (so the spline used is the converged
// trajectory), and AdaptiveQ::update() therefore produces a CANDIDATE for
// scan k only after scan k's own LiDAR update has already happened --
// poseControlEffectiveVarAcc()/VarGyr() (read at the START of the NEXT
// scan's own scan-start prior build) are what actually consume it. The
// current scan's own prior was already fixed before this file's update()
// call runs, so a scan's LiDAR measurement can never retroactively change
// the Q that built its own prior.
//
// BIAS/GRAVITY UNCERTAINTY CORRECTION (items 14/15/22/23): applyBiasGravity
// CorrectionToResidualStats() subtracts the analytically-exact bias/gravity
// contribution C_pred = J P J^T (J is exact and constant here: e_acc's
// bias_acc term is a plain +I, its gravity term is -R^T; e_gyr's bias_gyr
// term is a plain +I) from the empirical residual covariance BEFORE that
// covariance is treated as "extra" process noise -- replacing the
// previously-flagged trace(P_bias)/3 style proxy (not previously present in
// this file, since this path did not previously exist) with the real
// Jacobian-propagated contribution, including the ba/gravity CROSS term.
//
// 2026-09-24: TRAJECTORY-STATE term added. J_state (the sensitivity of
// e_acc/e_gyr to uncertainty in the converged spline's own control points,
// as distinct from bias/gravity) is now included via
// computePoseControlImuResidualStateJacobian(), evaluated at ONE
// representative time (the residual window's midpoint) rather than
// per-IMU-sample -- a documented simplification consistent with the
// bias/gravity terms' own treatment (both are systematic, not i.i.d.,
// across the sample window, and neither addresses the mean-centering
// interaction rigorously; this term is not a new category of
// approximation beyond what the bias/gravity terms already accept, see
// pose_control_implementation_report_v4.md). The Jacobian primitives
// (dAccDcp, dOmegaDcphi, dThetaDcphi) are the SAME ones
// addPoseControlLidarFactor() already uses, chained the same way the
// LiDAR factor's own dr_dcp_phi is; the sign/convention is verified by a
// finite-difference check in the registered test (not asserted from
// derivation alone).
// ============================================================================

namespace livo_recon
{

// Physical-residual statistics for the pose-control spline, reduced through
// the SAME machinery computeSplineImuResidual() uses for ScanSpline.
SplineImuResidualStats computePoseControlImuResidual(
    const PoseControlSpline& spline, const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity);

// Subtracts the EXACT bias_acc/gravity (resp. bias_gyr) contribution to the
// expected residual covariance from st.cov_acc/st.cov_gyr in place, clamped
// at 0 (the scalar PSD projection -- C_extra can never be reported negative).
// R_rep is the representative body-to-world rotation used for the gravity
// Jacobian (-R_rep^T); using ONE representative rotation for the whole scan
// (rather than per-sample) is a documented simplification consistent with
// this estimator's own isotropic/scalar (not full 3x3 anisotropic) process-
// noise convention elsewhere (AdaptiveQ itself only ever reports a single
// scalar variance per channel, matching state_->varAcc()/varGyr()'s own
// Constant(...)-broadcast contract).
void applyPoseControlAdaptiveQBiasGravityCorrection(
    SplineImuResidualStats& st, const M3D& R_rep,
    const Eigen::Matrix3d& P_ba, const Eigen::Matrix3d& P_bg,
    const Eigen::Matrix3d& P_g, const Eigen::Matrix3d& P_ba_g_cross);

// d(e_acc)/d(eta) and d(e_gyr)/d(eta) (each 3 x hns.freeDim()) at time t,
// via the SAME production Jacobian primitives (dAccDcp/dOmegaDcphi/
// dThetaDcphi) addPoseControlLidarFactor() uses for its own per-point
// rows, chained through the SAME head-nullspace basis Z the mean solve
// projects onto. e_acc = R(t)^T*(accAt(t)-gravity)+bias_acc - a_measured,
// e_gyr = omegaBodyAt(t)+bias_gyr - omega_measured -- only the accAt(t)/
// rotAt(t)/omegaBodyAt(t) terms depend on the trajectory (eta); bias/
// gravity/measurement terms are constants w.r.t. eta and contribute
// nothing here (handled separately, see
// applyPoseControlAdaptiveQBiasGravityCorrection above).
void computePoseControlImuResidualStateJacobian(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity,
    Eigen::MatrixXd& J_acc_eta, Eigen::MatrixXd& J_gyr_eta);

// Sandwiches J_acc_eta/J_gyr_eta (from the function above) through P_eta
// (the eta-block of the posterior/prior covariance -- dEta x dEta) and
// folds the result into st.cov_acc/st.cov_gyr the same way
// applyPoseControlAdaptiveQBiasGravityCorrection() folds in bias/gravity
// -- call this AFTER that function (both subtract from the SAME running
// st.cov_acc/st.cov_gyr, each independently PSD-projected before being
// subtracted, matching item 6's "project C_extra onto the PSD cone" for
// each contribution as it is removed).
void applyPoseControlAdaptiveQTrajectoryStateCorrection(
    SplineImuResidualStats& st,
    const Eigen::MatrixXd& J_acc_eta, const Eigen::MatrixXd& J_gyr_eta,
    const Eigen::MatrixXd& P_eta);

}  // namespace livo_recon
