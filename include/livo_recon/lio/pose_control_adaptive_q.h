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
//
// 2026-09-26 (pose_control_uncertainty_completion task, item 2): the
// trajectory-state term above previously sandwiched ONLY P_eta (the
// free-spline block), silently ignoring the HEAD state's own uncertainty
// (theta0/p0/v0, Sigma_full_post's topLeftCorner(9,9)) and the head/eta
// CROSS covariance -- both of which the joint pose-control posterior
// Sigma_full_post already carries. This was a real gap: e_acc depends on
// the head exactly the way pose_control_imu_prior_builder.cpp's own
// Jhead block already derives and uses for a different purpose (building
// this scan's IMU-only information matrix) -- d(e_acc)/d(theta0) =
// skew(a_spline_body) * poseControlHeadRotJacobian(spline,t), and
// d(e_acc)/d(p0)/d(v0) via poseControlHeadPosSensitivity's Minv chained
// through dAccDcp for the first 3 control points (abs_k<3), EXACTLY
// mirroring that file's derivation (not re-derived independently, to
// avoid the two diverging). e_gyr's head sensitivity is EXACTLY zero --
// omegaBodyAt(t) is built purely from cp_phi, whose particular
// (head-driven) component is fixed at phi(t0)=0 regardless of
// theta0/p0/v0 in this parameterization (see buildPoseControlHeadNullspace
// in pose_control_spline.cpp) -- confirmed both by direct inspection of
// that construction and by a finite-difference regression test here.
//
// computePoseControlImuResidualStateJacobian() now ALSO returns
// J_acc_head/J_gyr_head (each 3x9, columns [theta0(3);p0(3);v0(3)] --
// the SAME head-state layout Sigma_full_post's topLeftCorner(9,9) and
// pose_control_imu_prior_builder.cpp's Jhead already use), and
// applyPoseControlAdaptiveQTrajectoryStateCorrection() now sandwiches the
// FULL [J_head, J_eta] (3 x (9+dEta)) through Sigma_full_post's
// topLeftCorner(9+dEta, 9+dEta) (the SAME Sigma_head_eta_post block
// lio_coupled.cpp already computes for its own head-propagation
// diagnostics), which naturally includes head covariance, eta covariance,
// AND head/eta cross-covariance in one sandwich -- C_state = J_full *
// Sigma_head_eta * J_full^T. Bias and gravity uncertainty are
// DELIBERATELY EXCLUDED from this sandwich -- they are already subtracted
// by applyPoseControlAdaptiveQBiasGravityCorrection() above, using
// state_'s own EKF covariance block (P_T via idxBA()/idxBG()/idxG()), a
// SEPARATE covariance object from Sigma_full_post. Including them again
// here via Sigma_full_post's tail-state (sT) block would double-count the
// same physical uncertainty through two different covariance
// representations -- exactly the double-counting this task's own
// instructions warn against.
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

// d(e_acc)/d(eta), d(e_gyr)/d(eta) (each 3 x hns.freeDim()), AND
// d(e_acc)/d(head), d(e_gyr)/d(head) (each 3x9, head columns
// [theta0(3);p0(3);v0(3)] -- Sigma_full_post's topLeftCorner(9,9) layout)
// at time t, via the SAME production Jacobian primitives (dAccDcp/
// dOmegaDcphi/dThetaDcphi/poseControlHeadRotJacobian/
// poseControlHeadPosSensitivity) both addPoseControlLidarFactor() and
// pose_control_imu_prior_builder.cpp already use for their own per-point/
// per-sample rows -- the head-block derivation here is the SAME one that
// file already implements for building this scan's IMU-only information
// matrix, not re-derived independently. e_acc = R(t)^T*(accAt(t)-gravity)
// +bias_acc - a_measured, e_gyr = omegaBodyAt(t)+bias_gyr - omega_measured
// -- e_gyr's head sensitivity (J_gyr_head) is EXACTLY zero (omegaBodyAt(t)
// depends only on cp_phi, whose head-driven particular component is fixed
// at phi(t0)=0 regardless of head state in this parameterization); bias/
// gravity/measurement terms are constants w.r.t. both eta and head and
// contribute nothing here (handled separately, see
// applyPoseControlAdaptiveQBiasGravityCorrection above).
void computePoseControlImuResidualStateJacobian(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity,
    Eigen::MatrixXd& J_acc_eta, Eigen::MatrixXd& J_gyr_eta,
    Eigen::Matrix<double, 3, 9>& J_acc_head, Eigen::Matrix<double, 3, 9>& J_gyr_head);

// Sandwiches the FULL [J_*_head, J_*_eta] (3 x (9+dEta)) through
// Sigma_head_eta (the head+eta block of the posterior/prior covariance --
// (9+dEta) x (9+dEta), e.g. Sigma_full_post.topLeftCorner(9+dEta,9+dEta))
// and folds the result into st.cov_acc/st.cov_gyr the same way
// applyPoseControlAdaptiveQBiasGravityCorrection() folds in bias/gravity
// -- call this AFTER that function (both subtract from the SAME running
// st.cov_acc/st.cov_gyr, each independently PSD-projected before being
// subtracted, matching item 6's "project C_extra onto the PSD cone" for
// each contribution as it is removed). This naturally includes head
// covariance, eta covariance, AND head/eta cross-covariance in one
// sandwich -- see this file's header comment for why bias/gravity are
// deliberately excluded here (already handled, separately, above).
void applyPoseControlAdaptiveQTrajectoryStateCorrection(
    SplineImuResidualStats& st,
    const Eigen::MatrixXd& J_acc_eta, const Eigen::MatrixXd& J_gyr_eta,
    const Eigen::Matrix<double, 3, 9>& J_acc_head, const Eigen::Matrix<double, 3, 9>& J_gyr_head,
    const Eigen::MatrixXd& Sigma_head_eta);

}  // namespace livo_recon
