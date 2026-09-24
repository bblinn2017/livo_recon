#pragma once

#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/pose_control_spline.h"

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
// DISCLOSED SCOPE LIMITATION: J_state (the sensitivity of e_acc/e_gyr to
// TRAJECTORY-state uncertainty -- errors in the converged spline's own
// control points, as distinct from bias/gravity) is NOT included in C_pred
// here. Building it requires the sensitivity of accAt(t)/rotAt(t) at every
// raw IMU sample time to the eta/control-point posterior
// (coupled_pose_control_P_z_post_'s eta block) -- a per-sample Jacobian
// chain of the same shape as poseControlHeadRotJacobian()/dPosDcp()/
// dVelDcp() but evaluated at every IMU sample time within the scan rather
// than once at t1, then propagated through the FULL eta covariance (not
// just its diagonal). That is real, buildable work, but a distinct,
// larger addition than the bias/gravity terms (whose Jacobians are exact
// constants, not per-sample chain rules) -- not implemented in this pass.
// Consequence: the "extra" variance this file estimates is downward-biased
// whenever trajectory-state uncertainty is large relative to bias/gravity
// uncertainty (i.e. it does not yet fully satisfy item 14's C_pred
// definition, only its bias+gravity terms) -- disclosed here and in
// pose_control_implementation_report_v3.md rather than silently claimed
// complete.
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

}  // namespace livo_recon
