#pragma once

#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_knot_spline.h"   // buildImuStep9x9/integrateAndAccumulateStep

// ============================================================================
// The IMU process factor for PoseControlSpline, re-derived per the
// 2026-09-22 correctness gate (spec item 7): "Do NOT reuse the current 9N
// correction-space process construction... do not discard the predicted
// state from the IMU propagation routine... the process factor must be
// mathematically correct under relinearization."
//
// For breakpoint time t_j (j=0..n_seg, the SAME uniform grid the spline's
// own basis breakpoints sit on -- t_j = t0 + j*delta), define the DERIVED
// state from the CURRENT TRIAL spline:
//     x_j(c) = [theta(t_j;c) ; p(t_j;c) ; v(t_j;c)]
// where theta(t_j;c) is realised as the rotation R(t_j;c) itself (there is
// no separate additive theta variable -- see pose_control_spline.h).
//
// The nonlinear residual between consecutive breakpoints is
//     r_j(c) = x_{j+1}(c) boxminus f(x_j(c), u_j)
// with f() the SAME raw-IMU trapezoidal propagation
// imu_processing.cpp::ImuProc::propagate()/pose_knot_spline.cpp's
// integrateAndAccumulateStep() already implement -- reused here, not
// reimplemented, via relinearizePoseControlSegment() below, which calls
// f() STARTING FROM THE CURRENT TRIAL's own R(t_j;c)/p(t_j;c)/v(t_j;c)
// (relinearizing every GN iteration, per the spec) and returns BOTH the
// re-integrated prediction (rot_pred/pos_pred/vel_pred -- NEVER discarded)
// and the resulting F9/Q9.
//
//     r_theta = Log( rot_pred^T * R(t_{j+1};c) )
//     r_pos   = p(t_{j+1};c) - pos_pred
//     r_vel   = v(t_{j+1};c) - vel_pred
//
// Linearized: r_current + J_current * delta_c, with (per spec item 7)
//     dr/dx_j     = -F9        dr/dx_{j+1} = I
//     J_current   = -F9 * J_{x_j} + J_{x_{j+1}}
// where J_{x_j}/J_{x_{j+1}} (9 x 6N, sparse -- nonzero only in the 4
// control-point columns each breakpoint's basis touches) are built from
// PoseControlSpline::dPosDcp/dVelDcp/dThetaDcphi -- the SAME Jacobians
// validated (Test C2, Test D/E) in test_pose_control_spline.cpp.
//
// Lambda = pseudoInverse9(Q9, rel_thresh): the deterministic (zero-noise)
// IMU directions are handled EXACTLY the same way as the physical-knot
// (pose_knots) formulation's own Q9 pseudo-inverse already handles them --
// this is a property of the DISCRETIZED IMU NOISE MODEL (which segment of
// raw samples was integrated, at what dt), not of how the resulting
// prediction is subsequently reparameterized by position/rotation control
// points. Spec item 8's own answer: (a) they are automatically represented,
// because Lambda simply places zero precision on those directions of the
// 9-dim residual regardless of which 6N-dim Jacobian columns that residual
// is chained through -- no separate nullspace-elimination machinery is
// needed or retained for this state.
// ============================================================================

namespace livo_recon
{

// Buckets raw IMU samples into n_seg segments on the spline's own uniform
// breakpoint grid [t0, t0+delta, ..., t1], inserting one linearly
// interpolated boundary sample at each breakpoint -- the same bucketing
// PoseKnotSpline::init() performs (mirrored here rather than reused
// directly, since PoseKnotSpline's own bucketing is a private member of a
// class this file does not otherwise touch).
struct PoseControlImuSegments
{
  std::vector<std::vector<ImuSample>> seg_samples;   // size n_seg
};

PoseControlImuSegments bucketPoseControlImuSamples(
    const std::vector<ImuSample>& imu_raw, const PoseControlSpline& spline);

// Re-integrates segment j's raw IMU samples starting from the CURRENT
// TRIAL's own (rot_j, pos_j, vel_j) -- relinearizing at whatever the
// caller passes, every call, per the spec's correctness gate. Returns
// F9/Q9 (accumulated exactly like PoseKnotSpline::relinearizeSegment()'s
// own accumulation) AND the re-integrated prediction -- never discarded.
void relinearizePoseControlSegment(
    const std::vector<ImuSample>& samples,
    const M3D& rot_j, const V3D& pos_j, const V3D& vel_j,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9,
    M3D& rot_pred, V3D& pos_pred, V3D& vel_pred);

// Re-integrates segment j's raw IMU samples from (rot_j,pos_j,vel_j) EXACTLY
// like relinearizePoseControlSegment(), but ALSO accumulates the 9x9 bias/
// gravity Jacobian G = d[theta,pos,vel](t_{j+1}) / d[bias_gyr,bias_acc,gravity]
// -- the standard IMU-preintegration "bias-correction Jacobian" (Forster et
// al.), needed because in THIS architecture the head correction is held at
// EXACTLY zero (see solveHeadControlPoints()), so decoupled LIO's usual
// pathway for measurement information reaching bg/ba/g (the pose-correction
// coupling through the prior's cross-covariance) does not apply -- the
// process factor's OWN dependence on bias/gravity is the only remaining
// channel. Chained per micro-step via the same F9-recursion pattern
// buildImuStep9x9/integrateAndAccumulateStep already use for F9/Q9:
//     G_seg = F9_step * G_seg + L_step
// where L_step is that step's own LOCAL partial derivative (holding the
// entering state fixed) -- see pose_control_process_factor.cpp for the
// full per-block derivation. FD-validated in
// test_pose_control_process_factor.cpp (perturbing bias_acc/bias_gyr/
// gravity directly and re-walking the SAME raw samples).
void relinearizePoseControlSegmentWithBiasJac(
    const std::vector<ImuSample>& samples,
    const M3D& rot_j, const V3D& pos_j, const V3D& vel_j,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9,
    Eigen::Matrix<double, 9, 9>& G9,
    M3D& rot_pred, V3D& pos_pred, V3D& vel_pred);

// SPD pseudo-inverse via eigendecomposition of the symmetrized matrix,
// relative eigenvalue floor -- identical convention/algorithm to the
// existing pose_knots arm's own pseudoInverse9 lambda (lio_coupled.cpp),
// re-typed here as a free function since that lambda is not exposed.
Eigen::Matrix<double, 9, 9> poseControlPseudoInverse9(
    const Eigen::Matrix<double, 9, 9>& M, double rel_thresh);

// Builds and ADDS segment j's process-factor contribution
// (J^T Lambda J, -J^T Lambda r) into the caller's already-allocated
// A (6N x 6N) / b (6N) normal-equations system (the SAME flattened
// [c_p(3N);c_phi(3N)] layout buildHeadConstraintRows() uses).
// If out_E_process is non-null, accumulates 0.5*r^T*Lambda*r into it
// (E_process, for the campaign CSV / GN diagnostics).
void addPoseControlProcessFactor(
    const PoseControlSpline& spline, int j,
    const std::vector<ImuSample>& samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    double q_pinv_rel_thresh,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    double* out_E_process = nullptr);

// Accumulated head-block contribution (see pose_control_covariance.h's
// Schur-complement mechanism): A_hh (9x9, head-head, columns/rows
// [theta0,pos0,vel0]) and A_hf (9 x layout.dim(), head-free cross block).
// Every segment contributes to A_hh/A_hf's THETA0 columns (rotation's head
// sensitivity is GLOBAL -- see poseControlHeadRotJacobian()'s header
// comment); only segments whose 4-control-point window overlaps cp[0..2]
// (j<=2) contribute nonzero POS0/VEL0 columns (position's head
// sensitivity is LOCAL).
struct PoseControlProcessFactorHeadBlock
{
  Eigen::Matrix<double, 9, 9> A_hh = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::MatrixXd A_hf;   // lazily sized to (9, layout.dim()) on first use
};

// REDUCED-layout version of addPoseControlProcessFactor(): builds against
// z=[c_free; sT] (layout.dim() total) instead of the full 6N. Fixed
// (head, k<3) control-point columns are simply omitted from the free
// Jacobian (they are not GN variables); their sensitivity is captured
// separately in head_block via poseControlHeadRotJacobian()/
// poseControlHeadPosJacobians() (pass nullptr to skip that bookkeeping).
// sT gets a genuine Jacobian column via G9 (relinearizePoseControlSegmentWithBiasJac)
// -- the only channel information can reach bg/ba/g in this architecture
// (the head correction being fixed at zero removes decoupled LIO's usual
// pose-correction-via-prior-cross-covariance pathway -- see that
// function's header comment).
void addPoseControlProcessFactorReduced(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout, int j,
    const std::vector<ImuSample>& samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    double q_pinv_rel_thresh,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlProcessFactorHeadBlock* head_block,
    double* out_E_process = nullptr);

}  // namespace livo_recon
