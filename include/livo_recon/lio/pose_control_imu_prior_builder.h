#pragma once

#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_physical_diagnostics.h"   // ImuSplineResidualSample
#include "livo_recon/lio/imu_process_step9.h"   // buildImuStep9x9/integrateAndAccumulateStep (test-only reference math below)

// ============================================================================
// CONTINUOUS-TIME PRIOR REPLACEMENT (2026-09-24 targeted phase). Production
// now builds the pose-control IMU prior via buildPoseControlContinuousImuPrior()
// below: a direct continuous-time collocation factor that penalizes
// e_acc(t)/e_gyr(t) (the SAME residual convention pose_control_adaptive_q.h
// documents) AT EVERY RAW IMU SAMPLE against the spline's own p/v/a/omega,
// differentiated w.r.t. the control points/bias/gravity DIRECTLY -- there is
// no intermediate discrete-time state propagation/integration step. This
// REPLACES (does not augment) the old endpoint/segment-propagation prior
// below, which is retained ONLY as test-only reference math (exercised by
// test_pose_control_imu_prior_builder{,_reduced}.cpp for FD validation of
// the underlying G9/F9 IMU-integration Jacobian) -- it has NO production
// call site as of this phase. See buildPoseControlContinuousImuPrior()'s own
// header comment below for the full derivation/rationale.
// ============================================================================

// ============================================================================
// 2026-09-24 PRODUCTION ROLE CLARIFICATION (item 5 of the implementation +
// code-validation phase, SUPERSEDED BY THE CONTINUOUS-TIME REPLACEMENT
// ABOVE): this file's name and the "process factor" terminology below
// predate a change in how this math is actually used in production. Read
// this note before the derivation comment that follows. This describes the
// OLD (now test-only) endpoint/segment-propagation math, kept for FD
// reference only.
//
// FORMER PRODUCTION ROLE (through commit 3bf7151): the ONE production call site
// (LioProcCoupled's coupled-pose-control init block, lio_coupled.cpp, via
// buildPoseControlImuPriorContribution() below) invokes
// accumulatePoseControlImuPriorSegmentReduced() EXACTLY ONCE PER SCAN, at scan
// start, before any Gauss-Newton iteration runs. Its output (A/b) is folded
// into the fixed scan-start joint IMU/bias/gravity prior
// (Lambda_prior_z/z_imu_, see lio_coupled.cpp's init-block comment) and is
// NEVER relinearized or re-invoked during the GN loop. In other words, in
// production this file implements:
//     physical IMU process model  -->  fixed scan-start prior
// and NOT "a process factor relinearized every GN iteration" (that was the
// role of the now-removed legacy_process_factor/frozen_process_hessian_prior
// modes -- see git history, commit 12d61eb and earlier).
//
// accumulatePoseControlImuPriorSegment() (the non-reduced, full-6N-column variant)
// and relinearizePoseControlSegment() (without the bias/gravity Jacobian)
// have NO production call site at all -- they exist solely as reference
// implementations exercised by scripts/test/test_pose_control_process_
// factor{,_reduced}.cpp. Do not add a new production call site that
// relinearizes this math per-GN-iteration; that would silently reintroduce
// the double-counting/ambiguity the joint-prior reformulation removed.
// ============================================================================

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
// entering state fixed) -- see pose_control_imu_prior_builder.cpp for the
// full per-block derivation. FD-validated in
// test_pose_control_imu_prior_builder.cpp (perturbing bias_acc/bias_gyr/
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
void accumulatePoseControlImuPriorSegment(
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

// REDUCED-layout version of accumulatePoseControlImuPriorSegment(): builds against
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
void accumulatePoseControlImuPriorSegmentReduced(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout, int j,
    const std::vector<ImuSample>& samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    double q_pinv_rel_thresh,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlProcessFactorHeadBlock* head_block,
    double* out_E_process = nullptr);

// ============================================================================
// PRODUCTION: continuous-time IMU collocation prior (items 1-27 of the
// continuous-prior implementation phase).
//
// DISCRETIZATION CHOICE (item 15, audited explicitly per the continuous-
// prior-validation phase's item 22): sample-level, i.e. every raw IMU
// sample in [spline.t0(), spline.t1()] is treated as an independent
// physical measurement of e_acc(t)/e_gyr(t), weighted by var_acc/var_gyr.
//
// UNITS AUDIT (item 22): var_acc/var_gyr (state_->varAcc()/varGyr(), read
// here via poseControlEffectiveVarAcc()/VarGyr()) are PER-SAMPLE
// measurement variances AT THE IMU's OWN NATIVE SAMPLE RATE, not a
// continuous-time power spectral density. Traced to source:
// CalibProc::computeBiasAndNoise() (calib_processing.cpp) computes them as
// the literal sample variance (sum((x_i-mean)^2)/(N-1)) of N raw,
// consecutively-sampled stationary IMU readings taken at the sensor's
// actual streaming rate (~200Hz for this codebase's NTU_VIRAL configs) --
// this is unambiguously a discrete-time, fixed-rate quantity, not a
// PSD (which would carry units of variance-per-Hz and require dividing by
// dt to get a per-sample variance). Because production NEVER resamples
// the real IMU stream at a different rate than the one var_acc/var_gyr
// were calibrated at, applying var_acc/var_gyr identically to every raw
// sample (no additional dt-dependent rescaling) is the physically-
// consistent, self-calibrated discretization for THIS system -- not an
// approximation of a "real" continuous-time model that happens to be
// convenient. This is also the SAME per-sample interpretation every other
// consumer of varAcc()/varGyr() in this codebase already uses (e.g.
// computePoseControlImuResidual()'s reduceImuResidualSamples(), AdaptiveQ)
// -- no second interpretation exists anywhere in the codebase.
//
// A theoretical caveat, tested explicitly (item 23,
// test_pose_control_prior_mean_correctness.cpp / the timestep-refinement
// test in test_pose_control_continuous_prior_reference.cpp): if the SAME
// fixed var_acc/var_gyr were ever applied while artificially resampling a
// bag at a DIFFERENT rate than it was calibrated at (something production
// never does), total information would scale with sample count rather
// than converging to a fixed continuous-time limit -- this is the correct
// behavior for genuinely independent per-sample measurement noise (more
// real, independent samples of the same true signal ARE more information,
// not double-counting), and is NOT a bug in this implementation; it would
// only become a modeling error if var_acc/var_gyr's own calibration rate
// and the actual runtime IMU rate were ever allowed to diverge, which nothing
// in this codebase permits. Introducing a midpoint/trapezoidal/PSD-based
// scheme here would require inventing a new dt-scaling convention not used
// anywhere else, exactly what item 9 of an earlier phase prohibited ("do
// not invent a new empirical process-weight scalar"). No information is
// double-counted across samples: each sample contributes its own factor
// once, at its own timestamp, and nothing else touches A/b for that
// sample.
//
// RESIDUALS (items 7/8, EXACT convention, matching pose_control_adaptive_q.h
// byte-for-byte):
//   e_acc(t) = R(t)^T * (a_spline(t) - gravity) + bias_acc - a_measured(t)
//     units: m/s^2 (specific force). a_spline(t) = spline.accAt(t), the
//     WORLD-frame second derivative of the position spline. gravity is
//     SUBTRACTED in the world frame before rotating into the body frame
//     (matches this codebase's existing gravity-in-world convention used
//     throughout pose_control_adaptive_q.cpp/pose_control_process_factor's
//     own G9 derivation). bias_acc is ADDED (matches the IMU model
//     a_measured = R^T(a-g) + bias_acc + noise, solved for the residual
//     against the CURRENT bias estimate).
//   e_gyr(t) = spline.omegaBodyAt(t) + bias_gyr - omega_measured(t)
//     units: rad/s (body-frame angular rate). bias_gyr ADDED, matching
//     omega_measured = omega_body + bias_gyr + noise.
//
// JACOBIANS (items 6/10, reusing PoseControlSpline's own basis exclusively
// -- no second basis implementation):
//   de_acc/d(cp_p[k])   = R(t)^T * dAccDcp(jac,k)            (local, via a_spline)
//   de_acc/d(cp_phi[k]) = skew(R(t)^T*(a_spline(t)-gravity)) * dThetaDcphi(jac,k,t)
//   de_acc/d(ba)        = I                                  (item 10)
//   de_acc/d(g)         = -R(t)^T                             (item 10, since e_acc = R^T(a-g)+ba-a_meas)
//   de_gyr/d(cp_phi[k]) = dOmegaDcphi(jac,k,t)
//   de_gyr/d(bg)        = I                                  (item 10)
//   de_acc/d(cp_p[k]) for k in {0,1,2} additionally carries HEAD coupling
//     (item 4/11/13): d(cp_p[k])/d(p0)=Minv(k,0)*I, d(cp_p[k])/d(v0)=Minv(k,1)*I
//     (poseControlHeadPosSensitivity()) -- chained the SAME way
//     addPoseControlLidarFactor() already chains its own position head
//     coupling. de_acc/d(theta0) = skew(R^T*(a-g)) * poseControlHeadRotJacobian(t)
//     (theta0's GLOBAL rotation-sensitivity, exactly mirroring
//     addPoseControlLidarFactor()'s Jrow_head_theta). de_gyr/d(theta0) = 0
//     and de_gyr/d(p0)=de_gyr/d(v0)=0 EXACTLY: omegaBodyAt(t) is a function
//     of cp_phi/phiDotAt(t) alone (a LOCAL body-frame rate), and cp_phi does
//     not depend on theta0/p0/v0 at all (see pose_control_spline.h's own
//     head-covariance-sensitivity derivation) -- a rigid relabeling of the
//     fixed reference frame R_anchor cannot change a body-frame rate.
//
// This is what makes high-frequency position-knot jitter expensive (item
// 16): dAccDcp scales as 1/Delta_t^2, so a knot oscillation that shows up
// as a large a_spline(t) at collocation points directly produces a large
// e_acc(t), weighted by the SAME physical accelerometer variance every
// other sample uses -- no separate curvature/stiffness knob is needed or
// added (item 17).
//
// P0 (item 4/11/13/14): incoming state covariance enters via head_block
// (A_hh/A_hf), EXACTLY the same joint-[x0;z]-then-marginalize construction
// the LiDAR factor and (formerly) the old process factor already use --
// the caller (lio_coupled.cpp) builds Omega0=pinv(P0), adds it to A_hh, and
// Schur-complements x0 out, so P0's FULL cross-covariance structure
// (position/velocity/bias/gravity, never just a diagonal -- item 14)
// propagates into the resulting z=[eta;sT] prior by construction, not by
// an ad hoc diagonal approximation.
//
// THREADING (item 36): parallel loop over samples, thread-local
// accumulators (A/b/A_hh/A_hf/E_imu), single linear merge -- the SAME
// pattern addPoseControlLidarFactor() already uses. out_samples (if
// non-null) also collects each sample's residual (item 27/37: ONE
// evaluation feeds the prior, adaptive-Q, and diagnostics -- the caller
// passes these same samples to adaptive-Q instead of re-evaluating the
// spline).
// ============================================================================
struct PoseControlContinuousImuPriorStats
{
  int n_samples = 0;
  double E_imu = 0.0;
};

PoseControlContinuousImuPriorStats buildPoseControlContinuousImuPrior(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const std::vector<ImuSample>& imu_samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    const V3D& var_acc, const V3D& var_gyr,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlProcessFactorHeadBlock* head_block,
    std::vector<ImuSplineResidualSample>* out_samples = nullptr);

}  // namespace livo_recon
