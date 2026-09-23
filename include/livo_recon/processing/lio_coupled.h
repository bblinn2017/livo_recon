#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/coupled_estimator.h"
#include "livo_recon/lio/pose_spline_system.h"
#include "livo_recon/lio/pose_knot_spline.h"
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_process_factor.h"
#include "livo_recon/lio/pose_control_lidar_factor.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/utils/eval/nees_logger.h"

#include <limits>
#include <unordered_set>

namespace livo_recon
{

// CQ-49: options exclusive to the coupled estimator, under estimator/coupled/*
// -- a decoupled config never instantiates this class, so these keys are
// simply never claimed there, and checkAllParamsConsumed() throws naming any
// such key left in a decoupled config (item 3d(iii): object identity IS the
// nesting enforcement).
struct LioProcCoupledOptions
{
  // n_c control points for the correction basis (item 2: NOT
  // control_point_hz -- a separate resolution knob for a different spline).
  int n_c = 4;

  // CQ-82 Phase 2: which basis the n_c control points parameterize.
  //   "raw_imu" (DEFAULT): control points are corrections to the
  //     accelerometer/gyro signal (the basis this whole file already
  //     implements -- unchanged).
  //   "pose": control points are POSITION and ATTITUDE, evaluated via
  //     ScanSpline's own basis (spline.h) -- the LiDAR term is then linear
  //     in c_p directly (no chain through the IMU state), and the IMU
  //     enters as a measurement FACTOR (spline-implied accel/omega vs raw
  //     IMU), not a prior. See buildPoseSplineSystem().
  //   "pose_knots" (user instruction 2026-09-21, 32-item redesign):
  //     control points are ACTUAL PHYSICAL TRAJECTORY KNOTS (p_j,R_j,v_j at
  //     known times t_j -- PoseKnotSpline, pose_knot_spline.h) rather than
  //     B-spline basis coefficients (cp_j != p(tj) under "pose" -- this
  //     mode's whole point is closing that gap so a covariance genuinely
  //     attached to knot j means Cov[p(tj),theta(tj)], not
  //     Cov[coefficient j]). Kept as a THIRD, separate mode alongside
  //     "pose" (item 23: "I would not replace ScanSpline globally... this
  //     protects the existing decoupled estimator" -- and by the same
  //     logic, protects the existing coefficient-basis "pose" mode too,
  //     which stays exactly as shipped for comparison).
  // Config key: estimator/coupled/spline_mode.
  std::string spline_mode = "raw_imu";
  static constexpr const char* SPLINE_MODES[] = { "raw_imu", "pose", "pose_knots", "pose_control" };
  bool poseBasis() const { return spline_mode == "pose"; }
  bool poseKnotsBasis() const { return spline_mode == "pose_knots"; }
  // 2026-09-22: the pose-CONTROL-POINT-only trajectory state -- z=[c_free;
  // sT], no independent velocity/angular-velocity DOF, head mean fixed
  // (not a GN variable), tail free (spline-derived R/p/v + free bg/ba/g).
  // See pose_control_spline.h/pose_control_process_factor.h/
  // pose_control_lidar_factor.h/pose_control_covariance.h and
  // estimateCoupledPoseControlSpline()'s own doc comment for the mechanism.
  bool poseControlSplineBasis() const { return spline_mode == "pose_control"; }
  // Config key: estimator/coupled/pose_control/n_control_points.
  int pose_control_n = 13;
  // Config key: estimator/coupled/pose_control/q_pinv_rel_thresh -- same
  // convention as pose_knots_q_pinv_rel_thresh below.
  double pose_control_q_pinv_rel_thresh = 1e-6;
  // Diagnostic-only knobs for the 2026-09-22 correction's required
  // scan-1 test suite (items 16/18) -- both default to shipping behavior
  // (LiDAR on, P0 unscaled). Config keys:
  // estimator/coupled/pose_control/{lidar_enable,p0_scale}.
  bool pose_control_lidar_enable = true;
  double pose_control_p0_scale = 1.0;

  // CQ-82 Phase 2: pose-basis-only weights. Meaningless under raw_imu (the
  // refusal wiring never checks these -- they simply aren't read unless
  // poseBasis() is true). Config keys: estimator/coupled/pose_imu_weight_{acc,gyr},
  // estimator/coupled/pose_curvature_weight_{pos,rot}.
  double pose_imu_weight_acc = 1.0;
  double pose_imu_weight_gyr = 1.0;
  double pose_curvature_weight_pos = 0.0;
  double pose_curvature_weight_rot = 0.0;
  // CQ-86 item 1: the WEAKER anchor the card's own fallback authorized --
  // "IF COUPLING s TO c IS MORE THAN THIS ROUND CAN CARRY... instead FREEZE
  // the head control points to the raw-chain values". Freezes the first
  // `pose_head_freeze_cp` control points' correction (both c_p and c_phi
  // blocks) to EXACTLY ZERO by eliminating those columns from the linear
  // solve entirely (not a soft/large-weight prior) -- the spline's HEAD
  // stays at wherever this scan's ScanSpline::fit() (the IMU-propagated
  // raw chain) put it, for as many control points as this covers.
  // MEASURED NEGATIVE (CQ-86 item 3): freezing made stability WORSE than no
  // anchor at all (last_scan 62 vs 69) -- a hard elimination removes solve
  // freedom and adds no information. STAYS as a separate, named, opt-in
  // arm (default 0/off, md5-inert) -- CQ-87 item 1 did NOT delete it, per
  // Bryce's own instruction, since it is a measured result this register
  // cites. THE REAL head-tie (p(t0)=p_start+delta_p0 via the clamped-
  // B-spline identity, WITH a genuine Omega prior on delta_phi0/delta_pos0)
  // is now the DEFAULT pose-arm behavior whenever this is 0 -- see
  // reducePoseSplineHeadCoupling() and estimateCoupledCorrectionPoseBasis().
  // When this is > 0, that real coupling is SKIPPED and the older
  // hard-elimination arm runs instead, reproducing CQ-86's own measured
  // cells exactly. Config key: estimator/coupled/pose_head_freeze_cp.
  int pose_head_freeze_cp = 0;

  // CQ-87 item 5, Bryce asked for this directly: POSE_SPLINE_TIKHONOV_EPS
  // (pose_spline_system.h) is no longer a compile-time-only constant --
  // this is the live value threaded through buildPoseSplineCBlock() as an
  // explicit parameter. Default 1e-6 is EXACTLY that constant's own value,
  // so this key is md5-inert at its default. Config key:
  // estimator/coupled/pose_tikhonov_eps.
  double pose_tikhonov_eps = 1e-6;

  // POST-CQ-87-REVIEW FIX (item 4, user follow-up 2026-09-21): the pose
  // arm's IMU-factor weighting must use the SHIPPED config values
  // directly -- state/cov/acc, state/cov/gyr (state.cpp's own
  // paramWarn<double>(pnh, "state/cov/acc", ...), which seeds
  // state_->var_acc_/var_gyr_ -- VARIANCE, (m/s^2)^2 / (rad/s)^2, same
  // units/name as state.cpp's own local `var_acc`/`var_gyr`) -- rather
  // than reading them indirectly through state_->varAcc()/varGyr(). Under
  // the DEFAULT operating setup (CalibProcOptions::use_calib_var == false,
  // the shipped default) those two happen to be numerically identical,
  // since state_'s own members are seeded from these SAME two keys at
  // startup and never mutated by this arm -- but state_'s copy CAN
  // silently diverge from the config the moment use_calib_var=true (the
  // calibration pass overwrites it) or the decoupled arm's own AdaptiveQ
  // path runs (lio_decoupled.cpp:860, a DIFFERENT class's own mechanism
  // that also calls state_->setNoiseParams()). Reading these two config
  // keys directly, here, makes the pose arm's own weighting immune to
  // either -- exactly "use the config directly," not "usually equals the
  // config." Config keys: state/cov/acc, state/cov/gyr (the SAME keys
  // state.cpp reads -- this is a second, independent read of the same
  // param server entry, not a new key). Default 1e-4/1e-4 matches
  // state.cpp's own paramWarn() default exactly.
  double pose_imu_var_acc = 1e-4;
  double pose_imu_var_gyr = 1e-4;

  // User instruction 2026-09-21, item 15 ("fix the remaining numerical
  // stability issue... the observed ScanSpline::fit() crashes are
  // consistent with an upstream unstable pose-spline update"). Diagnosed
  // live (psd_audit_en instrumentation, ntu_viral/eee_01/n_c=13): the GN
  // solve diverges EXPONENTIALLY starting scan 1 iteration 1
  // (||delta_c_pos|| on the FREE control points: 0.036m -> 324m -> 3.1e6m
  // across 3 iterations, in every IMU/curvature weight configuration
  // tested) -- an unconstrained Gauss-Newton step with no line search or
  // trust region, exactly the gap item 15 names. This is a simple
  // dogleg-style safeguard: if the worst free-control-point step this
  // iteration exceeds the bound, the ENTIRE solved delta vector (both
  // pos and rot sub-blocks, plus the head prior's own delta) is scaled
  // down by one scalar so the direction is preserved and only the length
  // is capped -- cheaper than a real trust region (no re-solve), but
  // directly stops the geometric blowup at its source rather than only
  // detecting it after the fact (ldlt.info()/allFinite() checks above).
  // 0.0 = off (md5-inert at 0.0, matching every other numerics knob in
  // this file). Non-zero defaults below are a FIRST proposed value (the
  // sane iteration-0 step measured live was ~0.036m/2e-4rad -- these caps
  // sit roughly an order of magnitude above that), not a validated
  // choice -- report the actual number this produces before trusting it.
  double pose_gn_max_step_pos_m = 0.5;
  double pose_gn_max_step_rot_rad = 0.2;

  // ==========================================================================
  // User instruction 2026-09-21 ("do all of them", the 32-item physical-
  // knot redesign) -- estimator/coupled/pose_knots/*, meaningful only under
  // spline_mode=pose_knots (poseKnotsBasis()). See pose_knot_spline.h and
  // estimateCoupledPoseKnotSpline() for the actual mechanism; these are
  // just its tunables. Item 31's own "first experiment" config: n_knots=13
  // (matching this project's usual n_c range), smoothness OFF, IMU factors
  // ON, point_time by default (this arm reuses the SAME
  // estimator/coupled/jacobian_time_mode key the other two arms read --
  // item 9's own "keep point_time/end_time as an orthogonal experiment").
  // ==========================================================================
  int pose_knots_n = 13;
  // Item 20's own "simpler first implementation" fallback: a first-
  // difference penalty between ADJACENT knots' position/rotation
  // corrections (||delta_p_{j+1}-delta_p_j||^2 etc.), NOT the physically-
  // motivated integral jerk/angular-acceleration form item 20 states as
  // the ideal -- that would need a higher-order interpolation than the
  // cubic Hermite this first implementation uses. 0.0 = off (item 31: "First
  // prove physical knot representation + IMU factors + LiDAR works. Then
  // add smoothness" -- this class of prior is deliberately NOT what
  // enforces trajectory sanity in v1; the IMU process factor is).
  double pose_knots_smoothness_pos = 0.0;
  double pose_knots_smoothness_rot = 0.0;
  // Ablation knob (item 31's own YAML lists it explicitly): with this off,
  // consecutive knots have NO process-factor link at all (only the
  // optional smoothness prior above, if nonzero, and each knot's own
  // LiDAR factors) -- a genuinely degenerate configuration, kept only for
  // testing that the IMU factor is actually doing something.
  bool pose_knots_use_imu_factors = true;
  // POST-REVIEW ADDITION (item 3, "the threshold is still an arbitrary
  // numerical cutoff... add a config parameter and test 1e-4...1e-8,
  // record lambda_min(A)/lambda_max(A)/cond(A) plus the stationary
  // trajectory error. If results are stable, this issue can be
  // retired"): Q9's own pseudo-inverse relative eigenvalue threshold,
  // now a live knob instead of the hardcoded 1e-6 (which remains the
  // default -- this is deliberately NOT re-tuned by this change itself,
  // only made sweepable). Config key: estimator/coupled/pose_knots/
  // q_pinv_rel_thresh. Does NOT affect Omega0's own pseudo-inverse (the
  // head prior P0^-1) -- that one is not the rank-deficient quantity the
  // review is asking about, and stays at its own fixed 1e-6.
  double pose_knots_q_pinv_rel_thresh = 1e-6;

  // Phase-2 diagnostic (2026-09-22 process-factor nullspace investigation):
  // TEMPORARY soft penalty lambda_C*||N_j^T r_j||^2 added on top of the
  // existing r_j^T Q_j^+ r_j process cost, where N_j spans the 3 lowest
  // eigenvectors of that same GN iteration's relinearized Q9 (NOT the
  // configurable q_pinv_rel_thresh nullspace -- deliberately fixed at "the 3
  // smallest" per Phase 1's own dim(null(Q))=3 finding, to avoid confounding
  // this experiment with a q_pinv_rel_thresh sweep). Does NOT touch the
  // existing pseudoinverse, IMU/LiDAR weighting, or anything else -- this is
  // an additive diagnostic term only, default off. See pose_knots_G_rank_check
  // diagnostic / the nullspace-projection diagnostic this builds on.
  bool pose_knots_det_constraint_en = false;
  double pose_knots_det_constraint_weight = 1.0e4;

  // Phase-3 production candidate (2026-09-22): the soft penalty above
  // demonstrated (4/4 md5 gate, 719m->0.04m stationary drift, 16080->1.05
  // m/s^2 accel, kappa(A) 2.4e16->2.1e10 -- see the Phase-2 report) that
  // the pathological pose_knots stationary drift is explained by the bare
  // Q9 pseudoinverse omitting a DETERMINISTIC (zero-noise) component of
  // the process model, not by tuning. This flag replaces the soft penalty
  // with the mathematically exact version: N^T r = 0 enforced as a hard
  // equality constraint via nullspace elimination (delta = delta_p + Z*eta,
  // Z=null(C), C stacking each segment's N_j^T[-F_j I]), while the
  // stochastic term keeps EXACTLY r_stochastic^T Q^+ r_stochastic (Lambda
  // built from the SAME 6 non-null Q9 eigendirections N_j's complement
  // spans, not the separately-thresholded q_pinv_rel_thresh -- avoids
  // double-counting/gap between the two). Default false pending the
  // Phase-3 validation the user requested (exact-constraint solution vs
  // the lambda_C=1e6 soft-penalty solution on stationary eee_01 scan 1)
  // before this becomes the production default. When true, takes
  // precedence over pose_knots_det_constraint_en (soft path is skipped,
  // with a one-time diagnostic warning if both are set -- they are
  // mutually exclusive, not additive).
  bool pose_knots_exact_det_constraint_en = false;

  // Phase-4 (2026-09-22): whether F9_j/Q9_j are recomputed every GN
  // iteration from the trial's CURRENT (moving) state (true, matches
  // shipped/all-prior-phase behavior) or relinearized ONCE per scan at
  // the first iteration and then held fixed for the rest of that scan's
  // GN iterations (false). Re-testing this axis now that the exact
  // deterministic constraint exists -- the earlier finding ("frozen ->
  // oscillatory stationary error, relinearized -> catastrophic
  // divergence") was obtained while the process model was missing that
  // constraint, so it needs to be reconsidered. Default true (unchanged
  // live behavior).
  bool pose_knots_relinearize_fq = true;

  // Velocity-observability campaign (2026-09-22), Part 7: which knot
  // velocity DOFs the GN solve actually optimizes. Meaningful only under
  // spline_mode=pose_knots. Default "free_per_knot" is EXACTLY today's
  // shipped behavior (no behavioral change at default). Config key:
  // estimator/coupled/pose_knots/velocity_mode.
  //   free_per_knot -- current implementation, N independent 3D velocity
  //     corrections (one per knot), unchanged.
  //   fixed_nominal -- knot velocities stay at their IMU-propagated
  //     nominal values for the whole scan (delta_vel is a hard-
  //     constrained-to-zero homogeneous equality, NOT a huge prior and
  //     NOT set to zero VALUE -- the nominal velocity itself is whatever
  //     the IMU predicted, typically nonzero); zero optimized velocity
  //     DOFs.
  //   shared_scan -- exactly one shared 3D velocity correction
  //     delta_v_scan, hard-constrained equal across every knot
  //     (v_j = v_j_nominal + delta_v_scan for all j); 3 optimized
  //     velocity DOFs total instead of 3*N.
  //   derivative_defined (2026-09-22, pose-spline-derivative-state
  //     campaign) -- v_j is HARD-CONSTRAINED to equal the analytic
  //     time-derivative of the position-only (Catmull-Rom) spline
  //     through the position control points, NOT frozen and NOT free:
  //     v_j = (p_{j+1}-p_{j-1}) / (t_{j+1}-t_{j-1}) for interior knots,
  //     one-sided at the two boundary knots -- an AFFINE equality
  //     constraint (nonzero d_exact, unlike fixed_nominal/shared_scan's
  //     homogeneous rows), so v_j's optimized value changes automatically
  //     whenever a neighboring POSITION control point's own correction
  //     changes, with zero independent velocity DOFs. See
  //     estimateCoupledPoseKnotSpline()'s own derivation comment at the
  //     velmode_derivative constraint-row-fill site for the exact algebra
  //     and why this reuses the exact-constraint nullspace-elimination
  //     machinery rather than needing a new solver path.
  // Implemented as ADDITIONAL equality-constraint rows appended to the
  // SAME nullspace-elimination machinery exact_deterministic_constraint_
  // enable already uses (see estimateCoupledPoseKnotSpline()'s C_exact/
  // Z_ns construction) -- every velocity_mode value composes cleanly with
  // that flag (both independently on/off), rather than needing a
  // separate code path.
  std::string pose_knots_velocity_mode = "free_per_knot";

  // CQ-85 item 1: rule 58f's exact failure mode -- CQ-72's own 96-cell grid
  // produced a cell reporting completed=yes with ATE=396,499,288.300 mm (a
  // FAILED run that looked like a successful one; imu_deviation_weight=0,
  // point_time, n_c=8, eee_02). Loud, ABORT-style guard on the raw_imu
  // arm's own per-scan displacement (measured against the IMU-propagated
  // prediction, coupled_prop_.pos1, the same quantity a healthy scan's
  // correction should stay close to) -- NOT a fix for the underlying
  // failure, only a refusal to let it masquerade as a completed run, same
  // spirit as the residual-starvation guard (CQ-61 item 4) right above its
  // insertion site. <= 0.0 DISABLES the check entirely (the shipped
  // default -- a numerics default is Bryce's call, rule 26; this card
  // only PROPOSES turning it on). Config key:
  // estimator/coupled/max_scan_displacement_m.
  double max_scan_displacement_m = 0.0;
  // Item 3d(ii): the DC-component/bias split is exactly rank-deficient by 6
  // (a constant delta_a and a constant -delta_ba are indistinguishable over
  // one scan, same for delta_omega/delta_bg) and solvable only via the
  // Lambda-vs-bias-prior ratio unless this is set. When true, constrains
  // sum_samples beta(t)^T c = 0 per axis (6 linear equality rows) via a
  // KKT-bordered solve, giving c the within-scan SHAPE only and the bias the
  // offset. Default false (the un-constrained, prior-determined arm).
  bool zero_mean = false;
  // Diagnostic/ablation toggle: drops c_gyr (the within-scan rotation
  // CORRECTION basis) out of the joint solve entirely, leaving delta_bg
  // (a single rigid per-scan gyro-bias correction) as the only rotation
  // correction mechanism. Default false.
  bool disable_cgyr = false;
  // CQ-50/CQ-52 addendum: H's rotation-Jacobian column and Phix_pt/Phic_pt
  // (the state/coefficient sensitivity) must be evaluated at the SAME time
  // for H_k*Phi(t_k) to be a valid chain rule -- CQ-50's original diagnosis
  // was exactly this mismatch (H built once from the deskewed point + the
  // scan-end state_->rot() in lio_base.cpp's buildResiduals(), chained
  // against Phi interpolated at each residual's own capture time res.t;
  // valid only by coincidence at rest). phi_at_scan_end and h_at_point_time
  // used to be two INDEPENDENT booleans, which made all four combinations
  // reachable even though only two are ever meaningful -- both false is the
  // original mismatch (H@t1, Phi@t_k) and both true is an equally invalid,
  // never-used mirror-image mismatch (H@t_k, Phi@t1). Collapsed into one
  // mode, following the project's own standing rule (CQ-44 item 4c: "mutually
  // exclusive options are one mode, never two flags that can express a
  // meaningless combination"):
  //   "legacy_mismatched" (DEFAULT, unchanged from the prior bool defaults'
  //     shipped behavior -- H@t1, Phi@t_k -- kept as the default so this
  //     refactor changes no one's numerics without Bryce's say-so, rule 26
  //     item 1) -- the ORIGINAL bug CQ-50 diagnosed.
  //   "end_time" (was phi_at_scan_end=true): both H and Phi forced to the
  //     scan-end time t1 -- restores a valid chain rule, but every residual
  //     now shares the identical Phi, so all within-scan resolution is
  //     destroyed. CQ-52 item 1 found this INERT (correction magnitudes
  //     near-zero relative to sensor noise) -- consistent, but useless.
  //   "point_time" (was h_at_point_time=true): both H and Phi evaluated at
  //     each residual's own capture time t_k -- H via raw_body_point.cross(
  //     worldRotAt(t_k)^T * normal) instead of point_cross_normal (built
  //     once at t1). Valid chain rule AND genuine within-scan resolution
  //     preserved -- architecturally the correct fix, verified against
  //     CQ-49's md5 pairs.
  std::string jacobian_time_mode = "legacy_mismatched";
  // CQ-53 item 1: disabled-by-default pivot floor on the joint matrix's own
  // LDLT decomposition -- mirrors SplineOptions::PIVOT_MIN_FLOOR's shipped
  // no-op default (-1.0, below any real pivot this system produces) exactly.
  // No threshold has been validated for this system yet (rule 26: a real
  // numerics default is Bryce's call, not this session's) -- this exists so
  // the refusal PATH is exercised/testable, not to actually gate anything
  // at its shipped value.
  static constexpr double JOINT_PIVOT_MIN_FLOOR = -1.0;
  // CQ-53 item 6: commits the second throwaway debug harness this session
  // used (an env-var-gated print comparing each residual's delta_phi0/
  // delta_p0 Jacobian-column magnitude against its own capture-time
  // fraction within the scan) as a real, opt-in diagnostic instead of
  // leaving it deleted. Off by default -- per-residual, so real cost when
  // on. See jrow_leverage.txt's own write site for the exact columns.
  bool log_jrow_leverage_en = false;
  // CQ-52 item 2, the decisive test: when true, inflates delta_bg's own
  // prior-precision diagonal in Pi_ss to near-infinite, holding the gyro
  // bias fixed (the accumulator) while leaving c_gyr/c_acc's within-scan
  // correction (the forcing) fully active -- isolates whether h_at_point_
  // time's incomplete fix is the forcing's form or the accumulator
  // compounding a persistent one-sided error into a walk. Default false.
  bool freeze_bg = false;
  // CQ-54 item 1, the fix: sigma_a/sigma_g (which set Lambda's prior
  // stiffness, item 0c) are read ONCE from the pre-flight calibration
  // window and never revised -- during a vibration transient (e.g. drone
  // motor spin-up) the true IMU noise runs 15-55x that floor, so Lambda is
  // ~225-3000x too stiff exactly when the within-scan correction most needs
  // to absorb the inflated residuals, and the (exactly degenerate, item 0d)
  // signal is routed entirely into the persistent gyro bias instead. When
  // true, sigma_a/sigma_g are re-estimated per scan from a rolling window of
  // mg.imu_samples_raw (see ImuProc::loadParameters()'s keep_raw_samples
  // extension), floored at the calibration value (can only inflate, never
  // shrink below the sensor's own floor -- ImuProc's own calibration is
  // still trusted as the noise-floor lower bound). Default false, no-op
  // (md5-inert at defaults).
  bool adaptive_sigma = false;
  // CQ-54 item 3, the cheaper guard: when true, applies the SAME Pi_ss
  // inflation freeze_bg uses (see :523 in the .cpp), but only for scans
  // where the rolling-window IMU noise exceeds its calibration floor by
  // more than bias_freeze_vibration_factor -- released once the noise
  // settles, rather than held for the whole run the way freeze_bg is.
  // Physically: a gyro bias is a slow thermal drift with no reason to
  // update during a brief vibration transient, which is exactly the window
  // where the data cannot distinguish bias from noise (item 0d's
  // degeneracy). Default false.
  bool bias_freeze_on_vibration = false;
  // Threshold for bias_freeze_on_vibration, disabled-by-default in effect
  // since the guard itself defaults off -- no validated threshold exists
  // yet (rule 26: a real numerics default is Bryce's call), mirrors
  // JOINT_PIVOT_MIN_FLOOR's own shipped-inert-until-validated pattern.
  static constexpr double BIAS_FREEZE_VIBRATION_FACTOR_DEFAULT = 3.0;
  // CQ-55 item 3: runtime-overridable threshold (was previously a hardcoded
  // constexpr, blocking a sweep over its value) -- defaults to the same
  // BIAS_FREEZE_VIBRATION_FACTOR_DEFAULT, so this is md5-inert unless
  // explicitly overridden via estimator/coupled/bias_freeze_vibration_factor.
  double bias_freeze_vibration_factor = BIAS_FREEZE_VIBRATION_FACTOR_DEFAULT;
  // CQ-54 item 6: penalises the TOTAL departure of the gyro bias from its
  // calibration value (state_->biasGyr() + coupled_delta_bg_ -
  // coupled_bg_calib_), separately from Pi_ss's own per-scan-increment
  // prior, which never prices the accumulated total (the ratchet item 6
  // names). Default false, no-op at defaults.
  bool bias_anchor = false;
  // Provisional precision for bias_anchor, disabled-by-default in effect
  // since the flag itself defaults off -- no validated in-run bias-drift
  // bound exists yet (rule 26). Set generously above the quoted MEMS
  // in-run stability (~10 deg/hr = 0.0028 deg/s) at 0.05 deg/s (=
  // 8.72665e-4 rad/s) as a starting point for item 6's own run to react
  // against, not a validated physical spec.
  static constexpr double BIAS_ANCHOR_SIGMA_RAD_S_DEFAULT = 8.72665e-4;
  // CQ-59 item 1: second-difference curvature penalty weight on the
  // correction coefficients, added to Lambda alongside (not instead of)
  // the existing basis-Gram value term. SPLIT (was one raw
  // curvature_weight, CQ-55 item 6) and NORMALIZED by the same sigma^2 the
  // value term uses -- the old raw knob landed against a gyro base prior
  // ~4657x stiffer than the accelerometer's (sigma_a=0.00434,
  // sigma_g=6.36e-05 on eee_01/02: (sigma_a/sigma_g)^2 ~ 4657), so one raw
  // number could not mean the same thing on both blocks. Dimensionless now
  // ("this shape penalty is worth w times the value penalty" on THIS
  // block) -- 1.0 is a meaningful default, not an arbitrary number. Both
  // default 0.0 -- md5-inert. The old single curvature_weight knob is
  // REMOVED, not kept as a deprecated alias: it was never set in any
  // checked-in config (grepped config/, scripts/ -- zero hits) and always
  // defaulted to 0.0, so nothing depends on it continuing to exist, and a
  // permanent alias for a knob nothing ever used is pure upkeep cost.
  //
  // CQ-72 item 0a: renamed from curvature_weight_acc/curvature_weight_gyr
  // to smoothness_weight_acc/smoothness_weight_gyr (config key moves from
  // estimator/coupled/curvature_weight_{acc,gyr} to estimator/coupled/
  // prior/smoothness_weight_{acc,gyr}) -- states the BELIEF this term
  // encodes ("the correction should not wiggle") rather than the math
  // operator (D^T D) that implements it. Same type, same default, same
  // arithmetic -- a pure rename, not a new knob.
  double smoothness_weight_acc = 0.0;
  double smoothness_weight_gyr = 0.0;
  // CQ-69: "the trajectory change caused by the correction should be
  // small" as a THIRD prior band, distinct from gram (mid-band, IID-noise-
  // correct value prior) and curvature (high-band, second-difference shape
  // penalty). Lambda_traj = sum_k phi_head[k]^T W phi_head[k], W picking
  // out the POSITION rows only (rows 3-5 of phi_head[k]'s 9 rows) -- phi_head
  // already bakes in the basis weights, the world-frame rotation, and the
  // double integration (see estimateCoupledCorrection()'s own construction),
  // so this is an EXACT low-band trajectory-deviation penalty, not an
  // approximation built from a hand-rolled integration operator (which
  // would integrate the correction in the wrong frame). Normalized by
  // (t1-t0)^2 inside the construction -- the term's own DC weighting scales
  // as T^2, so an un-normalized knob would silently mean something different
  // at every scan duration/LiDAR rate. Default 0.0 -- md5-inert (adds
  // nothing to Lambda at that default). A regularization weight, not a
  // variance -- never described as one in a filing.
  //
  // CQ-72 item 0a: renamed from lambda_traj_pos to traj_deviation_weight
  // (estimator/coupled/lambda_traj_pos -> estimator/coupled/prior/
  // traj_deviation_weight) -- same reason as above, pure rename.
  double traj_deviation_weight = 0.0;
  // CQ-55 item 11(b): the principled, threshold-free alternative to
  // freeze_bg/bias_freeze_on_vibration -- see estimateCoupledCorrection()'s
  // own comment at the accumulation site for the full mechanism. Default
  // false.
  bool bias_observable_only = false;
  // CQ-72 items 0a/0b: REPLACES curvature_only (bool, default false --
  // "IMU prior off" and "mean prior on" were the SAME switch, the exact
  // thing this card exists to fix) with a plain multiplier on the existing
  // gram(i,j) value term: value_imu = imu_deviation_weight * gram(i,j).
  // Config key estimator/coupled/curvature_only -> estimator/coupled/
  // prior/imu_deviation_weight. Default 1.0 (== "the IMU prior is fully
  // on", i.e. today's curvature_only=false behavior, since 1.0*gram(i,j)
  // == gram(i,j)); 0.0 means fully off (== today's curvature_only=true's
  // effect on this one term, MINUS the mean-prior coupling -- see
  // mean_weight below, now independent). NOT a bool any more: rule 61/
  // CQ-72's own on/off grid only ever sets it to {0.0, 1.0}, but the type
  // is a double so a later card can ask a magnitude question without a
  // second rename.
  double imu_deviation_weight = 1.0;
  // CQ-72 items 0a/0b: REPLACES dc_weight (double, default 0.0, only ever
  // READ when curvature_only was true -- gated behind the SAME switch that
  // turned the IMU prior off, so "IMU prior off" and "mean prior on" could
  // never be set independently). Config key estimator/coupled/dc_weight ->
  // estimator/coupled/prior/mean_weight. Precision (relative to 1/sigma^2,
  // same scale the gram term uses) of an explicit DC/mean-direction-only
  // prior: mean_acc = mean_weight / sigma_a^2, mean_gyr = mean_weight /
  // sigma_g^2 -- now applied UNCONDITIONALLY (no longer gated on
  // imu_deviation_weight being 0), so the two beliefs ("corrections should
  // be small" and "corrections should not carry a constant offset") are
  // independently switchable, which is the whole point of this rename.
  // Default 0.0 -- md5-inert at that default (identical arithmetic to
  // today's curvature_only=false/dc_weight=0.0 case, since dc_acc/dc_gyr
  // were already always 0 then).
  double mean_weight = 0.0;
  // CQ-72 item 3: "how much does the trajectory the correction implies
  // actually move, per iteration and per frame" -- the literal thing the
  // card asks for and which delta_c_norm/delta_c_acc_norm/delta_c_gyr_norm
  // (CQ-53 item 3, coefficient-space) do not answer, since coefficient
  // space and trajectory space are different objects with different units.
  // Default false, md5-inert (pure logging, touches no solve/state path).
  bool log_traj_dev_en = false;
  // CQ-59 item 4: builds the Lambda VALUE term's 3x3 blocks as a diagonal
  // from the calibration's real per-axis noise floor (state_->varAccFloor()/
  // varGyrFloor(), now genuinely anisotropic -- see calib_processing.cpp)
  // instead of the scalar sigma^2 * Identity every other term still uses.
  // "Anisotropy with an unimpeachable source -- the sensor's own measured
  // noise" (the card's own words), independent of the residual-derived
  // anisotropy CQ-58 reports and deliberately does NOT feed back in (see
  // estimateCoupledCorrection()'s own comment on why that route is
  // wrong). Default false -- md5-inert.
  bool prior_per_axis_sigma = false;
  // CQ-61 arm (b): outlier-robust IRLS weighting on each residual's own
  // normalized value -- "none" (default, md5-inert), "huber", or "cauchy".
  // See estimateCoupledCorrection()'s own accumulation-loop comment for
  // the formula and why this treats the symptom (bad correspondences
  // dominating A), not Bug A's own mechanism (that's arm (a),
  // pose_cov_in_sigma, a pre-existing shared-code flag -- see voxelplane.cpp).
  std::string robust_loss = "none";
  // CQ-62 item 0b/1: staged PSD audit -- see the diagnostic block at
  // posterior18's own computation site in lio_coupled.cpp for what this
  // logs (psd_audit.txt). Default false, md5-inert (report-only, no
  // state mutation).
  bool psd_audit_en = false;
  // CQ-55 item 11(a): report-only, free -- log each scan's bg-block
  // posterior covariance eigenvalues (degenerate vs. well-observed
  // directions), independent of whether bias_observable_only itself is
  // engaged. Default false (the extra ncol x ncol inverse this requires is
  // not free enough to run unconditionally).
  bool log_bg_projection_en = false;
  // CQ-57 item 2: report-only, default false. Re-propagates a SECOND
  // covariance (R,P,V 9x9 only) alongside the corrected trajectory's own
  // Phi/Phix, WITH proper process-noise reinjection at each step (unlike
  // Phix's own bare propagation) -- answers "how far apart are the raw
  // (imu_processing.cpp) and corrected covariance propagations." Never
  // written into state_->covMut(). See coupled_estimator.h's propagateCoupled()
  // doc comment for the actual math.
  bool log_cov_repropagation_en = false;
  // Same imu/* keys imu_processing.cpp itself reads (imu/q_alpha_gyr,
  // imu/q_alpha_acc, imu/second_order) -- read again here (not shared)
  // since this class has no reference to ImuProc's own options struct.
  // Only used when log_cov_repropagation_en is true.
  double repro_q_alpha_gyr = 1.0, repro_q_alpha_acc = 1.0;
  bool repro_second_order = true;
  // CQ-57 item 3a: report-only... no, THIS one changes behaviour (adds
  // process noise to the written posterior) -- default false, md5-inert.
  // The joint solve carries ONE bias correction at t0; M*A^-1*M^T reports
  // the bias at t1 as exactly as well known as at t0, with no random-walk
  // growth over the scan's own duration. Adds q_alpha_bias*covBiasGyr()*dt
  // / q_alpha_bias*covBiasAcc()*dt (dt = the WHOLE scan duration, t1-t0 --
  // the gap this item names) to posterior18's own bg/ba diagonal blocks
  // before it is written, using the SAME imu/q_alpha_bias rate
  // imu_processing.cpp's own cov_w already uses (state.cpp:259-269's own
  // comment on why a per-axis RATE, not the initial-covariance value, is
  // the right quantity here).
  bool q_bias_rw_en = false;
  double q_alpha_bias = 1.0;  // imu/q_alpha_bias, same key imu_processing.cpp reads.
  // CQ-57 item 3b: default false/0.0, md5-inert. The correction spline can
  // represent corrections up to roughly n_c/(2*T_scan) Hz; IMU noise power
  // ABOVE that band contributes no column to A (hence no variance to
  // A^-1/posterior18) while being real error in the integrated
  // trajectory. Report-computed fractions (measured directly from the
  // bag's own raw IMU spectrum, NOT from a model -- see the card's own
  // "REPORT, DO NOT TUNE" instruction): eee_01, fs=197.55Hz (NOT the
  // 385Hz the card assumed -- corrected here), band_edge=20.0Hz at n_c=4/
  // 65.0Hz at n_c=13. Gyro out-of-band fraction is negligible at both
  // (0.05-1.8%); accel out-of-band fraction is LARGE at n_c=4 (31-69% per
  // axis) and small at n_c=13 (2.4-4.0%). These two fields are
  // config-SUPPLIED (not live-FFT-computed in this build -- that is
  // future work, out of this item's own scope) so a caller can inject the
  // already-measured fraction; default 0.0 (inert) leaves the shipped
  // behaviour unchanged regardless of q_out_of_band_en.
  bool q_out_of_band_en = false;
  double q_out_of_band_scale = 1.0;
  double q_out_of_band_fraction_acc = 0.0, q_out_of_band_fraction_gyr = 0.0;
  // CQ-58: report-only, free -- per-control-point 6-dof constraint report
  // (cp_constraint.csv). Reuses the SAME A^-1 inversion bias_observable_only/
  // log_bg_projection_en already pay for when either is also on; computes
  // its own if neither is (see estimateCoupledCorrection()'s shared-inversion
  // comment). One line per control point per scan -- n_c*n_scans lines,
  // buffered rather than flushed per line (same convention
  // log_jrow_leverage_en uses). Default false.
  bool log_cp_constraint_en = false;
  // CQ-75: re-deskew/re-propagate against the FINAL accepted GN step
  // (previously the published pose's own map-deskew input, mg.points,
  // and the covariance/A construction all reflected the PENULTIMATE
  // iteration's trajectory -- see this option's own use site for the
  // precise, verified structural finding). final_relinearize_cov requires
  // final_redeskew (enforced at load time).
  bool final_redeskew = false;
  bool final_relinearize_cov = false;
  // CQ-76 T2.1: when true, Pi_ss is built from the pre-propagation P
  // snapshot (imuProcQhatPeekPBefore(), "P(t0)" in the OTHER sense --
  // before THIS scan's own IMU propagation, i.e. the previous scan's own
  // posterior) instead of state_->cov() (the post-propagation prior,
  // confirmed correct by CQ-76 T0.1's own call-chain proof). A
  // deliberate sensitivity probe, not a claim this alternative is more
  // correct -- requires imu/log_qhat_en=true (enforced at load time).
  bool prior_at_scan_start = false;
  // CQ-76-R3: the principled pairing T2.1 alone did not test. Adds the
  // frame-local Q_scan (T1.0's own g_qhat_accum_cov_w, already captured
  // every scan by imuProcQhatRead()) to the rot/pos/vel 9x9 block of the
  // 18x18 posterior AFTER the normal M*A^-1*M^T write -- requires
  // imu/log_qhat_en=true (enforced at load time), md5-inert at false.
  bool add_q_scan_to_posterior = false;
  // CQ-75-R2: arm (e), isolating the map channel. final_redeskew still
  // runs (governs the solve-side report/residual rebuild as before) --
  // this flag ADDITIONALLY reverts mg.points to its pre-final-redeskew
  // value right before returning, so updateMap() (called by the caller
  // after this function returns) sees the ORIGINAL points rather than
  // the final-redeskewed ones. Requires final_redeskew=true (enforced at
  // load time, same pattern as final_relinearize_cov). md5-inert at false.
  bool final_redeskew_map_uses_pre = false;
  // CQ-79: report-only, log-only, reads quantities the accumulation loop
  // already holds (struct Residual's own r/t/plane_id/sigma_squared) --
  // no new computation, never touches the solve. Default false,
  // md5-inert. hist_start_scan/hist_n_scans name the one 20-scan window
  // Artifact 2's full per-bin detail is written for (Artifact 1's own
  // per-scan line fit is written for every scan, unconditionally, once
  // the flag is on).
  bool log_point_plane_en = false;
  int  log_point_plane_hist_start_scan = 0;
  int  log_point_plane_hist_n_scans = 20;
};

// CQ-49: the coupled estimator, reimplemented as its own class -- see
// lio_base.h's own doc comment for why (bug 4 on the old interleaved branch:
// the spline still ran under coupled mode because nothing PREVENTED it from
// running, only a flag that happened not to be set correctly). This class
// has NO ScanSpline member and NO spline call site anywhere -- the
// structural guarantee that bug 4's class of defect cannot recur here, not
// merely a fixed instance of it.
//
// ONE update, ONE set of variables, no endpoint discrepancy by construction:
// the LiDAR points correct an IMU measurement correction defined over the
// scan (coupled_estimator.h); that correction is re-propagated to produce
// the trajectory; the endpoint of that trajectory IS the state. There is no
// second correction applied afterwards and therefore no endpoint to
// reconcile.
//
// STATUS (2026-09-19): items 3c/3d/5's joint [delta_x(t0) 18, c] solve is
// implemented, with the three real bugs CQ-44 item 3f found (Fx never
// seeded with the identity; M missing the pose columns entirely; the
// gravity0/gravity stripping mismatch) already fixed from the start in this
// reimplementation -- see coupled_estimator.cpp. The motion-onset
// divergence TQ-40 found (coupled diverges to >100km once real dynamics
// begin, while the static window is healthy) is NOT yet root-caused and is
// NOT fixed here -- this class reproduces the same estimateCoupledCorrection()
// math, just with bug 4's contamination structurally removed. No G0/G1/G2
// gate has been passed. Default estimator/mode stays "decoupled"; nothing
// here is reachable unless a config explicitly sets estimator/mode: coupled.
class LioProcCoupled : public LioProcBase
{
public:
  explicit LioProcCoupled(NodeContext& ctx);
  // CQ-55 "effective-config report" standing requirement (CQ-54 item 8):
  // prints engagementReport() at shutdown and writes it to
  // debugLogPath("engagement.txt"), mirroring LioProcDecoupled's own
  // destructor exactly (see lio_decoupled.cpp) -- previously declared
  // =default here, so this class's own engagementReport() string was dead
  // code, never actually printed anywhere.
  ~LioProcCoupled() override;

  std::string loadParameters(ros::NodeHandle& pnh) override;
  std::string engagementReport() const override;
  void deskewAndDownsample(MeasureGroup& mg) override;
  std::string processLIO(MeasureGroup& mg) override;

  // CQ-44: ONE Gauss-Newton step of the coupled estimator. Re-propagates
  // mg.poses with the current coefficient AND delta_s(t0) estimate
  // (coupled_estimator.h), re-deskews mg.points against that corrected
  // trajectory, rebuilds residuals (buildResiduals() -- the SAME shared
  // method the decoupled path uses), solves the joint normal equations, and
  // applies the step. Does NOT touch estimateStateCorrection() at all.
  //
  // CQ-82 Phase 1: this is now a DISPATCHER. Everything before the
  // coefficient-block prior/residual-loop (propagate, re-deskew+downsample,
  // buildResiduals, the state-prior Pi_ss) is basis-independent and stays
  // here; the coefficient-block prior + the normal-equations residual loop
  // (previously inline) moved verbatim into buildImuCorrectionSystem() --
  // "moved", not rewritten: every accumulator/diagnostic it used to fill is
  // still filled, just via the returned CoupledSystemBuild rather than a
  // bare local. Everything from the LDLT solve onward (rank diagnostics,
  // the solve itself, the delta application, bg-projection, re-propagation)
  // is unchanged and still lives here, since none of it depends on which
  // basis built A/b -- only their CONTENTS do.
  double estimateCoupledCorrection(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);

private:
  // CQ-82 Phase 2, Artifact 1: the pose basis's own GN-iteration path.
  // Branched to from estimateCoupledCorrection()'s very top when
  // copts_.poseBasis() -- kept SEPARATE from the raw_imu dispatcher rather
  // than threaded through it, since the raw_imu path's own downstream
  // bookkeeping (bg-projection, delta_bg/delta_ba/delta_v/delta_g
  // accumulation, coupled_c_acc_/coupled_c_gyr_'s own semantics) does not
  // apply to a basis whose c-block is position/attitude, not accel/gyro
  // corrections -- a single top-level branch is easier to verify never
  // affects the raw_imu path than threading pose-specific conditionals
  // through ~600 lines of shared code. Deskews against the current trial
  // spline (coupled_pose_spline_ + coupled_c_pos_/coupled_c_rot_) via the
  // EXISTING, already-shared deskewPointsSpline() (lio/deskew.h, also used
  // by the decoupled spline path), reuses buildResiduals() (also shared,
  // non-virtual), builds the c-block system via buildPoseSplineCBlock(),
  // solves, applies the correction, and writes the resulting trajectory's
  // tail pose into state_ directly (no propagateCoupled()-equivalent
  // needed -- the pose basis's own trajectory already IS an absolute
  // pose, not a correction requiring re-propagation).
  double estimateCoupledCorrectionPoseBasis(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);
  // User instruction 2026-09-21 item 24: a NEW solver, not a further
  // extension of estimateCoupledCorrectionPoseBasis() ("that name now
  // describes the old coefficient-based architecture"). Owns: knot states/
  // covariance (coupled_pose_knots_), the joint per-GN-iteration
  // information system (item 14's "mathematically clean" batch form -- ALL
  // knots solved together, not independent sequential updates), IMU
  // process factors between adjacent knots (item 16), LiDAR factors
  // attached to each residual's own bracketing knot pair (item 8/9), and
  // the joint posterior A^-1 this class reads knot marginal/cross
  // covariance from directly (items 12/13/27 -- see pose_knot_spline.h's
  // own doc comment for why no basis-coefficient translation is needed
  // here, unlike coupled_pose_head_cov_ above).
  double estimateCoupledPoseKnotSpline(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);
  // 2026-09-22: one GN iteration of the pose-control-point-only estimator
  // (spline_mode=pose_control). Rebuilds LiDAR + process-factor normal
  // equations fresh each call against z=[c_free;sT] (coupled_pose_control_layout_),
  // solves, applies the mean update in place to coupled_pose_control_spline_
  // (free control points) and coupled_pose_control_sT_{bg,ba,g}_, reconciles
  // the tail (R/p/v read directly off the optimized spline at t1, never an
  // independent variable), and writes state_->setPropagatedState(...) every
  // call -- exactly like estimateCoupledPoseKnotSpline()'s own pattern.
  // Covariance is NOT written here (spec: mean every iteration, covariance
  // once after convergence) -- see processLIO()'s own
  // copts_.poseControlSplineBasis() early-return block for that.
  double estimateCoupledPoseControlSpline(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);
  // CQ-82 Phase 1: everything buildImuCorrectionSystem() (and, in Phase 2,
  // its pose-basis sibling) hands back to the dispatcher besides A/b itself
  // -- every sum/accumulator the residual loop used to leave in a bare local
  // for the bookkeeping below it to read. A struct, not a longer parameter
  // list of out-params, so the builder's own signature stays "the inputs it
  // needs" and nothing about its RETURN shape leaks into the call site
  // beyond one name. Deliberately NOT M -- solveCovarianceFromA()/the
  // covMut write are not called from inside estimateCoupledCorrection() at
  // all (they live in processLIO(), reading coupled_last_A_ after the GN
  // loop converges), so there is no "M" to thread through here; the card's
  // own dispatcher skeleton names it as an aspiration this function does not
  // actually need to satisfy.
  struct CoupledSystemBuild
  {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double sum_abs_r = 0.0, sum_sq_r = 0.0, sum_wr2 = 0.0, sum_sigma_squared = 0.0;
    double sum_floor_S = 0.0, sum_sdiag_S = 0.0, sum_pvar_S = 0.0, sum_prior_pose_S = 0.0;
    double sum_weight_this_iter = 0.0;
    std::vector<double> hcol_reldiff;
    Eigen::Matrix<double, 6, 6> HtH_pose_lidar = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> Htz_pose_lidar = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> H6_raw_accum = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::MatrixXd phic_spread_sum;
    double phic_spread_sumsq = 0.0;
    int phic_spread_n = 0;
    // CQ-83: per-residual nu^2/S and nu^2/(S-s_prior_pose), accumulated
    // alongside sum_floor_S et al (same acceptance condition: floor_term/
    // sigma_diag_squared/s_prior_pose all >= 0.0) so LioFrameDiag's nis/
    // nis_est columns have a genuine coupled source instead of sitting at
    // their -1.0 sentinel forever.
    double sum_nis = 0.0;
    int n_nis = 0;
    double sum_nis_est = 0.0;
    int n_nis_est = 0;
  };

  // CQ-82 Phase 1: the shipped arm ("raw_imu"/"imu_correction"), MOVED
  // verbatim from estimateCoupledCorrection() -- the coefficient-block prior
  // (gram/Curv/Lambda, item 3b/4/CQ-55/CQ-59/CQ-69) and the normal-equations
  // residual loop (H/Phix_pt/Phic_pt/Jrow, item 3c, CQ-50/CQ-53/CQ-61/CQ-66),
  // including the bias_anchor block and coupled_last_A_/coupled_last_Lambda_
  // writes. No new math anywhere in this function; only the split itself is
  // new. Pi_ss/s_vec (the state-block prior) are built by the caller and
  // passed in, since they do not depend on the coefficient-block basis.
  CoupledSystemBuild buildImuCorrectionSystem(
      MeasureGroup& mg, double t0, double t1, int n_c, int ncol, int ncol_s, int ncol_c,
      double sigma_a, double sigma_g, double sigma_a_floor, double sigma_g_floor,
      const Eigen::MatrixXd& Pi_ss, const Eigen::VectorXd& s_vec);

  LioProcCoupledOptions copts_;

  // Persisted across this scan's own GN iterations (reset at the top of
  // processLIO() each frame); NOT carried scan-to-scan -- c_prior = 0 every
  // scan (item 4).
  std::vector<V3D> coupled_c_acc_, coupled_c_gyr_;
  // CQ-82 Phase 2, Artifact 1: the pose-basis analogue. coupled_pose_spline_
  // is fit ONCE per scan (first GN iteration only, via ScanSpline::fit() on
  // the IMU-propagated mg.poses -- "the matching initial condition to the
  // other arm's c=0", per the card) and never re-fit within the scan;
  // coupled_c_pos_/coupled_c_rot_ are the accumulated corrections ON TOP of
  // that fixed fit's own control points, persisted across this scan's GN
  // iterations the same way coupled_c_acc_/coupled_c_gyr_ are, reset to
  // zero (and the spline re-fit) at the top of every new scan. Only
  // meaningful when copts_.poseBasis() -- untouched, unread, on the raw_imu
  // path.
  ScanSpline coupled_pose_spline_;
  bool coupled_pose_spline_valid_ = false;
  std::vector<V3D> coupled_c_pos_, coupled_c_rot_;
  // CQ-87 item 3: the pose arm's own posterior for [delta_phi0;delta_pos0]
  // (SAME layout as raw_imu's s_vec.segment<3>(0)/segment<3>(3)) -- the
  // marginal covariance of reducePoseSplineHeadCoupling()'s own reduced
  // system, recomputed every GN iteration (so the LAST one, used for the
  // post-loop state_->covMut() write, reflects the CONVERGED solve --
  // this is what makes final_relinearize_cov genuinely true on this arm,
  // see item 4). Valid only when poseBasis() && pose_head_freeze_cp==0
  // (the real-coupling path); untouched, unread otherwise.
  Eigen::Matrix<double, 6, 6> coupled_pose_head_cov_ = Eigen::Matrix<double, 6, 6>::Zero();

  // User instruction 2026-09-21 items 1-4/24: the physical-knot analogue
  // of coupled_pose_spline_/coupled_c_pos_/coupled_c_rot_ above. Built
  // ONCE per scan (first GN iteration) via PoseKnotSpline::init(); the
  // ACCUMULATED per-knot [delta_theta,delta_p,delta_v] corrections (9*N,
  // same flattened order as the joint solve) persist across this scan's
  // GN iterations, reset at scan start, exactly mirroring coupled_c_pos_/
  // coupled_c_rot_'s own pattern. Only meaningful when
  // copts_.poseKnotsBasis().
  PoseKnotSpline coupled_pose_knots_;
  bool coupled_pose_knots_valid_ = false;
  std::vector<V3D> coupled_knot_delta_theta_, coupled_knot_delta_pos_, coupled_knot_delta_vel_;
  // The joint posterior's own knot-diagonal covariance blocks after the
  // LAST (converged) GN iteration's solve -- items 12/13/27: read
  // DIRECTLY off A^-1 at each knot's own row/col range, no basis-
  // coefficient Jacobian chaining needed (see pose_knot_spline.h). Index
  // j holds knot j's own 9x9 [theta,p,v] marginal.
  std::vector<Eigen::Matrix<double, 9, 9>> coupled_knot_cov_;
  // Phase-4 (2026-09-22, frozen vs relinearized F/Q re-test now that the
  // exact deterministic process constraint exists): when
  // pose_knots_relinearize_fq is false, F9_j/Q9_j are relinearized ONCE
  // per scan (at the first GN iteration, from the scan-start trial --
  // i.e. coupled_pose_knots_ itself, all deltas still zero) and cached
  // here for reuse on every subsequent iteration of that same scan,
  // instead of being recomputed from the moving trial each iteration.
  // Sized N-1 (one per segment), reset alongside coupled_knot_delta_*
  // above at scan start.
  std::vector<Eigen::Matrix<double, 9, 9>> pose_knots_frozen_F9_, pose_knots_frozen_Q9_;
  bool pose_knots_frozen_fq_valid_ = false;
  // The pose-control-point analogue of coupled_pose_knots_ above -- only
  // meaningful when copts_.poseControlSplineBasis().
  //
  // 2026-09-22 correction: the head is a TRUE NULLSPACE ELIMINATION
  // (coupled_pose_control_hns_, built ONCE at scan start from p0/v0/R0 --
  // see buildPoseControlHeadNullspace()) over the RAW 6N control-point
  // space, not a fixed-control-point scheme. The actual GN mean variable
  // is coupled_pose_control_eta_ (6N-9 dim); the spline's cp_p/cp_phi are
  // a DERIVED view (c = c_particular + Z*eta), recomputed via
  // poseControlUnflatten() every time eta changes -- cp_p/cp_phi are
  // never treated as the optimization variable directly.
  PoseControlSpline coupled_pose_control_spline_;
  bool coupled_pose_control_valid_ = false;
  PoseControlFreeLayout coupled_pose_control_layout_;
  PoseControlHeadNullspace coupled_pose_control_hns_;
  Eigen::VectorXd coupled_pose_control_eta_;
  // Tail trial state (item 3/4 of the correction): tail_trial is the ONE
  // coherent current non-trajectory tail state, updated by INCREMENT
  // (delta_bg/ba/g solved by the GN step) every iteration -- never treated
  // as itself the optimization variable (the variable is the increment).
  // tail_prior is the FIXED scan-entry value the increment is measured
  // against (mean prior residual = tail_trial - tail_prior, consistent
  // with the covariance prior's own Omega_ss block -- item 10).
  V3D coupled_pose_control_bg_trial_ = V3D::Zero(), coupled_pose_control_ba_trial_ = V3D::Zero(),
      coupled_pose_control_g_trial_ = V3D::Zero();
  V3D coupled_pose_control_bg_prior_ = V3D::Zero(), coupled_pose_control_ba_prior_ = V3D::Zero(),
      coupled_pose_control_g_prior_ = V3D::Zero();
  // Raw IMU samples bucketed onto the spline's own breakpoint grid ONCE at
  // scan start (bucketing depends only on fixed breakpoint times, not on
  // the moving trial trajectory) -- reused every GN iteration, mirroring
  // coupled_pose_knots_'s own seg_samples_ caching.
  std::vector<std::vector<ImuSample>> coupled_pose_control_seg_samples_;
  // The reduced posterior z=[eta;delta_sT] covariance from the LAST
  // (converged) GN iteration's own information matrix -- written once,
  // post-loop, in processLIO()'s own poseControlSplineBasis() block (never
  // inside the per-iteration solve -- spec: mean every iteration,
  // covariance once after convergence).
  Eigen::MatrixXd coupled_pose_control_P_z_post_;
  // CQ-79: this scan's PREVIOUS iteration's own set of matched-plane
  // hashes, for carry_frac -- reset to empty at scan start (alongside
  // coupled_iters_'s own reset), updated after every iteration's own
  // point-plane logging pass.
  std::unordered_set<std::size_t> coupled_prev_iter_planes_;
  // CQ-44 items 3c/3d: this scan's own ACCUMULATED delta_s(t0) =
  // [delta_v, delta_bg, delta_ba, delta_g] across GN iterations -- added ON
  // TOP OF state_'s own pre-scan v/bg/ba/g (captured once at scan start).
  V3D coupled_delta_v_ = V3D::Zero(), coupled_delta_bg_ = V3D::Zero(),
      coupled_delta_ba_ = V3D::Zero(), coupled_delta_g_ = V3D::Zero();
  // Item 3e(v)/3f bug 2: the full 18-dim joint solve -- delta_phi(t0)/
  // delta_p(t0) are solved for too, not held at zero. This is a one-scan
  // fixed-lag smoother: the previous scan's published pose is genuinely
  // revised by the converged value of these two (the map already built from
  // the old pose is NOT retroactively updated -- a named consequence).
  V3D coupled_delta_phi0_ = V3D::Zero(), coupled_delta_pos0_ = V3D::Zero();
  V3D coupled_v0_pre_, coupled_bg0_pre_, coupled_ba0_pre_, coupled_g0_pre_;
  CoupledPropagation coupled_prop_;
  double coupled_solve_ms_ = -1.0;
  int    coupled_iters_ = 0;

  // Item 3d(i): this scan's own DC-component/bias-split diagnostics, filled
  // by the final GN iteration.
  double coupled_c_acc_dc_over_sigma_ = -1.0, coupled_c_gyr_dc_over_sigma_ = -1.0;
  double coupled_dba_over_sigma_ = -1.0, coupled_dbg_over_sigma_ = -1.0;
  double coupled_c_acc_over_sigma_ = -1.0, coupled_c_gyr_over_sigma_ = -1.0;
  // CQ-56 (spline-magnitude diagnostics): the SAME RMS-per-coefficient
  // accumulated c_acc/c_gyr magnitude as the _over_sigma pair above, but in
  // RAW physical units (m/s^2 for c_acc, rad/s for c_gyr) rather than
  // sigma-normalized -- lets the "how much curvature_weight suppresses the
  // spline" comparison be read directly, without needing sigma_a/sigma_g
  // from the same run to de-normalize it back out.
  double coupled_c_acc_total_norm_ = -1.0, coupled_c_gyr_total_norm_ = -1.0;
  // CQ-55 item 12: S = floor_term + sigma_diag_squared + plane_var_term +
  // s_prior_pose per residual, summed over this scan's accepted residuals
  // -- the absolute-units denominator the floor/sdiag/pvar/prior_pose
  // SHARES (elsewhere, decoupled-only so far) have always been reported as
  // fractions of, with the absolute magnitude itself never logged before
  // now, on EITHER path.
  double coupled_sum_S_ = -1.0;
  // CQ-55 item 11(a): this scan's bg-block posterior covariance (P_bg =
  // A^-1's 9:12,9:12 block), split into its "degenerate" (largest
  // eigenvalue, least-observed) and "well-observed" (smallest eigenvalue)
  // directions -- report-only, computed independent of
  // bias_observable_only's own confidence-weighted correction.
  double coupled_bg_var_degenerate_ = -1.0, coupled_bg_var_observed_ = -1.0;
  // CQ-60 item 5, Tier 1: constructed with copts_.nees_tier1_window_scans
  // once loadParameters() has run (see the constructor) -- one-shot,
  // inert after the bag's own stationary prefix has been consumed.
  Tier1NeesBuffer coupled_tier1_nees_{0};
  double coupled_delta_v_norm_ = -1.0, coupled_delta_g_norm_ = -1.0;
  // Item G1(a)/(c): velocity- and gravity-block posterior trace.
  double coupled_trP_vel_ = -1.0, coupled_trP_grav_ = -1.0;
  // The LAST GN iteration's own joint A matrix, kept so the item-5
  // covariance term can be applied ONCE after the loop converges.
  Eigen::MatrixXd coupled_last_A_;
  // CQ-58 item 2a: the c-block's PRIOR alone (Lambda, as added to A before
  // any residual accumulates into it) -- kept so the residual-only
  // information block I_j = A.block(j,j) - Lambda.block(j,j) can be
  // recovered without a second accumulation pass. Only meaningful when
  // log_cp_constraint_en is on; left default-empty otherwise.
  Eigen::MatrixXd coupled_last_Lambda_;

  // TQ-40 item 3: the coupled-arm equivalent of the decoupled path's own
  // ask/got/refusal discriminator, restricted to the [delta_phi0, delta_p0]
  // 6-dim sub-block of the FINAL GN iteration's pure-LiDAR-info
  // accumulation -- a NAMED APPROXIMATION (not marginalised over
  // v/bg/ba/g/c, so it overstates the information actually available to
  // phi0/p0 alone by whatever those directions correlate away).
  double coupled_ask_ = -1.0, coupled_got_ = -1.0, coupled_refusal_ = std::numeric_limits<double>::quiet_NaN();
  int    coupled_n_residuals_ = -1;
  // CQ-61 item 4: previous scan's own n_residuals, so residual-starvation
  // (a sudden drop, not just an absolute floor) can be detected -- see the
  // abort site in processLIO() for the full rationale. -1 means "no prior
  // scan yet" (never triggers the drop check on the very first scan).
  int    coupled_prev_n_residuals_ = -1;
  double coupled_sum_weight_ = -1.0;
  double coupled_h_pp_min_eig_ = -1.0, coupled_h_rr_min_eig_ = -1.0;
  // CQ-83: the remaining LioFrameDiag columns with a genuine coupled
  // analogue, routed here from quantities the dispatcher/builder already
  // compute (HtH_pose_lidar/Htz_pose_lidar, the sum_*_S components, ask/
  // got already wired above). h_pp_max_eig mirrors h_pp_min_eig's own
  // eigensolve (just the top eigenvalue instead of the bottom); h_rr_trace/
  // htth_pos_trace are plain traces of the same HtH_pose_lidar blocks;
  // htz_*_norm read Htz_pose_lidar the same way decoupled reads ekf_.Htz;
  // kappa_eff/kappa_gev* reuse decoupled's own formulas (kappa_eff =
  // sqrt(ask/got)-1; kappa_gev* = GeneralizedSelfAdjointEigenSolver(HtH,P)
  // eigenvalues) against coupled's own HtH_pose_lidar/prior_cov_ 6x6 block;
  // floor_share/sdiag_share/pvar_share/prior_pose_share are each sum_*_S
  // component's fraction of sum_S; nis/nis_est are the per-residual mean
  // nu^2/S and nu^2/(S-s_prior_pose), accumulated alongside sum_floor_S et
  // al in buildImuCorrectionSystem()'s own residual loop; dx_rot_deg/
  // dx_pos_mm read the SAME total_dtheta/total_dt this scan already
  // accumulates for its own engagement reporting.
  double coupled_h_pp_max_eig_ = -1.0;
  double coupled_h_rr_trace_ = -1.0, coupled_htth_pos_trace_ = -1.0;
  double coupled_htz_rot_norm_ = -1.0, coupled_htz_pos_norm_ = -1.0;
  double coupled_kappa_eff_ = -1.0;
  double coupled_kappa_gev_[6] = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
  bool   coupled_kappa_gev_ok_ = false;
  double coupled_floor_share_ = -1.0, coupled_sdiag_share_ = -1.0;
  double coupled_pvar_share_ = -1.0, coupled_prior_pose_share_ = -1.0;
  double coupled_nis_ = -1.0, coupled_nis_est_ = -1.0;
  double coupled_dx_rot_deg_ = 0.0, coupled_dx_pos_mm_ = 0.0;

  // CQ-53 item 1: joint matrix's own LDLT pivot floor/ceiling (vectorD()
  // min/max), plus the SAME diagnostic restricted to the state sub-block
  // (A.block(0,0,ncol_s,ncol_s), a SEPARATE LDLT of that block alone, not a
  // read of the joint LDLT's permuted D -- mirrors spline.cpp's own
  // "separate LDLT of AtA alone" pattern, see its doc comment on why: the
  // KKT/joint matrix's own permutation doesn't preserve block identity) and
  // the coefficient sub-block (A.block(ncol_s,ncol_s,ncol_c,ncol_c)).
  // Computed on every GN iteration's own A, overwritten each time so the
  // FINAL (converged) iteration's values are what survives to nees_diag.txt
  // -- same convention as the item-3d diagnostics above. -1.0 sentinel
  // (never a real pivot) if the diagnostic LDLT itself fails.
  double coupled_joint_dmin_ = -1.0, coupled_joint_dmax_ = -1.0;
  double coupled_state_dmin_ = -1.0, coupled_state_dmax_ = -1.0;
  double coupled_coeff_dmin_ = -1.0, coupled_coeff_dmax_ = -1.0;
  // True the iteration coupled_joint_dmin_ < JOINT_PIVOT_MIN_FLOOR -- at the
  // shipped -1.0 floor this can never fire; exists so the refusal SHAPE is
  // present and testable ahead of an actual validated threshold.
  bool   coupled_pivot_guard_ = false;

  // CQ-53 item 2: relative difference between point_cross_normal (H built
  // at t1) and the "point_time" formula (H built at each residual's own
  // t_k), computed for EVERY residual regardless of which jacobian_time_mode
  // is actually active (this column is diagnostic-only, read by none of the
  // modes), so every mode stays directly comparable on this number. This
  // REPLACES the
  // ad hoc, since-deleted debug print this session used to produce the
  // "1-3%" figure the CQ-50 filing cited -- that number no longer existed
  // on disk anywhere once the print was reverted; this makes it a
  // permanent, re-derivable column instead. p10/p50/p90/max over this
  // iteration's own residual set, overwritten each iteration (final
  // iteration survives).
  double coupled_hcol_reldiff_p10_ = -1.0, coupled_hcol_reldiff_p50_ = -1.0;
  double coupled_hcol_reldiff_p90_ = -1.0, coupled_hcol_reldiff_max_ = -1.0;

  // CQ-53 item 3: this GN iteration's own STEP norms (not the scan's
  // accumulated total, which coupled_delta_phi0_ etc. already track) --
  // |delta_s| (the 18-dim state step) and |delta_c| (the 6*n_c coefficient
  // step) from the joint solve, set at the end of estimateCoupledCorrection()
  // each call and read by processLIO()'s own iter_error.txt write (which
  // runs once per GN iteration already, right after the call).
  double coupled_last_delta_s_norm_ = -1.0, coupled_last_delta_c_norm_ = -1.0;
  // CQ-53 item (B): the s-block's own six 3-dim sub-blocks (same
  // [delta_phi0, delta_p0, delta_v, delta_bg, delta_ba, delta_g] order
  // s_vec/delta_s use throughout this file), split out from the single
  // combined delta_s_norm above -- needed to tell "phi0/p0 are still
  // moving but bg has converged" apart from "everything is shrinking
  // together", which the combined norm alone cannot.
  double coupled_last_delta_phi0_norm_ = -1.0, coupled_last_delta_p0_norm_ = -1.0,
         coupled_last_delta_v_norm_step_ = -1.0, coupled_last_delta_bg_norm_step_ = -1.0,
         coupled_last_delta_ba_norm_step_ = -1.0, coupled_last_delta_g_norm_step_ = -1.0;
  // CQ-56: the SAME per-iteration step, split into its c_acc (position
  // spline) and c_gyr (rotation spline) halves -- delta_c_norm above is
  // their combined norm, which can't distinguish "acc coefficients moved a
  // lot, gyr didn't" from the reverse.
  double coupled_last_delta_c_acc_norm_ = -1.0, coupled_last_delta_c_gyr_norm_ = -1.0;
  // CQ-58 item 3: the FULL per-iteration delta_c vector (not just its norm
  // above) -- layout [c_acc(3*n_c), c_gyr(3*n_c)], same as coupled_c_acc_/
  // coupled_c_gyr_'s own accumulation. Only meaningful when
  // log_cp_constraint_en is on; overwritten every GN iteration so the
  // post-loop cp_constraint.csv write (processLIO(), after the loop) sees
  // the FINAL iteration's own step.
  Eigen::VectorXd coupled_last_delta_c_;

  // CQ-72 item 3: the PREVIOUS GN iteration's own P_N*c VECTOR (P_N = the
  // last IMU step's phi_head position rows) -- traj_dev_step_m is defined
  // as ||P_N c_this_iter - P_N c_prev_iter||, a vector difference, not a
  // difference of norms, so the vector itself (not just its magnitude) has
  // to be kept. coupled_prev_traj_dev_valid_ = false at scan start
  // (alongside coupled_iters_'s own reset) means "no previous iteration
  // this scan", so the first iteration's own step is never reported
  // against a stale value from the prior scan.
  V3D coupled_prev_traj_dev_end_vec_ = V3D::Zero();
  bool coupled_prev_traj_dev_valid_ = false;

  // CQ-53 item 4: per-scan RMS (not mean-absolute -- sum_abs_r/error above
  // is already mean-|r|) residual, in meters, over the FINAL GN iteration's
  // own residual set -- the coupled-path analogue decoupled's own residual
  // RMS record has and coupled never had. sqrt(mean(r^2)), same units/scale
  // as a point-to-plane residual everywhere else in this codebase.
  double coupled_res_rms_ = -1.0;
  // CQ-60 item 0a: arithmetic mean of res.sigma_squared over this scan's
  // final-iteration residual set -- same definition lio_decoupled.cpp's own
  // mean_sigma_squared uses (sum_sigma_squared/n), so the two paths are
  // directly, identically comparable. sum_weight (already logged) is
  // sum(1/sigma_squared) -- its reciprocal-scaled harmonic mean is NOT the
  // same quantity and does not substitute for this one.
  double coupled_mean_sigma_squared_ = -1.0;

  // CQ-53 item 5: gravity-leak falsifier -- world-frame integrated
  // acceleration magnitude (|acc_avr_world|, averaged over this scan's own
  // segments, truth is ~9.81 m/s^2 at rest and under pure translation) and
  // the angle between the CURRENT posterior gravity estimate and this
  // scan's own dominant measured-acceleration direction (a stationary-
  // window proxy for "how much has gravity's direction rotated away from
  // vertical in this estimator's own frame"). Set once per scan (not per
  // iteration) from the FINAL converged coupled_prop_.
  double coupled_acc_world_mag_ = -1.0, coupled_gravity_dir_err_deg_ = -1.0;

  // CQ-44 G0: forced to exactly 0.0 every scan -- there is no separate
  // spline t0 to compare against (this scan's own propagation start IS the
  // previous scan's own coupled endpoint, by construction).
  double boundary_dpos_ = 0.0;
  double boundary_drot_deg_ = 0.0;

  // CQ-54 item 1: the sigma actually used this scan (== the calibration
  // floor unless adaptive_sigma inflated it) and its ratio to that floor --
  // the instrument item (1) asks for, ~1 while quiet, 15-55 through a
  // vibration transient.
  double coupled_sigma_a_used_ = -1.0, coupled_sigma_g_used_ = -1.0;
  double coupled_sigma_a_ratio_ = -1.0, coupled_sigma_g_ratio_ = -1.0;
  // CQ-54 item 3: whether bias_freeze_on_vibration's guard was active THIS
  // scan, and the running fraction of scans it has been active over the
  // process's life (active_count_/scan_count_, both accumulated here since
  // this options struct is not otherwise per-scan-reset).
  bool coupled_bias_freeze_active_ = false;
  long coupled_bias_freeze_active_count_ = 0, coupled_bias_freeze_scan_count_ = 0;
  // CQ-54 item 5: the within-scan correction's and the bias's own
  // contributions to angular rate (angvel_avr = seg.gyr - delta_bg +
  // delta_w), logged as norms in deg/s, plus their net -- overwritten every
  // GN iteration so the final converged call's values survive.
  double coupled_w_from_c_deg_s_ = -1.0, coupled_w_from_bg_deg_s_ = -1.0, coupled_w_net_deg_s_ = -1.0;

  // CQ-54 item 6: the gyro bias's calibration-time value, snapshotted once
  // (lazily, on the first scan bias_anchor is active) and never revised --
  // the anchor target for the whole run.
  V3D coupled_bg_calib_ = V3D::Zero();
  bool coupled_bg_calib_set_ = false;

  // CQ-54 item 4: reduced chi-square this scan -- mean(w_k * r_k^2) over the
  // final GN iteration's residual set, i.e. how big the residuals actually
  // are relative to how big the filter's own (possibly IMU-noise-blind,
  // item 0e) measurement-noise model says they should be. ~1 means the
  // model is honest; item 4 predicts a rise toward ~4 through a vibration
  // transient the weights never see.
  double coupled_reduced_chi2_ = -1.0;
};

}  // namespace livo_recon
