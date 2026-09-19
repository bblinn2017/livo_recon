#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/coupled_estimator.h"
#include "livo_recon/utils/eval/nees_logger.h"

#include <limits>

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
  // CQ-55 item 6: second-difference curvature penalty weight on the
  // correction coefficients, added to Lambda alongside (not instead of)
  // the existing basis-Gram value term. Default 0.0 -- md5-inert.
  double curvature_weight = 0.0;
  // CQ-55 item 11(b): the principled, threshold-free alternative to
  // freeze_bg/bias_freeze_on_vibration -- see estimateCoupledCorrection()'s
  // own comment at the accumulation site for the full mechanism. Default
  // false.
  bool bias_observable_only = false;
  // CQ-55 item 8, arm (c): when true, the basis-Gram value term
  // (gram(i,j)/sigma^2) is REPLACED (not supplemented) by curvature_weight's
  // second-difference term plus an explicit DC-only prior (dc_weight below)
  // -- see estimateCoupledCorrection()'s Lambda-construction comment for why
  // a pure curvature penalty alone is an invalid configuration (its 2D null
  // space per axis -- constant AND linear -- leaves the bias-degenerate
  // constant direction completely unpriced; the DC term prices exactly
  // that one direction back). Default false, no-op (curvature_weight's own
  // existing value+curvature behavior is unchanged when this is off).
  bool curvature_only = false;
  // Precision (relative to 1/sigma^2, same scale the removed gram term
  // used) of the explicit DC/mean-direction-only prior arm (c) adds back.
  // Only read when curvature_only is true. Default 0.0 -- no validated
  // value exists yet (rule 26); a test run must set this explicitly.
  double dc_weight = 0.0;
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
  double estimateCoupledCorrection(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);

private:
  LioProcCoupledOptions copts_;

  // Persisted across this scan's own GN iterations (reset at the top of
  // processLIO() each frame); NOT carried scan-to-scan -- c_prior = 0 every
  // scan (item 4).
  std::vector<V3D> coupled_c_acc_, coupled_c_gyr_;
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

  // TQ-40 item 3: the coupled-arm equivalent of the decoupled path's own
  // ask/got/refusal discriminator, restricted to the [delta_phi0, delta_p0]
  // 6-dim sub-block of the FINAL GN iteration's pure-LiDAR-info
  // accumulation -- a NAMED APPROXIMATION (not marginalised over
  // v/bg/ba/g/c, so it overstates the information actually available to
  // phi0/p0 alone by whatever those directions correlate away).
  double coupled_ask_ = -1.0, coupled_got_ = -1.0, coupled_refusal_ = std::numeric_limits<double>::quiet_NaN();
  int    coupled_n_residuals_ = -1;
  double coupled_sum_weight_ = -1.0;
  double coupled_h_pp_min_eig_ = -1.0, coupled_h_rr_min_eig_ = -1.0;

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
  // CQ-56: the SAME per-iteration step, split into its c_acc (position
  // spline) and c_gyr (rotation spline) halves -- delta_c_norm above is
  // their combined norm, which can't distinguish "acc coefficients moved a
  // lot, gyr didn't" from the reverse.
  double coupled_last_delta_c_acc_norm_ = -1.0, coupled_last_delta_c_gyr_norm_ = -1.0;

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
