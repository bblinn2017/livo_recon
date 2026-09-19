#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/coupled_estimator.h"

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
  // CQ-50 diagnostic toggle: when true, every residual's Jacobian row uses
  // Phix_pt/Phic_pt evaluated at the SCAN-END time t1 (coupled_prop_.
  // phi_x_head.back()/phi_head.back()) instead of interpolatePhiX/
  // interpolatePhi at the point's own capture time res.t. This matches the
  // mismatch already baked into H itself (built once from the deskewed
  // point + the scan-end state_->rot(), never re-evaluated per point -- see
  // lio_base.cpp's buildResiduals()), so at true the whole Jrow becomes a
  // consistent (if within-scan-blind) derivative w.r.t. the SAME t1 pose
  // throughout, rather than H(t1) chained through Phi(t_k). Purely
  // diagnostic: confirming the prediction collapses the motion-onset
  // divergence is expected to make within-scan sensitivity structurally
  // useless (every point's Jrow becomes identical up to H), not a fix in
  // its own right. Default false (the existing, mismatched behavior).
  bool phi_at_scan_end = false;
  // CQ-50 item (d), the real fix (not merely diagnostic, unlike
  // phi_at_scan_end above): builds H's rotation-Jacobian column at each
  // point's OWN capture time t_k -- raw_body_point.cross(worldRotAt(t_k)^T *
  // normal) -- instead of point_cross_normal (built once at t1 from the
  // deskewed point + state_->rot()). Phix_pt/Phic_pt stay interpolated at
  // t_k as before (unlike phi_at_scan_end, this keeps within-scan
  // sensitivity rather than discarding it): H_k*Phi(t_k) is now a valid
  // chain rule throughout, not merely a coincidence at rest. Default false
  // (the existing, mismatched H(t1)*Phi(t_k) behavior) until this arm is
  // verified against phi_at_scan_end's own result and CQ-49's md5 pairs.
  bool h_at_point_time = false;
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
  ~LioProcCoupled() override = default;

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
  // at t1) and the h_at_point_time formula (H built at each residual's own
  // t_k), computed for EVERY residual regardless of which arm is actually
  // active (h_at_point_time reads one side, phi_at_scan_end/default reads
  // the other -- this column is diagnostic-only, read by neither), so the
  // two arms stay directly comparable on this number. This REPLACES the
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

  // CQ-53 item 4: per-scan RMS (not mean-absolute -- sum_abs_r/error above
  // is already mean-|r|) residual, in meters, over the FINAL GN iteration's
  // own residual set -- the coupled-path analogue decoupled's own residual
  // RMS record has and coupled never had. sqrt(mean(r^2)), same units/scale
  // as a point-to-plane residual everywhere else in this codebase.
  double coupled_res_rms_ = -1.0;

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
};

}  // namespace livo_recon
