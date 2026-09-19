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

  // CQ-44 G0: forced to exactly 0.0 every scan -- there is no separate
  // spline t0 to compare against (this scan's own propagation start IS the
  // previous scan's own coupled endpoint, by construction).
  double boundary_dpos_ = 0.0;
  double boundary_drot_deg_ = 0.0;
};

}  // namespace livo_recon
