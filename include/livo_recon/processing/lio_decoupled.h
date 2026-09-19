#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/adaptive_q.h"
#include "livo_recon/utils/eval/nees_logger.h"

#include <array>

namespace livo_recon
{

// CQ-49: options exclusive to the decoupled (spline) estimator -- a coupled
// config may carry NEITHER of these namespaces at all (LioProcCoupled's own
// loadParameters() never claims a spline/* or adaptive_q/* key, so
// checkAllParamsConsumed() throws at startup naming any such key left in a
// coupled config -- object identity IS the nesting enforcement, item 3d).
struct LioProcDecoupledOptions
{
  SplineOptions    spline;
  AdaptiveQOptions adaptive_q;
};

// The shipped, DEFAULT estimator (estimator/mode: decoupled) -- unchanged
// arithmetic from before CQ-49's split (see item 0's byte-identical md5
// gate). estimateStateCorrection() (LioProcBase) runs the LiDAR pose-block
// IEKF update; refineSplineFromResiduals() separately shapes the position
// spline's interior against the same residuals; the two are reconciled at
// scan end via moveTailClamp(). NO estimator_mode branch survives anywhere
// in this class -- it always runs the decoupled path, by construction (the
// node never instantiates this class under estimator/mode: coupled).
class LioProcDecoupled : public LioProcBase
{
public:
  explicit LioProcDecoupled(NodeContext& ctx);
  ~LioProcDecoupled() override;

  std::string loadParameters(ros::NodeHandle& pnh) override;
  std::string engagementReport() const override;
  void deskewAndDownsample(MeasureGroup& mg) override;
  std::string processLIO(MeasureGroup& mg) override;

  // Re-place this frame's kept points against the spline after the spline
  // has been re-anchored to the corrected state. Called at the top of
  // every IEKF inner iteration after the first, when
  // spline.redeskew_each_iteration is on. No-op (returns false) if the
  // spline is not valid for this frame.
  bool redeskewFromSpline(MeasureGroup& mg);

  // Refine the spline's interior shape against the residuals the IEKF
  // just accumulated, before the same residuals are used to solve.
  bool refineSplineFromResiduals(const MeasureGroup& mg);

  // Fit the spline, and afterwards measure the IMU residual against it and
  // hand the result to AdaptiveQ.
  void finalizeSplineAndQ(MeasureGroup& mg);

private:
  // See LioProcOptions::dry_run_point_filter_num's doc comment.
  void runDryRunShadowPass(const MeasureGroup& mg);

  LioProcDecoupledOptions dopts_;
  // CQ-60 item 5, Tier 1: see lio_coupled.h's own copy of this field for
  // the full doc comment -- constructed with opts_.nees_tier1_window_scans
  // once loadParameters() has run.
  Tier1NeesBuffer decoupled_tier1_nees_{0};

  // ── Scan-spline state, all per-frame ────────────────────────────────────
  ScanSpline spline_;
  bool       spline_ok_ = false;      // fit succeeded for THIS frame
  // Raw-cloud indices of the points that survived the Stage-1 downsample,
  // captured once per frame so the per-iteration re-deskew can go back to
  // each kept point's raw LiDAR-frame coordinates.
  std::vector<int> ds_offsets_;
  std::vector<int> ds_members_;
  // Scratch for the control-point refinement, reused across iterations and
  // frames so the per-iteration path allocates nothing.
  std::vector<SplineLidarObs> lidar_obs_;

  std::vector<Pose6D> spline_poses_;    // replay target, reused
  int  spline_refits_ = 0;              // per frame, for spline_q.csv

  // The bias/gravity THIS FRAME's IMU propagation actually started from.
  V3D spline_frame_bias_acc_ = V3D::Zero();
  V3D spline_frame_bias_gyr_ = V3D::Zero();
  V3D spline_frame_gravity_  = V3D::Zero();

  // ── Engagement counters ─────────────────────────────────────────────────
  int    redeskew_calls_ = 0;           // per frame
  double redeskew_dp_rms_ = 0.0;        // m, RMS point move across a re-deskew
  std::vector<PointXYZCov> redeskew_prev_;   // scratch, for the displacement

  double refit_dtraj_rms_ = 0.0;   // m
  double refit_dtraj_max_ = 0.0;   // m
  double refit_drot_deg_  = 0.0;   // deg
  double cov_acc_pre_ = 0.0;
  double cov_gyr_pre_ = 0.0;

  // Run totals, for the engagement report.
  long   run_redeskew_calls_ = 0;
  long   run_refine_applied_ = 0;
  long   run_refine_rejects_ = 0;
  long   run_refits_ = 0;
  double run_refine_dcp_max_ = 0.0;
  double run_redeskew_dp_max_ = 0.0;
  long   run_aq_ok_frames_ = 0;
  double run_aq_applied_min_acc_ = 0.0, run_aq_applied_max_acc_ = 0.0;
  double run_aq_applied_min_gyr_ = 0.0, run_aq_applied_max_gyr_ = 0.0;
  long   run_frames_ = 0;

  // adaptive_q/noise_floor/mode, kept as a string so the resolver can
  // validate it; AdaptiveQOptions::use_noise_floor is derived from it.
  std::string adaptive_q_floor_mode_ = "allan";
  AdaptiveQ  adaptive_q_;
  bool       adaptive_q_primed_ = false;
  SplineImuResidualStats last_spline_stats_;
  int        spline_fit_fail_count_ = 0;
  int        spline_frame_count_ = 0;
  std::array<int, 11> spline_fail_cause_count_{};
  int        spline_chart_guard_warn_count_ = 0;

  // DX-2 preflight fix (boundary_dpos). Consecutive scans' spline windows
  // abut in time, so the last (t1) sample of one scan's finalized spline
  // and the first (t0) sample of the next MEASURE the inter-scan
  // discontinuity.
  bool  prev_scan_end_valid_ = false;
  V3D   prev_scan_end_pos_   = V3D::Zero();
  M3D   prev_scan_end_rot_   = M3D::Identity();
  V3D   prev_scan_end_vel_   = V3D::Zero();
  double last_total_dtheta_deg_ = 0.0;
  // TQ-38 item 1: the spline's control points EXACTLY as fit() left them
  // (boundary-constrained, UNREFINED).
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_p_unrefined_snapshot_;
  // This scan's own boundary_dpos/drot (vs the PREVIOUS scan's end),
  // -1 if unavailable.
  double boundary_dpos_ = -1.0;
  double boundary_drot_deg_ = -1.0;
};

}  // namespace livo_recon
