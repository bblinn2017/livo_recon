#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/q_adapt.h"
#include "livo_recon/diagnostics/eval/nees_logger.h"

#include <array>
#include <string>
#include <vector>

namespace livo_recon
{

// The estimator (R97: the only one; the estimator/mode key was removed). Points are deskewed once from the IMU poses;
// estimateStateCorrection() (LioProcBase) runs the LiDAR pose-block IEKF update.
class LioProcIekf : public LioProcBase
{
public:
  explicit LioProcIekf(NodeContext& ctx);
  ~LioProcIekf() override;

  std::string loadParameters(ros::NodeHandle& pnh) override;
  std::string engagementReport() const override;
  void deskewAndDownsample(MeasureGroup& mg) override;
  std::string processLIO(MeasureGroup& mg) override;
  const char* firstFrameArchitectureName() const override {
    return "iekf";
  }

private:
  // R73 (F-414, F-415): everything one update of one scan produced, for the per-scan logs. Write-only.
  struct UpdateRecord
  {
    bool valid = false;
    V3D start_pos = V3D::Zero(), start_rv = V3D::Zero();   // state before this scan's propagation (the previous scan's posterior)
    V3D prop_pos = V3D::Zero(), prop_rv = V3D::Zero();     // propagated (prior) state
    Eigen::MatrixXd prior_cov;                              // the prior covariance the solve used (after the scalar P controls)
    V3D post_pos = V3D::Zero(), post_rv = V3D::Zero();     // posterior state
    Eigen::MatrixXd post_cov;
    int n_iter = 0, n_resid0 = 0;
    double chi2_iter0 = 0.0;                                // mean r^2 / sigma^2 over the iteration-0 residuals
    std::vector<double> step_rot, step_pos;                 // |dtheta| (rad) and |dt| (m) of every executed iteration
    std::vector<int> n_resid_iter;
    Eigen::Matrix<double, 6, 6> HtH0 = Eigen::Matrix<double, 6, 6>::Zero(), HtHlast = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> Htz0 = Eigen::Matrix<double, 6, 1>::Zero(), Htzlast = Eigen::Matrix<double, 6, 1>::Zero();
    std::string stop;
    void reset() { *this = UpdateRecord(); }
  };
  UpdateRecord drv_rec_;
  void recordIteration(UpdateRecord& rec, int iter, const V3D& dtheta, const V3D& dt) const;
  void writeScanLogRow(const MeasureGroup& mg, const VoxelMapScanStats& st) const;
  VoxelMapScanStats scan_stats_;
  bool scan_stats_valid_ = false;

  // CQ-60 item 5, Tier 1: see lio_coupled.h's own copy of this field for
  // the full doc comment -- constructed with opts_.nees_tier1_window_scans
  // once loadParameters() has run.
  Tier1NeesBuffer iekf_tier1_nees_{0};

  // R72 (F-401): innovation-driven gain on the IMU step variance (imu/process_noise/gain/*). Default off (bit-identical); engagement counters below feed engagementReport().
  // R83: the zero-velocity pseudo-update (lio/zupt/*) was removed (Bryce 2026-10-06).
  QGainOptions q_gain_opts_;
  QGain        q_gain_;
  long   qa_scans_ = 0, qa_scans_gain_ne_1_ = 0;
  double qa_nis_sum_ = 0.0, qa_rho_sum_ = 0.0, qa_lag1_abs_sum_ = 0.0;
  void updateQGainFromResiduals();

  // R91H: rotation-warp scale engagement (written only when imu/undistort/rot_warp_scale != 1). Scans warped, scans on which the scale was applied (frame <= rot_warp_last_frame), and
  // the summed counters of the primary stream over those scans.
  long   rot_warp_scans_ = 0, rot_warp_scans_applied_ = 0, rot_warp_points_ = 0, rot_warp_points_rescaled_ = 0;
  double rot_warp_sum_theta_raw_ = 0.0, rot_warp_sum_theta_applied_ = 0.0, rot_warp_max_theta_raw_ = 0.0;
  bool   rot_warp_on_report() const { return opts_.deskew.rot_warp_scale != 1.0; }
};

}  // namespace livo_recon
