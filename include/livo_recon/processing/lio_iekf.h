#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/akf_q.h"
#include "livo_recon/lio/q_adapt.h"
#include "livo_recon/lio/zupt.h"
#include "livo_recon/diagnostics/eval/nees_logger.h"

#include <array>

namespace livo_recon
{

// The shipped, DEFAULT estimator (estimator/mode: iekf; renamed from
// `decoupled` 2026-09-30 when the scan spline and AdaptiveQ were removed).
// Points are deskewed once from the IMU poses; estimateStateCorrection() (LioProcBase)
// runs the LiDAR pose-block IEKF update. NO estimator_mode branch survives in
// this class (the node never instantiates it under estimator/mode: coupled).
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
  // See LioProcOptions::dry_run_point_filter_num's doc comment.
  void runDryRunShadowPass(const MeasureGroup& mg);

  // CQ-60 item 5, Tier 1: see lio_coupled.h's own copy of this field for
  // the full doc comment -- constructed with opts_.nees_tier1_window_scans
  // once loadParameters() has run.
  Tier1NeesBuffer iekf_tier1_nees_{0};

  AkfQ       akf_q_;   // R66 opt-in AKF-style Q adaptation (akf_q/enable, default false)

  // R72 (F-401): innovation-driven gain on the IMU step variance (lio/q_adapt/*) and zero-velocity pseudo-update (lio/zupt/*).
  // Both default off (bit-identical); engagement counters below feed engagementReport().
  QGainOptions q_gain_opts_;
  QGain        q_gain_;
  long   qa_scans_ = 0, qa_scans_gain_ne_1_ = 0;
  double qa_nis_sum_ = 0.0, qa_rho_sum_ = 0.0, qa_lag1_abs_sum_ = 0.0;
  void updateQGainFromResiduals();

  ZuptOptions  zupt_opts_;
  ZuptDetector zupt_det_;
  long   zupt_scans_ = 0, zupt_stationary_ = 0, zupt_applied_ = 0;
  double zupt_vnorm_sum_ = 0.0, zupt_vnorm_max_ = 0.0;
  void applyZuptIfStationary(MeasureGroup& mg);
};

}  // namespace livo_recon
