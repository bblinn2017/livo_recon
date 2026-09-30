#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/akf_q.h"
#include "livo_recon/diagnostics/eval/nees_logger.h"

#include <array>

namespace livo_recon
{

// The shipped, DEFAULT estimator (estimator/mode: decoupled) -- unchanged
// arithmetic from before CQ-49's split. Points are deskewed once from the IMU
// poses (spline/mode raw_imu, the only mode left: the decoupled spline and
// AdaptiveQ were removed 2026-09-30); estimateStateCorrection() (LioProcBase)
// runs the LiDAR pose-block IEKF update. NO estimator_mode branch survives in
// this class (the node never instantiates it under estimator/mode: coupled).
class LioProcDecoupled : public LioProcBase
{
public:
  explicit LioProcDecoupled(NodeContext& ctx);
  ~LioProcDecoupled() override;

  std::string loadParameters(ros::NodeHandle& pnh) override;
  std::string engagementReport() const override;
  void deskewAndDownsample(MeasureGroup& mg) override;
  std::string processLIO(MeasureGroup& mg) override;
  const char* firstFrameArchitectureName() const override {
    return "splineless";   // archived first-frame logs key on this name
  }

private:
  // See LioProcOptions::dry_run_point_filter_num's doc comment.
  void runDryRunShadowPass(const MeasureGroup& mg);

  // CQ-60 item 5, Tier 1: see lio_coupled.h's own copy of this field for
  // the full doc comment -- constructed with opts_.nees_tier1_window_scans
  // once loadParameters() has run.
  Tier1NeesBuffer decoupled_tier1_nees_{0};

  AkfQ       akf_q_;   // R66 opt-in AKF-style Q adaptation (akf_q/enable, default false)
};

}  // namespace livo_recon
