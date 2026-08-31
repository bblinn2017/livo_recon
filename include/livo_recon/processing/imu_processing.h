#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

struct ImuProcOptions
{
  // Gates the per-propagation /tmp/imu.txt dump (debugLogImu), which
  // includes a full flattened state covariance matrix every call --
  // previously unconditional (2026-08-09 cleanup, task #149). Off by
  // default.
  bool   log_debug_en = false;
  bool   second_order = true;

  // T0-E (2026-08-31): Q = q_alpha * Q0 -- uniformly scales the WHOLE
  // process-noise matrix (cov_w: rotation/velocity/position noise from
  // varGyr/varAcc, plus bias random-walk noise) built each IMU
  // propagation substep, before it's added into state_->cov(). 1.0
  // (default) is exactly today's behavior, bit-identical -- this option
  // exists purely so scripts/analysis/{consistency.py's NLL panel,
  // qsens.py's secant driver} can externally re-run the same sequence at
  // different alpha and compare total NLL (see LioProcOptions::
  // log_nll_en) to find the alpha that maximizes filter consistency,
  // without touching per-term (gyro-vs-accel-vs-bias) weighting -- a
  // single global scalar, not the 3-axis alpha_acc/alpha_gyr/alpha_bias
  // qsens.py's own doc comment mentions as the eventual generalization.
  double q_alpha = 1.0;
};

class ImuProc
{
public:
  explicit ImuProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  void propagate(MeasureGroup& mg);
  // Just propagate() -- deskewing/downsampling moved to LioProc (2026-08-18),
  // since it's now intrinsically a LIO concern, not a one-shot IMU-side
  // step. This is a thin wrapper rather than callers just calling
  // propagate() directly so the TimedScope("imu/propagate") profiler tag
  // stays intact.
  void processIMU(MeasureGroup& mg);

private:
  StateGroupPtr state_;
  ProfilerPtr profiler_;
  DataQueuesPtr data_queues_;  // for start_time -- see debugLogImu()'s absolute timestamps

  ImuProcOptions opts_;
  ImuSample last_imu_sample_;
};

}  // namespace livo_recon
