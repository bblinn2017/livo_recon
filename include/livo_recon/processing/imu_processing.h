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

  // T0-E (2026-08-31), split per T7 step 2 (2026-08-31): Q = per-block
  // alpha * Q0 -- each scales one part of the process-noise matrix
  // (cov_w) built each IMU propagation substep, before it's added into
  // state_->cov():
  //   q_alpha_gyr  -- rotation block (from varGyr)
  //   q_alpha_acc  -- velocity block, and (second_order only) the
  //                   position block and position/velocity cross terms
  //                   (all derived from varAcc via acc_noise_world)
  //   q_alpha_bias -- both bias random-walk blocks (BG and BA) together;
  //                   not split further, since T7-a/T7-b's per-block
  //                   estimators only distinguish acc/gyr/bias, not
  //                   bias_acc vs bias_gyr separately
  // All default 1.0 -- bit-identical to pre-split behavior when equal.
  // Setting all three to the same value reproduces the old single-scalar
  // q_alpha sweep (T0-E-2's 240 jobs used the pre-split scalar this way,
  // uniformly); this option exists so scripts/analysis/{consistency.py's
  // NLL panel, qsens.py's secant driver} can externally re-run a sequence
  // at different alpha (uniform or per-block) and compare total NLL (see
  // LioProcOptions::log_nll_en) to find the alpha that maximizes filter
  // consistency -- now with per-term weighting available, per T7-a/T7-b's
  // own per-block Q estimates.
  double q_alpha_acc = 1.0;
  double q_alpha_gyr = 1.0;
  double q_alpha_bias = 1.0;
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
