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
