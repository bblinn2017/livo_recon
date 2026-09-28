#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/processing/calib_processing.h"

namespace livo_recon
{
class LioProcBase;

// Owns the boundary between finalized stationary calibration and ordinary
// estimator processing. It never propagates or corrects the state.
class MapBootstrapper
{
public:
  explicit MapBootstrapper(NodeContext& ctx) : ctx_(ctx) {}

  std::string seedSequentialStationary(
      std::vector<CalibrationLidarObservation> observations,
      LioProcBase& lio_proc);

private:
  NodeContext& ctx_;
};
}  // namespace livo_recon
