#pragma once
// R73 (F-414): the shadow instance's propagation half. The update half lives in LioProcIekf::runShadowUpdate (it needs the residual
// machinery). See shadow_options.h for the semantics.
#include <memory>

#include "livo_recon/lio/shadow_options.h"
#include "livo_recon/node_context.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/state/state.h"

namespace livo_recon
{
class ShadowLio
{
public:
  // driver_imu supplies the options the shadow starts from; the shadow then overrides only the process-noise model and scales.
  ShadowLio(NodeContext& ctx, const ImuProc& driver_imu, const ShadowOptions& opts);

  // Called by the node immediately BEFORE the driver's processIMU(mg): the driver's state at that moment is its posterior of the
  // previous scan. The shadow state (mean and covariance) is reset to it, then one scan is propagated with the shadow's own Q on a
  // private copy of the IMU samples (mg itself is not modified).
  void propagateFromDriver(const MeasureGroup& mg, const StateGroup& driver_state_before_imu);

  const StateGroupPtr& state() const { return state_; }
  const ShadowOptions& opts() const { return opts_; }
  const ImuProc& imu() const { return *imu_; }
  const ImuProc& driverImu() const { return *driver_imu_; }
  long propagations() const { return propagations_; }

private:
  ShadowOptions opts_;
  const ImuProc* driver_imu_ = nullptr;   // owned by the node; outlives the shadow
  StateGroupPtr state_;
  std::unique_ptr<ImuProc> imu_;
  long propagations_ = 0;
};
}  // namespace livo_recon
