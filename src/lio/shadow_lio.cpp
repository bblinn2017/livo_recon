#include "livo_recon/lio/shadow_lio.h"

#include <stdexcept>

namespace livo_recon
{
ShadowLio::ShadowLio(NodeContext& ctx, const ImuProc& driver_imu, const ShadowOptions& opts) : opts_(opts), driver_imu_(&driver_imu)
{
  const std::string err = shadowOptionsError(opts_);
  if (!err.empty()) throw std::invalid_argument(err);
  state_ = std::make_shared<StateGroup>(*ctx.state);
  NodeContext sctx(ctx);   // same references and queues; only the state pointer differs
  sctx.state = state_;
  imu_ = std::make_unique<ImuProc>(sctx);
  imu_->markShadow("shadow");
  ImuProcOptions& o = imu_->optsMut();
  o = driver_imu.opts();
  // The shadow never writes the driver's debug files.
  o.log_debug_en = false;
  o.log_qhat_en = false;
  o.first_scans_log_en = false;
  if (opts_.process_noise_model == "fixed")
  {
    o.process_noise_model = "fixed";
    o.motion_acc_floor_scale = 1.0;
    o.motion_gyr_floor_scale = 1.0;
    o.energy_source = "instant";
    o.local_poly_order = 2;
  }
  else if (opts_.process_noise_model == "isotropic")
  {
    o.process_noise_model = "isotropic";
    o.motion_beta = opts_.motion_beta;
    o.motion_acc_scale = opts_.motion_acc_scale;
    o.motion_gyr_scale = opts_.motion_gyr_scale;
    o.motion_acc_max_dynamic_variance = opts_.motion_acc_cap;
    o.motion_gyr_max_dynamic_variance = opts_.motion_gyr_cap;
    o.motion_acc_floor_scale = 1.0;
    o.motion_gyr_floor_scale = 1.0;
    o.energy_source = "instant";
  }
  // "same": the shadow's Q configuration equals the driver's (a null shadow; used as the determinism and bookkeeping control).
}

void ShadowLio::propagateFromDriver(const MeasureGroup& mg, const StateGroup& driver_state_before_imu)
{
  *state_ = driver_state_before_imu;   // state AND covariance rolled back to the driver's posterior of the previous scan
  MeasureGroup sm;
  sm.image.t = mg.image.t;
  sm.imu_samples = mg.imu_samples;
  imu_->processIMU(sm);
  ++propagations_;
}
}  // namespace livo_recon
