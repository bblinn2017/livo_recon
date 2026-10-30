#pragma once
// cbk/imu/motion_band/* (see motion_band_canceller.h). Read by CbkProc (the
// canceller on the IMU stream, same hook point as cbk/imu/vibration -- right
// after VibrationCanceller::process(), before the sample is pushed to
// DataQueues). Only cbk/imu/motion_band/enable is read when it is false (the
// rest of the block is then marked consumed), mirroring vibration_params.h's
// own convention exactly.
#include <stdexcept>
#include <string>
#include <vector>

#include "livo_recon/utils/imu/motion_band_canceller.h"
#include "livo_recon/utils/log/param_warn.h"

namespace livo_recon
{

inline MotionBandCancellerOptions loadMotionBandConfig(ros::NodeHandle& pnh)
{
  MotionBandCancellerOptions o;
  const std::string k = "cbk/imu/motion_band/";
  paramWarn<bool>(pnh, k + "enable", o.enable, false);
  if (!o.enable)
  {
    markParamConsumed("cbk/imu/motion_band");
    return o;
  }
  paramWarn<double>(pnh, k + "sample_rate_hz", o.sample_rate_hz, 200.0);
  paramWarn<double>(pnh, k + "band_lo_hz", o.band_lo_hz, 8.0);
  paramWarn<double>(pnh, k + "band_hi_hz", o.band_hi_hz, 12.0);
  paramWarn<double>(pnh, k + "gyro_threshold_rad_s", o.gyro_threshold_rad_s, 0.03);
  paramWarn<double>(pnh, k + "gyro_lpf_tau_s", o.gyro_lpf_tau_s, 0.5);
  paramWarn<std::vector<double>>(pnh, k + "floor_amplitude", o.floor_amplitude, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "plateau_amplitude", o.plateau_amplitude, std::vector<double>{});

  if (!(o.sample_rate_hz > 0.0) || !(o.band_lo_hz > 0.0) || !(o.band_hi_hz > o.band_lo_hz) ||
      o.band_hi_hz >= 0.5 * o.sample_rate_hz)
    throw std::invalid_argument(k + "band_lo_hz/band_hi_hz must satisfy 0 < lo < hi < sample_rate_hz/2");
  if (!(o.gyro_threshold_rad_s > 0.0) || !(o.gyro_lpf_tau_s > 0.0))
    throw std::invalid_argument(k + "gyro_threshold_rad_s and gyro_lpf_tau_s must be > 0");
  for (const std::vector<double>* v : {&o.floor_amplitude, &o.plateau_amplitude})
  {
    if (v->size() != 3)
      throw std::invalid_argument(k + "floor_amplitude and plateau_amplitude must be set "
                                  "(3 values each, x/y/z) when the canceller is enabled");
    for (double x : *v)
      if (!(x >= 0.0)) throw std::invalid_argument(k + "floor_amplitude/plateau_amplitude must be >= 0");
  }
  for (int i = 0; i < 3; i++)
    if (o.plateau_amplitude[i] < o.floor_amplitude[i])
      throw std::invalid_argument(k + "plateau_amplitude must be >= floor_amplitude per axis "
                                  "(the model removes the EXCESS motion-induced amplitude, not a negative one)");
  return o;
}

}  // namespace livo_recon
