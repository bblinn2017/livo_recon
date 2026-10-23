#pragma once
// cbk/imu/vibration/* (see vibration_canceller.h). Read by CbkProc (the canceller on the IMU stream) and by
// CalibProc (removeLinesBatch() on the calibration window and the vibration-free noise floor), so both use
// the same values. Only cbk/imu/vibration/enable is read when it is false (the rest of the block is then
// marked consumed, so enable: false with the block left in place passes checkAllParamsConsumed()).
#include <stdexcept>
#include <string>
#include <vector>

#include "livo_recon/utils/imu/vibration_canceller.h"
#include "livo_recon/utils/log/param_warn.h"

namespace livo_recon
{

struct VibrationConfig
{
  VibrationCancellerOptions canceller;
  // Stationary noise floor of the IMU with the vibration removed (per sample, per axis), used by
  // CalibProc::computeBiasAndNoise() instead of its fixed raw floor when the canceller is enabled
  std::vector<double> floor_acc_variance{0.0, 0.0, 0.0};
  std::vector<double> floor_gyro_variance{0.0, 0.0, 0.0};
};

inline VibrationConfig loadVibrationConfig(ros::NodeHandle& pnh)
{
  VibrationConfig c;
  VibrationCancellerOptions& o = c.canceller;
  const std::string k = "cbk/imu/vibration/";
  paramWarn<bool>(pnh, k + "enable", o.enable, false);
  if (!o.enable)
  {
    // switched off with the block still in the config: its other keys are intentionally unread
    markParamConsumed("cbk/imu/vibration");
    return c;
  }
  paramWarn<double>(pnh, k + "sample_rate_hz", o.sample_rate_hz, 200.0);
  paramWarn<double>(pnh, k + "main_hz", o.main_hz, 0.0);
  paramWarn<double>(pnh, k + "sideband_offset_hz", o.sideband_offset_hz, 0.0);
  paramWarn<int>(pnh, k + "harmonics", o.harmonics, 1);
  paramWarn<std::vector<double>>(pnh, k + "extra_hz", o.extra_hz, std::vector<double>{});
  paramWarn<double>(pnh, k + "tau_fit_s", o.tau_fit_s, 4.0);
  paramWarn<double>(pnh, k + "highpass_hz", o.highpass_hz, 8.0);
  paramWarn<double>(pnh, k + "tau_freq_s", o.tau_freq_s, 20.0);
  paramWarn<double>(pnh, k + "tau_freq_extra_s", o.tau_freq_extra_s, 60.0);
  paramWarn<double>(pnh, k + "max_freq_dev_hz", o.max_freq_dev_hz, 0.05);
  paramWarn<double>(pnh, k + "sideband_delay_s", o.sideband_delay_s, 2.5);
  paramWarn<double>(pnh, k + "max_gap_s", o.max_gap_s, 0.5);
  paramWarn<std::vector<double>>(pnh, k + "seed_main", o.seed_main, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "seed_sideband_lo", o.seed_sideband_lo, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "seed_sideband_hi", o.seed_sideband_hi, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "seed_harmonic2", o.seed_harmonic2, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "seed_extra0", o.seed_extra0, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "seed_extra1", o.seed_extra1, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "floor_acc_variance", c.floor_acc_variance, std::vector<double>{});
  paramWarn<std::vector<double>>(pnh, k + "floor_gyro_variance", c.floor_gyro_variance, std::vector<double>{});

  if (!(o.sample_rate_hz > 0.0) || !(o.main_hz > 0.0) || o.main_hz >= 0.5 * o.sample_rate_hz)
    throw std::invalid_argument(k + "main_hz must be in (0, sample_rate_hz / 2) and sample_rate_hz > 0");
  if (!(o.sideband_offset_hz >= 0.0) || o.harmonics < 1 || !(o.tau_fit_s > 0.0) || !(o.highpass_hz > 0.0) ||
      !(o.tau_freq_s > 0.0) || !(o.tau_freq_extra_s > 0.0) || !(o.max_freq_dev_hz >= 0.0) ||
      !(o.sideband_delay_s >= 0.0) || !(o.max_gap_s > 0.0))
    throw std::invalid_argument(k + "*: out of range (see config/mid360.yaml)");
  for (const std::vector<double>* v : {&c.floor_acc_variance, &c.floor_gyro_variance})
  {
    if (v->size() != 3)
      throw std::invalid_argument(k + "floor_acc_variance and floor_gyro_variance must be set (3 values each) "
                                  "when the canceller is enabled: the fixed raw floor includes the vibration");
    for (double x : *v)
      if (!(x > 0.0)) throw std::invalid_argument(k + "floor_*_variance must be > 0");
  }
  for (const std::vector<double>* v : {&o.seed_main, &o.seed_sideband_lo, &o.seed_sideband_hi,
                                       &o.seed_harmonic2, &o.seed_extra0, &o.seed_extra1})
    if (!v->empty() && v->size() != 6)
      throw std::invalid_argument(k + "seed_*: must be empty or exactly 6 values "
                                  "(acc x,y,z then gyro x,y,z)");
  return c;
}

}  // namespace livo_recon
