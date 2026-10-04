#pragma once
// R74 (F-426 part 2): switch the driver's IMU process-noise configuration at one scan. ROS-free so tests/test_process_noise_switch.cpp can check it.
//
// What it is (fixed in advance): the driver runs its configured process noise (imu/process_noise/model and motion/*) UNTIL the scan before
// `scan`; from the scan whose map frame index is `scan` onward it runs the "after" configuration given here. Nothing else changes: state,
// covariance, voxel map and every other option continue. Default scan = 0 is off and bit-identical (the switch code is never entered).
// Note (mechanical consequence, stated): the motion-dependent Q low-pass filter only runs while the model is not "fixed", so a fixed -> isotropic
// switch starts that filter at the switch scan (primed from that scan's own excitation), and an isotropic -> fixed switch stops it.
#include <string>

namespace livo_recon
{
struct ProcessNoiseSwitch
{
  long scan = 0;                           // 0 = off. Otherwise the map frame index of the first scan that uses the "after" configuration (>= 1)
  std::string model = "fixed";             // after model: fixed | isotropic
  double motion_beta = 0.9;                // isotropic only
  double motion_acc_scale = 0.3;
  double motion_gyr_scale = 10.0;
  double motion_acc_cap = 1.0e6;
  double motion_gyr_cap = 1.0e6;
  bool on() const { return scan > 0; }
};

// Empty string when valid, else a message naming the offending key.
inline std::string processNoiseSwitchError(const ProcessNoiseSwitch& s)
{
  if (s.scan < 0) return "imu/process_noise/switch/scan must be >= 0 (0 = off)";
  if (!s.on()) return std::string();
  if (s.model != "fixed" && s.model != "isotropic") return "imu/process_noise/switch/model must be fixed or isotropic";
  if (s.model == "isotropic")
  {
    if (!(s.motion_beta >= 0.0 && s.motion_beta < 1.0)) return "imu/process_noise/switch/motion_beta must be in [0,1)";
    if (!(s.motion_acc_scale >= 0.0) || !(s.motion_gyr_scale >= 0.0)) return "imu/process_noise/switch/motion_acc_scale and motion_gyr_scale must be >= 0";
    if (!(s.motion_acc_cap > 0.0) || !(s.motion_gyr_cap > 0.0)) return "imu/process_noise/switch/motion_acc_cap and motion_gyr_cap must be > 0";
  }
  return std::string();
}

// True exactly once: the first time the scan about to be propagated will be map frame `next_frame` >= scan and the switch has not been applied.
inline bool processNoiseSwitchDue(const ProcessNoiseSwitch& s, long next_frame, bool already_applied)
{
  return s.on() && !already_applied && next_frame >= s.scan;
}

// Copies the "after" configuration into any options struct with ImuProcOptions' member names (a template so the test needs no ROS types).
template <class Opts>
inline void applyProcessNoiseSwitch(Opts& o, const ProcessNoiseSwitch& s)
{
  o.process_noise_model = s.model;
  if (s.model == "isotropic")
  {
    o.motion_beta = s.motion_beta;
    o.motion_acc_scale = s.motion_acc_scale;
    o.motion_gyr_scale = s.motion_gyr_scale;
    o.motion_acc_max_dynamic_variance = s.motion_acc_cap;
    o.motion_gyr_max_dynamic_variance = s.motion_gyr_cap;
  }
}
}  // namespace livo_recon
