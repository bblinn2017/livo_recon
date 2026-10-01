#pragma once
// R73 (F-414): options of the shadow LIO update. ROS-free so that tests/test_shadow_options.cpp can check the validation.
//
// What the shadow is (fixed in advance, see the follow-up register F-414):
//   The DRIVER run updates the filter and the voxel map with its own configuration, exactly as a normal run.
//   At every scan the SHADOW starts from the driver's posterior of the previous scan, state AND covariance both rolled back to it
//   (the shadow keeps nothing of its own between scans except the IMU motion-Q filter, which is driven by the raw IMU samples, not by
//   the state). It propagates one scan with its OWN process noise, and, for scans inside [window_start, window_end], does a full
//   LiDAR update against the driver's voxel map, read only (no insertion, no map side effect). Both the driver's and the shadow's
//   state, prior covariance and posterior covariance are recorded every window scan.
#include <string>

namespace livo_recon
{
struct ShadowOptions
{
  std::string mode = "off";                   // off | on
  long window_start = 0, window_end = 0;      // inclusive range of map frame indices (frame_idx) that run the shadow UPDATE
  std::string process_noise_model = "same";   // same | fixed | isotropic: the shadow's IMU process-noise model
  // used only when process_noise_model == isotropic (motion-dependent Q); defaults are the project's motion-Q values.
  double motion_beta = 0.9;
  double motion_acc_scale = 0.3;
  double motion_gyr_scale = 10.0;
  double motion_acc_cap = 1.0e6;
  double motion_gyr_cap = 1.0e6;

  bool on() const { return mode == "on"; }
  bool inWindow(long frame) const { return on() && frame >= window_start && frame <= window_end; }
};

// Empty string when valid, otherwise a message naming the offending key. mode=off is always valid (and its sub-keys are refused by
// ConfigResolver before this is reached).
inline std::string shadowOptionsError(const ShadowOptions& o)
{
  if (o.mode != "off" && o.mode != "on") return "lio/shadow/mode must be off or on";
  if (!o.on()) return std::string();
  if (o.window_start < 1) return "lio/shadow/window_start must be >= 1 (frame index 0 is the empty-map bootstrap scan)";
  if (o.window_end < o.window_start) return "lio/shadow/window_end must be >= lio/shadow/window_start";
  if (o.process_noise_model != "same" && o.process_noise_model != "fixed" && o.process_noise_model != "isotropic")
    return "lio/shadow/process_noise_model must be same, fixed or isotropic";
  if (o.process_noise_model == "isotropic")
  {
    if (!(o.motion_beta >= 0.0 && o.motion_beta < 1.0)) return "lio/shadow/motion_beta must be in [0,1)";
    if (!(o.motion_acc_scale >= 0.0) || !(o.motion_gyr_scale >= 0.0)) return "lio/shadow/motion_acc_scale and motion_gyr_scale must be >= 0";
    if (!(o.motion_acc_cap > 0.0) || !(o.motion_gyr_cap > 0.0)) return "lio/shadow/motion_acc_cap and motion_gyr_cap must be > 0";
  }
  return std::string();
}
}  // namespace livo_recon
