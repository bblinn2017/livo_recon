#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <ros/ros.h>
#include "livo_recon/utils/log/param_warn.h"

namespace livo_recon
{

enum class PrintCategory { PARAMS = 0, STATE, CALIB, VIO, LIO, PROFILER, VOXEL, EVO, COUNT };

class Printer
{
  static constexpr int N = static_cast<int>(PrintCategory::COUNT);

  static constexpr const char* kParamKeys[N] = {
    "outputs/printer/params/enable",
    "outputs/printer/state/enable",
    "outputs/printer/calib/enable",
    "outputs/printer/vio/enable",
    "outputs/printer/lio/enable",
    "outputs/printer/profiler/enable",
    "outputs/printer/voxel/enable",
    "outputs/printer/evo/enable",
  };

  static constexpr const char* kParamNames[N] = {
    "params", "state", "calib", "vio", "lio", "profiler", "voxel", "evo",
  };

public:
  Printer()
  {
    for (int i = 0; i < N; ++i) { enabled_[i] = true; last_s_[i] = 0.0; }
  }

  void print(PrintCategory cat, const std::string& msg) const
  {
    if (msg.empty()) return;
    const int i = static_cast<int>(cat);
    if (!enabled_[i]) return;

    if (cat == PrintCategory::PARAMS)
    {
      ROS_INFO_STREAM(msg);
      return;
    }

    const double now = ros::Time::now().toSec();
    if (now - last_s_[i] >= throttle_s_)
    {
      ROS_INFO_STREAM(msg);
      last_s_[i] = now;
    }
  }

  std::string loadParameters(ros::NodeHandle& pnh)
  {
    paramWarn<double>(pnh, "outputs/printer/throttle_s", throttle_s_, 5.0);
    for (int i = 0; i < N; ++i)
      paramWarn<bool>(pnh, kParamKeys[i], enabled_[i], true);

    std::ostringstream oss;
    oss << "[params/printer]\n  throttle_s: " << throttle_s_;
    for (int i = 0; i < N; ++i)
      oss << "\n  " << kParamNames[i] << "/enable: " << (enabled_[i] ? "true" : "false");
    return oss.str();
  }

private:
  double throttle_s_ = 5.0;
  bool   enabled_[N];
  mutable double last_s_[N];
};

using PrinterPtr = std::shared_ptr<Printer>;

}  // namespace livo_recon
