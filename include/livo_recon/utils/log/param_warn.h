#pragma once

#include <ros/ros.h>
#include <sstream>
#include <string>
#include <vector>

// History (8-22): see docs/livo_recon_changelog.md#include-livo_recon-utils-log-param_warn.h-8
namespace livo_recon
{

namespace detail
{
// std::vector<T> has no operator<<, unlike every other type this codebase's
// *ProcOptions structs ever load (bool/int/double/std::string) -- print it
// as "[v0 v1 ...]" instead. Everything else falls through to the generic
// overload just below, which streams the value directly.
template <typename T>
std::string paramWarnFormat(const std::vector<T>& v)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i)
    oss << (i ? " " : "") << v[i];
  oss << "]";
  return oss.str();
}

template <typename T>
std::string paramWarnFormat(const T& v)
{
  std::ostringstream oss;
  oss << v;
  return oss.str();
}
}  // namespace detail

template <typename T>
void paramWarn(ros::NodeHandle& pnh, const std::string& key, T& out, const T& default_value)
{
  if (!pnh.hasParam(key))
  {
    ROS_WARN_STREAM("[params] '" << key << "' not found on the param server "
                     "-- falling back to default: " << detail::paramWarnFormat(default_value));
  }
  pnh.param<T>(key, out, default_value);
}

}  // namespace livo_recon
