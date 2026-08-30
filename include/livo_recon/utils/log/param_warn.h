#pragma once

#include <ros/ros.h>
#include <sstream>
#include <string>
#include <vector>

// Every *ProcOptions field in this codebase is read via pnh.param<T>(key,
// out, default) -- if `key` was never set (typo'd config key, a rosparam
// that got renamed in code but not in every checked-in yaml, an option
// that's simply missing from a particular dataset's config), pnh.param()
// silently falls back to its C++ default with zero indication anything
// was missing. Confirmed this session (2026-08-16): a config-loading
// override placed under the wrong yaml section (e.g. common: instead of
// outputs:) silently no-op'd with no error, and separately, a hardcoded
// yaml value shadowing a just-changed C++ default silently kept the old
// behavior -- both would have been caught immediately by a fallback
// warning. paramWarn<T>() is a drop-in replacement for pnh.param<T>() that
// additionally logs a ROS_WARN when the key isn't found on the param
// server at all (i.e. every *.yaml actually loaded by this launch didn't
// set it), before falling back to `default_value` exactly like pnh.param()
// always has.
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
