#pragma once

#include <ros/ros.h>
#include <xmlrpcpp/XmlRpcValue.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// History (8-22): see docs/livo_recon_changelog.md#include-livo_recon-utils-log-param_warn.h-8
namespace livo_recon
{

namespace detail
{
// Every key any paramWarn()/markParamConsumed() call has ever asked for, by
// its RELATIVE (pnh-namespace-stripped) name -- e.g. "voxel_map/residual/
// pose_cov_in_sigma", matching exactly what a config.yaml override writes.
// Meyer's-singleton so header-only code across many translation units
// shares one set without a separate .cpp/global-init-order problem.
inline std::set<std::string>& consumedParamKeys()
{
  static std::set<std::string> keys;
  return keys;
}
}  // namespace detail

// Call once, right at the end of the top-level loadParameters() (after
// EVERY *ProcOptions::loadParameters(pnh) call has run), to catch a config
// key that was SET on the param server (i.e. present in config.yaml, a
// dataset default, or an --overrides_file) but never READ by any
// paramWarn()/markParamConsumed() call anywhere in the codebase -- the
// exact failure mode that silently no-ops a misspelled or wrong-nesting
// override key (e.g. "voxel_map/plane/pose_cov_in_sigma" when the real key
// is "voxel_map/residual/pose_cov_in_sigma": no crash, no warning, the
// override is just quietly discarded and the run proceeds on the default).
// Throws (loud, not a ROS_WARN) listing every unconsumed key it finds, so a
// bad job manifest fails BEFORE the run ever launches rather than
// producing a config that silently isn't what was asked for.
inline void checkAllParamsConsumed(ros::NodeHandle& pnh)
{
  std::vector<std::string> all_names;
  ros::param::getParamNames(all_names);
  const std::string ns = pnh.getNamespace() + "/";
  std::vector<std::string> unconsumed;
  for (const auto& full : all_names)
  {
    if (full.rfind(ns, 0) != 0) continue;  // not under this node's private namespace
    const std::string rel = full.substr(ns.size());
    // ROS param server entries for a struct/array (e.g. a rosparam-loaded
    // list) may enumerate as multiple leaf paths under one key a single
    // paramWarn<std::vector<T>> call consumed as a whole -- only flag a
    // leaf if NEITHER it nor any of its own ancestor paths was consumed.
    bool consumed = false;
    std::string probe = rel;
    while (true)
    {
      if (detail::consumedParamKeys().count(probe)) { consumed = true; break; }
      const auto slash = probe.find_last_of('/');
      if (slash == std::string::npos) break;
      probe = probe.substr(0, slash);
    }
    if (!consumed) unconsumed.push_back(rel);
  }
  if (!unconsumed.empty())
  {
    std::ostringstream oss;
    oss << "livo_recon: " << unconsumed.size() << " param(s) set on the server "
           "under '" << ns << "' were never read by any loadParameters() call "
           "-- almost always a misspelled or wrong-nesting config key that "
           "would otherwise silently no-op:\n";
    for (const auto& k : unconsumed) oss << "  " << k << "\n";
    throw std::runtime_error(oss.str());
  }
}

// For a param read via pnh.getParam()/pnh.param() directly rather than
// paramWarn() (e.g. an optional field with no single natural default to
// log) -- marks it consumed so checkAllParamsConsumed() doesn't flag it.
inline void markParamConsumed(const std::string& key)
{
  detail::consumedParamKeys().insert(key);
}

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

// 2026-10-27: a param present on the server with the WRONG XmlRpc type (the
// exact failure mode that let a YAML scalar like `1.0e6` -- parsed by
// PyYAML/ruamel as the STRING "1.0e6", not a float, since YAML's float
// grammar requires a signed exponent -- silently reach the server as a
// string where a double was wanted) used to be indistinguishable from a
// correctly-typed value: pnh.param<T>() coerces what it can and falls back
// to the default on ANY failure, missing-key or wrong-type alike, with no
// warning either way once hasParam() says true. Confirmed this session: a
// job's own `max_avg_error: 1.0e6` override silently never took effect,
// and nothing printed to say so. XmlRpcTypeCheck below closes exactly that
// gap for the four scalar types this codebase's *ProcOptions structs
// actually load (double/int/bool/std::string) -- same "refuse rather than
// resolve silently" contract ConfigResolver's mode()/nested() already
// enforce for scope/enum violations, extended to the raw param-server type.
namespace detail
{
template <typename T>
struct XmlRpcTypeCheck
{
  static constexpr bool kChecked = false;
  static void check(ros::NodeHandle&, const std::string&) {}
};

template <>
struct XmlRpcTypeCheck<double>
{
  static constexpr bool kChecked = true;
  static void check(ros::NodeHandle& pnh, const std::string& key)
  {
    XmlRpc::XmlRpcValue v;
    if (!pnh.getParam(key, v)) return;  // hasParam() raced/lied -- param()'s own fallback handles it
    if (v.getType() != XmlRpc::XmlRpcValue::TypeDouble && v.getType() != XmlRpc::XmlRpcValue::TypeInt)
      throw std::runtime_error("[params] '" + key + "' is set on the param server but is not a "
                               "number (likely a YAML scalar that parsed as a string, e.g. an "
                               "unsigned-exponent float like 1.0e6 instead of 1.0e+6) -- "
                               "expected a double");
  }
};

template <>
struct XmlRpcTypeCheck<int>
{
  static constexpr bool kChecked = true;
  static void check(ros::NodeHandle& pnh, const std::string& key)
  {
    XmlRpc::XmlRpcValue v;
    if (!pnh.getParam(key, v)) return;
    if (v.getType() != XmlRpc::XmlRpcValue::TypeInt)
      throw std::runtime_error("[params] '" + key + "' is set on the param server but is not an "
                               "int -- expected an int");
  }
};

template <>
struct XmlRpcTypeCheck<bool>
{
  static constexpr bool kChecked = true;
  static void check(ros::NodeHandle& pnh, const std::string& key)
  {
    XmlRpc::XmlRpcValue v;
    if (!pnh.getParam(key, v)) return;
    if (v.getType() != XmlRpc::XmlRpcValue::TypeBoolean)
      throw std::runtime_error("[params] '" + key + "' is set on the param server but is not a "
                               "bool (see the YAML 1.1 bool-keyword gotcha -- an unquoted on/off/"
                               "yes/no elsewhere in the same file is the usual cause) -- expected a bool");
  }
};

template <>
struct XmlRpcTypeCheck<std::string>
{
  static constexpr bool kChecked = true;
  static void check(ros::NodeHandle& pnh, const std::string& key)
  {
    XmlRpc::XmlRpcValue v;
    if (!pnh.getParam(key, v)) return;
    if (v.getType() != XmlRpc::XmlRpcValue::TypeString)
      throw std::runtime_error("[params] '" + key + "' is set on the param server but is not a "
                               "string -- expected a string");
  }
};
}  // namespace detail

template <typename T>
void paramWarn(ros::NodeHandle& pnh, const std::string& key, T& out, const T& default_value)
{
  detail::consumedParamKeys().insert(key);
  if (!pnh.hasParam(key))
  {
    ROS_WARN_STREAM("[params] '" << key << "' not found on the param server "
                     "-- falling back to default: " << detail::paramWarnFormat(default_value));
  }
  else if (detail::XmlRpcTypeCheck<T>::kChecked)
  {
    detail::XmlRpcTypeCheck<T>::check(pnh, key);
  }
  pnh.param<T>(key, out, default_value);
}

}  // namespace livo_recon
