#pragma once

#include <ros/ros.h>

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "livo_recon/utils/log/param_warn.h"

// ---------------------------------------------------------------------------
// ConfigResolver -- read a configuration whose STRUCTURE is the dependency
// structure, and refuse rather than resolve silently.
//
// WHY THIS EXISTS.  Options in this codebase were a flat list of
// independent-looking booleans and scalars, and every relationship between
// them -- nesting, mutual exclusion, enumeration -- lived only in the C++ and
// in a comment.  Three consequences, all of them observed rather than
// hypothesised:
//
//   NESTING.  spline/lidar_refine_cp and spline/reintegrate_each_iteration
//   read as peers of spline/redeskew_each_iteration and are in fact steps
//   INSIDE it.  A 276-job grid crossed them as orthogonal factors; 32 cells
//   per arm per sequence came back ATE-identical, and the only reason anyone
//   noticed was that two engagement counters had been added one batch earlier
//   for an unrelated reason.
//
//   MUTUAL EXCLUSION.  spline/n_control_points and spline/control_point_hz
//   parameterise one quantity.  Precedence resolved it (hz won whenever it was
//   > 0) and nothing was printed, so a config that set both was legal, ran one
//   of them, and said nothing about which.
//
//   ENUMERATION.  spline/rot_mode was compared against "tangent" with
//   everything else falling through to "cumulative", so a typo selected an
//   arm rather than failing.
//
// THE CONTRACT.  A key whose enclosing scope is off must not be read, and if
// it was set explicitly the run REFUSES -- a sweep cell that cannot mean what
// it says should fail at dispatch, not produce a duplicate forty minutes
// later.  A mode must be one of a named set.  And what gets printed and
// logged is the EFFECTIVE configuration, not the requested one, because that
// is the cell's real identity and it is what a scorer should parse.
// ---------------------------------------------------------------------------

namespace livo_recon
{

class ConfigResolver
{
public:
  explicit ConfigResolver(ros::NodeHandle& pnh) : pnh_(pnh) {}

  // Unconditional scalar.  Same behaviour as paramWarn(), recorded for the
  // effective-configuration report.
  template <typename T>
  void get(const std::string& key, T& out, const T& def)
  {
    paramWarn<T>(pnh_, key, out, def);
    effective_.emplace_back(key, detail::paramWarnFormat(out));
  }

  // Enumerated option.  An unrecognised value is a hard error: the run must
  // not silently land on a default that happens to be a valid arm.
  void mode(const std::string& key, std::string& out, const std::string& def,
            std::initializer_list<const char*> allowed)
  {
    paramWarn<std::string>(pnh_, key, out, def);
    if (!isAllowed(out, allowed))
    {
      std::ostringstream oss;
      oss << key << " = '" << out << "' is not one of " << allowedList(allowed);
      errors_.push_back(oss.str());
      out = def;   // keep the object well-formed; the caller aborts on !ok()
    }
    effective_.emplace_back(key, out);
  }

  // Scalar that is meaningful only inside a live scope.  When the scope is
  // off the default is written, and if the key was set explicitly that is an
  // error naming the scope that killed it.
  template <typename T>
  void nested(bool live, const std::string& scope, const std::string& key,
              T& out, const T& def)
  {
    if (live) { get<T>(key, out, def); return; }
    out = def;
    if (pnh_.hasParam(key)) ignore(scope, key);
  }

  // Enumerated option inside a live scope.
  void nestedMode(bool live, const std::string& scope, const std::string& key,
                  std::string& out, const std::string& def,
                  std::initializer_list<const char*> allowed)
  {
    if (live) { mode(key, out, def, allowed); return; }
    out = def;
    if (pnh_.hasParam(key)) ignore(scope, key);
  }

  // For a derived value the caller computed rather than read (n_cp resolved
  // from a rate, a floor converted from an error in metres, and so on).
  void derived(const std::string& label, const std::string& value)
  { effective_.emplace_back(label + "  [derived]", value); }

  bool ok() const { return errors_.empty(); }

  std::string report() const
  {
    std::ostringstream oss;
    oss << "[config/effective]  what this run actually does -- not what the "
           "YAML asked for";
    for (const auto& kv : effective_)
      oss << "\n  " << kv.first << " = " << kv.second;
    if (!errors_.empty())
    {
      oss << "\n[config/REFUSED]  " << errors_.size()
          << " problem(s); the run is aborting rather than resolving these "
             "silently";
      for (const auto& e : errors_) oss << "\n  " << e;
    }
    return oss.str();
  }

  const std::vector<std::string>& errors() const { return errors_; }

private:
  void ignore(const std::string& scope, const std::string& key)
  {
    std::ostringstream oss;
    oss << key << " was set explicitly but is dead: the scope that would read "
           "it (" << scope << ") is off. Remove the key or turn the scope on -- "
           "a run that silently drops it is a cell that does not mean what it "
           "says.";
    errors_.push_back(oss.str());
  }

  static bool isAllowed(const std::string& v,
                        std::initializer_list<const char*> allowed)
  {
    return std::any_of(allowed.begin(), allowed.end(),
                       [&](const char* a) { return v == a; });
  }

  static std::string allowedList(std::initializer_list<const char*> allowed)
  {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const char* a : allowed) { oss << (first ? "" : ", ") << a; first = false; }
    oss << "}";
    return oss.str();
  }

  ros::NodeHandle& pnh_;
  std::vector<std::pair<std::string, std::string>> effective_;
  std::vector<std::string> errors_;
};

}  // namespace livo_recon
