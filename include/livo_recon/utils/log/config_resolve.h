#pragma once

#include <ros/ros.h>

#include <algorithm>
#include <initializer_list>
#include <set>
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
    claim(key);
    paramWarn<T>(pnh_, key, out, def);
    effective_.emplace_back(key, detail::paramWarnFormat(out));
  }

  // Enumerated option.  An unrecognised value is a hard error: the run must
  // not silently land on a default that happens to be a valid arm.
  void mode(const std::string& key, std::string& out, const std::string& def,
            std::initializer_list<const char*> allowed)
  {
    claim(key);
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
    claim(key);
    out = def;
    if (pnh_.hasParam(key)) ignore(scope, key);
  }

  // Enumerated option inside a live scope.
  void nestedMode(bool live, const std::string& scope, const std::string& key,
                  std::string& out, const std::string& def,
                  std::initializer_list<const char*> allowed)
  {
    if (live) { mode(key, out, def, allowed); return; }
    claim(key);
    out = def;
    if (pnh_.hasParam(key)) ignore(scope, key);
  }

  // For a derived value the caller computed rather than read (n_cp resolved
  // from a rate, a floor converted from an error in metres, and so on).
  void derived(const std::string& label, const std::string& value)
  { effective_.emplace_back(label + "  [derived]", value); }

  // -------------------------------------------------------------------------
  // The other half of the contract, and its absence was a live defect.
  //
  // Everything above validates the keys this resolver READS.  Nothing
  // validated a key the config SETS that no reader ever asks for -- and
  // 5c93cc6 renamed every spline/* and adaptive_q/* key at once.  An override
  // YAML still written against the old names (which is every cell the sweep
  // runner generates) sets keys nobody consumes, and the run silently takes
  // defaults.  paramWarn's "not found, falling back to default" line is the
  // only trace, and it is a warning in a log nobody greps.  That is exactly
  // the failure this file was written to eliminate, one level up.
  //
  // Call this AFTER every read, with the namespace prefixes this resolver
  // owns COMPLETELY.  Do not pass a prefix whose keys are shared with a
  // paramWarn() elsewhere: those keys are unclaimed here and would be
  // reported as dead when they are merely read by someone else.
  void refuseUnclaimed(std::initializer_list<const char*> prefixes)
  {
    std::vector<std::string> names;
    if (!pnh_.getParamNames(names))
    {
      effective_.emplace_back("[config/unclaimed-scan]",
                              "SKIPPED -- the parameter server would not "
                              "enumerate; a stale key cannot be detected on "
                              "this run");
      return;
    }
    for (const char* pre : prefixes)
    {
      const std::string root = pnh_.resolveName(pre);
      const std::string root_slash = root + "/";
      for (const std::string& full : names)
      {
        if (full != root && full.rfind(root_slash, 0) != 0) continue;
        if (claimed_.count(full) != 0) continue;
        std::ostringstream oss;
        oss << full << " is set but NO reader claims it. Either it is a key "
               "renamed out from under this config, or a typo. Nothing would "
               "consume it and the run would silently use the default -- "
               "which is the failure this resolver exists to prevent, so it "
               "is an error rather than a warning.";
        errors_.push_back(oss.str());
      }
    }
  }

  // A key that was RENAMED rather than removed is the one case
  // refuseUnclaimed() above cannot catch on its own: the old name lives in a
  // namespace this resolver does not own completely (voxel_map/plane/* is
  // still half paramWarn), so scanning the whole prefix would flag keys that
  // other readers legitimately consume.  Name the retired key explicitly and
  // say what replaced it, so a stale override YAML fails loudly with the
  // migration in the error text instead of silently taking the default.
  void refuseIfSet(const std::string& key, const std::string& advice)
  {
    if (!pnh_.hasParam(key)) return;
    std::ostringstream oss;
    oss << pnh_.resolveName(key) << " is set but has been RETIRED. " << advice;
    errors_.push_back(oss.str());
    claim(key);   // so refuseUnclaimed() does not also report it
  }

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
  void claim(const std::string& key) { claimed_.insert(pnh_.resolveName(key)); }

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
  std::set<std::string> claimed_;
};

}  // namespace livo_recon
