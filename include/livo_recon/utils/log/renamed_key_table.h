#pragma once
// R73 (F-404): configuration keys that were renamed are REFUSED under their old names (no alias period). A run that sets an old name
// fails at start with a message that names the new key, instead of being reported as an anonymous unconsumed key.
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace livo_recon
{
inline const std::vector<std::pair<std::string, std::string>>& renamedKeyTable()
{
  static const std::vector<std::pair<std::string, std::string>> t = {
      {"lio/q_adapt/mode", "imu/process_noise/gain/mode"},
      {"lio/q_adapt/apply", "imu/process_noise/gain/apply"},
      {"lio/q_adapt/min_gain", "imu/process_noise/gain/min_gain"},
      {"lio/q_adapt/max_gain", "imu/process_noise/gain/max_gain"},
      {"lio/q_adapt/forgetting", "imu/process_noise/gain/forgetting"},
      {"lio/q_adapt/rho_clamp", "imu/process_noise/gain/rho_clamp"},
      {"lio/q_adapt/nis_hi", "imu/process_noise/gain/nis_hi"},
      {"lio/q_adapt/inflate", "imu/process_noise/gain/inflate"},
      {"lio/q_adapt/decay", "imu/process_noise/gain/decay"},
      {"lio/q_adapt/whiteness_max", "imu/process_noise/gain/whiteness_max"}};
  return t;
}

// R74 (F-427): keys of features that were REMOVED (local-polynomial energy source, additive WNOA disturbance). There is no replacement
// key. Setting one is an error at start, never an ignored or unconsumed key. The continuous_prior/* names are the R72 spelling of the
// disturbance keys, which R73 renamed; both spellings are refused.
inline const std::vector<std::string>& removedKeyTable()
{
  static const std::vector<std::string> t = {
      "imu/process_noise/motion/energy_source",
      "imu/process_noise/motion/local_poly/order",
      "imu/process_noise/disturbance/mode",
      "imu/process_noise/disturbance/qc_acc",
      "imu/process_noise/disturbance/qc_gyr",
      "imu/process_noise/continuous_prior/mode",
      "imu/process_noise/continuous_prior/qc_acc",
      "imu/process_noise/continuous_prior/qc_gyr"};
  return t;
}

// Pure function: the refusal message for a removed key, or "" when `key` is not a removed key.
inline std::string removedKeyMessage(const std::string& key)
{
  for (const auto& k : removedKeyTable())
    if (k == key) return "configuration key '" + key + "' was removed in R74 (F-427) and has no replacement; delete it from the configuration";
  return std::string();
}

// Pure function (unit-testable without a parameter server): returns the refusal message for `key`, or "" when `key` is not a retired name.
inline std::string renamedKeyMessage(const std::string& key)
{
  for (const auto& kv : renamedKeyTable())
    if (kv.first == key)
      return "configuration key '" + key + "' was renamed in R73 and is refused under its old name; use '" + kv.second + "'";
  return std::string();
}

}  // namespace livo_recon
