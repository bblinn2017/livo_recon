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
      "imu/process_noise/continuous_prior/qc_gyr",
      // R82: F-476 (two-sided NIS gate) and F-403 (hard cap on the iekf step) were retired by Bryce and their code removed.
      "imu/process_noise/gain/nis_lo",
      "imu/process_noise/gain/deflate",
      "lio/ekf/step_cap/mode",
      "lio/ekf/step_cap/rotation_rad",
      "lio/ekf/step_cap/position_m",
      // R83: the zero-velocity pseudo-update (F-401, calibrated thresholds F-508) was a dead lever in R82 and was removed by Bryce (2026-10-06).
      "lio/zupt/mode",
      "lio/zupt/threshold_mode",
      "lio/zupt/gyro_thresh",
      "lio/zupt/acc_thresh",
      "lio/zupt/calib_scans",
      "lio/zupt/calib_margin_acc",
      "lio/zupt/calib_margin_gyro",
      "lio/zupt/enter_scans",
      "lio/zupt/sigma_v",
      // R84 (F-521): the time-dependent point-covariance formulations were removed by Bryce (2026-10-07): R65 propagated mode, R83 window
      // accumulation (lio/point_cov/window_s, process-noise accumulation; the replacement is a different model under a new key) and the R82
      // shared per-scan timing term. (lio/point_cov/insertion_source was kept until R93A.)
      "lio/point_cov/mode",
      "lio/point_cov/window_s",
      "lio/time_cov/mode",
      "lio/time_cov/sigma_s",
      // R85 (F-520 removal): the sample-window covariance (positional spread over sample times) was tested in R84 (216 arms), showed no consistent
      // sign against window 0 and was retired by Bryce (2026-10-08). There is no replacement key.
      "lio/point_cov/sample_window_s",
      // R92: voxel_map/binned/update_count_mode (R91) selected between the R88 to R90 count (bins newly opened) and the R91 count (unique bins touched since the last fit). The touched count is now the only count; the key is refused.
      "voxel_map/binned/update_count_mode",
      // R93A (Bryce 2026-10-10): the pose-covariance levers and the switches whose second branch was never used were removed. voxel_map/residual/pose_cov_in_sigma (the residual weight never contains the pose covariance),
      // voxel_map/plane/plane_fit_pose_cov_mode (every fit uses the sensor covariance), lio/point_cov/insertion_source (posterior only), voxel_map/plane/plane_gap_floor_en and
      // voxel_map/plane/line_residual/classify_at_fit (both now unconditional), voxel_map/plane/debias_denominator_guard and voxel_map/plane/plane_var_denom_floor_en (replaced by the one gap floor).
      "voxel_map/residual/pose_cov_in_sigma",
      "voxel_map/plane/plane_fit_pose_cov_mode",
      "lio/point_cov/insertion_source",
      "voxel_map/plane/plane_gap_floor_en",
      "voxel_map/plane/line_residual/classify_at_fit",
      "voxel_map/plane/debias_denominator_guard",
      "voxel_map/plane/plane_var_denom_floor_en"};
  return t;
}

// The release and followup that removed `key` (R74 F-427 for the first eight keys, R82 F-476 / F-403 for the next five, R83 F-401 / F-508 for lio/zupt/*, R84 F-521 for the temporal point-covariance keys, R85 F-520 for lio/point_cov/sample_window_s, R92 for voxel_map/binned/update_count_mode, R93A for the seven keys at the end of the table).
inline std::string removedKeyOrigin(const std::string& key)
{
  if (key == "imu/process_noise/gain/nis_lo" || key == "imu/process_noise/gain/deflate") return "R82 (F-476)";
  if (key.rfind("lio/ekf/step_cap/", 0) == 0) return "R82 (F-403)";
  if (key.rfind("lio/zupt/", 0) == 0) return "R83 (F-401, F-508)";
  if (key == "lio/point_cov/sample_window_s") return "R85 (F-520)";
  if (key == "voxel_map/binned/update_count_mode") return "R92";
  if (key == "voxel_map/residual/pose_cov_in_sigma" || key == "voxel_map/plane/plane_fit_pose_cov_mode" || key == "lio/point_cov/insertion_source" || key == "voxel_map/plane/plane_gap_floor_en" ||
      key == "voxel_map/plane/line_residual/classify_at_fit" || key == "voxel_map/plane/debias_denominator_guard" || key == "voxel_map/plane/plane_var_denom_floor_en") return "R93A";
  if (key == "lio/point_cov/mode" || key == "lio/point_cov/window_s" || key.rfind("lio/time_cov/", 0) == 0) return "R84 (F-521)";
  return "R74 (F-427)";
}

// Pure function: the refusal message for a removed key, or "" when `key` is not a removed key.
inline std::string removedKeyMessage(const std::string& key)
{
  for (const auto& k : removedKeyTable())
    if (k == key) return "configuration key '" + key + "' was removed in " + removedKeyOrigin(key) + " and has no replacement; delete it from the configuration";
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
