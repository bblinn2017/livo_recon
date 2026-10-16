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
      "voxel_map/plane/plane_var_denom_floor_en",
      // R95 (Bryce 2026-10-11): the uniform bin weight (w = 1 per occupied bin) was removed; the weight of a bin is its return count (the former `count`). The key is refused, whatever its value.
      "voxel_map/binned/weight_mode",
      // R97 (Bryce 2026-10-15): the estimator has one mode (the IEKF), so the selector is gone. Both spellings are refused, whatever the value.
      "estimator/mode",
      "lio/estimator/mode",
      // R97 (Bryce 2026-10-15): the lidar time offset and the overlap policy (with its four retime parameters) are replaced by rewritten bags (scripts/r97_rewrite_bags.py); the debiased plane-fit path was removed with its five keys;
      // the inert keys of the R97 sweep (read and printed only, or dead) were removed. Keys of the removed akf and surface backends, shadow, akf_q and process_noise/switch/* sub-keys are refused as unknown keys.
      "cbk/lidar/time_offset",
      "cbk/lidar/overlap_policy",
      "cbk/lidar/retime_t0_s",
      "cbk/lidar/retime_period_s",
      "cbk/lidar/retime_seq0",
      "cbk/lidar/retime_ref_offset_s",
      "voxel_map/plane/plane_fit_mode",
      "voxel_map/plane/debias_mode",
      "voxel_map/plane/plane_threshold_hysteresis",
      "voxel_map/plane/centred_accumulation",
      "voxel_map/plane/debiased_line_equiv",
      "state/est/cov_acc",
      "state/est/cov_gyr",
      "frames/camera",
      "frames/lidar",
      "camera/distortion_model",
      "viz/grid_map",
      "imu/process_noise/switch/scan",
      // R97A Q1: the binned map is the only map (voxel_map/backend); min_frames_to_init was a no-op (default 1) and min_frames_to_converge/
      // convergence_mode=frame_gated are removed; plane_gate_mode is baked in at "disc" (ellipse/ellipse_area_matched removed); line
      // residuals are always on (line_residual/enabled, plane_var_min, plane_var_scan_s2 removed with the no-line path); map/
      // freeze_after_calibration and map/shuffle_insertion_seed were off/0 in every job.
      "voxel_map/backend",
      "voxel_map/plane/min_frames_to_init",
      "voxel_map/plane/min_frames_to_converge",
      "voxel_map/plane/plane_gate_mode",
      "voxel_map/plane/line_residual/enabled",
      "voxel_map/plane/plane_var_min",
      "voxel_map/plane/plane_var_scan_s2",
      "voxel_map/map/freeze_after_calibration",
      "voxel_map/map/shuffle_insertion_seed",
      // R97A Q2: the LIO EKF early-stop thresholds were unset at the yaml (0/0/-1, never firing); the dry-run second
      // decimation stream and the VIO dry-run mode; the residual-redundancy correction and the three scalar P
      // controls (all identity at the yaml); the startup/open-loop/bootstrap-inflate controls (all off at the yaml).
      "lio/ekf/min_norm_dtheta",
      "lio/ekf/min_norm_dt",
      "lio/ekf/min_diff_error",
      "lio/dry_run_point_filter_num",
      "lio/iekf_point_stream",
      "vio/log/dry_run",
      "lio/residual_redundancy/mode",
      "lio/residual_redundancy/rho",
      "lio/residual_redundancy/renorm",
      "lio/p_inflate/alpha",
      "lio/p_floor/min_eig",
      "lio/p_fading/lambda",
      "lio/bootstrap_inflate/scans",
      "lio/bootstrap_inflate/factor",
      "lio/startup_hold/scans",
      "lio/open_loop/mode",
      "lio/open_loop/reset_period_s"};
  return t;
}

// The release and followup that removed `key` (R74 F-427 for the first eight keys, R82 F-476 / F-403 for the next five, R83 F-401 / F-508 for lio/zupt/*, R84 F-521 for the temporal point-covariance keys, R85 F-520 for lio/point_cov/sample_window_s, R92 for voxel_map/binned/update_count_mode, R93A for the next seven keys, R95 for voxel_map/binned/weight_mode, R97 for estimator/mode, lio/estimator/mode and the 18 keys listed after them).
inline std::string removedKeyOrigin(const std::string& key)
{
  if (key == "imu/process_noise/gain/nis_lo" || key == "imu/process_noise/gain/deflate") return "R82 (F-476)";
  if (key.rfind("lio/ekf/step_cap/", 0) == 0) return "R82 (F-403)";
  if (key.rfind("lio/zupt/", 0) == 0) return "R83 (F-401, F-508)";
  if (key == "lio/point_cov/sample_window_s") return "R85 (F-520)";
  if (key == "voxel_map/binned/update_count_mode") return "R92";
  if (key == "voxel_map/residual/pose_cov_in_sigma" || key == "voxel_map/plane/plane_fit_pose_cov_mode" || key == "lio/point_cov/insertion_source" || key == "voxel_map/plane/plane_gap_floor_en" ||
      key == "voxel_map/plane/line_residual/classify_at_fit" || key == "voxel_map/plane/debias_denominator_guard" || key == "voxel_map/plane/plane_var_denom_floor_en") return "R93A";
  if (key == "voxel_map/binned/weight_mode") return "R95";
  if (key == "estimator/mode" || key == "lio/estimator/mode") return "R97";
  for (const char* k : {"cbk/lidar/time_offset", "cbk/lidar/overlap_policy", "cbk/lidar/retime_t0_s", "cbk/lidar/retime_period_s", "cbk/lidar/retime_seq0", "cbk/lidar/retime_ref_offset_s",
                        "voxel_map/plane/plane_fit_mode", "voxel_map/plane/debias_mode", "voxel_map/plane/plane_threshold_hysteresis", "voxel_map/plane/centred_accumulation", "voxel_map/plane/debiased_line_equiv",
                        "state/est/cov_acc", "state/est/cov_gyr", "frames/camera", "frames/lidar", "camera/distortion_model", "viz/grid_map", "imu/process_noise/switch/scan"})
    if (key == k) return "R97";
  if (key == "lio/point_cov/mode" || key == "lio/point_cov/window_s" || key.rfind("lio/time_cov/", 0) == 0) return "R84 (F-521)";
  for (const char* k : {"voxel_map/backend", "voxel_map/plane/min_frames_to_init", "voxel_map/plane/min_frames_to_converge",
                        "voxel_map/plane/plane_gate_mode", "voxel_map/plane/line_residual/enabled", "voxel_map/plane/plane_var_min",
                        "voxel_map/plane/plane_var_scan_s2", "voxel_map/map/freeze_after_calibration", "voxel_map/map/shuffle_insertion_seed",
                        "lio/ekf/min_norm_dtheta", "lio/ekf/min_norm_dt", "lio/ekf/min_diff_error",
                        "lio/dry_run_point_filter_num", "lio/iekf_point_stream", "vio/log/dry_run",
                        "lio/residual_redundancy/mode", "lio/residual_redundancy/rho", "lio/residual_redundancy/renorm",
                        "lio/p_inflate/alpha", "lio/p_floor/min_eig", "lio/p_fading/lambda",
                        "lio/bootstrap_inflate/scans", "lio/bootstrap_inflate/factor", "lio/startup_hold/scans",
                        "lio/open_loop/mode", "lio/open_loop/reset_period_s"})
    if (key == k) return "R97A";
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
