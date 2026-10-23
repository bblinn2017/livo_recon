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
  // R97A Q3: the 10 lio/q_adapt/* -> imu/process_noise/gain/* renames are removed -- the rename TARGET
  // (imu/process_noise/gain/*, the whole adaptive-gain module) no longer exists either, so both the old
  // and new names now move to removedKeyTable() instead (a key cannot be both a renamed-target and removed).
  // R97A Q5: 73 old_key -> new_key moves under log/ and run/ (r97a_inputs/log_key_map.tsv). The two outputs/auto_terminate_*
  // keys move to run/*; every other row moves to log/*.
  static const std::vector<std::pair<std::string, std::string>> t = {
      {"outputs/debug_log_dir", "log/dir"},
      {"outputs/profiler/export_path", "log/profiler_path"},
      {"outputs/pose_pair/export", "log/pose_pair/export"},
      {"outputs/printer/throttle_s", "log/printer/throttle_s"},
      {"outputs/printer/params/enable", "log/printer/params"},
      {"outputs/printer/state/enable", "log/printer/state"},
      {"outputs/printer/calib/enable", "log/printer/calib"},
      {"outputs/printer/vio/enable", "log/printer/vio"},
      {"outputs/printer/lio/enable", "log/printer/lio"},
      {"outputs/printer/profiler/enable", "log/printer/profiler"},
      {"outputs/printer/voxel/enable", "log/printer/voxel"},
      {"outputs/printer/evo/enable", "log/printer/evo"},
      {"eval/state_trace_en", "log/state_trace/enable"},
      {"eval/state_trace_run_id", "log/state_trace/run_id"},
      {"eval/state_trace_pre_imu_frames", "log/state_trace/pre_imu_frames"},
      {"eval/pose_cov_trace_en", "log/pose_cov_trace/enable"},
      {"eval/pose_cov_trace_end_s", "log/pose_cov_trace/end_s"},
      {"eval/nees_per_dof_en", "log/nees/per_dof"},
      {"eval/nees_tier1_window_scans", "log/nees/tier1_window_scans"},
      {"eval/imu_first_scans_en", "log/imu_first_scans"},
      {"diagnostics/frame_window_log", "log/frame_window"},
      {"data_queues/log_sync_debug", "log/data_queues/sync_debug"},
      {"data_queues/sync_debug_log_path", "log/data_queues/sync_debug_path"},
      {"data_queues/log_raw_point_count", "log/data_queues/raw_point_count"},
      {"data_queues/log_sensor_stamps", "log/data_queues/sensor_stamps"},
      {"data_queues/log_lidar_msgs", "log/data_queues/lidar_msgs"},
      {"imu/log_debug_en", "log/imu/debug"},
      {"imu/log_qhat_en", "log/imu/qhat"},
      {"imu/motion_q_step_log_en", "log/imu/motion_q_step"},
      {"lio/log_debug_en", "log/lio/debug"},
      {"lio/log_pair_corr_en", "log/lio/pair_corr"},
      {"lio/log_eigenspectrum_en", "log/lio/eigenspectrum"},
      {"lio/log_consistency_scan_en", "log/lio/consistency_scan"},
      {"lio/log_nll_en", "log/lio/nll"},
      {"lio/update_diag_en", "log/lio/update_diag"},
      {"lio/live_point_cov_en", "log/lio/live_point_cov"},
      {"lio/first_frame_dump_frames", "log/lio/first_frame_dump/frames"},
      {"lio/first_frame_dump_frame_list", "log/lio/first_frame_dump/frame_list"},
      {"lio/first_frame_dump_slim", "log/lio/first_frame_dump/slim"},
      {"lio/scan_log_en", "log/lio/scan"},
      {"lio/residual_dump_frames", "log/lio/residual_dump/frames"},
      {"lio/residual_dump_last_iter", "log/lio/residual_dump/last_iter"},
      {"lio/residual_dump_all_iters", "log/lio/residual_dump/all_iters"},
      {"lio/residual_dump_voxel_info", "log/lio/residual_dump/voxel_info"},
      {"lio/residual_dump_line_geometry", "log/lio/residual_dump/line_geometry"},
      {"lio/residual_dump_slim", "log/lio/residual_dump/slim"},
      {"lio/residual_geometry_en", "log/lio/residual_geometry"},
      {"lio/line_rows_log_en", "log/lio/line_rows"},
      {"lio/zsq_log_en", "log/lio/zsq"},
      {"voxel_map/map/log_frame_stats_en", "log/voxel_map/frame_stats"},
      {"voxel_map/map/snapshot_post_calibration", "log/voxel_map/snapshot_post_calibration"},
      {"voxel_map/map/snapshot_frames", "log/voxel_map/snapshot_frames"},
      {"voxel_map/plane/plane_var_diag_en", "log/voxel_map/plane_var_diag"},
      {"voxel_map/point_cov_stats_en", "log/voxel_map/point_cov_stats"},
      {"voxel_map/plane/log_debug_en", "log/voxel_map/plane_debug"},
      {"voxel_map/plane/plane_fit_reasons_en", "log/voxel_map/plane_fit_reasons"},
      {"voxel_map/plane/plane_fit_trace_frames", "log/voxel_map/plane_fit_trace/frames"},
      {"voxel_map/plane/plane_fit_trace_max_rows", "log/voxel_map/plane_fit_trace/max_rows"},
      {"voxel_map/plane/plane_fit_eigs_en", "log/voxel_map/plane_fit_eigs"},
      {"voxel_map/plane/log_variance_shares_en", "log/voxel_map/variance_shares"},
      {"voxel_map/plane/log_consistency_mode", "log/voxel_map/consistency/mode"},
      {"voxel_map/plane/log_consistency_corr_stride", "log/voxel_map/consistency/corr_stride"},
      {"vio/log/log_iterations", "log/vio/iterations"},
      {"vio/log/log_point_residuals", "log/vio/point_residuals"},
      {"vio/log/log_residual_counts", "log/vio/residual_counts"},
      {"vio/log/log_consistency_corr_en", "log/vio/consistency_corr"},
      {"vio/log/log_raw_tracker_output", "log/vio/raw_tracker_output"},
      {"vio/log/log_norm_l_low_frac", "log/vio/norm_l_low_frac"},
      {"vio/log/log_path", "log/vio/path"},
      {"combined/log_debug_en", "log/combined/debug"},
      {"combined/log_consistency_en", "log/combined/consistency"},
      {"outputs/auto_terminate_on_idle", "run/auto_terminate_on_idle"},
      {"outputs/auto_terminate_idle_secs", "run/auto_terminate_idle_secs"}

  };
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
      "lio/open_loop/reset_period_s",
      // R97A Q3 item 3: the adaptive Q gain module (imu/process_noise/gain/*, off at the yaml) and its
      // predecessor names (lio/q_adapt/*, already-renamed as of R73 -- removed here with their target).
      "lio/q_adapt/mode",
      "lio/q_adapt/apply",
      "lio/q_adapt/min_gain",
      "lio/q_adapt/max_gain",
      "lio/q_adapt/forgetting",
      "lio/q_adapt/rho_clamp",
      "lio/q_adapt/nis_hi",
      "lio/q_adapt/inflate",
      "lio/q_adapt/decay",
      "lio/q_adapt/whiteness_max",
      "imu/process_noise/gain/mode",
      "imu/process_noise/gain/apply",
      "imu/process_noise/gain/min_gain",
      "imu/process_noise/gain/max_gain",
      "imu/process_noise/gain/forgetting",
      "imu/process_noise/gain/rho_clamp",
      "imu/process_noise/gain/nis_hi",
      "imu/process_noise/gain/inflate",
      "imu/process_noise/gain/decay",
      "imu/process_noise/gain/whiteness_max",
      "imu/process_noise/gain/nis_slope",
      "imu/process_noise/gain/nis_clip",
      // R97A Q3 item 1: the Gauss-Markov bias model.
      "imu/process_noise/bias_model",
      "imu/process_noise/gauss_markov/gyro/correlation_time_s",
      "imu/process_noise/gauss_markov/gyro/sigma",
      "imu/process_noise/gauss_markov/accel/correlation_time_s",
      "imu/process_noise/gauss_markov/accel/sigma",
      // R97A Q3 item 2: the motion-dependent Q floor scales.
      "imu/process_noise/motion/acc_floor_scale",
      "imu/process_noise/motion/gyro_floor_scale",
      // R97B: motion/beta (EMA smoothing of the per-step dynamic excitation
      // energy) and the acc/gyro_max_dynamic_variance clamp are removed --
      // the mid360 beta=0 vs beta=0.9 acc/gyro_scale sweep comparison found
      // no meaningful difference either way, so the per-step instantaneous
      // energy is now used directly and unclamped. NOTE: R97A Q3-fix had
      // previously restored the clamp after bisection found removing it
      // caused a real divergence on r97a_c_nya_02_id1 (NTU_VIRAL) -- that
      // risk was explicitly re-accepted here, it was not re-verified safe.
      "imu/process_noise/motion/beta",
      "imu/process_noise/motion/acc_max_dynamic_variance",
      "imu/process_noise/motion/gyro_max_dynamic_variance",
      // R97A Q3 item 4: the three Q scale alphas (all identity at 1.0).
      "imu/q_alpha_acc",
      "imu/q_alpha_gyr",
      "imu/q_alpha_bias",
      // R97A Q3 item 5: the rotation-warp branch.
      "imu/undistort/rot_warp_scale",
      "imu/undistort/rot_warp_last_frame",
      // R97A Q4 item 1: the stall-abort.
      "calib/stationary/stall_calls_max",
      // R97A Q4 item 2: both behaviours are always on (the yaml sets both true).
      "calib/stationary/apply_to_state/gyro_bias",
      "calib/stationary/apply_to_state/accel_bias",
      // R97A Q4 item 3: P0's position, velocity and attitude blocks are always exactly zero (the pinned path the yaml always selected).
      "calib/p0/mode",
      "calib/p0/known_pos_variance",
      "calib/p0/known_vel_variance",
      "calib/p0/known_theta_variance",
      // R97A Q4 item 5: state/est/{bg,ba,gravity} are always true.
      "state/est/bg",
      "state/est/ba",
      "state/est/gravity",
      // R97A Q4 item 6: no consumer survives items 3-5 (position/velocity/attitude P0 blocks always pinned to zero; bias P0
      // blocks always calibration-derived; initial vel/bias always overwritten by the mandatory stationary calibration).
      "state/cov/rot_tilt",
      "state/cov/rot_yaw",
      "state/cov/pos",
      "state/cov/vel",
      "state/cov/bg",
      "state/cov/ba",
      "state/init_state/vel",
      "state/init_state/ba",
      "state/init_state/bg"};
  return t;
}

// The release and followup that removed `key` (R74 F-427 for the first eight keys, R82 F-476 / F-403 for the next five, R83 F-401 / F-508 for lio/zupt/*, R84 F-521 for the temporal point-covariance keys, R85 F-520 for lio/point_cov/sample_window_s, R92 for voxel_map/binned/update_count_mode, R93A for the next seven keys, R95 for voxel_map/binned/weight_mode, R97 for estimator/mode, lio/estimator/mode and the 18 keys listed after them).
inline std::string removedKeyOrigin(const std::string& key)
{
  if (key == "imu/process_noise/motion/beta" || key == "imu/process_noise/motion/acc_max_dynamic_variance" ||
      key == "imu/process_noise/motion/gyro_max_dynamic_variance") return "R97B";
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
                        "lio/open_loop/mode", "lio/open_loop/reset_period_s",
                        "imu/process_noise/bias_model",
                        "imu/process_noise/motion/acc_floor_scale", "imu/process_noise/motion/gyro_floor_scale",
                        "imu/q_alpha_acc", "imu/q_alpha_gyr", "imu/q_alpha_bias",
                        "imu/undistort/rot_warp_scale", "imu/undistort/rot_warp_last_frame",
                        "calib/stationary/stall_calls_max",
                        "calib/stationary/apply_to_state/gyro_bias", "calib/stationary/apply_to_state/accel_bias",
                        "calib/p0/mode", "calib/p0/known_pos_variance", "calib/p0/known_vel_variance", "calib/p0/known_theta_variance",
                        "state/est/bg", "state/est/ba", "state/est/gravity",
                        "state/cov/rot_tilt", "state/cov/rot_yaw", "state/cov/pos", "state/cov/vel", "state/cov/bg", "state/cov/ba",
                        "state/init_state/vel", "state/init_state/ba", "state/init_state/bg"})
    if (key == k) return "R97A";
  if (key.rfind("lio/q_adapt/", 0) == 0 || key.rfind("imu/process_noise/gain/", 0) == 0 ||
      key.rfind("imu/process_noise/gauss_markov/", 0) == 0) return "R97A";
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
  // R97A Q5: the table's only entries now are the log/run move (R73's own entries were removed with their target in Q3).
  for (const auto& kv : renamedKeyTable())
    if (kv.first == key)
      return "configuration key '" + key + "' was renamed in R97A and is refused under its old name; use '" + kv.second + "'";
  return std::string();
}

}  // namespace livo_recon
