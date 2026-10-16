// R73 unit test (F-404): retired configuration key names are refused with a message that names the new key.
//   RK1 every retired name maps to a message that contains both the old and the new name.
//   RK2 the new names and unrelated keys give an empty message (not refused).
//   RK3 (R97A Q3) the ten q_adapt keys' RENAME TARGET (imu/process_noise/gain/*) is now itself removed, so both the old and new
//       names moved to the removed table instead of staying a rename; the table was empty after Q3 and Q5 repopulates it with
//       the 73 log/run key moves (r97a_inputs/log_key_map.tsv).
//   RK4 (R74 F-427, R82 F-476 and F-403, R83 F-401 and F-508, R84 F-521, R85 F-520) every removed key gives a message that contains the key and its release; the removed table has 137 entries (..., R97A Q1: voxel_map/backend and 8 more, R97A Q2: 17 more, R97A Q3: 36 more, R97A Q4: 19 more).
//   RK5 a removed key is never also a renamed-key target or source in the renamed table (no key can be both renamed and removed).
#include "livo_recon/utils/log/renamed_key_table.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  CHECK(renamedKeyTable().size() == 73, "RK3 renamed table has 73 entries (R97A Q5: the log/run key moves; R73's own entries were removed with their target in Q3)");
  for (const auto& kv : renamedKeyTable()) {
    const std::string m = renamedKeyMessage(kv.first);
    if (m.find(kv.first) == std::string::npos || m.find(kv.second) == std::string::npos) CHECK(false, "RK1 message names old and new key");
    if (!renamedKeyMessage(kv.second).empty()) CHECK(false, "RK2 new name is not refused");
  }
  CHECK(true, "RK1 every renamed-table entry's message names both keys");
  CHECK(renamedKeyMessage("lio/zupt/mode").empty(), "RK2 lio/zupt/mode is removed, not renamed");
  CHECK(renamedKeyMessage("imu/process_noise/model").empty(), "RK2 imu/process_noise/model is not retired");
  CHECK(renamedKeyMessage("lio/q_adapt").empty(), "RK2 a prefix alone is not a retired key");
  CHECK(removedKeyTable().size() == 137, "RK4 removed table has 137 entries (8 from R74, 5 from R82, 9 from R83, 4 from R84, 1 from R85, 1 from R92, 7 from R93A, 1 from R95, 20 from R97, 9 from R97A Q1, 17 from R97A Q2, 36 from R97A Q3, 19 from R97A Q4)");
  for (const auto& k : removedKeyTable()) {
    const std::string m = removedKeyMessage(k);
    if (m.find(k) == std::string::npos || m.find("removed in R7") == std::string::npos && m.find("removed in R82") == std::string::npos && m.find("removed in R83") == std::string::npos && m.find("removed in R84") == std::string::npos && m.find("removed in R85") == std::string::npos && m.find("removed in R92") == std::string::npos && m.find("removed in R93A") == std::string::npos && m.find("removed in R95") == std::string::npos && m.find("removed in R97") == std::string::npos && m.find("removed in R97A") == std::string::npos) CHECK(false, "RK4 message names the key and its release");
  }
  CHECK(true, "RK4 all 137 removed keys refused with a message that names the key");
  CHECK(removedKeyMessage("lio/zupt/mode").find("R83 (F-401, F-508)") != std::string::npos, "RK6 lio/zupt/mode names R83 F-401 F-508");
  CHECK(removedKeyMessage("lio/zupt/calib_margin_gyro").find("R83") != std::string::npos && removedKeyMessage("lio/zupt/sigma_v").find("R83") != std::string::npos, "RK6 every lio/zupt key is refused");
  CHECK(removedKeyMessage("lio/zupt").empty(), "RK5 the lio/zupt prefix alone is not a removed key");
  CHECK(removedKeyMessage("lio/point_cov/mode").find("R84 (F-521)") != std::string::npos
        && removedKeyMessage("lio/point_cov/window_s").find("R84 (F-521)") != std::string::npos
        && removedKeyMessage("lio/time_cov/mode").find("R84 (F-521)") != std::string::npos
        && removedKeyMessage("lio/time_cov/sigma_s").find("R84 (F-521)") != std::string::npos, "RK7 the removed temporal point-covariance keys name R84 F-521");
  CHECK(removedKeyMessage("lio/point_cov/sample_window_s").find("R85 (F-520)") != std::string::npos, "RK9 lio/point_cov/sample_window_s is refused and names R85 F-520");
  CHECK(removedKeyMessage("voxel_map/binned/update_count_mode").find("removed in R92") != std::string::npos, "RK10 voxel_map/binned/update_count_mode is refused and names R92");
  CHECK(removedKeyMessage("voxel_map/binned/cov_mode").empty() && removedKeyMessage("voxel_map/binned/bin_size_fraction").empty(), "RK10 the other binned keys are not removed");
  CHECK(removedKeyMessage("voxel_map/binned/weight_mode").find("removed in R95") != std::string::npos, "RK11 voxel_map/binned/weight_mode is refused and names R95");
  for (const char* k : {"voxel_map/residual/pose_cov_in_sigma", "voxel_map/plane/plane_fit_pose_cov_mode", "lio/point_cov/insertion_source", "voxel_map/plane/plane_gap_floor_en",
                        "voxel_map/plane/line_residual/classify_at_fit", "voxel_map/plane/debias_denominator_guard", "voxel_map/plane/plane_var_denom_floor_en"})
    CHECK(removedKeyMessage(k).find("removed in R93A") != std::string::npos, "RK8 the R93A keys are refused and name R93A");
  CHECK(removedKeyMessage("estimator/mode").find("removed in R97") != std::string::npos && removedKeyMessage("lio/estimator/mode").find("removed in R97") != std::string::npos, "RK12 estimator/mode and lio/estimator/mode are refused and name R97");
  CHECK(removedKeyMessage("estimator").empty(), "RK12 the estimator prefix alone is not a removed key");
  for (const char* k : {"cbk/lidar/time_offset", "cbk/lidar/overlap_policy", "cbk/lidar/retime_t0_s", "cbk/lidar/retime_period_s", "cbk/lidar/retime_seq0", "cbk/lidar/retime_ref_offset_s",
                        "voxel_map/plane/plane_fit_mode", "voxel_map/plane/debias_mode", "voxel_map/plane/plane_threshold_hysteresis", "voxel_map/plane/centred_accumulation", "voxel_map/plane/debiased_line_equiv",
                        "state/est/cov_acc", "state/est/cov_gyr", "frames/camera", "frames/lidar", "camera/distortion_model", "viz/grid_map", "imu/process_noise/switch/scan"})
    CHECK(removedKeyMessage(k).find("removed in R97") != std::string::npos, "RK13 the R97 keys are refused and name R97");
  CHECK(removedKeyMessage("cbk/lidar/blind").empty(), "RK13 cbk/lidar/blind is a live key");
  CHECK(removedKeyMessage("lio/gate_state_uncertainty").empty(), "RK8 lio/gate_state_uncertainty is a live key");
  CHECK(removedKeyMessage("lio/ekf/step_cap/mode").find("R82 (F-403)") != std::string::npos, "RK6 step_cap/mode names R82 F-403");
  CHECK(removedKeyMessage("imu/process_noise/gain/nis_lo").find("R82 (F-476)") != std::string::npos, "RK6 nis_lo names R82 F-476");
  CHECK(removedKeyMessage("imu/process_noise/disturbance/mode").find("R74 (F-427)") != std::string::npos, "RK6 R74 keys keep their R74 message");
  CHECK(removedKeyMessage("imu/process_noise/gain/nis_hi").find("R97A") != std::string::npos && removedKeyMessage("imu/process_noise/gain/inflate").find("R97A") != std::string::npos,
        "RK6 nis_gate keys that stayed through R82 are themselves removed by R97A Q3 (the whole gain module)");
  for (const auto& kv : renamedKeyTable())
    for (const auto& k : removedKeyTable())
      if (kv.first == k || kv.second == k) CHECK(false, "RK5 a key is both renamed and removed");
  CHECK(true, "RK5 no overlap between renamed and removed tables");
  CHECK(removedKeyMessage("imu/process_noise/gain/mode").find("R97A") != std::string::npos, "RK5 gain/mode (live through R82) is itself removed by R97A Q3 (the whole gain module)");
  CHECK(removedKeyMessage("imu/process_noise/model").empty(), "RK5 process_noise/model is not removed");
  CHECK(removedKeyMessage("imu/process_noise/disturbance").empty(), "RK5 a prefix alone is not a removed key");
  CHECK(removedKeyMessage("imu/process_noise/motion/beta").empty(), "RK5 motion/beta (kept) is not removed");
  for (const char* k : {"voxel_map/backend", "voxel_map/plane/min_frames_to_init", "voxel_map/plane/min_frames_to_converge", "voxel_map/plane/plane_gate_mode",
                        "voxel_map/plane/line_residual/enabled", "voxel_map/plane/plane_var_min", "voxel_map/plane/plane_var_scan_s2",
                        "voxel_map/map/freeze_after_calibration", "voxel_map/map/shuffle_insertion_seed"})
    CHECK(removedKeyMessage(k).find("removed in R97A") != std::string::npos, "RK14 the R97A Q1 keys are refused and name R97A");
  CHECK(removedKeyMessage("voxel_map/binned/bin_size_fraction").empty() && removedKeyMessage("voxel_map/plane/convergence_mode").empty()
        && removedKeyMessage("voxel_map/plane/line_residual/form").empty(), "RK14 the keys that stay (bin_size_fraction, convergence_mode, line_residual/form) are not removed");
  for (const char* k : {"lio/ekf/min_norm_dtheta", "lio/ekf/min_norm_dt", "lio/ekf/min_diff_error",
                        "lio/dry_run_point_filter_num", "lio/iekf_point_stream", "vio/log/dry_run",
                        "lio/residual_redundancy/mode", "lio/residual_redundancy/rho", "lio/residual_redundancy/renorm",
                        "lio/p_inflate/alpha", "lio/p_floor/min_eig", "lio/p_fading/lambda",
                        "lio/bootstrap_inflate/scans", "lio/bootstrap_inflate/factor", "lio/startup_hold/scans",
                        "lio/open_loop/mode", "lio/open_loop/reset_period_s"})
    CHECK(removedKeyMessage(k).find("removed in R97A") != std::string::npos, "RK15 the R97A Q2 keys are refused and name R97A");
  CHECK(removedKeyMessage("lio/ekf/max_iterations").empty() && removedKeyMessage("lio/gate_state_uncertainty").empty()
        && removedKeyMessage("lio/residual_binning/mode").empty() && removedKeyMessage("lio/info_scale/mode").empty(),
        "RK15 the keys that stay (max_iterations, gate_state_uncertainty, residual_binning, info_scale) are not removed");
  for (const char* k : {"lio/q_adapt/mode", "lio/q_adapt/apply", "lio/q_adapt/min_gain", "lio/q_adapt/max_gain",
                        "lio/q_adapt/forgetting", "lio/q_adapt/rho_clamp", "lio/q_adapt/nis_hi", "lio/q_adapt/inflate",
                        "lio/q_adapt/decay", "lio/q_adapt/whiteness_max",
                        "imu/process_noise/gain/mode", "imu/process_noise/gain/apply", "imu/process_noise/gain/min_gain",
                        "imu/process_noise/gain/max_gain", "imu/process_noise/gain/forgetting", "imu/process_noise/gain/rho_clamp",
                        "imu/process_noise/gain/nis_hi", "imu/process_noise/gain/inflate", "imu/process_noise/gain/decay",
                        "imu/process_noise/gain/whiteness_max", "imu/process_noise/gain/nis_slope", "imu/process_noise/gain/nis_clip",
                        "imu/process_noise/bias_model",
                        "imu/process_noise/gauss_markov/gyro/correlation_time_s", "imu/process_noise/gauss_markov/gyro/sigma",
                        "imu/process_noise/gauss_markov/accel/correlation_time_s", "imu/process_noise/gauss_markov/accel/sigma",
                        "imu/process_noise/motion/acc_floor_scale", "imu/process_noise/motion/gyro_floor_scale",
                        "imu/process_noise/motion/acc_max_dynamic_variance", "imu/process_noise/motion/gyro_max_dynamic_variance",
                        "imu/q_alpha_acc", "imu/q_alpha_gyr", "imu/q_alpha_bias",
                        "imu/undistort/rot_warp_scale", "imu/undistort/rot_warp_last_frame"})
    CHECK(removedKeyMessage(k).find("removed in R97A") != std::string::npos, "RK16 the R97A Q3 keys are refused and name R97A");
  CHECK(removedKeyMessage("imu/process_noise/bias_random_walk/gyro_variance_rate").empty()
        && removedKeyMessage("imu/process_noise/bias_random_walk/accel_variance_rate").empty()
        && removedKeyMessage("imu/process_noise/model").empty()
        && removedKeyMessage("imu/undistort/time_based_process_noise").empty(),
        "RK16 the keys that stay (bias_random_walk rates, process_noise/model, undistort/time_based_process_noise) are not removed");
  std::printf("ALL PASS\n");
  return 0;
}
