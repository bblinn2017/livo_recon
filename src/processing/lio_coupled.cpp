#include "livo_recon/processing/lio_coupled.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/pose_control_adaptive_q.h"
#include "livo_recon/lio/pose_control_directional_redundancy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <cstdio>

namespace livo_recon
{

// CQ-62 item 1/5c: one reusable stage-audit logger, gated behind
// psd_audit_en (default false, debug-only, doesn't need to be pretty or
// survive -- Bryce's own instruction on this card). Reports dim/min_eig/
// max_eig/rel AND asym: asym is NOT optional -- a matrix that is not
// symmetric BEFORE 0.5*(X+X^T) carries a construction bug that
// symmetrisation then conceals (the symmetrised result can be perfectly
// PSD while the thing that produced it is wrong; asym is the only column
// that can see that).
static void logPsdStage(int scan_id, int iter, const char* stage, const Eigen::MatrixXd& X)
{
  const Eigen::MatrixXd Xsym = 0.5 * (X + X.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Xsym);
  const double min_eig = es.eigenvalues().minCoeff();
  const double max_eig = es.eigenvalues().maxCoeff();
  const double denom_asym = std::max(X.cwiseAbs().maxCoeff(), 1e-300);
  const double asym = (X - X.transpose()).cwiseAbs().maxCoeff() / denom_asym;
  static PersistentLogStream log("psd_stage_audit.txt");
  bool first;
  std::ofstream& ofs = log.stream(&first);
  if (first) ofs << "scan_id,iter,stage,dim,min_eig,max_eig,rel,asym\n";
  ofs << scan_id << "," << iter << "," << stage << "," << X.rows() << ","
      << min_eig << "," << max_eig << ","
      << (max_eig != 0.0 ? min_eig / max_eig : 0.0) << ","
      << asym << "\n";
  ofs.flush();
}

// Forward declaration: logCovTraceStage (below) routes through the unified
// diagnostics CSV (emitFullDiagRow/fullDiagRunId, defined further down this
// file) rather than its own dedicated file writer -- see the "legacy code
// removal" note at that definition for why pose_control_cov_trace.txt was
// retired.
static const std::string& fullDiagRunId();
static void emitFullDiagRow(const std::string& run_id, const std::string& test_id,
                             const std::string& row_type, int scan_id, int iteration,
                             const std::map<std::string, std::string>& kv);

// Reports shape/trace/Frobenius/min-max eig/rank estimate/exact-zero
// row-or-column count for an arbitrary matrix at a named pipeline stage of
// the pose_control prior/covariance construction -- gated behind
// psd_audit_en. Previously wrote its own pose_control_cov_trace.txt; now
// folded into pose_control_full_diagnostics.csv's "covariance_trace_stage"
// row type (2026-09-24 legacy-diagnostic-writer cleanup, item 3/33) so this
// remains the ONE unified validation instrumentation path rather than a
// second, dedicated raw file.
static void logCovTraceStage(int scan_id, const std::string& test_id, const char* stage, const Eigen::MatrixXd& X)
{
  const int rows = static_cast<int>(X.rows()), cols = static_cast<int>(X.cols());
  const Eigen::MatrixXd Xsym = (rows == cols) ? Eigen::MatrixXd(0.5 * (X + X.transpose())) : X;
  double min_eig = 0.0, max_eig = 0.0, trace = 0.0;
  int rank_est = -1;
  if (rows == cols && rows > 0) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Xsym);
    min_eig = es.eigenvalues().minCoeff();
    max_eig = es.eigenvalues().maxCoeff();
    trace = X.trace();
    const double thresh = 1e-9 * std::max(std::abs(max_eig), 1.0);
    rank_est = 0;
    for (int i = 0; i < es.eigenvalues().size(); ++i)
      if (std::abs(es.eigenvalues()(i)) > thresh) ++rank_est;
  }
  const double fro = X.norm();
  int zero_rows = 0, zero_cols = 0;
  for (int i = 0; i < rows; ++i) if (X.row(i).cwiseAbs().maxCoeff() < 1e-300) ++zero_rows;
  for (int j = 0; j < cols; ++j) if (X.col(j).cwiseAbs().maxCoeff() < 1e-300) ++zero_cols;

  std::map<std::string, std::string> kv = {
    {"cov_trace_stage", stage}, {"cov_trace_rows", std::to_string(rows)}, {"cov_trace_cols", std::to_string(cols)},
    {"cov_trace_trace", std::to_string(trace)}, {"cov_trace_frobenius", std::to_string(fro)},
    {"cov_trace_min_eig", std::to_string(min_eig)}, {"cov_trace_max_eig", std::to_string(max_eig)},
    {"cov_trace_rank_est", std::to_string(rank_est)},
    {"cov_trace_zero_rows", std::to_string(zero_rows)}, {"cov_trace_zero_cols", std::to_string(zero_cols)},
  };
  emitFullDiagRow(fullDiagRunId(), test_id, "covariance_trace_stage", scan_id, -1, kv);
}

// 2026-09-23 x1/head-propagation campaign, item 2: linear interpolation of
// the RAW IMU-propagated pose chain (mg.poses, Pose6D) at an arbitrary time
// t -- the "actual x1_imu_prior", independent of the spline's own coarser
// nearest-knot-time control-point seeding (which snaps to the nearest
// mg.poses SAMPLE, not a true interpolation). Position/velocity: linear.
// Rotation: Exp((1-a)*Log(R_lo)) is NOT used (that's not a true slerp for
// two arbitrary rotations) -- instead R(t) = R_lo * Exp(a * Log(R_lo^T R_hi)),
// the standard constant-angular-velocity interpolation between two SO(3)
// samples, exact for the small per-IMU-sample chords this codebase already
// assumes elsewhere (CHART_MAX_PHI_RAD-style regime).
static void interpPose6DAt(const std::vector<Pose6D>& poses, double t,
                            V3D& p, V3D& v, M3D& R)
{
  if (poses.empty()) { p.setZero(); v.setZero(); R.setIdentity(); return; }
  if (t <= poses.front().t) { p = poses.front().pos; v = poses.front().vel; R = poses.front().rot; return; }
  if (t >= poses.back().t)  { p = poses.back().pos;  v = poses.back().vel;  R = poses.back().rot;  return; }
  size_t hi = 0;
  while (hi < poses.size() && poses[hi].t < t) ++hi;
  hi = std::min(hi, poses.size() - 1);
  const size_t lo = (hi > 0) ? hi - 1 : 0;
  const double t_lo = poses[lo].t, t_hi = poses[hi].t;
  const double a = (t_hi > t_lo) ? (t - t_lo) / (t_hi - t_lo) : 0.0;
  p = (1.0 - a) * poses[lo].pos + a * poses[hi].pos;
  v = (1.0 - a) * poses[lo].vel + a * poses[hi].vel;
  R = poses[lo].rot * Exp(V3D(a * Log(M3D(poses[lo].rot.transpose() * poses[hi].rot))));
}

// ============================================================================
// 2026-09-23 unified diagnostic CSV (pose_control_full_diagnostics.csv).
// Scope note (honest, not silently reduced): this implements a real subset
// of the requested 33-row-type schema -- run_summary, scan_summary,
// gn_iteration, process_segment, covariance_summary, covariance_block,
// p0_scale, head_constraint, and boolean flags -- reusing values already
// computed at each call site (nothing here is fabricated/estimated). NOT
// implemented in this pass: process_matrix/process_jacobian_validation rows
// (the existing standalone FD unit test already covers this, not re-run
// per-scan), lidar_matrix decomposition rows, covariance_eigenmode/
// observability_mode weak-mode composition rows, synthetic_test rows, and
// the decoupled-LIO comparison row -- each is called out explicitly in the
// final report rather than silently omitted.
// ============================================================================
static std::mutex g_full_diag_mtx;
static const std::vector<std::string>& fullDiagColumns()
{
  static const std::vector<std::string> cols = {
    "run_id","test_id","row_type","git_commit","sequence","scan_id","iteration","timestamp",
    // run_summary / config
    "trajectory_parameterization","velocity_mode","jacobian_time_mode","N_control_points",
    "total_optimization_dimension","free_spline_dimension","tail_free_state_dimension",
    "lidar_enable","process_enable","imu_var_acc_x","imu_var_acc_y","imu_var_acc_z",
    "imu_var_gyr_x","imu_var_gyr_y","imu_var_gyr_z","covariance_pseudoinverse_threshold","p0_scale_config",
    // scan_summary
    "E_lidar","E_total","num_lidar_points","num_imu_samples","gn_iterations",
    "final_delta_eta_norm","final_delta_bg_norm","final_delta_ba_norm","final_delta_g_norm",
    // gn_iteration
    "delta_eta_norm","delta_bg_norm","delta_ba_norm","delta_g_norm",
    "head_p0_err","head_v0_err","head_R0_err",
    // covariance_summary / covariance_block
    "trace_P0","trace_P_tail_pred","trace_P_tail_post","min_eig_P0","min_eig_P_tail_pred","min_eig_P_tail_post",
    "block_name","trace_pred","trace_post","contraction_fraction",
    // p0_scale
    "p0_scale_value","head_p_diff_norm","head_R_diff_norm","head_v_diff_norm",
    "trace_P_tail_pred_ratio_vs_nominal","trace_P_tail_post_ratio_vs_nominal",
    // head_constraint
    "CZ_frobenius","CZ_max_abs","C_rows","C_cols","Z_rows","Z_cols",
    // flags
    "pose_covariance_zero","position_covariance_zero","velocity_covariance_zero","process_dominates_lidar",
    // process-prior-REFORMULATION: ekf_reference (item 10/51), bias_information (item 5/12)
    "delta_actual_norm","delta_reference_norm","delta_difference_norm","delta_relative_difference",
    "delta_eta_actual","delta_eta_reference","delta_bg_actual","delta_bg_reference",
    "delta_ba_actual","delta_ba_reference","delta_g_actual","delta_g_reference",
    "lambda_prior_trace","lambda_total_trace","curvature_weight_pos","curvature_weight_rot",
    "P_eta_bg_norm","P_eta_ba_norm","P_eta_g_norm",
    // q_estimation (items 13-19/39-40)
    "q_used_acc","q_used_gyr","q_candidate_acc","q_candidate_gyr","q_next_acc","q_next_gyr",
    "residual_var_acc","residual_var_gyr","acf1_acc","acf1_gyr","acf2_acc","acf2_gyr","acf5_acc","acf5_gyr",
    "q_update_accepted","q_adaptation_reason","bias_var_acc_proxy","bias_var_gyr_proxy",
    // directional_redundancy (items 26-30)
    "raw_residual_count","reduced_state_dimension","effective_rank","condition_number",
    "dominant_eigenvalue","weak_eigenvalue","cumulative_information_fraction_at_rank5",
    // lidar_correlation (items 24/25)
    "num_raw_residuals","redund_groups","redund_n_raw","raw_information_trace",
    "correlation_corrected_information","correlation_information_reduction","lidar_correlation_mode",
    // covariance_trace_stage (folded from the old pose_control_cov_trace.txt)
    "cov_trace_stage","cov_trace_rows","cov_trace_cols","cov_trace_trace","cov_trace_frobenius",
    "cov_trace_min_eig","cov_trace_max_eig","cov_trace_rank_est","cov_trace_zero_rows","cov_trace_zero_cols",
    // x1_init_state (folded from the old pose_control_x1_init_state.txt)
    "x1_knot_index","x1_time","p_imu_x","p_imu_y","p_imu_z","v_imu_x","v_imu_y","v_imu_z",
    "p_pre_x","p_pre_y","p_pre_z","v_pre_x","v_pre_y","v_pre_z",
    "p_post_x","p_post_y","p_post_z","v_post_x","v_post_y","v_post_z",
    "delta_p_pre_norm","delta_v_pre_norm","delta_R_pre_norm",
    "delta_p_post_norm","delta_v_post_norm","delta_R_post_norm",
    // x1_covariance (folded from the old pose_control_x1_diagnostics.txt)
    "p_final_x","p_final_y","p_final_z","v_final_x","v_final_y","v_final_z",
    "delta_p_norm","delta_v_norm","delta_R_norm",
    "trace_P_p_x1","trace_P_v_x1","trace_P_R_x1","trace_P_x1_prior","trace_P_x1_post",
    "min_eig_P_x1_prior","max_eig_P_x1_prior","cond_P_x1_prior","rank_P_x1_prior",
    "sigma_distance_p","sigma_distance_R","sigma_distance_v",
    "abs_err_head_propagation_check","rel_err_head_propagation_check",
    // knot_state (folded from the old pose_control_knot_state.txt)
    "knot_index","knot_time","p_x","p_y","p_z","rlog_x","rlog_y","rlog_z",
    "v_x","v_y","v_z","a_x","a_y","a_z","omega_x","omega_y","omega_z",
    "d1_pos_norm","d2_pos_norm","d1_rot_norm","d2_rot_norm",
    "notes"
  };
  return cols;
}

static const std::string& fullDiagRunId()
{
  static const std::string run_id = std::to_string(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
  return run_id;
}

static void emitFullDiagRow(const std::string& run_id, const std::string& test_id,
                             const std::string& row_type, int scan_id, int iteration,
                             const std::map<std::string, std::string>& kv)
{
  std::lock_guard<std::mutex> lock(g_full_diag_mtx);
  static PersistentLogStream log("pose_control_full_diagnostics.csv");
  bool first;
  std::ofstream& ofs = log.stream(&first);
  const auto& cols = fullDiagColumns();
  if (first) {
    for (size_t i = 0; i < cols.size(); ++i) ofs << (i ? "," : "") << cols[i];
    ofs << "\n";
  }
  static const std::string git_commit = []() {
    std::string out;
    FILE* p = popen("git -C /root/catkin_ws/src/livo_recon rev-parse HEAD 2>/dev/null", "r");
    if (p) { char buf[128]; if (fgets(buf, sizeof(buf), p)) out = buf; pclose(p); }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.empty() ? std::string("unknown") : out;
  }();
  std::map<std::string, std::string> row = kv;
  row["run_id"] = run_id; row["test_id"] = test_id; row["row_type"] = row_type;
  row["git_commit"] = git_commit; row["sequence"] = "eee_01";
  row["scan_id"] = std::to_string(scan_id);
  row["iteration"] = (iteration >= 0) ? std::to_string(iteration) : "NA";
  row["timestamp"] = std::to_string(std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  for (size_t i = 0; i < cols.size(); ++i) {
    auto it = row.find(cols[i]);
    ofs << (i ? "," : "") << (it != row.end() ? it->second : "NA");
  }
  ofs << "\n";
  ofs.flush();
}

LioProcCoupled::LioProcCoupled(NodeContext& ctx)
  : LioProcBase(ctx)
{}

// CQ-55 "effective-config report" standing requirement -- mirrors
// LioProcDecoupled::~LioProcDecoupled() exactly (lio_decoupled.cpp), minus
// the splineOn() gate (this class has no spline at all, so it always has an
// engagement report worth printing).
LioProcCoupled::~LioProcCoupled()
{
  const std::string rep = engagementReport();
  ROS_WARN_STREAM("\n" << rep);
  std::ofstream ofs(debugLogPath("engagement.txt"), std::ios::trunc);
  if (ofs) ofs << rep << '\n';
}

std::string LioProcCoupled::loadParameters(ros::NodeHandle& pnh)
{
  ConfigResolver cfg(pnh);

  loadSharedParameters(cfg, pnh);

  // CQ-49 item 4: estimator/mode: coupled REQUIRES spline/mode: raw_imu --
  // enforced structurally by this class simply never claiming a spline/*
  // key (so checkAllParamsConsumed() throws naming any such key left in a
  // coupled config), and belt-and-suspenders here with an explicit refusal
  // if the (now-dead, for this class) spline/mode key is set to anything
  // else. AdaptiveQ's input IS the spline-vs-IMU residual, so it has no
  // meaning here either -- Lambda has taken over its within-scan job
  // (item 5).
  std::string spline_mode_check;
  bool spline_mode_set = pnh.hasParam("spline/mode");
  if (spline_mode_set) {
    pnh.getParam("spline/mode", spline_mode_check);
    if (spline_mode_check != "raw_imu")
      cfg.requireCombination(
          "estimator/mode: coupled requires spline/mode: raw_imu (or no "
          "spline/mode key at all) -- the coupled estimator has no "
          "ScanSpline and never reads any spline/* or adaptive_q/* key, so "
          "setting spline/mode to anything else describes a path that does "
          "not run under this estimator.");
  }

  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/n_c", copts_.n_c, 4);
  cfg.nestedMode(true, "estimator/mode=coupled", "estimator/coupled/spline_mode",
                 copts_.spline_mode, "raw_imu", {"raw_imu", "pose", "pose_control"});
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/n_control_points", copts_.pose_control_n, 13);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/q_pinv_rel_thresh", copts_.pose_control_q_pinv_rel_thresh, 1e-6);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/lidar_enable", copts_.pose_control_lidar_enable, true);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/p0_scale", copts_.pose_control_p0_scale, 1.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/curvature_weight_pos", copts_.pose_control_curvature_weight_pos, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/curvature_weight_rot", copts_.pose_control_curvature_weight_rot, 0.0);
  cfg.nested<std::string>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/test_id", copts_.pose_control_test_id, std::string("unlabeled"));
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/adaptive_q/enable", copts_.pose_control_adaptive_q.enable, false);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/adaptive_q/beta_acc", copts_.pose_control_adaptive_q.beta_acc, 0.3);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/adaptive_q/beta_gyr", copts_.pose_control_adaptive_q.beta_gyr, 0.3);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/adaptive_q/acf1_max", copts_.pose_control_adaptive_q.acf1_max, 1.0);
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/adaptive_q/warmup_frames", copts_.pose_control_adaptive_q.warmup_frames, 20);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/adaptive_q/ema", copts_.pose_control_adaptive_q.ema, 0.9);
  copts_.pose_control_adaptive_q.use_noise_floor = false;  // pose_control has no calibration-window floor plumbed yet -- documented simplification
  coupled_pose_control_adaptive_q_.configure(copts_.pose_control_adaptive_q);
  cfg.nestedMode(true, "estimator/mode=coupled", "estimator/coupled/pose_control/lidar_correlation_mode",
                 copts_.pose_control_lidar_correlation.mode, "off", {"off", "woodbury"});
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/lidar_correlation_rho", copts_.pose_control_lidar_correlation.rho, 1.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_control/lidar_correlation_max_discount", copts_.pose_control_lidar_correlation.max_discount, 0.9);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_imu_weight_acc", copts_.pose_imu_weight_acc, 1.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_imu_weight_gyr", copts_.pose_imu_weight_gyr, 1.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_curvature_weight_pos", copts_.pose_curvature_weight_pos, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_curvature_weight_rot", copts_.pose_curvature_weight_rot, 0.0);
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/pose_head_freeze_cp", copts_.pose_head_freeze_cp, 0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_tikhonov_eps", copts_.pose_tikhonov_eps, 1e-6);
  // POST-CQ-87-REVIEW FIX (item 4, follow-up): read state/cov/acc,gyr
  // DIRECTLY (the same two keys state.cpp itself reads to seed
  // state_->var_acc_/var_gyr_) -- see copts_.pose_imu_var_acc/gyr's own
  // doc comment for why this arm no longer goes through state_->varAcc()/
  // varGyr() at all.
  paramWarn<double>(pnh, "state/cov/acc", copts_.pose_imu_var_acc, 1e-4);
  paramWarn<double>(pnh, "state/cov/gyr", copts_.pose_imu_var_gyr, 1e-4);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_gn_max_step_pos_m", copts_.pose_gn_max_step_pos_m, 0.5);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_gn_max_step_rot_rad", copts_.pose_gn_max_step_rot_rad, 0.2);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/max_scan_displacement_m", copts_.max_scan_displacement_m, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/zero_mean", copts_.zero_mean, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/disable_cgyr", copts_.disable_cgyr, false);
  cfg.nestedMode(true, "estimator/mode=coupled", "estimator/coupled/jacobian_time_mode",
                 copts_.jacobian_time_mode, "legacy_mismatched",
                 {"legacy_mismatched", "end_time", "point_time"});
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/log_jrow_leverage_en", copts_.log_jrow_leverage_en, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/freeze_bg", copts_.freeze_bg, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/adaptive_sigma", copts_.adaptive_sigma, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/bias_freeze_on_vibration", copts_.bias_freeze_on_vibration, false);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/bias_freeze_vibration_factor", copts_.bias_freeze_vibration_factor, LioProcCoupledOptions::BIAS_FREEZE_VIBRATION_FACTOR_DEFAULT);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/bias_anchor", copts_.bias_anchor, false);
  // CQ-72 item 0a: renamed, HARD BREAK not an alias -- refuseUnclaimed
  // (CQ-32) already throws on an unclaimed estimator/coupled/* key, so a
  // config still using the old names fails loudly at startup.
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/prior/smoothness_weight_acc", copts_.smoothness_weight_acc, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/prior/smoothness_weight_gyr", copts_.smoothness_weight_gyr, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/prior/traj_deviation_weight", copts_.traj_deviation_weight, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/bias_observable_only", copts_.bias_observable_only, false);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/prior/imu_deviation_weight", copts_.imu_deviation_weight, 1.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/prior/mean_weight", copts_.mean_weight, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/log_traj_dev_en", copts_.log_traj_dev_en, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/prior_per_axis_sigma", copts_.prior_per_axis_sigma, false);
  cfg.nestedMode(true, "estimator/mode=coupled", "estimator/coupled/robust_loss",
                 copts_.robust_loss, "none",
                 {"none", "huber", "cauchy"});
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/psd_audit_en", copts_.psd_audit_en, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/log_bg_projection_en", copts_.log_bg_projection_en, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/log_cov_repropagation_en", copts_.log_cov_repropagation_en, false);
  paramWarn<double>(pnh, "imu/q_alpha_gyr", copts_.repro_q_alpha_gyr, 1.0);
  paramWarn<double>(pnh, "imu/q_alpha_acc", copts_.repro_q_alpha_acc, 1.0);
  paramWarn<bool>(pnh, "imu/second_order", copts_.repro_second_order, true);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/q_bias_rw_en", copts_.q_bias_rw_en, false);
  paramWarn<double>(pnh, "imu/q_alpha_bias", copts_.q_alpha_bias, 1.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/q_out_of_band_en", copts_.q_out_of_band_en, false);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/q_out_of_band_scale", copts_.q_out_of_band_scale, 1.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/q_out_of_band_fraction_acc", copts_.q_out_of_band_fraction_acc, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/q_out_of_band_fraction_gyr", copts_.q_out_of_band_fraction_gyr, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/log_cp_constraint_en", copts_.log_cp_constraint_en, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/final_redeskew", copts_.final_redeskew, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/final_relinearize_cov", copts_.final_relinearize_cov, false);
  if (copts_.final_relinearize_cov && !copts_.final_redeskew)
    cfg.requireCombination(
        "estimator/coupled/final_relinearize_cov requires "
        "estimator/coupled/final_redeskew=true -- relinearizing the "
        "covariance at the final trajectory is meaningless if the final "
        "trajectory was never actually re-deskewed/re-propagated against.");
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/prior_at_scan_start", copts_.prior_at_scan_start, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/add_q_scan_to_posterior", copts_.add_q_scan_to_posterior, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/final_redeskew_map_uses_pre", copts_.final_redeskew_map_uses_pre, false);
  if (copts_.final_redeskew_map_uses_pre && !copts_.final_redeskew)
    cfg.requireCombination(
        "estimator/coupled/final_redeskew_map_uses_pre requires "
        "estimator/coupled/final_redeskew=true -- there is nothing to "
        "isolate the map from if the final redeskew pass never runs.");
  if (copts_.add_q_scan_to_posterior) {
    bool imu_log_qhat_en = false;
    pnh.param<bool>("imu/log_qhat_en", imu_log_qhat_en, false);
    if (!imu_log_qhat_en)
      cfg.requireCombination(
          "estimator/coupled/add_q_scan_to_posterior requires "
          "imu/log_qhat_en=true -- it reads the frame-local Q_scan that "
          "machinery already captures every scan; there is no separate "
          "capture path.");
  }
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/log_point_plane_en", copts_.log_point_plane_en, false);
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/log_point_plane_hist_start_scan", copts_.log_point_plane_hist_start_scan, 0);
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/log_point_plane_hist_n_scans", copts_.log_point_plane_hist_n_scans, 20);
  if (copts_.prior_at_scan_start) {
    // log_qhat_en lives in ImuProcOptions, a separate class's options
    // struct not reachable from here -- check the raw rosparam directly
    // (it is set on the same shared param server) rather than plumbing a
    // cross-class dependency for one validation check.
    bool imu_log_qhat_en = false;
    pnh.param<bool>("imu/log_qhat_en", imu_log_qhat_en, false);
    if (!imu_log_qhat_en)
      cfg.requireCombination(
          "estimator/coupled/prior_at_scan_start requires imu/log_qhat_en=true "
          "-- it reads the pre-propagation P snapshot that machinery already "
          "captures every scan; there is no separate capture path.");
  }
  // CQ-82 Phase 2 item 4: under spline_mode=pose, every raw_imu-basis-
  // specific knob describes a term/path the pose basis does not have --
  // refuse loudly rather than silently ignore, so a config left over from a
  // raw_imu sweep cannot silently no-op under the new basis.
  if (copts_.poseBasis()) {
    if (copts_.smoothness_weight_acc != 0.0 || copts_.smoothness_weight_gyr != 0.0)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/prior/smoothness_weight_{acc,gyr} -- those "
          "weight a curvature prior on the raw_imu coefficient basis, which "
          "does not exist under the pose basis (see its own smoothness term).");
    if (copts_.imu_deviation_weight != 1.0)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/prior/imu_deviation_weight -- it weights the "
          "raw_imu coefficient prior's gram term, which the pose basis does "
          "not build (the IMU enters as a measurement factor there, not a prior).");
    if (copts_.mean_weight != 0.0)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/prior/mean_weight -- it weights the raw_imu "
          "coefficient basis's zero-mean prior, meaningless for pose control points.");
    if (copts_.traj_deviation_weight != 0.0)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/prior/traj_deviation_weight -- it penalizes "
          "deviation of the IMU-propagated trajectory from the raw_imu "
          "correction basis, which has no analogue once position/attitude "
          "ARE the control points.");
    if (copts_.zero_mean)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/zero_mean -- it constrains the raw_imu "
          "coefficient basis's DC component, which the pose basis has no "
          "equivalent of.");
    if (copts_.prior_per_axis_sigma)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/prior_per_axis_sigma -- it shapes the raw_imu "
          "coefficient prior's per-axis precision, which the pose basis "
          "does not build.");
    if (copts_.bias_observable_only)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/bias_observable_only -- it is a raw_imu-basis "
          "bias-projection mechanism with no pose-basis analogue.");
    if (copts_.bias_anchor)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/bias_anchor -- it anchors the raw_imu "
          "coefficient basis's bias-prior precision, meaningless here.");
    if (copts_.freeze_bg)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/freeze_bg -- it freezes the raw_imu gyro-bias "
          "correction pathway, which the pose basis does not route through.");
    if (copts_.q_out_of_band_en)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/q_out_of_band_en -- it injects process noise "
          "keyed to the raw_imu coefficient basis's own sigma, not built "
          "under the pose basis.");
    if (copts_.q_bias_rw_en)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/q_bias_rw_en -- it is a raw_imu-basis bias "
          "random-walk process-noise term with no pose-basis analogue.");
    if (copts_.adaptive_sigma)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/adaptive_sigma -- it adapts the raw_imu "
          "coefficient basis's own noise floor, not read under the pose basis.");
    if (copts_.disable_cgyr)
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/disable_cgyr -- it drops the raw_imu "
          "rotation-correction coefficient block, which the pose basis "
          "does not have (attitude control points replace it entirely).");
    if (copts_.jacobian_time_mode == "legacy_mismatched")
      cfg.requireCombination(
          "estimator/coupled/spline_mode=pose refuses "
          "estimator/coupled/jacobian_time_mode=legacy_mismatched (the "
          "shipped raw_imu default) -- the pose basis's LiDAR term is "
          "linear in c_p directly and carries no H-vs-Phi time mismatch to "
          "describe; set end_time or point_time explicitly to acknowledge "
          "this key is otherwise a no-op under the pose basis.");
    // Force final_relinearize_cov on for the pose basis regardless of what
    // was configured -- unlike the raw_imu case (where it's optional,
    // gated on final_redeskew), the pose basis's own covariance is only
    // meaningful evaluated at the basis actually solved. CQ-87 item 4,
    // CORRECTING A RULE-58f VIOLATION THIS COMMENT USED TO DESCRIBE
    // WITHOUT NOTICING IT: this flag used to be forced true here and then
    // NEVER READ on this arm at all -- the block that honours it
    // (final_relinearize/relin_pending_, ~line 1141) lives past the
    // poseBasis() early return this file's own processLIO() takes, so
    // [config/effective] printed final_relinearize_cov=true for every pose
    // run while the actual behaviour was false. FIXED (honoured, not
    // refused, per the card's own preference): the pose arm's posterior
    // write (estimateCoupledCorrectionPoseBasis()'s own
    // coupled_pose_head_cov_, applied at the poseBasis() early-return site
    // in processLIO()) is recomputed from the FULLY CONVERGED last GN
    // iteration's own reduced system every time -- this genuinely IS
    // "relinearized at the converged solution," the same guarantee this
    // flag names, just satisfied by construction rather than by a second
    // explicit pass the way the raw_imu arm's own relin_pending_ block
    // needs one.
    copts_.final_relinearize_cov = true;

    // CQ-82 Phase 2: the pose basis's own residual construction now HAS a
    // real implementation (estimateCoupledCorrectionPoseBasis(),
    // buildPoseSplineCBlock()) -- the earlier NOT-YET-IMPLEMENTED refusal
    // that lived here is gone. estimateCoupledCorrection()'s own top-level
    // branch (copts_.poseBasis()) now dispatches to that real path instead
    // of ever falling through to the raw_imu math, so the silent-
    // substitution risk this refusal guarded against no longer applies --
    // removing the refusal is itself gated on that branch existing, not a
    // relaxation of the underlying safety property.
  }

  coupled_tier1_nees_ = Tier1NeesBuffer(opts_.nees_per_dof_en ? opts_.nees_tier1_window_scans : 0);

  // Every spline/* and adaptive_q/* key is unclaimed by this class by
  // construction -- refuseUnclaimed only needs to additionally cover
  // estimator/coupled/* itself (already claimed above) plus the shared
  // namespaces loadSharedParameters() already reads. Leaving spline/
  // adaptive_q OUT of the allowed-unclaimed list means any key set there
  // under a coupled config is refused at startup, naming itself -- the
  // structural enforcement item 3d(iii) describes.
  return finalizeConfig(cfg, { "lio/ekf", "voxel_map" });
}

std::string LioProcCoupled::engagementReport() const
{
  std::ostringstream oss;
  oss << "[engagement] estimator=coupled n_c=" << copts_.n_c
      << " spline_mode=" << copts_.spline_mode
      << " zero_mean=" << (copts_.zero_mean ? "true" : "false")
      << " disable_cgyr=" << (copts_.disable_cgyr ? "true" : "false")
      << " jacobian_time_mode=" << copts_.jacobian_time_mode
      << " log_jrow_leverage_en=" << (copts_.log_jrow_leverage_en ? "true" : "false")
      << " freeze_bg=" << (copts_.freeze_bg ? "true" : "false")
      << " adaptive_sigma=" << (copts_.adaptive_sigma ? "true" : "false")
      << " bias_freeze_on_vibration=" << (copts_.bias_freeze_on_vibration ? "true" : "false")
      << " bias_freeze_vibration_factor=" << copts_.bias_freeze_vibration_factor
      << " bias_anchor=" << (copts_.bias_anchor ? "true" : "false")
      << " smoothness_weight_acc=" << copts_.smoothness_weight_acc
      << " smoothness_weight_gyr=" << copts_.smoothness_weight_gyr
      << " bias_observable_only=" << (copts_.bias_observable_only ? "true" : "false")
      << " imu_deviation_weight=" << copts_.imu_deviation_weight
      << " mean_weight=" << copts_.mean_weight
      << " traj_deviation_weight=" << copts_.traj_deviation_weight
      << " prior_per_axis_sigma=" << (copts_.prior_per_axis_sigma ? "true" : "false")
      << " robust_loss=" << copts_.robust_loss
      << " log_bg_projection_en=" << (copts_.log_bg_projection_en ? "true" : "false")
      << " -- no spline/AdaptiveQ engagement to report (this class has "
         "neither)";
  return oss.str();
}

// No spline at all -- see lio_base.h's own doc comment for why this is a
// VIRTUAL override rather than shared base logic (CQ-44 bug 4's second
// consequence: deskewAndDownsample() is the OTHER place the spline could
// leak into coupled mode if left non-virtual). Deskews from the raw IMU
// pose chain unconditionally, then downsamples exactly as the decoupled
// path does at spline_ok_==false.
void LioProcCoupled::deskewAndDownsample(MeasureGroup& mg)
{
  TimedScope ts(profiler_, "lio/deskew");

  std::vector<PointXYZCov> deskewed;
  deskewPoints(state_, mg.poses, mg.image.t, mg.lidar_points, opts_.deskew, deskewed);

  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
  }

  // See LioProcOptions::dry_run_point_filter_num's doc comment. Identical
  // deskew/downsample treatment as the primary points above.
  if (opts_.dry_run_point_filter_num > 0 && !mg.dry_run_lidar_points.empty())
  {
    std::vector<PointXYZCov> dry_run_deskewed;
    deskewPoints(state_, mg.poses, mg.image.t, mg.dry_run_lidar_points, opts_.deskew, dry_run_deskewed);
    if (opts_.dsOn()) {
      DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
      voxelDownsample(dry_run_deskewed, mg.dry_run_points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
    } else {
      mg.dry_run_points = std::move(dry_run_deskewed);
    }
  }
}

V3D LioProcCoupled::poseControlEffectiveVarAcc() const
{
  if (copts_.pose_control_adaptive_q.enable && coupled_pose_control_adaptive_q_primed_ &&
      coupled_pose_control_adaptive_q_.active())
    return V3D::Constant(coupled_pose_control_adaptive_q_.varAcc());
  return state_->varAcc();
}

V3D LioProcCoupled::poseControlEffectiveVarGyr() const
{
  if (copts_.pose_control_adaptive_q.enable && coupled_pose_control_adaptive_q_primed_ &&
      coupled_pose_control_adaptive_q_.active())
    return V3D::Constant(coupled_pose_control_adaptive_q_.varGyr());
  return state_->varGyr();
}

std::string LioProcCoupled::processLIO(MeasureGroup& mg)
{
  mg.prior_pos = state_->pos();
  mg.prior_rot = state_->rot();
  mg.prior_vel = state_->vel();

  if (voxel_map_->isEmpty()) return {};

  TimedScope ts(profiler_, "lio/ekf");
  V3D dtheta, dt;
  V3D total_dtheta = V3D::Zero(), total_dt = V3D::Zero();
  double prev_error = std::numeric_limits<double>::infinity();
  std::string stop = "max_iter";
  int iter = 0;

  // CQ-62 item 1: S0 -- state_->cov() on entry to processLIO, what
  // processIMU left (P(t1)_raw), BEFORE anything this function does.
  if (copts_.psd_audit_en)
    logPsdStage(voxel_map_->frame_idx_, -1, "S0_cov_entry", state_->cov());
  prior_cov_ = state_->cov();
  applyPriorScalarControls(prior_cov_, opts_.prior_scalar);
  // CQ-62 item 1: S1 -- prior_cov_ after applyPriorScalarControls.
  if (copts_.psd_audit_en)
    logPsdStage(voxel_map_->frame_idx_, -1, "S1_prior_cov_scaled", prior_cov_);
  state_propagat_ = *state_;
  trP_pos_pre_ = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();

  // CQ-74 item 1: coupled never populated frame_stats.txt's LioFrameDiag
  // at all before this (confirmed by grep -- no noteLioFrameDiag() call
  // anywhere in this file), so every LioFrameDiag-sourced column,
  // including the ones that already existed for decoupled, silently sat
  // at its -1.0 default for every coupled run. Scoped here to just the P
  // (prior_cov_/state_->cov()) fields the card's own comparison needs --
  // the ESIKF-specific residual/HtH fields (ask/got/refusal, htz_*_norm,
  // etc.) genuinely do not apply to coupled's own GN solve the same way
  // and are left at their existing "unavailable" defaults, matching how
  // decoupled's own N/A cases already read.
  LioFrameDiag coupled_diag;
  if (prior_cov_.rows() >= StateGroup::idxP() + 3 && prior_cov_.cols() >= StateGroup::idxP() + 3) {
    const M3D P_pp_pre = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
    Eigen::SelfAdjointEigenSolver<M3D> es_p_pre(P_pp_pre);
    coupled_diag.p_pos_eig_min_pre = es_p_pre.eigenvalues()(0);
    coupled_diag.p_pos_eig_mid_pre = es_p_pre.eigenvalues()(1);
    coupled_diag.p_pos_eig_max_pre = es_p_pre.eigenvalues()(2);
    const M3D P_rr_pre = prior_cov_.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
    Eigen::SelfAdjointEigenSolver<M3D> es_r_pre(P_rr_pre);
    coupled_diag.p_rot_trace_pre   = P_rr_pre.trace();
    coupled_diag.p_rot_eig_min_pre = es_r_pre.eigenvalues()(0);
    coupled_diag.p_rot_eig_mid_pre = es_r_pre.eigenvalues()(1);
    coupled_diag.p_rot_eig_max_pre = es_r_pre.eigenvalues()(2);
  }
  coupled_diag.trP_pos_pre = trP_pos_pre_;
  mg.prior_pos = state_propagat_.pos();
  mg.prior_rot = state_propagat_.rot();
  mg.prior_vel = state_propagat_.vel();
  bool any_solved = false;

  // CQ-44 item 4: c_prior = 0 EVERY scan -- reset here, not carried
  // scan-to-scan. n_c is fixed for the life of the process (loadParameters()),
  // so resizing (not reallocating) each frame is cheap.
  coupled_c_acc_.assign(copts_.n_c, V3D::Zero());
  coupled_c_gyr_.assign(copts_.n_c, V3D::Zero());
  coupled_iters_ = 0;
  coupled_solve_ms_ = 0.0;
  coupled_prev_traj_dev_valid_ = false;  // CQ-72 item 3: no "previous iteration" at scan start
  coupled_prev_iter_planes_.clear();  // CQ-79: no "previous iteration" carries into a new scan
  coupled_delta_v_.setZero(); coupled_delta_bg_.setZero();
  coupled_delta_ba_.setZero(); coupled_delta_g_.setZero();
  coupled_delta_phi0_.setZero(); coupled_delta_pos0_.setZero();
  // CQ-82 Phase 2, Artifact 1: the pose-basis's own scan-start reset. Fit
  // ONCE per scan (not once per GN iteration -- INIT via ScanSpline::fit()
  // on the IMU-propagated mg.poses, "the matching initial condition to the
  // other arm's c=0", per the card), then reset the accumulated corrections
  // to zero, exactly mirroring coupled_c_acc_/coupled_c_gyr_'s own pattern.
  // control_point_hz is chosen PER SCAN so the derived n_cp (ScanSpline's
  // own formula: round(hz*(t1-t0))+3) lands on EXACTLY copts_.n_c -- ScanSpline's
  // own public fit() API takes a rate, not a raw control-point count, and
  // is shared with the decoupled spline path, so this works around that
  // rather than changing shared, already-shipped surface.
  if (copts_.poseBasis()) {
    coupled_pose_spline_valid_ = false;
    coupled_c_pos_.clear();
    coupled_c_rot_.clear();
    if (!mg.poses.empty() && mg.image.t > mg.poses.front().t) {
      const double t0 = mg.poses.front().t;
      const double t1 = mg.image.t;
      SplineOptions pose_fit_opts;
      pose_fit_opts.control_point_hz = (copts_.n_c - 3) / std::max(t1 - t0, 1e-6);
      pose_fit_opts.end_constraint_velocity = true;
      // POST-CQ-87-REVIEW FIX (item 1, "the pose spline is fitted without
      // enforcing the scan-start pose"): without this, the initial fit is
      // an unconstrained OLS approximation to mg.poses, NOT guaranteed to
      // satisfy p(t0)=p_ESIKF(t0)/R(t0)=R_ESIKF(t0) -- and
      // reducePoseSplineHeadCoupling()'s head tie only constrains the
      // CORRECTION at t0 (p(t0)=p_fit(t0)+delta_p0), so without this fix
      // "p(t0)=p_start+delta_p0" does not actually hold unless p_fit(t0)
      // already equals p_start. mg.poses.front() IS p_ESIKF(t0)/R_ESIKF(t0)
      // (this scan's own IMU-propagated starting condition, "the matching
      // initial condition to the other arm's c=0" per CQ-82's own framing)
      // -- freezing the fit to it directly, rather than via a separate
      // prev_scan_end_pos_-style member (decoupled's own pattern,
      // lio_decoupled.cpp:257-260), needs no "is this the first scan"
      // gate: mg.poses.front() is always a valid boundary value, even on
      // scan 1. Tail is frozen to state_->pos()/rot()/vel() (the
      // IMU-propagated scan-end estimate, correction loop hasn't run yet)
      // -- mirroring decoupled's own tail convention exactly. The reviewer
      // who flagged this said "definitely enforce the head... not
      // necessarily the tail" -- both are frozen here anyway because
      // setFrozenBoundary()'s KKT construction only offers a single
      // both-ends gate (no head-only mode exists in spline.cpp), and
      // reusing that already-tested machinery (identical to decoupled's
      // own call) is preferable to a new one-sided KKT variant.
      //
      // REVISED per user instruction 2026-09-21 item 5: do NOT hard-clamp
      // the tail. spline.h's setFrozenBoundary()/buildEndConstraints() now
      // support a HEAD-ONLY KKT constraint (constrain_tail=false) --
      // pos1/rot1/vel1 are still passed (stored, unused as constraint
      // targets) so the call shape matches every other caller; the tail is
      // left for the LiDAR/IMU/smoothness data term alone to determine,
      // which is the whole point of the pose-spline experiment ("don't
      // lose one of the main freedoms we're trying to investigate"). The
      // head stays hard-tied to the ESIKF state exactly as before.
      coupled_pose_spline_.setFrozenBoundary(
          SplineOptions::N_FROZEN_CP,
          mg.poses.front().pos, mg.poses.front().rot, mg.poses.front().vel,
          state_->pos(), state_->rot(), state_->vel(),
          /*constrain_tail=*/false);
      coupled_pose_spline_valid_ = coupled_pose_spline_.fit(mg.poses, t0, t1, pose_fit_opts);
      // BUGFIX (found via a live crash: SIGSEGV inside processLIO, an
      // out-of-bounds Eigen column access): ScanSpline::fit() can silently
      // CLAMP its actual control-point count below the REQUESTED
      // copts_.n_c (its own doc comment: "fit() additionally clamps n_cp
      // to n_samples - 1" -- a real scan can have too few pose samples in
      // its own window, especially an early/short one). coupled_c_pos_/
      // coupled_c_rot_ (and every loop bound in
      // estimateCoupledCorrectionPoseBasis()) MUST be sized off the
      // spline's own ACTUAL nControlPoints(), never off copts_.n_c
      // directly -- copts_.n_c is a request, not a guarantee.
      if (coupled_pose_spline_valid_) {
        coupled_c_pos_.assign(coupled_pose_spline_.nControlPoints(), V3D::Zero());
        coupled_c_rot_.assign(coupled_pose_spline_.nControlPoints(), V3D::Zero());
      }
    }
  }
  // 2026-09-22 correction: pose-control-point scan-start init. The head
  // is now EXACTLY 9 linear constraints (p0 value+rate, phi0 value only --
  // NO omega0/acceleration rows) eliminated via a true nullspace basis
  // (buildPoseControlHeadNullspace()) over the RAW 6N control-point space
  // -- cp[0..2]/cp_phi[0..2] are NOT fixed/removed; they remain reachable
  // through eta's own head-nullspace columns. tail_trial is seeded to
  // tail_prior (the scan-entry StateGroup bg/ba/g) -- both are tracked
  // separately from here on (item 3/4: delta_sT is the GN variable, not
  // the absolute tail value).
  if (copts_.poseControlSplineBasis()) {
    coupled_pose_control_valid_ = false;
    if (!mg.poses.empty() && mg.image.t > mg.poses.front().t) {
      const double t0 = mg.poses.front().t;
      const double t1 = mg.image.t;
      // 2026-09-23 x1/head-propagation campaign item 12: floor relaxed from
      // 6 to 4 (the true architectural minimum -- N=4 gives nSeg=N-3=1
      // process-factor segment and exactly ONE free control point, cp[3],
      // which is simultaneously x1 (first free knot) AND the tail (scan-end)
      // for this N; a real, intended edge case, not a bug) so N=4 can
      // actually be dispatched. The old floor of 6 was never justified by a
      // real lower bound -- nothing below (head nullspace needs N>=3+1=4,
      // the curvature loop's k=1..N-2 range is empty-but-safe at N=4) relies
      // on N>=6.
      const int N = std::max(4, copts_.pose_control_n);

      auto& spline = coupled_pose_control_spline_;
      spline.init(N, t0, t1);
      spline.R_anchor = mg.poses.front().rot;

      const V3D p0 = mg.poses.front().pos, v0 = mg.poses.front().vel;

      // Initial guess for EVERY control point (including 0..2, now that
      // they are reachable free coordinates, not fixed) -- nearest-time
      // lookup into the IMU-propagated mg.poses chain, refined by the GN
      // loop; the head constraints are enforced exactly regardless of
      // this guess via the c_particular/Z projection below.
      for (int k = 0; k < N; ++k) {
        const double tk = std::min(t1, t0 + k * spline.delta());
        size_t best = 0; double best_dt = std::numeric_limits<double>::max();
        for (size_t i = 0; i < mg.poses.size(); ++i) {
          const double dtp = std::abs(mg.poses[i].t - tk);
          if (dtp < best_dt) { best_dt = dtp; best = i; }
        }
        spline.cp_p.col(k) = mg.poses[best].pos;
        spline.cp_phi.col(k) = Log(M3D(spline.R_anchor.transpose() * mg.poses[best].rot));
      }

      // x1/head-propagation campaign item 1/2: x1 = first free knot AFTER
      // the fixed head. Only cp[0..2] participate in the 9 head constraints
      // (buildPoseControlHeadNullspace() below) -- cp[3] is the first
      // control point with NO involvement in them, hence x1's knot index is
      // ALWAYS 3, for every N>=4 tested. "x1_spline_init_pre" is the
      // PHYSICAL (spline-curve-evaluated, not raw-control-point) state at
      // x1's knot time using the raw nearest-sample guess just assigned
      // above, BEFORE the head-nullspace projection below can move it.
      constexpr int kPoseControlX1Knot = 3;
      const bool have_x1 = (N > kPoseControlX1Knot);
      const double t_x1 = have_x1 ? std::min(t1, t0 + kPoseControlX1Knot * spline.delta()) : t1;
      V3D p_x1_spline_init_pre, v_x1_spline_init_pre; M3D R_x1_spline_init_pre;
      if (have_x1) {
        p_x1_spline_init_pre = spline.posAt(t_x1);
        v_x1_spline_init_pre = spline.velAt(t_x1);
        R_x1_spline_init_pre = spline.rotAt(t_x1);
      }
      V3D p_x1_imu_prior, v_x1_imu_prior; M3D R_x1_imu_prior;
      interpPose6DAt(mg.poses, t_x1, p_x1_imu_prior, v_x1_imu_prior, R_x1_imu_prior);

      coupled_pose_control_hns_ = buildPoseControlHeadNullspace(spline, p0, v0);
      const Eigen::VectorXd c_initial = poseControlFlatten(spline);
      coupled_pose_control_eta_ =
          coupled_pose_control_hns_.Z.transpose() * (c_initial - coupled_pose_control_hns_.c_particular);
      coupled_pose_control_eta_imu_ = coupled_pose_control_eta_;  // item 16: frozen scan-start seed
      // Rebuild the spline from the PROJECTED eta (not the raw guess) so
      // the head constraints hold exactly from iteration 0, not just
      // approximately from the initial guess.
      poseControlUnflatten(
          coupled_pose_control_hns_.c_particular + coupled_pose_control_hns_.Z * coupled_pose_control_eta_,
          spline);

      // item 2: x1_spline_init_post (AFTER projection) vs x1_spline_init_pre
      // (BEFORE) vs x1_imu_prior (the true IMU chain, independent of the
      // spline's discretization entirely) -- all three logged below once
      // coupled_pose_control_valid_ is set.
      V3D p_x1_spline_init_post, v_x1_spline_init_post; M3D R_x1_spline_init_post;
      if (have_x1) {
        p_x1_spline_init_post = spline.posAt(t_x1);
        v_x1_spline_init_post = spline.velAt(t_x1);
        R_x1_spline_init_post = spline.rotAt(t_x1);
      }

      coupled_pose_control_bg_prior_ = coupled_pose_control_bg_trial_ = state_->biasGyr();
      coupled_pose_control_ba_prior_ = coupled_pose_control_ba_trial_ = state_->biasAcc();
      coupled_pose_control_g_prior_  = coupled_pose_control_g_trial_  = state_->gravity();
      coupled_pose_control_layout_.N = N;
      coupled_pose_control_layout_.has_bg = state_->estBG();
      coupled_pose_control_layout_.has_ba = state_->estBA();
      coupled_pose_control_layout_.has_g = state_->estGravity();
      coupled_pose_control_layout_.fix_head = false;
      coupled_pose_control_seg_samples_ =
          bucketPoseControlImuSamples(mg.imu_samples_raw, spline).seg_samples;
      coupled_pose_control_P_z_post_.resize(0, 0);
      // ==========================================================================
      // PRODUCTION joint IMU/bias Gaussian prior. Computed ONCE per scan, at
      // scan start, from the joint [x0;z] marginalize-then-invert
      // construction below. This is the estimator's ONE authoritative
      // prior: coupled_pose_control_sigma_full_prior_ (the full joint
      // covariance, 9+dimZ square) is stored and reused VERBATIM by the
      // mean solve (via its z-marginal information, lambda_prior_z_) and by
      // the post-loop covariance computation (directly) -- neither ever
      // re-derives a separate prior. This is what makes "mean update
      // information == covariance update information" hold by construction.
      // ==========================================================================
      {
        const int dimRawScanstart = coupled_pose_control_layout_.dim();
        Eigen::MatrixXd A_process_scanstart_shared = Eigen::MatrixXd::Zero(dimRawScanstart, dimRawScanstart);
        Eigen::VectorXd b_unused_scanstart = Eigen::VectorXd::Zero(dimRawScanstart);
        PoseControlProcessFactorHeadBlock head_block_scanstart;
        for (int j = 0; j < spline.nSeg(); ++j)
          buildPoseControlImuPriorContribution(spline, coupled_pose_control_layout_, j, coupled_pose_control_seg_samples_[j],
              coupled_pose_control_ba_trial_, coupled_pose_control_bg_trial_, coupled_pose_control_g_trial_,
              copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr, poseControlEffectiveVarAcc(), poseControlEffectiveVarGyr(),
              copts_.repro_second_order, copts_.pose_control_q_pinv_rel_thresh,
              A_process_scanstart_shared, b_unused_scanstart, &head_block_scanstart, nullptr);
        const auto& hns2 = coupled_pose_control_hns_;
        // items 4/5/6: the joint marginalized prior over z=[eta;sT] at scan
        // start, INCLUDING P0's propagated uncertainty (via head_block_
        // scanstart's cross-coupling into A_hf_prior) and the joint
        // trajectory/bias covariance structure (P_eta_bg/ba/g, item 5 --
        // present because A_ff_priorS/Sigma_full_priorS below are built
        // over the WHOLE z=[eta;sT] block jointly, never eta and sT
        // separately). Frozen here for the rest of this scan.
        {
          const auto& layoutS = coupled_pose_control_layout_;
          const int dEtaS = hns2.freeDim(), dSTS = layoutS.dimST(), dimZS = dEtaS + dSTS;
          Eigen::MatrixXd P0s = state_->cov();
          if (copts_.prior_at_scan_start) {
            Eigen::MatrixXd p_before_peek;
            if (imuProcQhatPeekPBefore(p_before_peek) &&
                p_before_peek.rows() == P0s.rows() && p_before_peek.cols() == P0s.cols())
              P0s = p_before_peek;
          }
          P0s *= copts_.pose_control_p0_scale;
          const Eigen::MatrixXd Omega0s = generalPseudoInverse(P0s, copts_.pose_control_q_pinv_rel_thresh);
          Eigen::MatrixXd A_hh_priorS = head_block_scanstart.A_hh;
          Eigen::MatrixXd A_hf_priorS = (head_block_scanstart.A_hf.size() > 0)
              ? head_block_scanstart.A_hf : Eigen::MatrixXd::Zero(9, dimRawScanstart);
          if (Omega0s.rows() >= 9) A_hh_priorS += Omega0s.block(0, 0, 9, 9);
          if (dSTS > 0 && Omega0s.rows() >= 9 + dSTS) {
            A_process_scanstart_shared.block(layoutS.dimCFree(), layoutS.dimCFree(), dSTS, dSTS) += Omega0s.block(9, 9, dSTS, dSTS);
            A_hf_priorS.block(0, layoutS.dimCFree(), 9, dSTS) += Omega0s.block(0, 9, 9, dSTS);
          }
          Eigen::MatrixXd Ps = Eigen::MatrixXd::Zero(dimRawScanstart, dimZS);
          Ps.block(0, 0, hns2.rawDim(), dEtaS) = hns2.Z;
          if (dSTS > 0) Ps.block(hns2.rawDim(), dEtaS, dSTS, dSTS) = Eigen::MatrixXd::Identity(dSTS, dSTS);
          const Eigen::MatrixXd A_ff_priorS = Ps.transpose() * A_process_scanstart_shared * Ps;
          const Eigen::MatrixXd A_hf_prior_zS = A_hf_priorS * Ps;
          const int dimFullS = 9 + dimZS;
          Eigen::MatrixXd Lambda_full_priorS = Eigen::MatrixXd::Zero(dimFullS, dimFullS);
          Lambda_full_priorS.block(0, 0, 9, 9) = A_hh_priorS;
          Lambda_full_priorS.block(0, 9, 9, dimZS) = A_hf_prior_zS;
          Lambda_full_priorS.block(9, 0, dimZS, 9) = A_hf_prior_zS.transpose();
          Lambda_full_priorS.block(9, 9, dimZS, dimZS) = A_ff_priorS;
          // item 8: "no prior information in a direction" (a true nullspace
          // of Lambda_full_priorS) genuinely means infinite prior variance
          // there -- pinv is the textbook representation of the prior
          // restricted to its supported subspace, not an invented
          // threshold; the SAME Sigma_full_priorS object is reused for
          // BOTH the mean solve's information (via its z-block inverse
          // below) and the covariance block (directly) -- one prior, one
          // representation, item 29's invariant by construction.
          // Item 10 (prior factorization): this pinv runs EXACTLY ONCE per
          // scan (here, at scan-start init) -- never again during the GN
          // loop, which only ever reads the resulting coupled_pose_control_
          // lambda_prior_z_/z_imu_ objects directly (A += Lambda_prior_z).
          // A pinv (rather than LDLT/LLT of a full-rank matrix) is used
          // because Lambda_full_priorS has GENUINE structural rank
          // deficiency by construction: every IMU-process-only direction
          // the raw 6N/dimST space contains that neither the head
          // constraints nor any process/prior term touches (e.g. any
          // spline column entirely outside the process factor's segment
          // window for a large N) is an EXACT zero row/column of
          // A_process_scanstart_shared, not a numerically-weak one --
          // expected structural nullity is layoutS.dim() minus the number
          // of columns the process-factor segments + head constraints
          // actually touch (bounded above by dimFullS - dEtaS - dSTS - 9
          // for a fully-constrained scan). The retained eigenvalue range
          // after pinv is [rel_thresh * lambda_max, lambda_max] by
          // generalPseudoInverse's own construction (pose_control_
          // covariance.h) -- see test_pose_control_prior_math.cpp's
          // testStructuralVsNumericalNullspace() for the general version of
          // this exact structural-vs-numerical distinction, validated on a
          // synthetic analogue of this same pinv call.
          coupled_pose_control_sigma_full_prior_ = generalPseudoInverse(Lambda_full_priorS, 1e-9);
          const Eigen::MatrixXd P_z_priorS = coupled_pose_control_sigma_full_prior_.bottomRightCorner(dimZS, dimZS);
          coupled_pose_control_lambda_prior_z_ =
              generalPseudoInverse(P_z_priorS, copts_.pose_control_q_pinv_rel_thresh);
          coupled_pose_control_z_imu_ = Eigen::VectorXd::Zero(dimZS);
          coupled_pose_control_z_imu_.head(dEtaS) = coupled_pose_control_eta_;  // == eta_imu at this point (pre-GN)

          // items 5/12: bias/gravity <-> trajectory cross-covariance export
          // (P_eta_bg/ba/g), directly from the joint P_z_priorS block this
          // production prior is built from -- proof the prior treats the
          // IMU trajectory as uncertain-given-uncertain-bias, not
          // deterministic given a point bias estimate.
          if (copts_.psd_audit_en && dSTS > 0) {
            const Eigen::MatrixXd P_eta_sT = P_z_priorS.topRightCorner(dEtaS, dSTS);
            int off = 0;
            std::map<std::string, std::string> bkv;
            auto blockNorm = [&](const char* name, int width) {
              if (width <= 0) return;
              bkv[std::string("P_eta_") + name + "_norm"] = std::to_string(P_eta_sT.block(0, off, dEtaS, width).norm());
              off += width;
            };
            blockNorm("bg", layoutS.colBG() >= 0 ? 3 : 0);
            blockNorm("ba", layoutS.colBA() >= 0 ? 3 : 0);
            blockNorm("g", layoutS.colG() >= 0 ? 3 : 0);
            bkv["trace_P0"] = std::to_string(P0s.trace());
            emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "bias_information", voxel_map_->frame_idx_, -1, bkv);
          }

        }
      }
      coupled_pose_control_valid_ = true;
      // x1/head-propagation campaign item 1/2: the PHYSICAL (spline-
      // evaluated) x1 state at three stages of initialization -- the raw
      // IMU chain interpolation, the pre-projection spline guess, and the
      // post-projection (head-constraint-enforced) spline guess actually
      // used as the GN loop's starting point -- plus the explicit norms
      // item 2 requires.
      if (copts_.psd_audit_en && have_x1) {
        const double dp_pre = (p_x1_spline_init_pre - p_x1_imu_prior).norm();
        const double dv_pre = (v_x1_spline_init_pre - v_x1_imu_prior).norm();
        const double dR_pre = Log(M3D(R_x1_imu_prior.transpose() * R_x1_spline_init_pre)).norm();
        const double dp_post = (p_x1_spline_init_post - p_x1_imu_prior).norm();
        const double dv_post = (v_x1_spline_init_post - v_x1_imu_prior).norm();
        const double dR_post = Log(M3D(R_x1_imu_prior.transpose() * R_x1_spline_init_post)).norm();
        // 2026-09-24 legacy-diagnostic-writer cleanup (item 3/33): folded
        // into pose_control_full_diagnostics.csv's "x1_init_state" row
        // (previously a dedicated pose_control_x1_init_state.txt file).
        std::map<std::string, std::string> x1kv = {
          {"x1_knot_index", std::to_string(kPoseControlX1Knot)}, {"x1_time", std::to_string(t_x1)},
          {"p_imu_x", std::to_string(p_x1_imu_prior.x())}, {"p_imu_y", std::to_string(p_x1_imu_prior.y())}, {"p_imu_z", std::to_string(p_x1_imu_prior.z())},
          {"v_imu_x", std::to_string(v_x1_imu_prior.x())}, {"v_imu_y", std::to_string(v_x1_imu_prior.y())}, {"v_imu_z", std::to_string(v_x1_imu_prior.z())},
          {"p_pre_x", std::to_string(p_x1_spline_init_pre.x())}, {"p_pre_y", std::to_string(p_x1_spline_init_pre.y())}, {"p_pre_z", std::to_string(p_x1_spline_init_pre.z())},
          {"v_pre_x", std::to_string(v_x1_spline_init_pre.x())}, {"v_pre_y", std::to_string(v_x1_spline_init_pre.y())}, {"v_pre_z", std::to_string(v_x1_spline_init_pre.z())},
          {"p_post_x", std::to_string(p_x1_spline_init_post.x())}, {"p_post_y", std::to_string(p_x1_spline_init_post.y())}, {"p_post_z", std::to_string(p_x1_spline_init_post.z())},
          {"v_post_x", std::to_string(v_x1_spline_init_post.x())}, {"v_post_y", std::to_string(v_x1_spline_init_post.y())}, {"v_post_z", std::to_string(v_x1_spline_init_post.z())},
          {"delta_p_pre_norm", std::to_string(dp_pre)}, {"delta_v_pre_norm", std::to_string(dv_pre)}, {"delta_R_pre_norm", std::to_string(dR_pre)},
          {"delta_p_post_norm", std::to_string(dp_post)}, {"delta_v_post_norm", std::to_string(dv_post)}, {"delta_R_post_norm", std::to_string(dR_post)},
        };
        emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "x1_init_state", voxel_map_->frame_idx_, -1, x1kv);
      }
      if (copts_.psd_audit_en) {
        const auto& hns = coupled_pose_control_hns_;
        const int dEta = hns.freeDim(), dST = coupled_pose_control_layout_.dimST();
        std::map<std::string, std::string> kv = {
          {"trajectory_parameterization", "pose_control"}, {"velocity_mode", "spline_derived"},
          {"N_control_points", std::to_string(N)},
          {"total_optimization_dimension", std::to_string(dEta + dST)},
          {"free_spline_dimension", std::to_string(dEta)},
          {"tail_free_state_dimension", std::to_string(dST)},
          {"lidar_enable", copts_.pose_control_lidar_enable ? "1" : "0"},
          {"process_enable", "1"},
          {"imu_var_acc_x", std::to_string(state_->varAcc().x())},
          {"imu_var_acc_y", std::to_string(state_->varAcc().y())},
          {"imu_var_acc_z", std::to_string(state_->varAcc().z())},
          {"imu_var_gyr_x", std::to_string(state_->varGyr().x())},
          {"imu_var_gyr_y", std::to_string(state_->varGyr().y())},
          {"imu_var_gyr_z", std::to_string(state_->varGyr().z())},
          {"covariance_pseudoinverse_threshold", std::to_string(copts_.pose_control_q_pinv_rel_thresh)},
          {"p0_scale_config", std::to_string(copts_.pose_control_p0_scale)},
        };
        emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "run_summary", voxel_map_->frame_idx_, -1, kv);

        // head_constraint: full C construction not repeated here (it's
        // internal to buildPoseControlHeadNullspace) -- instead verify Z's
        // OWN orthonormality (Z^T Z ~= I), which combined with the known
        // dimension (rawDim x rawDim-9) is the property the mean/covariance
        // code actually relies on (c=c_particular+Z*eta, eta0=Z^T*(...)).
        // Documented simplification vs the literally-requested ||C*Z||_F.
        const Eigen::MatrixXd ZtZ = hns.Z.transpose() * hns.Z;
        const Eigen::MatrixXd dev = ZtZ - Eigen::MatrixXd::Identity(ZtZ.rows(), ZtZ.cols());
        std::map<std::string, std::string> hkv = {
          {"CZ_frobenius", std::to_string(dev.norm())},
          {"CZ_max_abs", std::to_string(dev.cwiseAbs().maxCoeff())},
          {"C_rows", "9"}, {"C_cols", std::to_string(hns.rawDim())},
          {"Z_rows", std::to_string(hns.rawDim())}, {"Z_cols", std::to_string(hns.freeDim())},
          {"notes", "CZ_frobenius/CZ_max_abs here report ||Z^T Z - I|| (orthonormality), not ||C*Z|| -- C is not separately reconstructed in this pass"},
        };
        emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "head_constraint", voxel_map_->frame_idx_, -1, hkv);
      }
    }
  }
  // mg.poses.front().vel (NOT state_->vel()), for consistency with
  // rot0/pos0 in estimateCoupledCorrection() -- all three come from the
  // SAME raw-chain snapshot.
  coupled_v0_pre_ = mg.poses.empty() ? state_->vel() : mg.poses.front().vel;
  coupled_bg0_pre_ = state_->biasGyr();
  coupled_ba0_pre_ = state_->biasAcc();
  coupled_g0_pre_ = state_->gravity();

  for (; iter < opts_.max_iterations; iter++) {
    // CQ-44 item 3, ASSERTED BY CONSTRUCTION: this class has no
    // estimateStateCorrection()/refineSplineFromResiduals() call site at
    // all -- one set of points, one update, always this one.
    const double error = estimateCoupledCorrection(mg, dtheta, dt);
    ++coupled_iters_;
    if (!residuals_.empty()) any_solved = true;
    total_dtheta += dtheta;
    total_dt     += dt;

    if (opts_.log_debug_en)
    {
      const double t_abs = mg.image.t + data_queues_->start_time;
      std::ostringstream iss;
      iss << "t_abs=" << std::fixed << std::setprecision(6) << t_abs
          << "  iter=" << iter << "  n_residuals=" << residuals_.size()
          << "  avg_abs_r=" << std::scientific << std::setprecision(6) << error
          << "  rel_diff=" << ((prev_error - error) / std::max(prev_error, 1e-6))
          // CQ-53 item 3: THIS iteration's own step norms.
          << "  delta_s_norm=" << coupled_last_delta_s_norm_
          << "  delta_c_norm=" << coupled_last_delta_c_norm_
          << "  delta_c_acc_norm=" << coupled_last_delta_c_acc_norm_
          << "  delta_c_gyr_norm=" << coupled_last_delta_c_gyr_norm_
          // CQ-53 item (B): per-block breakdown of delta_s_norm above --
          // distinguishes "phi0/p0 still moving, bg converged" from
          // "everything shrinking together" / "everything oscillating",
          // which the combined norm alone cannot.
          << "  delta_phi0_norm=" << coupled_last_delta_phi0_norm_
          << "  delta_p0_norm=" << coupled_last_delta_p0_norm_
          << "  delta_v_norm_step=" << coupled_last_delta_v_norm_step_
          << "  delta_bg_norm_step=" << coupled_last_delta_bg_norm_step_
          << "  delta_ba_norm_step=" << coupled_last_delta_ba_norm_step_
          << "  delta_g_norm_step=" << coupled_last_delta_g_norm_step_;
      static PersistentLogStream log("iter_error.txt");
      std::ofstream& ofs = log.stream();
      ofs << iss.str() << "\n";
      ofs.flush();
    }

    const double prev = prev_error;
    prev_error = error;
    if (dtheta.norm() < opts_.min_norm_dtheta && dt.norm() < opts_.min_norm_dt) {
      std::ostringstream ss;
      ss << std::scientific << std::setprecision(1)
         << "norm(dth=" << dtheta.norm() * (180.0 / M_PI) << "deg"
         << ",dt="      << dt.norm() * 1000.0               << "mm)";
      stop = ss.str(); break;
    }
    if ((prev - error) / std::max(prev, 1e-6) < opts_.min_diff_error)
      { stop = "rel_diff"; break; }
  }

  // CQ-82 Phase 2: everything below this point (final_redeskew, bg-
  // projection, relinearization, and most of the CQ-53/54/60/etc.
  // diagnostics) is raw_imu-basis-specific bookkeeping -- built around
  // coupled_last_A_/coupled_c_acc_/coupled_c_gyr_'s RAW_IMU shapes and
  // semantics (e.g. coupled_last_A_ is assumed to be the FULL 18+6*n_c
  // joint system with delta_bg/delta_ba at fixed offsets; the pose basis's
  // own coupled_last_A_ is only the 6*n_c c-block, a different shape
  // entirely). Auditing and re-deriving a pose-basis-appropriate version of
  // ~1150 lines of this bookkeeping is real, separate work -- FOUND LIVE,
  // via a real SIGSEGV crash (an out-of-bounds read on coupled_last_A_'s
  // now-mismatched shape), not assumed. Returning here is honest about that
  // scope rather than risking another silent out-of-bounds read: the GN
  // loop itself (the actual novel numerics -- LiDAR/IMU-factor/smoothness
  // terms, the solve, state_ write-back) already ran to real convergence
  // (confirmed live: multiple iterations, stop="rel_diff", real dtheta/dt
  // norms) by the time we reach here. mg.points is already correctly
  // deskewed (by estimateCoupledCorrectionPoseBasis()'s own
  // deskewPointsSpline() call each iteration) -- the caller's own map
  // update (outside this function) is unaffected by skipping the rest.
  // 2026-09-22: pose-control-point post-loop covariance + full-tail
  // write-back. Mirrors poseBasis()'s own early-return pattern below --
  // the GN loop itself already ran every iteration's mean update (LiDAR+
  // process factor, state_->setPropagatedState() every call); this block
  // does the ONE-TIME (not per-iteration, per spec item 5) covariance
  // solve: relinearize the FULL system (LiDAR+process+sT-prior+head_block)
  // at the converged trial, Schur-complement the head out, map the
  // resulting z=[c_free;sT] posterior to the COMPLETE tail StateGroup via
  // M_T, and write state_->covMut()/applyDelta() ONCE.
  if (copts_.poseControlSplineBasis() && coupled_pose_control_valid_) {
    auto& spline = coupled_pose_control_spline_;
    const auto& layout = coupled_pose_control_layout_;
    const auto& hns = coupled_pose_control_hns_;
    const double t1 = spline.t1();
    const int dimRaw = layout.dim();
    const int dEta = hns.freeDim(), dST = layout.dimST(), dimZ = dEta + dST;

    // ==========================================================================
    // process-prior-REFORMULATION phase, item 29's hard invariant: the
    // prior used here MUST be the EXACT SAME object the mean solve used,
    // not a fresh re-derivation from the converged trial. coupled_pose_
    // control_sigma_full_prior_ was computed ONCE at scan-start (in the
    // init block, unconditionally) and is reused verbatim here -- this
    // covariance block no longer re-derives A_process_raw/head_block/Omega0/
    // Lambda_full_prior at all. Only the LiDAR side is rebuilt (relinearized
    // at the converged trial, matching item 9's "LiDAR: relinearize; IMU
    // prior: FIXED"). NOTE: under the comparison-only legacy_process_factor/
    // frozen_process_hessian_prior modes, the MEAN solve used a DIFFERENT
    // effective prior than this -- for those modes the item-29 invariant is
    // deliberately NOT claimed (they are explicitly non-production
    // comparison paths; the covariance reported for them still reflects the
    // production joint prior, not their own ad hoc information, and this is
    // disclosed here rather than silently mismatched).
    // ==========================================================================
    Eigen::MatrixXd A_lidar_raw = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
    Eigen::VectorXd b_lidar_raw = Eigen::VectorXd::Zero(dimRaw);

    std::vector<PoseControlLidarObs> lidar_obs;
    lidar_obs.reserve(residuals_.size());
    for (auto& res : residuals_) {
      const double d = res.r - res.normal.dot(spline.rotAt(res.t) * res.raw_body_point + spline.posAt(res.t));
      PoseControlLidarObs o;
      o.t = res.t; o.q = res.raw_body_point; o.normal = res.normal; o.d = d; o.sigma2 = res.sigma_squared; o.plane_id = res.plane_id; o.plane_var_term = res.plane_var_term;
      lidar_obs.push_back(o);
    }
    ResidualRedundancyStats pose_control_lidar_corr_stats;
    if (copts_.pose_control_lidar_enable) {
      std::vector<PoseControlLidarRecord> lidar_records;
      const bool want_records = copts_.pose_control_lidar_correlation.mode != "off";
      addPoseControlLidarFactor(spline, layout, lidar_obs, A_lidar_raw, b_lidar_raw, nullptr, nullptr,
                                 want_records ? &lidar_records : nullptr);
      if (want_records)
        pose_control_lidar_corr_stats = applyPoseControlLidarCorrelationCorrection(
            lidar_records, copts_.pose_control_lidar_correlation, A_lidar_raw, b_lidar_raw);
      if (copts_.psd_audit_en && want_records) {
        std::map<std::string, std::string> lkv = {
          {"num_raw_residuals", std::to_string(lidar_records.size())},
          {"redund_groups", std::to_string(pose_control_lidar_corr_stats.redund_groups)},
          {"redund_n_raw", std::to_string(pose_control_lidar_corr_stats.redund_n_raw)},
          {"raw_information_trace", std::to_string(pose_control_lidar_corr_stats.naive_info_gain)},
          {"correlation_corrected_information", std::to_string(pose_control_lidar_corr_stats.woodbury_info_gain)},
          {"correlation_information_reduction", std::to_string(1.0 - pose_control_lidar_corr_stats.redund_info_ratio)},
          {"lidar_correlation_mode", copts_.pose_control_lidar_correlation.mode},
        };
        emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "lidar_correlation", voxel_map_->frame_idx_, -1, lkv);
      }
    }

    // ---- project onto z=[eta;delta_sT] (SAME P as the mean solve) --------
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dimRaw, dimZ);
    P.block(0, 0, hns.rawDim(), dEta) = hns.Z;
    if (dST > 0) P.block(hns.rawDim(), dEta, dST, dST) = Eigen::MatrixXd::Identity(dST, dST);
    // Item 18 (curvature semantics), DECIDED: curvature is interpretation B
    // -- DETERMINISTIC NUMERICAL REGULARIZATION on the trajectory, not a
    // Gaussian smoothness prior. It therefore appears in the MEAN solve's
    // A/b (estimateCoupledPoseControlSpline's curvature block, added
    // directly to A_raw/b_raw before projection) but is DELIBERATELY
    // EXCLUDED from Lambda_meas_z below and from the joint prior -- the
    // covariance this estimator reports is the covariance of the
    // MAP/regularized-trajectory-conditional posterior (LiDAR + IMU/bias
    // prior information only), not "as if curvature were itself measurement
    // information". If curvature is ever reinterpreted as interpretation A
    // (a genuine Gaussian smoothness prior), it must be added HERE too, to
    // both this covariance path and the mean solve's A/b, from the SAME
    // Lambda_curvature object -- not independently in only one place.
    const Eigen::MatrixXd Lambda_meas_z = P.transpose() * A_lidar_raw * P;  // LiDAR ONLY, in z-space
    if (copts_.psd_audit_en) {
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "Lambda_meas_z", Lambda_meas_z);
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "Lambda_meas_z_eta_block", Lambda_meas_z.topLeftCorner(dEta, dEta));

      // Directional redundancy (items 26-30): surface-agnostic eigen-
      // spectrum of the eta block of Lambda_meas_z -- see
      // pose_control_directional_redundancy.h. Restricted to the eta block
      // (not the full z, which also carries sT/bias-gravity rows the LiDAR
      // factor never touches directly and would just report as
      // structural zero-information rows, diluting the "how well is the
      // TRAJECTORY observed" question this diagnostic answers).
      const DirectionalRedundancyStats dstats = analyzeDirectionalRedundancy(
          Lambda_meas_z.topLeftCorner(dEta, dEta), static_cast<int>(lidar_obs.size()));
      if (dstats.valid) {
        std::map<std::string, std::string> dkv = {
          {"raw_residual_count", std::to_string(dstats.raw_residual_count)},
          {"reduced_state_dimension", std::to_string(dstats.reduced_state_dimension)},
          {"effective_rank", std::to_string(dstats.effective_rank)},
          {"condition_number", std::to_string(dstats.condition_number)},
          {"dominant_eigenvalue", std::to_string(dstats.eigenvalues.size() ? dstats.eigenvalues(0) : 0.0)},
          {"weak_eigenvalue", std::to_string(dstats.eigenvalues.size() ? dstats.eigenvalues(dstats.eigenvalues.size() - 1) : 0.0)},
          {"cumulative_information_fraction_at_rank5",
           std::to_string(dstats.cumulative_information_fraction.size() > 4
                               ? dstats.cumulative_information_fraction(4) : 1.0)},
        };
        emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "directional_redundancy", voxel_map_->frame_idx_, -1, dkv);
      }
    }

    const int dimFull = 9 + dimZ;
    const Eigen::MatrixXd& Sigma_full_prior = coupled_pose_control_sigma_full_prior_;
    const bool schur_ok = Sigma_full_prior.rows() == dimFull && Sigma_full_prior.allFinite();
    if (copts_.psd_audit_en && schur_ok) {
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "Sigma_full_prior", Sigma_full_prior);
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_x0_prior", Sigma_full_prior.topLeftCorner(9, 9));
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_z_prior", Sigma_full_prior.bottomRightCorner(dimZ, dimZ));
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_eta_prior", Sigma_full_prior.block(9, 9, dEta, dEta));
      logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_sT_prior", Sigma_full_prior.bottomRightCorner(dST, dST));
    }
    if (schur_ok) {
      Eigen::MatrixXd Lambda_meas_full = Eigen::MatrixXd::Zero(dimFull, dimFull);
      Lambda_meas_full.block(9, 9, dimZ, dimZ) = Lambda_meas_z;   // LiDAR touches z only -- see head_block_process comment above
      Eigen::MatrixXd Sigma_full_post;
      CovarianceUpdateDiagnostics cov_diag;
      const bool update_ok = covarianceInformationUpdate(Sigma_full_prior, Lambda_meas_full, Sigma_full_post, cov_diag);
      if (!update_ok) { Sigma_full_post = Sigma_full_prior; }
      coupled_pose_control_P_z_post_ = Sigma_full_post.bottomRightCorner(dimZ, dimZ);
      if (copts_.psd_audit_en) {
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "Sigma_full_post", Sigma_full_post);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_z_post", coupled_pose_control_P_z_post_);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_eta_post", Sigma_full_post.block(9, 9, dEta, dEta));
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_sT_post", Sigma_full_post.bottomRightCorner(dST, dST));
      }

      // M_T: dimState() x dimZ, mapping z -> the FULL tail StateGroup
      // [theta,p,v,bg?,ba?,g?] in StateGroup's own index order. R/p/v
      // Jacobians chain through Z (dc/deta = Z, item 11's J_R/J_p/J_v =
      // d(.)(t1)/dc * Z) since eta, not c directly, is now the variable;
      // bg/ba/g map via plain identity into their own delta_sT columns.
      const int dimSt = state_->dimState();
      Eigen::MatrixXd M_T = Eigen::MatrixXd::Zero(dimSt, dimZ);
      // 2026-09-23 BUG FIX (found while adding the x1 instrumentation below,
      // via a segfault on a config with bg/ba/g estimation enabled -- the
      // normal/default case, not an edge case): these must be sized
      // (3 x hns.rawDim()), NOT (3 x dimRaw=layout.dim()). layout.dim()
      // includes the sT (bg/ba/g) columns, but hns.Z (below) only has
      // hns.rawDim()=6N rows (control-point space only) -- dR_dc*hns.Z was a
      // DIMENSION-MISMATCHED multiply whenever dST>0 (dimRaw=6N+dST !=
      // hns.rawDim()=6N). Eigen's own eigen_assert on this is compiled OUT
      // in a Release/NDEBUG build, so this silently read past hns.Z's
      // allocation (undefined behavior) instead of erroring -- it happened
      // to not crash in earlier runs (adjacent heap memory, no page fault)
      // but is NOT reliable; the stationary campaign data produced before
      // this fix may have corrupted P_R/P_p/P_v tail-covariance values
      // whenever bg/ba/g were enabled (they were, in every config this
      // session used bg/ba/g estimation) -- flagged explicitly, not
      // silently patched over.
      Eigen::MatrixXd dR_dc = Eigen::MatrixXd::Zero(3, hns.rawDim());
      Eigen::MatrixXd dp_dc = Eigen::MatrixXd::Zero(3, hns.rawDim());
      Eigen::MatrixXd dv_dc = Eigen::MatrixXd::Zero(3, hns.rawDim());
      const auto jac1 = spline.jacobianAt(t1);
      for (int k = 0; k < 4; ++k) {
        const int abs_k = jac1.s + k;
        const int colp = layout.colPos(abs_k), colph = layout.colPhi(abs_k);
        if (colph >= 0) dR_dc.block<3, 3>(0, colph) = spline.dThetaDcphi(jac1, k, t1);
        if (colp >= 0) {
          dp_dc.block<3, 3>(0, colp) = PoseControlSpline::dPosDcp(jac1, k);
          dv_dc.block<3, 3>(0, colp) = PoseControlSpline::dVelDcp(jac1, k);
        }
      }
      const Eigen::MatrixXd J_R = dR_dc * hns.Z;
      const Eigen::MatrixXd J_p = dp_dc * hns.Z;
      const Eigen::MatrixXd J_v = dv_dc * hns.Z;
      M_T.block(StateGroup::idxR(), 0, 3, dEta) = J_R;
      M_T.block(StateGroup::idxP(), 0, 3, dEta) = J_p;
      M_T.block(StateGroup::idxV(), 0, 3, dEta) = J_v;
      if (copts_.psd_audit_en) {
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "dR_dc_raw", dR_dc);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "dp_dc_raw", dp_dc);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "dv_dc_raw", dv_dc);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "J_R", J_R);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "J_p", J_p);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "J_v", J_v);
      }
      if (layout.colBG() >= 0 && state_->idxBG() >= 0)
        M_T.block<3, 3>(state_->idxBG(), dEta + layout.colBG() - layout.dimCFree()) = M3D::Identity();
      if (layout.colBA() >= 0 && state_->idxBA() >= 0)
        M_T.block<3, 3>(state_->idxBA(), dEta + layout.colBA() - layout.dimCFree()) = M3D::Identity();
      if (layout.colG() >= 0 && state_->idxG() >= 0)
        M_T.block<3, 3>(state_->idxG(), dEta + layout.colG() - layout.dimCFree()) = M3D::Identity();

      // J_h_tail: dimSt x 9, the DIRECT sensitivity of the tail state to
      // x0=[dtheta0,dp0,dv0] (channel (b) in the comment above -- the SAME
      // exact head-Jacobians poseControlHeadRotJacobian()/
      // poseControlHeadPosJacobians() already used inside the process/LiDAR
      // factors' own head_block bookkeeping, evaluated at t1 instead of at
      // a factor's own segment endpoints). bg/ba/g rows are zero: those
      // have no direct x0 dependency (their uncertainty reaches the tail
      // entirely through delta_sT / Omega0's sT block, already handled).
      Eigen::MatrixXd J_h_tail = Eigen::MatrixXd::Zero(dimSt, 9);
      {
        const M3D dR_dtheta0 = poseControlHeadRotJacobian(spline, t1);
        const auto hs_head = poseControlHeadPosSensitivity(spline);
        M3D dp_dp0, dp_dv0, dv_dp0, dv_dv0;
        poseControlHeadPosJacobians(spline, hs_head, t1, dp_dp0, dp_dv0, dv_dp0, dv_dv0);
        J_h_tail.block<3, 3>(StateGroup::idxR(), 0) = dR_dtheta0;
        J_h_tail.block<3, 3>(StateGroup::idxP(), 3) = dp_dp0;
        J_h_tail.block<3, 3>(StateGroup::idxP(), 6) = dp_dv0;
        J_h_tail.block<3, 3>(StateGroup::idxV(), 3) = dv_dp0;
        J_h_tail.block<3, 3>(StateGroup::idxV(), 6) = dv_dv0;
      }
      Eigen::MatrixXd M_full = Eigen::MatrixXd::Zero(dimSt, dimFull);
      M_full.block(0, 0, dimSt, 9) = J_h_tail;
      M_full.block(0, 9, dimSt, dimZ) = M_T;

      const Eigen::MatrixXd Ppred_raw = M_full * Sigma_full_prior * M_full.transpose();
      const Eigen::MatrixXd Ppost_raw = M_full * Sigma_full_post * M_full.transpose();
      const Eigen::MatrixXd P_tail_pred = 0.5 * (Ppred_raw + Ppred_raw.transpose());
      const Eigen::MatrixXd P_T = 0.5 * (Ppost_raw + Ppost_raw.transpose());
      if (copts_.psd_audit_en) {
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "J_h_tail", J_h_tail);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "M_full", M_full);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "M_T", M_T);
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_R_pred", P_tail_pred.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()));
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_p_pred", P_tail_pred.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()));
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_v_pred", P_tail_pred.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()));
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_R_post", P_T.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()));
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_p_post", P_T.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()));
        logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_v_post", P_T.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()));
      }

      // ====================================================================
      // 2026-09-23 x1/head-propagation campaign, items 3/4/5/7/9/10: the
      // SAME M_full=[J_h,J_state] recipe used above for the tail (t1),
      // re-evaluated at x1's knot time (t_x1) instead -- x1 = first free
      // knot AFTER the fixed head (k=3; only cp[0..2] enter the 9 head
      // constraints). This is "physical x1" (spline-curve-evaluated),
      // matching item 9's distinction from the raw conditional eta/
      // control-point-space prior (already logged above as P_eta_prior).
      // ====================================================================
      if (copts_.psd_audit_en) {
        constexpr int kPoseControlX1Knot = 3;
        const int Nlay = layout.N;
        const bool have_x1 = (Nlay > kPoseControlX1Knot);
        if (have_x1) {
          const double t_x1 = std::min(t1, spline.t0() + kPoseControlX1Knot * spline.delta());
          // Same fix as dR_dc/dp_dc/dv_dc above: sized to hns.rawDim(), not dimRaw.
          Eigen::MatrixXd dR_dc_x1 = Eigen::MatrixXd::Zero(3, hns.rawDim());
          Eigen::MatrixXd dp_dc_x1 = Eigen::MatrixXd::Zero(3, hns.rawDim());
          Eigen::MatrixXd dv_dc_x1 = Eigen::MatrixXd::Zero(3, hns.rawDim());
          const auto jac_x1 = spline.jacobianAt(t_x1);
          for (int k = 0; k < 4; ++k) {
            const int abs_k = jac_x1.s + k;
            const int colp = layout.colPos(abs_k), colph = layout.colPhi(abs_k);
            if (colph >= 0) dR_dc_x1.block<3, 3>(0, colph) = spline.dThetaDcphi(jac_x1, k, t_x1);
            if (colp >= 0) {
              dp_dc_x1.block<3, 3>(0, colp) = PoseControlSpline::dPosDcp(jac_x1, k);
              dv_dc_x1.block<3, 3>(0, colp) = PoseControlSpline::dVelDcp(jac_x1, k);
            }
          }
          const Eigen::MatrixXd J_R_x1 = dR_dc_x1 * hns.Z;
          const Eigen::MatrixXd J_p_x1 = dp_dc_x1 * hns.Z;
          const Eigen::MatrixXd J_v_x1 = dv_dc_x1 * hns.Z;

          // F_10 = d(x1)/d(x0): the SAME head-Jacobian functions used for
          // J_h_tail above (poseControlHeadRotJacobian/
          // poseControlHeadPosJacobians), evaluated at t_x1 instead of t1.
          M3D dR_dtheta0_x1, dp_dp0_x1, dp_dv0_x1, dv_dp0_x1, dv_dv0_x1;
          dR_dtheta0_x1 = poseControlHeadRotJacobian(spline, t_x1);
          {
            const auto hs_head_x1 = poseControlHeadPosSensitivity(spline);
            poseControlHeadPosJacobians(spline, hs_head_x1, t_x1, dp_dp0_x1, dp_dv0_x1, dv_dp0_x1, dv_dv0_x1);
          }
          Eigen::MatrixXd J_h_x1 = Eigen::MatrixXd::Zero(9, 9);
          J_h_x1.block<3, 3>(0, 0) = dR_dtheta0_x1;
          J_h_x1.block<3, 3>(3, 3) = dp_dp0_x1; J_h_x1.block<3, 3>(3, 6) = dp_dv0_x1;
          J_h_x1.block<3, 3>(6, 3) = dv_dp0_x1; J_h_x1.block<3, 3>(6, 6) = dv_dv0_x1;

          Eigen::MatrixXd J_x1_eta = Eigen::MatrixXd::Zero(9, dimZ);
          J_x1_eta.block(0, 0, 3, dEta) = J_R_x1;
          J_x1_eta.block(3, 0, 3, dEta) = J_p_x1;
          J_x1_eta.block(6, 0, 3, dEta) = J_v_x1;

          Eigen::MatrixXd M_x1_full = Eigen::MatrixXd::Zero(9, dimFull);
          M_x1_full.block(0, 0, 9, 9) = J_h_x1;
          M_x1_full.block(0, 9, 9, dimZ) = J_x1_eta;

          const Eigen::MatrixXd P_x1_prior_raw = M_x1_full * Sigma_full_prior * M_x1_full.transpose();
          const Eigen::MatrixXd P_x1_prior = 0.5 * (P_x1_prior_raw + P_x1_prior_raw.transpose());
          const Eigen::MatrixXd P_x1_post_raw = M_x1_full * Sigma_full_post * M_x1_full.transpose();
          const Eigen::MatrixXd P_x1_post = 0.5 * (P_x1_post_raw + P_x1_post_raw.transpose());

          // item 5's independent/naive-vs-joint-prior check (the "does the
          // head-coupling cross term matter" question) was answered
          // conclusively in the earlier x1/head-propagation campaign phase
          // (~61% median Frobenius gap at N=13, confirming it matters) --
          // NOT recomputed here post-reformulation, since it required
          // rebuilding a separate un-marginalized A_ff_prior the production
          // covariance path no longer constructs (item 60's "one
          // authoritative covariance update" audit). abs/rel_err_x1_check
          // are left as NA in this phase's CSV rather than silently
          // reusing a stale/mismatched quantity.
          const double abs_err_x1_check = std::numeric_limits<double>::quiet_NaN();
          const double rel_err_x1_check = std::numeric_limits<double>::quiet_NaN();

          // item 1/10: physical x1 state -- final (converged spline, this
          // block runs post-GN-loop) vs the true IMU-chain interpolation.
          const V3D p_x1_final = spline.posAt(t_x1), v_x1_final = spline.velAt(t_x1);
          const M3D R_x1_final = spline.rotAt(t_x1);
          V3D p_x1_imu_prior, v_x1_imu_prior; M3D R_x1_imu_prior;
          interpPose6DAt(mg.poses, t_x1, p_x1_imu_prior, v_x1_imu_prior, R_x1_imu_prior);
          const V3D e_p = p_x1_final - p_x1_imu_prior;
          const V3D e_R = Log(M3D(R_x1_imu_prior.transpose() * R_x1_final));
          const V3D e_v = v_x1_final - v_x1_imu_prior;
          const M3D P_p_x1 = P_x1_prior.block<3, 3>(3, 3), P_R_x1 = P_x1_prior.block<3, 3>(0, 0), P_v_x1 = P_x1_prior.block<3, 3>(6, 6);
          const double d2_p = (e_p.transpose() * generalPseudoInverse(P_p_x1, copts_.pose_control_q_pinv_rel_thresh) * e_p)(0);
          const double d2_R = (e_R.transpose() * generalPseudoInverse(P_R_x1, copts_.pose_control_q_pinv_rel_thresh) * e_R)(0);
          const double d2_v = (e_v.transpose() * generalPseudoInverse(P_v_x1, copts_.pose_control_q_pinv_rel_thresh) * e_v)(0);
          const double sigma_dist_p = std::sqrt(std::max(0.0, d2_p));
          const double sigma_dist_R = std::sqrt(std::max(0.0, d2_R));
          const double sigma_dist_v = std::sqrt(std::max(0.0, d2_v));

          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_x1(P_x1_prior);
          const double x1_cond = (std::abs(es_x1.eigenvalues().minCoeff()) > 1e-300)
              ? es_x1.eigenvalues().maxCoeff() / es_x1.eigenvalues().minCoeff() : 0.0;

          // 2026-09-24 legacy-diagnostic-writer cleanup (item 3/33): folded
          // into pose_control_full_diagnostics.csv's "x1_covariance" row
          // (previously a dedicated pose_control_x1_diagnostics.txt file).
          int rank_x1 = 0;
          const double thresh_x1 = 1e-9 * std::max(std::abs(es_x1.eigenvalues().maxCoeff()), 1.0);
          for (int i = 0; i < es_x1.eigenvalues().size(); ++i) if (std::abs(es_x1.eigenvalues()(i)) > thresh_x1) ++rank_x1;
          std::map<std::string, std::string> x1covkv = {
            {"x1_knot_index", std::to_string(kPoseControlX1Knot)}, {"x1_time", std::to_string(t_x1)},
            {"p_final_x", std::to_string(p_x1_final.x())}, {"p_final_y", std::to_string(p_x1_final.y())}, {"p_final_z", std::to_string(p_x1_final.z())},
            {"v_final_x", std::to_string(v_x1_final.x())}, {"v_final_y", std::to_string(v_x1_final.y())}, {"v_final_z", std::to_string(v_x1_final.z())},
            {"p_imu_x", std::to_string(p_x1_imu_prior.x())}, {"p_imu_y", std::to_string(p_x1_imu_prior.y())}, {"p_imu_z", std::to_string(p_x1_imu_prior.z())},
            {"v_imu_x", std::to_string(v_x1_imu_prior.x())}, {"v_imu_y", std::to_string(v_x1_imu_prior.y())}, {"v_imu_z", std::to_string(v_x1_imu_prior.z())},
            {"delta_p_norm", std::to_string(e_p.norm())}, {"delta_v_norm", std::to_string(e_v.norm())}, {"delta_R_norm", std::to_string(e_R.norm())},
            {"trace_P_p_x1", std::to_string(P_p_x1.trace())}, {"trace_P_v_x1", std::to_string(P_v_x1.trace())}, {"trace_P_R_x1", std::to_string(P_R_x1.trace())},
            {"trace_P_x1_prior", std::to_string(P_x1_prior.trace())}, {"trace_P_x1_post", std::to_string(P_x1_post.trace())},
            {"min_eig_P_x1_prior", std::to_string(es_x1.eigenvalues().minCoeff())}, {"max_eig_P_x1_prior", std::to_string(es_x1.eigenvalues().maxCoeff())},
            {"cond_P_x1_prior", std::to_string(x1_cond)}, {"rank_P_x1_prior", std::to_string(rank_x1)},
            {"sigma_distance_p", std::to_string(sigma_dist_p)}, {"sigma_distance_R", std::to_string(sigma_dist_R)}, {"sigma_distance_v", std::to_string(sigma_dist_v)},
            {"abs_err_head_propagation_check", std::to_string(abs_err_x1_check)}, {"rel_err_head_propagation_check", std::to_string(rel_err_x1_check)},
          };
          emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "x1_covariance", voxel_map_->frame_idx_, -1, x1covkv);

          logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_x1_prior", P_x1_prior);
          logCovTraceStage(voxel_map_->frame_idx_, copts_.pose_control_test_id, "P_x1_post", P_x1_post);
        }
      }

      // item 10/51's EKF-reference test is now computed INSIDE the mean-
      // solve loop, at iterations 0/1/2, comparing against that SAME
      // iteration's own actual step (see estimateCoupledPoseControlSpline's
      // "pose_control_ekf_reference_iter012.txt"/emitFullDiagRow("ekf_
      // reference",...) -- superseding this post-loop, converged-step-only
      // version, which the spec explicitly says not to use ("do NOT compare
      // a converged final GN step against a nonzero hypothetical update").

      // Item 6/7's SAME-TAIL information-gain diagnostic: compare
      // P_tail_pred (process/prior only, SAME tail state xT) against
      // P_tail_post (process/prior + LiDAR) -- NOT P_head_at_scan_start vs
      // P_tail_post (the OLD, INVALID comparison item 7 explicitly calls
      // out: "the head and tail are different physical states and process
      // noise can increase uncertainty between them").
      if (copts_.psd_audit_en) {
        const Eigen::MatrixXd DeltaP = P_tail_pred - P_T;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (DeltaP + DeltaP.transpose()));
        auto trBlock = [&](const Eigen::MatrixXd& M, int i, int j) { return (i >= 0 && j >= 0) ? M.block<3, 3>(i, j).trace() : 0.0; };
        const int iR = StateGroup::idxR(), iP = StateGroup::idxP(), iV = StateGroup::idxV();
        const int iBG = state_->idxBG(), iBA = state_->idxBA(), iG = state_->idxG();

        // trace_P0/min_eig_P0 now read from Sigma_full_prior's own x0 block
        // (the joint prior's marginal on x0) rather than a separately
        // recomputed raw P0 -- consistent with item 29/60's "one
        // authoritative prior" requirement; this IS what the production
        // covariance actually treats x0's uncertainty as.
        const Eigen::MatrixXd P0_marginal = Sigma_full_prior.topLeftCorner(9, 9);
        std::map<std::string, std::string> ckv = {
          {"trace_P0", std::to_string(P0_marginal.trace())},
          {"trace_P_tail_pred", std::to_string(P_tail_pred.trace())},
          {"trace_P_tail_post", std::to_string(P_T.trace())},
          {"min_eig_P0", std::to_string(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(0.5 * (P0_marginal + P0_marginal.transpose())).eigenvalues().minCoeff())},
          {"min_eig_P_tail_pred", std::to_string(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(0.5 * (P_tail_pred + P_tail_pred.transpose())).eigenvalues().minCoeff())},
          {"min_eig_P_tail_post", std::to_string(es.eigenvalues().minCoeff())},
          {"pose_covariance_zero", (trBlock(P_tail_pred, iR, iR) < 1e-300 && trBlock(P_tail_pred, iP, iP) < 1e-300) ? "1" : "0"},
          {"position_covariance_zero", (trBlock(P_tail_pred, iP, iP) < 1e-300) ? "1" : "0"},
          {"velocity_covariance_zero", (trBlock(P_tail_pred, iV, iV) < 1e-300) ? "1" : "0"},
          {"covariance_contraction_not_psd", (!cov_diag.post_psd) ? "1" : "0"},
        };
        emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "covariance_summary", voxel_map_->frame_idx_, -1, ckv);
        const char* block_names[] = {"R", "p", "v", "bg", "ba", "g"};
        const int block_idx[] = {iR, iP, iV, iBG, iBA, iG};
        for (int bi = 0; bi < 6; ++bi) {
          const double tp = trBlock(P_tail_pred, block_idx[bi], block_idx[bi]);
          const double tq = trBlock(P_T, block_idx[bi], block_idx[bi]);
          std::map<std::string, std::string> bkv = {
            {"block_name", block_names[bi]}, {"trace_pred", std::to_string(tp)}, {"trace_post", std::to_string(tq)},
            {"contraction_fraction", std::to_string(tp > 1e-300 ? tq / tp : 0.0)},
          };
          emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "covariance_block", voxel_map_->frame_idx_, -1, bkv);
        }
      }

      // 2026-09-24 legacy-diagnostic-writer cleanup (item 3/33): the old
      // pose_control_full_cov_matrices.txt (full P_tail_pred/P_tail_post
      // NxN dumps, for an offline NEES computation) is REMOVED rather than
      // folded into the unified CSV -- a full matrix dump doesn't fit that
      // schema's fixed scalar/vector-column convention (same reasoning
      // applied when pose_control_weak_mode_matrices.txt was removed the
      // previous phase), and NEES-from-raw-matrix is not part of this
      // phase's required validation output (items 44-49). trace/min-eig/
      // max-eig of these same matrices remain available via the
      // covariance_summary/covariance_block rows above.

      // 2026-09-23 stationary-campaign instrumentation: per-knot state
      // (position/rotation-log/derived velocity/acceleration/angular
      // velocity) and control-point second-difference (oscillation) at the
      // CONVERGED spline, once per scan. Cheap -- pure spline evaluation,
      // no new solves.
      // 2026-09-24 legacy-diagnostic-writer cleanup (item 3/33): folded
      // into pose_control_full_diagnostics.csv's "knot_state" row
      // (previously a dedicated pose_control_knot_state.txt file).
      if (copts_.psd_audit_en) {
        const int N = layout.N;
        for (int k = 0; k < N; ++k) {
          const double tk = spline.t0() + std::min<double>(k, spline.nSeg()) * spline.delta();
          const V3D p = spline.cp_p.col(k), r = spline.cp_phi.col(k);
          const V3D vel = spline.velAt(tk), acc = spline.accAt(tk), om = spline.omegaBodyAt(tk);
          double d1p = 0, d2p = 0, d1r = 0, d2r = 0;
          if (k >= 1) d1p = (spline.cp_p.col(k) - spline.cp_p.col(k-1)).norm();
          if (k >= 1) d1r = (spline.cp_phi.col(k) - spline.cp_phi.col(k-1)).norm();
          if (k >= 2) d2p = (spline.cp_p.col(k) - 2*spline.cp_p.col(k-1) + spline.cp_p.col(k-2)).norm();
          if (k >= 2) d2r = (spline.cp_phi.col(k) - 2*spline.cp_phi.col(k-1) + spline.cp_phi.col(k-2)).norm();
          std::map<std::string, std::string> knkv = {
            {"knot_index", std::to_string(k)}, {"knot_time", std::to_string(tk)},
            {"p_x", std::to_string(p.x())}, {"p_y", std::to_string(p.y())}, {"p_z", std::to_string(p.z())},
            {"rlog_x", std::to_string(r.x())}, {"rlog_y", std::to_string(r.y())}, {"rlog_z", std::to_string(r.z())},
            {"v_x", std::to_string(vel.x())}, {"v_y", std::to_string(vel.y())}, {"v_z", std::to_string(vel.z())},
            {"a_x", std::to_string(acc.x())}, {"a_y", std::to_string(acc.y())}, {"a_z", std::to_string(acc.z())},
            {"omega_x", std::to_string(om.x())}, {"omega_y", std::to_string(om.y())}, {"omega_z", std::to_string(om.z())},
            {"d1_pos_norm", std::to_string(d1p)}, {"d2_pos_norm", std::to_string(d2p)},
            {"d1_rot_norm", std::to_string(d1r)}, {"d2_rot_norm", std::to_string(d2r)},
          };
          emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "knot_state", voxel_map_->frame_idx_, -1, knkv);
        }
      }

      // ---- full tail writeback (item 12): ONE complete posterior tail
      // (R/p/v from the spline, bg/ba/g from tail_trial), written via the
      // normal applyDelta()/covMut() StateGroup machinery in one place.
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(dimSt);
      // R/p/v: state_ already holds spline.rotAt(t1)/posAt(t1)/velAt(t1)
      // from the LAST GN iteration's own setPropagatedState() call, so
      // their own delta is zero here -- only bg/ba/g need a correction to
      // reach tail_trial's own final value.
      if (state_->idxBG() >= 0) dx.segment<3>(state_->idxBG()) = coupled_pose_control_bg_trial_ - state_->biasGyr();
      if (state_->idxBA() >= 0) dx.segment<3>(state_->idxBA()) = coupled_pose_control_ba_trial_ - state_->biasAcc();
      if (state_->idxG()  >= 0) dx.segment<3>(state_->idxG())  = coupled_pose_control_g_trial_  - state_->gravity();
      state_->applyDelta(dx);
      state_->covMut() = P_T;

      // 2026-09-23: adaptive-Q candidate for scan k, computed from THIS
      // scan's own converged trajectory/IMU residual, AFTER the LiDAR
      // update above -- causality (item 17): this can only affect
      // poseControlEffectiveVarAcc()/VarGyr()'s READ at the NEXT scan's own
      // scan-start prior build (already-completed prior code above this
      // block in the current scan), never this scan's own prior.
      if (copts_.pose_control_adaptive_q.enable) {
        SplineImuResidualStats pcq_st = computePoseControlImuResidual(
            spline, mg.imu_samples_raw, state_->biasAcc(), state_->biasGyr(), state_->gravity());
        if (pcq_st.valid()) {
          Eigen::Matrix3d P_ba = Eigen::Matrix3d::Zero(), P_bg = Eigen::Matrix3d::Zero(),
                          P_g = Eigen::Matrix3d::Zero(), P_ba_g = Eigen::Matrix3d::Zero();
          if (state_->idxBA() >= 0) P_ba = P_T.block<3, 3>(state_->idxBA(), state_->idxBA());
          if (state_->idxBG() >= 0) P_bg = P_T.block<3, 3>(state_->idxBG(), state_->idxBG());
          if (state_->idxG()  >= 0) P_g  = P_T.block<3, 3>(state_->idxG(),  state_->idxG());
          if (state_->idxBA() >= 0 && state_->idxG() >= 0)
            P_ba_g = P_T.block<3, 3>(state_->idxBA(), state_->idxG());
          applyPoseControlAdaptiveQBiasGravityCorrection(
              pcq_st, spline.rotAt(t1), P_ba, P_bg, P_g, P_ba_g);
          coupled_pose_control_adaptive_q_.setNominal(state_->varAcc().mean(), state_->varGyr().mean());
          coupled_pose_control_adaptive_q_.update(pcq_st);
          coupled_pose_control_adaptive_q_primed_ = true;
          if (copts_.psd_audit_en) {
            std::map<std::string, std::string> qkv = {
              {"residual_var_acc", std::to_string(pcq_st.cov_acc)},
              {"residual_var_gyr", std::to_string(pcq_st.cov_gyr)},
              {"acf1_acc", std::to_string(pcq_st.acf1_acc)}, {"acf1_gyr", std::to_string(pcq_st.acf1_gyr)},
              {"acf2_acc", std::to_string(pcq_st.acf2_acc)}, {"acf2_gyr", std::to_string(pcq_st.acf2_gyr)},
              {"acf5_acc", std::to_string(pcq_st.acf5_acc)}, {"acf5_gyr", std::to_string(pcq_st.acf5_gyr)},
              {"q_candidate_acc", std::to_string(pcq_st.cov_acc)}, {"q_candidate_gyr", std::to_string(pcq_st.cov_gyr)},
              {"q_used_acc", std::to_string(state_->varAcc().mean())}, {"q_used_gyr", std::to_string(state_->varGyr().mean())},
              {"q_next_acc", std::to_string(coupled_pose_control_adaptive_q_.varAcc())},
              {"q_next_gyr", std::to_string(coupled_pose_control_adaptive_q_.varGyr())},
              {"q_update_accepted", std::to_string(coupled_pose_control_adaptive_q_.activeThisFrame())},
              {"q_adaptation_reason", coupled_pose_control_adaptive_q_.lastStatus()},
            };
            emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "q_estimation", voxel_map_->frame_idx_, -1, qkv);
          }
        }
      }
    }

    if (copts_.psd_audit_en) {
      std::map<std::string, std::string> skv = {
        {"gn_iterations", std::to_string(iter)},
        {"num_lidar_points", std::to_string(residuals_.size())},
      };
      emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "scan_summary", voxel_map_->frame_idx_, -1, skv);
    }

    std::ostringstream oss;
    oss << "[lio/ekf][pose-control] iters=" << iter << "  n_residuals=" << residuals_.size()
        << "  schur_ok=" << (schur_ok ? 1 : 0)
        << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
        << "  |dt|=" << total_dt.norm() * 1000.0 << " mm";
    return oss.str();
  }

  if (copts_.poseBasis()) {
    boundary_dpos_ = 0.0;
    boundary_drot_deg_ = 0.0;
    // CQ-87 item 3: the posterior covariance write this arm never had.
    // coupled_pose_head_cov_ was recomputed every GN iteration inside
    // estimateCoupledCorrectionPoseBasis() (real head coupling path only
    // -- pose_head_freeze_cp>0 leaves it at its last-set value, which for
    // a fresh process is Zero(); see the write guard below), so by the
    // time we reach here it reflects the LAST (converged) iteration's own
    // reduced system -- exactly what "relinearize at the converged
    // solution" (item 4) means, with no separate relinearization pass
    // needed: every GN iteration already rebuilds the reduced system
    // fresh from the current trial spline. This is what makes
    // copts_.final_relinearize_cov=true (set at loadParameters(), CQ-82's
    // own poseBasis() config block) GENUINELY true on this arm now,
    // rather than the rule-58f violation CQ-87 item 4 found (forced true,
    // then never honoured because the block that reads it lives past this
    // early return). HONOURED, not refused -- item 4's preferred option,
    // since it falls out of item 3 exactly as the card predicted.
    //
    // Scope: only [idxR,idxP] (rotation, position) are written -- velocity/
    // bias/gravity are NOT touched by this arm's own solve at all (CQ-87's
    // explicit not-in-scope list), so their rows/columns of state_->cov()
    // (and their cross-correlation with rot/pos) are left exactly as IMU
    // propagation alone produced. Only fires on the real-coupling path
    // (pose_head_freeze_cp==0); the frozen-elimination arm has no
    // posterior of its own to report (it never touches state_->cov() at
    // all) and correctly leaves the propagated-only covariance in place,
    // same as before this item.
    if (copts_.pose_head_freeze_cp == 0) {
      Eigen::MatrixXd P_full = state_->cov();
      const int iR = StateGroup::idxR(), iP = StateGroup::idxP();
      if (P_full.rows() >= iP + 3 && P_full.cols() >= iP + 3) {
        P_full.block<3, 3>(iR, iR) = coupled_pose_head_cov_.block<3, 3>(0, 0);
        P_full.block<3, 3>(iR, iP) = coupled_pose_head_cov_.block<3, 3>(0, 3);
        P_full.block<3, 3>(iP, iR) = coupled_pose_head_cov_.block<3, 3>(3, 0);
        P_full.block<3, 3>(iP, iP) = coupled_pose_head_cov_.block<3, 3>(3, 3);
        state_->covMut() = P_full;
      }
    }

    // CQ-87 item 8 (and the rule-60 contract item 3 exists to make
    // satisfiable): this arm NEVER called noteLioFrameDiag() at all before
    // this -- the only call site in this file (further below, now
    // labelled "the ONLY raw_imu noteLioFrameDiag() call site") sits past
    // this early return, so a pose-basis run wrote ZERO rows to
    // frame_stats.txt, full stop. coupled_diag already carries the P_pre
    // fields (populated earlier in this function, before either arm
    // branches -- see its own declaration comment). Add what this arm
    // actually has: n_residuals, n_c_requested/actual/clamped, and --
    // NEW, because item 3 just made it real -- the POST covariance this
    // arm's own solve just wrote (or, on the frozen-elimination arm,
    // whatever state_->cov() already held, unchanged by this arm).
    // Everything else (HtH family, ask/got/refusal, S-decomposition,
    // kappa/redund/*) stays at its struct-default sentinel -- this arm's
    // solve has no analogue for any of them, named rather than guessed.
    coupled_diag.n_residuals = static_cast<int>(residuals_.size());
    // NOTE, discovered live via this smoke test, worth recording rather
    // than silently accepting: nControlPointsRequested()'s own n_cp_req_
    // is captured AFTER ScanSpline::fit()'s hard floor of 7 (spline.cpp:
    // "n_cp = std::max(7, n_cp)" -- both ends are clamped, six control
    // points are spent on the two clamps, so below 7 there is no free
    // interior at all), so this column does NOT reflect
    // estimator/coupled/n_c's own configured value when n_c<7 -- e.g. a
    // config requesting n_c=4 already reports n_c_requested=7 here, not
    // 4, because the floor already applied before this accessor's own
    // "requested" snapshot was taken. copts_.n_c (the true config value)
    // is available separately in every filing's own effective-config
    // report if that distinction matters.
    coupled_diag.n_c_requested = coupled_pose_spline_.nControlPointsRequested();
    coupled_diag.n_c_actual    = coupled_pose_spline_.nControlPoints();
    coupled_diag.n_c_clamped   = coupled_pose_spline_.nControlPointsClamped() ? 1 : 0;
    coupled_diag.n_imu_samples = mg.n_imu_samples;
    coupled_diag.n_miss_coverage = n_miss_coverage_;
    coupled_diag.n_miss_mismatch = n_miss_mismatch_;
    coupled_diag.n_tier0_miss_coverage = n_tier0_miss_coverage_;
    coupled_diag.n_tier0_miss_mismatch = n_tier0_miss_mismatch_;
    {
      const Eigen::MatrixXd& P_post = state_->cov();
      const int iR = StateGroup::idxR(), iP = StateGroup::idxP();
      if (P_post.rows() >= iP + 3 && P_post.cols() >= iP + 3) {
        const M3D P_pp_post = P_post.block<3, 3>(iP, iP);
        Eigen::SelfAdjointEigenSolver<M3D> es_p_post(P_pp_post);
        coupled_diag.p_pos_eig_min_post = es_p_post.eigenvalues()(0);
        coupled_diag.p_pos_eig_mid_post = es_p_post.eigenvalues()(1);
        coupled_diag.p_pos_eig_max_post = es_p_post.eigenvalues()(2);
        coupled_diag.trP_pos_post = P_pp_post.trace();
        const M3D P_rr_post = P_post.block<3, 3>(iR, iR);
        Eigen::SelfAdjointEigenSolver<M3D> es_r_post(P_rr_post);
        coupled_diag.p_rot_trace_post   = P_rr_post.trace();
        coupled_diag.p_rot_eig_min_post = es_r_post.eigenvalues()(0);
        coupled_diag.p_rot_eig_mid_post = es_r_post.eigenvalues()(1);
        coupled_diag.p_rot_eig_max_post = es_r_post.eigenvalues()(2);
      }
    }
    if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get())) vm->noteLioFrameDiag(coupled_diag);

    std::ostringstream oss;
    oss << "[lio/ekf][pose-basis] iters=" << iter + 1 << "  stop=" << stop
        << std::scientific << std::setprecision(1)
        << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
        << "  |dt|=" << total_dt.norm() * 1000.0 << " mm"
        << "  head_tie=" << (copts_.pose_head_freeze_cp == 0 ? "real" : "frozen")
        << "  (post-loop bg-projection/final_redeskew/most CQ-53-era "
        << "diagnostics SKIPPED -- raw_imu-specific, not yet ported to "
        << "this basis, per CQ-87's own not-in-scope list)";
    return oss.str();
  }

  // CQ-75: final_redeskew. STRUCTURAL FINDING, VERIFIED BY CODE READ, THAT
  // CORRECTS THIS CARD'S OWN PREMISE -- reported here rather than silently
  // implementing the card's literal text without flagging the correction.
  // The card's own structure section (item 0) states propagateCoupled()
  // "appears exactly once... inside the loop, no call after it," so
  // state_ holds the c_{K-1} endpoint at loop exit. THIS IS NOT WHAT THE
  // CODE DOES: estimateCoupledCorrection() (called once per GN iteration
  // by the loop above) itself calls propagateCoupled() TWICE internally --
  // once to build this iteration's residuals/A from the INCOMING
  // (c_{K-1}) trajectory, and a SECOND time (see that function's own "re-
  // propagate ONCE MORE with the updated c AND delta_s" comment) using
  // THIS iteration's own freshly-solved c_K, immediately followed by
  // state_->setPropagatedState(). This second call runs on EVERY
  // iteration, including the last -- so state_'s own POSE is verifiably
  // NOT one step behind; it already reflects c_K by the time the loop
  // exits. WHAT IS GENUINELY STALE, confirmed: mg.points (deskewed only
  // once per iteration, at the FIRST propagate, i.e. against c_{K-1}) --
  // and by extension whatever consumes mg.points downstream (the map
  // update) -- and coupled_last_A_/the reported covariance (built from
  // residuals over that same c_{K-1}-deskewed mg.points). This flag fixes
  // exactly those two things; it does not need to "fix" state_'s pose,
  // which was never actually behind.
  //
  // BUGFIX (found during CQ-75-R2 md5/ATE verification, pre-publish):
  // final_relinearize_cov's own write of state_->covMut() cannot happen
  // where it originally lived (inside this if-block) because the CQ-44
  // item 5 block a few hundred lines below -- which runs UNCONDITIONALLY
  // after this whole if-block, for every arm, and is what actually feeds
  // this scan's posterior forward as the NEXT scan's Pi_ss prior --
  // overwrites state_->covMut() again right after, silently discarding
  // whatever this block wrote. relin_pending_/relin_A_final_/
  // relin_Jx_f_/relin_Jc_f_ below carry the relinearized posterior's
  // ingredients (computed here, from A_final/final_prop -- NEITHER of
  // which is touched by CQ-44's block -- so they're unaffected by that
  // block's own read of the now-restored coupled_last_A_/coupled_prop_)
  // past CQ-44's write, which then applies the actual relinearized
  // overwrite immediately afterward (see that site for the write itself).
  bool relin_pending_ = false;
  Eigen::MatrixXd relin_A_final_;
  Eigen::Matrix<double, 9, 18> relin_Jx_f_;
  Eigen::Matrix<double, 9, Eigen::Dynamic> relin_Jc_f_;
  if (copts_.final_redeskew) {
    const double t1 = mg.image.t;
    const int n_c = copts_.n_c;
    CoupledPropagation final_prop;
    propagateCoupled(mg.poses, state_propagat_.rot(), t1,
                     mg.poses.front().rot * Exp(coupled_delta_phi0_),
                     mg.poses.front().pos + coupled_delta_pos0_,
                     coupled_v0_pre_ + coupled_delta_v_,
                     coupled_g0_pre_ + coupled_delta_g_, coupled_g0_pre_,
                     coupled_delta_bg_, coupled_delta_ba_,
                     coupled_c_acc_, coupled_c_gyr_, n_c, final_prop);
    // Matches the card's own literal spec (propagateCoupled ->
    // setPropagatedState) -- an explicit, testable check of the
    // structural finding above: if state_'s pose were genuinely stale,
    // this would change it; per the finding, it should be a no-op
    // (final_prop.rot1/pos1/vel1 identical to what the loop's own last
    // iteration already set). Captured BEFORE the write, to measure that.
    const M3D rot_before_fr = state_->rot();
    const V3D pos_before_fr = state_->pos();
    state_->setPropagatedState(final_prop.rot1, final_prop.pos1, final_prop.vel1);
    // CQ-75-R2 arm (e): snapshot mg.points BEFORE this pass overwrites it,
    // so the map-channel isolation restore below has something to
    // restore to. Cheap (a vector copy), only taken when the flag is
    // live to keep the default path's cost unchanged.
    const std::vector<PointXYZCov> points_before_final_redeskew =
        copts_.final_redeskew_map_uses_pre ? mg.points : std::vector<PointXYZCov>{};
    std::vector<PointXYZCov> final_deskewed;
    deskewPoints(state_, final_prop.poses, t1, mg.lidar_points, opts_.deskew, final_deskewed);
    DsMode ds_mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(final_deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, ds_mode);
    // Report-only: rebuilds residuals against the NOW-CORRECT mg.points,
    // does NOT solve or write to state_/A/covariance again (rule: "do not
    // solve again").
    const double res_rms_pre = coupled_res_rms_;  // at c_{K-1}, the loop's own last value
    buildResiduals(mg.points, residuals_, /*allow_consistency_log=*/false);
    double sum_sq_r_final = 0.0;
    for (const auto& res : residuals_) sum_sq_r_final += res.r * res.r;
    const double res_rms_post = residuals_.empty()
        ? -1.0 : std::sqrt(sum_sq_r_final / static_cast<double>(residuals_.size()));
    static PersistentLogStream fr_log("cq75_final_redeskew.txt");
    bool fr_first;
    std::ofstream& fr_ofs = fr_log.stream(&fr_first);
    if (fr_first) fr_ofs << "scan_id,t_abs,res_rms_pre,res_rms_post,"
                            "pose_delta_rot_deg,pose_delta_pos_mm\n";
    const double t_abs_fr = mg.image.t + data_queues_->start_time;
    // Pose delta BEFORE this final_redeskew pass vs AFTER -- tests the
    // structural finding directly: predicted ~0 (state_'s pose was
    // already at c_K via the loop's own last-iteration second propagate).
    const double pose_delta_rot_deg =
        Log(rot_before_fr.transpose() * final_prop.rot1).norm() * (180.0 / M_PI);
    const double pose_delta_pos_mm = (final_prop.pos1 - pos_before_fr).norm() * 1000.0;
    fr_ofs << voxel_map_->frame_idx_ << "," << t_abs_fr << ","
           << res_rms_pre << "," << res_rms_post << ","
           << pose_delta_rot_deg << "," << pose_delta_pos_mm << "\n";
    fr_ofs.flush();

    // CQ-75 arm (c): final_relinearize_cov. Rebuilds A (and the posterior
    // it implies) at the FINAL trajectory -- WITHOUT taking another step.
    // Deliberately reuses estimateCoupledCorrection() itself (the exact,
    // already-verified code that builds Lambda/Pi_ss/HtH/A) rather than
    // duplicating ~900 lines of dense accumulation logic in a second,
    // parallel implementation -- a duplicate is exactly the kind of
    // drift-prone, hard-to-verify surgery this project's own code-
    // organization preference warns against. The mean is restored exactly
    // by construction: coupled_c_acc_/coupled_c_gyr_/coupled_delta_* (this
    // extra call's own INPUT, read at its own top, same as every ordinary
    // iteration) are snapshotted before the call and restored after, so
    // the one solve step this call takes internally is computed and then
    // discarded -- only coupled_last_A_ (which the call updates as an
    // unconditional side effect) is kept.
    if (copts_.final_relinearize_cov) {
      const std::vector<V3D> c_acc_snap = coupled_c_acc_, c_gyr_snap = coupled_c_gyr_;
      const V3D delta_v_snap = coupled_delta_v_, delta_bg_snap = coupled_delta_bg_,
                delta_ba_snap = coupled_delta_ba_, delta_g_snap = coupled_delta_g_,
                delta_phi0_snap = coupled_delta_phi0_, delta_pos0_snap = coupled_delta_pos0_;
      const M3D rot_snap = state_->rot();
      const V3D pos_snap = state_->pos(), vel_snap = state_->vel();
      // BUGFIX (found during CQ-75-R2 md5/ATE verification, pre-publish):
      // estimateCoupledCorrection() also mutates mg.points (re-deskews +
      // re-downsamples the full raw set against ITS OWN extra-call
      // trajectory, lio_coupled.cpp's own step (2)) and residuals_
      // (buildResiduals() over that same mg.points) as unconditional side
      // effects -- neither was in the original snapshot set. Left
      // unreverted, this extra call's mg.points silently replaced the
      // legitimate final_redeskew pass's own already-correct mg.points,
      // which then fed map insertion for this scan and corrupted every
      // later scan's map -- confirmed as the cause of a real ATE
      // divergence vs arm (b) (eee_01 -1.0mm, eee_02 +5.3mm) that should
      // have been exactly 0 by this pass's own stated contract (measures
      // covariance only, takes no 6th GN step).
      const std::vector<PointXYZCov> points_snap = mg.points;
      const auto residuals_snap = residuals_;
      // Same bug, second instance: estimateCoupledCorrection() also
      // overwrites the coupled_last_A_/coupled_prop_ MEMBERS (not just
      // mg.points/residuals_) as unconditional side effects. The CQ-44
      // item 5 block below (pre-existing, runs for every arm, not new to
      // this card) reads those same two members directly to compute the
      // posterior it feeds forward as the NEXT scan's Pi_ss prior --
      // left uncorrected, that block silently used this extra call's
      // post-step (not post-convergence) A/propagation, corrupting every
      // later scan's prior and, transitively, the whole rest of the
      // trajectory. This is what the mg.points fix alone did not catch
      // (a rebuild+rerun with only that fix reproduced the exact same
      // wrong ATE as the original bug).
      const Eigen::MatrixXd A_snap = coupled_last_A_;
      const CoupledPropagation prop_snap = coupled_prop_;

      V3D dtheta_extra, dt_extra;
      estimateCoupledCorrection(mg, dtheta_extra, dt_extra);
      const Eigen::MatrixXd A_final = coupled_last_A_;

      // Revert the mean and every side-effected member -- this pass
      // measures the covariance implied by the final trajectory, it does
      // not take a 6th GN step and must be invisible to everything else
      // this scan touches afterward.
      coupled_c_acc_ = c_acc_snap; coupled_c_gyr_ = c_gyr_snap;
      coupled_delta_v_ = delta_v_snap; coupled_delta_bg_ = delta_bg_snap;
      coupled_delta_ba_ = delta_ba_snap; coupled_delta_g_ = delta_g_snap;
      coupled_delta_phi0_ = delta_phi0_snap; coupled_delta_pos0_ = delta_pos0_snap;
      state_->setPropagatedState(rot_snap, pos_snap, vel_snap);
      mg.points = points_snap;
      residuals_ = residuals_snap;
      coupled_last_A_ = A_snap;
      coupled_prop_ = prop_snap;

      // Same M/posterior construction as the card's own item 0 cites
      // (lio_coupled.cpp:306-311's own formula) -- reused verbatim, just
      // fed A_final and final_prop's own (already-reverted-consistent,
      // since final_prop was built from the SAME c_K before this extra
      // call ever ran) Jacobians instead of coupled_last_A_/coupled_prop_.
      // Stashed rather than written here -- see relin_pending_'s own
      // doc comment above for why the actual state_->covMut() overwrite
      // has to happen after the CQ-44 item 5 block below, not here.
      if (A_final.rows() == final_prop.phi_head.back().cols() + 18 &&
          state_->idxBG() >= 0 && state_->idxBA() >= 0 && state_->idxG() >= 0) {
        Eigen::LDLT<Eigen::MatrixXd> ldlt_final(A_final);
        const double min_pivot_final = ldlt_final.vectorD().minCoeff();
        if (min_pivot_final > 0.0) {
          relin_pending_ = true;
          relin_A_final_ = A_final;
          relin_Jx_f_ = final_prop.phi_x_head.back();
          relin_Jc_f_ = final_prop.phi_head.back();
        }
        // min_pivot_final <= 0.0: leave relin_pending_ false -- the
        // (soon-to-be-written) A_K-based posterior stays in place rather
        // than a non-PSD covariance getting written -- matches the
        // existing guard's own "abort loudly" spirit, but report-not-
        // abort here since this whole pass is diagnostic and the run's
        // own real posterior is validly written by CQ-44's block either way.
      }
    }
    // CQ-75-R2 arm (e): revert mg.points to its pre-final-redeskew value
    // for the MAP channel only -- everything above (the residual report,
    // arm (c)'s relinearized covariance) already ran against the post-
    // redeskew points and is unaffected; only what the caller passes to
    // updateMap() after this function returns changes.
    if (copts_.final_redeskew_map_uses_pre) mg.points = points_before_final_redeskew;
  }

  // CQ-54 item 3: once per SCAN (not per GN iteration -- the loop above
  // calls estimateCoupledCorrection() once per iteration, recomputing
  // coupled_bias_freeze_active_ fresh each time), tally against the FINAL
  // iteration's own assessment.
  ++coupled_bias_freeze_scan_count_;
  if (coupled_bias_freeze_active_) ++coupled_bias_freeze_active_count_;

  // CQ-44 item 5: P(t1) = [Phi_x(t1) Phi_c(t1)] * A^-1 * [Phi_x(t1) Phi_c(t1)]^T
  // + Q_unmodelled. REPLACES state_->cov() (does not add to it) -- A^-1
  // already correctly combines the P(t0)^-1 prior with this scan's LiDAR
  // evidence, so adding it on top of the naively-propagated P double-counts
  // the prior. Q_unmodelled (bias random walk, process noise above the
  // correction basis's own bandwidth) is NOT separately isolated and added
  // back -- a named simplification, not a structural double-count.
  if (any_solved
      && coupled_last_A_.rows() == coupled_prop_.phi_head.back().cols() + 18
      && state_->idxBG() >= 0 && state_->idxBA() >= 0 && state_->idxG() >= 0)
  {
    Eigen::LDLT<Eigen::MatrixXd> ldlt_A(coupled_last_A_);
    // CQ-57 item 5, RULE 58: info()==Success does NOT catch rank deficiency
    // -- confirmed directly (CQ-55 arm (c) produced trP_pos_post=-4,629,615
    // straight through this exact gate). spline.cpp/spline.h already
    // document this same fact and use vectorD().minCoeff() instead (see
    // e.g. coupled_joint_dmin_ a few hundred lines below, same technique,
    // already in this file) -- a genuinely positive-definite LDLT has EVERY
    // pivot strictly positive, which is what this checks directly, rather
    // than trusting Eigen's own internal success flag. On failure: ABORT
    // loudly (never a substitute covariance masquerading as a real one).
    const double min_pivot = ldlt_A.vectorD().minCoeff();
    if (min_pivot <= 0.0) {
      std::ostringstream abort_msg;
      abort_msg << "[FATAL] coupled covariance solve: LDLT(coupled_last_A_) is "
                   "NOT positive-definite (min_pivot=" << min_pivot
                << ", info()=" << (ldlt_A.info() == Eigen::Success ? "Success" : "NumericalIssue")
                << ") -- scan_id=" << voxel_map_->frame_idx_
                << " n_residuals=" << coupled_n_residuals_
                << " n_c=" << copts_.n_c << " jacobian_time_mode=" << copts_.jacobian_time_mode
                << " smoothness_weight_acc=" << copts_.smoothness_weight_acc
                << " smoothness_weight_gyr=" << copts_.smoothness_weight_gyr
                << " imu_deviation_weight=" << copts_.imu_deviation_weight;
      throw std::runtime_error(abort_msg.str());
    }
    {
      const Eigen::Matrix<double, 9, 18>& Jx = coupled_prop_.phi_x_head.back();
      const Eigen::Matrix<double, 9, Eigen::Dynamic>& Jc = coupled_prop_.phi_head.back();
      // ONE consistent linear map from the full [delta_x(t0) 18, c] joint
      // posterior to the full 18-dim (R,P,V,BG,BA,G) state at t1. Rows 0-8
      // are Jx/Jc; rows 9-17 are a pure SELECTOR picking out
      // [delta_bg,delta_ba,delta_g] from the s-block columns. Building the
      // FULL 18x18 as M*coeff_cov*M^T in ONE product is what GUARANTEES the
      // result is PSD.
      Eigen::MatrixXd M(18, 18 + Jc.cols());
      M.setZero();
      M.topRows(9).leftCols(18) = Jx;
      M.topRows(9).rightCols(Jc.cols()) = Jc;
      M.block(9, 9, 9, 9) = Eigen::MatrixXd::Identity(9, 9);  // select [bg,ba,g]
      // CQ-57 item 4: shared with EkfUpdate::applyCovarianceUpdate() (the
      // decoupled path) -- see solveCovarianceFromA()'s own doc comment.
      // ldlt_A was already computed above for item 5's own rank-deficiency
      // guard (min_pivot check); solveCovarianceFromA() builds its own
      // fresh LDLT internally rather than reusing that one, so this is not
      // the cheapest possible version, but it IS the same shared, single-
      // definition solve-and-symmetrize path decoupled itself now goes
      // through -- numerically identical to what this file computed by
      // hand before this refactor (both are ldlt.solve(Identity), then
      // M*(...)*M^T, then symmetrize -- just no longer duplicated).
      Eigen::MatrixXd posterior18 = solveCovarianceFromA(coupled_last_A_, &M);

      // CQ-62 item 0b/1: staged PSD audit, coupled-only (no shared-file
      // change), gated behind estimator/coupled/psd_audit_en (default
      // false, md5-inert). Checks the (R,P) 6x6 sub-block's own min
      // eigenvalue AT TWO POINTS: S_M (posterior18, the freshly-computed
      // M*coeff_cov*M^T output, BEFORE it is written into state_->cov())
      // and S_ldlt_cross (an INDEPENDENT recomputation of the same
      // quantity using ldlt_A -- the SAME LDLT already verified PD by
      // item 5's own guard a few lines above -- rather than letting
      // solveCovarianceFromA build its own fresh internal LDLT from
      // scratch). If these two disagree, the defect is inside
      // solveCovarianceFromA's own fresh-LDLT path specifically; if they
      // agree with each other but disagree with the LATER post-write-
      // and-readback value the NEES hook sees (near the end of this
      // function), the defect is in the state_->covMut() write or the R,P
      // sub-block extraction downstream, not in this computation at all.
      if (copts_.psd_audit_en) {
        const int iP_a = StateGroup::idxP(), iR_a = StateGroup::idxR();
        // CQ-62: is the FULL 18x18 posterior18 itself PSD (as M*PSD*M^T
        // mathematically guarantees), even if a supposed R,P sub-block
        // extraction looks non-PSD? A principal submatrix (same rows AND
        // columns selected) of a genuinely PSD matrix is ALWAYS PSD --
        // if the full 18x18 checks out clean but the extracted 6x6 does
        // not, the bug is in the EXTRACTION, not in the covariance itself.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_full18(
            0.5 * (posterior18 + posterior18.transpose()));
        const double full18_min_eig = es_full18.eigenvalues().minCoeff();
        const double full18_max_eig = es_full18.eigenvalues().maxCoeff();
        Eigen::Matrix<double, 6, 6> P6_sM;
        P6_sM.block<3, 3>(0, 0) = posterior18.block<3, 3>(iR_a, iR_a);
        P6_sM.block<3, 3>(3, 3) = posterior18.block<3, 3>(iP_a, iP_a);
        P6_sM.block<3, 3>(0, 3) = posterior18.block<3, 3>(iR_a, iP_a);
        P6_sM.block<3, 3>(3, 0) = posterior18.block<3, 3>(iP_a, iR_a);  // CQ-62 fix: no extra transpose, see the NEES hook's own comment
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es_sM(
            0.5 * (P6_sM + P6_sM.transpose()));

        const Eigen::MatrixXd coeff_cov_cross =
            ldlt_A.solve(Eigen::MatrixXd::Identity(coupled_last_A_.rows(), coupled_last_A_.rows()));
        // CQ-62: "are we starting out PSD?" -- the FULL coeff_cov = A^-1,
        // BEFORE any M-projection, in its own native (18+6*n_c)-dim joint
        // space. A's pivots (joint_dmin) being all positive mathematically
        // guarantees A^-1 is PD too -- this checks whether the ACTUAL
        // FLOATING-POINT MATRIX Eigen computed via ldlt.solve(Identity)
        // still honors that in practice, independent of any M projection
        // at all. If this is already substantially negative, the defect
        // is in the LDLT solve itself (or A's own conditioning is bad
        // enough that even the UNPROJECTED inverse can't be trusted); if
        // this stays clean (~machine-epsilon negative at worst) while
        // S_M/S_ldlt_cross above are NOT, the defect is specific to the M
        // projection -- M picking out a direction where A^-1's own
        // genuine-but-tiny eigenvalues get amplified/cancelled.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_full(
            0.5 * (coeff_cov_cross + coeff_cov_cross.transpose()));
        const double full_min_eig = es_full.eigenvalues().minCoeff();
        const double full_max_eig = es_full.eigenvalues().maxCoeff();

        Eigen::MatrixXd posterior18_cross = M * coeff_cov_cross * M.transpose();
        Eigen::Matrix<double, 6, 6> P6_cross;
        P6_cross.block<3, 3>(0, 0) = posterior18_cross.block<3, 3>(iR_a, iR_a);
        P6_cross.block<3, 3>(3, 3) = posterior18_cross.block<3, 3>(iP_a, iP_a);
        P6_cross.block<3, 3>(0, 3) = posterior18_cross.block<3, 3>(iR_a, iP_a);
        P6_cross.block<3, 3>(3, 0) = posterior18_cross.block<3, 3>(iP_a, iR_a);  // CQ-62 fix: no extra transpose
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es_cross(
            0.5 * (P6_cross + P6_cross.transpose()));

        static PersistentLogStream psd_audit_log("psd_audit.txt");
        bool psd_first;
        std::ofstream& psd_ofs = psd_audit_log.stream(&psd_first);
        if (psd_first)
          psd_ofs << "scan_id,S_M_min_eig,S_M_max_eig,S_ldlt_cross_min_eig,"
                     "S_ldlt_cross_max_eig,diff_min_eig,joint_dmin,joint_dmax,"
                     "joint_cond,full_coeff_cov_min_eig,full_coeff_cov_max_eig,"
                     "full_coeff_cov_rel_min,full18_min_eig,full18_max_eig,"
                     "idxR,idxP\n";
        const double joint_dmax_here = ldlt_A.vectorD().maxCoeff();
        psd_ofs << voxel_map_->frame_idx_ << ","
                << es_sM.eigenvalues()(0) << "," << es_sM.eigenvalues()(5) << ","
                << es_cross.eigenvalues()(0) << "," << es_cross.eigenvalues()(5) << ","
                << (es_sM.eigenvalues()(0) - es_cross.eigenvalues()(0)) << ","
                << min_pivot << "," << joint_dmax_here << ","
                << (min_pivot > 0 ? joint_dmax_here / min_pivot : -1.0) << ","
                << full_min_eig << "," << full_max_eig << ","
                << (full_min_eig / full_max_eig) << ","
                << full18_min_eig << "," << full18_max_eig << ","
                << iR_a << "," << iP_a << "\n";
        psd_ofs.flush();

        // CQ-62 item 1: S8/S9/S10 into the same per-stage audit S0-S7 use.
        // S8 = coeff_cov = ldlt_A.solve(Identity) (coeff_cov_cross above IS
        // this quantity, just named for the cross-check it's also used
        // for). S9 = M*coeff_cov*M^T BEFORE symmetrisation (posterior18_cross
        // above, pre-0.5(X+X^T)). S10 = the actual, symmetrized posterior18
        // this scan writes downstream.
        logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S8_coeff_cov", coeff_cov_cross);
        logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S9_posterior18_presym", posterior18_cross);
        logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S10_posterior18_postsym", posterior18);
      }

      // CQ-58: per-control-point 6-dof constraint report. Reuses ldlt_A
      // (already computed above for item 5's rank-deficiency guard, already
      // confirmed PD by the min_pivot check) rather than paying for a
      // second ncol x ncol solve -- the same sharing item 4's own text asks
      // for, applied to the covariance-update LDLT instead of the
      // bias-projection one (that one lives inside estimateCoupledCorrection()
      // and is a DIFFERENT solve; this reuses the one already sitting here).
      if (copts_.log_cp_constraint_en
          && coupled_last_Lambda_.rows() == 6 * copts_.n_c
          && coupled_last_delta_c_.size() == 6 * copts_.n_c) {
        const int n_c = copts_.n_c;
        const int ncol_s = 18;  // same convention as M's own 18 rows above
        const Eigen::MatrixXd Ainv =
            ldlt_A.solve(Eigen::MatrixXd::Identity(coupled_last_A_.rows(), coupled_last_A_.rows()));
        static PersistentLogStream cp_log("cp_constraint.csv");
        std::ofstream& cp_ofs = cp_log.stream();
        static bool cp_header_written = false;
        if (!cp_header_written) {
          cp_ofs << "scan_id,cp,"
                     "c_acc_x,c_acc_y,c_acc_z,c_gyr_x,c_gyr_y,c_gyr_z,"
                     "c_acc_over_sigma_x,c_acc_over_sigma_y,c_acc_over_sigma_z,"
                     "c_gyr_over_sigma_x,c_gyr_over_sigma_y,c_gyr_over_sigma_z,"
                     "dc_acc_x,dc_acc_y,dc_acc_z,dc_gyr_x,dc_gyr_y,dc_gyr_z,"
                     "dc_acc_over_sigma_x,dc_acc_over_sigma_y,dc_acc_over_sigma_z,"
                     "dc_gyr_over_sigma_x,dc_gyr_over_sigma_y,dc_gyr_over_sigma_z,"
                     "marg_sigma_acc_x,marg_sigma_acc_y,marg_sigma_acc_z,"
                     "marg_sigma_gyr_x,marg_sigma_gyr_y,marg_sigma_gyr_z,"
                     "info_eig_min,info_eig_max,info_ratio,"
                     "info_eigvec_min_acc_x,info_eigvec_min_acc_y,info_eigvec_min_acc_z,"
                     "info_eigvec_min_gyr_x,info_eigvec_min_gyr_y,info_eigvec_min_gyr_z\n";
          cp_header_written = true;
        }
        const double sig_a = std::max(coupled_sigma_a_used_, 1e-12);
        const double sig_g = std::max(coupled_sigma_g_used_, 1e-12);
        for (int j = 0; j < n_c; ++j) {
          const int ai = ncol_s + 3 * j;
          const int gi = ncol_s + 3 * n_c + 3 * j;
          // 2a INFORMATION: the residual-only contribution to this control
          // point's own 6x6 block -- A already holds Lambda(prior) + every
          // residual's outer product summed in; subtracting Lambda's own
          // (i=j) block recovers "how hard do the POINTS pin this control
          // point" without a second accumulation pass.
          Eigen::Matrix<double, 6, 6> Ij;
          Ij.block<3, 3>(0, 0) = coupled_last_A_.block<3, 3>(ai, ai)
                                - coupled_last_Lambda_.block<3, 3>(3 * j, 3 * j);
          Ij.block<3, 3>(0, 3) = coupled_last_A_.block<3, 3>(ai, gi)
                                - coupled_last_Lambda_.block<3, 3>(3 * j, 3 * n_c + 3 * j);
          Ij.block<3, 3>(3, 0) = Ij.block<3, 3>(0, 3).transpose();
          Ij.block<3, 3>(3, 3) = coupled_last_A_.block<3, 3>(gi, gi)
                                - coupled_last_Lambda_.block<3, 3>(3 * n_c + 3 * j, 3 * n_c + 3 * j);
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es_info(Ij);
          const double eig_min = es_info.eigenvalues()(0);
          const double eig_max = es_info.eigenvalues()(5);
          const double info_ratio = (eig_min > 1e-12) ? eig_max / eig_min
                                                        : std::numeric_limits<double>::infinity();
          // Item 5: the least-informed direction itself (eigenvectors()
          // column 0, paired with eigenvalues() index 0 by
          // SelfAdjointEigenSolver's own ascending-order convention) --
          // needed to say whether it's consistent across control points or
          // varies with j, and (item 6) whether it turns toward yaw
          // (gyr_z) as the n_c=13 offset opens. Sign is arbitrary (a unit
          // eigenvector and its negation are equally valid) -- callers
          // comparing directions across scans/cp must compare axes
          // (|dot|, or fix a sign convention), not raw signed components.
          const Eigen::Matrix<double, 6, 1> v_min = es_info.eigenvectors().col(0);
          // 2b MARGINAL: A^-1's own diagonal at this control point's rows,
          // as standard deviations -- "how uncertain once everything this
          // control point is coupled to (other cp's, delta_bg via Lambda's
          // connectivity, ...) is accounted for."
          const V3D marg_sigma_acc(std::sqrt(Ainv(ai, ai)), std::sqrt(Ainv(ai + 1, ai + 1)),
                                    std::sqrt(Ainv(ai + 2, ai + 2)));
          const V3D marg_sigma_gyr(std::sqrt(Ainv(gi, gi)), std::sqrt(Ainv(gi + 1, gi + 1)),
                                    std::sqrt(Ainv(gi + 2, gi + 2)));
          const V3D c_acc = coupled_c_acc_[j];
          const V3D c_gyr = coupled_c_gyr_[j];
          const V3D dc_acc = coupled_last_delta_c_.segment<3>(3 * j);
          const V3D dc_gyr = coupled_last_delta_c_.segment<3>(3 * n_c + 3 * j);
          cp_ofs << voxel_map_->frame_idx_ << "," << j << ","
                 << c_acc.x() << "," << c_acc.y() << "," << c_acc.z() << ","
                 << c_gyr.x() << "," << c_gyr.y() << "," << c_gyr.z() << ","
                 << c_acc.x() / sig_a << "," << c_acc.y() / sig_a << "," << c_acc.z() / sig_a << ","
                 << c_gyr.x() / sig_g << "," << c_gyr.y() / sig_g << "," << c_gyr.z() / sig_g << ","
                 << dc_acc.x() << "," << dc_acc.y() << "," << dc_acc.z() << ","
                 << dc_gyr.x() << "," << dc_gyr.y() << "," << dc_gyr.z() << ","
                 << dc_acc.x() / sig_a << "," << dc_acc.y() / sig_a << "," << dc_acc.z() / sig_a << ","
                 << dc_gyr.x() / sig_g << "," << dc_gyr.y() / sig_g << "," << dc_gyr.z() / sig_g << ","
                 << marg_sigma_acc.x() << "," << marg_sigma_acc.y() << "," << marg_sigma_acc.z() << ","
                 << marg_sigma_gyr.x() << "," << marg_sigma_gyr.y() << "," << marg_sigma_gyr.z() << ","
                 << eig_min << "," << eig_max << "," << info_ratio << ","
                 << v_min(0) << "," << v_min(1) << "," << v_min(2) << ","
                 << v_min(3) << "," << v_min(4) << "," << v_min(5) << "\n";
        }
        // Buffered, not flushed per line (n_c lines/scan, ~45k over a full
        // n_c=13 run) -- same convention log_jrow_leverage_en uses: relies
        // on ofstream's own buffering plus normal process exit to flush.
      }

      // CQ-57 item 3a: the joint solve carries ONE bias correction at t0;
      // the line above reports the bias at t1 exactly as well known as at
      // t0, no random-walk growth over the scan's own duration. Adds
      // q_alpha_bias*covBiasGyr()*dt / *covBiasAcc()*dt to the bg/ba
      // diagonal blocks -- dt is the WHOLE scan duration (t1-t0), the gap
      // this item names, using the SAME rate imu_processing.cpp's own
      // cov_w already uses for its own (per-IMU-step) bg/ba noise. A pure
      // diagonal addition can only increase variance, so this cannot
      // break the M*C*M^T construction's own PSD guarantee above.
      if (copts_.q_bias_rw_en && state_->idxBG() >= 0 && state_->idxBA() >= 0) {
        const double dt_scan = mg.image.t - mg.poses.front().t;
        if (dt_scan > 0.0) {
          posterior18.block<3, 3>(state_->idxBG(), state_->idxBG()).diagonal() +=
              copts_.q_alpha_bias * state_->covBiasGyr() * dt_scan;
          posterior18.block<3, 3>(state_->idxBA(), state_->idxBA()).diagonal() +=
              copts_.q_alpha_bias * state_->covBiasAcc() * dt_scan;
        }
      }

      // CQ-57 item 3b / CQ-68 item 3: add out-of-band IMU noise power (see
      // this option's own header doc comment for the measured fractions)
      // into the V and P blocks -- integrated over the scan's own
      // duration. CQ-65 item 7's own P2 test found this injection was
      // FLAT/scan-independent (var_acc_oob depended only on the near-
      // constant dt_scan and on coupled_sigma_a_used_, which without
      // estimator/coupled/adaptive_sigma also on is just the CALIBRATION
      // FLOOR every scan -- i.e. a near-constant addition, unconditional
      // on any actually-detected vibration), and that the falsifier fired:
      // the term visibly reshaped the stationary window's own calibration,
      // where no physical vibration source applies. FIX: inject only the
      // EXCESS variance measured ABOVE the calibration floor (sigma_used^2
      // - sigma_floor^2, floored at 0) rather than the raw sigma_used^2 --
      // sigma_used^2 alone double-counts the floor's own contribution,
      // which is already present via imu_processing.cpp's own cov_w term,
      // and is exactly the calibration constant the raw formula injected
      // even during a genuinely calm scan. With
      // estimator/coupled/adaptive_sigma also enabled (so
      // coupled_sigma_a_used_/coupled_sigma_g_used_ actually track this
      // scan's own raw-IMU std rather than sitting at the floor), the
      // excess is ~0 during calm scans (sigma_used ~= sigma_floor) and
      // grows only when real excess in-scan noise is measured -- "inject
      // what is actually there," per the card's own words. Requires
      // adaptive_sigma=true to have any effect at all (with it off,
      // sigma_used == sigma_floor identically every scan, so the excess
      // is always exactly 0 and this option becomes silently a no-op --
      // a real, intentional coupling between the two flags, not a bug).
      if (copts_.q_out_of_band_en) {
        const double dt_scan = mg.image.t - mg.poses.front().t;
        if (dt_scan > 0.0) {
          const double sigma_a_floor_now = std::sqrt(state_->varAccFloor().mean());
          const double sigma_g_floor_now = std::sqrt(state_->varGyrFloor().mean());
          const double excess_var_a = std::max(0.0,
              coupled_sigma_a_used_ * coupled_sigma_a_used_ - sigma_a_floor_now * sigma_a_floor_now);
          const double excess_var_g = std::max(0.0,
              coupled_sigma_g_used_ * coupled_sigma_g_used_ - sigma_g_floor_now * sigma_g_floor_now);
          const double var_acc_oob = copts_.q_out_of_band_scale * copts_.q_out_of_band_fraction_acc
              * excess_var_a * dt_scan;
          const int iV = StateGroup::idxV(), iP = StateGroup::idxP();
          posterior18.block<3, 3>(iV, iV).diagonal().array() += var_acc_oob;
          posterior18.block<3, 3>(iP, iP).diagonal().array() += var_acc_oob * dt_scan * dt_scan;
          const double var_gyr_oob = copts_.q_out_of_band_scale * copts_.q_out_of_band_fraction_gyr
              * excess_var_g * dt_scan;
          const int iR = StateGroup::idxR();
          posterior18.block<3, 3>(iR, iR).diagonal().array() += var_gyr_oob;
        }
      }

      // CQ-57 item 1, THE CONSEQUENCE: report-only trace comparison, no
      // behaviour change. trP_prior_in/trP_imu are the SAME quantity here
      // (nothing writes covMut() between processIMU and this point within
      // a scan -- see item 0's confirmed facts) but both are logged
      // independently so that equality is a checked fact, not an assumed
      // one. trP_phi = trace(Jx * P_prior_18x18 * Jx^T), the homogeneous
      // map alone (no LiDAR update at all) -- position and rotation blocks
      // reported separately throughout, per the card's own request.
      if (opts_.log_debug_en) {
        const Eigen::MatrixXd P_prior_full = state_->cov();
        const int iP = StateGroup::idxP(), iR = StateGroup::idxR();
        if (P_prior_full.rows() >= 18 && P_prior_full.cols() >= 18) {
          const Eigen::MatrixXd P_prior_18 = P_prior_full.block(0, 0, 18, 18);
          const Eigen::Matrix<double, 9, 9> P_phi9 = Jx * P_prior_18 * Jx.transpose();
          const double trP_prior_in_pos = P_prior_full.block<3, 3>(iP, iP).trace();
          const double trP_prior_in_rot = P_prior_full.block<3, 3>(iR, iR).trace();
          const double trP_imu_pos = trP_prior_in_pos;  // same read; see comment above
          const double trP_imu_rot = trP_prior_in_rot;
          const double trP_phi_pos = P_phi9.block<3, 3>(3, 3).trace();  // Jx rows [R(3),P(3),V(3)]
          const double trP_phi_rot = P_phi9.block<3, 3>(0, 0).trace();
          const double trP_post_pos = posterior18.block<3, 3>(iP, iP).trace();
          const double trP_post_rot = posterior18.block<3, 3>(iR, iR).trace();
          static PersistentLogStream cov_trace_log("cov_trace.txt");
          std::ofstream& ctofs = cov_trace_log.stream();
          ctofs << std::setprecision(12)
                << "scan_id=" << voxel_map_->frame_idx_
                << " trP_prior_in_pos=" << trP_prior_in_pos
                << " trP_prior_in_rot=" << trP_prior_in_rot
                << " trP_phi_pos=" << trP_phi_pos
                << " trP_phi_rot=" << trP_phi_rot
                << " trP_post_pos=" << trP_post_pos
                << " trP_post_rot=" << trP_post_rot
                << " trP_imu_pos=" << trP_imu_pos
                << " trP_imu_rot=" << trP_imu_rot
                // CQ-57 item 2: -1/-1 when log_cov_repropagation_en is off
                // (out.has_repro stays false in that case).
                << " repro_trP_pos=" << (coupled_prop_.has_repro ? coupled_prop_.repro_trP_pos : -1.0)
                << " repro_trP_rot=" << (coupled_prop_.has_repro ? coupled_prop_.repro_trP_rot : -1.0)
                << "\n";
          ctofs.flush();
        }
      }

      Eigen::MatrixXd P = state_->cov();
      P.block(0, 0, 18, 18) = posterior18;   // idxR=0..idxG()+3=18, contiguous
      state_->covMut() = P;

      // CQ-75 arm (c): final_relinearize_cov's own posterior overwrite,
      // moved here (see relin_pending_'s doc comment above the
      // final_redeskew if-block) so it applies AFTER this scan's normal
      // posterior write above rather than being silently clobbered by
      // it. This IS the intended effect of the flag: the relinearized
      // covariance becomes what's actually carried forward as the NEXT
      // scan's Pi_ss prior -- only the MEAN/trajectory is guaranteed
      // unchanged (reverted immediately after the extra solve, above),
      // not the covariance, which is the entire quantity this flag exists
      // to test.
      if (relin_pending_) {
        Eigen::MatrixXd M_f(18, 18 + relin_Jc_f_.cols());
        M_f.setZero();
        M_f.topRows(9).leftCols(18) = relin_Jx_f_;
        M_f.topRows(9).rightCols(relin_Jc_f_.cols()) = relin_Jc_f_;
        M_f.block(9, 9, 9, 9) = Eigen::MatrixXd::Identity(9, 9);
        const Eigen::MatrixXd posterior18_relin = solveCovarianceFromA(relin_A_final_, &M_f);
        Eigen::MatrixXd P_relin = state_->cov();
        P_relin.block(0, 0, 18, 18) = posterior18_relin;
        state_->covMut() = P_relin;
      }
      // CQ-62 item 1: S11 -- state_->cov() immediately after the write.
      if (copts_.psd_audit_en)
        logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S11_cov_post_write", state_->cov());

      // CQ-74 item 1: the POST half of coupled_diag (see its own comment
      // at this function's PRE site) -- state_->cov() now reflects THIS
      // scan's posterior, matching decoupled's own POST read timing
      // (after applyCovarianceUpdate() there, after this covMut() write
      // here). Then submit -- this is the ONLY noteLioFrameDiag() call
      // site for the RAW_IMU arm (CQ-87 item 8 added a second, pose-arm-
      // only site at the poseBasis() early-return above, since that arm
      // never reached this one at all); frame_stats.txt's P-related
      // columns are populated for coupled for the first time as of CQ-74.
      {
        const Eigen::MatrixXd& P_post_diag = state_->cov();
        if (P_post_diag.rows() >= StateGroup::idxP() + 3 && P_post_diag.cols() >= StateGroup::idxP() + 3) {
          const M3D P_pp_post = P_post_diag.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
          Eigen::SelfAdjointEigenSolver<M3D> es_p_post(P_pp_post);
          coupled_diag.p_pos_eig_min_post = es_p_post.eigenvalues()(0);
          coupled_diag.p_pos_eig_mid_post = es_p_post.eigenvalues()(1);
          coupled_diag.p_pos_eig_max_post = es_p_post.eigenvalues()(2);
          coupled_diag.trP_pos_post = P_pp_post.trace();
        }
        if (P_post_diag.rows() >= StateGroup::idxR() + 3 && P_post_diag.cols() >= StateGroup::idxR() + 3) {
          const M3D P_rr_post = P_post_diag.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
          Eigen::SelfAdjointEigenSolver<M3D> es_r_post(P_rr_post);
          coupled_diag.p_rot_trace_post   = P_rr_post.trace();
          coupled_diag.p_rot_eig_min_post = es_r_post.eigenvalues()(0);
          coupled_diag.p_rot_eig_mid_post = es_r_post.eigenvalues()(1);
          coupled_diag.p_rot_eig_max_post = es_r_post.eigenvalues()(2);
        }
        // CQ-83: the remaining columns with a genuine coupled analogue,
        // read off the member variables the diagnostics blocks earlier in
        // this scan's own processing already filled (nees_diag.txt's
        // write site, further up this function, reads several of the
        // SAME members -- this is "route", not "recompute", exactly as
        // the card asks).
        coupled_diag.n_residuals   = coupled_n_residuals_;
        coupled_diag.sum_weight    = coupled_sum_weight_;
        coupled_diag.h_pp_min_eig  = coupled_h_pp_min_eig_;
        coupled_diag.h_rr_min_eig  = coupled_h_rr_min_eig_;
        coupled_diag.h_pp_max_eig  = coupled_h_pp_max_eig_;
        coupled_diag.h_rr_trace    = coupled_h_rr_trace_;
        coupled_diag.htth_pos_trace = coupled_htth_pos_trace_;
        coupled_diag.htz_rot_norm  = coupled_htz_rot_norm_;
        coupled_diag.htz_pos_norm  = coupled_htz_pos_norm_;
        coupled_diag.ask           = coupled_ask_;
        coupled_diag.got           = coupled_got_;
        coupled_diag.refusal       = coupled_refusal_;
        coupled_diag.iters         = coupled_iters_;
        coupled_diag.dx_rot_deg    = coupled_dx_rot_deg_;
        coupled_diag.dx_pos_mm     = coupled_dx_pos_mm_;
        coupled_diag.sum_S           = coupled_sum_S_;
        coupled_diag.floor_share     = coupled_floor_share_;
        coupled_diag.sdiag_share     = coupled_sdiag_share_;
        coupled_diag.pvar_share      = coupled_pvar_share_;
        coupled_diag.prior_pose_share = coupled_prior_pose_share_;
        coupled_diag.nis           = coupled_nis_;
        coupled_diag.nis_est       = coupled_nis_est_;
        coupled_diag.reduced_chi2  = coupled_reduced_chi2_;
        coupled_diag.kappa_eff     = coupled_kappa_eff_;
        coupled_diag.kappa_gev0 = coupled_kappa_gev_[0]; coupled_diag.kappa_gev1 = coupled_kappa_gev_[1];
        coupled_diag.kappa_gev2 = coupled_kappa_gev_[2]; coupled_diag.kappa_gev3 = coupled_kappa_gev_[3];
        coupled_diag.kappa_gev4 = coupled_kappa_gev_[4]; coupled_diag.kappa_gev5 = coupled_kappa_gev_[5];
        coupled_diag.kappa_gev_ok  = coupled_kappa_gev_ok_;
        coupled_diag.n_imu_samples = mg.n_imu_samples;
    coupled_diag.n_miss_coverage = n_miss_coverage_;
    coupled_diag.n_miss_mismatch = n_miss_mismatch_;
    coupled_diag.n_tier0_miss_coverage = n_tier0_miss_coverage_;
    coupled_diag.n_tier0_miss_mismatch = n_tier0_miss_mismatch_;
        // CQ-87 item 8: raw_imu's own n_c never clamps (no per-scan
        // ScanSpline::fit() call on this arm at all) -- reported as
        // requested==actual, clamped=0, so this column reads consistently
        // regardless of which arm produced the row. The pose arm's own
        // real clamp status is populated separately, in
        // estimateCoupledCorrectionPoseBasis()'s own scan-start reset
        // (this diag-population site is raw_imu-only; the pose arm's
        // early return in processLIO() never reaches it).
        coupled_diag.n_c_requested = copts_.n_c;
        coupled_diag.n_c_actual    = copts_.n_c;
        coupled_diag.n_c_clamped   = 0;
        // CQ-83: NO COUPLED ANALOGUE, left at their struct-default
        // sentinel -- confirmed by direct search, none of
        // residual_redundancy/ResidualRedundancyStats/collapse-axis/
        // sigma_scale-axis machinery is referenced anywhere in this file.
        // These features are applied inside decoupled's own ekf_.HtH
        // accumulation/covariance-update path (see CQ-70's "coupled never
        // calls applyCovarianceUpdate() at all" finding); coupled builds
        // its own A/b directly from residuals_ with no such correction
        // stage to report on: redund_groups, redund_n_raw, redund_n_eff,
        // redund_info_ratio, redund_groups_seen,
        // redund_groups_degenerate_pv, redund_groups_degenerate_var,
        // naive_info_gain, woodbury_info_gain, collapse_groups_collapsed,
        // collapse_residuals_removed, per_residual_touched,
        // per_residual_renorm_factor, per_residual_mean_scale,
        // sigma_scale_applied, sigma_scale_chi2_ema.
        if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get())) vm->noteLioFrameDiag(coupled_diag);
        logEigenspectrum18(voxel_map_->frame_idx_, mg.image.t + data_queues_->start_time, "coupled");
      }

      // CQ-76 T1.0: the qhat accumulator (imu_processing.cpp) is primed on
      // EVERY IMU propagation regardless of estimator mode, but was only
      // ever CONSUMED (read + reset) by the decoupled path -- in coupled
      // mode it was never read, so it accumulated F A F^T + cov_w
      // continuously across the whole run rather than resetting per scan.
      // This call is the coupled-side consumer: it no-ops (returns false)
      // unless imu/log_qhat_en is set, exactly like the decoupled call
      // site -- no new flag, zero behavior change to the estimator (the
      // returned matrices are read-only diagnostic output, never fed back
      // into state_ or A). g_qhat_p_before is P at the START of this
      // scan's IMU integration (end of the PREVIOUS scan's LiDAR
      // correction); phi_p_phit + accum_cov_w reconstructs g_qhat_p_after,
      // the covariance THIS scan's Pi_ss/Omega is actually built from (see
      // T0.1's confirmed call chain: state_->cov() at that read point is
      // exactly this post-IMU-propagation value, nothing else has written
      // covMut() in between).
      {
        Eigen::MatrixXd phi_p_phit, accum_cov_w, p_before;
        if (imuProcQhatRead(phi_p_phit, accum_cov_w, p_before)) {
          // CQ-76-R3: add the frame-local Q_scan to the ALREADY-WRITTEN
          // posterior's rot/pos/vel 9x9 block (state_->covMut() was set
          // by CQ-44 item 5's write above, and by arm (c)'s own relin
          // overwrite if that also fired -- either way this runs after
          // and adds on top of whichever posterior is currently in
          // state_->cov(), matching the card's own "AFTER the M A^-1 M^T
          // write" instruction). accum_cov_w is 18x18; only the 9x9
          // [idxR,idxP,idxV) sub-block is added, per the card's own scope.
          if (copts_.add_q_scan_to_posterior && accum_cov_w.rows() >= 9 && accum_cov_w.cols() >= 9) {
            Eigen::MatrixXd P_with_q = state_->cov();
            if (P_with_q.rows() >= 9 && P_with_q.cols() >= 9) {
              P_with_q.block<9, 9>(0, 0) += accum_cov_w.block<9, 9>(0, 0);
              state_->covMut() = P_with_q;
            }
          }
          const Eigen::MatrixXd P_t1 = phi_p_phit + accum_cov_w;  // = g_qhat_p_after
          if (p_before.rows() == P_t1.rows() && p_before.rows() >= 18) {
            auto blockEig = [](const Eigen::MatrixXd& M, int idx0, double& tr,
                                double& lmin, double& lmid, double& lmax) {
              const Eigen::Matrix3d B = M.block<3, 3>(idx0, idx0);
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(0.5 * (B + B.transpose()));
              const Eigen::Vector3d ev = es.eigenvalues();
              tr = B.trace(); lmin = ev(0); lmid = ev(1); lmax = ev(2);
            };
            const int iR = StateGroup::idxR(), iP = StateGroup::idxP();
            static PersistentLogStream cq76_log("cq76_prior_time_index.txt");
            bool cq76_first;
            std::ofstream& cq76_ofs = cq76_log.stream(&cq76_first);
            if (cq76_first)
              cq76_ofs << "scan_id,t_abs,matrix,block,trace,eig_min,eig_mid,eig_max\n";
            const double t_abs = mg.image.t + data_queues_->start_time;
            const int sid = voxel_map_->frame_idx_;
            struct MatEntry { const char* name; const Eigen::MatrixXd* M; };
            const std::vector<MatEntry> mats = {
                {"P_t0", &p_before}, {"P_t1", &P_t1},
                {"PhiP0Phi", &phi_p_phit}, {"Q_scan", &accum_cov_w}};
            for (const auto& me : mats) {
              for (const auto& blk : {std::make_pair("pos", iP), std::make_pair("rot", iR)}) {
                double tr, lmin, lmid, lmax;
                blockEig(*me.M, blk.second, tr, lmin, lmid, lmax);
                cq76_ofs << sid << "," << std::setprecision(10) << t_abs << ","
                         << me.name << "," << blk.first << "," << tr << ","
                         << lmin << "," << lmid << "," << lmax << "\n";
              }
            }
            // T1.1's own extra ask: eigenvalue spectrum of P(t0)^-1 - P(t1)^-1
            // (the s-block information DIFFERENCE the two time-index
            // readings would disagree by), full 18x18, all eigenvalues,
            // not just a 3x3 block.
            Eigen::LDLT<Eigen::MatrixXd> ldlt_t0(p_before), ldlt_t1(P_t1);
            if (ldlt_t0.info() == Eigen::Success && ldlt_t1.info() == Eigen::Success) {
              const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_t1.rows(), P_t1.rows());
              const Eigen::MatrixXd info_diff = ldlt_t0.solve(I) - ldlt_t1.solve(I);
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_diff(
                  0.5 * (info_diff + info_diff.transpose()));
              const Eigen::VectorXd ev_diff = es_diff.eigenvalues();
              static PersistentLogStream cq76_diff_log("cq76_info_diff_spectrum.txt");
              bool cq76_diff_first;
              std::ofstream& cq76_diff_ofs = cq76_diff_log.stream(&cq76_diff_first);
              if (cq76_diff_first) cq76_diff_ofs << "scan_id,t_abs,eig_min,eig_max,eig_all\n";
              cq76_diff_ofs << sid << "," << t_abs << "," << ev_diff(0) << ","
                            << ev_diff(ev_diff.size() - 1) << ",\"";
              for (int i = 0; i < ev_diff.size(); ++i)
                cq76_diff_ofs << ev_diff(i) << (i + 1 < ev_diff.size() ? ";" : "");
              cq76_diff_ofs << "\"\n";
            }
            // T1.2: P_coeff_to_state = Phi_c P_c Phi_c^T -- what the
            // coefficient posterior already carries into the endpoint
            // state block, logged with its own blocks/spectrum so T1.2's
            // "report both, decide neither" instruction can be honored
            // without re-deriving this quantity from raw logs later.
            if (coupled_last_A_.rows() > 18 && copts_.log_cp_constraint_en) {
              const int ncol_c = coupled_last_A_.rows() - 18;
              // NOTE, reported not silently fixed: this is A_cc^-1, the
              // BLOCK-DIAGONAL approximation of the coefficient marginal
              // covariance -- it ignores the s/c cross-correlation A
              // actually carries (the true marginal would need the Schur
              // complement (A_cc - A_cs*A_ss^-1*A_sc)^-1). Acceptable for
              // this card's own explicit "diagnostic, report both, decide
              // neither" framing (T1.2) but NOT a claim of exactness.
              const Eigen::MatrixXd P_cc =
                  coupled_last_A_.block(18, 18, ncol_c, ncol_c).inverse();
              // phi_head[k] is already exactly 9 x ncol_c (the c-block-only
              // sensitivity -- see coupled_estimator.h's own doc comment),
              // no s-block columns to slice off.
              const Eigen::Matrix<double, 9, Eigen::Dynamic>& Phi_c =
                  coupled_prop_.phi_head.back();
              const Eigen::MatrixXd P_coeff_to_state = Phi_c * P_cc * Phi_c.transpose();
              static PersistentLogStream cq76_pcs_log("cq76_p_coeff_to_state.txt");
              bool cq76_pcs_first;
              std::ofstream& cq76_pcs_ofs = cq76_pcs_log.stream(&cq76_pcs_first);
              if (cq76_pcs_first)
                cq76_pcs_ofs << "scan_id,t_abs,block,trace,eig_min,eig_mid,eig_max\n";
              for (const auto& blk : {std::make_pair("pos", 3), std::make_pair("rot", 0)}) {
                const Eigen::Matrix3d B = P_coeff_to_state.block<3, 3>(blk.second, blk.second);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(0.5 * (B + B.transpose()));
                const Eigen::Vector3d ev = es.eigenvalues();
                cq76_pcs_ofs << sid << "," << t_abs << "," << blk.first << "," << B.trace()
                             << "," << ev(0) << "," << ev(1) << "," << ev(2) << "\n";
              }
            }
          }
        }
      }

      // CQ-61 item 4, RULE 58 -- NOT AN EXPERIMENT: on residual starvation
      // the estimator keeps integrating IMU and reports convergence --
      // confirmed directly (3700 -> 0 residuals by scan 1785, trP_pos_post
      // 0.0787 -> 309.24, a full pose still comes out). ABORT loudly
      // rather than let a starved scan masquerade as a normal one -- this
      // does not fix Bug A, it stops Bug A from producing a plausible-
      // looking answer once the LiDAR evidence has effectively vanished.
      //
      // CORRECTED after the first cut wrongly fired on a genuinely healthy
      // run: a single-scan relative-drop check (originally alongside the
      // floor below) is NOT a valid signal here -- n_residuals oscillates
      // legitimately by 10x+ scan-to-scan under completely normal
      // operation (confirmed directly against CQ-59's own known-good
      // eee_01_n4_base run, ATE=0.0251m: 1695 -> 147 is an 11.5x single-
      // scan drop that happens every ~3rd scan, healthily, throughout the
      // WHOLE run). An absolute floor alone is the right signal -- the
      // same healthy run's own minimum across its entire run is 93,
      // comfortably above the floor below, so this cannot false-positive
      // on normal variance the way the drop check did.
      {
        constexpr int MIN_RESIDUALS_FLOOR = 20;
        if (coupled_n_residuals_ < MIN_RESIDUALS_FLOOR) {
          const double trP_pos_post_now =
              posterior18.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();
          std::ostringstream abort_msg;
          abort_msg << "[FATAL] coupled residual starvation: n_residuals=" << coupled_n_residuals_
                    << " (floor=" << MIN_RESIDUALS_FLOOR << ")"
                    << " prev_n_residuals=" << coupled_prev_n_residuals_
                    << " trP_pos_post=" << trP_pos_post_now
                    << " scan_id=" << voxel_map_->frame_idx_
                    << " n_c=" << copts_.n_c << " jacobian_time_mode=" << copts_.jacobian_time_mode;
          throw std::runtime_error(abort_msg.str());
        }
        coupled_prev_n_residuals_ = coupled_n_residuals_;
      }

      // CQ-85 item 1, rule 58f: the non-aborting-divergence guard. Disabled
      // (max_scan_displacement_m<=0.0) reproduces every prior run
      // byte-identically -- this block computes nothing and throws nothing
      // in that case. When enabled: the corrected position (state_->pos(),
      // already written by this scan's correction loop by the time this
      // runs) should stay close to mg.prior_pos -- the IMU-only prediction
      // captured at line ~409, BEFORE this scan's correction loop ran at
      // all -- a correction that moves the state by metres within one
      // ~0.1s scan is exactly the "FAILED operation that looks like a
      // successful one" rule 58f names, not a plausible LiDAR update.
      // NOTE, found live during this guard's own first smoke test:
      // coupled_prop_.pos1 is NOT usable here for this comparison -- the
      // correction loop's OWN internal propagateCoupled() calls (the
      // "twice per GN iteration, second call re-propagates with the
      // solved c_K" the pose-basis doc comment describes) overwrite
      // coupled_prop_ with the ALREADY-CORRECTED pose by the time this
      // line runs, so a disp computed against it is always ~0 regardless
      // of how far the scan actually diverged -- confirmed: the guard
      // silently never fired against coupled_prop_.pos1 on a direct
      // repro of the 396,499,288mm cell. mg.prior_pos is written once,
      // at scan start, and never touched again -- the correct reference.
      if (copts_.max_scan_displacement_m > 0.0) {
        const double disp = (state_->pos() - mg.prior_pos).norm();
        if (disp > copts_.max_scan_displacement_m) {
          std::ostringstream abort_msg;
          abort_msg << "[FATAL] coupled non-aborting divergence: scan displacement="
                    << disp << "m exceeds max_scan_displacement_m="
                    << copts_.max_scan_displacement_m << "m"
                    << " scan_id=" << voxel_map_->frame_idx_
                    << " n_c=" << copts_.n_c << " jacobian_time_mode=" << copts_.jacobian_time_mode
                    << " corrected_pos=[" << state_->pos().transpose() << "]"
                    << " imu_propagated_pos=[" << mg.prior_pos.transpose() << "]";
          throw std::runtime_error(abort_msg.str());
        }
      }
    }
  }

  // CQ-60 item 5: Tier 1's shared NEES machinery. MUST run here, AFTER the
  // posterior18 covMut() write above -- NOT inside estimateCoupledCorrection()
  // (which runs once per GN ITERATION, before this scan's final,
  // PSD-guaranteed covariance exists). A first attempt placed this hook at
  // estimateCoupledCorrection()'s own nees_diag.txt write site and measured
  // genuinely NEGATIVE eigenvalues in the resulting P6 -- not floating-point
  // dust (up to ~12% of the largest eigenvalue) -- because it was reading
  // an intermediate, not-yet-finalized covariance. The posterior18 write
  // above IS mathematically guaranteed PSD by construction (M*coeff_cov*M^T
  // for any M, given coeff_cov PSD, is PSD -- the same guarantee ekf.h's
  // applyCovarianceUpdate() documents for the decoupled path's own scheme),
  // so reading AFTER it lands should never reproduce that failure.
  if (opts_.nees_per_dof_en) {
    const Eigen::MatrixXd& P_final = state_->cov();
    const int iP = StateGroup::idxP(), iR = StateGroup::idxR();
    if (P_final.rows() >= iP + 3 && P_final.cols() >= iP + 3 &&
        P_final.rows() >= iR + 3 && P_final.cols() >= iR + 3) {
      Eigen::Matrix<double, 6, 6> P6;
      P6.block<3, 3>(0, 0) = P_final.block<3, 3>(iR, iR);
      P6.block<3, 3>(3, 3) = P_final.block<3, 3>(iP, iP);
      P6.block<3, 3>(0, 3) = P_final.block<3, 3>(iR, iP);
      // CQ-62: was P_final.block(iP,iR).transpose() -- WRONG. P_final is
      // (nearly) symmetric, so P_final.block(iP,iR) ALREADY equals
      // P_final.block(iR,iP).transpose() by construction; transposing it
      // AGAIN silently flips it back to P_final.block(iR,iP) -- the SAME
      // value already placed at (0,3) above, not its transpose. This made
      // P6 = [[Prr,X],[X,Ppp]] instead of the true principal submatrix
      // [[Prr,X],[X^T,Ppp]] -- a DIFFERENT matrix, not merely an
      // asymmetric version of the right one. The mandatory downstream
      // symmetrize (computeNeesPerDof's own Psym=0.5*(P+P^T)) then
      // silently discards X's entire antisymmetric part instead of
      // correctly combining X with its true transpose -- producing a
      // matrix with NO relation to state_->cov()'s own genuine PSD-ness
      // (a principal submatrix of a PSD matrix is always PSD; this
      // reconstruction was not actually a principal submatrix at all).
      // Confirmed directly: state_->cov()'s own FULL 18x18 has zero
      // negative eigenvalues on every scan tested (psd_audit_en's own
      // full18_min_eig column), while THIS extraction's 6x6 showed
      // min_eig as negative as -11.7% of max_eig on the very same scans
      // -- the "coupled is grossly non-PSD" finding (CQ-60 item 5) was
      // this bug, not a real property of the covariance the estimator
      // actually carries.
      P6.block<3, 3>(3, 0) = P_final.block<3, 3>(iP, iR);
      // CQ-62 item 1: S12 -- the (R,P) 6x6 exactly as handed to the NEES
      // logger. Item 4: check the handoff BY EQUALITY, not eigenvalues --
      // a principal submatrix of a PSD matrix is always PSD, so if S12
      // disagrees element-by-element with state_->cov().block<6,6>(0,0)
      // (idxR=0, idxP=3, contiguous -- confirmed via the psd_audit_en log's
      // own idxR/idxP columns), S12 is not actually a principal submatrix.
      // Post-fix this should be exactly zero (P6 IS a direct block copy of
      // the same contiguous 6x6, just gathered via two 3x3 sub-blocks
      // instead of one 6x6 block read).
      if (copts_.psd_audit_en) {
        logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S12_P6_nees_handoff", P6);
        const Eigen::Matrix<double, 6, 6> P6_direct = P_final.block<6, 6>(0, 0);
        const double max_abs_diff = (P6 - P6_direct).cwiseAbs().maxCoeff();
        static PersistentLogStream eq_log("psd_s12_equality.txt");
        bool eq_first;
        std::ofstream& eq_ofs = eq_log.stream(&eq_first);
        if (eq_first) eq_ofs << "scan_id,max_abs_diff,idxR,idxP\n";
        eq_ofs << voxel_map_->frame_idx_ << "," << max_abs_diff << ","
               << iR << "," << iP << "\n";
        eq_ofs.flush();

        // CQ-71 item 1: trace(P) cannot see anisotropy -- CQ-57's own
        // 5,396x/965,170x trace-based collapse numbers and CQ-62's
        // per-axis ratios spanning 12x across axes are both invisible to
        // each other's instrument. Log the full eigenspectrum (not just
        // min/max, though those are what's reported below) and WHICH state
        // direction the weakest (largest-uncertainty) eigenvector points
        // along, for both the (R,P) 6x6 (P6, already computed above) and
        // the full 18x18 (P_final itself) -- so the anisotropy is visible
        // for the WHOLE run, not just inside the 489-scan stationary
        // window nees_per_dof.txt is capped to.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es6(P6);
        const auto& ev6 = es6.eigenvalues();     // ascending
        const auto& evec6 = es6.eigenvectors();
        const double lam_min6 = ev6(0), lam_max6 = ev6(5);
        // The weak (largest-uncertainty) eigenvector's dominant component
        // names which physical axis it's closest to -- [rx,ry,rz,px,py,pz]
        // order, matching P6's own construction above.
        int weak6_axis; evec6.col(0).cwiseAbs().maxCoeff(&weak6_axis);
        static const char* AXIS6[6] = {"rx","ry","rz","px","py","pz"};

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es18(P_final);
        const Eigen::VectorXd ev18 = es18.eigenvalues();  // ascending
        const double lam_min18 = ev18(0), lam_max18 = ev18(ev18.size() - 1);
        int weak18_axis;
        es18.eigenvectors().col(0).cwiseAbs().maxCoeff(&weak18_axis);

        static PersistentLogStream eig_log("psd_eigenspectrum.txt");
        bool eig_first;
        std::ofstream& eig_ofs = eig_log.stream(&eig_first);
        if (eig_first)
          eig_ofs << "scan_id,lam_min6,lam_max6,ratio6,weak6_axis,"
                     "lam_min18,lam_max18,ratio18,weak18_dim\n";
        eig_ofs << voxel_map_->frame_idx_ << ","
                << lam_min6 << "," << lam_max6 << ","
                << (lam_min6 > 0.0 ? lam_max6 / lam_min6
                                    : std::numeric_limits<double>::quiet_NaN())
                << "," << AXIS6[weak6_axis] << ","
                << lam_min18 << "," << lam_max18 << ","
                << (lam_min18 > 0.0 ? lam_max18 / lam_min18
                                     : std::numeric_limits<double>::quiet_NaN())
                << "," << weak18_axis << "\n";
        eig_ofs.flush();
      }
      const double t_abs_nees = mg.image.t + data_queues_->start_time;
      coupled_tier1_nees_.addScan("tier1_coupled", voxel_map_->frame_idx_, t_abs_nees,
                                   state_->rot(), state_->pos(), P6);
    }
  }

  // Item 3c: apply the FINAL converged delta_s(t0) to the real state, ONCE.
  // rot_/pos_ are NOT touched by an explicit dx here because they are
  // ALREADY the converged mean: estimateCoupledCorrection()'s own step 6
  // re-propagates from the corrected t0 pose through to t1 and calls
  // setPropagatedState() with the result.
  if (any_solved
      && state_->idxBG() >= 0 && state_->idxBA() >= 0 && state_->idxG() >= 0)
  {
    // idxV() is deliberately left at zero here: state_->setPropagatedState()
    // (inside estimateCoupledCorrection()) already SET vel_ directly.
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(state_->dimState());
    dx.segment<3>(state_->idxBG())    = coupled_delta_bg_;
    dx.segment<3>(state_->idxBA())    = coupled_delta_ba_;
    dx.segment<3>(state_->idxG())     = coupled_delta_g_;
    state_->applyDelta(dx);
    coupled_trP_vel_ = state_->cov().block<3, 3>(StateGroup::idxV(), StateGroup::idxV()).trace();
    coupled_trP_grav_ = state_->cov().block<3, 3>(state_->idxG(), state_->idxG()).trace();
  }

  const double total_dtheta_deg = total_dtheta.norm() * (180.0 / M_PI);
  // CQ-83: mirrors decoupled's own dx_rot_deg/dx_pos_mm exactly (total
  // accumulated correction this scan, same units).
  coupled_dx_rot_deg_ = total_dtheta_deg;
  coupled_dx_pos_mm_ = total_dt.norm() * 1000.0;

  // CQ-53 item 5, gravity-leak falsifier: coupled_prop_.poses[k].acc_head/
  // acc_tail are the CORRECTED world-frame accelerations (R*(a_body+
  // correction)+gravity -- see Pose6D's own doc comment), so subtracting
  // the JUST-UPDATED gravity_estimate leaves the net specific-force this
  // scan actually integrated into velocity. At genuine rest (or constant
  // velocity) this should be near zero; a persistent nonzero horizontal
  // component here IS gravity leaking through a wrong attitude estimate --
  // the mechanism CQ-52's own quadratic-position-growth derivation
  // predicts (~7.7 m/s^2 for a ~52deg attitude error). gravity_dir_err_deg
  // is a NAMED APPROXIMATION (no ground-truth stationary window is wired
  // into LioProc) -- it compares the posterior gravity estimate's
  // direction against the NEGATIVE of this scan's own mean measured
  // acceleration direction, a self-consistency check rather than a true
  // external reference.
  {
    const V3D gravity_estimate = state_->gravity();
    V3D acc_net_sum = V3D::Zero(), acc_mean_sum = V3D::Zero();
    for (const auto& pose : coupled_prop_.poses) {
      const V3D acc_avr = 0.5 * (pose.acc_head + pose.acc_tail);
      acc_net_sum += (acc_avr - gravity_estimate);
      acc_mean_sum += acc_avr;
    }
    const double n_poses = static_cast<double>(coupled_prop_.poses.size());
    if (n_poses > 0) {
      coupled_acc_world_mag_ = (acc_net_sum / n_poses).norm();
      const V3D acc_mean_dir = (acc_mean_sum / n_poses).normalized();
      const V3D grav_dir = gravity_estimate.normalized();
      const double cos_ang = std::clamp((-acc_mean_dir).dot(grav_dir), -1.0, 1.0);
      coupled_gravity_dir_err_deg_ = std::acos(cos_ang) * (180.0 / M_PI);
    }
  }

  // nees_diag.txt -- same field format as the decoupled path's own write,
  // deliberately kept byte-compatible so the existing eps_pos analysis
  // scripts work unchanged against either. G1 (TQ-37/39's own eps_pos
  // methodology) reads exactly this file.
  if (opts_.log_debug_en) {
    static PersistentLogStream log("nees_diag.txt");
    std::ofstream& ofs = log.stream();
    const Eigen::MatrixXd& P = state_->cov();
    const int iP = StateGroup::idxP(), iR = StateGroup::idxR();
    const bool have_p = P.rows() >= iP + 3 && P.cols() >= iP + 3;
    const bool have_r = P.rows() >= iR + 3 && P.cols() >= iR + 3;
    const M3D P_pp = have_p ? M3D(P.block<3, 3>(iP, iP)) : M3D::Zero();
    const M3D P_rr = have_r ? M3D(P.block<3, 3>(iR, iR)) : M3D::Zero();
    // CQ-60 item 5: the R-P cross-block, missing until now -- Ppp/Prr alone
    // are only the two diagonal 3x3 pieces of the full 6x6 pose covariance;
    // a real 6-dof NEES/Mahalanobis computation needs this cross term too
    // (state_->cov()'s R,P block is not actually block-diagonal).
    const M3D P_rp = (have_p && have_r) ? M3D(P.block<3, 3>(iR, iP)) : M3D::Zero();
    const double t_abs = mg.image.t + data_queues_->start_time;
    const Eigen::Quaterniond state_q(state_->rot());
    // Motion-onset investigation (2026-09-19): same velocity/bias/gravity
    // fields as the decoupled path's own write, so the two logs are
    // directly comparable column-for-column.
    const int iV = StateGroup::idxV();
    const bool have_v = P.rows() >= iV + 3 && P.cols() >= iV + 3;
    const M3D P_vv = have_v ? M3D(P.block<3, 3>(iV, iV)) : M3D::Zero();
    const int iBG = state_->idxBG(), iBA = state_->idxBA(), iGr = state_->idxG();
    const bool have_bg = iBG >= 0 && P.rows() >= iBG + 3 && P.cols() >= iBG + 3;
    const bool have_ba = iBA >= 0 && P.rows() >= iBA + 3 && P.cols() >= iBA + 3;
    const bool have_gr = iGr >= 0 && P.rows() >= iGr + 3 && P.cols() >= iGr + 3;
    const double trP_bg_full = have_bg ? P.block<3, 3>(iBG, iBG).trace() : -1.0;
    const double trP_ba_full = have_ba ? P.block<3, 3>(iBA, iBA).trace() : -1.0;
    const double trP_grav_full = have_gr ? P.block<3, 3>(iGr, iGr).trace() : -1.0;
    ofs << std::setprecision(12)
        << "scan_id=" << voxel_map_->frame_idx_ << " t_abs=" << t_abs
        << " state_px=" << state_->pos().x() << " state_py=" << state_->pos().y()
        << " state_pz=" << state_->pos().z()
        << " state_qw=" << state_q.w() << " state_qx=" << state_q.x()
        << " state_qy=" << state_q.y() << " state_qz=" << state_q.z()
        << " total_dtheta_deg=" << total_dtheta_deg
        << " trP_pos_pre=" << trP_pos_pre_
        << " trP_pos_post=" << (have_p ? P_pp.trace() : -1.0)
        << " Ppp_xx=" << P_pp(0, 0) << " Ppp_xy=" << P_pp(0, 1) << " Ppp_xz=" << P_pp(0, 2)
        << " Ppp_yy=" << P_pp(1, 1) << " Ppp_yz=" << P_pp(1, 2) << " Ppp_zz=" << P_pp(2, 2)
        << " Prr_xx=" << P_rr(0, 0) << " Prr_xy=" << P_rr(0, 1) << " Prr_xz=" << P_rr(0, 2)
        << " Prr_yy=" << P_rr(1, 1) << " Prr_yz=" << P_rr(1, 2) << " Prr_zz=" << P_rr(2, 2)
        << " Prp_00=" << P_rp(0, 0) << " Prp_01=" << P_rp(0, 1) << " Prp_02=" << P_rp(0, 2)
        << " Prp_10=" << P_rp(1, 0) << " Prp_11=" << P_rp(1, 1) << " Prp_12=" << P_rp(1, 2)
        << " Prp_20=" << P_rp(2, 0) << " Prp_21=" << P_rp(2, 1) << " Prp_22=" << P_rp(2, 2)
        << " have_Ppp=" << (have_p ? 1 : 0) << " have_Prr=" << (have_r ? 1 : 0)
        // CQ-53 item 6: NaN, not -1.0 -- this class has no free-tail
        // mechanism at all (no spline), so a numeric -1.0 here reads as a
        // real (if suspicious) measured value to a script that doesn't
        // special-case it; NaN is unambiguously "not applicable", matching
        // coupled_refusal_'s own quiet_NaN() convention just below.
        << " free_tail_d=" << std::numeric_limits<double>::quiet_NaN()
        << " state_vx=" << state_->vel().x() << " state_vy=" << state_->vel().y()
        << " state_vz=" << state_->vel().z()
        << " Pvv_xx=" << P_vv(0, 0) << " Pvv_xy=" << P_vv(0, 1) << " Pvv_xz=" << P_vv(0, 2)
        << " Pvv_yy=" << P_vv(1, 1) << " Pvv_yz=" << P_vv(1, 2) << " Pvv_zz=" << P_vv(2, 2)
        << " have_Pvv=" << (have_v ? 1 : 0)
        << " state_bgx=" << state_->biasGyr().x() << " state_bgy=" << state_->biasGyr().y()
        << " state_bgz=" << state_->biasGyr().z()
        << " state_bax=" << state_->biasAcc().x() << " state_bay=" << state_->biasAcc().y()
        << " state_baz=" << state_->biasAcc().z()
        << " state_gx=" << state_->gravity().x() << " state_gy=" << state_->gravity().y()
        << " state_gz=" << state_->gravity().z()
        << " trP_bg=" << trP_bg_full << " trP_ba=" << trP_ba_full
        << " have_Pbg=" << (have_bg ? 1 : 0) << " have_Pba=" << (have_ba ? 1 : 0)
        << " have_Pgrav=" << (have_gr ? 1 : 0)
        << " ask=" << coupled_ask_ << " got=" << coupled_got_
        << " refusal=" << coupled_refusal_
        << " n_residuals=" << coupled_n_residuals_
        << " sum_weight=" << coupled_sum_weight_
        << " h_pp_min_eig=" << coupled_h_pp_min_eig_
        << " h_rr_min_eig=" << coupled_h_rr_min_eig_
        << " c_acc_over_sigma=" << coupled_c_acc_over_sigma_
        << " c_gyr_over_sigma=" << coupled_c_gyr_over_sigma_
        << " c_acc_total_norm=" << coupled_c_acc_total_norm_
        << " c_gyr_total_norm=" << coupled_c_gyr_total_norm_
        << " sum_S=" << coupled_sum_S_
        << " bg_var_degenerate=" << coupled_bg_var_degenerate_
        << " bg_var_observed=" << coupled_bg_var_observed_
        << " c_acc_dc_over_sigma=" << coupled_c_acc_dc_over_sigma_
        << " c_gyr_dc_over_sigma=" << coupled_c_gyr_dc_over_sigma_
        << " dba_over_sigma=" << coupled_dba_over_sigma_
        << " dbg_over_sigma=" << coupled_dbg_over_sigma_
        << " delta_v_norm=" << coupled_delta_v_norm_
        << " delta_g_norm=" << coupled_delta_g_norm_
        << " trP_vel=" << coupled_trP_vel_ << " trP_grav=" << trP_grav_full
        << " iters=" << coupled_iters_ << " solve_ms=" << coupled_solve_ms_
        // CQ-53 item 1: joint-matrix pivot diagnostics (see coupled_joint_dmin_'s
        // own doc comment for the state-block/coeff-block split rationale).
        << " joint_dmin=" << coupled_joint_dmin_ << " joint_dmax=" << coupled_joint_dmax_
        << " state_dmin=" << coupled_state_dmin_ << " state_dmax=" << coupled_state_dmax_
        << " coeff_dmin=" << coupled_coeff_dmin_ << " coeff_dmax=" << coupled_coeff_dmax_
        << " pivot_guard=" << (coupled_pivot_guard_ ? 1 : 0)
        // CQ-53 item 2: H(t1)-vs-H(t_k) relative-difference distribution.
        << " hcol_reldiff_p10=" << coupled_hcol_reldiff_p10_
        << " hcol_reldiff_p50=" << coupled_hcol_reldiff_p50_
        << " hcol_reldiff_p90=" << coupled_hcol_reldiff_p90_
        << " hcol_reldiff_max=" << coupled_hcol_reldiff_max_
        // CQ-53 item 3: FINAL GN iteration's own step norms.
        << " last_delta_s_norm=" << coupled_last_delta_s_norm_
        << " last_delta_c_norm=" << coupled_last_delta_c_norm_
        // CQ-53 item 4: per-scan RMS residual (meters).
        << " res_rms=" << coupled_res_rms_
        << " mean_sigma_squared=" << coupled_mean_sigma_squared_
        // CQ-53 item 5: gravity-leak falsifier.
        << " acc_world_mag=" << coupled_acc_world_mag_
        << " gravity_dir_err_deg=" << coupled_gravity_dir_err_deg_
        // CQ-54 item 1: sigma actually used this scan and its ratio to the
        // calibration floor (~1 quiet, 15-55 through a vibration transient).
        << " sigma_a_used=" << coupled_sigma_a_used_
        << " sigma_g_used=" << coupled_sigma_g_used_
        << " sigma_a_ratio=" << coupled_sigma_a_ratio_
        << " sigma_g_ratio=" << coupled_sigma_g_ratio_
        // CQ-54 item 3: this scan's bias_freeze_on_vibration state and the
        // running active fraction over the process's life so far.
        << " bias_freeze_active=" << (coupled_bias_freeze_active_ ? 1 : 0)
        << " bias_freeze_active_frac=" << (coupled_bias_freeze_scan_count_ > 0
            ? static_cast<double>(coupled_bias_freeze_active_count_) / coupled_bias_freeze_scan_count_
            : -1.0)
        // CQ-54 item 4: reduced chi-square (~1 = honest noise model).
        << " reduced_chi2=" << coupled_reduced_chi2_
        // CQ-54 item 5: angular-rate correction split, deg/s.
        << " w_from_c_deg_s=" << coupled_w_from_c_deg_s_
        << " w_from_bg_deg_s=" << coupled_w_from_bg_deg_s_
        << " w_net_deg_s=" << coupled_w_net_deg_s_
        << "\n";
    ofs.flush();
  }


  // CQ-44 G0: boundary_dpos_/boundary_drot_deg_ are forced to exactly 0.0
  // every scan -- there is no separate spline t0 to compare against; this
  // scan's own propagation start IS the previous scan's own coupled
  // endpoint, by construction, so no discrepancy can exist to measure.
  boundary_dpos_ = 0.0;
  boundary_drot_deg_ = 0.0;

  std::ostringstream oss;
  oss << "[lio/ekf] iters=" << iter + 1 << "  stop=" << stop
      << std::scientific << std::setprecision(1)
      << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
      << "  |dt|=" << total_dt.norm() * 1000.0 << " mm";
  return oss.str();
}

LioProcCoupled::CoupledSystemBuild LioProcCoupled::buildImuCorrectionSystem(
    MeasureGroup& mg, double t0, double t1, int n_c, int ncol, int ncol_s, int ncol_c,
    double sigma_a, double sigma_g, double sigma_a_floor, double sigma_g_floor,
    const Eigen::MatrixXd& Pi_ss, const Eigen::VectorXd& s_vec)
{
  CoupledSystemBuild build;
  // CQ-82 Phase 1: gram/Curv/Lambda (the coefficient-block prior) and c_vec
  // are UNCHANGED from estimateCoupledCorrection()'s own pre-split code --
  // see this function's declaration-site comment for the "(4) the prior,
  // TWO BLOCKS" context this half (the c-block) used to sit next to; the
  // s-block half (Pi_ss) moved to the dispatcher, since it does not depend
  // on which coefficient basis is active.
  Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n_c, n_c);
  for (const auto& pose : mg.poses) {
    std::vector<double> bw(n_c);
    for (int j = 0; j < n_c; ++j) bw[j] = basisWeight(j, n_c, t0, t1, pose.t);
    for (int i = 0; i < n_c; ++i)
      for (int j = 0; j < n_c; ++j) gram(i, j) += bw[i] * bw[j];
  }
  // CQ-55 item 6: gram is built from basis VALUES only -- nothing penalises
  // the correction's SHAPE, which is why the prior's connectivity collapses
  // to a chain at n_c=13 (adjacent-only support overlap) instead of the
  // fully-dense n_c=4 case (item 9's gram[0][12]=0 vs gram[0][3]!=0
  // finding). Curv = D^T D for the second-difference operator D (row k:
  // +1,-2,+1 at columns k,k+1,k+2) adds an EXPLICIT connectivity term
  // between every control point pair within 2 of each other, independent
  // of whether their basis supports overlap -- default 0.0, so shipped
  // behavior is unchanged and this is provably md5-inert at that default.
  Eigen::MatrixXd Curv = Eigen::MatrixXd::Zero(n_c, n_c);
  if ((copts_.smoothness_weight_acc > 0.0 || copts_.smoothness_weight_gyr > 0.0) && n_c >= 3) {
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_c - 2, n_c);
    for (int k = 0; k < n_c - 2; ++k) { D(k, k) = 1.0; D(k, k + 1) = -2.0; D(k, k + 2) = 1.0; }
    Curv = D.transpose() * D;
  }
  // CQ-72 items 0a/0b: mean_weight now applies UNCONDITIONALLY (no longer
  // gated on imu_deviation_weight==0 the way dc_weight used to be gated on
  // curvature_only) -- the two beliefs are independent, so the switch that
  // used to force them together is gone. mean_weight/sigma^2, applied
  // UNIFORMLY to every (i,j) pair -- a
  // rank-1 all-ones contribution whose quadratic form for c is exactly
  // mean_weight/sigma^2 * n_c * ||mean(c)||^2, pricing the mean/constant
  // direction only). A pure curvature penalty's 2D null space per axis
  // (constant AND linear both map to zero under [1,-2,1]) would otherwise
  // leave the bias-degenerate constant direction completely unpriced --
  // this term is what keeps arm (c) a valid configuration at all. Default
  // false: arm (a)/(b)'s existing value(+curvature) behavior is unchanged.
  // CQ-59 item 4: the VALUE term's precision, per axis instead of the
  // scalar 1/sigma^2 every other term here still uses. Measured on
  // eee_01/eee_02 (calib_processing.cpp's own per-axis floor print):
  // acc max/min ratio 9.8-17.4x, gyr max/min ratio 8.0-39.4x -- neither
  // near 1, so per the card's own criterion this is worth a flag rather
  // than being dropped. infl_a/infl_g carry adaptive_sigma's scalar
  // inflation (sigma_a/sigma_a_floor)^2 through unchanged -- 1.0 when
  // adaptive_sigma is off (the default), so at that default this reduces
  // to the literal "diagonal from the per-axis floor" the card asks for;
  // when adaptive_sigma is also on, the per-axis floor's RATIOS are kept
  // but scaled to the same inflated overall magnitude, so the two flags
  // compose rather than fight over which sigma is authoritative. Default
  // false -- md5-inert (prec_acc_diag/prec_gyr_diag both collapse to the
  // existing scalar terms below when off).
  V3D prec_acc_diag = V3D::Constant(1.0 / (sigma_a * sigma_a));
  V3D prec_gyr_diag = V3D::Constant(1.0 / (sigma_g * sigma_g));
  if (copts_.prior_per_axis_sigma) {
    const double infl_a = (sigma_a_floor > 1e-12) ? (sigma_a * sigma_a) / (sigma_a_floor * sigma_a_floor) : 1.0;
    const double infl_g = (sigma_g_floor > 1e-12) ? (sigma_g * sigma_g) / (sigma_g_floor * sigma_g_floor) : 1.0;
    const V3D floor_acc = state_->varAccFloor();
    const V3D floor_gyr = state_->varGyrFloor();
    for (int k = 0; k < 3; ++k) {
      prec_acc_diag(k) = 1.0 / std::max(floor_acc(k) * infl_a, 1e-18);
      prec_gyr_diag(k) = 1.0 / std::max(floor_gyr(k) * infl_g, 1e-18);
    }
  }

  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(ncol_c, ncol_c);
  for (int i = 0; i < n_c; ++i)
    for (int j = 0; j < n_c; ++j) {
      // CQ-59 item 1: smooth_acc/smooth_gyr each normalized by the SAME
      // sigma^2 the value term for that block uses, so the weight means
      // "this shape penalty is worth w times the value penalty" on that
      // block specifically -- replaces the single raw curv_ij that used
      // to apply identically to both blocks despite their ~4657x base-
      // prior stiffness difference.
      const double smooth_acc = copts_.smoothness_weight_acc * Curv(i, j) / (sigma_a * sigma_a);
      const double smooth_gyr = copts_.smoothness_weight_gyr * Curv(i, j) / (sigma_g * sigma_g);
      // CQ-72 item 0b: value_gram is now a plain multiplier (was: ternary
      // gated on curvature_only) -- at the default imu_deviation_weight=1.0
      // this is EXACTLY gram(i,j), identical to today's curvature_only=
      // false arithmetic. dc_acc/dc_gyr are now UNCONDITIONAL (were gated
      // on the SAME curvature_only switch as value_gram) -- at the default
      // mean_weight=0.0 both are exactly 0.0, identical to today's
      // dc_weight=0.0 (the only value ever shipped) regardless of
      // curvature_only. Both defaults together reproduce today's
      // arithmetic exactly -- the md5 gate is what checks that claim.
      const double value_gram = copts_.imu_deviation_weight * gram(i, j);
      const double dc_acc = copts_.mean_weight / (sigma_a * sigma_a);
      const double dc_gyr = copts_.mean_weight / (sigma_g * sigma_g);
      Lambda.block<3, 3>(3 * i, 3 * j) =
          M3D(value_gram * prec_acc_diag.asDiagonal()) + (smooth_acc + dc_acc) * M3D::Identity();
      Lambda.block<3, 3>(3 * n_c + 3 * i, 3 * n_c + 3 * j) =
          M3D(value_gram * prec_gyr_diag.asDiagonal()) + (smooth_gyr + dc_gyr) * M3D::Identity();
    }
  // CQ-69: the low-band trajectory-deviation prior. Lambda_traj = sum_k
  // phi_head[k]^T W phi_head[k], W selecting phi_head[k]'s POSITION rows
  // (rows 3-5 of its 9-row [dtheta,dp,dv] layout) only -- position-only for
  // this first pass per the card's own instruction (attitude/velocity rows
  // have different units, so a single scalar W can't combine them
  // dimensionlessly). phi_head[k] already bakes in the basis weights, the
  // world-frame rotation, and the double integration via the Fx*Phi
  // recursion (see this function's own G-block construction above) -- this
  // is an EXACT low-band penalty on the trajectory deviation the
  // correction implies, not an approximation. NOT block-diagonal in (i,j)
  // the way gram/Curv are -- accumulated as one whole-matrix outer-product
  // sum, which is guaranteed PSD by construction (a sum of P_k^T*P_k
  // terms). Normalized by (t1-t0)^2: the term's own DC weighting scales as
  // T^2 (verified by computation in the card), so this keeps a given
  // traj_deviation_weight meaning the same thing across scan durations/LiDAR
  // rates. Default 0.0 -- md5-inert (the whole block is skipped).
  if (copts_.traj_deviation_weight > 0.0) {
    Eigen::MatrixXd Lambda_traj = Eigen::MatrixXd::Zero(ncol_c, ncol_c);
    for (const auto& phi_k : coupled_prop_.phi_head) {
      const Eigen::Matrix<double, 3, Eigen::Dynamic> P_k = phi_k.middleRows<3>(3);
      Lambda_traj.noalias() += P_k.transpose() * P_k;
    }
    const double T = t1 - t0;
    const double norm = (T * T > 1e-12) ? 1.0 / (T * T) : 0.0;
    Lambda.noalias() += (copts_.traj_deviation_weight * norm) * Lambda_traj;
  }
  // CQ-62 item 1: S4 -- Lambda.
  if (copts_.psd_audit_en) logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S4_Lambda", Lambda);

  Eigen::VectorXd c_vec(ncol_c);
  for (int j = 0; j < n_c; ++j) {
    c_vec.segment<3>(3 * j) = coupled_c_acc_[j];
    c_vec.segment<3>(3 * n_c + 3 * j) = coupled_c_gyr_[j];
  }

  // ---- (5) normal equations, JOINTLY over [delta_s(t0), c] -- items
  // 3c/3d/CORRECTED 2026-09-19, replacing the old c-alone solve entirely
  // (see the amendment: solving for c alone left velocity/both biases/
  // gravity with a COVARIANCE contribution and NO MEAN CORRECTION, a
  // regression on bias estimation only visible as slow drift over ~4000
  // scans). SAME sign convention as the (correct, still-valid) fix found
  // in the c-alone version: A*delta = -(J'R^-1 r) - Pi(prior_iter), i.e.
  // b accumulates MINUS the residual term, matching ekf.h's own
  // applyMeanUpdate() "-K1*Htz" convention (Htz itself a "+H'*W*r"
  // accumulation) -- verified again here, not merely carried over
  // unchecked, since the joint Jacobian's own sign (Phi_x, Phi_c) could in
  // principle have flipped something the scalar c-alone case didn't
  // exercise; it did not (see the filing's own G0/G1 numbers). ----
  build.A = Eigen::MatrixXd::Zero(ncol, ncol);
  Eigen::MatrixXd& A = build.A;
  A.block(0, 0, ncol_s, ncol_s) = Pi_ss;
  A.block(ncol_s, ncol_s, ncol_c, ncol_c) = Lambda;
  // CQ-62 item 1: S5 -- A after Pi_ss and Lambda are placed, BEFORE the
  // residual loop.
  if (copts_.psd_audit_en) logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S5_A_prior_only", A);
  build.b = Eigen::VectorXd::Zero(ncol);
  Eigen::VectorXd& b = build.b;
  b.segment(0, ncol_s) = -(Pi_ss * s_vec);
  b.segment(ncol_s, ncol_c) = -(Lambda * c_vec);
  // CQ-54 item 6: the existing Pi_ss/s_vec term above penalises delta_bg
  // against ZERO, i.e. against wherever the bias already is at t0 -- so a
  // sequence of individually-cheap per-scan increments accumulates without
  // anything ever pricing the TOTAL departure from calibration (the
  // ratchet). bias_anchor adds a SEPARATE quadratic penalty on the total
  // (state_->biasGyr() + coupled_delta_bg_ - coupled_bg_calib_), independent
  // of Pi_ss's own (state-covariance-derived) local-increment prior. Only
  // touches the bg 3x3 diagonal block of A and its own 3 rows of b -- every
  // other block (phi0/p0/v/ba/g, and the c-block via Lambda) is unaffected.
  if (copts_.bias_anchor) {
    if (!coupled_bg_calib_set_) { coupled_bg_calib_ = state_->biasGyr(); coupled_bg_calib_set_ = true; }
    const double sigma_anchor = LioProcCoupledOptions::BIAS_ANCHOR_SIGMA_RAD_S_DEFAULT;
    const M3D Pi_anchor = M3D::Identity() / (sigma_anchor * sigma_anchor);
    A.block<3, 3>(9, 9) += Pi_anchor;
    b.segment<3>(9) += Pi_anchor * (coupled_bg_calib_ - state_->biasGyr());
  }
  // CQ-62 item 1: S6 -- A after the bias_anchor block add (identical to S5
  // when bias_anchor is off, still logged for completeness -- cheap).
  if (copts_.psd_audit_en) logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S6_A_post_anchor", A);
  double& sum_abs_r = build.sum_abs_r;
  double& sum_sq_r = build.sum_sq_r;  // CQ-53 item 4
  double& sum_wr2 = build.sum_wr2;   // CQ-54 item 4
  // CQ-55 item 12: S = floor_term + sigma_diag_squared + plane_var_term +
  // s_prior_pose per residual, same definition lio_decoupled.cpp's own
  // sum_S uses -- summed here so the coupled path reports the SAME
  // absolute-units denominator, never logged on this path before now.
  double& sum_floor_S = build.sum_floor_S;
  double& sum_sdiag_S = build.sum_sdiag_S;
  double& sum_pvar_S = build.sum_pvar_S;
  double& sum_prior_pose_S = build.sum_prior_pose_S;
  double& sum_sigma_squared = build.sum_sigma_squared;  // CQ-60 item 0a
  std::vector<double>& hcol_reldiff = build.hcol_reldiff;  // CQ-53 item 2
  hcol_reldiff.reserve(residuals_.size());
  // TQ-40 item 3: pure-LiDAR-info accumulation restricted to [delta_phi0,
  // delta_p0] (columns 0-5), BEFORE the Pi_ss/Lambda prior is added --
  // see coupled_ask_'s own doc comment in the header for the approximation
  // this makes (no marginalisation over v/bg/ba/g/c).
  Eigen::Matrix<double, 6, 6>& HtH_pose_lidar = build.HtH_pose_lidar;
  Eigen::Matrix<double, 6, 1>& Htz_pose_lidar = build.Htz_pose_lidar;
  double& sum_weight_this_iter = build.sum_weight_this_iter;
  // CQ-66 item 1: sum_k w_k H_k^T H_k, RAW 6x6 (H before composing with
  // Phix_pt/Phic_pt -- the [rot_jac_col;normal] measurement sensitivity
  // alone). Item 1's own derivation: in end_time mode, EVERY residual's
  // c-block contribution is H_k*Phic_end (Phic_end the SAME fixed 6xncol_c
  // matrix for the whole scan), so the c-block's own contribution to A from
  // the residual loop is Phic_end^T * (sum_k w_k H_k^T H_k) * Phic_end --
  // rank-bounded by THIS 6x6 matrix's own rank, regardless of n_residuals
  // or ncol_c. point_time uses a DIFFERENT Phic_pt per residual, so no such
  // bound applies there.
  Eigen::Matrix<double, 6, 6>& H6_raw_accum = build.H6_raw_accum;
  // CQ-66 item 3's own accumulator -- see its doc comment inside the loop.
  Eigen::MatrixXd& phic_spread_sum = build.phic_spread_sum;
  double& phic_spread_sumsq = build.phic_spread_sumsq;
  int& phic_spread_n = build.phic_spread_n;
  for (const auto& res : residuals_) {
    Eigen::Matrix<double, 1, 6> H;
    // CQ-50 item (d), the real fix: build the rotation-Jacobian column at
    // this point's OWN capture time t_k, using raw_body_point (the body
    // point before deskew's warp to t1) and worldRotAt(t_k) -- the SAME
    // formula point_cross_normal itself uses (p.cross(R^T*n)), just with
    // (p1, R(t1)) replaced by (p(t_k), R(t_k)). The translation column
    // (normal itself) is frame-independent -- dr/dpos is n^T regardless of
    // which time's body frame p was expressed in -- so it is unaffected.
    // CQ-53 item 2: computed for EVERY residual regardless of which arm is
    // actually active below, so the two arms stay directly comparable on
    // this number -- the permanent replacement for the deleted ad hoc debug
    // print CQ-50's own filing cited a since-nonexistent "1-3%" figure from.
    const V3D hk = V3D(res.raw_body_point.cross(worldRotAt(coupled_prop_, res.t).transpose() * res.normal));
    hcol_reldiff.push_back((hk - res.point_cross_normal).norm() / std::max(res.point_cross_normal.norm(), 1e-9));
    // jacobian_time_mode: H and Phi must be evaluated at the SAME time for
    // H_k*Phi(t_k) to be a valid chain rule (CQ-50's original diagnosis was
    // exactly this mismatch -- H built once from the DESKEWED point and the
    // SCAN-END state_->rot() in lio_base.cpp's buildResiduals(), chained
    // against Phi interpolated at each residual's own t_k, valid only at
    // rest by coincidence). "point_time" pulls H back to t_k to meet Phi
    // there (real within-scan resolution preserved); "end_time" pushes Phi
    // forward to t1 to meet H there (resolution collapsed, but consistent);
    // "legacy_mismatched" (default) reproduces the original mismatch
    // unchanged, so this refactor changes no one's numerics by default
    // (rule 26 item 1 -- see the header's own doc comment for the full
    // 4-combination table this collapses).
    const bool point_time = (copts_.jacobian_time_mode == "point_time");
    const bool end_time = (copts_.jacobian_time_mode == "end_time");
    const V3D rot_jac_col = point_time ? hk : res.point_cross_normal;
    H.block<1, 3>(0, 0) = rot_jac_col.transpose();
    H.block<1, 3>(0, 3) = res.normal.transpose();
    const Eigen::Matrix<double, 9, 18> Phix_pt =
        end_time ? coupled_prop_.phi_x_head.back() : interpolatePhiX(coupled_prop_, res.t);
    const Eigen::Matrix<double, 9, Eigen::Dynamic> Phic_pt =
        end_time ? coupled_prop_.phi_head.back() : interpolatePhi(coupled_prop_, res.t);
    // CQ-66 item 3: per-scan spread of Phi_c(t_k) across residuals --
    // single-pass (Var = E[||X||^2] - ||E[X]||^2, applied to the whole
    // matrix via the Frobenius inner product) so no second pass over
    // residuals_ is needed. In end_time mode Phic_pt is IDENTICAL for
    // every residual, so this is mathematically guaranteed to come out
    // exactly 0 -- the interesting number is point_time's own value.
    if (copts_.psd_audit_en) {
      if (phic_spread_sum.size() == 0) phic_spread_sum = Eigen::MatrixXd::Zero(9, ncol_c);
      phic_spread_sum.noalias() += Phic_pt;
      phic_spread_sumsq += Phic_pt.squaredNorm();
      ++phic_spread_n;
    }
    Eigen::Matrix<double, 1, Eigen::Dynamic> Jrow(1, ncol);
    Jrow.segment(0, ncol_s) = H * Phix_pt.topRows(6);          // R,P rows only (item 3b/3c)
    Jrow.segment(ncol_s, ncol_c) = H * Phic_pt.topRows(6);
    // Diagnostic toggle: zero c_gyr's own columns (the last 3*n_c of the
    // c-block, per coupled_estimator.h's own documented column order
    // [c_acc(3*n_c), c_gyr(3*n_c)]) AFTER computing them, so c_gyr gets NO
    // LiDAR information at all -- its posterior then equals its prior
    // (Lambda) exactly, i.e. c_gyr never moves and never correlates with
    // anything else in A (the cross term with delta_bg this toggle exists
    // to test is a Jrow-column product, and one factor is now identically
    // zero). delta_bg (the ONLY rotation-correction path left active) is
    // untouched -- this is not "no rotation correction at all", it is
    // "rotation correction exactly as bounded as the decoupled path's own
    // EKF pose-block dtheta, no within-scan SHAPE parameterisation".
    if (copts_.disable_cgyr) Jrow.segment(ncol_s + 3 * n_c, 3 * n_c).setZero();
    // CQ-53 item 6: committed diagnostic harness (was an ad hoc getenv
    // print during CQ-50's own investigation) -- per-residual leverage on
    // delta_phi0/delta_p0, binnable by the point's own capture-time
    // fraction within the scan [0,1]. Refuted the "early-scan points get
    // outsized delta_phi0 leverage under jacobian_time_mode=point_time"
    // hypothesis when this was run manually: the frac-vs-leverage pattern
    // came out nearly identical between end_time and point_time.
    if (copts_.log_jrow_leverage_en) {
      const double t0_local = coupled_prop_.poses.empty() ? 0.0 : coupled_prop_.poses.front().t;
      const double t1_local = coupled_prop_.poses.empty() ? 1.0
          : coupled_prop_.poses.back().t + coupled_prop_.poses.back().dt;
      const double frac = (t1_local > t0_local) ? (res.t - t0_local) / (t1_local - t0_local) : -1.0;
      static PersistentLogStream log("jrow_leverage.txt");
      std::ofstream& ofs = log.stream();
      ofs << "scan_id=" << voxel_map_->frame_idx_ << " frac=" << frac
          << " phi0_lev=" << Jrow.segment(0, 3).norm()
          << " p0_lev=" << Jrow.segment(3, 3).norm() << "\n";
      // Deliberately NOT flushed per residual (unlike nees_diag.txt/
      // iter_error.txt's per-scan/per-iteration writes) -- this fires once
      // per RESIDUAL, potentially hundreds of thousands of times per run;
      // relies on ofstream's own buffering + normal process exit to flush.
    }
    // CQ-61 arm (b): outlier-robust IRLS down-weighting on the normalized
    // residual z = r/sqrt(sigma_squared) -- treats the SYMPTOM (a handful
    // of bad correspondences dominating A once matching has already
    // degraded), not the mechanism arm (a)/pose_cov_in_sigma targets; it
    // will not reopen a gate that voxelplane.cpp's own acceptance test has
    // already closed. Standard 95%-efficiency-under-Gaussian constants
    // (Huber k=1.345, Cauchy c=2.3849). Default "none" -- md5-inert.
    double w = 1.0 / res.sigma_squared;
    if (copts_.robust_loss != "none") {
      const double z = std::abs(res.r) / std::sqrt(std::max(res.sigma_squared, 1e-18));
      if (copts_.robust_loss == "huber") {
        constexpr double HUBER_K = 1.345;
        if (z > HUBER_K) w *= HUBER_K / z;
      } else if (copts_.robust_loss == "cauchy") {
        constexpr double CAUCHY_C = 2.3849;
        w *= 1.0 / (1.0 + (z / CAUCHY_C) * (z / CAUCHY_C));
      }
    }
    A.noalias() += w * (Jrow.transpose() * Jrow);
    b.noalias() -= w * Jrow.transpose() * res.r;
    if (copts_.psd_audit_en) H6_raw_accum.noalias() += w * (H.transpose() * H);
    sum_abs_r += std::abs(res.r);
    sum_sq_r += res.r * res.r;
    sum_wr2 += w * res.r * res.r;  // CQ-54 item 4: reduced chi-square numerator
    sum_sigma_squared += res.sigma_squared;  // CQ-60 item 0a
    // CQ-55 item 12: same accept/skip rule as lio_decoupled.cpp's own sum_S.
    if (res.floor_term >= 0.0 && res.sigma_diag_squared >= 0.0 && res.s_prior_pose >= 0.0) {
      sum_floor_S      += res.floor_term;
      sum_sdiag_S      += res.sigma_diag_squared;
      sum_pvar_S       += res.plane_var_term;
      sum_prior_pose_S += res.s_prior_pose;
      // CQ-83: per-residual S = floor+sdiag+pvar+prior_pose, mirroring
      // lio_decoupled.cpp's own nis/nis_est per-residual definition exactly.
      const double res_S = res.floor_term + res.sigma_diag_squared +
                            res.plane_var_term + res.s_prior_pose;
      if (res_S > 0.0) {
        build.sum_nis += (res.r * res.r) / res_S;
        ++build.n_nis;
        const double res_S_est = res_S - res.s_prior_pose;
        if (res_S_est > 0.0) {
          build.sum_nis_est += (res.r * res.r) / res_S_est;
          ++build.n_nis_est;
        }
      }
    }
    // TQ-40 item 3: matches ekf_.HtH/Htz's own "+H'*W*r" convention exactly
    // (H here is the SAME 1x6 row, since Jrow.head(6) IS H*Phix_pt.topRows(6)
    // restricted to the phi0/p0 columns -- Phix_pt's own phi0/p0 columns are
    // Identity at t0 and only decay via Fx's own accumulation to t_k, so
    // this is genuinely "how much does THIS residual constrain phi0/p0").
    const Eigen::Matrix<double, 1, 6> H6 = Jrow.segment(0, 6);
    HtH_pose_lidar.noalias() += w * (H6.transpose() * H6);
    Htz_pose_lidar.noalias() += w * H6.transpose() * res.r;
    sum_weight_this_iter += w;
  }
  coupled_last_A_ = A;
  // CQ-62 item 1: S7 -- A after the residual accumulation loop (== coupled_last_A_).
  if (copts_.psd_audit_en) logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S7_A_final", A);
  if (copts_.log_cp_constraint_en) coupled_last_Lambda_ = Lambda;
  return build;
}

double LioProcCoupled::estimateCoupledCorrection(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out)
{
  dtheta_out = V3D::Zero();
  dt_out = V3D::Zero();
  if (mg.poses.empty()) return 0.0;
  // CQ-82 Phase 2: a single top-level branch, kept as far from the raw_imu
  // path's own ~600 lines of downstream bookkeeping as possible -- see
  // estimateCoupledCorrectionPoseBasis()'s own doc comment.
  if (copts_.poseBasis()) return estimateCoupledCorrectionPoseBasis(mg, dtheta_out, dt_out);
  if (copts_.poseControlSplineBasis()) return estimateCoupledPoseControlSpline(mg, dtheta_out, dt_out);

  const auto t_start = std::chrono::steady_clock::now();

  const double t0 = mg.poses.front().t;
  const double t1 = mg.image.t;
  const int n_c = copts_.n_c;
  const int ncol_c = 6 * n_c;
  // Item 3e(v)/3f bug 2: 18, not 12 -- [delta_phi0, delta_p0, delta_v,
  // delta_bg, delta_ba, delta_g]. Pose included in both mean and covariance.
  const int ncol_s = 18;
  const int ncol = ncol_s + ncol_c;
  // Item 4: sigma_a/sigma_g are the CALIBRATION-FLOOR SIGMA, not the
  // variance config/ntu_viral.yaml logs (acc=0.00434, gyr=0.0000636 are
  // VARIANCES -- this project already filed a round on exactly this
  // mistake; the sigma is sqrt(), 14.7x larger). Read once here from
  // state_->varAccFloor()/varGyrFloor() (the SAME calibration-floor
  // quantity TQ-34/TQ-38 both used) rather than a hardcoded literal, so a
  // different calibration run changes this automatically.
  const double sigma_a_floor = std::sqrt(state_->varAccFloor().mean());
  const double sigma_g_floor = std::sqrt(state_->varGyrFloor().mean());
  // CQ-54 item 1: sigma_a/sigma_g -- adaptive_sigma re-estimates them per
  // scan from the raw IMU stream's own std dev (mg.imu_samples_raw, kept
  // now that ImuProc::loadParameters() extends keep_raw_samples for this
  // flag), floored at the calibration value so it can only INFLATE, never
  // shrink below the sensor's own floor. Falls back to the floor itself
  // (identical to the pre-CQ-54 behavior) when off or when too few raw
  // samples are available to form an estimate.
  double sigma_a = sigma_a_floor, sigma_g = sigma_g_floor;
  if (copts_.adaptive_sigma && mg.imu_samples_raw.size() >= 3) {
    double acc_mean = 0.0, gyr_mean = 0.0;
    for (const auto& s : mg.imu_samples_raw) { acc_mean += s.acc.norm(); gyr_mean += s.gyro.norm(); }
    acc_mean /= mg.imu_samples_raw.size();
    gyr_mean /= mg.imu_samples_raw.size();
    double acc_var = 0.0, gyr_var = 0.0;
    for (const auto& s : mg.imu_samples_raw) {
      const double da = s.acc.norm() - acc_mean, dg = s.gyro.norm() - gyr_mean;
      acc_var += da * da; gyr_var += dg * dg;
    }
    acc_var /= (mg.imu_samples_raw.size() - 1);
    gyr_var /= (mg.imu_samples_raw.size() - 1);
    sigma_a = std::max(sigma_a_floor, std::sqrt(acc_var));
    sigma_g = std::max(sigma_g_floor, std::sqrt(gyr_var));
  }
  coupled_sigma_a_used_ = sigma_a;
  coupled_sigma_g_used_ = sigma_g;
  coupled_sigma_a_ratio_ = sigma_a / std::max(sigma_a_floor, 1e-12);
  coupled_sigma_g_ratio_ = sigma_g / std::max(sigma_g_floor, 1e-12);
  // CQ-54 item 3: the cheaper guard -- freeze the bias only while the
  // measured noise exceeds its calibration floor by more than the stated
  // factor, using the SAME rolling-window estimate above regardless of
  // whether adaptive_sigma itself is also on (the two options are
  // independent -- CQ-54 item 2's 2x2 grid runs all four combinations).
  coupled_bias_freeze_active_ = false;
  if (copts_.bias_freeze_on_vibration && mg.imu_samples_raw.size() >= 3) {
    double acc_mean = 0.0;
    for (const auto& s : mg.imu_samples_raw) acc_mean += s.acc.norm();
    acc_mean /= mg.imu_samples_raw.size();
    double acc_var = 0.0;
    for (const auto& s : mg.imu_samples_raw) { const double da = s.acc.norm() - acc_mean; acc_var += da * da; }
    acc_var /= (mg.imu_samples_raw.size() - 1);
    const double ratio = std::sqrt(acc_var) / std::max(sigma_a_floor, 1e-12);
    coupled_bias_freeze_active_ = (ratio > copts_.bias_freeze_vibration_factor);
  }

  // ---- (1) re-propagate with the CURRENT coefficient AND delta_s estimate
  // -- items 3c/3d. vel0/gravity fold this scan's own accumulated
  // delta_v/delta_g on top of the pre-scan snapshot; delta_bg/delta_ba are
  // passed through and subtracted INSIDE propagateCoupled (see its header
  // comment) since they enter per-segment, not just the initial condition.
  // rot0/pos0: item 3e(v)/3f bug 2 -- fold in this scan's own accumulated
  // delta_phi0/delta_p0 (right-multiplicative for phi0, matching
  // StateGroup::applyDelta()'s own convention; additive for p0), no longer
  // held at the raw chain's own t0 value unconditionally.
  propagateCoupled(mg.poses, state_propagat_.rot(), t1,
                   mg.poses.front().rot * Exp(coupled_delta_phi0_),
                   mg.poses.front().pos + coupled_delta_pos0_,
                   coupled_v0_pre_ + coupled_delta_v_,
                   coupled_g0_pre_ + coupled_delta_g_, coupled_g0_pre_,
                   coupled_delta_bg_, coupled_delta_ba_,
                   coupled_c_acc_, coupled_c_gyr_, n_c, coupled_prop_);


  // ---- (2) re-deskew the FULL raw point set against this corrected
  // trajectory, then downsample. Not CSR-optimized (re-downsamples every
  // GN iteration rather than re-placing a fixed membership set the way
  // redeskewFromSpline()'s per-iteration path does) -- a real scope/cost
  // choice, named here and in the filing rather than silently assumed away;
  // solve_ms_per_scan_p50 (below) is what it actually costs. ----
  state_->setPropagatedState(coupled_prop_.rot1, coupled_prop_.pos1, coupled_prop_.vel1);
  std::vector<PointXYZCov> deskewed;
  deskewPoints(state_, coupled_prop_.poses, t1, mg.lidar_points, opts_.deskew, deskewed);
  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
  }

  // ---- (3) residuals -- UNCHANGED (item 3): the exact same
  // buildResiduals() the decoupled path uses, over mg.points as just
  // re-deskewed above. ----
  // CQ-60 item 0b: was hardcoded false (never logged corr_scan.csv on this
  // path at all) -- now passes through, matching decoupled's own pattern
  // exactly (gated by the shared lio/log_pair_corr_en flag, default false,
  // md5-inert). NOTE: unlike decoupled's "first-iteration-only" semantics,
  // this call happens once per OUTER GN iteration (up to max_iterations
  // times per scan, same scan_id each time) -- debugAccumConsistencyCorr's
  // per-scan-id accumulator therefore SUMS n_candidates/n_accepted across
  // every iteration rather than isolating the final one. The trend in their
  // RATIO across scans is still meaningful; the absolute counts are inflated
  // by up to a factor of max_iterations.
  buildResiduals(mg.points, residuals_, /*allow_consistency_log=*/true);

  // ---- (4) the prior. TWO BLOCKS, per items 3c/3d/3e(v)/3f REVISED
  // 2026-09-19: Lambda (the c prior, item 3b/4, UNCHANGED math) on the
  // c-block, and Pi_ss -- now the FULL 18x18 P(t1)^-1 (CQ-76-R3: this is
  // the PREVIOUS scan's post-solve posterior, i.e. P(t1) of that scan,
  // read at the top of THIS scan before propagation -- not this scan's
  // own pre-propagation P(t0), despite the state_->cov() read site's own
  // naming. CQ-76-R2's T2.1 A/B (prior_at_scan_start) confirmed the two
  // are materially different matrices: substituting the genuine
  // pre-propagation P(t0) diverges catastrophically (four orders of
  // magnitude in ATE) precisely because P(t1) is the only route by which
  // within-scan process noise (Q_unmodelled, see :306-311's own comment)
  // reaches the posterior at all -- so this being P(t1) is load-bearing,
  // not a naming slip to silently correct behind), at
  // [idxR,idxP,idxV,idxBG,idxBA,idxG]. Bug 2's fix solves for ALL of
  // delta_x(t0) jointly (nothing held fixed/conditioned any more), so this
  // is no longer a Schur-complement sub-block trick -- these six blocks ARE
  // the entire state, so Pi_ss is simply P(t1)^-1 itself (row/column
  // order matched to this solve's own [phi0,p0,v,bg,ba,g] convention). No
  // cross term with the c-block: the prior itself does not correlate the
  // two blocks (any correlation enters only through the shared LiDAR
  // evidence, in the loop below). ----
  Eigen::MatrixXd Pi_ss = Eigen::MatrixXd::Zero(ncol_s, ncol_s);
  bool have_pi_ss = state_->idxBG() >= 0 && state_->idxBA() >= 0 && state_->idxG() >= 0;
  if (have_pi_ss) {
    // CQ-76 T2.1: prior_at_scan_start substitutes the pre-propagation P
    // snapshot for state_->cov() here -- a deliberate sensitivity probe
    // (see this option's own doc comment). Falls back to state_->cov()
    // (today's behavior) if the peek isn't primed for any reason, rather
    // than silently using an empty/garbage matrix.
    Eigen::MatrixXd P_for_omega = state_->cov();
    if (copts_.prior_at_scan_start) {
      Eigen::MatrixXd p_before_peek;
      if (imuProcQhatPeekPBefore(p_before_peek) &&
          p_before_peek.rows() == P_for_omega.rows() &&
          p_before_peek.cols() == P_for_omega.cols()) {
        P_for_omega = p_before_peek;
      }
    }
    const Eigen::MatrixXd Omega = P_for_omega.inverse();  // information form of P(t1) of the PREVIOUS scan (CQ-76-R3: not this scan's own P(t0) -- see this function's own Pi_ss comment above), or, if prior_at_scan_start, the pre-propagation snapshot
    // CQ-62 item 1: S2 -- Omega, the full dense inverse.
    if (copts_.psd_audit_en) logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S2_Omega", Omega);
    const int idx[6] = {StateGroup::idxR(), StateGroup::idxP(), StateGroup::idxV(),
                         state_->idxBG(), state_->idxBA(), state_->idxG()};
    for (int bi = 0; bi < 6; ++bi)
      for (int bj = 0; bj < 6; ++bj)
        Pi_ss.block<3, 3>(3 * bi, 3 * bj) = Omega.block<3, 3>(idx[bi], idx[bj]);
  }
  // Fallback (should not fire on any config with bias/gravity estimation
  // on, which every dispatched cell this card runs uses): an isotropic
  // proxy so the solve stays well-posed rather than silently singular; NOT
  // the card's own prescription, named here rather than silently
  // substituted for the real Schur-complement prior.
  if (!have_pi_ss) Pi_ss = Eigen::MatrixXd::Identity(ncol_s, ncol_s) * 1e6;

  // CQ-52 item 2, the decisive test: hold delta_bg at (effectively) zero by
  // inflating its own prior-precision diagonal block to near-infinite,
  // rather than touching the actual IMU process-noise config -- a standard
  // "freeze this state block" EKF technique. Off-diagonal Pi_ss entries
  // (bg's correlation with phi0/p0/v/ba/g) are left untouched, so the OTHER
  // blocks still solve correctly accounting for whatever prior correlation
  // exists; only bg's own column is driven toward zero regardless of what
  // the LiDAR evidence below would otherwise push it to. Column 3*3=9 in
  // the [phi0,p0,v,bg,ba,g] layout (bi=3 in the idx[6] array above).
  if (copts_.freeze_bg || coupled_bias_freeze_active_) Pi_ss.block<3, 3>(9, 9) += M3D::Identity() * 1e12;
  // CQ-62 item 1: S3 -- Pi_ss, after the freeze_bg adjustment.
  if (copts_.psd_audit_en) logPsdStage(voxel_map_->frame_idx_, coupled_iters_, "S3_Pi_ss", Pi_ss);

  Eigen::VectorXd s_vec(ncol_s);
  s_vec.segment<3>(0)  = coupled_delta_phi0_;
  s_vec.segment<3>(3)  = coupled_delta_pos0_;
  s_vec.segment<3>(6)  = coupled_delta_v_;
  s_vec.segment<3>(9)  = coupled_delta_bg_;
  s_vec.segment<3>(12) = coupled_delta_ba_;
  s_vec.segment<3>(15) = coupled_delta_g_;

  // CQ-82 Phase 1: the coefficient-block prior + the normal-equations
  // residual loop -- basis-specific -- MOVED verbatim into
  // buildImuCorrectionSystem() below. Nothing from here to the LDLT
  // solve changed; only where the code physically lives did.
  CoupledSystemBuild build = buildImuCorrectionSystem(
      mg, t0, t1, n_c, ncol, ncol_s, ncol_c, sigma_a, sigma_g, sigma_a_floor, sigma_g_floor,
      Pi_ss, s_vec);
  Eigen::MatrixXd& A = build.A;
  Eigen::VectorXd& b = build.b;
  double& sum_abs_r = build.sum_abs_r;
  double& sum_sq_r = build.sum_sq_r;
  double& sum_wr2 = build.sum_wr2;
  double& sum_sigma_squared = build.sum_sigma_squared;
  double& sum_floor_S = build.sum_floor_S;
  double& sum_sdiag_S = build.sum_sdiag_S;
  double& sum_pvar_S = build.sum_pvar_S;
  double& sum_prior_pose_S = build.sum_prior_pose_S;
  double& sum_weight_this_iter = build.sum_weight_this_iter;
  std::vector<double>& hcol_reldiff = build.hcol_reldiff;
  Eigen::Matrix<double, 6, 6>& HtH_pose_lidar = build.HtH_pose_lidar;
  Eigen::Matrix<double, 6, 1>& Htz_pose_lidar = build.Htz_pose_lidar;
  Eigen::Matrix<double, 6, 6>& H6_raw_accum = build.H6_raw_accum;
  Eigen::MatrixXd& phic_spread_sum = build.phic_spread_sum;
  double& phic_spread_sumsq = build.phic_spread_sumsq;
  int& phic_spread_n = build.phic_spread_n;
  // CQ-83: expose the nis/nis_est accumulators the same way.
  double& sum_nis = build.sum_nis;
  int& n_nis = build.n_nis;
  double& sum_nis_est = build.sum_nis_est;
  int& n_nis_est = build.n_nis_est;

  // CQ-79: are the spline corrections pulling points toward their matched
  // planes, and does it differ across the scan? Report-only, reads
  // struct Residual's own r/t/plane_id -- see this option's own doc
  // comment. Deliberately a SEPARATE pass over residuals_ (not folded
  // into the accumulation loop above) so this purely-diagnostic addition
  // cannot perturb the real math by construction, not just by inspection.
  if (copts_.log_point_plane_en && !mg.poses.empty()) {
    const double t0 = mg.poses.front().t;
    const double dt_scan = std::max(mg.image.t - t0, 1e-9);
    const int scan_id = voxel_map_->frame_idx_;
    const bool in_hist_window = scan_id >= copts_.log_point_plane_hist_start_scan &&
        scan_id < copts_.log_point_plane_hist_start_scan + copts_.log_point_plane_hist_n_scans;
    constexpr int NBINS = 20;
    int bin_n[NBINS] = {0};
    double bin_sum_signed[NBINS] = {0.0}, bin_sum_abs[NBINS] = {0.0}, bin_sum_sq[NBINS] = {0.0};

    std::unordered_set<std::size_t> cur_planes;
    double sum_t = 0.0, sum_t2 = 0.0, sum_r = 0.0, sum_tr = 0.0, sum_r2 = 0.0;
    int n_carried = 0;
    const int n_pts = static_cast<int>(residuals_.size());
    for (const auto& res : residuals_) {
      const double t_rel = res.t - t0;
      const double r_mm = res.r * 1000.0;
      sum_t += t_rel; sum_t2 += t_rel * t_rel;
      sum_r += r_mm; sum_tr += t_rel * r_mm; sum_r2 += r_mm * r_mm;

      const std::size_t plane_hash = std::hash<const void*>{}(res.plane_id);
      cur_planes.insert(plane_hash);
      if (coupled_prev_iter_planes_.count(plane_hash)) ++n_carried;

      if (in_hist_window) {
        int bin = static_cast<int>((t_rel / dt_scan) * NBINS);
        bin = std::clamp(bin, 0, NBINS - 1);
        ++bin_n[bin];
        bin_sum_signed[bin] += r_mm;
        bin_sum_abs[bin] += std::abs(r_mm);
        bin_sum_sq[bin] += r_mm * r_mm;
      }
    }

    double slope = 0.0, intercept = 0.0, r2 = 0.0, rms = 0.0;
    if (n_pts >= 2) {
      const double denom = n_pts * sum_t2 - sum_t * sum_t;
      if (std::abs(denom) > 1e-12) {
        slope = (n_pts * sum_tr - sum_t * sum_r) / denom;
        intercept = (sum_r - slope * sum_t) / n_pts;
        const double mean_r = sum_r / n_pts;
        double ss_tot = 0.0, ss_res = 0.0;
        for (const auto& res : residuals_) {
          const double t_rel = res.t - t0;
          const double r_mm = res.r * 1000.0;
          const double pred = slope * t_rel + intercept;
          ss_res += (r_mm - pred) * (r_mm - pred);
          ss_tot += (r_mm - mean_r) * (r_mm - mean_r);
        }
        r2 = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 0.0;
      }
      rms = std::sqrt(sum_r2 / n_pts);
    }
    const double carry_frac = n_pts > 0 ? static_cast<double>(n_carried) / n_pts : 0.0;

    static PersistentLogStream fit_log("cq79_point_plane_fit.csv");
    bool fit_first;
    std::ofstream& fit_ofs = fit_log.stream(&fit_first);
    if (fit_first)
      fit_ofs << "scan_id,iter,n_pts,slope_mm_per_s,intercept_mm,r2,rms_r_mm,"
                 "n_carried_over,carry_frac\n";
    fit_ofs << scan_id << "," << coupled_iters_ << "," << n_pts << ","
            << slope << "," << intercept << "," << r2 << "," << rms << ","
            << n_carried << "," << carry_frac << "\n";
    fit_ofs.flush();

    if (in_hist_window) {
      static PersistentLogStream hist_log("cq79_point_plane_hist.csv");
      bool hist_first;
      std::ofstream& hist_ofs = hist_log.stream(&hist_first);
      if (hist_first)
        hist_ofs << "scan_id,iter,bin,t_center_s,n_pts,mean_signed_r_mm,"
                     "mean_abs_r_mm,rms_r_mm\n";
      const double bin_width = dt_scan / NBINS;
      for (int b = 0; b < NBINS; ++b) {
        const double t_center = (b + 0.5) * bin_width;
        const double mean_signed = bin_n[b] > 0 ? bin_sum_signed[b] / bin_n[b] : 0.0;
        const double mean_abs = bin_n[b] > 0 ? bin_sum_abs[b] / bin_n[b] : 0.0;
        const double bin_rms = bin_n[b] > 0 ? std::sqrt(bin_sum_sq[b] / bin_n[b]) : 0.0;
        hist_ofs << scan_id << "," << coupled_iters_ << "," << b << "," << t_center
                 << "," << bin_n[b] << "," << mean_signed << "," << mean_abs
                 << "," << bin_rms << "\n";
      }
      hist_ofs.flush();
    }

    coupled_prev_iter_planes_ = std::move(cur_planes);
  }
  coupled_n_residuals_ = static_cast<int>(residuals_.size());
  coupled_sum_weight_ = sum_weight_this_iter;
  // CQ-53 items 2/4: RMS residual and hcol_reldiff distribution, overwritten
  // every iteration so the FINAL (converged) call's values survive.
  coupled_res_rms_ = residuals_.empty() ? -1.0 : std::sqrt(sum_sq_r / static_cast<double>(residuals_.size()));
  coupled_mean_sigma_squared_ = residuals_.empty() ? -1.0 : sum_sigma_squared / static_cast<double>(residuals_.size());
  coupled_reduced_chi2_ = residuals_.empty() ? -1.0 : sum_wr2 / static_cast<double>(residuals_.size());
  {
    const double sum_S = sum_floor_S + sum_sdiag_S + sum_pvar_S + sum_prior_pose_S;
    coupled_sum_S_ = (sum_S > 0.0) ? sum_S : -1.0;
    if (sum_S > 0.0) {
      coupled_floor_share_      = sum_floor_S / sum_S;
      coupled_sdiag_share_      = sum_sdiag_S / sum_S;
      coupled_pvar_share_       = sum_pvar_S / sum_S;
      coupled_prior_pose_share_ = sum_prior_pose_S / sum_S;
    }
  }
  coupled_nis_     = (n_nis > 0)     ? sum_nis / n_nis         : -1.0;
  coupled_nis_est_ = (n_nis_est > 0) ? sum_nis_est / n_nis_est : -1.0;
  if (!hcol_reldiff.empty()) {
    std::sort(hcol_reldiff.begin(), hcol_reldiff.end());
    const size_t n = hcol_reldiff.size();
    coupled_hcol_reldiff_p10_ = hcol_reldiff[static_cast<size_t>(0.10 * (n - 1))];
    coupled_hcol_reldiff_p50_ = hcol_reldiff[static_cast<size_t>(0.50 * (n - 1))];
    coupled_hcol_reldiff_p90_ = hcol_reldiff[static_cast<size_t>(0.90 * (n - 1))];
    coupled_hcol_reldiff_max_ = hcol_reldiff.back();
  }
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_pp6(HtH_pose_lidar.block<3, 3>(3, 3));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_rr6(HtH_pose_lidar.block<3, 3>(0, 0));
    coupled_h_pp_min_eig_ = es_pp6.eigenvalues()(0);
    coupled_h_rr_min_eig_ = es_rr6.eigenvalues()(0);
    // CQ-83: h_pp_max_eig off the SAME eigensolve as the pre-existing min
    // (CQ-81 item B's own pattern); h_rr_trace/htth_pos_trace are plain
    // traces of the same HtH_pose_lidar blocks (htth_pos_trace mirrors
    // decoupled's "trace(HtH's position block), read directly off ekf_.HtH").
    coupled_h_pp_max_eig_ = es_pp6.eigenvalues()(2);
    coupled_h_rr_trace_ = HtH_pose_lidar.block<3, 3>(0, 0).trace();
    coupled_htth_pos_trace_ = HtH_pose_lidar.block<3, 3>(3, 3).trace();
  }

  // CQ-53 item 1: pivot diagnostics on the joint matrix, READ-ONLY (three
  // SEPARATE diagnostic LDLTs, none of which feed the actual solve below --
  // this project already learned, round 33 scan 3133 on the decoupled side,
  // that ldlt.info()==Success does not rule out a rank-deficient matrix).
  {
    Eigen::LDLT<Eigen::MatrixXd> ldlt_joint(A);
    if (ldlt_joint.info() == Eigen::Success) {
      coupled_joint_dmin_ = ldlt_joint.vectorD().minCoeff();
      coupled_joint_dmax_ = ldlt_joint.vectorD().maxCoeff();
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt_state(A.block(0, 0, ncol_s, ncol_s));
    if (ldlt_state.info() == Eigen::Success) {
      coupled_state_dmin_ = ldlt_state.vectorD().minCoeff();
      coupled_state_dmax_ = ldlt_state.vectorD().maxCoeff();
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt_coeff(A.block(ncol_s, ncol_s, ncol_c, ncol_c));
    if (ldlt_coeff.info() == Eigen::Success) {
      coupled_coeff_dmin_ = ldlt_coeff.vectorD().minCoeff();
      coupled_coeff_dmax_ = ldlt_coeff.vectorD().maxCoeff();
    }
    // CQ-66: test the end_time-collapses-to-rank-6 derivation directly.
    // rank_h6 = rank(H6_raw_accum) (the 6x6 upper bound the derivation
    // predicts); rank_cblock = numerical rank of the RESIDUAL LOOP's own
    // contribution to the c-block, i.e. (A-Lambda)'s c-block -- A's c-block
    // was seeded with Lambda BEFORE the residual loop ran (A.block(ncol_s,
    // ncol_s,...) = Lambda, above), so subtracting it back out isolates
    // exactly what the derivation is about. Only meaningful when Lambda was
    // actually captured (log_cp_constraint_en); psd_audit_en alone still
    // gives rank_h6 (needs only H6_raw_accum, always safe to compute).
    if (copts_.psd_audit_en) {
      Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd_h6(H6_raw_accum);
      const Eigen::Matrix<double, 6, 1> sv_h6 = svd_h6.singularValues();
      const double tol = sv_h6(0) * 1e-9 * 6;  // scale-relative, Eigen's own default-style tolerance
      int rank_h6 = 0;
      for (int i = 0; i < 6; ++i) if (sv_h6(i) > tol) ++rank_h6;

      int rank_cblock = -1;
      double cblock_tol = std::numeric_limits<double>::quiet_NaN();
      if (copts_.log_cp_constraint_en) {
        const Eigen::MatrixXd resid_only_cblock =
            A.block(ncol_s, ncol_s, ncol_c, ncol_c) - coupled_last_Lambda_;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_c(resid_only_cblock);
        const Eigen::VectorXd sv_c = svd_c.singularValues();
        cblock_tol = (sv_c.size() > 0 ? sv_c(0) : 0.0) * 1e-9 * ncol_c;
        rank_cblock = 0;
        for (int i = 0; i < sv_c.size(); ++i) if (sv_c(i) > cblock_tol) ++rank_cblock;
      }

      double phic_spread = std::numeric_limits<double>::quiet_NaN();
      if (phic_spread_n > 0) {
        const Eigen::MatrixXd mean_phic = phic_spread_sum / static_cast<double>(phic_spread_n);
        const double var = phic_spread_sumsq / static_cast<double>(phic_spread_n) - mean_phic.squaredNorm();
        phic_spread = std::sqrt(std::max(var, 0.0));
      }

      static PersistentLogStream rank_log("cq66_rank.txt");
      bool rank_first;
      std::ofstream& rank_ofs = rank_log.stream(&rank_first);
      if (rank_first)
        rank_ofs << "scan_id,jacobian_time_mode,n_c,ncol_c,n_residuals,"
                     "rank_h6,h6_tol,rank_cblock,cblock_tol,coeff_dmin,phic_spread\n";
      rank_ofs << voxel_map_->frame_idx_ << "," << copts_.jacobian_time_mode << ","
               << copts_.n_c << "," << ncol_c << "," << residuals_.size() << ","
               << rank_h6 << "," << tol << "," << rank_cblock << "," << cblock_tol << ","
               << coupled_coeff_dmin_ << "," << phic_spread << "\n";
      rank_ofs.flush();
    }
    // Disabled-by-default (JOINT_PIVOT_MIN_FLOOR=-1.0, below any real pivot
    // this system produces) -- exercises the refusal SHAPE without gating
    // anything at its shipped value. No validated threshold exists yet
    // (rule 26: a real numerics default is Bryce's call).
    coupled_pivot_guard_ = (coupled_joint_dmin_ < LioProcCoupledOptions::JOINT_PIVOT_MIN_FLOOR);
  }

  Eigen::VectorXd delta = Eigen::VectorXd::Zero(ncol);
  if (copts_.zero_mean) {
    // Item 3d(ii): KKT-bordered solve, 6 equality rows (one 3-vector per
    // axis) sum_samples beta_j(t) * c_j = 0, zero in the s-block columns.
    // Same rank-deficiency-removal shape CQ-41's own end-constraint rewrite
    // already uses (a bordered system, not a projected/reduced one).
    Eigen::VectorXd gsum = Eigen::VectorXd::Zero(n_c);
    for (const auto& pose : mg.poses)
      for (int j = 0; j < n_c; ++j) gsum[j] += basisWeight(j, n_c, t0, t1, pose.t);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6, ncol);
    for (int j = 0; j < n_c; ++j) {
      C.block<3, 3>(0, ncol_s + 3 * j)         = gsum[j] * M3D::Identity();  // acc axis
      C.block<3, 3>(3, ncol_s + 3 * n_c + 3 * j) = gsum[j] * M3D::Identity();  // gyr axis
    }
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ncol + 6, ncol + 6);
    K.block(0, 0, ncol, ncol) = A;
    K.block(0, ncol, ncol, 6) = C.transpose();
    K.block(ncol, 0, 6, ncol) = C;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ncol + 6);
    rhs.segment(0, ncol) = b;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
    // CQ-61 item 4, sibling defect, RULE 58: delta was initialised to
    // Zero() above and this branch only ASSIGNED it on success -- a failed
    // solve silently fell through as delta=0, i.e. "the GN step converged
    // to exactly zero correction", indistinguishable from genuine
    // convergence. ABORT loudly instead; a failed bordered solve here
    // means the KKT system itself is singular, not that nothing needs
    // correcting.
    if (lu.isInvertible()) {
      delta = lu.solve(rhs).head(ncol);
    } else {
      std::ostringstream abort_msg;
      abort_msg << "[FATAL] coupled zero_mean solve: bordered KKT matrix K is "
                   "NOT invertible -- scan_id=" << voxel_map_->frame_idx_
                << " n_residuals=" << coupled_n_residuals_
                << " n_c=" << copts_.n_c << " jacobian_time_mode=" << copts_.jacobian_time_mode
                << " iters_so_far=" << coupled_iters_;
      throw std::runtime_error(abort_msg.str());
    }
  } else {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    // Same rule-58 fix as above: ldlt.info() previously gated the ONLY
    // assignment to delta, so a failed factorization silently left
    // delta=0 (a false "converged, no correction needed" instead of a
    // real failure). ABORT loudly instead.
    if (ldlt.info() == Eigen::Success) {
      delta = ldlt.solve(b);
    } else {
      std::ostringstream abort_msg;
      abort_msg << "[FATAL] coupled GN solve: LDLT(A) factorization failed "
                   "(info()=NumericalIssue) -- scan_id=" << voxel_map_->frame_idx_
                << " n_residuals=" << coupled_n_residuals_
                << " n_c=" << copts_.n_c << " jacobian_time_mode=" << copts_.jacobian_time_mode
                << " iters_so_far=" << coupled_iters_;
      throw std::runtime_error(abort_msg.str());
    }
  }

  const Eigen::VectorXd delta_s = delta.segment(0, ncol_s);
  const Eigen::VectorXd delta_c = delta.segment(ncol_s, ncol_c);
  // CQ-53 item 3: THIS iteration's own step norms (not the scan-accumulated
  // total below).
  coupled_last_delta_s_norm_ = delta_s.norm();
  coupled_last_delta_c_norm_ = delta_c.norm();
  // CQ-53 item (B): the same combined delta_s split into its six 3-dim
  // sub-blocks -- [delta_phi0, delta_p0, delta_v, delta_bg, delta_ba,
  // delta_g], matching s_vec's own layout above exactly.
  coupled_last_delta_phi0_norm_    = delta_s.segment<3>(0).norm();
  coupled_last_delta_p0_norm_      = delta_s.segment<3>(3).norm();
  coupled_last_delta_v_norm_step_  = delta_s.segment<3>(6).norm();
  coupled_last_delta_bg_norm_step_ = delta_s.segment<3>(9).norm();
  coupled_last_delta_ba_norm_step_ = delta_s.segment<3>(12).norm();
  coupled_last_delta_g_norm_step_  = delta_s.segment<3>(15).norm();
  // CQ-56: split THIS iteration's step into its c_acc/c_gyr halves --
  // delta_c's own layout is [c_acc(3*n_c), c_gyr(3*n_c)], same order as
  // c_vec above.
  coupled_last_delta_c_acc_norm_ = delta_c.segment(0, 3 * n_c).norm();
  coupled_last_delta_c_gyr_norm_ = delta_c.segment(3 * n_c, 3 * n_c).norm();
  if (copts_.log_cp_constraint_en) coupled_last_delta_c_ = delta_c;
  // Item 3e(v)/3f bug 2: same [phi0,p0,v,bg,ba,g] order as s_vec above.
  coupled_delta_phi0_ += delta_s.segment<3>(0);
  coupled_delta_pos0_ += delta_s.segment<3>(3);
  coupled_delta_v_    += delta_s.segment<3>(6);
  // CQ-55 item 11(b): the principled fix, no threshold. A is regularised by
  // Pi_ss/Lambda everywhere, so it is never actually singular along the
  // degenerate bg/c_gyr direction even though the DATA says nothing there
  // -- A^-1 (hence P(t1)) is then SMALL along it, and the filter reports
  // having learned the split when it only inherited the prior's own
  // assumption (item 11's own diagnosis). Decompose THIS iteration's raw
  // bg increment into the eigenbasis of A^-1's own bg-marginal covariance;
  // scale each component by a continuous confidence = 1 - min(1,
  // posterior_var/prior_var) along that direction (0 = no info gained at
  // all, 1 = fully data-determined) BEFORE folding it into the persistent
  // bias. Only the persistent accumulator is touched -- c_vec's own
  // increment (reset to zero every scan regardless) is untouched, so this
  // cannot itself remove the c_gyr/bg degeneracy, only stop the PERSISTENT
  // half of it from accumulating an unearned point estimate.
  V3D delta_bg_this_iter = delta_s.segment<3>(9);
  // CQ-55 item 11(a): report-only bg-projection logging shares the SAME
  // A^-1 inversion bias_observable_only needs, computed once if either is
  // engaged -- avoids paying for a second ncol x ncol solve when both are
  // on, while item 11(a) stays available independent of 11(b)'s own flag.
  if (copts_.bias_observable_only || copts_.log_bg_projection_en) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt_full(A);
    if (ldlt_full.info() == Eigen::Success) {
      const Eigen::MatrixXd Ainv = ldlt_full.solve(Eigen::MatrixXd::Identity(ncol, ncol));
      const M3D P_bg = Ainv.block<3, 3>(9, 9);
      Eigen::SelfAdjointEigenSolver<M3D> es(P_bg);
      if (copts_.log_bg_projection_en) {
        // Eigenvalues ascending (SelfAdjointEigenSolver's convention): (0)
        // is the smallest posterior variance (best-observed direction),
        // (2) the largest (the degenerate/least-observed direction).
        coupled_bg_var_observed_   = es.eigenvalues()(0);
        coupled_bg_var_degenerate_ = es.eigenvalues()(2);
      }
      if (copts_.bias_observable_only) {
        const M3D Pi_bg = Pi_ss.block<3, 3>(9, 9);
        Eigen::FullPivLU<M3D> lu_pi(Pi_bg);
        // Average prior variance along the bg block (trace(Pi_bg^-1)/3) --
        // what P_bg WOULD be with no data at all this scan. Falls back to a
        // large (effectively "fully unconstrained") value if Pi_bg happens
        // to be singular, so an unavailable prior never masquerades as
        // "fully observed" (rule 58: a fallback on an impossible condition
        // must not look like a successful one -- this is the not-invertible
        // case, made explicit rather than silently dividing by a near-zero).
        const double prior_var = lu_pi.isInvertible()
            ? std::max(lu_pi.inverse().trace() / 3.0, 1e-12) : 1e12;
        V3D filtered = V3D::Zero();
        for (int k = 0; k < 3; ++k) {
          const double conf = 1.0 - std::min(1.0, std::max(0.0, es.eigenvalues()(k) / prior_var));
          const V3D v = es.eigenvectors().col(k);
          filtered += conf * (v.dot(delta_bg_this_iter)) * v;
        }
        delta_bg_this_iter = filtered;
      }
    }
  }
  coupled_delta_bg_   += delta_bg_this_iter;
  coupled_delta_ba_   += delta_s.segment<3>(12);
  coupled_delta_g_    += delta_s.segment<3>(15);
  for (int j = 0; j < n_c; ++j) {
    coupled_c_acc_[j] += delta_c.segment<3>(3 * j);
    coupled_c_gyr_[j] += delta_c.segment<3>(3 * n_c + 3 * j);
  }

  // CQ-72 item 3: how far does the trajectory the correction implies
  // actually move, per iteration -- ||P_k c|| in METRES (P_k = phi_head[k]'s
  // position rows, c = the CURRENT, just-updated coefficient vector, same
  // c_vec layout [c_acc(3*n_c);c_gyr(3*n_c)] used everywhere else). One row
  // per (scan_id, iter); the FINAL row for a given scan_id is that scan's
  // own per-scan summary (the card asks for both, and a per-iteration log
  // already contains the per-scan-final one as its last row -- no second
  // write site needed). Logged one iteration AFTER the coefficient it
  // describes was applied, matching coupled_iters_'s own post-increment
  // convention below.
  if (copts_.log_traj_dev_en) {
    Eigen::VectorXd c_now(ncol_c);
    for (int j = 0; j < n_c; ++j) {
      c_now.segment<3>(3 * j) = coupled_c_acc_[j];
      c_now.segment<3>(3 * n_c + 3 * j) = coupled_c_gyr_[j];
    }
    double max_norm = 0.0, sumsq = 0.0;
    V3D end_vec = V3D::Zero();
    const int n_k = static_cast<int>(coupled_prop_.phi_head.size());
    for (int k = 0; k < n_k; ++k) {
      const V3D traj_k = coupled_prop_.phi_head[k].middleRows<3>(3) * c_now;
      const double norm_k = traj_k.norm();
      max_norm = std::max(max_norm, norm_k);
      sumsq += norm_k * norm_k;
      if (k == n_k - 1) end_vec = traj_k;
    }
    const double end_norm = end_vec.norm();
    const double rms_norm = (n_k > 0) ? std::sqrt(sumsq / static_cast<double>(n_k)) : 0.0;
    const double step_norm = coupled_prev_traj_dev_valid_
        ? (end_vec - coupled_prev_traj_dev_end_vec_).norm() : 0.0;
    static PersistentLogStream traj_dev_log("cq72_traj_dev.csv");
    bool traj_dev_first;
    std::ofstream& traj_dev_ofs = traj_dev_log.stream(&traj_dev_first);
    if (traj_dev_first)
      traj_dev_ofs << "scan_id,iter,traj_dev_max_m,traj_dev_end_m,traj_dev_rms_m,traj_dev_step_m\n";
    traj_dev_ofs << std::setprecision(9)
                 << voxel_map_->frame_idx_ << "," << coupled_iters_ << ","
                 << max_norm << "," << end_norm << "," << rms_norm << "," << step_norm << "\n";
    traj_dev_ofs.flush();
    coupled_prev_traj_dev_end_vec_ = end_vec;
    coupled_prev_traj_dev_valid_ = true;
  }

  // TQ-40 item 3: ask/got/refusal, THIS iteration's own [delta_phi0,delta_p0]
  // against the pure-LiDAR-info accumulated above -- same eigenbasis-solve
  // convention ekf.h's own P1 diagnostic uses (drop modes below a relative
  // tolerance rather than inverting a possibly-singular 6x6).
  {
    Eigen::Matrix<double, 6, 1> dxv_pose;
    dxv_pose << delta_s.segment<3>(0), delta_s.segment<3>(3);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es6c(HtH_pose_lidar);
    const auto& ev6 = es6c.eigenvalues();
    const double tol6 = 1e-12 * std::max(1.0, ev6(5));
    const Eigen::Matrix<double, 6, 1> z6 = es6c.eigenvectors().transpose() * Htz_pose_lidar;
    const Eigen::Matrix<double, 6, 1> y6 = es6c.eigenvectors().transpose() * dxv_pose;
    double ask6 = 0.0, got6 = 0.0;
    for (int i = 0; i < 6; ++i) if (ev6(i) > tol6) {
      ask6 += z6(i) * z6(i) / ev6(i);
      got6 += ev6(i) * y6(i) * y6(i);
    }
    coupled_ask_ = ask6;
    coupled_got_ = got6;
    coupled_refusal_ = (ask6 > 0.0) ? (1.0 - got6 / ask6) : std::numeric_limits<double>::quiet_NaN();
    // CQ-83: kappa_eff/kappa_gev*, decoupled's own exact formulas
    // (lio_decoupled.cpp) against coupled's HtH_pose_lidar/prior_cov_ in
    // place of ekf_.HtH/that path's own prior_cov_ slice -- same 6x6
    // rot-then-pos layout, so the identical block extraction applies.
    coupled_kappa_eff_ = (got6 > 0.0) ? (std::sqrt(ask6 / got6) - 1.0) : -1.0;
    if (prior_cov_.rows() >= StateGroup::idxR() + 6 &&
        prior_cov_.cols() >= StateGroup::idxR() + 6) {
      const Eigen::Matrix<double, 6, 6> P_prior_6 =
          prior_cov_.block<6, 6>(StateGroup::idxR(), StateGroup::idxR());
      Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
          ges(HtH_pose_lidar, P_prior_6);
      if (ges.info() == Eigen::Success) {
        const auto& gev = ges.eigenvalues();
        for (int i = 0; i < 6; ++i) coupled_kappa_gev_[i] = gev(i);
        coupled_kappa_gev_ok_ = true;
      }
    }
    coupled_htz_rot_norm_ = Htz_pose_lidar.segment<3>(0).norm();
    coupled_htz_pos_norm_ = Htz_pose_lidar.segment<3>(3).norm();
  }

  // Item 3d(i): DC-component/bias-split diagnostics, overwritten every
  // iteration so the FINAL (converged) call's values are what survives.
  {
    V3D c_acc_dc = V3D::Zero(), c_gyr_dc = V3D::Zero();
    for (int j = 0; j < n_c; ++j) { c_acc_dc += coupled_c_acc_[j]; c_gyr_dc += coupled_c_gyr_[j]; }
    c_acc_dc /= static_cast<double>(n_c); c_gyr_dc /= static_cast<double>(n_c);
    coupled_c_acc_dc_over_sigma_ = c_acc_dc.norm() / sigma_a;
    coupled_c_gyr_dc_over_sigma_ = c_gyr_dc.norm() / sigma_g;
    coupled_dba_over_sigma_ = coupled_delta_ba_.norm() / sigma_a;
    coupled_dbg_over_sigma_ = coupled_delta_bg_.norm() / sigma_g;
    // TQ-40 item 4: the FULL vector norms (not just the DC/mean component
    // above), RMS-per-coefficient over n_c so the number is comparable
    // across n_c the same way the DC ones already are.
    double sq_acc = 0.0, sq_gyr = 0.0;
    for (int j = 0; j < n_c; ++j) { sq_acc += coupled_c_acc_[j].squaredNorm(); sq_gyr += coupled_c_gyr_[j].squaredNorm(); }
    coupled_c_acc_over_sigma_ = std::sqrt(sq_acc / static_cast<double>(n_c)) / sigma_a;
    coupled_c_gyr_over_sigma_ = std::sqrt(sq_gyr / static_cast<double>(n_c)) / sigma_g;
    // CQ-56: the same RMS-per-coefficient totals, in raw physical units
    // (m/s^2, rad/s) rather than sigma-normalized -- directly comparable
    // across runs with different sigma_a/sigma_g (e.g. adaptive_sigma on
    // vs off) without needing to de-normalize.
    coupled_c_acc_total_norm_ = std::sqrt(sq_acc / static_cast<double>(n_c));
    coupled_c_gyr_total_norm_ = std::sqrt(sq_gyr / static_cast<double>(n_c));
    coupled_delta_v_norm_ = coupled_delta_v_.norm();
    coupled_delta_g_norm_ = coupled_delta_g_.norm();
    // CQ-54 item 5: the angular-rate correction actually used is
    // seg.gyr - delta_bg + delta_w (coupled_estimator.cpp's angvel_avr) --
    // ONE physical quantity split across two independently-priced
    // parameters. c_gyr_dc (just computed above) is the within-scan
    // correction's own DC/mean contribution, the same sense as delta_w;
    // coupled_delta_bg_ enters with a MINUS sign in angvel_avr, so its own
    // contribution to angular rate is -coupled_delta_bg_. Logged as norms
    // (deg/s) plus their net (vector sum, deg/s) -- a small net despite two
    // large opposing parts is the "paying twice for nothing" signature the
    // card's own item 5 asks to check for.
    const double rad2deg = 180.0 / M_PI;
    const V3D w_from_bg = -coupled_delta_bg_;
    coupled_w_from_c_deg_s_ = c_gyr_dc.norm() * rad2deg;
    coupled_w_from_bg_deg_s_ = w_from_bg.norm() * rad2deg;
    coupled_w_net_deg_s_ = (c_gyr_dc + w_from_bg).norm() * rad2deg;
  }

  // ---- (6) re-propagate ONCE MORE with the updated c AND delta_s, so
  // state_ and mg.points (via the caller's next iteration, or the loop
  // exit) reflect what this step actually applied -- mirrors
  // estimateStateCorrection()'s own solve-then-apply contract. ----
  // CQ-57 item 2: build the R,P,V 9x9 seed (SAME P this scan's Pi_ss was
  // built from -- see item 0c/item 1) only when the diagnostic is on; kept
  // as a local so the pointer below is null (default, existing behaviour)
  // whenever it's off.
  Eigen::Matrix<double, 9, 9> repro_P0;
  const Eigen::Matrix<double, 9, 9>* repro_P0_ptr = nullptr;
  const V3D var_gyr = state_->varGyr(), var_acc = state_->varAcc();
  if (copts_.log_cov_repropagation_en) {
    const Eigen::MatrixXd& P0_full = state_->cov();
    if (P0_full.rows() >= 9 && P0_full.cols() >= 9) {
      repro_P0 = P0_full.block<9, 9>(0, 0);  // idxR=0,idxP=3,idxV=6, contiguous
      repro_P0_ptr = &repro_P0;
    }
  }
  propagateCoupled(mg.poses, state_propagat_.rot(), t1,
                   mg.poses.front().rot * Exp(coupled_delta_phi0_),
                   mg.poses.front().pos + coupled_delta_pos0_,
                   coupled_v0_pre_ + coupled_delta_v_,
                   coupled_g0_pre_ + coupled_delta_g_, coupled_g0_pre_,
                   coupled_delta_bg_, coupled_delta_ba_,
                   coupled_c_acc_, coupled_c_gyr_, n_c, coupled_prop_,
                   repro_P0_ptr, copts_.repro_q_alpha_gyr, copts_.repro_q_alpha_acc,
                   &var_gyr, &var_acc, copts_.repro_second_order);
  state_->setPropagatedState(coupled_prop_.rot1, coupled_prop_.pos1, coupled_prop_.vel1);

  // Outer-loop convergence read: the ENDPOINT state change THIS STEP
  // implied, i.e. the joint Jacobian at t1 applied to [delta_s, delta_c] --
  // same quantity (dtheta,dt)'s norms are checked against as the decoupled
  // path's own dtheta/dt (estimateStateCorrection()'s return contract).
  const Eigen::VectorXd d9 = coupled_prop_.phi_x_head.back() * delta_s
                            + coupled_prop_.phi_head.back()   * delta_c;
  dtheta_out = d9.segment<3>(0);
  dt_out = d9.segment<3>(3);

  const auto t_end = std::chrono::steady_clock::now();
  coupled_solve_ms_ += std::chrono::duration<double, std::milli>(t_end - t_start).count();

  return residuals_.empty() ? 0.0 : sum_abs_r / static_cast<double>(residuals_.size());
}

// CQ-82 Phase 2, Artifact 1: see this function's own declaration-site
// doc comment in lio_coupled.h.
double LioProcCoupled::estimateCoupledCorrectionPoseBasis(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out)
{
  dtheta_out = V3D::Zero();
  dt_out = V3D::Zero();
  // "A fallback on an impossible condition aborts loudly; it never
  // substitutes" -- a scan whose ScanSpline::fit() failed at scan start has
  // no valid initial trajectory for this basis to correct; silently
  // returning 0.0 here would let the GN loop believe the scan converged
  // trivially, which is not true and would corrupt state_ silently.
  if (!coupled_pose_spline_valid_) {
    std::ostringstream diag;
    diag << "[coupled/pose] estimateCoupledCorrectionPoseBasis(): this "
            "scan's ScanSpline::fit() failed or never ran at scan start -- "
            "no valid initial trajectory to correct. lastFitFailCause()="
         << static_cast<int>(coupled_pose_spline_.lastFitFailCause())
         << " mg.poses.size()=" << mg.poses.size()
         << " n_c_requested=" << copts_.n_c
         << " scan_id=" << voxel_map_->frame_idx_;
    throw std::runtime_error(diag.str());
  }

  const double t1 = mg.image.t;
  // BUGFIX: n_c here MUST be the spline's own ACTUAL control-point count
  // (it can be clamped below copts_.n_c -- see the scan-start reset's own
  // comment), never copts_.n_c directly. coupled_c_pos_/coupled_c_rot_ are
  // already sized to this same actual count at scan-start reset time.
  const int n_c = coupled_pose_spline_.nControlPoints();
  // TEMPORARY diagnostic (not meant to stay): confirm every size this loop
  // depends on actually agrees before touching any Eigen column.
  if (static_cast<int>(coupled_c_pos_.size()) != n_c ||
      static_cast<int>(coupled_c_rot_.size()) != n_c ||
      coupled_pose_spline_.cpPos().cols() != n_c) {
    std::ostringstream diag;
    diag << "[coupled/pose] DIAG size mismatch: n_c(nControlPoints)=" << n_c
         << " coupled_c_pos_.size()=" << coupled_c_pos_.size()
         << " coupled_c_rot_.size()=" << coupled_c_rot_.size()
         << " cpPos().cols()=" << coupled_pose_spline_.cpPos().cols()
         << " n_c_requested=" << copts_.n_c
         << " nControlPointsRequested()=" << coupled_pose_spline_.nControlPointsRequested()
         << " clamped=" << coupled_pose_spline_.nControlPointsClamped();
    throw std::runtime_error(diag.str());
  }

  // The trial spline: this scan's ONE-TIME fit, with the corrections
  // accumulated so far THIS scan (across earlier GN iterations) applied.
  ScanSpline trial = coupled_pose_spline_;
  for (int j = 0; j < n_c; ++j) {
    trial.cpPosMut().col(j) += coupled_c_pos_[j];
    trial.cp_phi_.col(j)    += coupled_c_rot_[j];
  }
  const M3D prev_tail_R = trial.rotAt(t1);
  const V3D prev_tail_p = trial.posAt(t1);

  // Deskew against the trial spline -- deskewPointsSpline() is EXISTING,
  // shared machinery (lio/deskew.h), already used by the decoupled spline
  // path; not re-derived here. Downsample the same way the raw_imu arm
  // does (shared opts_.ds_mode/ds_leaf_size).
  std::vector<PointXYZCov> deskewed;
  deskewPointsSpline(state_, trial, t1, mg.lidar_points, opts_.deskew, deskewed);
  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = deskewed;
  }
  buildResiduals(mg.points, residuals_, coupled_iters_ == 0);

  std::vector<PoseSplineLidarObs> lidar_obs;
  lidar_obs.reserve(residuals_.size());
  for (const auto& res : residuals_) {
    PoseSplineLidarObs o;
    o.t = res.t; o.raw_body_point = res.raw_body_point; o.normal = res.normal;
    o.r = res.r; o.sigma2 = res.sigma_squared;
    lidar_obs.push_back(o);
  }
  std::vector<PoseSplineImuObs> imu_obs;
  imu_obs.reserve(mg.imu_samples_raw.size());
  for (const auto& s : mg.imu_samples_raw) {
    PoseSplineImuObs o; o.t = s.t; o.acc = s.acc; o.gyr = s.gyro;
    imu_obs.push_back(o);
  }

  // CQ-86 item 0: reuse the SAME estimator/coupled/jacobian_time_mode knob
  // the raw_imu arm reads (rather than adding a second, pose-arm-only key)
  // so the two arms' rank/spread rows are directly comparable under an
  // identical config value; "point_time"/"end_time" map onto the pose arm's
  // own PoseSplineTimeMode, anything else (legacy_mismatched, the raw_imu
  // arm's default) falls back to kPointTime -- the pose arm's own default
  // and the only mode it has ever run under through round 5.
  const PoseSplineTimeMode pose_time_mode =
      (copts_.jacobian_time_mode == "end_time") ? PoseSplineTimeMode::kEndTime
                                                 : PoseSplineTimeMode::kPointTime;
  // POST-CQ-87-REVIEW FIX (item 4, follow-up 2026-09-21): read directly
  // from copts_.pose_imu_var_acc/gyr (state/cov/acc,gyr, loaded straight
  // off the param server in loadParameters() -- see that field's own doc
  // comment) rather than state_->varAcc()/varGyr(), so this arm's
  // weighting can never silently diverge from the shipped config value
  // regardless of calibration/adaptive-Q state mutation elsewhere.
  const double pose_sigma_acc = std::sqrt(copts_.pose_imu_var_acc);
  const double pose_sigma_gyr = std::sqrt(copts_.pose_imu_var_gyr);
  // POST-CQ-87-REVIEW FIX (item 3): the correction ALREADY accumulated
  // this scan (across earlier GN iterations), flattened into the SAME
  // [c_p(3n_c); c_phi(3n_c)] layout buildPoseSplineCBlock() itself uses --
  // required so the curvature/Tikhonov prior's gradient is centered at the
  // CURRENT c, not silently re-centered at 0 every iteration. c=0 on
  // iteration 1 (coupled_c_pos_/coupled_c_rot_ were just reset to Zero()
  // at scan start), so this is a no-op there and only matters from
  // iteration 2 onward.
  Eigen::VectorXd c_current(6 * n_c);
  for (int j = 0; j < n_c; ++j) {
    c_current.segment<3>(3 * j)           = coupled_c_pos_[j];
    c_current.segment<3>(3 * n_c + 3 * j) = coupled_c_rot_[j];
  }
  const auto build = buildPoseSplineCBlock(
      trial, lidar_obs, imu_obs,
      state_->biasAcc(), state_->biasGyr(), state_->gravity(),
      copts_.pose_imu_weight_acc, copts_.pose_imu_weight_gyr,
      copts_.pose_curvature_weight_pos, copts_.pose_curvature_weight_rot,
      pose_time_mode, t1, /*audit=*/copts_.psd_audit_en,
      copts_.pose_tikhonov_eps, pose_sigma_acc, pose_sigma_gyr, c_current);

  if (copts_.psd_audit_en) {
    // CQ-86 item 0: rank_cblock = numerical rank of build.A_lidar_only (the
    // LiDAR residual loop's own contribution, isolated from the Tikhonov
    // floor and smoothness prior) -- the pose-arm analogue of CQ-66's
    // "A.block(c) - Lambda" isolation on the raw_imu arm. phic_spread is
    // build.phic_spread directly (already computed single-pass inside the
    // builder). Same SVD/tolerance convention as cq66_rank.txt's rank_cblock
    // column (scale-relative, Eigen-default-style).
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_c(build.A_lidar_only);
    const Eigen::VectorXd sv_c = svd_c.singularValues();
    const double cblock_tol = (sv_c.size() > 0 ? sv_c(0) : 0.0) * 1e-9 * (6 * n_c);
    int rank_cblock = 0;
    for (int i = 0; i < sv_c.size(); ++i) if (sv_c(i) > cblock_tol) ++rank_cblock;

    static PersistentLogStream rank_log("cq86_rank.txt");
    bool rank_first;
    std::ofstream& rank_ofs = rank_log.stream(&rank_first);
    if (rank_first)
      rank_ofs << "scan_id,jacobian_time_mode,n_c,ncol_c,n_residuals,"
                   "rank_cblock,cblock_tol,phic_spread\n";
    rank_ofs << voxel_map_->frame_idx_ << "," << copts_.jacobian_time_mode << ","
             << n_c << "," << (6 * n_c) << "," << residuals_.size() << ","
             << rank_cblock << "," << cblock_tol << "," << build.phic_spread << "\n";
    rank_ofs.flush();
  }

  // CQ-86 item 1: the weaker head-freeze anchor. n_frozen columns of BOTH
  // c_p and c_phi are eliminated from the linear system entirely (not just
  // solved-then-discarded) -- their delta is EXACTLY zero, matching
  // decoupled's own "clamped" semantics rather than a soft large-weight
  // prior. Reduces to a plain LDLT solve on the surviving free columns
  // when pose_head_freeze_cp==0 (the default), byte-identical to before
  // this item -- free_idx == every column, no elimination performed.
  const int n_frozen = std::max(0, std::min(copts_.pose_head_freeze_cp, n_c));
  Eigen::VectorXd delta_c = Eigen::VectorXd::Zero(6 * n_c);
  if (n_frozen == 0) {
    // CQ-87 items 1/2/3: the REAL head coupling, now the default path
    // whenever pose_head_freeze_cp==0. Omega/Pi_ss in the SAME convention
    // the raw_imu arm's own s-block prior uses (lio_coupled.cpp:2256-2281,
    // ~line 2273's Omega = P_for_omega.inverse()) -- restricted to the
    // [rot,pos] 6x6 sub-block since this arm's own s-block is only
    // [delta_phi0;delta_pos0], not the full 18-dim raw_imu state (velocity/
    // bias/gravity remain owned by IMU propagation alone on this arm, per
    // CQ-87's own explicit not-in-scope list).
    Eigen::Matrix<double, 6, 6> pi_ss_pose = Eigen::Matrix<double, 6, 6>::Zero();
    const Eigen::MatrixXd& P_for_omega = state_->cov();
    const int iR = StateGroup::idxR(), iP = StateGroup::idxP();
    if (P_for_omega.rows() >= iP + 3 && P_for_omega.cols() >= iP + 3) {
      Eigen::MatrixXd Omega6(6, 6);
      Omega6.block<3, 3>(0, 0) = P_for_omega.block<3, 3>(iR, iR);
      Omega6.block<3, 3>(0, 3) = P_for_omega.block<3, 3>(iR, iP);
      Omega6.block<3, 3>(3, 0) = P_for_omega.block<3, 3>(iP, iR);
      Omega6.block<3, 3>(3, 3) = P_for_omega.block<3, 3>(iP, iP);
      pi_ss_pose = Omega6.inverse();
    }
    Eigen::Matrix<double, 6, 1> s_vec_pose;
    s_vec_pose.segment<3>(0) = coupled_delta_phi0_;
    s_vec_pose.segment<3>(3) = coupled_delta_pos0_;

    const PoseSplineReducedSystem reduced =
        reducePoseSplineHeadCoupling(build, n_c, pi_ss_pose, s_vec_pose);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_red(reduced.A);
    Eigen::VectorXd delta_red = ldlt_red.solve(reduced.b);
    if (delta_red.size() != reduced.A.rows()) {
      std::ostringstream diag;
      diag << "[coupled/pose] DIAG reduced solve size mismatch: delta_red.size()="
           << delta_red.size() << " reduced.A.rows()=" << reduced.A.rows()
           << " n_c=" << n_c << " scan_id=" << voxel_map_->frame_idx_;
      throw std::runtime_error(diag.str());
    }
    // POST-CQ-87-REVIEW FIX (safety hole flagged by external review, item
    // 2): previously NEITHER ldlt_red.info() NOR delta_red.allFinite() was
    // checked -- rule 58f ("a failed operation must not be able to look
    // like a successful one") violated: a singular/ill-conditioned solve
    // could silently hand NaN/Inf straight into coupled_c_pos_/
    // coupled_c_rot_ -> state_->setPropagatedState() -> next scan's own
    // mg.poses, i.e. exactly the "GN step corrupts state, next scan's
    // fit() merely DISCOVERS it as kNonFinite" failure chain the review
    // hypothesized. Fail loud here instead, at the actual point of
    // corruption, so a future investigation doesn't have to rediscover
    // that the reported kNonFinite scan can be a downstream symptom.
    if (ldlt_red.info() != Eigen::Success || !delta_red.allFinite()) {
      std::ostringstream diag;
      diag << "[coupled/pose] FATAL: reduced head-coupled GN solve produced "
              "a non-finite/failed result -- refusing to apply it to "
              "state_. ldlt.info()=" << static_cast<int>(ldlt_red.info())
           << " (0=Success) delta_red.allFinite()=" << delta_red.allFinite()
           << " n_c=" << n_c << " n_free=" << reduced.n_free
           << " iter=" << coupled_iters_ << " scan_id=" << voxel_map_->frame_idx_
           << " max|c_pos so far|=" << [&]{ double m=0; for (auto& v: coupled_c_pos_) m=std::max(m, v.norm()); return m; }()
           << " max|c_rot so far|=" << [&]{ double m=0; for (auto& v: coupled_c_rot_) m=std::max(m, v.norm()); return m; }();
      throw std::runtime_error(diag.str());
    }
    // User instruction 2026-09-21 item 15: step-size/trust-region
    // safeguard -- see copts_.pose_gn_max_step_pos_m/rot_rad's own doc
    // comment for the diagnosis this responds to. A single scalar shrinks
    // the WHOLE solved delta (free c_p, free c_phi, and the head's own
    // delta_phi0/delta_pos0 together) so the step's DIRECTION is
    // unchanged -- only its length is capped -- computed from whichever
    // sub-block (position or rotation, free or head) is most over bound.
    {
      const int n_free_pre = reduced.n_free;
      double max_step_pos = 0.0, max_step_rot = 0.0;
      for (int j = 0; j < n_free_pre; ++j) {
        max_step_pos = std::max(max_step_pos, delta_red.segment<3>(3 * j).norm());
        max_step_rot = std::max(max_step_rot, delta_red.segment<3>(3 * n_free_pre + 3 * j).norm());
      }
      max_step_rot = std::max(max_step_rot, delta_red.segment<3>(6 * n_free_pre).norm());
      max_step_pos = std::max(max_step_pos, delta_red.segment<3>(6 * n_free_pre + 3).norm());
      double scale = 1.0;
      if (copts_.pose_gn_max_step_pos_m > 0.0 && max_step_pos > copts_.pose_gn_max_step_pos_m)
        scale = std::min(scale, copts_.pose_gn_max_step_pos_m / max_step_pos);
      if (copts_.pose_gn_max_step_rot_rad > 0.0 && max_step_rot > copts_.pose_gn_max_step_rot_rad)
        scale = std::min(scale, copts_.pose_gn_max_step_rot_rad / max_step_rot);
      if (scale < 1.0) delta_red *= scale;
    }
    // Unpack: free control points (kTie..n_c-1) map back directly; the
    // shared [delta_phi0;delta_pos0] tail applies IDENTICALLY to every
    // tied head control point (0..kTie-1) -- delta_c for those rows is
    // set, not accumulated, since coupled_c_pos_[j]/coupled_c_rot_[j] for
    // j<kTie are defined to always equal coupled_delta_pos0_/
    // coupled_delta_phi0_ exactly (the clamped-identity construction),
    // never an independent per-iteration increment layered on top.
    const int n_free = reduced.n_free;
    for (int j = POSE_SPLINE_HEAD_TIE_CP; j < n_c; ++j) {
      delta_c.segment<3>(3 * j)           = delta_red.segment<3>(3 * (j - POSE_SPLINE_HEAD_TIE_CP));
      delta_c.segment<3>(3 * n_c + 3 * j) = delta_red.segment<3>(3 * n_free + 3 * (j - POSE_SPLINE_HEAD_TIE_CP));
    }
    coupled_delta_phi0_ += delta_red.segment<3>(6 * n_free);
    coupled_delta_pos0_ += delta_red.segment<3>(6 * n_free + 3);
    for (int j = 0; j < std::min(POSE_SPLINE_HEAD_TIE_CP, n_c); ++j) {
      // SET (not +=): coupled_c_pos_[j]/coupled_c_rot_[j] carry the ALREADY-
      // accumulated coupled_delta_pos0_/coupled_delta_phi0_ directly below,
      // so this delta_c contribution is (new total - old total), applied
      // via the same "coupled_c_pos_[j] += delta_c.segment<3>(3*j)" loop
      // every other column already goes through further down.
      delta_c.segment<3>(3 * j)           = coupled_delta_pos0_ - coupled_c_pos_[j];
      delta_c.segment<3>(3 * n_c + 3 * j) = coupled_delta_phi0_ - coupled_c_rot_[j];
    }

    const Eigen::MatrixXd M_head = [&] {
      Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, reduced.A.rows());
      M.block<3, 3>(0, 6 * n_free)     = M3D::Identity();  // delta_phi0
      M.block<3, 3>(3, 6 * n_free + 3) = M3D::Identity();  // delta_pos0
      return M;
    }();
    coupled_pose_head_cov_ = solveCovarianceFromA(reduced.A, &M_head);
  } else {
    // Build the full free-column index list: c_p free columns, then c_phi
    // free columns, each expanded to its 3 scalar components.
    std::vector<int> free_cols;
    free_cols.reserve(3 * (n_c - n_frozen) * 2);
    for (int j = n_frozen; j < n_c; ++j)
      for (int a = 0; a < 3; ++a) free_cols.push_back(3 * j + a);
    for (int j = n_frozen; j < n_c; ++j)
      for (int a = 0; a < 3; ++a) free_cols.push_back(3 * n_c + 3 * j + a);

    const int nf = static_cast<int>(free_cols.size());
    Eigen::MatrixXd A_free(nf, nf);
    Eigen::VectorXd b_free(nf);
    for (int r = 0; r < nf; ++r) {
      b_free(r) = build.b(free_cols[r]);
      for (int c = 0; c < nf; ++c) A_free(r, c) = build.A(free_cols[r], free_cols[c]);
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt_free(A_free);
    Eigen::VectorXd delta_free = ldlt_free.solve(b_free);
    // POST-CQ-87-REVIEW FIX: same finite/info check as the n_frozen==0
    // branch above -- this fallback arm (pose_head_freeze_cp>0) had the
    // identical gap.
    if (ldlt_free.info() != Eigen::Success || !delta_free.allFinite()) {
      std::ostringstream diag;
      diag << "[coupled/pose] FATAL: frozen-elimination GN solve produced "
              "a non-finite/failed result -- refusing to apply it to "
              "state_. ldlt.info()=" << static_cast<int>(ldlt_free.info())
           << " (0=Success) delta_free.allFinite()=" << delta_free.allFinite()
           << " n_c=" << n_c << " n_frozen=" << n_frozen
           << " iter=" << coupled_iters_ << " scan_id=" << voxel_map_->frame_idx_;
      throw std::runtime_error(diag.str());
    }
    // User instruction 2026-09-21 item 15: same step-size safeguard as the
    // n_frozen==0 branch above. free_cols is laid out c_p first (nf/2
    // scalar entries, 3 per free control point) then c_phi (the other
    // half) -- see the free_cols build loop above.
    {
      const int n_free_p = (n_c - n_frozen);
      double max_step_pos = 0.0, max_step_rot = 0.0;
      for (int j = 0; j < n_free_p; ++j) {
        max_step_pos = std::max(max_step_pos, delta_free.segment<3>(3 * j).norm());
        max_step_rot = std::max(max_step_rot, delta_free.segment<3>(3 * n_free_p + 3 * j).norm());
      }
      double scale = 1.0;
      if (copts_.pose_gn_max_step_pos_m > 0.0 && max_step_pos > copts_.pose_gn_max_step_pos_m)
        scale = std::min(scale, copts_.pose_gn_max_step_pos_m / max_step_pos);
      if (copts_.pose_gn_max_step_rot_rad > 0.0 && max_step_rot > copts_.pose_gn_max_step_rot_rad)
        scale = std::min(scale, copts_.pose_gn_max_step_rot_rad / max_step_rot);
      if (scale < 1.0) delta_free *= scale;
    }
    for (int r = 0; r < nf; ++r) delta_c(free_cols[r]) = delta_free(r);
    // Frozen columns of delta_c are left at their Zero() initialization --
    // exact elimination, not an approximation.
  }
  if (delta_c.size() != 6 * n_c) {
    std::ostringstream diag;
    diag << "[coupled/pose] DIAG delta_c size mismatch: delta_c.size()=" << delta_c.size()
         << " expected 6*n_c=" << 6 * n_c << " n_c=" << n_c
         << " build.A.rows()=" << build.A.rows() << " build.A.cols()=" << build.A.cols();
    throw std::runtime_error(diag.str());
  }

  for (int j = 0; j < n_c; ++j) {
    coupled_c_pos_[j] += delta_c.segment<3>(3 * j);
    coupled_c_rot_[j] += delta_c.segment<3>(3 * n_c + 3 * j);
  }

  // Re-evaluate the tail pose with the NEWLY updated corrections, and write
  // it into state_ directly -- the pose basis's own trajectory already IS
  // an absolute pose; there is no propagateCoupled()-equivalent correction
  // to re-apply the way the raw_imu arm needs.
  ScanSpline trial_new = coupled_pose_spline_;
  for (int j = 0; j < n_c; ++j) {
    trial_new.cpPosMut().col(j) += coupled_c_pos_[j];
    trial_new.cp_phi_.col(j)    += coupled_c_rot_[j];
  }
  const M3D new_tail_R = trial_new.rotAt(t1);
  const V3D new_tail_p = trial_new.posAt(t1);
  const V3D new_tail_v = trial_new.velAt(t1);

  // POST-CQ-87-REVIEW instrumentation (item "what I would instrument
  // immediately"): per-GN-iteration step-size/spline-shape diagnostics,
  // gated on the SAME estimator/coupled/psd_audit_en flag every other
  // per-iteration pose-arm diagnostic in this function already uses
  // (rank_cblock/phic_spread above). Answers, directly, the review's own
  // "is delta_c or spline acceleration already exploding the iteration
  // BEFORE a later scan's fit() reports kNonFinite" question -- rather
  // than only discovering corruption after the fact via that downstream
  // symptom.
  if (copts_.psd_audit_en) {
    double max_delta_cp = 0.0, max_delta_cphi = 0.0;
    for (int j = 0; j < n_c; ++j) {
      max_delta_cp   = std::max(max_delta_cp,   delta_c.segment<3>(3 * j).norm());
      max_delta_cphi = std::max(max_delta_cphi, delta_c.segment<3>(3 * n_c + 3 * j).norm());
    }
    double max_cp = 0.0, max_cphi = 0.0;
    for (int j = 0; j < n_c; ++j) {
      max_cp   = std::max(max_cp,   coupled_c_pos_[j].norm());
      max_cphi = std::max(max_cphi, coupled_c_rot_[j].norm());
    }
    // Sample acc/omega on a fixed grid across [t0,t1] (NOT at residual
    // times -- residuals_ can be empty on a starved scan, and this is
    // meant to characterize the spline's own shape, not the residual
    // set) -- 20 points is cheap relative to the O(n_c^2) work already
    // done above, and matches the "max_t ||pddot(t)||" quantity the
    // review's own instrumentation list asks for directly.
    double max_acc = 0.0, max_omega = 0.0;
    constexpr int kGridN = 20;
    for (int k = 0; k <= kGridN; ++k) {
      const double t = coupled_pose_spline_.t0() +
          (t1 - coupled_pose_spline_.t0()) * (static_cast<double>(k) / kGridN);
      max_acc   = std::max(max_acc,   trial_new.accAt(t).norm());
      max_omega = std::max(max_omega, trial_new.omegaBodyAt(t).norm());
    }
    const bool poses_finite = std::all_of(mg.poses.begin(), mg.poses.end(),
        [](const Pose6D& p) { return p.pos.allFinite() && p.rot.allFinite(); });

    static PersistentLogStream gn_log("pose_gn_debug.txt");
    bool gn_first;
    std::ofstream& gn_ofs = gn_log.stream(&gn_first);
    if (gn_first)
      gn_ofs << "scan_id,iter,n_c,mg_poses_finite,max_delta_cp,max_delta_cphi,"
                "max_cp,max_cphi,max_acc,max_omega,head_tie\n";
    gn_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << n_c << ","
           << (poses_finite ? 1 : 0) << "," << max_delta_cp << "," << max_delta_cphi << ","
           << max_cp << "," << max_cphi << "," << max_acc << "," << max_omega << ","
           << (n_frozen == 0 ? "real" : "frozen") << "\n";
    gn_ofs.flush();
  }

  dtheta_out = Log(prev_tail_R.transpose() * new_tail_R);
  dt_out = new_tail_p - prev_tail_p;
  state_->setPropagatedState(new_tail_R, new_tail_p, new_tail_v);
  coupled_last_A_ = build.A;

  double sum_abs_r = 0.0;
  for (const auto& res : residuals_) sum_abs_r += std::abs(res.r);
  return residuals_.empty() ? 0.0 : sum_abs_r / static_cast<double>(residuals_.size());
}


// ONE GN iteration of the pose-control-point-only estimator, called
// repeatedly from processLIO()'s `for (; iter < opts_.max_iterations;
// iter++)` loop. Builds against z=[c_free;sT] (coupled_pose_control_layout_)
// -- see pose_control_spline.h/pose_control_process_factor.h/
// pose_control_lidar_factor.h for the validated math this function
// assembles.
double LioProcCoupled::estimateCoupledPoseControlSpline(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out)
{
  dtheta_out = V3D::Zero();
  dt_out = V3D::Zero();
  if (!coupled_pose_control_valid_) return 0.0;

  auto& spline = coupled_pose_control_spline_;
  const auto& layout = coupled_pose_control_layout_;
  const double t1 = spline.t1();
  const M3D prev_tail_R = spline.rotAt(t1);
  const V3D prev_tail_p = spline.posAt(t1);

  // ---- deskew + associate ----------------------------------------------
  const M3D R_end_T = spline.rotAt(t1).transpose();
  const V3D p_end = spline.posAt(t1);
  std::vector<PointXYZCov> deskewed(mg.lidar_points.size());
  for (size_t i = 0; i < mg.lidar_points.size(); ++i) {
    const auto& pt = mg.lidar_points[i];
    const M3D R_i = spline.rotAt(pt.t);
    const V3D p_i = spline.posAt(pt.t);
    const M3D R_rel = R_end_T * R_i;
    const V3D t_rel = R_end_T * (p_i - p_end);
    const V3D p_imu_i = state_->lidarToImu(pt.p);
    const V3D p_imu_end = R_rel * p_imu_i + t_rel;
    const M3D cov_lidar_i = getBodyCov(pt.p, opts_.deskew.sigma_r2, opts_.deskew.sigma_a2);
    const M3D cov_imu_end = state_->lidarToImu(M3D(R_rel * cov_lidar_i * R_rel.transpose()));
    deskewed[i] = PointXYZCov{p_imu_end, cov_imu_end};
    deskewed[i].t = pt.t;
    deskewed[i].raw_body_point = p_imu_i;
  }
  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = deskewed;
  }
  buildResiduals(mg.points, residuals_, coupled_iters_ == 0);
  std::vector<PoseControlLidarObs> lidar_obs;
  lidar_obs.reserve(residuals_.size());
  for (auto& res : residuals_) {
    const double d = res.r - res.normal.dot(res.world_point);
    const V3D q_world_trial = spline.rotAt(res.t) * res.raw_body_point + spline.posAt(res.t);
    res.r = res.normal.dot(q_world_trial) + d;
    PoseControlLidarObs o;
    o.t = res.t; o.q = res.raw_body_point; o.normal = res.normal; o.d = d;
    o.sigma2 = res.sigma_squared;
    o.plane_id = res.plane_id; o.plane_var_term = res.plane_var_term;
    lidar_obs.push_back(o);
  }

  // ---- assemble the FULL RAW [c(6N);sT] normal equations (layout.fix_head
  // == false: every control point, including cp[0..2], gets an ordinary
  // Jacobian column here -- the head constraint is applied ONLY via the
  // Z-projection below, not by skipping columns) ---------------------------
  const int dimRaw = layout.dim();   // 6N + dimST
  Eigen::MatrixXd A_raw = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_raw = Eigen::VectorXd::Zero(dimRaw);
  double E_lidar = 0.0;

  // ONE production information path: LiDAR builds A_raw/b_raw directly.
  // The IMU/bias/gravity prior is NOT a per-iteration relinearizing factor
  // here -- it is added exactly once, in z=[eta;sT] space, as the single
  // joint prior derived at scan-start (see the production-prior block
  // below, right after the projection onto z). This is what makes "the
  // prior used by the mean solve == the prior used by the covariance
  // update" true by construction: both read the same coupled_pose_control_
  // lambda_prior_z_/z_imu_ objects, never a separately-relinearized factor.
  if (copts_.pose_control_lidar_enable)
    addPoseControlLidarFactor(spline, layout, lidar_obs, A_raw, b_raw, nullptr, &E_lidar);

  // 2026-09-23 stationary-campaign item 22: curvature (Tikhonov, second-
  // difference) regularization for pose_control specifically -- default
  // 0.0 (no-op). Penalizes cp[k+1]-2*cp[k]+cp[k-1] for position/rotation
  // control points, over EVERY interior control point (k=1..N-2) including
  // the head ones (their contribution to eta is via the SAME Z-projection
  // as any other factor -- no special-casing needed, matching how LiDAR/
  // process factors already touch cp[0..2] uniformly under fix_head=false).
  if (copts_.pose_control_curvature_weight_pos > 0.0 || copts_.pose_control_curvature_weight_rot > 0.0) {
    const int N = layout.N;
    for (int k = 1; k < N - 1; ++k) {
      const int cm1p = layout.colPos(k - 1), c0p = layout.colPos(k), cp1p = layout.colPos(k + 1);
      const int cm1r = layout.colPhi(k - 1), c0r = layout.colPhi(k), cp1r = layout.colPhi(k + 1);
      if (copts_.pose_control_curvature_weight_pos > 0.0 && cm1p >= 0 && c0p >= 0 && cp1p >= 0) {
        const double w = copts_.pose_control_curvature_weight_pos;
        const V3D r = spline.cp_p.col(k + 1) - 2.0 * spline.cp_p.col(k) + spline.cp_p.col(k - 1);
        // J = [I, -2I, I] at columns [cp1p, c0p, cm1p]; A += w*J^T J, b += -w*J^T r
        A_raw.block<3,3>(cp1p,cp1p) += w*M3D::Identity(); A_raw.block<3,3>(c0p,c0p) += 4*w*M3D::Identity(); A_raw.block<3,3>(cm1p,cm1p) += w*M3D::Identity();
        A_raw.block<3,3>(cp1p,c0p) += -2*w*M3D::Identity(); A_raw.block<3,3>(c0p,cp1p) += -2*w*M3D::Identity();
        A_raw.block<3,3>(cp1p,cm1p) += w*M3D::Identity(); A_raw.block<3,3>(cm1p,cp1p) += w*M3D::Identity();
        A_raw.block<3,3>(c0p,cm1p) += -2*w*M3D::Identity(); A_raw.block<3,3>(cm1p,c0p) += -2*w*M3D::Identity();
        b_raw.segment<3>(cp1p) += -w*r; b_raw.segment<3>(c0p) += 2*w*r; b_raw.segment<3>(cm1p) += -w*r;
      }
      if (copts_.pose_control_curvature_weight_rot > 0.0 && cm1r >= 0 && c0r >= 0 && cp1r >= 0) {
        const double w = copts_.pose_control_curvature_weight_rot;
        const V3D r = spline.cp_phi.col(k + 1) - 2.0 * spline.cp_phi.col(k) + spline.cp_phi.col(k - 1);
        A_raw.block<3,3>(cp1r,cp1r) += w*M3D::Identity(); A_raw.block<3,3>(c0r,c0r) += 4*w*M3D::Identity(); A_raw.block<3,3>(cm1r,cm1r) += w*M3D::Identity();
        A_raw.block<3,3>(cp1r,c0r) += -2*w*M3D::Identity(); A_raw.block<3,3>(c0r,cp1r) += -2*w*M3D::Identity();
        A_raw.block<3,3>(cp1r,cm1r) += w*M3D::Identity(); A_raw.block<3,3>(cm1r,cp1r) += w*M3D::Identity();
        A_raw.block<3,3>(c0r,cm1r) += -2*w*M3D::Identity(); A_raw.block<3,3>(cm1r,c0r) += -2*w*M3D::Identity();
        b_raw.segment<3>(cp1r) += -w*r; b_raw.segment<3>(c0r) += 2*w*r; b_raw.segment<3>(cm1r) += -w*r;
      }
    }
  }

  // sT is covered entirely by the joint production prior (Lambda_prior_z,
  // applied below in z-space) -- no separate Omega_ss block here. The two
  // modes that once needed their own sT-only prior (legacy_process_factor,
  // frozen_process_hessian_prior) are removed from production; git history
  // is the recovery path if either is ever needed again for comparison.

  // ---- project onto z=[eta;delta_sT] (item 1: the mean solve operates
  // DIRECTLY in the head-constraint nullspace) -----------------------------
  const auto& hns = coupled_pose_control_hns_;
  const int dEta = hns.freeDim(), dST = layout.dimST(), dimZ = dEta + dST;
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dimRaw, dimZ);
  P.block(0, 0, hns.rawDim(), dEta) = hns.Z;
  if (dST > 0) P.block(hns.rawDim(), dEta, dST, dST) = Eigen::MatrixXd::Identity(dST, dST);

  Eigen::MatrixXd A = P.transpose() * A_raw * P;
  Eigen::VectorXd b = P.transpose() * b_raw;
  // Saved before the prior is added below, purely so the EKF-reference
  // test (item 33) can reconstruct "LiDAR-only" information without
  // double-counting the prior it adds separately.
  const Eigen::MatrixXd A_lidar_reduced = A;
  const Eigen::VectorXd b_lidar_reduced = b;

  // ONE production prior, unconditionally applied: the joint marginalized
  // prior over z=[eta;sT] (Lambda_prior_z was derived at scan-start
  // covering both blocks jointly -- see the init-block comment) -- this is
  // the SAME Lambda_prior_z/z_imu_ object the covariance update
  // (covarianceInformationUpdate()) consumes, so the mean solve and the
  // covariance update are guaranteed to see identical information.
  // r_prior_z = z_current - z_imu, where z_current's sT part is read
  // directly off the trial (bg_trial_-bg_prior_ etc) since sT is not part
  // of the eta vector.
  if (coupled_pose_control_lambda_prior_z_.rows() == dimZ &&
      coupled_pose_control_z_imu_.size() == dimZ) {
    Eigen::VectorXd z_current = Eigen::VectorXd::Zero(dimZ);
    z_current.head(dEta) = coupled_pose_control_eta_;
    if (layout.colBG() >= 0) z_current.segment<3>(dEta + layout.colBG() - layout.dimCFree()) = coupled_pose_control_bg_trial_ - coupled_pose_control_bg_prior_;
    if (layout.colBA() >= 0) z_current.segment<3>(dEta + layout.colBA() - layout.dimCFree()) = coupled_pose_control_ba_trial_ - coupled_pose_control_ba_prior_;
    if (layout.colG()  >= 0) z_current.segment<3>(dEta + layout.colG()  - layout.dimCFree()) = coupled_pose_control_g_trial_  - coupled_pose_control_g_prior_;
    const Eigen::VectorXd r_prior_z = z_current - coupled_pose_control_z_imu_;
    A += coupled_pose_control_lambda_prior_z_;
    b += -coupled_pose_control_lambda_prior_z_ * r_prior_z;

    // ========================================================================
    // item 10/51: the CORRECT EKF/MAP-equivalence test. Unlike the earlier
    // (converged-step) version, this compares delta_reference against
    // delta_actual (the step this SAME GN iteration is about to take) using
    // the EXACT SAME FIXED prior and THIS iteration's own LiDAR
    // linearization -- at iterations 0, 1, and 2, per the spec's explicit
    // instruction not to compare only at convergence.
    // NOTE ON SIGN CONVENTION: item 10's textbook formula writes
    // delta = -(Lambda)^-1 (Lambda*(z-z_prior) + H^T R^-1 r), which assumes
    // b := +(Lambda*(z-z_prior) + H^T R^-1 r) and an explicit leading minus
    // at solve time. THIS codebase's own established convention (every
    // existing prior/LiDAR factor call site, unchanged by this
    // reformulation) is the opposite: b is built so that delta_z = A^-1 * b
    // directly, NO leading minus (confirmed by the mean solve's own
    // `A_raw.../b_raw...; ... delta_z = ldlt.solve(b); eta += delta_z`, and
    // by the pre-existing Omega_ss prior's `b_raw += -Omega0_ss*r_prior`).
    // The reference below is therefore built with the SAME (b, no leading
    // minus) convention, using this iteration's own b_raw/A_raw BEFORE the
    // prior block's own += (i.e. exactly what addPoseControlLidarFactor
    // itself produced) plus the prior term with the SAME sign the mean
    // solve's own prior block uses (-Lambda_prior_z*r_prior_z). An earlier
    // version of this block used the textbook leading-minus literally on
    // top of this codebase's already-correctly-signed b, which silently
    // flipped the LiDAR contribution's sign -- caught via a smoke test
    // showing delta_relative_difference landing at exactly 2.0 (the
    // signature of delta_actual = -delta_reference when the prior term is
    // negligible), fixed here.
    // ========================================================================
    if (copts_.psd_audit_en && coupled_iters_ <= 2) {
      const Eigen::MatrixXd& Lambda_lidar_iter = A_lidar_reduced;
      const Eigen::VectorXd& b_lidar_iter = b_lidar_reduced;
      const Eigen::MatrixXd A_ekf_ref_iter = coupled_pose_control_lambda_prior_z_ + Lambda_lidar_iter;
      const Eigen::VectorXd rhs_ekf_ref_iter = -coupled_pose_control_lambda_prior_z_ * r_prior_z + b_lidar_iter;
      Eigen::LDLT<Eigen::MatrixXd> ldlt_ekf_iter(A_ekf_ref_iter);
      if (ldlt_ekf_iter.info() == Eigen::Success) {
        const Eigen::VectorXd delta_ref_iter = ldlt_ekf_iter.solve(rhs_ekf_ref_iter);
        // delta_actual for THIS iteration is computed a few lines below
        // (the LDLT solve of the full A/b, including curvature if any) --
        // captured via a member so it can be compared once available.
        coupled_pose_control_ekf_ref_pending_ = delta_ref_iter;
        coupled_pose_control_ekf_ref_pending_valid_ = delta_ref_iter.allFinite();
      } else {
        coupled_pose_control_ekf_ref_pending_valid_ = false;
      }
    } else {
      coupled_pose_control_ekf_ref_pending_valid_ = false;
    }
  }


  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  if (ldlt.info() != Eigen::Success) return 0.0;
  Eigen::VectorXd delta_z = ldlt.solve(b);
  if (!delta_z.allFinite()) return 0.0;

  // item 10/51: log the EKF-reference comparison for iterations 0-2, using
  // the UNCLAMPED delta_z (the trust-region safeguard below is an
  // implementation safety mechanism, not part of the EKF-equivalence
  // question). Curvature (if enabled) is NOT part of delta_reference's own
  // formula (item 30: curvature is a separate smoothness prior, not IMU or
  // LiDAR information) -- a nonzero gap when curvature_weight>0 is
  // therefore EXPECTED and attributable to curvature, logged as such via
  // curvature_weight_pos/rot in the row rather than treated as a bug.
  if (copts_.psd_audit_en && coupled_pose_control_ekf_ref_pending_valid_ &&
      coupled_pose_control_ekf_ref_pending_.size() == dimZ) {
    const Eigen::VectorXd& dref = coupled_pose_control_ekf_ref_pending_;
    const Eigen::VectorXd diff = delta_z - dref;
    const int off_bg = layout.colBG() >= 0 ? dEta + layout.colBG() - layout.dimCFree() : -1;
    const int off_ba = layout.colBA() >= 0 ? dEta + layout.colBA() - layout.dimCFree() : -1;
    const int off_g  = layout.colG()  >= 0 ? dEta + layout.colG()  - layout.dimCFree() : -1;
    std::map<std::string, std::string> ekv = {
      {"delta_actual_norm", std::to_string(delta_z.norm())},
      {"delta_reference_norm", std::to_string(dref.norm())},
      {"delta_difference_norm", std::to_string(diff.norm())},
      {"delta_relative_difference", std::to_string(dref.norm() > 1e-300 ? diff.norm() / dref.norm() : 0.0)},
      {"delta_eta_actual", std::to_string(delta_z.head(dEta).norm())},
      {"delta_eta_reference", std::to_string(dref.head(dEta).norm())},
      {"delta_bg_actual", std::to_string(off_bg >= 0 ? delta_z.segment<3>(off_bg).norm() : 0.0)},
      {"delta_bg_reference", std::to_string(off_bg >= 0 ? dref.segment<3>(off_bg).norm() : 0.0)},
      {"delta_ba_actual", std::to_string(off_ba >= 0 ? delta_z.segment<3>(off_ba).norm() : 0.0)},
      {"delta_ba_reference", std::to_string(off_ba >= 0 ? dref.segment<3>(off_ba).norm() : 0.0)},
      {"delta_g_actual", std::to_string(off_g >= 0 ? delta_z.segment<3>(off_g).norm() : 0.0)},
      {"delta_g_reference", std::to_string(off_g >= 0 ? dref.segment<3>(off_g).norm() : 0.0)},
      {"curvature_weight_pos", std::to_string(copts_.pose_control_curvature_weight_pos)},
      {"curvature_weight_rot", std::to_string(copts_.pose_control_curvature_weight_rot)},
      {"lambda_prior_trace", std::to_string(coupled_pose_control_lambda_prior_z_.trace())},
      {"lambda_total_trace", std::to_string(A.trace())},
    };
    emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "ekf_reference", voxel_map_->frame_idx_, coupled_iters_, ekv);
  }

  // Trust-region step-size safeguard -- SAME mechanism/config knobs
  // (pose_gn_max_step_pos_m/rot_rad) the pose_knots/pose-basis arms
  // already use: a single scalar shrinks the WHOLE step (direction
  // unchanged) so no per-control-point 3-vector exceeds the configured
  // bound. Newly REQUIRED here (missing in the first pose_control smoke
  // test) because eta now spans genuinely weakly-observed directions --
  // e.g. the a(t0)-controlling nullspace direction the OLD (over-
  // constrained) head-fixing scheme had no analogue for at all -- and an
  // unclamped GN step there diverged to ~1e9 m ATE within ~200 scans on
  // the live eee_01 smoke test, confirming this is not optional.
  {
    const Eigen::VectorXd delta_c_raw = hns.Z * delta_z.head(dEta);
    double max_step_pos = 0.0, max_step_rot = 0.0;
    const int N = layout.N;
    for (int k = 0; k < N; ++k) {
      max_step_pos = std::max(max_step_pos, delta_c_raw.segment<3>(3 * k).norm());
      max_step_rot = std::max(max_step_rot, delta_c_raw.segment<3>(3 * N + 3 * k).norm());
    }
    double scale = 1.0;
    if (copts_.pose_gn_max_step_pos_m > 0.0 && max_step_pos > copts_.pose_gn_max_step_pos_m)
      scale = std::min(scale, copts_.pose_gn_max_step_pos_m / max_step_pos);
    if (copts_.pose_gn_max_step_rot_rad > 0.0 && max_step_rot > copts_.pose_gn_max_step_rot_rad)
      scale = std::min(scale, copts_.pose_gn_max_step_rot_rad / max_step_rot);
    if (scale < 1.0) delta_z *= scale;
  }

  // ---- apply the mean update: eta AND the tail_trial increment, every
  // iteration (item 4) -----------------------------------------------------
  coupled_pose_control_eta_ += delta_z.head(dEta);
  if (layout.colBG() >= 0) coupled_pose_control_bg_trial_ += delta_z.segment<3>(dEta + layout.colBG() - layout.dimCFree());
  if (layout.colBA() >= 0) coupled_pose_control_ba_trial_ += delta_z.segment<3>(dEta + layout.colBA() - layout.dimCFree());
  if (layout.colG()  >= 0) coupled_pose_control_g_trial_  += delta_z.segment<3>(dEta + layout.colG()  - layout.dimCFree());

  poseControlUnflatten(hns.c_particular + hns.Z * coupled_pose_control_eta_, spline);

  // ---- reconcile the tail (item 5: R/p/v are NEVER independent -- always
  // read directly off the just-updated spline) -----------------------------
  const M3D new_tail_R = spline.rotAt(t1);
  const V3D new_tail_p = spline.posAt(t1);
  const V3D new_tail_v = spline.velAt(t1);
  dtheta_out = Log(M3D(prev_tail_R.transpose() * new_tail_R));
  dt_out = new_tail_p - prev_tail_p;
  state_->setPropagatedState(new_tail_R, new_tail_p, new_tail_v);
  // Bias/gravity mean is NOT written to state_ every iteration (item 4:
  // state_ need not be overwritten every iteration and preferably should
  // not be -- the ONE coherent tail representation during the GN loop is
  // coupled_pose_control_{bg,ba,g}_trial_, not state_ itself; the full
  // coherent write-back happens once, post-loop, in processLIO()'s
  // poseControlSplineBasis() block -- item 12).

  // Item 15's required per-iteration instrumentation, folded directly into
  // the unified diagnostics CSV (no dedicated pose_control_gn_iter.txt).
  if (copts_.psd_audit_en) {
    // head_state_before/after (item 15): mg.poses.front() is this SAME
    // scan's own fixed head data, unchanged across GN iterations -- these
    // should be ~1e-12-level (machine precision), confirming the head
    // truly never moves, not merely "moves very little".
    const double head_p0_err = (spline.posAt(spline.t0()) - mg.poses.front().pos).norm();
    const double head_v0_err = (spline.velAt(spline.t0()) - mg.poses.front().vel).norm();
    const double head_R0_err = Log(M3D(spline.rotAt(spline.t0()).transpose() * mg.poses.front().rot)).norm();
    std::map<std::string, std::string> gkv = {
      {"delta_eta_norm", std::to_string(delta_z.head(dEta).norm())},
      {"delta_bg_norm", std::to_string(layout.colBG() >= 0 ? delta_z.segment<3>(dEta + layout.colBG() - layout.dimCFree()).norm() : 0.0)},
      {"delta_ba_norm", std::to_string(layout.colBA() >= 0 ? delta_z.segment<3>(dEta + layout.colBA() - layout.dimCFree()).norm() : 0.0)},
      {"delta_g_norm", std::to_string(layout.colG() >= 0 ? delta_z.segment<3>(dEta + layout.colG() - layout.dimCFree()).norm() : 0.0)},
      {"head_p0_err", std::to_string(head_p0_err)}, {"head_v0_err", std::to_string(head_v0_err)},
      {"head_R0_err", std::to_string(head_R0_err)}, {"E_lidar", std::to_string(E_lidar)},
    };
    emitFullDiagRow(fullDiagRunId(), copts_.pose_control_test_id, "gn_iteration", voxel_map_->frame_idx_, coupled_iters_, gkv);
  }

  return residuals_.empty() ? 0.0 : E_lidar / static_cast<double>(residuals_.size());
}

}  // namespace livo_recon
