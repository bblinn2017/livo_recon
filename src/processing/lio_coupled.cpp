#include "livo_recon/processing/lio_coupled.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/map/voxelmap.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

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
                 copts_.spline_mode, "raw_imu", {"raw_imu", "pose", "pose_knots"});
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
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/n_knots", copts_.pose_knots_n, 13);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/smoothness_position", copts_.pose_knots_smoothness_pos, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/smoothness_rotation", copts_.pose_knots_smoothness_rot, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/use_imu_factors", copts_.pose_knots_use_imu_factors, true);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/q_pinv_rel_thresh", copts_.pose_knots_q_pinv_rel_thresh, 1e-6);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/deterministic_constraint_enable", copts_.pose_knots_det_constraint_en, false);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/deterministic_constraint_weight", copts_.pose_knots_det_constraint_weight, 1.0e4);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/exact_deterministic_constraint_enable", copts_.pose_knots_exact_det_constraint_en, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/pose_knots/relinearize_fq", copts_.pose_knots_relinearize_fq, true);
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
  // User instruction 2026-09-21 items 1-6/18/19: the physical-knot arm's
  // own scan-start init, mirroring the poseBasis() block above but built
  // ONCE via PoseKnotSpline::init() rather than ScanSpline::fit() --
  // item 18's prior (P0 = state_->cov()'s own [theta,p,v] block) and item
  // 3's own initialization from mg.poses are both handled inside init()
  // itself; there is no separate setFrozenBoundary()-style hard-clamp call
  // here at all (items 4/19: the head is a SOFT prior consumed by the
  // solver's own prior factor, the tail is free by construction -- see
  // pose_knot_spline.h's own doc comment for why the old clamped-B-spline
  // head-tie machinery has no analogue needed here).
  if (copts_.poseKnotsBasis()) {
    coupled_pose_knots_valid_ = false;
    coupled_knot_delta_theta_.clear();
    coupled_knot_delta_pos_.clear();
    coupled_knot_delta_vel_.clear();
    coupled_knot_cov_.clear();
    if (!mg.poses.empty() && mg.image.t > mg.poses.front().t) {
      const double t0 = mg.poses.front().t;
      const double t1 = mg.image.t;
      Eigen::Matrix<double, 9, 9> P0 = Eigen::Matrix<double, 9, 9>::Zero();
      // POST-REVIEW FIX (external review 2026-09-21, item 1, "P_prior for
      // pose knots is still using the wrong covariance epoch"):
      // processIMU() runs before processLIO(), and has already done
      // P <- F P F^T + Q -- so state_->cov() here is the POST-propagation
      // covariance, not the pre-scan one the raw_imu arm's own Pi_ss
      // already knows to correct for via prior_at_scan_start (see that
      // option's own use at ~line 2550, the identical pattern applied
      // here). Falls back to state_->cov() if the peek isn't primed, same
      // as the raw_imu arm's own fallback.
      Eigen::MatrixXd P_full = state_->cov();
      if (copts_.prior_at_scan_start) {
        Eigen::MatrixXd p_before_peek;
        if (imuProcQhatPeekPBefore(p_before_peek) &&
            p_before_peek.rows() == P_full.rows() && p_before_peek.cols() == P_full.cols()) {
          P_full = p_before_peek;
        }
      }
      if (P_full.rows() >= 9 && P_full.cols() >= 9) P0 = P_full.block<9, 9>(0, 0);
      // POST-REVIEW FIX: init() now walks mg.imu_samples_raw directly
      // (previously silently ignored -- the review's own "imu_raw isn't
      // actually used" finding) to build knots whose nominal states are
      // the actual integrated IMU trajectory, not an independent
      // interpolation of mg.poses.
      coupled_pose_knots_valid_ = coupled_pose_knots_.init(
          t0, t1, copts_.pose_knots_n,
          mg.poses.front().pos, mg.poses.front().rot, mg.poses.front().vel, P0,
          mg.imu_samples_raw, state_->biasAcc(), state_->biasGyr(), state_->gravity(),
          copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr, state_->varAcc(), state_->varGyr(),
          copts_.repro_second_order);
      if (coupled_pose_knots_valid_) {
        const int n = coupled_pose_knots_.nKnots();
        coupled_knot_delta_theta_.assign(n, V3D::Zero());
        coupled_knot_delta_pos_.assign(n, V3D::Zero());
        coupled_knot_delta_vel_.assign(n, V3D::Zero());
        coupled_knot_cov_.assign(n, Eigen::Matrix<double, 9, 9>::Zero());
        pose_knots_frozen_F9_.assign(std::max(n - 1, 0), Eigen::Matrix<double, 9, 9>::Zero());
        pose_knots_frozen_Q9_.assign(std::max(n - 1, 0), Eigen::Matrix<double, 9, 9>::Zero());
        pose_knots_frozen_fq_valid_ = false;
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

  // User instruction 2026-09-21 items 12/13/18/27: the physical-knot
  // arm's own early return, parallel to poseBasis() above -- same
  // rationale (the raw_imu-specific post-loop bookkeeping below has no
  // analogue for this arm's own [theta,p,v]-only, N-knot state).
  if (copts_.poseKnotsBasis()) {
    boundary_dpos_ = 0.0;
    boundary_drot_deg_ = 0.0;
    // Item 18's own "this is also where your previously identified
    // P(t0)/P(t1) issue needs to be resolved" -- the posterior write here
    // uses the TAIL knot's own marginal (items 12/13), read directly off
    // the joint solve's A^-1 diagonal block, not a basis-coefficient
    // translation. Rotation/position/velocity are ALL written (unlike the
    // coefficient-basis arm, which could only ever offer [rot,pos] since
    // it never solved velocity jointly) -- bias/gravity remain untouched
    // (item 22: held fixed this round).
    if (!coupled_knot_cov_.empty()) {
      Eigen::MatrixXd P_full = state_->cov();
      const int iR = StateGroup::idxR(), iP = StateGroup::idxP(), iV = StateGroup::idxV();
      if (P_full.rows() >= iV + 3 && P_full.cols() >= iV + 3) {
        const auto& Ptail = coupled_knot_cov_.back();
        P_full.block<3, 3>(iR, iR) = Ptail.block<3, 3>(0, 0);
        P_full.block<3, 3>(iR, iP) = Ptail.block<3, 3>(0, 3);
        P_full.block<3, 3>(iP, iR) = Ptail.block<3, 3>(3, 0);
        P_full.block<3, 3>(iP, iP) = Ptail.block<3, 3>(3, 3);
        P_full.block<3, 3>(iR, iV) = Ptail.block<3, 3>(0, 6);
        P_full.block<3, 3>(iV, iR) = Ptail.block<3, 3>(6, 0);
        P_full.block<3, 3>(iP, iV) = Ptail.block<3, 3>(3, 6);
        P_full.block<3, 3>(iV, iP) = Ptail.block<3, 3>(6, 3);
        P_full.block<3, 3>(iV, iV) = Ptail.block<3, 3>(6, 6);
        state_->covMut() = P_full;
      }
    }
    coupled_diag.n_residuals = static_cast<int>(residuals_.size());
    coupled_diag.n_c_requested = copts_.pose_knots_n;
    coupled_diag.n_c_actual    = coupled_pose_knots_.nKnots();
    coupled_diag.n_c_clamped   = 0;  // PoseKnotSpline::init() has no clamp -- n_knots is exact, always.
    coupled_diag.n_imu_samples = mg.n_imu_samples;
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
    oss << "[lio/ekf][pose-knots] iters=" << iter + 1 << "  stop=" << stop
        << std::scientific << std::setprecision(1)
        << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
        << "  |dt|=" << total_dt.norm() * 1000.0 << " mm"
        << "  n_knots=" << coupled_pose_knots_.nKnots()
        << "  (post-loop bg-projection/final_redeskew/most CQ-53-era "
        << "diagnostics SKIPPED -- raw_imu-specific, no analogue for this "
        << "arm's own N-knot state, same as the coefficient-basis arm)";
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
  if (copts_.poseKnotsBasis()) return estimateCoupledPoseKnotSpline(mg, dtheta_out, dt_out);

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

// ============================================================================
// User instruction 2026-09-21 items 1-27 ("do all of them"): the physical-
// knot pose spline. See pose_knot_spline.h's own doc comment for the
// architecture and this file's own scan-start init block (above) for
// items 3/5/6/18. Every GN iteration:
//   1. build a TRIAL copy of coupled_pose_knots_ with this scan's
//      accumulated corrections applied (mirrors estimateCoupledCorrection-
//      PoseBasis()'s own `trial` pattern exactly);
//   2. deskew + buildResiduals() against it (item 8, point_time/end_time
//      per item 9/10, via a local deskew loop -- deskewPointsSpline()
//      itself is ScanSpline-specific, not reusable for this class);
//   3. assemble the JOINT (9*N)x(9*N) information system (item 14's
//      "mathematically clean" batch form, not independent per-knot
//      updates): a head prior (item 18), IMU process factors between
//      adjacent knots (item 16, using the CACHED per-segment F9/Q9 from
//      init() -- see PoseKnotSpline::segF9/segQ9's own doc comment for the
//      named simplification), an optional discrete smoothness prior (item
//      20's own "simpler first implementation" fallback, off by default),
//      and LiDAR factors (item 8, analytic position/velocity Jacobian via
//      the Hermite weights, FD rotation Jacobian per item 16's own
//      allowance to "retain FD initially for validation");
//   4. the SAME step-size trust-region safeguard the coefficient-basis arm
//      needed (copts_.pose_gn_max_step_pos_m/rot_rad) -- carried over as a
//      precaution, not asked for again in this item list but clearly still
//      warranted given the earlier round's own findings;
//   5. solve, apply, and -- since x_j IS the physical knot state directly
//      here -- read Cov(x_j,x_j) straight off A^-1's own diagonal blocks
//      (items 12/13/27), no basis-coefficient Jacobian chaining needed at
//      all (pose_knot_spline.h's own point).
// ============================================================================
double LioProcCoupled::estimateCoupledPoseKnotSpline(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out)
{
  dtheta_out = V3D::Zero();
  dt_out = V3D::Zero();
  if (!coupled_pose_knots_valid_) {
    std::ostringstream diag;
    diag << "[coupled/pose_knots] estimateCoupledPoseKnotSpline(): this "
            "scan's PoseKnotSpline::init() failed or never ran at scan "
            "start -- no valid initial trajectory to correct. "
            "mg.poses.size()=" << mg.poses.size()
         << " n_knots=" << copts_.pose_knots_n << " scan_id=" << voxel_map_->frame_idx_;
    throw std::runtime_error(diag.str());
  }

  const double t1 = mg.image.t;
  const int N = coupled_pose_knots_.nKnots();
  if (static_cast<int>(coupled_knot_delta_theta_.size()) != N ||
      static_cast<int>(coupled_knot_delta_pos_.size()) != N ||
      static_cast<int>(coupled_knot_delta_vel_.size()) != N) {
    std::ostringstream diag;
    diag << "[coupled/pose_knots] DIAG size mismatch: N=" << N
         << " delta_theta.size()=" << coupled_knot_delta_theta_.size()
         << " delta_pos.size()=" << coupled_knot_delta_pos_.size()
         << " delta_vel.size()=" << coupled_knot_delta_vel_.size();
    throw std::runtime_error(diag.str());
  }

  // Trial: coupled_pose_knots_ itself is FIXED (this scan's one-time
  // init() result, never mutated) -- a copy with this scan's accumulated
  // corrections applied, exactly mirroring the coefficient-basis arm's own
  // `ScanSpline trial = coupled_pose_spline_` pattern.
  PoseKnotSpline trial = coupled_pose_knots_;
  for (int j = 0; j < N; ++j) {
    trial.knotMut(j).pos = coupled_pose_knots_.knot(j).pos + coupled_knot_delta_pos_[j];
    trial.knotMut(j).rot = coupled_pose_knots_.knot(j).rot * Exp(coupled_knot_delta_theta_[j]);
    trial.knotMut(j).vel = coupled_pose_knots_.knot(j).vel + coupled_knot_delta_vel_[j];
  }
  const M3D prev_tail_R = trial.rotationAt(t1);
  const V3D prev_tail_p = trial.positionAt(t1);

  // Item 8: deskew every point at its OWN capture-time pose against the
  // trial trajectory, expressed in the scan-end frame -- the same
  // computation deskewOnePointSpline() (lio/deskew.cpp) performs for
  // ScanSpline, re-derived here since that function is not generic over
  // PoseKnotSpline.
  const M3D R_end_T = trial.rotationAt(t1).transpose();
  const V3D p_end = trial.positionAt(t1);
  std::vector<PointXYZCov> deskewed(mg.lidar_points.size());
  for (size_t i = 0; i < mg.lidar_points.size(); ++i) {
    const auto& pt = mg.lidar_points[i];
    const M3D R_i = trial.rotationAt(pt.t);
    const V3D p_i = trial.positionAt(pt.t);
    const M3D R_rel = R_end_T * R_i;
    const V3D t_rel = R_end_T * (p_i - p_end);
    const V3D p_imu_i = state_->lidarToImu(pt.p);
    const V3D p_imu_end = R_rel * p_imu_i + t_rel;
    const M3D cov_lidar_i = getBodyCov(pt.p, opts_.deskew.sigma_r2, opts_.deskew.sigma_a2);
    const M3D cov_imu_end = state_->lidarToImu(M3D(R_rel * cov_lidar_i * R_rel.transpose()));
    deskewed[i] = PointXYZCov{p_imu_end, cov_imu_end};
    deskewed[i].t = pt.t;
    // POST-REVIEW FIX (external review 2026-09-21, "the LiDAR residuals...
    // are not actually evaluated against the trial pose-spline trajectory
    // that the comments say they are"): raw_body_point was NOT being set
    // here at all (defaulting to V3D::Zero()), even though it's exactly
    // what the world-frame residual recomputation below needs -- the
    // pre-warp, own-capture-time IMU-frame point, same convention
    // deskewOnePointSpline() itself uses for the SAME field (deskew.cpp:
    // "out.raw_body_point = p_imu_i").
    deskewed[i].raw_body_point = p_imu_i;
  }
  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = deskewed;
  }
  buildResiduals(mg.points, residuals_, coupled_iters_ == 0);

  // POST-REVIEW FIX, the P0 item ("the most important one"): buildResiduals()
  // internally does `state_->toWorld(pts[i])` (lio_base.cpp) -- state_'s
  // CURRENT pose, which state_->setPropagatedState() only updates at the
  // very END of this function, i.e. from the PREVIOUS GN iteration's own
  // result, not this iteration's trial. So every res.r above was computed
  // against a point expressed via a STALE frame, inconsistent with the
  // trial-based deskew that chose which plane it matched and inconsistent
  // with the Jrow Jacobian built below (which IS w.r.t. trial). Fixed by
  // adopting the review's own "Option A": recompute each residual's r
  // directly in WORLD frame via trial, using the identity that a plane's
  // own (n,d) doesn't depend on which point evaluation is used --
  // d = res.r_stale - n.dot(res.world_point_stale) is recovered exactly
  // from what buildResiduals() already computed, then r is rebuilt at
  // trial's own q_i^w = R(t_i)*raw_body_point + p(t_i). This is EXACT, not
  // an approximation (correspondence/plane MATCHING still uses the stale
  // point -- a standard, acceptable ICP-style choice, since which plane a
  // point associates with is a discrete decision refreshed every
  // iteration regardless; only the residual VALUE and its Jacobian must be
  // mutually consistent, which they now are). This ALSO resolves the
  // review's second finding (the Jacobian missing the scan-end-pose
  // dependence term) for free: a world-frame residual has NO p(t1)/R(t1)
  // dependence at all, so the existing Jrow (already built purely w.r.t.
  // knot j/j+1 at t_i, never referencing the endpoint) is now correct as
  // written, rather than needing a second Jacobian block added.
  for (auto& res : residuals_) {
    const double d = res.r - res.normal.dot(res.world_point);
    const V3D q_world_trial = trial.rotationAt(res.t) * res.raw_body_point + trial.positionAt(res.t);
    res.r = res.normal.dot(q_world_trial) + d;
  }

  const bool use_end_time = (copts_.jacobian_time_mode == "end_time");

  constexpr int kDim = 9;
  const int total = kDim * N;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(total, total);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(total);
  // Small fixed floor, same role as POSE_SPLINE_TIKHONOV_EPS -- keeps A
  // invertible even with every other term off (e.g. use_imu_factors=false
  // and no LiDAR residuals this scan).
  A.diagonal().array() += 1e-6;

  // POST-REVIEW ADDITION (item 11, "factor-scale decomposition... this is
  // how I'd answer the remaining fundamental question: why does the
  // optimizer want ~0.5m of stationary motion? Without this, tuning Q,
  // curvature, or LiDAR weights is mostly guesswork"): mirror every
  // addition to A into the matching per-FACTOR-TYPE accumulator below, so
  // each term's own eigenstructure/trace can be inspected in isolation.
  // Zero extra cost when psd_audit_en is off (the accumulation itself is
  // cheap -- one extra += per block write -- only the eigendecompositions
  // at the bottom are gated).
  Eigen::MatrixXd A_prior = Eigen::MatrixXd::Zero(total, total);
  Eigen::MatrixXd A_imu = Eigen::MatrixXd::Zero(total, total);
  Eigen::MatrixXd A_lidar = Eigen::MatrixXd::Zero(total, total);
  Eigen::MatrixXd A_smooth = Eigen::MatrixXd::Zero(total, total);
  // Phase-2 diagnostic accumulator (deterministic-nullspace soft penalty),
  // isolated the same way A_imu/A_lidar/etc are so v_min^T A_det v_min can
  // be reported below alongside the other factor-strength numbers.
  Eigen::MatrixXd A_det = Eigen::MatrixXd::Zero(total, total);
  // Phase-2 telemetry: actual scalar objective-value contributions
  // (r^T W r per factor, not just A's quadratic form at some arbitrary
  // vector), accumulated alongside their matrix blocks below so the GN
  // telemetry log can report E_lidar/E_process/E_det/E_total per iteration.
  double e_lidar_total = 0.0, e_process_total = 0.0, e_det_total = 0.0;

  // Phase-3 (2026-09-22): exact deterministic equality constraint, via
  // nullspace elimination -- C_exact stacks each segment's own
  // C_j = N_j^T[-F_j I] (3x18, embedded into the 3x(9N) row block for
  // knots j,j+1), d_exact stacks each segment's own -N_j^T*r_j (the
  // target that drives N_j^T r toward zero over this GN step). Built
  // only when pose_knots_exact_det_constraint_en && pose_knots_use_imu_factors.
  const bool exact_det_en = copts_.pose_knots_exact_det_constraint_en && copts_.pose_knots_use_imu_factors;
  const int n_exact_constraints = exact_det_en ? 3 * (N - 1) : 0;
  Eigen::MatrixXd C_exact = Eigen::MatrixXd::Zero(n_exact_constraints, total);
  Eigen::VectorXd d_exact = Eigen::VectorXd::Zero(n_exact_constraints);
  if (copts_.pose_knots_exact_det_constraint_en && copts_.pose_knots_det_constraint_en) {
    ROS_WARN_STREAM_ONCE(
        "[coupled/pose_knots] both pose_knots_exact_det_constraint_en and "
        "pose_knots_det_constraint_en are set -- the exact constraint takes "
        "precedence, the soft penalty is skipped entirely (mutually "
        "exclusive alternatives, not additive).");
  }

  // POST-REVIEW FIX (external review 2026-09-21, "Q9 is rank-deficient
  // before the artificial floor... the code is artificially filling those
  // null directions with 1e-9... that can create enormous artificial
  // information along directions that have essentially zero modeled
  // process noise"): a flat additive 1e-9 on Q9's diagonal is dimensionally
  // meaningless (mixes rad^2/m^2/(m/s)^2) AND, for the structurally
  // low-rank position/velocity sub-block (both derive from the same
  // second-order acceleration noise), can dominate the inverse along the
  // null directions that floor is filling. Fixed via a proper SPD
  // pseudo-inverse: eigendecompose the symmetrized matrix, invert only
  // eigenvalues above a RELATIVE (dimensionless) threshold, treat smaller
  // ones as exactly-zero information rather than 1e9-scale fake precision.
  auto pseudoInverse9 = [](const Eigen::Matrix<double, 9, 9>& M, double rel_thresh) {
    const Eigen::Matrix<double, 9, 9> Ms = 0.5 * (M + M.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(Ms);
    const auto& evals = es.eigenvalues();
    const auto& evecs = es.eigenvectors();
    const double thresh = rel_thresh * std::max(evals.maxCoeff(), 0.0);
    Eigen::Matrix<double, 9, 9> out = Eigen::Matrix<double, 9, 9>::Zero();
    for (int k = 0; k < 9; ++k)
      if (evals(k) > thresh) out += (1.0 / evals(k)) * (evecs.col(k) * evecs.col(k).transpose());
    return out;
  };

  // ---- item 18: head prior, Omega0 = pinv(P0) (P0 == knot 0's own P, set
  // ONCE at init() from state_->cov(), never mutated) ----
  {
    const Eigen::Matrix<double, 9, 9> Omega0 = pseudoInverse9(coupled_pose_knots_.knot(0).P_prior, 1e-6);
    Eigen::Matrix<double, 9, 1> s0;
    s0.segment<3>(0) = coupled_knot_delta_theta_[0];
    s0.segment<3>(3) = coupled_knot_delta_pos_[0];
    s0.segment<3>(6) = coupled_knot_delta_vel_[0];
    A.block<9, 9>(0, 0) += Omega0;
    A_prior.block<9, 9>(0, 0) += Omega0;
    b.segment<9>(0) -= Omega0 * s0;
  }

  // ---- item 16: IMU process factors between adjacent knots ----
  if (copts_.pose_knots_use_imu_factors) {
    for (int j = 0; j + 1 < N; ++j) {
      // POST-REVIEW FIX (item 2, "the process Jacobians F_j are frozen at
      // initialization... recompute F_j(k)=F(x_j(k),u_j) at every GN
      // iteration"): re-walk this segment's own cached raw-sample
      // sequence starting from the TRIAL's current knot-j state (not the
      // frozen nominal) -- F9's world-frame acceleration terms depend on
      // rot_imu at each micro-step, so this genuinely changes once the
      // trajectory has moved (which the earlier stationary-window
      // finding -- up to ~0.5m -- says it does, non-negligibly).
      // Phase-4 (2026-09-22): pose_knots_relinearize_fq gates whether F9/Q9
      // are recomputed from the trial's CURRENT (moving) state every GN
      // iteration (true, unchanged prior behavior) or relinearized ONCE
      // per scan at iteration 0 (from the still-zero-delta trial, i.e.
      // the scan-start nominal) and held fixed for the rest of the scan's
      // iterations (false). The iter-0 result is ALWAYS cached into
      // pose_knots_frozen_F9_/Q9_ (regardless of this flag) so the
      // relinearized-run diagnostic below (||F^(k)-F^(0)||_F/||F^(0)||_F)
      // has a baseline to compare against even when relinearize_fq=true.
      Eigen::Matrix<double, 9, 9> F9, Q9;
      if (copts_.pose_knots_relinearize_fq || coupled_iters_ == 0) {
        coupled_pose_knots_.relinearizeSegment(
            j, trial.knot(j).rot, trial.knot(j).pos, trial.knot(j).vel,
            state_->biasAcc(), state_->biasGyr(), state_->gravity(),
            copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr,
            state_->varAcc(), state_->varGyr(), copts_.repro_second_order, F9, Q9);
        if (coupled_iters_ == 0 && j < static_cast<int>(pose_knots_frozen_F9_.size())) {
          pose_knots_frozen_F9_[j] = F9;
          pose_knots_frozen_Q9_[j] = Q9;
          pose_knots_frozen_fq_valid_ = true;
        }
      } else {
        F9 = pose_knots_frozen_F9_[j];
        Q9 = pose_knots_frozen_Q9_[j];
      }
      if (copts_.psd_audit_en && copts_.pose_knots_relinearize_fq && coupled_iters_ > 0 &&
          pose_knots_frozen_fq_valid_ && (j == 0 || j == 6)) {
        const Eigen::Matrix<double, 9, 9>& F0 = pose_knots_frozen_F9_[j];
        const Eigen::Matrix<double, 9, 9>& Q0 = pose_knots_frozen_Q9_[j];
        const double f_rel = F0.norm() > 1e-300 ? (F9 - F0).norm() / F0.norm() : -1.0;
        const double q_rel = Q0.norm() > 1e-300 ? (Q9 - Q0).norm() / Q0.norm() : -1.0;
        static PersistentLogStream fq_drift_log("pose_knots_fq_relin_drift.txt");
        bool fq_drift_first;
        std::ofstream& fq_drift_ofs = fq_drift_log.stream(&fq_drift_first);
        if (fq_drift_first) fq_drift_ofs << "scan_id,iter,seg,F_rel_fro_diff,Q_rel_fro_diff\n";
        fq_drift_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << j << ","
                     << f_rel << "," << q_rel << "\n";
        fq_drift_ofs.flush();
      }
      // POST-REVIEW FIX (item 3): threshold is now a live config knob
      // (default 1e-6, unchanged) rather than hardcoded -- see
      // copts_.pose_knots_q_pinv_rel_thresh's own doc comment for the
      // sensitivity-sweep this is meant to enable.
      //
      // Phase-3 (2026-09-22): when exact_det_en, Lambda is instead built
      // from the SAME eigendecomposition that supplies N_j below (6
      // non-null directions only), rather than the separately-thresholded
      // q_pinv_rel_thresh -- guarantees the stochastic term and the exact
      // equality constraint partition Q9's spectrum with no gap/overlap
      // (the 3 null directions get EXACTLY zero stochastic weight here,
      // since that information is now supplied by C_exact/d_exact
      // instead, not by an independently-thresholded pseudoinverse that
      // might not zero exactly the same 3 directions).
      Eigen::Matrix<double, 9, 9> Lambda;
      Eigen::Matrix<double, 9, 3> N3;
      if (exact_det_en) {
        const Eigen::Matrix<double, 9, 9> Qs = 0.5 * (Q9 + Q9.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es_q(Qs);
        N3 = es_q.eigenvectors().leftCols(3);
        Lambda.setZero();
        for (int k = 3; k < 9; ++k) {
          const double ev = es_q.eigenvalues()(k);
          if (ev > 1e-300) {
            const Eigen::Matrix<double, 9, 1> ek = es_q.eigenvectors().col(k);
            Lambda += (1.0 / ev) * (ek * ek.transpose());
          }
        }
      } else {
        Lambda = pseudoInverse9(Q9, copts_.pose_knots_q_pinv_rel_thresh);
      }
      Eigen::Matrix<double, 9, 1> xj, xj1;
      xj.segment<3>(0) = coupled_knot_delta_theta_[j];
      xj.segment<3>(3) = coupled_knot_delta_pos_[j];
      xj.segment<3>(6) = coupled_knot_delta_vel_[j];
      xj1.segment<3>(0) = coupled_knot_delta_theta_[j + 1];
      xj1.segment<3>(3) = coupled_knot_delta_pos_[j + 1];
      xj1.segment<3>(6) = coupled_knot_delta_vel_[j + 1];
      const Eigen::Matrix<double, 9, 1> r = xj1 - F9 * xj;
      const Eigen::Matrix<double, 9, 9> FtL = F9.transpose() * Lambda;
      A.block<9, 9>(kDim * j, kDim * j)         += FtL * F9;
      A.block<9, 9>(kDim * j, kDim * (j + 1))   += -FtL;
      A.block<9, 9>(kDim * (j + 1), kDim * j)   += -Lambda * F9;
      A.block<9, 9>(kDim * (j + 1), kDim * (j + 1)) += Lambda;
      A_imu.block<9, 9>(kDim * j, kDim * j)         += FtL * F9;
      A_imu.block<9, 9>(kDim * j, kDim * (j + 1))   += -FtL;
      A_imu.block<9, 9>(kDim * (j + 1), kDim * j)   += -Lambda * F9;
      A_imu.block<9, 9>(kDim * (j + 1), kDim * (j + 1)) += Lambda;
      b.segment<9>(kDim * j)       += FtL * r;
      b.segment<9>(kDim * (j + 1)) += -Lambda * r;
      e_process_total += (r.transpose() * Lambda * r)(0);

      // Phase-3: fill this segment's own 3 constraint rows. C_j = N_j^T*J
      // where J=[-F9,I] acts on [xj;xj1] -- embedded at columns
      // [kDim*j:kDim*j+9] (=-N3^T*F9) and [kDim*(j+1):kDim*(j+1)+9]
      // (=N3^T). Target d_j = -N3^T*r drives N3^T*r toward zero over this
      // GN step (r linearizes as r+J*delta, want N3^T*(r+J*delta)=0).
      if (exact_det_en) {
        C_exact.block<3, 9>(3 * j, kDim * j)       = -N3.transpose() * F9;
        C_exact.block<3, 9>(3 * j, kDim * (j + 1)) = N3.transpose();
        const Eigen::Vector3d Ntr = N3.transpose() * r;
        d_exact.segment<3>(3 * j) = -Ntr;
        if (copts_.psd_audit_en) {
          static PersistentLogStream exact_log("pose_knots_exact_constraint.txt");
          bool exact_first;
          std::ofstream& exact_ofs = exact_log.stream(&exact_first);
          if (exact_first) exact_ofs << "scan_id,iter,seg,r_norm,Ntr_norm\n";
          exact_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << j << ","
                    << r.norm() << "," << Ntr.norm() << "\n";
          exact_ofs.flush();
        }
      }

      // Phase-2 diagnostic (2026-09-22, user-specified experiment): temporary
      // soft deterministic-nullspace penalty lambda_C*||N^T r||^2 added on
      // top of the existing r^T Q^+ r process cost, where N spans THIS
      // iteration's own 3 lowest Q9 eigenvectors (relinearized above, same
      // Q9 the Lambda pseudoinverse just used -- NOT frozen at init, per the
      // user's explicit instruction to keep this consistent with the
      // per-iteration relinearization). Sign convention verified against the
      // existing process factor immediately above: that factor implements
      // J=[-F9,I], A+=J^T Lambda J, b+=-J^T Lambda r (i.e. FtL=F9^T*Lambda,
      // b.left+=FtL*r=F9^T*Lambda*r, b.right+=-Lambda*r -- matches standard
      // GN normal equations A dx=-J^T W r with W=Lambda). The deterministic
      // term is the same J with W replaced by lambda_C*N*N^T (so that
      // dx^T A_det dx = lambda_C*||N^T J dx||^2 exactly, and at r=0 the
      // gradient contribution reduces to the same -J^T W r form). Derivation:
      // J_det = N^T J = [-N^T F9, N^T]; A_det = lambda_C*J_det^T*J_det;
      // b_det = -lambda_C*J_det^T*(N^T r). Expanding with NNt := N*N^T
      // reduces exactly to the existing Lambda-shaped block pattern with
      // Lambda -> lambda_C*NNt.
      if (copts_.pose_knots_det_constraint_en && !exact_det_en) {
        const Eigen::Matrix<double, 9, 9> Qs = 0.5 * (Q9 + Q9.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es_q(Qs);
        const Eigen::Matrix<double, 9, 3> N = es_q.eigenvectors().leftCols(3);
        const Eigen::Matrix<double, 9, 9> NNt = N * N.transpose();
        const double lam_c = copts_.pose_knots_det_constraint_weight;
        const Eigen::Matrix<double, 9, 9> FtNNt = F9.transpose() * NNt;
        A.block<9, 9>(kDim * j, kDim * j)         += lam_c * FtNNt * F9;
        A.block<9, 9>(kDim * j, kDim * (j + 1))   += -lam_c * FtNNt;
        A.block<9, 9>(kDim * (j + 1), kDim * j)   += -lam_c * NNt * F9;
        A.block<9, 9>(kDim * (j + 1), kDim * (j + 1)) += lam_c * NNt;
        A_det.block<9, 9>(kDim * j, kDim * j)         += lam_c * FtNNt * F9;
        A_det.block<9, 9>(kDim * j, kDim * (j + 1))   += -lam_c * FtNNt;
        A_det.block<9, 9>(kDim * (j + 1), kDim * j)   += -lam_c * NNt * F9;
        A_det.block<9, 9>(kDim * (j + 1), kDim * (j + 1)) += lam_c * NNt;
        b.segment<9>(kDim * j)       += lam_c * FtNNt * r;
        b.segment<9>(kDim * (j + 1)) += -lam_c * NNt * r;
        const Eigen::Vector3d Ntr = N.transpose() * r;
        e_det_total += lam_c * Ntr.squaredNorm();

        if (copts_.psd_audit_en) {
          static PersistentLogStream det_log("pose_knots_det_constraint.txt");
          bool det_first;
          std::ofstream& det_ofs = det_log.stream(&det_first);
          if (det_first)
            det_ofs << "scan_id,iter,seg,lambda_c,r_norm,Ntr_norm\n";
          det_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << j << ","
                   << lam_c << "," << r.norm() << "," << Ntr.norm() << "\n";
          det_ofs.flush();
        }
      }

      // POST-REVIEW ADDITION (item 12, "add the actual nonlinear process
      // residual to the diagnostics... this tells us whether the bizarre
      // stationary trajectory is process-consistent (r~=0, LiDAR-driven
      // motion the optimizer thinks IS compatible with the IMU) or
      // process-inconsistent (r>>0, LiDAR dragging the spline away from
      // inertial dynamics)"). NOTE ON SCOPE: `r` here is the residual in
      // CORRECTION-SPACE against the FROZEN (at-init) F_j -- i.e. exactly
      // what this GN system's own process factor penalizes -- not a
      // freshly re-integrated nonlinear residual against the trial's own
      // moved trajectory (that would need re-walking mg.imu_samples_raw
      // per segment from the trial's own knot j state, duplicating
      // init()'s own boundary-interpolation logic; not done this round).
      // Still directly answers the question: r||_theta/p/v large means
      // the CURRENT correction already disagrees with what the process
      // model predicts, regardless of whether that disagreement is "real"
      // nonlinearity or LiDAR overpowering the (frozen-linearized) prior.
      if (copts_.psd_audit_en) {
        static PersistentLogStream proc_res_log("pose_knots_process_residual.txt");
        bool proc_res_first;
        std::ofstream& proc_res_ofs = proc_res_log.stream(&proc_res_first);
        if (proc_res_first)
          proc_res_ofs << "scan_id,iter,seg,r_theta_norm,r_p_norm,r_v_norm\n";
        proc_res_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << j << ","
                     << r.segment<3>(0).norm() << "," << r.segment<3>(3).norm() << ","
                     << r.segment<3>(6).norm() << "\n";
        proc_res_ofs.flush();
      }
    }
  }

  // ---- item 20's own fallback: discrete first-difference smoothness,
  // off (0.0) by default per item 31's own "first prove... then add
  // smoothness" ----
  if (copts_.pose_knots_smoothness_pos > 0.0 || copts_.pose_knots_smoothness_rot > 0.0) {
    for (int j = 0; j + 1 < N; ++j) {
      if (copts_.pose_knots_smoothness_pos > 0.0) {
        const double lp = copts_.pose_knots_smoothness_pos;
        const V3D rp = coupled_knot_delta_pos_[j + 1] - coupled_knot_delta_pos_[j];
        A.block<3, 3>(kDim * j + 3, kDim * j + 3)         += lp * M3D::Identity();
        A.block<3, 3>(kDim * j + 3, kDim * (j + 1) + 3)   += -lp * M3D::Identity();
        A.block<3, 3>(kDim * (j + 1) + 3, kDim * j + 3)   += -lp * M3D::Identity();
        A.block<3, 3>(kDim * (j + 1) + 3, kDim * (j + 1) + 3) += lp * M3D::Identity();
        A_smooth.block<3, 3>(kDim * j + 3, kDim * j + 3)         += lp * M3D::Identity();
        A_smooth.block<3, 3>(kDim * j + 3, kDim * (j + 1) + 3)   += -lp * M3D::Identity();
        A_smooth.block<3, 3>(kDim * (j + 1) + 3, kDim * j + 3)   += -lp * M3D::Identity();
        A_smooth.block<3, 3>(kDim * (j + 1) + 3, kDim * (j + 1) + 3) += lp * M3D::Identity();
        b.segment<3>(kDim * j + 3)       += lp * rp;
        b.segment<3>(kDim * (j + 1) + 3) += -lp * rp;
      }
      if (copts_.pose_knots_smoothness_rot > 0.0) {
        const double lr = copts_.pose_knots_smoothness_rot;
        const V3D rr = coupled_knot_delta_theta_[j + 1] - coupled_knot_delta_theta_[j];
        A.block<3, 3>(kDim * j, kDim * j)                 += lr * M3D::Identity();
        A.block<3, 3>(kDim * j, kDim * (j + 1))           += -lr * M3D::Identity();
        A.block<3, 3>(kDim * (j + 1), kDim * j)           += -lr * M3D::Identity();
        A.block<3, 3>(kDim * (j + 1), kDim * (j + 1))     += lr * M3D::Identity();
        A_smooth.block<3, 3>(kDim * j, kDim * j)                 += lr * M3D::Identity();
        A_smooth.block<3, 3>(kDim * j, kDim * (j + 1))           += -lr * M3D::Identity();
        A_smooth.block<3, 3>(kDim * (j + 1), kDim * j)           += -lr * M3D::Identity();
        A_smooth.block<3, 3>(kDim * (j + 1), kDim * (j + 1))     += lr * M3D::Identity();
        b.segment<3>(kDim * j)       += lr * rr;
        b.segment<3>(kDim * (j + 1)) += -lr * rr;
      }
    }
  }

  // ---- item 8/9/10: LiDAR factors, analytic position/velocity Jacobian
  // (Hermite weights), FD rotation Jacobian (item 16's own allowance) ----
  constexpr double kFdEps = 1e-6;
  for (const auto& res : residuals_) {
    const double t_eval = use_end_time ? t1 : res.t;
    int j; double u;
    trial.bracket(t_eval, j, u);
    const double dt_seg = coupled_pose_knots_.knot(j + 1).t - coupled_pose_knots_.knot(j).t;
    const double h00 = 2*u*u*u - 3*u*u + 1, h10 = u*u*u - 2*u*u + u;
    const double h01 = -2*u*u*u + 3*u*u,    h11 = u*u*u - u*u;
    const double w = 1.0 / std::max(res.sigma_squared, 1e-18);

    Eigen::Matrix<double, 1, 18> Jrow;
    Jrow.setZero();
    // Position/velocity columns: analytic (position does not depend on
    // rotation at all under this Hermite parameterization -- decoupled).
    Jrow.segment<3>(3)  = h00 * res.normal.transpose();          // d/d(delta_pos_j)
    Jrow.segment<3>(6)  = h10 * dt_seg * res.normal.transpose(); // d/d(delta_vel_j)
    Jrow.segment<3>(12) = h01 * res.normal.transpose();          // d/d(delta_pos_{j+1})
    Jrow.segment<3>(15) = h11 * dt_seg * res.normal.transpose(); // d/d(delta_vel_{j+1})
    // Rotation columns: central FD, perturbing the TRIAL knot's own
    // rotation (right-multiplicative) and re-evaluating the true
    // residual r(c) = n^T(R(t)q + p(t)) + d, d held fixed at THIS
    // residual's own current value (res.r already IS r(c_current) since
    // it was computed against the trial spline above).
    auto residualAt = [&](int knot_idx, int axis, double eps) {
      PoseKnotSpline pert = trial;
      pert.knotMut(knot_idx).rot = trial.knot(knot_idx).rot * Exp(V3D(eps * V3D::Unit(axis)));
      const M3D R_t = pert.rotationAt(t_eval);
      const V3D p_t = pert.positionAt(t_eval);
      return res.normal.dot(R_t * res.raw_body_point + p_t) + (res.r - res.normal.dot(
                 trial.rotationAt(t_eval) * res.raw_body_point + trial.positionAt(t_eval)));
    };
    for (int axis = 0; axis < 3; ++axis) {
      const double rp = residualAt(j, axis, kFdEps);
      const double rm = residualAt(j, axis, -kFdEps);
      Jrow(0 + axis) = (rp - rm) / (2 * kFdEps);
      const double rp1 = residualAt(j + 1, axis, kFdEps);
      const double rm1 = residualAt(j + 1, axis, -kFdEps);
      Jrow(9 + axis) = (rp1 - rm1) / (2 * kFdEps);
    }

    A.block<18, 18>(kDim * j, kDim * j).noalias() += w * (Jrow.transpose() * Jrow);
    A_lidar.block<18, 18>(kDim * j, kDim * j).noalias() += w * (Jrow.transpose() * Jrow);
    b.segment<18>(kDim * j).noalias() -= w * Jrow.transpose() * res.r;
    e_lidar_total += w * res.r * res.r;
  }

  // POST-REVIEW FIX ("Test 1 -- prove whether A is genuinely indefinite"):
  // symmetrize defensively (each off-diagonal block pair above was
  // computed via two DIFFERENT matrix products -- e.g. -F9^T*Lambda vs
  // -Lambda*F9 -- which are exact transposes analytically but not
  // necessarily bit-identical in floating point) and log the symmetrized
  // matrix's own min/max eigenvalue, condition number, and ||A-A^T||_F
  // BEFORE inverting -- so a future negative "covariance" trace can be
  // read against ground truth (lambda_min(A_s)<0 == a real construction
  // problem; lambda_min(A_s)>0 but A^-1 still has negative diagonal ==
  // the inversion itself is numerically unreliable) instead of asserted.
  const double asym_norm = (A - A.transpose()).norm();
  A = 0.5 * (A + A.transpose());
  if (copts_.psd_audit_en) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(A);
    const double lmin = es_a.eigenvalues().minCoeff();
    const double lmax = es_a.eigenvalues().maxCoeff();
    static PersistentLogStream a_diag_log("pose_knots_A_diag.txt");
    bool a_diag_first;
    std::ofstream& a_diag_ofs = a_diag_log.stream(&a_diag_first);
    if (a_diag_first) a_diag_ofs << "scan_id,iter,N,total,lambda_min,lambda_max,cond,asym_norm\n";
    a_diag_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << N << "," << total << ","
               << lmin << "," << lmax << "," << (lmin > 0.0 ? lmax / lmin : -1.0) << "," << asym_norm << "\n";
    a_diag_ofs.flush();

    // POST-REVIEW ADDITION (item 11, factor-scale decomposition): each
    // term's own eigenstructure/trace, isolated. A_prior/A_imu/A_lidar/
    // A_smooth are NOT individually PD in general (e.g. A_imu alone is
    // singular wherever no process factor touches a given knot pair --
    // only their SUM plus the Tikhonov floor is guaranteed PD), so this
    // reports min/max/trace of the symmetrized matrix directly rather
    // than pretending each is its own well-posed system.
    static PersistentLogStream decomp_log("pose_knots_factor_decomp.txt");
    bool decomp_first;
    std::ofstream& decomp_ofs = decomp_log.stream(&decomp_first);
    if (decomp_first)
      decomp_ofs << "scan_id,iter,N,factor,lambda_min,lambda_max,trace\n";
    auto logFactor = [&](const char* name, const Eigen::MatrixXd& Af) {
      const Eigen::MatrixXd Afs = 0.5 * (Af + Af.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_f(Afs);
      decomp_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << N << ","
                 << name << "," << es_f.eigenvalues().minCoeff() << ","
                 << es_f.eigenvalues().maxCoeff() << "," << Afs.trace() << "\n";
    };
    logFactor("prior", A_prior);
    logFactor("imu", A_imu);
    logFactor("lidar", A_lidar);
    logFactor("smooth", A_smooth);
    logFactor("det", A_det);
    decomp_ofs.flush();

    // USER REQUEST 2026-09-21 (diagnostic round): the dominant eigenvector
    // of the "problematic" direction -- lambda_min's own eigenvector
    // (the direction A constrains WEAKEST, exactly where a near-cost-free
    // drift would live) -- and each factor's own quadratic-form
    // projection onto it: v^T A_x v, x in {prior,imu,lidar,smooth}. This
    // answers "which factor actually controls the weakest direction"
    // directly, rather than inferring it from each factor's own global
    // eigenvalues (which don't say whether they agree on WHICH direction
    // is weak).
    {
      const Eigen::VectorXd v = es_a.eigenvectors().col(0);  // Eigen sorts ascending -- col(0) == lambda_min
      static PersistentLogStream veig_log("pose_knots_dominant_eigvec.txt");
      bool veig_first;
      std::ofstream& veig_ofs = veig_log.stream(&veig_first);
      if (veig_first)
        veig_ofs << "scan_id,iter,lambda_min,vT_Aprior_v,vT_Aimu_v,vT_Alidar_v,vT_Asmooth_v,vT_Adet_v,vT_Atotal_v\n";
      veig_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << lmin << ","
               << (v.transpose() * A_prior * v)(0) << "," << (v.transpose() * A_imu * v)(0) << ","
               << (v.transpose() * A_lidar * v)(0) << "," << (v.transpose() * A_smooth * v)(0) << ","
               << (v.transpose() * A_det * v)(0) << ","
               << (v.transpose() * A * v)(0) << "\n";
      veig_ofs.flush();
    }

    // USER REQUEST: Q9's own 9-eigenvalue spectrum, block breakdown
    // (theta-theta/pp/pv/vv), and rank at a range of relative thresholds
    // -- for segment 0 only (the "one representative stationary
    // interval" -- every segment this scan is stationary, segment 0 is
    // as representative as any), scan 1, iteration 0 only (the spectrum
    // is a property of the raw IMU samples + trial rotation, essentially
    // static across the handful of iterations on a stationary scan --
    // logging it once avoids N_iter redundant copies of the same answer).
    if (voxel_map_->frame_idx_ == 1 && coupled_iters_ == 0 && N > 1) {
      // Dump EVERY segment, not just one "representative" one -- the
      // process-residual finding above shows segments are NOT
      // interchangeable (segment 6, the chain midpoint, carries a much
      // larger/growing residual than the others), so a single segment's
      // spectrum could be misleading about whether the threshold matters.
      static PersistentLogStream q9_log("pose_knots_q9_spectrum.txt");
      bool q9_first;
      std::ofstream& q9_ofs = q9_log.stream(&q9_first);
      if (q9_first)
        q9_ofs << "scan_id,seg,eig_idx,eigenvalue,relative_eigenvalue,"
                  "tr_theta_theta,tr_pp,norm_pv,tr_vv,"
                  "rank_1e-3,rank_1e-4,rank_1e-5,rank_1e-6,rank_1e-7,rank_1e-8\n";
      for (int seg = 0; seg + 1 < N; ++seg) {
        const Eigen::Matrix<double, 9, 9>& Q9_rep = coupled_pose_knots_.segQ9(seg);
        const Eigen::Matrix<double, 9, 9> Q9s = 0.5 * (Q9_rep + Q9_rep.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es_q(Q9s);
        const auto& qevals = es_q.eigenvalues();  // ascending
        const double qmax = qevals.maxCoeff();
        int rank_1e3 = 0, rank_1e4 = 0, rank_1e5 = 0, rank_1e6 = 0, rank_1e7 = 0, rank_1e8 = 0;
        for (int k = 0; k < 9; ++k) {
          const double rel = qevals(k) / std::max(qmax, 1e-300);
          if (rel > 1e-3) ++rank_1e3;
          if (rel > 1e-4) ++rank_1e4;
          if (rel > 1e-5) ++rank_1e5;
          if (rel > 1e-6) ++rank_1e6;
          if (rel > 1e-7) ++rank_1e7;
          if (rel > 1e-8) ++rank_1e8;
        }
        for (int k = 8; k >= 0; --k) {  // descending (largest first)
          q9_ofs << voxel_map_->frame_idx_ << "," << seg << "," << (8 - k) << "," << qevals(k) << ","
                 << (qevals(k) / std::max(qmax, 1e-300)) << ","
                 << Q9s.block<3, 3>(0, 0).trace() << "," << Q9s.block<3, 3>(3, 3).trace() << ","
                 << Q9s.block<3, 3>(3, 6).norm() << "," << Q9s.block<3, 3>(6, 6).trace() << ","
                 << rank_1e3 << "," << rank_1e4 << "," << rank_1e5 << "," << rank_1e6 << ","
                 << rank_1e7 << "," << rank_1e8 << "\n";
        }
      }
      q9_ofs.flush();

      // USER REQUEST 2026-09-21 (Phase 1, "nullspace projection of the
      // bad A eigenvector" -- "test first, do not change the estimator"):
      // for segments 0 and 6 (the two segments the process-residual log
      // already showed are special), project v_min's own implied process
      // residual r_j=[-F_j I]v_min onto Q_j's range vs null subspace, and
      // report r^T Q_j^+ r -- directly tests whether the weak direction
      // is a genuine zero-noise process mode (||r_null||>>||r_range||,
      // r^T Q^+ r tiny) rather than inferring it from condition numbers.
      const Eigen::VectorXd v_min = es_a.eigenvectors().col(0);
      for (int seg : {0, 6}) {
        if (seg + 1 >= N) continue;
        const Eigen::Matrix<double, 9, 1> vj = v_min.segment<9>(kDim * seg);
        const Eigen::Matrix<double, 9, 1> vj1 = v_min.segment<9>(kDim * (seg + 1));
        // Same relinearized F9/Q9 this iteration's own solve used (trial
        // == nominal at iter0, so this also matches segF9/segQ9 exactly).
        Eigen::Matrix<double, 9, 9> F9_seg, Q9_seg;
        coupled_pose_knots_.relinearizeSegment(
            seg, trial.knot(seg).rot, trial.knot(seg).pos, trial.knot(seg).vel,
            state_->biasAcc(), state_->biasGyr(), state_->gravity(),
            copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr,
            state_->varAcc(), state_->varGyr(), copts_.repro_second_order, F9_seg, Q9_seg);
        const Eigen::Matrix<double, 9, 1> r = vj1 - F9_seg * vj;

        const Eigen::Matrix<double, 9, 9> Q9s = 0.5 * (Q9_seg + Q9_seg.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es_q2(Q9s);
        const auto& qevals2 = es_q2.eigenvalues();
        const auto& qvecs2 = es_q2.eigenvectors();
        const double qmax2 = std::max(qevals2.maxCoeff(), 0.0);
        const double thresh2 = copts_.pose_knots_q_pinv_rel_thresh * qmax2;
        Eigen::Matrix<double, 9, 1> r_range = Eigen::Matrix<double, 9, 1>::Zero();
        Eigen::Matrix<double, 9, 1> r_null = Eigen::Matrix<double, 9, 1>::Zero();
        Eigen::Matrix<double, 9, 9> Qplus = Eigen::Matrix<double, 9, 9>::Zero();
        for (int k = 0; k < 9; ++k) {
          const Eigen::Matrix<double, 9, 1> uk = qvecs2.col(k);
          const double proj = uk.dot(r);
          if (qevals2(k) > thresh2) {
            r_range += proj * uk;
            Qplus += (1.0 / qevals2(k)) * (uk * uk.transpose());
          } else {
            r_null += proj * uk;
          }
        }
        const double rTQplusR = (r.transpose() * Qplus * r)(0);

        static PersistentLogStream nullproj_log("pose_knots_nullspace_projection.txt");
        bool nullproj_first;
        std::ofstream& nullproj_ofs = nullproj_log.stream(&nullproj_first);
        if (nullproj_first)
          nullproj_ofs << "scan_id,seg,r_norm,r_range_norm,r_null_norm,rT_Qplus_r\n";
        nullproj_ofs << voxel_map_->frame_idx_ << "," << seg << "," << r.norm() << ","
                     << r_range.norm() << "," << r_null.norm() << "," << rTQplusR << "\n";
        nullproj_ofs.flush();
      }

      // USER REQUEST (Phase 1, "finite-difference validation of F9"):
      // independent of the covariance question -- perturb x_j (position/
      // velocity/rotation, one axis at a time) and compare the TRUE
      // re-integrated nonlinear end-state against F9's own linear
      // prediction, for segments 0 and 6.
      constexpr double kFdEpsSeg = 1e-6;
      for (int seg : {0, 6}) {
        if (seg + 1 >= N) continue;
        Eigen::Matrix<double, 9, 9> F9_base, Q9_dummy;
        coupled_pose_knots_.relinearizeSegment(
            seg, trial.knot(seg).rot, trial.knot(seg).pos, trial.knot(seg).vel,
            state_->biasAcc(), state_->biasGyr(), state_->gravity(),
            copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr,
            state_->varAcc(), state_->varGyr(), copts_.repro_second_order, F9_base, Q9_dummy);

        auto integrateFrom = [&](const M3D& r0, const V3D& p0, const V3D& v0,
                                 M3D& r1, V3D& p1, V3D& v1) {
          Eigen::Matrix<double, 9, 9> Fd, Qd;
          coupled_pose_knots_.relinearizeSegment(
              seg, r0, p0, v0, state_->biasAcc(), state_->biasGyr(), state_->gravity(),
              copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr,
              state_->varAcc(), state_->varGyr(), copts_.repro_second_order, Fd, Qd, &r1, &p1, &v1);
        };
        double err_theta_sum = 0.0, err_pos_sum = 0.0, err_vel_sum = 0.0;
        int n_theta = 0, n_pos = 0, n_vel = 0;
        for (int cat = 0; cat < 3; ++cat) {  // 0=theta, 1=pos, 2=vel
          for (int axis = 0; axis < 3; ++axis) {
            M3D rp = trial.knot(seg).rot, rm = trial.knot(seg).rot;
            V3D pp = trial.knot(seg).pos, pm = trial.knot(seg).pos;
            V3D vp = trial.knot(seg).vel, vm = trial.knot(seg).vel;
            const V3D e = V3D::Unit(axis);
            if (cat == 0) { rp = rp * Exp(V3D(kFdEpsSeg * e)); rm = rm * Exp(V3D(-kFdEpsSeg * e)); }
            else if (cat == 1) { pp += kFdEpsSeg * e; pm -= kFdEpsSeg * e; }
            else { vp += kFdEpsSeg * e; vm -= kFdEpsSeg * e; }
            M3D r1p, r1m; V3D p1p, p1m, v1p, v1m;
            integrateFrom(rp, pp, vp, r1p, p1p, v1p);
            integrateFrom(rm, pm, vm, r1m, p1m, v1m);
            Eigen::Matrix<double, 9, 1> fd_col;
            fd_col.segment<3>(0) = V3D(Log(M3D(r1m.transpose() * r1p))) / (2 * kFdEpsSeg);
            fd_col.segment<3>(3) = (p1p - p1m) / (2 * kFdEpsSeg);
            fd_col.segment<3>(6) = (v1p - v1m) / (2 * kFdEpsSeg);
            const int col_idx = cat * 3 + axis;
            const Eigen::Matrix<double, 9, 1> analytic_col = F9_base.col(col_idx);
            const double rel_err = (fd_col - analytic_col).norm() / std::max(analytic_col.norm(), 1e-12);
            if (cat == 0) { err_theta_sum += rel_err; ++n_theta; }
            else if (cat == 1) { err_pos_sum += rel_err; ++n_pos; }
            else { err_vel_sum += rel_err; ++n_vel; }
          }
        }
        static PersistentLogStream fd_log("pose_knots_F9_fd_check.txt");
        bool fd_first;
        std::ofstream& fd_ofs = fd_log.stream(&fd_first);
        if (fd_first) fd_ofs << "scan_id,seg,mean_rel_err_theta,mean_rel_err_pos,mean_rel_err_vel\n";
        fd_ofs << voxel_map_->frame_idx_ << "," << seg << "," << err_theta_sum / n_theta << ","
               << err_pos_sum / n_pos << "," << err_vel_sum / n_vel << "\n";
        fd_ofs.flush();
      }

      // USER REQUEST 2026-09-21 (Phase 2, "independent analytic Q9
      // verification"): build Q9 for segments 0/6 via a SEPARATELY-CODED
      // chain (never calls buildImuStep9x9()'s own Q9 output -- only its
      // F9 output, for the transition matrix, since Phase 3 above already
      // validated F9 to near machine precision and re-deriving it again
      // would test nothing new) using the explicit physical noise-map
      // form: G_a=[0.5*R*dt^2;R*dt] (world frame), Q_pv,step = G_a *
      // Sigma_a * G_a^T with Sigma_a=q_alpha_acc*diag(var_acc);
      // Q_theta,step = dt^2*q_alpha_gyr*diag(var_gyr) (G_g=dt*I). This
      // isolates the check to Q's own per-step construction and the
      // chaining loop, independent of the shipped buildImuStep9x9()'s own
      // Q9 block-assignment code path.
      for (int seg : {0, 6}) {
        if (seg + 1 >= N) continue;
        const auto& samples = coupled_pose_knots_.segSamples(seg);
        M3D rot_imu = trial.knot(seg).rot;
        Eigen::Matrix<double, 9, 9> Q_analytic = Eigen::Matrix<double, 9, 9>::Zero();
        double dt_total = 0.0;
        int n_steps = 0;
        for (size_t k = 0; k + 1 < samples.size(); ++k) {
          const double dt = samples[k + 1].t - samples[k].t;
          if (!(dt > 0.0)) continue;
          dt_total += dt;
          ++n_steps;
          const M3D acc_noise_world = rot_imu * (copts_.repro_q_alpha_acc * state_->varAcc()).asDiagonal() * rot_imu.transpose();
          Eigen::Matrix<double, 9, 9> Qstep = Eigen::Matrix<double, 9, 9>::Zero();
          Qstep.block<3, 3>(0, 0) = (copts_.repro_q_alpha_gyr * state_->varGyr() * (dt * dt)).asDiagonal();
          Qstep.block<3, 3>(3, 3) = 0.25 * dt * dt * dt * dt * acc_noise_world;
          Qstep.block<3, 3>(3, 6) = 0.5 * dt * dt * dt * acc_noise_world;
          Qstep.block<3, 3>(6, 3) = Qstep.block<3, 3>(3, 6);
          Qstep.block<3, 3>(6, 6) = dt * dt * acc_noise_world;
          const V3D acc_avr = 0.5 * (samples[k].acc + samples[k + 1].acc) - state_->biasAcc();
          const V3D angvel_avr = 0.5 * (samples[k].gyro + samples[k + 1].gyro) - state_->biasGyr();
          Eigen::Matrix<double, 9, 9> Fstep, Qstep_shipped_unused;
          buildImuStep9x9(rot_imu, acc_avr, angvel_avr, dt, state_->varAcc(), state_->varGyr(),
                          copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr, copts_.repro_second_order,
                          Fstep, Qstep_shipped_unused);
          Q_analytic = Fstep * Q_analytic * Fstep.transpose() + Qstep;
          rot_imu = rot_imu * Exp(angvel_avr, dt);
        }
        const Eigen::Matrix<double, 9, 9>& Q_code = coupled_pose_knots_.segQ9(seg);
        const double fro_err = (Q_code - Q_analytic).norm();
        const double fro_rel = fro_err / std::max(Q_analytic.norm(), 1e-300);

        // USER REQUEST 2026-09-22 (redesigned Phase 1, "verify N^T G = 0
        // and identify the deterministic nullspace... via the ACCUMULATED
        // process noise map G_j, not just Q_j's own eigendecomposition"):
        // build G_total (9 x 6*n_steps: rows=[theta,p,v], each step
        // contributes 6 columns=[gyro-noise(3),accel-noise(3)]) by
        // chaining G_new=[F_step*G_old, G_step] the SAME way Q itself
        // chains (Q_new=F*Q_old*F^T+Q_step) -- a genuinely SEPARATE
        // numerical object from Q_code/Q_analytic above, not a relabeling
        // of the same eigendecomposition. rank(G_total) and its LEFT null
        // space (via SVD) are then compared against Q9's own
        // eigendecomposition-derived null space as two INDEPENDENTLY
        // computed answers to the same question.
        Eigen::MatrixXd G_total(9, 0);
        {
          M3D rot_g = trial.knot(seg).rot;
          for (size_t k = 0; k + 1 < samples.size(); ++k) {
            const double dt = samples[k + 1].t - samples[k].t;
            if (!(dt > 0.0)) continue;
            Eigen::Matrix<double, 9, 6> Gstep = Eigen::Matrix<double, 9, 6>::Zero();
            Gstep.block<3, 3>(0, 0) = dt * M3D::Identity();          // theta <- gyro noise
            Gstep.block<3, 3>(3, 3) = 0.5 * rot_g * dt * dt;         // p     <- accel noise
            Gstep.block<3, 3>(6, 3) = rot_g * dt;                    // v     <- accel noise
            const V3D acc_avr = 0.5 * (samples[k].acc + samples[k + 1].acc) - state_->biasAcc();
            const V3D angvel_avr = 0.5 * (samples[k].gyro + samples[k + 1].gyro) - state_->biasGyr();
            Eigen::Matrix<double, 9, 9> Fstep, Qstep_unused2;
            buildImuStep9x9(rot_g, acc_avr, angvel_avr, dt, state_->varAcc(), state_->varGyr(),
                            copts_.repro_q_alpha_acc, copts_.repro_q_alpha_gyr, copts_.repro_second_order,
                            Fstep, Qstep_unused2);
            Eigen::MatrixXd G_new(9, G_total.cols() + 6);
            if (G_total.cols() > 0) G_new.leftCols(G_total.cols()) = Fstep * G_total;
            G_new.rightCols(6) = Gstep;
            G_total = G_new;
            rot_g = rot_g * Exp(angvel_avr, dt);
          }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_g(G_total, Eigen::ComputeFullU);
        const auto& sv_g = svd_g.singularValues();
        const double sv_g_max = (sv_g.size() > 0) ? sv_g(0) : 0.0;
        int rank_g = 0;
        // USER FOLLOW-UP FIX: threshold on G's SINGULAR values must be the
        // SQUARE ROOT of the threshold applied to Q's EIGENVALUES
        // elsewhere (Q~G*Sigma*G^T => eigenvalue ratios ~ singular-value
        // ratios SQUARED for the same direction) -- an inconsistent 1e-9
        // (vs the shipped pose_knots_q_pinv_rel_thresh=1e-6 on Q)
        // produced a MISLEADING rank_G=9/null_dim=0 (0-dim null space,
        // making N^T*G and the principal angles vacuously zero) on the
        // first run of this diagnostic. Fixed to sqrt(threshold) for a
        // consistent comparison.
        const double sv_thresh_rel = std::sqrt(copts_.pose_knots_q_pinv_rel_thresh);
        for (int i = 0; i < sv_g.size(); ++i) if (sv_g(i) > sv_thresh_rel * std::max(sv_g_max, 1.0)) ++rank_g;
        // U's LAST (9-rank_g) columns (JacobiSVD orders singular values
        // descending) span G_total's own LEFT null space -- computed
        // purely from G_total's own SVD, independent of Q9's eigenvectors.
        const Eigen::MatrixXd U_null_g = svd_g.matrixU().rightCols(9 - rank_g);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es_code(0.5 * (Q_code + Q_code.transpose()));
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es_ana(0.5 * (Q_analytic + Q_analytic.transpose()));

        static PersistentLogStream qcheck_log("pose_knots_q9_analytic_check.txt");
        bool qcheck_first;
        std::ofstream& qcheck_ofs = qcheck_log.stream(&qcheck_first);
        if (qcheck_first)
          qcheck_ofs << "scan_id,seg,n_steps,dt_total,mean_var_acc,mean_var_gyr,"
                        "q_alpha_acc,q_alpha_gyr,fro_err,fro_rel,eig_idx,"
                        "eig_code,eig_analytic\n";
        for (int k = 8; k >= 0; --k) {
          qcheck_ofs << voxel_map_->frame_idx_ << "," << seg << "," << n_steps << "," << dt_total << ","
                     << state_->varAcc().mean() << "," << state_->varGyr().mean() << ","
                     << copts_.repro_q_alpha_acc << "," << copts_.repro_q_alpha_gyr << ","
                     << fro_err << "," << fro_rel << "," << (8 - k) << ","
                     << es_code.eigenvalues()(k) << "," << es_ana.eigenvalues()(k) << "\n";
        }
        qcheck_ofs.flush();

        // USER REQUEST 2026-09-22 (redesigned Phase 1): N (Q9's own null
        // eigenvectors, from es_code -- 9x9 ascending, columns 0..8-rank
        // are the null ones) vs G_total's own SVD-derived left null space
        // (U_null_g, computed above, entirely independently). Report
        // ||N^T G_total||_F directly (should be ~0 given N spans
        // null(Q)=null(G^T) exactly, PROVIDED Sigma is full rank -- this
        // is the direct numerical confirmation the user asked for rather
        // than an inference from Phase 2's matrix-level agreement), and
        // the principal angles between subspace(N) and subspace(U_null_g)
        // -- both computed FRESH here, not reused from earlier phases.
        {
          const int null_dim = 9 - rank_g;
          const Eigen::MatrixXd N = es_code.eigenvectors().leftCols(std::max(null_dim, 0));
          const double NtG_fro = (N.transpose() * G_total).norm();
          // Principal angles: singular values of N^T * U_null_g are
          // cos(theta_i) for the principal angles between the two
          // subspaces (both orthonormal bases, N from a SelfAdjoint
          // eigensolver, U_null_g from JacobiSVD -- both orthonormal by
          // construction).
          std::vector<double> angles_deg;
          if (N.cols() > 0 && U_null_g.cols() > 0) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd_pa(N.transpose() * U_null_g);
            for (int i = 0; i < svd_pa.singularValues().size(); ++i) {
              const double c = std::min(1.0, std::max(-1.0, svd_pa.singularValues()(i)));
              angles_deg.push_back(std::acos(c) * 180.0 / M_PI);
            }
          }
          static PersistentLogStream grank_log("pose_knots_G_rank_check.txt");
          bool grank_first;
          std::ofstream& grank_ofs = grank_log.stream(&grank_first);
          if (grank_first)
            grank_ofs << "scan_id,seg,n_steps,dt_total,rank_G,rank_Q,null_dim,"
                         "NtG_fro,max_principal_angle_deg,mean_principal_angle_deg,"
                         "sv_g_all\n";
          double max_ang = 0.0, sum_ang = 0.0;
          for (double a : angles_deg) { max_ang = std::max(max_ang, a); sum_ang += a; }
          std::ostringstream sv_list;
          for (int i = 0; i < sv_g.size(); ++i) sv_list << sv_g(i) << (i + 1 < sv_g.size() ? ";" : "");
          grank_ofs << voxel_map_->frame_idx_ << "," << seg << "," << n_steps << "," << dt_total << ","
                    << rank_g << "," << (9 - null_dim) << "," << null_dim << "," << NtG_fro << ","
                    << max_ang << "," << (angles_deg.empty() ? 0.0 : sum_ang / angles_deg.size()) << ","
                    << sv_list.str() << "\n";
          grank_ofs.flush();
        }

        // USER REQUEST 2026-09-21 (Phase 4, "deterministic constraint
        // verification"): does Q9's own (already-validated) null
        // eigenspace correspond to a genuine zero-noise KINEMATIC
        // relation, or is it an arbitrary numerical artifact? Q9 has NO
        // theta<->[p,v] coupling by construction (buildImuStep9x9 never
        // writes Q9.block(iR,iP)/Q9.block(iR,iV)), so the 3 near-zero
        // eigenvalues' own eigenvectors should live ENTIRELY in the 6-dim
        // [p,v] sub-block -- verified explicitly (theta_norm below).
        // For a SINGLE accelerometer-noise step, Q_pv,step = G_a*Sigma_a*
        // G_a^T with G_a=[0.5*R*dt^2;R*dt] (Phase 2's own construction) --
        // a standard linear-algebra fact (Sigma_a positive definite) gives
        // null(Q_pv,step) = null(G_a^T) = {[n_p;n_v] : n_v=-0.5*dt*n_p}
        // EXACTLY, for ANY n_p -- i.e. G_a^T serves as the deterministic-
        // constraint matrix C DIRECTLY (C@G_a=0 holds trivially by this
        // same construction, not something separately worth re-verifying
        // numerically). For a MULTI-STEP chained segment (both 0 and 6
        // are >1 step), the exact relation is no longer a single closed
        // form (each step's null direction gets rotated/mixed by that
        // step's own F), so this checks whether the FULL segment's own
        // (already Phase-2-validated) null eigenvectors still
        // approximately satisfy n_v=-0.5*dt_total*n_p -- reporting the
        // fit residual honestly rather than assuming it.
        for (int k = 0; k < 3; ++k) {  // ascending order -- the 3 smallest eigenvalues
          const Eigen::Matrix<double, 9, 1> ek = es_code.eigenvectors().col(k);
          const double theta_norm = ek.segment<3>(0).norm();
          const V3D n_p = ek.segment<3>(3);
          const V3D n_v = ek.segment<3>(6);
          // Least-squares alpha such that n_v ~= -alpha*n_p (alpha's
          // physical prediction, if the trapezoidal-like relation held,
          // would be alpha=0.5*dt_total).
          const double alpha_fit = (n_p.squaredNorm() > 1e-300) ? -(n_p.dot(n_v)) / n_p.squaredNorm() : 0.0;
          const double resid = (n_v + alpha_fit * n_p).norm();  // residual of the BEST-FIT linear relation
          static PersistentLogStream nullvec_log("pose_knots_q9_nullvec_check.txt");
          bool nullvec_first;
          std::ofstream& nullvec_ofs = nullvec_log.stream(&nullvec_first);
          if (nullvec_first)
            nullvec_ofs << "scan_id,seg,null_idx,eigenvalue,theta_norm,"
                          "n_p_norm,n_v_norm,alpha_fit,half_dt_total,resid_norm\n";
          nullvec_ofs << voxel_map_->frame_idx_ << "," << seg << "," << k << ","
                     << es_code.eigenvalues()(k) << "," << theta_norm << ","
                     << n_p.norm() << "," << n_v.norm() << "," << alpha_fit << ","
                     << (0.5 * dt_total) << "," << resid << "\n";
          nullvec_ofs.flush();
        }
      }
    }
  }

  // ---- item 15's own step-size safeguard, carried over ----
  // Phase-3 (2026-09-22): when exact_det_en, A/b as assembled above are
  // the UNCONSTRAINED (stochastic-only) system -- Lambda already excludes
  // the 3 null directions per segment entirely, so A alone is exactly as
  // rank-deficient in those directions as Phase 1/2 found (that's by
  // design: the missing information is supplied by the equality
  // constraint C_exact*delta=d_exact via Lagrange multipliers/nullspace
  // elimination below, NOT by regularizing A itself). Solving A*delta=b
  // directly here (the old path) would let LiDAR+prior alone determine
  // the constrained direction -- exactly the pathological mechanism Phase
  // 1/2 diagnosed. So the exact-constraint delta MUST go through the
  // reduced system, never through ldlt_a.solve(A,b) directly.
  //
  // Nullspace elimination: delta = delta_p + Z*eta, where delta_p is the
  // minimum-norm particular solution of C_exact*delta_p=d_exact (via
  // C_exact's own SVD) and Z=null(C_exact) (same SVD's V-columns below
  // the rank). Standard equality-constrained normal equations reduce to
  // (Z^T A Z) eta = Z^T (b - A*delta_p) -- see the commit message /
  // config doc comment for the full derivation.
  Eigen::MatrixXd Z_ns;             // null(C_exact) basis, total x (total-rank_c) -- only when exact_det_en
  Eigen::MatrixXd A_reduced;        // Z^T A Z -- only when exact_det_en
  Eigen::VectorXd delta;
  if (exact_det_en && n_exact_constraints > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_c(C_exact, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::VectorXd& sv_c = svd_c.singularValues();
    const double sv_c_max = sv_c.size() ? sv_c(0) : 0.0;
    const double sv_c_thresh = 1e-6 * std::max(sv_c_max, 1.0);
    int rank_c = 0;
    for (int i = 0; i < sv_c.size(); ++i)
      if (sv_c(i) > sv_c_thresh) ++rank_c;
    const Eigen::MatrixXd& Vm = svd_c.matrixV();
    const Eigen::MatrixXd& Um = svd_c.matrixU();
    Z_ns = Vm.rightCols(total - rank_c);
    Eigen::VectorXd delta_p = Eigen::VectorXd::Zero(total);
    for (int i = 0; i < rank_c; ++i)
      delta_p += (Um.col(i).dot(d_exact) / sv_c(i)) * Vm.col(i);
    A_reduced = Z_ns.transpose() * A * Z_ns;
    const Eigen::VectorXd b_reduced = Z_ns.transpose() * (b - A * delta_p);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_reduced(A_reduced);
    const Eigen::VectorXd eta = ldlt_reduced.solve(b_reduced);
    delta = delta_p + Z_ns * eta;
    if (ldlt_reduced.info() != Eigen::Success || !delta.allFinite() ||
        delta.size() != total || rank_c != n_exact_constraints) {
      std::ostringstream diag;
      diag << "[coupled/pose_knots] FATAL: exact-constraint nullspace-elimination "
              "solve produced a non-finite/failed/rank-deficient result -- refusing "
              "to apply it to state_. ldlt_reduced.info()=" << static_cast<int>(ldlt_reduced.info())
           << " (0=Success) allFinite=" << delta.allFinite() << " rank_c=" << rank_c
           << " expected=" << n_exact_constraints << " N=" << N << " total=" << total
           << " iter=" << coupled_iters_ << " scan_id=" << voxel_map_->frame_idx_;
      throw std::runtime_error(diag.str());
    }
    if (copts_.psd_audit_en) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_red(A_reduced);
      const double lmin_r = es_red.eigenvalues().minCoeff();
      const double lmax_r = es_red.eigenvalues().maxCoeff();
      static PersistentLogStream exact_solve_log("pose_knots_exact_solve_diag.txt");
      bool exact_solve_first;
      std::ofstream& exact_solve_ofs = exact_solve_log.stream(&exact_solve_first);
      if (exact_solve_first)
        exact_solve_ofs << "scan_id,iter,rank_c,expected_rank_c,reduced_dim,lambda_min_reduced,"
                            "lambda_max_reduced,cond_reduced\n";
      exact_solve_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << rank_c << ","
                       << n_exact_constraints << "," << (total - rank_c) << "," << lmin_r << ","
                       << lmax_r << "," << (lmin_r > 0.0 ? lmax_r / lmin_r : -1.0) << "\n";
      exact_solve_ofs.flush();
    }
  } else {
    Eigen::LDLT<Eigen::MatrixXd> ldlt_a(A);
    delta = ldlt_a.solve(b);
    if (ldlt_a.info() != Eigen::Success || !delta.allFinite() || delta.size() != total) {
      std::ostringstream diag;
      diag << "[coupled/pose_knots] FATAL: joint GN solve produced a "
              "non-finite/failed result -- refusing to apply it to state_. "
              "ldlt.info()=" << static_cast<int>(ldlt_a.info())
           << " (0=Success) allFinite=" << delta.allFinite()
           << " N=" << N << " total=" << total
           << " iter=" << coupled_iters_ << " scan_id=" << voxel_map_->frame_idx_;
      throw std::runtime_error(diag.str());
    }
  }

  // POST-REVIEW ADDITION ("compare batch pose-knot against causal
  // pose-knot filter/smoother"): A is exactly block-tridiagonal (every
  // factor -- head prior, process, LiDAR -- only ever couples adjacent
  // knots), so solveBlockTridiagonal9() computes the EXACT SAME estimator
  // via a genuinely different algorithm (forward information-filter +
  // RTS backward smoother, see its own doc comment for the equivalence
  // argument). Run every iteration (cheap: N<=~20 independent 9x9 LDLTs,
  // not the 9Nx9N dense one), gated on psd_audit_en for the logging cost
  // only -- the cross-check itself always runs so a silent divergence
  // can't hide behind a config flag nobody happened to set. The DENSE
  // result (`delta`, below) remains what's actually applied to state_ --
  // this does not change live behavior, only verifies it.
  //
  // NOT applicable when exact_det_en: the equality-constrained
  // nullspace-elimination solve above is not the plain block-tridiagonal
  // information filter/smoother recursion this check verifies against --
  // comparing them would report a spurious "large diff" that reflects the
  // algorithm change, not a bug. Skipped entirely in that mode.
  if (!exact_det_en) {
    std::vector<Eigen::Matrix<double, 9, 9>> Ajj(N), Aoff(N - 1);
    std::vector<Eigen::Matrix<double, 9, 1>> bj(N);
    for (int j = 0; j < N; ++j) {
      Ajj[j] = A.block<9, 9>(kDim * j, kDim * j);
      bj[j] = b.segment<9>(kDim * j);
      if (j + 1 < N) Aoff[j] = A.block<9, 9>(kDim * j, kDim * (j + 1));
    }
    const Eigen::VectorXd delta_bt = solveBlockTridiagonal9(Ajj, Aoff, bj);
    const bool bt_ok = (delta_bt.size() == total) && delta_bt.allFinite();
    const double bt_diff = bt_ok ? (delta_bt - delta).norm() : -1.0;
    if (copts_.psd_audit_en) {
      static PersistentLogStream bt_log("pose_knots_blocktri_check.txt");
      bool bt_first;
      std::ofstream& bt_ofs = bt_log.stream(&bt_first);
      if (bt_first) bt_ofs << "scan_id,iter,N,bt_ok,dense_norm,bt_diff_norm,bt_diff_rel\n";
      const double dense_norm = delta.norm();
      bt_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << N << ","
             << (bt_ok ? 1 : 0) << "," << dense_norm << "," << bt_diff << ","
             << (dense_norm > 1e-300 ? bt_diff / dense_norm : -1.0) << "\n";
      bt_ofs.flush();
    }
  }

  {
    double max_step_pos = 0.0, max_step_rot = 0.0;
    for (int j = 0; j < N; ++j) {
      max_step_pos = std::max(max_step_pos, delta.segment<3>(kDim * j + 3).norm());
      max_step_rot = std::max(max_step_rot, delta.segment<3>(kDim * j + 0).norm());
    }
    // USER REQUEST: GN telemetry -- ||delta_c_p||_inf, ||delta_c_phi||_inf
    // (max_step_pos/rot above ARE exactly these, taken BEFORE the trust-
    // region clamp below) and max spline acceleration over [t0,t1] this
    // iteration's own trial.
    if (copts_.psd_audit_en) {
      double max_acc = 0.0, max_angvel = 0.0;
      constexpr int kAccGridN = 40;
      const double t0g = coupled_pose_knots_.knot(0).t;
      for (int k = 0; k <= kAccGridN; ++k) {
        const double tg = t0g + (t1 - t0g) * (static_cast<double>(k) / kAccGridN);
        max_acc = std::max(max_acc, trial.accelerationAt(tg).norm());
        // Phase-6 (2026-09-22, campaign CSV's max_spline_angular_velocity
        // column): same grid, reusing PoseKnotSpline::angularVelocityAt().
        max_angvel = std::max(max_angvel, trial.angularVelocityAt(tg).norm());
      }
      static PersistentLogStream gn_tel_log("pose_knots_gn_telemetry.txt");
      bool gn_tel_first;
      std::ofstream& gn_tel_ofs = gn_tel_log.stream(&gn_tel_first);
      if (gn_tel_first)
        gn_tel_ofs << "scan_id,iter,max_delta_cp_inf,max_delta_cphi_inf,max_accel,max_angvel,"
                      "e_lidar,e_process,e_det,e_total\n";
      gn_tel_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << max_step_pos << ","
                 << max_step_rot << "," << max_acc << "," << max_angvel << "," << e_lidar_total << ","
                 << e_process_total << "," << e_det_total << ","
                 << (e_lidar_total + e_process_total + e_det_total) << "\n";
      gn_tel_ofs.flush();
    }
    double scale = 1.0;
    if (copts_.pose_gn_max_step_pos_m > 0.0 && max_step_pos > copts_.pose_gn_max_step_pos_m)
      scale = std::min(scale, copts_.pose_gn_max_step_pos_m / max_step_pos);
    if (copts_.pose_gn_max_step_rot_rad > 0.0 && max_step_rot > copts_.pose_gn_max_step_rot_rad)
      scale = std::min(scale, copts_.pose_gn_max_step_rot_rad / max_step_rot);
    if (scale < 1.0) delta *= scale;
  }

  for (int j = 0; j < N; ++j) {
    coupled_knot_delta_theta_[j] += delta.segment<3>(kDim * j + 0);
    coupled_knot_delta_pos_[j]   += delta.segment<3>(kDim * j + 3);
    coupled_knot_delta_vel_[j]   += delta.segment<3>(kDim * j + 6);
  }

  // Items 12/13: read Cov(x_j,x_j) straight off A^-1's own diagonal
  // blocks -- x_j IS the physical knot state here, no chaining needed.
  //
  // Phase-3: when exact_det_en, plain A^-1 is wrong (A is deliberately
  // rank-deficient along the 3 constrained directions per segment, same
  // as noted at the solve site above) -- the correct posterior covariance
  // restricted to the constraint manifold is Z*(Z^T A Z)^-1*Z^T (zero
  // variance along the constrained directions themselves, which is
  // correct: a deterministic zero-noise constraint has no uncertainty).
  {
    Eigen::MatrixXd A_inv;
    if (exact_det_en && n_exact_constraints > 0) {
      Eigen::LDLT<Eigen::MatrixXd> ldlt_reduced_cov(A_reduced);
      const Eigen::MatrixXd A_reduced_inv =
          ldlt_reduced_cov.solve(Eigen::MatrixXd::Identity(A_reduced.rows(), A_reduced.rows()));
      A_inv = Z_ns * A_reduced_inv * Z_ns.transpose();
    } else {
      Eigen::LDLT<Eigen::MatrixXd> ldlt_a_cov(A);
      A_inv = ldlt_a_cov.solve(Eigen::MatrixXd::Identity(total, total));
    }
    for (int j = 0; j < N; ++j) coupled_knot_cov_[j] = A_inv.block<9, 9>(kDim * j, kDim * j);
  }

  // Item 27: trajectory covariance logging, gated the same way every
  // other per-iteration pose-arm diagnostic in this file is.
  if (copts_.psd_audit_en) {
    // POST-REVIEW FIX (item 4, "the pose-knot covariance being logged is
    // still a smoothed posterior, not P_prior... log both explicitly"):
    // knot(j).P_prior is the IMU-propagated prior (P_j^-, fixed at
    // init()); coupled_knot_cov_[j] (A^-1's own diagonal block) is the
    // FULL-scan smoothed posterior (P_j^+ after every factor in the
    // batch, including future LiDAR) -- materially different quantities,
    // now both named and logged so neither can be mistaken for the other.
    static PersistentLogStream cov_log("pose_knots_cov.txt");
    bool cov_first;
    std::ofstream& cov_ofs = cov_log.stream(&cov_first);
    if (cov_first)
      cov_ofs << "scan_id,iter,knot,t,"
                 "tr_Pprior_pos,tr_Pprior_theta,tr_Pprior_vel,"
                 "tr_Ppost_pos,tr_Ppost_theta,tr_Ppost_vel\n";
    for (int j = 0; j < N; ++j) {
      const auto& Pprior = coupled_pose_knots_.knot(j).P_prior;
      cov_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << "," << j << ","
              << coupled_pose_knots_.knot(j).t << ","
              << Pprior.block<3, 3>(3, 3).trace() << ","
              << Pprior.block<3, 3>(0, 0).trace() << ","
              << Pprior.block<3, 3>(6, 6).trace() << ","
              << coupled_knot_cov_[j].block<3, 3>(3, 3).trace() << ","
              << coupled_knot_cov_[j].block<3, 3>(0, 0).trace() << ","
              << coupled_knot_cov_[j].block<3, 3>(6, 6).trace() << "\n";
    }
    cov_ofs.flush();

    // Phase-5 (2026-09-22, prior_at_scan_start A/B): head-knot (j=0)
    // prior eigenvalues (pos/theta blocks of P_prior, which is exactly
    // the P0 fed into Omega0 at init() -- toggled by
    // copts_.prior_at_scan_start between state_->cov() [P(t1), post-
    // propagation] and the pre-propagation snapshot [P(t0)]) plus the
    // actual converged correction magnitude ||delta_p0||/||delta_theta0||
    // this iteration, so the two arms' effect on the actual optimized
    // head state can be compared directly, not just their input priors.
    if (N > 0) {
      const auto& P0 = coupled_pose_knots_.knot(0).P_prior;
      Eigen::SelfAdjointEigenSolver<M3D> es_p0(P0.block<3, 3>(3, 3));
      Eigen::SelfAdjointEigenSolver<M3D> es_th0(P0.block<3, 3>(0, 0));
      static PersistentLogStream head_prior_log("pose_knots_head_prior_diag.txt");
      bool head_prior_first;
      std::ofstream& head_prior_ofs = head_prior_log.stream(&head_prior_first);
      if (head_prior_first)
        head_prior_ofs << "scan_id,iter,tr_Pp0,eig_Pp0_min,eig_Pp0_max,"
                           "tr_Pth0,eig_Pth0_min,eig_Pth0_max,"
                           "delta_p0_norm,delta_theta0_norm\n";
      head_prior_ofs << voxel_map_->frame_idx_ << "," << coupled_iters_ << ","
                     << P0.block<3, 3>(3, 3).trace() << "," << es_p0.eigenvalues().minCoeff() << ","
                     << es_p0.eigenvalues().maxCoeff() << "," << P0.block<3, 3>(0, 0).trace() << ","
                     << es_th0.eigenvalues().minCoeff() << "," << es_th0.eigenvalues().maxCoeff() << ","
                     << coupled_knot_delta_pos_[0].norm() << "," << coupled_knot_delta_theta_[0].norm()
                     << "\n";
      head_prior_ofs.flush();
    }
  }

  PoseKnotSpline trial_new = coupled_pose_knots_;
  for (int j = 0; j < N; ++j) {
    trial_new.knotMut(j).pos = coupled_pose_knots_.knot(j).pos + coupled_knot_delta_pos_[j];
    trial_new.knotMut(j).rot = coupled_pose_knots_.knot(j).rot * Exp(coupled_knot_delta_theta_[j]);
    trial_new.knotMut(j).vel = coupled_pose_knots_.knot(j).vel + coupled_knot_delta_vel_[j];
  }
  const M3D new_tail_R = trial_new.rotationAt(t1);
  const V3D new_tail_p = trial_new.positionAt(t1);
  const V3D new_tail_v = trial_new.velocityAt(t1);

  dtheta_out = Log(prev_tail_R.transpose() * new_tail_R);
  dt_out = new_tail_p - prev_tail_p;
  state_->setPropagatedState(new_tail_R, new_tail_p, new_tail_v);

  double sum_abs_r = 0.0;
  for (const auto& res : residuals_) sum_abs_r += std::abs(res.r);
  return residuals_.empty() ? 0.0 : sum_abs_r / static_cast<double>(residuals_.size());
}

}  // namespace livo_recon
