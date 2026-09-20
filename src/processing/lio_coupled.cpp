#include "livo_recon/processing/lio_coupled.h"
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
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/curvature_weight_acc", copts_.curvature_weight_acc, 0.0);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/curvature_weight_gyr", copts_.curvature_weight_gyr, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/bias_observable_only", copts_.bias_observable_only, false);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/curvature_only", copts_.curvature_only, false);
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/dc_weight", copts_.dc_weight, 0.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/prior_per_axis_sigma", copts_.prior_per_axis_sigma, false);
  cfg.nestedMode(true, "estimator/mode=coupled", "estimator/coupled/robust_loss",
                 copts_.robust_loss, "none",
                 {"none", "huber", "cauchy"});
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
  coupled_tier1_nees_ = Tier1NeesBuffer(opts_.nees_per_dof_en ? opts_.nees_tier1_window_scans : 0);

  // Every spline/* and adaptive_q/* key is unclaimed by this class by
  // construction -- refuseUnclaimed only needs to additionally cover
  // estimator/coupled/* itself (already claimed above) plus the shared
  // namespaces loadSharedParameters() already reads. Leaving spline/
  // adaptive_q OUT of the allowed-unclaimed list means any key set there
  // under a coupled config is refused at startup, naming itself -- the
  // structural enforcement item 3d(iii) describes.
  return finalizeConfig(cfg, {});
}

std::string LioProcCoupled::engagementReport() const
{
  std::ostringstream oss;
  oss << "[engagement] estimator=coupled n_c=" << copts_.n_c
      << " zero_mean=" << (copts_.zero_mean ? "true" : "false")
      << " disable_cgyr=" << (copts_.disable_cgyr ? "true" : "false")
      << " jacobian_time_mode=" << copts_.jacobian_time_mode
      << " log_jrow_leverage_en=" << (copts_.log_jrow_leverage_en ? "true" : "false")
      << " freeze_bg=" << (copts_.freeze_bg ? "true" : "false")
      << " adaptive_sigma=" << (copts_.adaptive_sigma ? "true" : "false")
      << " bias_freeze_on_vibration=" << (copts_.bias_freeze_on_vibration ? "true" : "false")
      << " bias_freeze_vibration_factor=" << copts_.bias_freeze_vibration_factor
      << " bias_anchor=" << (copts_.bias_anchor ? "true" : "false")
      << " curvature_weight_acc=" << copts_.curvature_weight_acc
      << " curvature_weight_gyr=" << copts_.curvature_weight_gyr
      << " bias_observable_only=" << (copts_.bias_observable_only ? "true" : "false")
      << " curvature_only=" << (copts_.curvature_only ? "true" : "false")
      << " dc_weight=" << copts_.dc_weight
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

  prior_cov_ = state_->cov();
  applyPriorScalarControls(prior_cov_, opts_.prior_scalar);
  state_propagat_ = *state_;
  trP_pos_pre_ = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();
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
  coupled_delta_v_.setZero(); coupled_delta_bg_.setZero();
  coupled_delta_ba_.setZero(); coupled_delta_g_.setZero();
  coupled_delta_phi0_.setZero(); coupled_delta_pos0_.setZero();
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
                << " curvature_weight_acc=" << copts_.curvature_weight_acc
                << " curvature_weight_gyr=" << copts_.curvature_weight_gyr
                << " curvature_only=" << (copts_.curvature_only ? "true" : "false");
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

      // CQ-57 item 3b: add out-of-band IMU noise power (see this option's
      // own header doc comment for the measured fractions) into the V and
      // P blocks -- integrated over the scan's own duration, using the
      // SAME sigma_a/sigma_g this scan already calibrated (coupled_sigma_a_used_/
      // coupled_sigma_g_used_), not a separate model.
      if (copts_.q_out_of_band_en) {
        const double dt_scan = mg.image.t - mg.poses.front().t;
        if (dt_scan > 0.0) {
          const double var_acc_oob = copts_.q_out_of_band_scale * copts_.q_out_of_band_fraction_acc
              * coupled_sigma_a_used_ * coupled_sigma_a_used_ * dt_scan;
          const int iV = StateGroup::idxV(), iP = StateGroup::idxP();
          posterior18.block<3, 3>(iV, iV).diagonal().array() += var_acc_oob;
          posterior18.block<3, 3>(iP, iP).diagonal().array() += var_acc_oob * dt_scan * dt_scan;
          const double var_gyr_oob = copts_.q_out_of_band_scale * copts_.q_out_of_band_fraction_gyr
              * coupled_sigma_g_used_ * coupled_sigma_g_used_ * dt_scan;
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
      P6.block<3, 3>(3, 0) = P_final.block<3, 3>(iP, iR).transpose();
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

double LioProcCoupled::estimateCoupledCorrection(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out)
{
  dtheta_out = V3D::Zero();
  dt_out = V3D::Zero();
  if (mg.poses.empty()) return 0.0;

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
  // c-block, and Pi_ss -- now the FULL 18x18 P(t0)^-1, at
  // [idxR,idxP,idxV,idxBG,idxBA,idxG]. Bug 2's fix solves for ALL of
  // delta_x(t0) jointly (nothing held fixed/conditioned any more), so this
  // is no longer a Schur-complement sub-block trick -- these six blocks ARE
  // the entire state, so Pi_ss is simply P(t0)^-1 itself (row/column
  // order matched to this solve's own [phi0,p0,v,bg,ba,g] convention). No
  // cross term with the c-block: the prior itself does not correlate the
  // two blocks (any correlation enters only through the shared LiDAR
  // evidence, in the loop below). ----
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
  if ((copts_.curvature_weight_acc > 0.0 || copts_.curvature_weight_gyr > 0.0) && n_c >= 3) {
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_c - 2, n_c);
    for (int k = 0; k < n_c - 2; ++k) { D(k, k) = 1.0; D(k, k + 1) = -2.0; D(k, k + 2) = 1.0; }
    Curv = D.transpose() * D;
  }
  // CQ-55 item 8, arm (c): curvature_only REPLACES the gram/sigma^2 value
  // term with curvature_weight's shape penalty plus an explicit DC-only
  // prior (dc_weight/sigma^2, applied UNIFORMLY to every (i,j) pair -- a
  // rank-1 all-ones contribution whose quadratic form for c is exactly
  // dc_weight/sigma^2 * n_c * ||mean(c)||^2, pricing the mean/constant
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
      // CQ-59 item 1: curv_acc/curv_gyr each normalized by the SAME
      // sigma^2 the value term for that block uses, so the weight means
      // "this shape penalty is worth w times the value penalty" on that
      // block specifically -- replaces the single raw curv_ij that used
      // to apply identically to both blocks despite their ~4657x base-
      // prior stiffness difference.
      const double curv_acc = copts_.curvature_weight_acc * Curv(i, j) / (sigma_a * sigma_a);
      const double curv_gyr = copts_.curvature_weight_gyr * Curv(i, j) / (sigma_g * sigma_g);
      const double value_gram = copts_.curvature_only ? 0.0 : gram(i, j);
      const double dc_acc = copts_.curvature_only ? copts_.dc_weight / (sigma_a * sigma_a) : 0.0;
      const double dc_gyr = copts_.curvature_only ? copts_.dc_weight / (sigma_g * sigma_g) : 0.0;
      Lambda.block<3, 3>(3 * i, 3 * j) =
          M3D(value_gram * prec_acc_diag.asDiagonal()) + (curv_acc + dc_acc) * M3D::Identity();
      Lambda.block<3, 3>(3 * n_c + 3 * i, 3 * n_c + 3 * j) =
          M3D(value_gram * prec_gyr_diag.asDiagonal()) + (curv_gyr + dc_gyr) * M3D::Identity();
    }

  Eigen::MatrixXd Pi_ss = Eigen::MatrixXd::Zero(ncol_s, ncol_s);
  bool have_pi_ss = state_->idxBG() >= 0 && state_->idxBA() >= 0 && state_->idxG() >= 0;
  if (have_pi_ss) {
    const Eigen::MatrixXd Omega = state_->cov().inverse();  // information form of P(t0)
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

  Eigen::VectorXd c_vec(ncol_c);
  for (int j = 0; j < n_c; ++j) {
    c_vec.segment<3>(3 * j) = coupled_c_acc_[j];
    c_vec.segment<3>(3 * n_c + 3 * j) = coupled_c_gyr_[j];
  }
  // Item 3e(v)/3f bug 2: order [delta_phi0, delta_p0, delta_v, delta_bg,
  // delta_ba, delta_g] -- matches phi_x_head's own column order exactly.
  Eigen::VectorXd s_vec(ncol_s);
  s_vec.segment<3>(0)  = coupled_delta_phi0_;
  s_vec.segment<3>(3)  = coupled_delta_pos0_;
  s_vec.segment<3>(6)  = coupled_delta_v_;
  s_vec.segment<3>(9)  = coupled_delta_bg_;
  s_vec.segment<3>(12) = coupled_delta_ba_;
  s_vec.segment<3>(15) = coupled_delta_g_;

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
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncol, ncol);
  A.block(0, 0, ncol_s, ncol_s) = Pi_ss;
  A.block(ncol_s, ncol_s, ncol_c, ncol_c) = Lambda;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(ncol);
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
  double sum_abs_r = 0.0;
  double sum_sq_r = 0.0;  // CQ-53 item 4
  double sum_wr2 = 0.0;   // CQ-54 item 4
  // CQ-55 item 12: S = floor_term + sigma_diag_squared + plane_var_term +
  // s_prior_pose per residual, same definition lio_decoupled.cpp's own
  // sum_S uses -- summed here so the coupled path reports the SAME
  // absolute-units denominator, never logged on this path before now.
  double sum_floor_S = 0.0, sum_sdiag_S = 0.0, sum_pvar_S = 0.0, sum_prior_pose_S = 0.0;
  double sum_sigma_squared = 0.0;  // CQ-60 item 0a
  std::vector<double> hcol_reldiff;  // CQ-53 item 2
  hcol_reldiff.reserve(residuals_.size());
  // TQ-40 item 3: pure-LiDAR-info accumulation restricted to [delta_phi0,
  // delta_p0] (columns 0-5), BEFORE the Pi_ss/Lambda prior is added --
  // see coupled_ask_'s own doc comment in the header for the approximation
  // this makes (no marginalisation over v/bg/ba/g/c).
  Eigen::Matrix<double, 6, 6> HtH_pose_lidar = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> Htz_pose_lidar = Eigen::Matrix<double, 6, 1>::Zero();
  double sum_weight_this_iter = 0.0;
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
  if (copts_.log_cp_constraint_en) coupled_last_Lambda_ = Lambda;
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
  }
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

}  // namespace livo_recon
