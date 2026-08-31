#include "livo_recon/processing/lio_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/lio_accumulator.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>

namespace livo_recon
{

namespace
{

// Debug trace of LIO's per-measure-group state correction and covariance,
// for diagnosing where trajectory divergence originates (see imu_processing
// .cpp's debugLogImu() for the IMU-side counterpart). Truncated at the
// start of each process (first call), appended thereafter. Absolute (bag/
// wall-clock) timestamps throughout, matching evo_processing.cpp's
// /tmp/evo.txt, imu_processing.cpp's /tmp/imu.txt, and FAST-LIVO2's own
// logs, so all of these can be compared directly against each other.
// Remove once done debugging.
void debugLogLio(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("lio.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// Temporary: per-iteration avg|r| trace within a single frame's IEKF loop,
// for the 2026-08-03 early-stopping/convergence-shape question. Remove
// once done debugging.
void debugLogIterError(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("iter_error.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// 2026-08-24: shadow dry-run diagnostic trace -- see LioProcOptions::
// dry_run_point_filter_num's doc comment and LioProc::runDryRunShadowPass().
// Same truncate-on-first-call/append convention as debugLogLio() above, own
// file so it never interleaves with the real per-frame [lio] lines.
void debugLogLioDryRun(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("lio_dryrun.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// T0-D (2026-08-31): scan.csv for scripts/analysis/consistency.py -- see
// LioProcOptions::log_consistency_scan_en's doc comment for the column
// list. Called once per frame from processLIO(), single-threaded (no OMP
// context here, unlike voxelplane.cpp's per-correspondence logs), so no
// mutex needed -- same truncate-on-first-call/append convention as every
// other debug log in this file.
void debugLogConsistencyScan(int scan_id, double t, double dt, double trP_pos,
                              double trP_vel, double trP_att, double omega_norm, double acc_norm)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("scan.csv"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call)
    ofs << "scan_id,t,dt,trP_pos,trP_vel,trP_att,omega_norm,acc_norm\n";
  first_call = false;
  ofs << scan_id << "," << t << "," << dt << "," << trP_pos << "," << trP_vel << ","
      << trP_att << "," << omega_norm << "," << acc_norm << "\n";
}

// T0-E (2026-08-31): nll.txt -- see LioProcOptions::log_nll_en's doc
// comment for the column/formula. Called once per frame (single-
// threaded, no OMP context here), same truncate-on-first-call/append
// convention as every other debug log in this file.
void debugLogNll(double t_abs, double nll, int n_residuals)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("nll.txt"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call)
    ofs << "t,nll,n_residuals\n";
  first_call = false;
  ofs << t_abs << "," << nll << "," << n_residuals << "\n";
}

// 2026-08-24: REMOVED debugLogSplineIter()/debugLogSplineFit()/
// debugLogSplineFitPoints()/debugLogSplineKinFit() -- temporary debug
// logging for the removed iterative-deskew Hermite-spline mechanism. See
// docs/removed_livo_recon_spline_deskew_2026aug24.md.

}  // namespace

LioProc::LioProc(NodeContext& ctx)
  : state_(ctx.state), voxel_map_(ctx.voxel_map), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

std::string LioProc::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<int>(pnh, "lio/ekf/max_iterations",     opts_.max_iterations,  5);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dtheta", opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dt",     opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "lio/ekf/min_diff_error",  opts_.min_diff_error,  -1.0);
  paramWarn<bool>(pnh, "lio/log_debug_en",          opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "lio/log_consistency_scan_en", opts_.log_consistency_scan_en, false);
  paramWarn<bool>(pnh, "lio/log_nll_en", opts_.log_nll_en, false);
  paramWarn<int>(pnh, "lio/dry_run_point_filter_num", opts_.dry_run_point_filter_num, 0);
  paramWarn<double>(pnh, "lio/ekf/density_sigma_ref", opts_.density_sigma_ref, 0.0);
  paramWarn<std::string>(pnh, "lio/ekf/density_sigma_mode", opts_.density_sigma_mode, "linear");
  paramWarn<bool>(pnh, "cuda/enable",               cuda_enable_,          false);

  // Same rosparam keys the (removed) ImuProc deskew/downsample code used
  // to read -- see LioProcOptions::deskew/ds_leaf_size/ds_mode's doc
  // comments (deskewing/downsampling moved here from ImuProc 2026-08-18).
  double range_err;
  paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
  opts_.deskew.sigma_r2 = range_err * range_err;
  double angle_err_deg;
  paramWarn<double>(pnh, "imu/sensor/angle_err_deg", angle_err_deg, 0.2);
  const double sin_angle_err = std::sin(std::max(1e-6, angle_err_deg * M_PI / 180.0));
  opts_.deskew.sigma_a2 = sin_angle_err * sin_angle_err;
  paramWarn<std::string>(pnh, "imu/undistort/time_based_process_noise",
                          opts_.deskew.time_based_process_noise, "var_acc");
  paramWarn<double>(pnh, "imu/ds/ds_leaf_size", opts_.ds_leaf_size, 0.15);
  paramWarn<std::string>(pnh, "imu/ds/mode", opts_.ds_mode, "first");

  std::ostringstream oss;
  oss << "[params/lio]"
      << "\n  log_consistency_scan_en:   " << (opts_.log_consistency_scan_en ? "true" : "false")
      << "\n  log_nll_en:                " << (opts_.log_nll_en ? "true" : "false")
      << "\n  ekf/max_iterations:        " << opts_.max_iterations
      << "\n  ekf/min_norm_dtheta:       " << opts_.min_norm_dtheta
      << "\n  ekf/min_norm_dt:           " << opts_.min_norm_dt
      << "\n  ekf/min_diff_error:        " << opts_.min_diff_error
      << "\n  ekf/density_sigma_ref:     " << opts_.density_sigma_ref
      << "\n  ekf/density_sigma_mode:    " << opts_.density_sigma_mode
      << "\n  deskew/sigma_r2:           " << opts_.deskew.sigma_r2
      << "\n  deskew/sigma_a2:           " << opts_.deskew.sigma_a2
      << "\n  deskew/time_based_process_noise: " << opts_.deskew.time_based_process_noise
      << "\n  ds/ds_leaf_size:           " << opts_.ds_leaf_size
      << "\n  ds/mode:                   " << opts_.ds_mode
      << "\n  cuda/enable:               " << (cuda_enable_ ? "true" : "false");
  return oss.str();
}

// Residual and EKF code
void LioProc::buildResiduals(
  const std::vector<PointXYZCov>& pts,
  std::vector<Residual>& residuals,
  bool allow_consistency_log) const {
  // Single-threaded, before the OMP region starts below -- see
  // VoxelMap::setAllowConsistencyLog()'s doc comment.
  if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get()))
    vm->setAllowConsistencyLog(allow_consistency_log);

  const int n = (int)pts.size();
  const int threads = cappedOmpThreads();

  build_thread_residuals_.resize(threads);
  for (auto& v : build_thread_residuals_) v.clear();
  build_thread_miss_.assign(threads, {0, 0});
  build_thread_tier0_miss_.assign(threads, {0, 0});

  // Frame-constant context for T0-D's corr.csv S column (H P- H^T + R) --
  // see WorldPointCov::body_point/rot_transpose/prior_cov_rp's doc
  // comment. Computed once here (not per point) and copied onto every
  // pt_world below; cheap regardless of whether logging is actually on
  // this call, so no separate gate is needed.
  const M3D rot_transpose = state_->rot().transpose();
  Eigen::Matrix<double, 6, 6> prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();
  if (prior_cov_.rows() >= StateGroup::idxR() + 6 && prior_cov_.cols() >= StateGroup::idxR() + 6)
    prior_cov_rp = prior_cov_.block<6, 6>(StateGroup::idxR(), StateGroup::idxR());

  #pragma omp parallel for schedule(static) num_threads(threads)
  for (int i = 0; i < n; ++i)
  {
    const PointXYZCov sensor_world = state_->toWorld(pts[i]);
    WorldPointCov pt_world{
        sensor_world.point, sensor_world.sensor_cov, state_->poseCovAt(pts[i].point)};
    pt_world.body_point = pts[i].point;
    pt_world.rot_transpose = rot_transpose;
    pt_world.prior_cov_rp = prior_cov_rp;
    Residual res{};
    bool tier0_had_plane = false;
    bool tier0_missed = true;
    if (voxel_map_->findPlaneResidual(pt_world, res, &tier0_had_plane)) {
      res.point_cross_normal = pts[i].point.cross(state_->rot().transpose() * res.normal);
      res.sigma_squared += res.plane_var_term;
      build_thread_residuals_[omp_get_thread_num()].push_back(res);
      tier0_missed = (res.match_tier != 0);
    } else {
      const int idx = voxel_map_->hasConvergedNeighbor(pt_world.point) ? 1 : 0;
      ++build_thread_miss_[omp_get_thread_num()][idx];
    }
    if (tier0_missed) {
      const int idx0 = tier0_had_plane ? 1 : 0;
      ++build_thread_tier0_miss_[omp_get_thread_num()][idx0];
    }
  }

  residuals.clear();
  for (const auto& local : build_thread_residuals_)
    residuals.insert(residuals.end(), local.begin(), local.end());

  n_miss_coverage_ = 0;
  n_miss_mismatch_ = 0;
  for (const auto& m : build_thread_miss_) {
    n_miss_coverage_ += m[0];
    n_miss_mismatch_ += m[1];
  }

  n_tier0_miss_coverage_ = 0;
  n_tier0_miss_mismatch_ = 0;
  for (const auto& m : build_thread_tier0_miss_) {
    n_tier0_miss_coverage_ += m[0];
    n_tier0_miss_mismatch_ += m[1];
  }
}

// Both paths now just call the shared accumulator (lio/lio_accumulator.h,
// 2026-08-14 -- extracted so CombinedProc can build the exact same HtH/Htz
// without depending on LioProc's internals, per the plan for an opt-in
// combined LIO+VIO EKF step) and then apply the update, unchanged from
// before the extraction.
void LioProc::solveSystem_cuda(const std::vector<Residual>& residuals) const {
  accumulateLioResidualsCuda(residuals, ekf_, cuda_buf_);
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
}

void LioProc::solveSystem(const std::vector<Residual>& residuals) const {
  accumulateLioResiduals(residuals, ekf_);

  // Mean-only update against the frame's FIXED prior (prior_cov_/
  // state_propagat_, snapshotted once in processLIO() before this frame's
  // iteration loop began) -- does not touch state_->cov(). Both this CPU
  // path and solveSystem_cuda() now delegate to the same ekf_
  // applyMeanUpdate()/applyCovarianceUpdate() machinery instead of each
  // inlining/duplicating the blend math themselves (see ekf.h for why the
  // old per-call state_->covMut() = P_new here was a correctness bug).
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
}

double LioProc::estimateStateCorrection(
  const std::vector<PointXYZCov>& pts,
  V3D &dtheta,
  V3D &dt,
  bool allow_consistency_log) {

  {
    TimedScope ts(profiler_, "lio/ekf/build_residuals");
    buildResiduals(pts, residuals_, allow_consistency_log);
  }
  if (residuals_.empty())
    return 0.0;

  if (opts_.density_sigma_ref > 0.0) {
    const double x = residuals_.size() / opts_.density_sigma_ref;
    double scale = x;
    if (opts_.density_sigma_mode == "sqrt")           scale = std::sqrt(x);
    else if (opts_.density_sigma_mode == "quadratic") scale = x * x;
    scale = std::max(1.0, scale);
    for (auto& r : residuals_) r.sigma_squared *= scale;
    last_density_scale_ = scale;
  }

  double avg_res = 0.0;
  for (const auto& r : residuals_)
    avg_res += std::abs(r.r);
  avg_res /= residuals_.size();

  {
    TimedScope ts(profiler_, "lio/ekf/solve");
    if (cuda_enable_)
      solveSystem_cuda(residuals_);
    else
      solveSystem(residuals_);
    dtheta = ekf_.dtheta;
    dt     = ekf_.dt;
  }

  return avg_res;
}

bool LioProc::accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res)
{
  if (voxel_map_->isEmpty()) return false;

  {
    TimedScope ts(profiler_, "lio/ekf/build_residuals");
    buildResiduals(mg.points, residuals_);
  }
  if (residuals_.empty()) return false;

  if (opts_.density_sigma_ref > 0.0) {
    const double x = residuals_.size() / opts_.density_sigma_ref;
    double scale = x;
    if (opts_.density_sigma_mode == "sqrt")           scale = std::sqrt(x);
    else if (opts_.density_sigma_mode == "quadratic") scale = x * x;
    scale = std::max(1.0, scale);
    for (auto& r : residuals_) r.sigma_squared *= scale;
    last_density_scale_ = scale;
  }

  avg_res = 0.0;
  for (const auto& r : residuals_)
    avg_res += std::abs(r.r);
  avg_res /= residuals_.size();

  {
    TimedScope ts(profiler_, "lio/ekf/accumulate_combined");
    if (cuda_enable_)
      accumulateLioResidualsCuda(residuals_, out, cuda_buf_);
    else
      accumulateLioResiduals(residuals_, out);
  }
  return true;
}

// LIO Processing

void LioProc::deskewAndDownsample(MeasureGroup& mg)
{
  TimedScope ts(profiler_, "lio/deskew");

  std::vector<PointXYZCov> deskewed;
  deskewPoints(state_, mg.poses, mg.image.t, mg.lidar_points, opts_.deskew, deskewed);
  if (opts_.ds_leaf_size > 0.0) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
  }

  // See LioProcOptions::dry_run_point_filter_num's doc comment. Identical
  // deskew/downsample treatment as the primary points above, just against
  // mg.dry_run_lidar_points -- left empty (no-op) unless the feature is on
  // and this frame actually got a second-decimation scan.
  if (opts_.dry_run_point_filter_num > 0 && !mg.dry_run_lidar_points.empty())
  {
    std::vector<PointXYZCov> dry_run_deskewed;
    deskewPoints(state_, mg.poses, mg.image.t, mg.dry_run_lidar_points, opts_.deskew, dry_run_deskewed);
    if (opts_.ds_leaf_size > 0.0) {
      DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
      voxelDownsample(dry_run_deskewed, mg.dry_run_points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
    } else {
      mg.dry_run_points = std::move(dry_run_deskewed);
    }
  }
}

// See LioProcOptions::dry_run_point_filter_num's doc comment and this
// method's declaration in lio_processing.h.
void LioProc::runDryRunShadowPass(const MeasureGroup& mg)
{
  if (mg.dry_run_points.empty()) return;

  // Swap state_ to a throwaway copy for the duration of this shadow pass --
  // every member function below (buildResiduals/solveSystem/
  // estimateStateCorrection) reads/writes state_ by following this pointer,
  // so reassigning it here means none of them ever touch the real shared
  // StateGroup other components (ImuProc/PubProc) hold a reference to.
  StateGroupPtr real_state = state_;
  state_ = std::make_shared<StateGroup>(*real_state);

  // prior_cov_/state_propagat_ are class members read directly by
  // solveSystem() -- save/restore around the shadow pass's own use of them,
  // same reasoning as the state_ swap above. residuals_/ekf_ are also class
  // members clobbered by estimateStateCorrection() below, but don't need
  // save/restore: the real loop right after this call unconditionally
  // overwrites both via its own first buildResiduals()/solveSystem() call
  // before ever reading them.
  Eigen::MatrixXd real_prior_cov = prior_cov_;
  StateGroup real_state_propagat = state_propagat_;

  prior_cov_ = state_->cov();
  state_propagat_ = *state_;

  V3D dtheta, dt;
  V3D total_dtheta = V3D::Zero(), total_dt = V3D::Zero();
  double prev_error = std::numeric_limits<double>::infinity();
  std::string stop = "max_iter";
  int iter = 0;
  bool any_solved = false;

  for (; iter < opts_.max_iterations; iter++) {
    // allow_consistency_log=false unconditionally -- this whole pass is
    // discarded (see this method's own doc comment), so none of its
    // residuals, first-iteration or not, belong in corr.csv.
    double error = estimateStateCorrection(mg.dry_run_points, dtheta, dt, /*allow_consistency_log=*/false);
    if (!residuals_.empty()) any_solved = true;
    total_dtheta += dtheta;
    total_dt     += dt;

    const double prev = prev_error;
    prev_error = error;

    if (dtheta.norm() < opts_.min_norm_dtheta && dt.norm() < opts_.min_norm_dt)
      { stop = "norm"; break; }
    if ((prev - error) / std::max(prev, 1e-6) < opts_.min_diff_error)
      { stop = "rel_diff"; break; }
  }

  // Applying the shadow covariance update (against the shadow's own
  // prior_cov_) is harmless -- state_ still points at the throwaway copy,
  // not the real shared StateGroup -- and lets trace(P_PP)/trace(P_RR)/
  // trace(P_VV) below reflect what a REAL update at this point_filter_num
  // would have produced, for direct comparison against the real frame's
  // own posterior trace further down in processLIO().
  if (any_solved)
    ekf_.applyCovarianceUpdate(state_, prior_cov_);

  // Same eigen-structure / frac_weak / frac_strong diagnostics as
  // processLIO()'s own [lio] block (see that block's doc comments for the
  // full rationale) -- computed here against residuals_ from this shadow
  // pass's LAST estimateStateCorrection() call, i.e. the shadow's own final
  // iteration, exactly mirroring how the real block's residuals_ reflects
  // its own final iteration.
  V3D mean_normal = V3D::Zero();
  double mean_abs_r = 0.0, max_abs_r = 0.0;
  double sum_sigma_squared = 0.0, min_sigma_squared = std::numeric_limits<double>::infinity(), max_sigma_squared = 0.0;
  double sum_weight = 0.0;
  int n_tier0 = 0, n_tier1 = 0, n_tier2 = 0;
  double sum_chi2 = 0.0;
  for (const auto& r : residuals_)
  {
    mean_normal += r.normal;
    const double abs_r = std::abs(r.r);
    mean_abs_r += abs_r;
    max_abs_r = std::max(max_abs_r, abs_r);

    sum_sigma_squared += r.sigma_squared;
    min_sigma_squared = std::min(min_sigma_squared, r.sigma_squared);
    max_sigma_squared = std::max(max_sigma_squared, r.sigma_squared);
    sum_weight += 1.0 / r.sigma_squared;
    sum_chi2 += (r.r * r.r) / r.sigma_squared;

    if (r.match_tier == 0) ++n_tier0;
    else if (r.match_tier == 1) ++n_tier1;
    else ++n_tier2;
  }
  if (!residuals_.empty())
  {
    mean_normal /= static_cast<double>(residuals_.size());
    mean_abs_r  /= static_cast<double>(residuals_.size());
  }
  const double mean_sigma_squared = residuals_.empty() ? 0.0 : sum_sigma_squared / residuals_.size();
  const double reduced_chi2 = residuals_.empty() ? 0.0 : sum_chi2 / residuals_.size();

  const M3D H_pp = ekf_.HtH.block<3, 3>(3, 3);
  Eigen::SelfAdjointEigenSolver<M3D> h_es(H_pp);
  const V3D h_pp_eig = h_es.eigenvalues();
  const M3D H_rr = ekf_.HtH.block<3, 3>(0, 0);
  Eigen::SelfAdjointEigenSolver<M3D> h_rr_es(H_rr);
  const V3D h_rr_eig = h_rr_es.eigenvalues();

  const V3D v_pp_weak   = h_es.eigenvectors().col(0);
  const V3D v_pp_strong = h_es.eigenvectors().col(2);
  const V3D v_rr_weak   = h_rr_es.eigenvectors().col(0);
  const V3D v_rr_strong = h_rr_es.eigenvectors().col(2);
  double sum_frac_weak_pp = 0.0, sum_frac_strong_pp = 0.0;
  double sum_frac_weak_rr = 0.0, sum_frac_strong_rr = 0.0;
  for (const auto& res : residuals_)
  {
    const double hp_sqnorm = res.normal.squaredNorm();
    if (hp_sqnorm > 1e-12) {
      sum_frac_weak_pp   += std::pow(res.normal.dot(v_pp_weak),   2) / hp_sqnorm;
      sum_frac_strong_pp += std::pow(res.normal.dot(v_pp_strong), 2) / hp_sqnorm;
    }
    const double hr_sqnorm = res.point_cross_normal.squaredNorm();
    if (hr_sqnorm > 1e-12) {
      sum_frac_weak_rr   += std::pow(res.point_cross_normal.dot(v_rr_weak),   2) / hr_sqnorm;
      sum_frac_strong_rr += std::pow(res.point_cross_normal.dot(v_rr_strong), 2) / hr_sqnorm;
    }
  }
  const double n_res_d = std::max<double>(1.0, static_cast<double>(residuals_.size()));
  const double mean_frac_weak_pp   = sum_frac_weak_pp   / n_res_d;
  const double mean_frac_strong_pp = sum_frac_strong_pp / n_res_d;
  const double mean_frac_weak_rr   = sum_frac_weak_rr   / n_res_d;
  const double mean_frac_strong_rr = sum_frac_strong_rr / n_res_d;

  const auto& P = state_->cov();
  const M3D P_RR = P.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
  const M3D P_PP = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
  const M3D P_VV = P.block<3, 3>(StateGroup::idxV(), StateGroup::idxV());

  const double t_abs = mg.image.t + data_queues_->start_time;
  const Eigen::Quaterniond q(state_->rot());

  std::ostringstream dbg;
  dbg << std::fixed << std::setprecision(6)
      << "[lio_dryrun]  t_abs=" << t_abs
      << "  dry_run_point_filter_num=" << opts_.dry_run_point_filter_num
      << "  iters=" << iter + 1
      << "  stop=" << stop
      << "  n_residuals=" << residuals_.size()
      << "  tier0=" << n_tier0 << "  tier1=" << n_tier1 << "  tier2=" << n_tier2
      << "  n_map_pts=" << voxel_map_->last_n_map_pts_ << "  n_active_voxels=" << voxel_map_->last_n_active_voxels_
      << "  pos=[" << state_->pos().transpose() << "]"
      << "  vel=[" << state_->vel().transpose() << "]"
      << "  quat(wxyz)=[" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "]"
      << "  total_dtheta=[" << total_dtheta.transpose() << "]"
      << "  total_dt=[" << total_dt.transpose() << "]"
      << "  mean_normal=[" << mean_normal.transpose() << "]"
      << "  mean_normal_norm=" << mean_normal.norm()
      << "  mean_abs_r=" << mean_abs_r
      << "  max_abs_r=" << max_abs_r
      << std::scientific
      << "  reduced_chi2=" << reduced_chi2
      << "  mean_sigma_squared=" << mean_sigma_squared
      << "  min_sigma_squared=" << (residuals_.empty() ? 0.0 : min_sigma_squared)
      << "  max_sigma_squared=" << max_sigma_squared
      << "  sum_weight=" << sum_weight
      << "  H_pp_eig=[" << h_pp_eig.transpose() << "]"
      << "  H_rr_eig=[" << h_rr_eig.transpose() << "]"
      << "  mean_frac_weak_pp=" << mean_frac_weak_pp
      << "  mean_frac_strong_pp=" << mean_frac_strong_pp
      << "  mean_frac_weak_rr=" << mean_frac_weak_rr
      << "  mean_frac_strong_rr=" << mean_frac_strong_rr
      << "  trace(P_RR)=" << P_RR.trace()
      << "  trace(P_PP)=" << P_PP.trace()
      << "  trace(P_VV)=" << P_VV.trace();

  debugLogLioDryRun(dbg.str());

  // Restore everything the shadow pass touched -- the real loop right after
  // this call must see exactly the same state_/prior_cov_/state_propagat_
  // it would have if this method had never run.
  state_ = real_state;
  prior_cov_ = real_prior_cov;
  state_propagat_ = real_state_propagat;
}

std::string LioProc::processLIO(MeasureGroup& mg)
{
  if (voxel_map_->isEmpty()) return {};

  if (voxel_map_->isEmpty()) return {};

  if (opts_.dry_run_point_filter_num > 0 && opts_.log_debug_en)
    runDryRunShadowPass(mg);

  {
    TimedScope ts(profiler_, "lio/ekf");
    V3D dtheta, dt;
    V3D total_dtheta = V3D::Zero(), total_dt = V3D::Zero();
    double prev_error = std::numeric_limits<double>::infinity();
    std::string stop = "max_iter";
    int iter = 0;

    // Fixed IEKF prior for this frame's ENTIRE inner loop -- see ekf.h's
    // applyMeanUpdate() doc comment. Set once here, read (never rewritten)
    // by every solveSystem()/solveSystem_cuda() call below until the loop
    // finishes and applyCovarianceUpdate() is called exactly once.
    prior_cov_ = state_->cov();
    state_propagat_ = *state_;
    bool any_solved = false;

    for (; iter < opts_.max_iterations; iter++) {
      // T0-D wants the first-iteration (pre-update, un-relinearized)
      // innovation only -- later iterations relinearize at an
      // already-partially-corrected state, which is not the quantity NIS
      // is defined over.
      double error = estimateStateCorrection(mg.points, dtheta, dt, /*allow_consistency_log=*/iter == 0);
      if (!residuals_.empty()) any_solved = true;
      total_dtheta += dtheta;
      total_dt     += dt;

      // T0-E: same first-iteration-only scope as T0-D's corr.csv/scan.csv
      // (see estimateStateCorrection()'s allow_consistency_log doc
      // comment) -- this frame's prior_cov_/ekf_.HtH/ekf_.Htz are exactly
      // what this iteration's solveSystem() just accumulated.
      // 2026-08-31 code-audit fixes (see LioProcOptions::log_nll_en's doc
      // comment): (1) log EVERY frame, including zero-residual ones
      // (n_residuals=0, nll=0.5*(logdet(P0)+logdet(A)) is still
      // well-defined there since HtH/Htz vanish) -- line count is now
      // frame-count-invariant, not correspondence-count-invariant, so
      // runs at different alpha stay row-comparable by index/timestamp.
      // (2) n_residuals logged explicitly so callers can normalize
      // per-correspondence (nll/n_residuals, or sum(nll)/sum(n_residuals)
      // across a run) instead of summing a quantity whose per-frame N
      // varies with both frame content AND alpha. (3) the previously-
      // dropped n*log(2*pi) Gaussian normalization term is now included
      // -- omitting it is harmless for a FIXED N but not when N varies
      // between the runs being compared, which is exactly this
      // objective's use case.
      if (iter == 0 && opts_.log_nll_en)
      {
        const int n_res = static_cast<int>(residuals_.size());
        double nll = 0.0;
        if (n_res > 0) {
          // Only defined when solveSystem() actually ran this iteration
          // (estimateStateCorrection() returns early on an empty
          // residuals_ WITHOUT calling solveSystem(), which is the only
          // thing that (re-)accumulates ekf_.HtH/Htz -- calling
          // nllQuadraticAndLogdet() in the n_res==0 branch would silently
          // read a STALE HtH/Htz left over from a previous frame).
          double sum_chi2 = 0.0, sum_log_sigma2 = 0.0;
          for (const auto& r : residuals_) {
            sum_chi2 += (r.r * r.r) / r.sigma_squared;
            sum_log_sigma2 += std::log(r.sigma_squared);
          }
          nll = 0.5 * (n_res * std::log(2.0 * M_PI) + sum_log_sigma2 + sum_chi2
                       + ekf_.nllQuadraticAndLogdet(prior_cov_));
        }
        debugLogNll(mg.image.t + data_queues_->start_time, nll, n_res);
      }

      const double prev = prev_error;
      prev_error = error;

      if (opts_.log_debug_en)
      {
        const double t_abs = mg.image.t + data_queues_->start_time;
        std::ostringstream iss;
        iss << "t_abs=" << std::fixed << std::setprecision(6) << t_abs
            << "  iter=" << iter << "  n_residuals=" << residuals_.size()
            << "  avg_abs_r=" << std::scientific << std::setprecision(6) << error
            << "  rel_diff=" << ((prev - error) / std::max(prev, 1e-6));
        debugLogIterError(iss.str());
      }

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

    // Posterior covariance write happens exactly ONCE here, using the SAME
    // fixed prior_cov_ every real iteration above blended against (see
    // ekf.h) -- not per-iteration like the old applyUpdate() did. No-op
    // (skipped) when any_solved is false: no real measurement info was ever
    // accumulated this frame (ekf_'s H_full would be stale from a previous
    // frame), so there's nothing to blend and state_->cov() should stay
    // exactly as it was going into this frame.
    if (any_solved)
      ekf_.applyCovarianceUpdate(state_, prior_cov_);

    if (opts_.log_consistency_scan_en)
    {
      const double t_abs = mg.image.t + data_queues_->start_time;
      const double dt = (last_scan_t_abs_ < 0.0) ? 0.0 : (t_abs - last_scan_t_abs_);
      last_scan_t_abs_ = t_abs;

      const auto& P = state_->cov();
      const double trP_pos = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();
      const double trP_vel = P.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()).trace();
      const double trP_att = P.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()).trace();

      // mg.imu_samples is already cleared by ImuProc::propagate() (see its
      // doc comment there) by the time processLIO() runs -- mg.poses is
      // the per-IMU-step record propagate() leaves behind instead. `gyr`
      // is the bias-corrected body-frame angular velocity used for each
      // step's rotation propagation; acc_head/acc_tail are WORLD-frame
      // true linear acceleration (gravity already added back, see Pose6D's
      // doc comment) -- their average norm is near 0 for a near-stationary/
      // constant-velocity segment and grows with genuine dynamic motion,
      // which is the excitation signal T0-D's dynamics-slope panel wants
      // (a cleaner proxy than raw specific-force magnitude, which would be
      // dominated by the ~9.8 m/s^2 gravity constant).
      double sum_omega = 0.0, sum_acc = 0.0;
      for (const auto& p : mg.poses) {
        sum_omega += p.gyr.norm();
        sum_acc   += (0.5 * (p.acc_head + p.acc_tail)).norm();
      }
      const double n_poses = std::max<size_t>(1, mg.poses.size());
      debugLogConsistencyScan(voxel_map_->frame_idx_, t_abs, dt, trP_pos, trP_vel, trP_att,
                               sum_omega / n_poses, sum_acc / n_poses);
    }

    std::ostringstream oss;
    oss << "[lio/ekf] iters=" << iter + 1 << "  stop=" << stop
        << std::scientific << std::setprecision(1)
        << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
        << "  |dt|=" << total_dt.norm() * 1000.0 << " mm";

    if (opts_.log_debug_en)
    {
      const double t_abs = mg.image.t + data_queues_->start_time;
      const Eigen::Quaterniond q(state_->rot());
      // Pre-correction pose, captured by estimateState() right after
      // processIMU() (pure propagation, no LIO update applied yet) --
      // lets a reader compute this frame's actual correction vector
      // (state_->pos() - mg.pos_after_imu) directly, instead of only
      // inferring it indirectly via aligned-ATE deltas in evo.txt.
      const Eigen::Quaterniond q_before(mg.rot_after_imu);
      const auto& P = state_->cov();
      const M3D P_RR = P.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
      const M3D P_PP = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
      const M3D P_VV = P.block<3, 3>(StateGroup::idxV(), StateGroup::idxV());
      // Position-velocity cross-covariance -- this is the ONLY channel
      // through which a position-only measurement (LIO's HtH/Htz only
      // ever touch the R/P block, never V directly) can correct velocity
      // at all: the update's effect on V is entirely mediated by how
      // correlated P and V already are in the prior. Its norm answers
      // "how much of this frame's position correction, if any, leaked into
      // a velocity correction" -- near 0 means velocity got essentially no
      // benefit from this update, regardless of how good the position fix was.
      const M3D P_PV = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxV());

      // Plane/residual derivation summary for this frame's final iteration
      // (residuals_ reflects the last buildResiduals() call inside the loop
      // above) -- mean_normal_norm close to 1 means nearly every residual
      // this frame shares almost the same plane normal (dominated by one
      // surface -- expected while facing a single wall, but also the
      // signature of degenerate/spurious planes all landing on the same
      // artifact direction, e.g. along the LIDAR's own scan lines during a
      // near-stationary period); close to 0 means normals are diverse
      // (multiple, differently-oriented real surfaces contributing).
      V3D mean_normal = V3D::Zero();
      double mean_abs_r = 0.0, max_abs_r = 0.0;
      // sigma/weight stats (VARIANCE throughout, not std-dev, despite the
      // "sigma" name -- see Residual::sigma_squared's docs) -- testing
      // whether livo_recon's per-residual measurement noise (res.
      // sigma_squared = 1e-3 + sigma_diag_squared, sigma_diag_squared =
      // point sensor noise projected onto the plane normal + the plane's
      // own fit uncertainty) is systematically smaller (more confident)
      // than FAST-LIVO2's structurally-identical R_inv formula, which would
      // explain livo_recon's tighter aggregate H_full/posterior even after
      // matching residual density via ds_leaf_size.
      double sum_sigma_squared = 0.0, min_sigma_squared = std::numeric_limits<double>::infinity(), max_sigma_squared = 0.0;
      double sum_weight = 0.0;
      // Tier breakdown (see Residual::match_tier's docs): how many of this
      // frame's residuals came from the primary voxel vs. the single
      // directional neighbor (also present in FAST-LIVO2) vs. the full
      // neighborhood-box search (livo_recon-only) -- debugging whether a
      // spike in tier-2 usage correlates with degraded match quality
      // (rising mean_sigma_squared/max_abs_r) during dynamic motion.
      int n_tier0 = 0, n_tier1 = 0, n_tier2 = 0;
      // Post-fit normalized-residual consistency check (2026-08-03, NIS-
      // style -- see the Woodbury/Sherman-Morrison-per-plane discussion):
      // unlike trace(P_PP)/trace(P_RR) above (which only reflect the
      // filter's own INTERNAL belief about its uncertainty), this checks
      // that belief against what the ACTUAL post-fit residuals look like.
      // sum_chi2 = Sum(r_i^2/sigma_squared_i) over this frame's residuals;
      // reduced_chi2 = sum_chi2/n_residuals should average ~1 for a
      // correctly-calibrated noise model (each residual's squared error
      // should be on the same scale as its own claimed variance). >>1 means
      // sigma_squared is set too small somewhere (overconfident -- actual
      // errors are bigger than the model expects); <<1 means it's set too
      // large (overly conservative).
      double sum_chi2 = 0.0;
      for (const auto& r : residuals_)
      {
        mean_normal += r.normal;
        const double abs_r = std::abs(r.r);
        mean_abs_r += abs_r;
        max_abs_r = std::max(max_abs_r, abs_r);

        sum_sigma_squared += r.sigma_squared;
        min_sigma_squared = std::min(min_sigma_squared, r.sigma_squared);
        max_sigma_squared = std::max(max_sigma_squared, r.sigma_squared);
        sum_weight += 1.0 / r.sigma_squared;
        sum_chi2 += (r.r * r.r) / r.sigma_squared;

        if (r.match_tier == 0) ++n_tier0;
        else if (r.match_tier == 1) ++n_tier1;
        else ++n_tier2;
      }
      if (!residuals_.empty())
      {
        mean_normal /= static_cast<double>(residuals_.size());
        mean_abs_r  /= static_cast<double>(residuals_.size());
      }
      const double mean_sigma_squared = residuals_.empty() ? 0.0 : sum_sigma_squared / residuals_.size();
      const double reduced_chi2 = residuals_.empty() ? 0.0 : sum_chi2 / residuals_.size();

      // Per-axis structure of this frame's position information sub-block
      // (H^T*W*H, position rows/cols only) -- unlike the scalar sum_weight
      // above, this tests whether the aggregate weight is concentrated
      // (anisotropic, one huge eigenvalue) vs. well-distributed across all
      // three translation axes. ekf_.HtH is 6x6 (local layout: rows/cols
      // 0-2 rotation, 3-5 position) as accumulated by solveSystem() this
      // iteration.
      const M3D H_pp = ekf_.HtH.block<3, 3>(3, 3);
      Eigen::SelfAdjointEigenSolver<M3D> h_es(H_pp);
      const V3D h_pp_eig = h_es.eigenvalues();

      // Rotation analog of H_pp_eig above: eigenvalues of the rotation-rows/
      // cols sub-block of this frame's H^T*W*H. A small minimum eigenvalue
      // here means at least one rotational DOF is weakly constrained by
      // this frame's residual set (an aperture-problem/gauge-freedom
      // signature -- e.g. residuals concentrated on too few distinct plane
      // orientations to pin down all 3 rotational DOF), regardless of how
      // low mean_sigma_squared/max_abs_r look per-residual. Logged to test whether
      // livo_recon's rotation estimate is chronically under-constrained
      // relative to FAST-LIVO2's (same field added there, see voxel_map.cpp)
      // or only becomes so transiently during fast motion.
      const M3D H_rr = ekf_.HtH.block<3, 3>(0, 0);
      Eigen::SelfAdjointEigenSolver<M3D> h_rr_es(H_rr);
      const V3D h_rr_eig = h_rr_es.eigenvalues();

      // Per-residual eigen-projection diagnostic (2026-08-03, temporary --
      // see the pfn1_ds000-vs-pfn1_ds010 confidence-analysis finding that
      // H_pp/H_rr's AGGREGATE eigenvalues were nearly identical between the
      // two despite ds000 having ~15% more residuals every frame). Tests
      // whether that extra residual mass is structurally redundant --
      // concentrated on the ALREADY-well-constrained (largest-eigenvalue)
      // direction -- rather than contributing to the weakest direction,
      // by projecting each residual's own Jacobian (unweighted by its own
      // sigma_squared, a purely geometric/structural quantity) onto H_pp/
      // H_rr's eigenvectors and averaging the normalized squared-projection
      // fraction across all residuals this frame. A LOW mean_frac_weak
      // means most residuals barely touch the weakest direction at all
      // (piling onto the strong one instead) -- direct, per-residual
      // evidence for or against the "extra residuals mostly redundant"
      // hypothesis this whole investigation has been built on, complementing
      // (not replacing) the aggregate H_pp_eig/H_rr_eig numbers already
      // logged above. Eigen::SelfAdjointEigenSolver returns ascending
      // eigenvalues, so col(0)=weakest, col(2)=strongest.
      const V3D v_pp_weak   = h_es.eigenvectors().col(0);
      const V3D v_pp_strong = h_es.eigenvectors().col(2);
      const V3D v_rr_weak   = h_rr_es.eigenvectors().col(0);
      const V3D v_rr_strong = h_rr_es.eigenvectors().col(2);
      double sum_frac_weak_pp = 0.0, sum_frac_strong_pp = 0.0;
      double sum_frac_weak_rr = 0.0, sum_frac_strong_rr = 0.0;
      for (const auto& res : residuals_)
      {
        const double hp_sqnorm = res.normal.squaredNorm();
        if (hp_sqnorm > 1e-12) {
          sum_frac_weak_pp   += std::pow(res.normal.dot(v_pp_weak),   2) / hp_sqnorm;
          sum_frac_strong_pp += std::pow(res.normal.dot(v_pp_strong), 2) / hp_sqnorm;
        }
        const double hr_sqnorm = res.point_cross_normal.squaredNorm();
        if (hr_sqnorm > 1e-12) {
          sum_frac_weak_rr   += std::pow(res.point_cross_normal.dot(v_rr_weak),   2) / hr_sqnorm;
          sum_frac_strong_rr += std::pow(res.point_cross_normal.dot(v_rr_strong), 2) / hr_sqnorm;
        }
      }
      const double n_res_d = std::max<double>(1.0, static_cast<double>(residuals_.size()));
      const double mean_frac_weak_pp   = sum_frac_weak_pp   / n_res_d;
      const double mean_frac_strong_pp = sum_frac_strong_pp / n_res_d;
      const double mean_frac_weak_rr   = sum_frac_weak_rr   / n_res_d;
      const double mean_frac_strong_rr = sum_frac_strong_rr / n_res_d;

      // VoxelStats (converged/disabled/init_frames_sum/etc) describes a
      // convergence-lock-in concept only VoxelMap has -- AkfMap (or any
      // future backend) has no equivalent, so this diagnostic is gated on
      // an actual VoxelMap rather than promoted to the MapBackend
      // interface (see plan at ~/.claude/plans/imperative-tumbling-karp.md).
      std::ostringstream voxel_stats_dbg;
      if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get())) {
        voxel_stats_dbg << "  init_frames_sum=" << vm->stats()->init_frames_sum.load()
                        << "  init_count=" << vm->stats()->init_count.load()
                        << "  update_frames_sum=" << vm->stats()->update_frames_sum.load()
                        << "  update_count=" << vm->stats()->update_count.load();
      } else {
        voxel_stats_dbg << "  init_frames_sum=n/a  init_count=n/a  update_frames_sum=n/a  update_count=n/a";
      }

      std::ostringstream dbg;
      dbg << std::fixed << std::setprecision(6)
          << "[lio]  t_abs=" << t_abs
          << "  iters=" << iter + 1
          << "  stop=" << stop
          << "  n_residuals=" << residuals_.size()
          << "  tier0=" << n_tier0 << "  tier1=" << n_tier1 << "  tier2=" << n_tier2
          << "  miss_coverage=" << n_miss_coverage_ << "  miss_mismatch=" << n_miss_mismatch_
          << "  tier0_miss_coverage=" << n_tier0_miss_coverage_ << "  tier0_miss_mismatch=" << n_tier0_miss_mismatch_
          << "  n_map_pts=" << voxel_map_->last_n_map_pts_ << "  n_active_voxels=" << voxel_map_->last_n_active_voxels_
          << voxel_stats_dbg.str()
          << "  pos=[" << state_->pos().transpose() << "]"
          << "  vel=[" << state_->vel().transpose() << "]"
          << "  quat(wxyz)=[" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "]"
          << "  pos_before=[" << mg.pos_after_imu.transpose() << "]"
          << "  quat_before(wxyz)=[" << q_before.w() << " " << q_before.x() << " " << q_before.y() << " " << q_before.z() << "]"
          << "  total_dtheta=[" << total_dtheta.transpose() << "]"
          << "  total_dt=[" << total_dt.transpose() << "]"
          << "  mean_normal=[" << mean_normal.transpose() << "]"
          << "  mean_normal_norm=" << mean_normal.norm()
          << "  mean_abs_r=" << mean_abs_r
          << "  max_abs_r=" << max_abs_r
          << std::scientific
          << "  reduced_chi2=" << reduced_chi2
          << "  mean_sigma_squared=" << mean_sigma_squared
          << "  min_sigma_squared=" << (residuals_.empty() ? 0.0 : min_sigma_squared)
          << "  max_sigma_squared=" << max_sigma_squared
          << "  sum_weight=" << sum_weight
          << "  H_pp_eig=[" << h_pp_eig.transpose() << "]"
          << "  H_rr_eig=[" << h_rr_eig.transpose() << "]"
          << "  mean_frac_weak_pp=" << mean_frac_weak_pp
          << "  mean_frac_strong_pp=" << mean_frac_strong_pp
          << "  mean_frac_weak_rr=" << mean_frac_weak_rr
          << "  mean_frac_strong_rr=" << mean_frac_strong_rr
          << "  density_scale=" << last_density_scale_
          << "  trace(P_RR)=" << P_RR.trace()
          << "  trace(P_PP)=" << P_PP.trace()
          << "  trace(P_VV)=" << P_VV.trace()
          << "  norm(P_PV)=" << P_PV.norm();

      // Full post-LIO state + covariance dump, for direct raw-value
      // comparison against FAST-LIVO2's equivalent (see LIVMapper.cpp's
      // handleLIO() debug block) -- everything above this point already
      // summarizes P via trace/norm of individual blocks, but a from-
      // scratch cross-system comparison (e.g. per-axis correction size vs.
      // per-axis covariance, or reconstructing the full Kalman gain) needs
      // the raw entries, not just those derived scalars. cov_dim is
      // state_->dimState() (9 always: R,P,V, plus BG/BA/gravity if
      // estimated); cov_flat is P flattened ROW-MAJOR (Eigen's default
      // storage is column-major, so this is NOT a raw memory dump).
      dbg << "  dim=" << P.rows()
          << "  bias_gyr=[" << state_->biasGyr().transpose() << "]"
          << "  bias_acc=[" << state_->biasAcc().transpose() << "]"
          << "  gravity=[" << state_->gravity().transpose() << "]"
          << "  cov_flat=[";
      for (int r = 0; r < P.rows(); r++)
        for (int c = 0; c < P.cols(); c++)
          dbg << P(r, c) << (r == P.rows() - 1 && c == P.cols() - 1 ? "" : " ");
      dbg << "]";

      debugLogLio(dbg.str());
    }

    return oss.str();
  }
}

// Voxel map update
}
