#include "livo_recon/processing/lio_processing.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/consistency_log.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/voxelplane.h"   // voxelPlaneInformationFitCount()
#include "livo_recon/lio/lio_accumulator.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>

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

// History (40-42): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-40
void debugLogIterError(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("iter_error.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// History (51-54): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-51
void debugLogLioDryRun(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("lio_dryrun.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// History (63-70): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-63

// History (72-75): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-72

// History (77-89): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-77
void debugLogQhat(int scan_id, double t_abs, const Eigen::VectorXd& dx,
                  const Eigen::VectorXd& expected_diag,
                  const Eigen::VectorXd& acw_diag)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  const int n = static_cast<int>(dx.size());
  std::ofstream ofs(debugLogPath("qhat.csv"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call) {
    ofs << "scan_id,t,dim";
    for (int i = 0; i < n; ++i) ofs << ",dx" << i;
    for (int i = 0; i < n; ++i) ofs << ",exp" << i;
    for (int i = 0; i < n; ++i) ofs << ",acw" << i;
    ofs << "\n";
  }
  first_call = false;
  ofs << scan_id << "," << t_abs << "," << n;
  for (int i = 0; i < n; ++i) ofs << "," << dx(i);
  for (int i = 0; i < n; ++i) ofs << "," << expected_diag(i);
  for (int i = 0; i < n; ++i) ofs << "," << acw_diag(i);
  ofs << "\n";
}

}  // namespace

LioProc::LioProc(NodeContext& ctx)
  : state_(ctx.state), voxel_map_(ctx.voxel_map), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

std::string LioProc::loadParameters(ros::NodeHandle& pnh)
{
  // One resolver for the whole function: modes validated against a named
  // set, nested keys read only inside a live scope and REFUSED if set into a
  // dead one, and the EFFECTIVE configuration returned rather than the
  // requested one.  See config_resolve.h.
  ConfigResolver cfg(pnh);

  paramWarn<int>(pnh, "lio/ekf/max_iterations",     opts_.max_iterations,  5);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dtheta", opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dt",     opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "lio/ekf/min_diff_error",  opts_.min_diff_error,  -1.0);
  paramWarn<bool>(pnh, "lio/log_debug_en",          opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "lio/log_consistency_scan_en", opts_.log_consistency_scan_en, false);
  paramWarn<bool>(pnh, "lio/log_nll_en", opts_.log_nll_en, false);
  paramWarn<int>(pnh, "lio/dry_run_point_filter_num", opts_.dry_run_point_filter_num, 0);
  paramWarn<bool>(pnh, "cuda/enable",               cuda_enable_,          false);

  // History (134-136): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-134
  double range_err;
  paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
  opts_.deskew.sigma_r2 = range_err * range_err;
  double angle_err_deg;
  paramWarn<double>(pnh, "imu/sensor/angle_err_deg", angle_err_deg, 0.2);
  const double sin_angle_err = std::sin(std::max(1e-6, angle_err_deg * M_PI / 180.0));
  opts_.deskew.sigma_a2 = sin_angle_err * sin_angle_err;
  // ── scan spline ──────────────────────────────────────────────────────
  // Everything from here down goes through ConfigResolver, not paramWarn:
  // nested keys are read only inside a live scope and REFUSED if set into a
  // dead one, modes are validated against a named set, and what gets printed
  // is the effective configuration.  See config_resolve.h for why.
  cfg.get<bool>("spline/enable", opts_.spline.enable, false);
  const bool sp = opts_.spline.enable;

  cfg.nestedMode(sp, "spline/enable", "spline/control_points/mode",
                 opts_.spline.cp_mode, "n", { "n", "hz" });
  cfg.nested<int>(sp && !opts_.spline.cpFromHz(), "spline/control_points/mode=n",
                  "spline/control_points/n", opts_.spline.n_control_points, 8);
  cfg.nested<double>(sp && opts_.spline.cpFromHz(), "spline/control_points/mode=hz",
                     "spline/control_points/hz", opts_.spline.control_point_hz, 0.0);
  cfg.nested<int>(sp, "spline/enable", "spline/control_points/n_max",
                  opts_.spline.n_control_points_max, 32);

  cfg.nested<double>(sp, "spline/enable", "spline/fit/regularization",
                     opts_.spline.fit_regularization, 1e-6);
  cfg.nested<double>(sp, "spline/enable", "spline/fit/reg_max_frac",
                     opts_.spline.fit_reg_max_frac, 0.05);
  cfg.nestedMode(sp, "spline/enable", "spline/fit/rotation/mode",
                 opts_.spline.rot_mode, "cumulative", { "cumulative", "tangent" });
  // A single linear solve under "tangent" -- the iteration count is not a
  // parameter of that arm at all.
  cfg.nested<int>(sp && opts_.spline.rotCumulative(),
                  "spline/fit/rotation/mode=cumulative",
                  "spline/fit/rotation/iters", opts_.spline.rot_fit_iters, 4);

  cfg.nestedMode(sp, "spline/enable", "spline/fit/imu_term/mode",
                 opts_.spline.imu_fit_mode, "off", { "off", "acc", "gyr", "both" });
  cfg.nested<double>(sp && opts_.spline.imuFitAcc(), "spline/fit/imu_term/mode",
                     "spline/fit/imu_term/w_acc", opts_.spline.imu_fit_w_acc, 0.0);
  cfg.nested<double>(sp && opts_.spline.imuFitGyr(), "spline/fit/imu_term/mode",
                     "spline/fit/imu_term/w_gyr", opts_.spline.imu_fit_w_gyr, 0.0);
  cfg.nested<int>(sp && opts_.spline.imuFitOn(), "spline/fit/imu_term/mode",
                  "spline/fit/imu_term/iters", opts_.spline.imu_fit_iters, 2);

  // The per-iteration pipeline: ONE mode with five values, replacing the
  // three booleans whose 2x2x2 space had only five distinct behaviours.  See
  // SplineOptions::per_iteration.
  cfg.nestedMode(sp, "spline/enable", "spline/per_iteration/mode",
                 opts_.spline.per_iteration, "redeskew",
                 { "off", "redeskew", "redeskew+refine",
                   "redeskew+reintegrate", "redeskew+refine+reintegrate" });
  const bool rf = sp && opts_.spline.refineOn();
  const bool ri = sp && opts_.spline.reintegrateOn();

  cfg.nested<double>(rf, "spline/per_iteration/mode (no +refine)",
                     "spline/per_iteration/refine/damping",
                     opts_.spline.lidar_refine_damping, 1e-2);
  cfg.nested<double>(rf, "spline/per_iteration/mode (no +refine)",
                     "spline/per_iteration/refine/prior_w",
                     opts_.spline.lidar_refine_prior_w, 1.0);
  cfg.nested<int>(rf, "spline/per_iteration/mode (no +refine)",
                  "spline/per_iteration/refine/iters",
                  opts_.spline.lidar_refine_iters, 1);
  cfg.nested<double>(rf, "spline/per_iteration/mode (no +refine)",
                     "spline/per_iteration/refine/max_step",
                     opts_.spline.lidar_refine_max_step, 0.10);
  cfg.nested<double>(ri, "spline/per_iteration/mode (no +reintegrate)",
                     "spline/per_iteration/reintegrate/min_dbg",
                     opts_.spline.reintegrate_min_dbg, 1e-9);
  cfg.nested<double>(ri, "spline/per_iteration/mode (no +reintegrate)",
                     "spline/per_iteration/reintegrate/min_dba",
                     opts_.spline.reintegrate_min_dba, 1e-9);

  cfg.nested<bool>(sp, "spline/enable", "spline/log_en", opts_.spline.log_en, false);
  cfg.nested<bool>(sp, "spline/enable", "spline/keep_time_noise",
                   opts_.spline_keep_time_noise, false);

  // ── live process-noise estimation ────────────────────────────────────
  // The statistic IS the spline-vs-IMU residual, so this scope is nested
  // under the spline rather than merely warned about.
  cfg.nested<bool>(sp, "spline/enable", "adaptive_q/enable",
                   opts_.adaptive_q.enable, false);
  const bool aq = sp && opts_.adaptive_q.enable;
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/beta_acc", opts_.adaptive_q.beta_acc, 0.3);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/beta_gyr", opts_.adaptive_q.beta_gyr, 0.3);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/z_rate_limit", opts_.adaptive_q.z_rate_limit, 0.02);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/acf1_max", opts_.adaptive_q.acf1_max, 0.35);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/bounds/max_ratio", opts_.adaptive_q.max_ratio, 100.0);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/bounds/min_ratio", opts_.adaptive_q.min_ratio, 0.01);
  cfg.nested<int>(aq, "adaptive_q/enable", "adaptive_q/warmup_frames", opts_.adaptive_q.warmup_frames, 20);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/ema", opts_.adaptive_q.ema, 0.9);
  cfg.nestedMode(aq, "adaptive_q/enable", "adaptive_q/noise_floor/mode",
                 adaptive_q_floor_mode_, "allan", { "allan", "off" });
  opts_.adaptive_q.use_noise_floor = (adaptive_q_floor_mode_ == "allan");
  cfg.nested<double>(aq && opts_.adaptive_q.use_noise_floor,
                     "adaptive_q/noise_floor/mode=allan",
                     "adaptive_q/noise_floor/scale", opts_.adaptive_q.noise_floor_scale, 1.0);
  cfg.nested<bool>(aq, "adaptive_q/enable", "adaptive_q/log_en", opts_.adaptive_q.log_en, false);
  adaptive_q_.configure(opts_.adaptive_q);

  // Downsampling is one axis with three states, not a mode plus a magic
  // zero.  "ds_leaf_size = 0.0 means off" made imu/ds/mode silently INERT
  // whenever downsampling was disabled -- and LD-1's R4 rung turns
  // downsampling off deliberately, so that cell would have carried a mode
  // setting nothing consumed.  Off is now a value of the mode, and the leaf
  // size is a sub-option of it being on.
  cfg.mode("imu/ds/mode", opts_.ds_mode, "first", { "off", "first", "average" });
  cfg.nested<double>(opts_.dsOn(), "imu/ds/mode != off",
                     "imu/ds/ds_leaf_size", opts_.ds_leaf_size, 0.15);
  if (!opts_.dsOn()) opts_.ds_leaf_size = 0.0;
  else if (!(opts_.ds_leaf_size > 0.0))
    cfg.requireCombination(
        "imu/ds/ds_leaf_size must be > 0 when imu/ds/mode is '" + opts_.ds_mode +
        "' -- a zero leaf under an ON mode is the sentinel this mode set "
        "exists to remove; write imu/ds/mode: off instead");
  cfg.mode("imu/undistort/time_based_process_noise",
           opts_.deskew.time_based_process_noise, "var_acc",
           { "none", "state", "var_acc" });
  // Same defect, same fix.  density_sigma_mode was a validated enum whose
  // entire mechanism is gated on density_sigma_ref > 0.0 -- so the mode was
  // inert unless a DIFFERENT key was set, and that key used 0.0 as its own
  // off switch.  No config in tree sets either, so the live behaviour is
  // unchanged; what changes is that the dead state is now named.
  cfg.mode("lio/ekf/density_sigma_mode", opts_.density_sigma_mode, "off",
           { "off", "linear", "sqrt", "quadratic" });
  cfg.nested<double>(opts_.densitySigmaOn(), "lio/ekf/density_sigma_mode != off",
                     "lio/ekf/density_sigma_ref", opts_.density_sigma_ref, 0.0);
  if (!opts_.densitySigmaOn()) opts_.density_sigma_ref = 0.0;
  else if (!(opts_.density_sigma_ref > 0.0))
    cfg.requireCombination(
        "lio/ekf/density_sigma_ref must be > 0 when lio/ekf/density_sigma_mode "
        "is '" + opts_.density_sigma_mode + "' -- otherwise the mode is inert");

  // Every spline/* and adaptive_q/* key is read by the resolver above and by
  // nothing else, so anything left over in those namespaces is a key nobody
  // consumes -- a rename this config was never migrated through, or a typo.
  // Deliberately NOT extended to voxel_map/: several of its keys are still
  // read by paramWarn() in voxelmap.cpp, so they are unclaimed here and would
  // be reported as dead when they are merely read elsewhere. Widen this only
  // as those move onto a resolver.
  cfg.refuseUnclaimed({ "spline", "adaptive_q" });

  // The build's accumulation precision, recorded rather than assumed. The
  // CUDA path returns float-precision HtH/Htz, so nll.txt is NOT comparable
  // between a CPU and a GPU build -- an alpha ladder run half on each is a
  // ladder with a step in it. The bug ledger's complaint was that this has to
  // be pinned and is nowhere documented; documenting it in the run's own
  // effective config is the cheap half of the fix.
  cfg.derived("build/accumulation_precision",
              cuda_enable_ ? "float (CUDA HtH/Htz) -- nll.txt NOT comparable "
                             "with a CPU build"
                           : "double (CPU HtH/Htz)");

  // The EFFECTIVE configuration, not the requested one -- and a hard refusal
  // if any key was set into a scope that cannot read it.  A sweep cell that
  // cannot mean what it says fails here rather than producing a duplicate
  // forty minutes later.
  if (!cfg.ok())
  {
    ROS_FATAL_STREAM("\n" << cfg.report());
    throw std::runtime_error(
        "[config] refused: " + std::to_string(cfg.errors().size()) +
        " option(s) set into a dead scope or outside their allowed set -- "
        "see the [config/REFUSED] block above");
  }
  return cfg.report();
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
      res.t = pts[i].t;   // for the spline control-point refinement
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

// History (242-246): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-242
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

  if (opts_.densitySigmaOn()) {
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

  if (opts_.densitySigmaOn()) {
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

  // ── fit this frame's trajectory spline ────────────────────────────────
  // The window runs from the first IMU pose to the frame reference time so
  // that every LiDAR return in the scan lands inside the fitted domain --
  // ScanSpline refuses to extrapolate, and a point outside the window would
  // otherwise be silently clamped to an endpoint.
  spline_ok_ = false;
  ds_offsets_.clear();
  ds_members_.clear();
  spline_refits_ = 0;
  redeskew_calls_ = 0;
  redeskew_dp_rms_ = 0.0;
  reint_dp_max_ = 0.0;
  reint_drot_deg_max_ = 0.0;
  spline_poses0_.clear();
  // Per-FRAME, not per-fit: fit() no longer resets these, because with
  // reintegrate_each_iteration on it runs once per IEKF iteration.
  spline_.resetRefineStats();
  if (opts_.spline.enable && !mg.poses.empty())
  {
    TimedScope ts_fit(profiler_, "lio/spline/fit");
    spline_frame_count_++;
    const SplineImuFitData ifd = splineImuFitData(mg);
    spline_ok_ = spline_.fit(mg.poses, mg.poses.front().t, mg.image.t, opts_.spline, &ifd);
    // Keep the pristine sequence only if something will replay it.
    if (spline_ok_ && opts_.spline.reintegrateOn())
      spline_poses0_ = mg.poses;
    if (!spline_ok_)
    {
      // Never substitute a bad spline for a working deskew.  Count the
      // failures loudly (reported in finalizeSplineAndQ()) rather than
      // degrading silently -- a run where the fit failed on most frames is
      // not the experiment anyone thinks they ran.
      spline_fit_fail_count_++;
    }
  }

  std::vector<PointXYZCov> deskewed;
  if (spline_ok_)
    deskewPointsSpline(state_, spline_, mg.image.t, mg.lidar_points,
                       opts_.deskew, opts_.spline_keep_time_noise, deskewed);
  else
    deskewPoints(state_, mg.poses, mg.image.t, mg.lidar_points, opts_.deskew, deskewed);

  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    // Track which raw points each surviving output point was built from, so
    // the per-iteration re-deskew can re-place exactly this set from their raw
    // coordinates.  Both modes: FIRST gives one member per cell, AVERAGE gives
    // the whole cell and the re-deskew re-averages it.  Output is bit-identical
    // to voxelDownsample() in the matching mode -- see
    // voxelDownsampleIndexedCsr()'s doc comment.
    if (spline_ok_ && opts_.spline.redeskewOn())
      voxelDownsampleIndexedCsr(deskewed, mg.points, ds_offsets_, ds_members_,
                                PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
    else
      voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
    if (spline_ok_ && opts_.spline.redeskewOn()) {
      ds_members_.resize(mg.points.size());
      ds_offsets_.resize(mg.points.size() + 1);
      for (size_t i = 0; i < ds_members_.size(); ++i) {
        ds_members_[i] = static_cast<int>(i);
        ds_offsets_[i] = static_cast<int>(i);
      }
      ds_offsets_.back() = static_cast<int>(ds_members_.size());
    }
  }

  // See LioProcOptions::dry_run_point_filter_num's doc comment. Identical
  // deskew/downsample treatment as the primary points above, just against
  // mg.dry_run_lidar_points -- left empty (no-op) unless the feature is on
  // and this frame actually got a second-decimation scan.
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

SplineImuFitData LioProc::splineImuFitData(const MeasureGroup& mg) const
{
  SplineImuFitData d;
  if (opts_.spline.imuFitOn())
    d.samples = &mg.imu_samples_raw;
  d.bias_acc = state_->biasAcc();
  d.bias_gyr = state_->biasGyr();
  d.gravity  = state_->gravity();
  return d;
}

bool LioProc::redeskewFromSpline(MeasureGroup& mg)
{
  if (!spline_ok_ || !opts_.spline.redeskewOn()) return false;
  if (ds_offsets_.size() < 2) return false;
  if (ds_offsets_.size() != mg.points.size() + 1) return false;

  TimedScope ts(profiler_, "lio/spline/redeskew");

  // (0) SHAPE, from the IMU, under the bias this iteration has arrived at.
  //
  // propagate() ran once, before the loop, with the pre-update bias, and is
  // never re-run -- so without this the spline's shape is frozen there while
  // every iteration moves the bias.  anchorTo() cannot repair it: a bias delta
  // is a shape change (rotation drifting linearly in t, position
  // quadratically) and anchorTo is a rigid transform.  propagate() does not
  // touch the bias or gravity blocks, so state_propagat_ still holds exactly
  // the values the stored poses were built with.
  //
  // The replay always starts from spline_poses0_, never from the previous
  // replay, so the correction is applied once rather than compounded.  Note
  // this moves the spline out from under the LiDAR residuals that step (1)
  // below is about to be linearised at, by the size of the bias delta -- the
  // reason reintegrate_min_db{a,g} exist.
  if (opts_.spline.reintegrateOn() && spline_poses0_.size() >= 2)
  {
    TimedScope ts_ri(profiler_, "lio/spline/reintegrate");
    const V3D dba = state_->biasAcc() - state_propagat_.biasAcc();
    const V3D dbg = state_->biasGyr() - state_propagat_.biasGyr();
    const V3D dg  = state_->gravity() - state_propagat_.gravity();

    if (reintegratePoses(spline_poses0_, dba, dbg, dg, state_propagat_.gravity(),
                         opts_.spline.reintegrate_min_dba,
                         opts_.spline.reintegrate_min_dbg,
                         spline_poses_))
    {
      // Fit into a COPY and adopt only on success: fit() clears valid_ on
      // entry, so a failed re-fit would leave the frame with no spline at all
      // -- strictly worse than the slightly stale one it was replacing.
      const SplineImuFitData ifd = splineImuFitData(mg);
      ScanSpline trial = spline_;
      if (trial.fit(spline_poses_, spline_poses_.front().t, mg.image.t,
                    opts_.spline, &ifd))
      {
        spline_ = trial;
        spline_refits_++;
        // MAGNITUDE, not just the refit count -- see the engagement counters
        // in the header.  How far the replay actually moved the poses the
        // spline is fitted to; a refit that moves them by nanometres is a
        // refit that did nothing.
        const size_t np = std::min(spline_poses0_.size(), spline_poses_.size());
        for (size_t k = 0; k < np; ++k)
        {
          const double dp =
              (spline_poses_[k].pos - spline_poses0_[k].pos).norm();
          const double dr =
              Log(spline_poses0_[k].rot.transpose() * spline_poses_[k].rot).norm()
              * (180.0 / M_PI);
          if (std::isfinite(dp)) reint_dp_max_ = std::max(reint_dp_max_, dp);
          if (std::isfinite(dr)) reint_drot_deg_max_ = std::max(reint_drot_deg_max_, dr);
        }
      }
    }
  }

  // (1) SHAPE, from the map.  Must run BEFORE anchorTo(), which moves every
  // point and would leave these residuals describing a spline that no longer
  // exists.  See SplineOptions::lidar_refine_cp for the division of labour.
  if (opts_.spline.refineOn() && !residuals_.empty())
  {
    lidar_obs_.clear();
    lidar_obs_.reserve(residuals_.size());
    for (const auto& r : residuals_)
    {
      // Inside the fitted window, not "> 0".  The window runs from the first
      // IMU pose to the frame reference time precisely so that every LiDAR
      // return in the scan lands in it (see the fit call above), so this
      // rejects nothing in normal operation -- which is the point: the old
      // t > 0 test was a sentinel check reading a real coordinate.  Frame
      // times are start_time-relative and start_time is the calibration-end
      // image stamp (DataQueues::setStartTime), so a scan straddling the end
      // of calibration has genuinely NEGATIVE point times, and that test
      // silently dropped exactly those points.
      if (!(r.t >= spline_.t0() && r.t <= spline_.t1())) continue;
      SplineLidarObs o;
      o.t      = r.t;
      o.normal = r.normal;
      o.r      = r.r;
      o.sigma2 = r.sigma_squared;
      lidar_obs_.push_back(o);
    }
    spline_.refineWithLidar(lidar_obs_, opts_.spline);
  }

  // (2) POSE, from the IEKF.
  // Carry the IEKF's correction into EVERY control point, not just the
  // scan-end pose.  anchorTo() is a rigid transform of the whole spline
  // that makes its pose at the frame reference time equal the corrected
  // state, so the intra-scan SHAPE the fit found is preserved exactly
  // while the whole trajectory moves with the correction.  This is the
  // difference from the legacy one-shot deskew, where only the scan-end
  // pose is ever corrected and every earlier point keeps the pose the
  // IMU-only propagation gave it.
  spline_.anchorTo(mg.image.t, state_->rot(), state_->pos());

  // How far the re-deskew actually moved the points it re-placed.  This is
  // the only quantity that says whether the per-iteration mechanism is doing
  // anything at all, and nothing recorded it before.
  redeskew_prev_ = mg.points;
  deskewPointsSplineCsr(state_, spline_, mg.image.t, mg.lidar_points,
                        ds_offsets_, ds_members_, opts_.deskew,
                        opts_.spline_keep_time_noise, mg.points);
  ++redeskew_calls_;
  if (redeskew_prev_.size() == mg.points.size() && !mg.points.empty())
  {
    double sum_sq = 0.0;
    for (size_t k = 0; k < mg.points.size(); ++k)
    {
      const double d = (mg.points[k].point - redeskew_prev_[k].point).norm();
      if (std::isfinite(d)) sum_sq += d * d;
    }
    redeskew_dp_rms_ = std::max(
        redeskew_dp_rms_,
        std::sqrt(sum_sq / static_cast<double>(mg.points.size())));
  }
  return true;
}

void LioProc::finalizeSplineAndQ(MeasureGroup& mg)
{
  if (!opts_.spline.enable) return;

  last_spline_stats_ = SplineImuResidualStats{};

  if (spline_ok_)
  {
    // Anchor to the CONVERGED state before measuring.  Measuring against
    // the propagated-only spline would fold this frame's own correction
    // error into a number we are about to call "IMU noise".
    spline_.anchorTo(mg.image.t, state_->rot(), state_->pos());

    if (!mg.imu_samples_raw.empty())
    {
      TimedScope ts(profiler_, "lio/spline/imu_residual");
      last_spline_stats_ = computeSplineImuResidual(
          spline_, mg.imu_samples_raw,
          state_->biasAcc(), state_->biasGyr(), state_->gravity());
    }
  }

  if (opts_.adaptive_q.enable)
  {
    if (!adaptive_q_primed_)
    {
      // The NOMINAL is whatever the state was configured with before any
      // adaptation -- the YAML's state/cov/{acc,gyr} (or the calibration
      // values, if use_calib_var is on).  The FLOOR is the calibration
      // window's measured variance, always available now regardless of
      // use_calib_var.  Isotropic scalars, matching the trace/3 reduction
      // used at every other point in the pipeline.
      adaptive_q_.setNominal(state_->varAcc().mean(), state_->varGyr().mean());
      adaptive_q_.setFloor(state_->varAccFloor().mean(), state_->varGyrFloor().mean());
      adaptive_q_primed_ = true;
    }

    adaptive_q_.update(last_spline_stats_);

    // Push the applied values into the state, where ImuProc::propagate()
    // reads them to build cov_w on the NEXT frame.  One physical quantity,
    // measured as a measurement covariance and applied as process noise --
    // see adaptive_q.h's header for why that is legitimate rather than a
    // category error.  Note this deliberately does NOT touch the floor.
    if (adaptive_q_.active())
      state_->setNoiseParams(V3D::Constant(adaptive_q_.varAcc()),
                             V3D::Constant(adaptive_q_.varGyr()));
  }

  if (opts_.spline.log_en || opts_.adaptive_q.log_en)
  {
    static bool first = true;
    std::ofstream ofs(debugLogPath("spline_q.csv"), first ? std::ios::trunc : std::ios::app);
    if (first)
    {
      ofs << "scan_id," << adaptive_q_.csvHeader()
          << ",spline_ok,n_cp,fit_res_pos,fit_res_rot,fit_reg_frac,rot_chord_deg,"
             "fit_fail_count,frame_count,max_abs_acc,max_abs_gyr,"
             "n_cp_req,rot_mode,refine_applied,refine_rejects,last_refine_step,"
             "refits,per_iteration,refine_dcp_max,refine_dcp_rms,"
             "reint_dp_max,reint_drot_deg_max,redeskew_calls,redeskew_dp_rms\n";
      first = false;
    }
    const double t_abs = mg.image.t + data_queues_->start_time;
    ofs << voxel_map_->frame_idx_ << ','
        << adaptive_q_.csvRow(t_abs, last_spline_stats_) << ','
        << (spline_ok_ ? 1 : 0) << ','
        << (spline_ok_ ? spline_.nControlPoints() : 0) << ','
        << (spline_ok_ ? spline_.fitResidualPos() : 0.0) << ','
        << (spline_ok_ ? spline_.fitResidualRot() : 0.0) << ','
        << (spline_ok_ ? spline_.fitRegFrac() : 0.0) << ','
        << (spline_ok_ ? spline_.rotationChordDeg() : 0.0) << ','
        << spline_fit_fail_count_ << ',' << spline_frame_count_ << ','
        << last_spline_stats_.max_abs_acc << ',' << last_spline_stats_.max_abs_gyr << ','
        << (spline_ok_ ? spline_.nControlPointsRequested() : 0) << ','
        << (spline_ok_ ? (spline_.cumulative() ? "cumulative" : "tangent") : "-") << ','
        << (spline_ok_ ? spline_.refineApplied() : 0) << ','
        << (spline_ok_ ? spline_.refineRejects() : 0) << ','
        << (spline_ok_ ? spline_.lastRefineStep() : 0.0) << ','
        << spline_refits_ << ','
        // The engagement half: what each toggle was set to, and how far its
        // mechanism actually moved things this frame.  A cell with
        // per_iteration naming a step whose magnitude column is 0 across the
        // whole run is an INERT cell, not a null result.
        << opts_.spline.per_iteration << ','
        << (spline_ok_ ? spline_.refineDcpMax() : 0.0) << ','
        << (spline_ok_ ? spline_.refineDcpRms() : 0.0) << ','
        << reint_dp_max_ << ',' << reint_drot_deg_max_ << ','
        << redeskew_calls_ << ',' << redeskew_dp_rms_
        << '\n';
  }

  // Run totals for the end-of-run engagement report.
  ++run_frames_;
  run_redeskew_calls_ += redeskew_calls_;
  run_refits_ += spline_refits_;
  run_redeskew_dp_max_ = std::max(run_redeskew_dp_max_, redeskew_dp_rms_);
  run_reint_dp_max_ = std::max(run_reint_dp_max_, reint_dp_max_);
  if (spline_ok_)
  {
    run_refine_applied_ += spline_.refineApplied();
    run_refine_rejects_ += spline_.refineRejects();
    run_refine_dcp_max_ = std::max(run_refine_dcp_max_, spline_.refineDcpMax());
  }
  if (opts_.adaptive_q.enable && adaptive_q_.active())
  {
    ++run_aq_ok_frames_;
    const double aa = adaptive_q_.varAcc(), ag = adaptive_q_.varGyr();
    if (run_aq_ok_frames_ == 1)
    {
      run_aq_applied_min_acc_ = run_aq_applied_max_acc_ = aa;
      run_aq_applied_min_gyr_ = run_aq_applied_max_gyr_ = ag;
    }
    else
    {
      run_aq_applied_min_acc_ = std::min(run_aq_applied_min_acc_, aa);
      run_aq_applied_max_acc_ = std::max(run_aq_applied_max_acc_, aa);
      run_aq_applied_min_gyr_ = std::min(run_aq_applied_min_gyr_, ag);
      run_aq_applied_max_gyr_ = std::max(run_aq_applied_max_gyr_, ag);
    }
  }
}

// ── Engagement report ──────────────────────────────────────────────────────
// See the declaration in lio_processing.h.  One line per toggle; a flag that
// was ON with a zero counter prints INERT.
LioProc::~LioProc()
{
  if (!opts_.spline.enable) return;
  const std::string rep = engagementReport();
  ROS_WARN_STREAM("\n" << rep);
  std::ofstream ofs(debugLogPath("engagement.txt"), std::ios::trunc);
  if (ofs) ofs << rep << '\n';
}

std::string LioProc::engagementReport() const
{
  std::ostringstream o;
  o << "[engagement]  did each flag actually do anything?  frames="
    << run_frames_;

  auto line = [&](const char* name, bool on, long count, const char* mag_label,
                  double mag) {
    o << "\n  " << name << ": " << (on ? "ON " : "off");
    if (!on) return;
    o << " count=" << count << ' ' << mag_label << '=' << mag;
    if (count == 0)
      o << "   <-- INERT: the flag is on and its mechanism never ran";
    else if (!(mag > 0.0))
      o << "   <-- INERT: the mechanism ran " << count
        << " times and moved nothing";
  };

  o << "\n  spline/per_iteration = " << opts_.spline.per_iteration;
  line("  redeskew          ", opts_.spline.redeskewOn(), run_redeskew_calls_,
       "max_dp_rms_m", run_redeskew_dp_max_);
  line("  refine            ", opts_.spline.refineOn(), run_refine_applied_,
       "max_dcp_m", run_refine_dcp_max_);
  if (opts_.spline.refineOn())
    o << " rejects=" << run_refine_rejects_;
  line("  reintegrate       ", opts_.spline.reintegrateOn(), run_refits_,
       "max_dp_m", run_reint_dp_max_);

  o << "\n  adaptive_q/enable = "
    << (opts_.adaptive_q.enable ? "true" : "false");
  if (opts_.adaptive_q.enable)
  {
    o << "\n    applied frames=" << run_aq_ok_frames_
      << "  var_acc [" << run_aq_applied_min_acc_ << ", "
      << run_aq_applied_max_acc_ << "]"
      << "  var_gyr [" << run_aq_applied_min_gyr_ << ", "
      << run_aq_applied_max_gyr_ << "]";
    if (run_aq_ok_frames_ == 0)
      o << "\n    <-- INERT: adaptive_q never left warmup/gating, so the "
           "process noise was never touched";
    else if (run_aq_applied_max_acc_ <= run_aq_applied_min_acc_ &&
             run_aq_applied_max_gyr_ <= run_aq_applied_min_gyr_)
      o << "\n    <-- INERT: the applied process noise never CHANGED, so "
           "'engaged' here means 'reported healthy', not 'adapted'";
  }

  // The plane model is a VoxelMap-side flag, but its engagement belongs in
  // the same report -- a run that selected "information" and never built one
  // is an inert cell for exactly the same reason a spline flag with a zero
  // counter is, and the scorer only reads this file.
  {
    const long info_fits = voxelPlaneInformationFitCount();
    o << "\n  voxel_map/plane/plane_var_mode = "
      << voxel_map_->opts()->plane_var_mode;
    if (voxel_map_->opts()->plane_var_mode == "information")
    {
      o << " count=" << info_fits;
      if (info_fits == 0)
        o << "   <-- INERT: the information model was selected and no plane "
             "was ever fitted under it";
    }
    o << "\n  voxel_map/plane/weight_floor/mode = "
      << voxel_map_->opts()->weight_floor_mode;
  }

  o << "\n  imu_fit/mode = " << opts_.spline.imu_fit_mode
    << "\n  fit/rotation/mode = " << opts_.spline.rot_mode
    << "\n  control_points/mode = " << opts_.spline.cp_mode;
  o << "\n[engagement] a flag marked INERT above did not test anything; the "
       "cell is invalid, not null.";
  return o.str();
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
      // Re-place every kept point against the spline, re-anchored to
      // whatever the previous iteration corrected the state to.  Skipped on
      // iteration 0 (mg.points is already the freshly-deskewed set) and a
      // no-op unless spline/redeskew_each_iteration is on.  This is the
      // mechanism's point: the residuals the next solve sees are computed
      // from points placed by the CURRENT trajectory estimate, not by the
      // IMU-only propagation that produced the frame.
      if (iter > 0) redeskewFromSpline(mg);

      // T0-D wants the first-iteration (pre-update, un-relinearized)
      // innovation only -- later iterations relinearize at an
      // already-partially-corrected state, which is not the quantity NIS
      // is defined over.
      double error = estimateStateCorrection(mg.points, dtheta, dt, /*allow_consistency_log=*/iter == 0);
      if (!residuals_.empty()) any_solved = true;
      total_dtheta += dtheta;
      total_dt     += dt;

      // History (590-607): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-590
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
        const double pivot_ratio = (n_res > 0) ? ekf_.pivotRatio() : std::numeric_limits<double>::quiet_NaN();
        const double kalman_gain_norm = (n_res > 0) ? ekf_.kalmanGainNorm() : std::numeric_limits<double>::quiet_NaN();
        logConsistencyNll("lio", mg.image.t + data_queues_->start_time, nll, n_res, pivot_ratio, kalman_gain_norm);
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

    // FINAL re-deskew, against the converged state.  redeskewFromSpline()
    // runs at the TOP of each iteration, so without this call the last
    // solve's correction never reaches mg.points -- and the node calls
    // VoxelMap::updateMap() on exactly those points once processLIO()
    // returns.  The map would be built, permanently, from points placed by
    // the second-to-last state.  Also runs one last shape refinement against
    // the final residual set.
    redeskewFromSpline(mg);

    // Measure the IMU against the converged spline and, if enabled, update
    // the applied process noise for the NEXT frame.  Placed after the
    // covariance update so the spline is anchored to the final state --
    // see finalizeSplineAndQ()'s own doc comment for why measuring earlier
    // would contaminate the statistic with this frame's correction error.
    finalizeSplineAndQ(mg);

    // T7-a. dx must be read here and not inside the loop: it is the FRAME's
    // correction, and reading it mid-iteration would report one Gauss-Newton
    // step rather than the process-model error the measurement disagreed with.
    {
      Eigen::MatrixXd phi_p_phit, accum_cov_w;
      if (imuProcQhatRead(phi_p_phit, accum_cov_w)) {
        const Eigen::VectorXd dx = state_->boxminusFromPropagat(state_propagat_);
        const Eigen::MatrixXd& P_post = state_->cov();
        if (phi_p_phit.rows() == P_post.rows()) {
          const double t_abs = mg.image.t + data_queues_->start_time;
          debugLogQhat(voxel_map_->frame_idx_, t_abs, dx, (phi_p_phit - P_post).diagonal(),
                       accum_cov_w.diagonal());
        }
      }
    }

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
      logConsistencyScan("lio", voxel_map_->frame_idx_, t_abs, dt, trP_pos, trP_vel, trP_att,
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
      // History (774-785): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-774
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

      // History (837-854): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-837
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
