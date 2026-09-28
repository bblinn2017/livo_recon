#include "livo_recon/processing/lio_base.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/voxelplane.h"   // voxelPlaneInformationFitCount(), flushVarianceShareLog/flushConsistencyCorrLog
#include "livo_recon/lio/lio_accumulator.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace livo_recon
{

namespace
{

// CQ-46: per-residual pair-correlation diagnostic. Mirrors voxelplane.cpp's
// t_corr_buf/g_corr_with_covariates idiom exactly (thread_local append
// buffer, one flush per thread after the OMP region's implicit barrier) for
// the same reason: writing per-candidate, not just per-accepted-residual,
// through a shared ofstream would serialize buildResiduals()'s
// embarrassingly-parallel loop. Gated behind log_pair_corr_en (default
// false) AND allow_consistency_log (true only on a scan's first IEKF
// iteration) -- neither condition is met by any shipped config, so this is
// dead weight, not a live cost, at every default.
thread_local std::string t_pair_corr_buf;

void appendPairCorrResidual(int scan_id, const Residual& res)
{
  std::ostringstream oss;
  oss << std::setprecision(9)
      << scan_id << ','
      << res.world_point.x() << ',' << res.world_point.y() << ',' << res.world_point.z() << ','
      << res.normal.x() << ',' << res.normal.y() << ',' << res.normal.z() << ','
      << res.point_cross_normal.x() << ',' << res.point_cross_normal.y() << ',' << res.point_cross_normal.z() << ','
      << reinterpret_cast<uint64_t>(res.plane_id) << ','
      << res.r << ',' << res.floor_term << ',' << res.sigma_diag_squared << ','
      << res.plane_var_term << ',' << res.s_prior_pose << '\n';
  t_pair_corr_buf += oss.str();
}

void flushPairCorrLog()
{
  if (t_pair_corr_buf.empty()) return;
  static PersistentLogStream log("cq46_residuals.txt");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call)
    ofs << "scan_id,px,py,pz,nx,ny,nz,jx,jy,jz,plane_id,r,floor_term,sigma_diag_squared,plane_var_term,s_prior_pose\n";
  ofs << t_pair_corr_buf;
  ofs.flush();
  t_pair_corr_buf.clear();
}

void logPairCorrPrior(int scan_id, const Eigen::Matrix<double, 6, 6>& prior_cov_rp)
{
  static PersistentLogStream log("cq46_prior.txt");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call) {
    ofs << "scan_id";
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        ofs << ",p" << i << j;
    ofs << "\n";
  }
  ofs << scan_id << std::setprecision(9);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      ofs << ',' << prior_cov_rp(i, j);
  ofs << "\n";
  ofs.flush();
}



void writeFirstFrameSolveMatrix(std::ofstream& ofs, const char* name, const Eigen::MatrixXd& M)
{
  ofs << "matrix " << name << " rows=" << M.rows() << " cols=" << M.cols() << "\n";
  ofs << std::setprecision(17);
  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      if (c) ofs << ' ';
      ofs << M(r, c);
    }
    ofs << '\n';
  }
}

void writeFirstFrameSolveVector(std::ofstream& ofs, const char* name, const Eigen::VectorXd& v)
{
  ofs << "vector " << name << " size=" << v.size() << "\n";
  ofs << std::setprecision(17);
  for (int i = 0; i < v.size(); ++i) {
    if (i) ofs << ' ';
    ofs << v(i);
  }
  ofs << '\n';
}

void logFirstFrameDecoupledSolve(
    int frame_idx, int scan_id, int iteration, double t_abs,
    const char* architecture, const std::vector<PointXYZCov>& pts,
    const std::vector<Residual>& residuals, const StateGroup& before,
    const StateGroup& after, const StateGroup& propagat,
    const Eigen::MatrixXd& prior_cov, const EkfUpdate& ekf,
    const Eigen::Matrix<double,6,6>& HtH,
    const Eigen::Matrix<double,6,1>& Htz)
{
  if (frame_idx != 1) return;
  static PersistentLogStream csv_log("pose_control_first_frame_solve.csv");
  bool first = false;
  std::ofstream& csv = csv_log.stream(&first);
  if (first) csv << "architecture,frame_idx,scan_id,iteration,t_abs,n_points,n_residuals,"
                     "delta_px,delta_py,delta_pz,delta_pnorm,delta_rx,delta_ry,delta_rz,delta_rnorm,"
                     "HtH_trace,Htz_norm,prior_cov_trace,A_trace,K1_norm,solution_norm\n";
  const V3D dp = after.pos() - before.pos();
  const V3D dtheta = Log(before.rot().transpose() * after.rot());
  const Eigen::VectorXd& solution = ekf.lastSolution();
  csv << std::setprecision(17)
      << architecture << ',' << frame_idx << ',' << scan_id << ',' << iteration << ',' << t_abs << ','
      << pts.size() << ',' << residuals.size() << ','
      << dp.x() << ',' << dp.y() << ',' << dp.z() << ',' << dp.norm() << ','
      << dtheta.x() << ',' << dtheta.y() << ',' << dtheta.z() << ',' << dtheta.norm() << ','
      << HtH.trace() << ',' << Htz.norm() << ',' << prior_cov.trace() << ','
      << (ekf.lastA().size() ? ekf.lastA().trace() : 0.0) << ','
      << (ekf.lastK1().size() ? ekf.lastK1().norm() : 0.0) << ','
      << (solution.size() ? solution.norm() : 0.0) << '\n';
  csv.flush();
  static PersistentLogStream dump_log("pose_control_first_frame_solve_matrices.txt");
  std::ofstream& dump = dump_log.stream();
  dump << "=== decoupled_solve_snapshot ===\n"
       << "architecture=" << architecture << "\n"
       << "frame_idx=" << frame_idx << "\n"
       << "scan_id=" << scan_id << "\n"
       << "iteration=" << iteration << "\n"
       << "t_abs=" << std::setprecision(17) << t_abs << "\n"
       << "n_points=" << pts.size() << "\n"
       << "n_residuals=" << residuals.size() << "\n"
       << "state_before_pos=" << before.pos().transpose() << "\n"
       << "state_after_pos=" << after.pos().transpose() << "\n"
       << "delta_pos=" << dp.transpose() << "\n"
       << "delta_rot=" << dtheta.transpose() << "\n"
       << "propagated_pos=" << propagat.pos().transpose() << "\n"
       << "propagated_rot_log=" << Log(propagat.rot()).transpose() << "\n";
  // FIX (found this round): the supplied patch replaced this diagnostic's
  // write of `vec` with the exact production value (ekf.lastVec()), per its
  // own explicit rationale that a hand-recomputed vec here (previously
  // `before.boxminusFromPropagat(propagat)`, using the pre-iteration
  // snapshot `before` rather than production's own live `state` pointer at
  // the actual solve point) "could use a different state... and would
  // defeat the audit" -- but left the now-dead local declaration and its
  // redundant boxminus computation in place. Removed as dead code, exactly
  // the kind of duplicate-computation this patch exists to eliminate.
  writeFirstFrameSolveMatrix(dump, "prior_cov", prior_cov);
  writeFirstFrameSolveMatrix(dump, "H_full", ekf.lastHFull());
  writeFirstFrameSolveMatrix(dump, "A_exact", ekf.lastA());
  writeFirstFrameSolveMatrix(dump, "K1_exact", ekf.lastK1());
  writeFirstFrameSolveMatrix(dump, "HtH", Eigen::MatrixXd(HtH));
  writeFirstFrameSolveVector(dump, "Htz", Eigen::VectorXd(Htz));
  writeFirstFrameSolveVector(dump, "prior_boxminus_propagated", ekf.lastVec());
  writeFirstFrameSolveVector(dump, "prior_boxminus_propagated_rp", ekf.lastVecRp());
  writeFirstFrameSolveMatrix(dump, "G_exact", ekf.lastG());
  writeFirstFrameSolveVector(dump, "measurement_term_exact", ekf.lastMeasurementTerm());
  writeFirstFrameSolveVector(dump, "prior_term_exact", ekf.lastPriorTerm());
  writeFirstFrameSolveVector(dump, "solution_exact", solution);
  for (size_t i = 0; i < residuals.size(); ++i) {
    const auto& r = residuals[i];
    dump << "residual " << i << " t=" << r.t << " r=" << r.r
         << " sigma_squared=" << r.sigma_squared << " plane_var_term=" << r.plane_var_term
         << " plane_id=" << reinterpret_cast<uintptr_t>(r.plane_id)
         << " h_rot=" << r.point_cross_normal.transpose() << " h_pos=" << r.normal.transpose()
         << " world_point=" << r.world_point.transpose()
         << " raw_body_point=" << r.raw_body_point.transpose() << '\n';
  }
  dump << "=== end_decoupled_solve_snapshot ===\n";
  dump.flush();
}

}  // namespace

LioProcBase::LioProcBase(NodeContext& ctx)
  : state_(ctx.state), voxel_map_(ctx.voxel_map), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

// CQ-49 item 1: extracted ONCE from the pre-split LioProc::loadParameters(),
// called from both derived classes' own loadParameters() overrides against
// their OWN ConfigResolver -- see each override for the estimator-specific
// keys (spline/adaptive_q for decoupled, estimator/coupled/* for coupled)
// this does NOT read. Order relative to the original function is changed
// (this reads its keys before the derived override reads its own), which is
// safe: item 0's byte-identical gate is on odometry.txt, and the
// [config/effective] echo's key ORDER is explicitly excluded from that
// check (CQ-49 item 3f(iv)).
void LioProcBase::loadSharedParameters(ConfigResolver& cfg, ros::NodeHandle& pnh)
{
  paramWarn<int>(pnh, "lio/ekf/max_iterations",     opts_.max_iterations,  5);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dtheta", opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dt",     opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "lio/ekf/min_diff_error",  opts_.min_diff_error,  -1.0);
  paramWarn<bool>(pnh, "lio/log_debug_en",          opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "lio/log_pair_corr_en",      opts_.log_pair_corr_en, false);
  paramWarn<bool>(pnh, "lio/log_eigenspectrum_en",  opts_.log_eigenspectrum_en, false);
  paramWarn<bool>(pnh, "lio/log_consistency_scan_en", opts_.log_consistency_scan_en, false);
  paramWarn<bool>(pnh, "lio/log_nll_en", opts_.log_nll_en, false);
  paramWarn<int>(pnh, "lio/dry_run_point_filter_num", opts_.dry_run_point_filter_num, 0);
  paramWarn<bool>(pnh, "eval/nees_per_dof_en", opts_.nees_per_dof_en, false);
  paramWarn<int>(pnh, "eval/nees_tier1_window_scans", opts_.nees_tier1_window_scans, 489);
  paramWarn<bool>(pnh, "cuda/enable",               cuda_enable_,          false);

  // History (134-136): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-134
  double range_err;
  paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
  opts_.deskew.sigma_r2 = range_err * range_err;
  double angle_err_deg;
  paramWarn<double>(pnh, "imu/sensor/angle_err_deg", angle_err_deg, 0.2);
  const double sin_angle_err = std::sin(std::max(1e-6, angle_err_deg * M_PI / 180.0));
  opts_.deskew.sigma_a2 = sin_angle_err * sin_angle_err;

  // Independent control or exact per-VoxelPlane covariance marginalization.
  cfg.mode("lio/residual_redundancy/mode", opts_.residual_redundancy.mode, "off",
           { "off", "woodbury" });
  const bool rr = opts_.residual_redundancy.mode != "off";
  cfg.nested<double>(rr, "lio/residual_redundancy/mode!=off", "lio/residual_redundancy/rho",
                     opts_.residual_redundancy.rho, 1.0);
  cfg.nested<double>(rr, "lio/residual_redundancy/mode!=off", "lio/residual_redundancy/max_discount",
                     opts_.residual_redundancy.max_discount, 0.9);

  // CQ-31 item 5: three independently-switchable scalar P controls, all
  // default-identity.
  cfg.get<double>("lio/p_inflate/alpha", opts_.prior_scalar.p_inflate_alpha, 1.0);
  cfg.get<double>("lio/p_floor/min_eig", opts_.prior_scalar.p_floor_min_eig, 0.0);
  cfg.get<double>("lio/p_fading/lambda", opts_.prior_scalar.p_fading_lambda, 1.0);

  // Downsampling is one axis with three states, not a mode plus a magic
  // zero.
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

}

// CQ-49: the shared finish -- refuse anything left unclaimed in the given
// namespaces (each derived class passes its OWN set: {"spline","adaptive_q"}
// for decoupled, {"estimator"} for coupled), record the accumulation
// precision, and either refuse (throw) or return the effective config.
std::string LioProcBase::finalizeConfig(ConfigResolver& cfg, std::initializer_list<const char*> allowed_unclaimed_namespaces)
{
  cfg.refuseUnclaimed(allowed_unclaimed_namespaces);

  // The build's accumulation precision, recorded rather than assumed. The
  // CUDA path returns float-precision HtH/Htz, so nll.txt is NOT comparable
  // between a CPU and a GPU build.
  cfg.derived("build/accumulation_precision",
              cuda_enable_ ? "float (CUDA HtH/Htz) -- nll.txt NOT comparable "
                             "with a CPU build"
                           : "double (CPU HtH/Htz)");

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
void LioProcBase::ensureStationaryReference(const MeasureGroup& mg)
{
  if (stationary_reference_valid_) return;
  stationary_reference_R_ = mg.rot_before_imu;
  stationary_reference_p_ = mg.pos_before_imu;
  stationary_reference_v_ = mg.vel_before_imu;
  stationary_reference_valid_ = true;
}

void LioProcBase::buildResiduals(
  const std::vector<PointXYZCov>& pts,
  std::vector<Residual>& residuals,
  bool allow_consistency_log,
  bool include_state_uncertainty_in_gate,
  const std::vector<M3D>* pose_cov_overrides) const {
  // Single-threaded, before the OMP region starts below -- see
  // VoxelMap::setAllowConsistencyLog()'s doc comment.
  if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get()))
    vm->setAllowConsistencyLog(allow_consistency_log);

  const int n = (int)pts.size();
  const int threads = cappedOmpThreads();

  build_thread_residuals_.resize(threads);
  for (auto& v : build_thread_residuals_) v.clear();
  build_thread_miss_.assign(threads, {0, 0});
  build_thread_gate_audit_.assign(threads, {0, 0});
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

  // CQ-35: an explicit `omp parallel` region wrapping an `omp for`, rather
  // than a combined `omp parallel for`, so that flushVarianceShareLog() can
  // be called once per thread AFTER the implicit barrier at the end of
  // `omp for` -- every thread is guaranteed to have finished its share of
  // the loop (and therefore finished appending to its own thread_local
  // buffer) before any thread reaches the flush call, and it is still the
  // SAME team of threads that did the logging, so draining "this thread's
  // buffer" here is well-defined (see voxelplane.h's doc comment on why
  // this must be called from inside the same parallel region).
  #pragma omp parallel num_threads(threads)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i)
    {
      const PointXYZCov sensor_world = state_->toWorld(pts[i]);
      M3D pose_cov = M3D::Zero();
      if (include_state_uncertainty_in_gate) {
        if (pose_cov_overrides && i < static_cast<int>(pose_cov_overrides->size()))
          pose_cov = (*pose_cov_overrides)[i];
        else
          pose_cov = state_->poseCovAt(pts[i].point);
      }
      WorldPointCov pt_world{sensor_world.point, sensor_world.sensor_cov, pose_cov};
      pt_world.include_pose_cov_in_gate = include_state_uncertainty_in_gate;
      pt_world.body_point = pts[i].point;
      pt_world.rot_transpose = rot_transpose;
      pt_world.prior_cov_rp = prior_cov_rp;
      GateAudit gate_audit;
      pt_world.gate_audit = &gate_audit;
      Residual res{};
      bool tier0_had_plane = false;
      bool tier0_missed = true;
      // had_converged_neighbor: was a duplicate hasConvergedNeighbor(pt_world.point)
      // call on the else branch below (a second full neighborhood_size box
      // scan of the SAME box findPlaneResidual()'s own tier1/tier2 fallback
      // just scanned to look for a match) -- findPlaneResidual() now
      // accumulates the identical answer for free from cells it already
      // visits, via this out-param, instead of scanning the box twice on
      // every total miss.
      bool had_converged_neighbor = false;
      const bool admitted = voxel_map_->findPlaneResidual(
          pt_world, res, &tier0_had_plane, &had_converged_neighbor);
      if (admitted) {
        res.source_index = i;
        res.point_cross_normal = pts[i].point.cross(state_->rot().transpose() * res.normal);
        res.world_point = pt_world.point;
        res.sigma_squared += res.plane_var_term;
        res.t = pts[i].t;   // for the spline control-point refinement
        res.raw_body_point = pts[i].raw_body_point;  // CQ-50
        build_thread_residuals_[omp_get_thread_num()].push_back(res);
        if (allow_consistency_log && opts_.log_pair_corr_en)
          appendPairCorrResidual(voxel_map_->frame_idx_, res);
        tier0_missed = (res.match_tier != 0);
      } else {
        const int idx = had_converged_neighbor ? 1 : 0;
        ++build_thread_miss_[omp_get_thread_num()][idx];
      }
      auto& gate_counts = build_thread_gate_audit_[omp_get_thread_num()];
      if (gate_audit.reached_statistical_gate) ++gate_counts[0];
      if (!admitted && gate_audit.rejected_by_statistical_gate) ++gate_counts[1];
      if (tier0_missed) {
        const int idx0 = tier0_had_plane ? 1 : 0;
        ++build_thread_tier0_miss_[omp_get_thread_num()][idx0];
      }
    }
    // Implicit barrier at the end of `omp for` above already happened --
    // every thread's residual-building work (and its diagnostic logging,
    // gated behind log_variance_shares_en) is done by the time any thread
    // reaches here.
    flushVarianceShareLog();
    // CQ-38: same call-site reasoning as flushVarianceShareLog() above --
    // debugLogConsistencyCorr() (called from computeResidual(), same OMP
    // region) now buffers per-thread too. See voxelplane.h's doc comment.
    flushConsistencyCorrLog();
    // CQ-60: same once-per-thread obligation as flushVarianceShareLog()/
    // flushConsistencyCorrLog() above -- merges this thread's t_corr_scan
    // into the shared g_corr_scan, no lock held longer than one struct-add.
    mergeCorrScanThreadLocal();
    if (allow_consistency_log && opts_.log_pair_corr_en)
      flushPairCorrLog();
  }

  if (allow_consistency_log && opts_.log_pair_corr_en)
    logPairCorrPrior(voxel_map_->frame_idx_, prior_cov_rp);
  // CQ-60: single-threaded, after the parallel region above has fully
  // closed -- every thread's contribution is merged into g_corr_scan by
  // now (mergeCorrScanThreadLocal() ran for all of them, inside the
  // region, before its implicit join). One row per buildResiduals() call.
  flushCorrScanRow();

  residuals.clear();
  for (const auto& local : build_thread_residuals_)
    residuals.insert(residuals.end(), local.begin(), local.end());

  n_miss_coverage_ = 0;
  n_miss_mismatch_ = 0;
  for (const auto& m : build_thread_miss_) {
    n_miss_coverage_ += m[0];
    n_miss_mismatch_ += m[1];
  }
  n_statistical_gate_candidates_ = 0;
  n_statistical_gate_rejections_ = 0;
  for (const auto& counts : build_thread_gate_audit_) {
    n_statistical_gate_candidates_ += counts[0];
    n_statistical_gate_rejections_ += counts[1];
  }

  n_tier0_miss_coverage_ = 0;
  n_tier0_miss_mismatch_ = 0;
  for (const auto& m : build_thread_tier0_miss_) {
    n_tier0_miss_coverage_ += m[0];
    n_tier0_miss_mismatch_ += m[1];
  }
}

// History (242-246): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-242
void LioProcBase::solveSystem_cuda(const std::vector<Residual>& residuals) const {
  accumulateLioResidualsCuda(residuals, ekf_, cuda_buf_);
  // CQ-34: called UNCONDITIONALLY -- residual_redundancy.h's own header
  // already promises this is safe at mode=="off" (the module's internal
  // gate on ekf.HtH/Htz mutation is what keeps "off" byte-identical, not
  // this call site).
  { TimedScope ts(profiler_, "lio/ekf/redundancy");
    redundancy_stats_ = applyResidualRedundancyCorrection(residuals, opts_.residual_redundancy, ekf_); }
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
}

void LioProcBase::solveSystem(const std::vector<Residual>& residuals) const {
  accumulateLioResiduals(residuals, ekf_);

  // CQ-34: see solveSystem_cuda()'s comment above -- same fix, same reason.
  { TimedScope ts(profiler_, "lio/ekf/redundancy");
    redundancy_stats_ = applyResidualRedundancyCorrection(residuals, opts_.residual_redundancy, ekf_); }

  // Mean-only update against the frame's FIXED prior (prior_cov_/
  // state_propagat_, snapshotted once in processLIO() before this frame's
  // iteration loop began) -- does not touch state_->cov().
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
}

double LioProcBase::estimateStateCorrection(
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

  double avg_res = 0.0;
  for (const auto& r : residuals_)
    avg_res += std::abs(r.r);
  avg_res /= residuals_.size();

  const V3D p_before_update = state_->pos();
  const V3D v_before_update = state_->vel();
  const M3D R_before_update = state_->rot();
  StateGroup state_before_update;
  state_before_update = *state_;
  {
    TimedScope ts(profiler_, "lio/ekf/solve");
    if (cuda_enable_)
      solveSystem_cuda(residuals_);
    else
      solveSystem(residuals_);
    dtheta = ekf_.dtheta;
    dt     = ekf_.dt;
  }

  if (voxel_map_->frame_idx_ == 1 && diagnostic_gn_iteration_ >= 0) {
    const double t_abs = pts.empty() ? data_queues_->start_time : pts.front().t + data_queues_->start_time;
    logFirstFrameDecoupledSolve(
        voxel_map_->frame_idx_, 0, diagnostic_gn_iteration_, t_abs,
        firstFrameArchitectureName(), pts, residuals_, state_before_update, *state_,
        state_propagat_, prior_cov_, ekf_, ekf_.HtH, ekf_.Htz);
  }

  if (opts_.log_debug_en && diagnostic_gn_iteration_ >= 0) {
    static PersistentLogStream log("decoupled_factor_iteration.csv");
    bool first_call;
    std::ofstream& ofs = log.stream(&first_call);
    if (first_call) {
      ofs << "scan_id,iter,timestamp,p_before_x,p_before_y,p_before_z,p_after_x,p_after_y,p_after_z,"
             "v_before_x,v_before_y,v_before_z,v_after_x,v_after_y,v_after_z,dp_x,dp_y,dp_z,dp_norm,"
             "dtheta_x,dtheta_y,dtheta_z,dtheta_norm,rank,lambda_min,lambda_max";
      for (int i = 0; i < 36; ++i) ofs << ",A" << i;
      for (int i = 0; i < 6; ++i) ofs << ",b" << i;
      ofs << "\n";
    }
    const M3D dR = R_before_update.transpose() * state_->rot();
    Eigen::AngleAxisd aa(dR);
    const V3D dtheta_phys = aa.angle() * aa.axis();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(
        0.5 * (ekf_.HtH + ekf_.HtH.transpose()));
    int rank = 0;
    double lmin = 0.0, lmax = 0.0;
    if (es.info() == Eigen::Success && es.eigenvalues().size() > 0) {
      lmin = es.eigenvalues()(0);
      lmax = es.eigenvalues()(5);
      const double thresh = 1e-12 * std::max(1.0, std::abs(lmax));
      for (int i = 0; i < 6; ++i) if (es.eigenvalues()(i) > thresh) ++rank;
    }
    ofs << std::setprecision(12)
        << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ','
        << (pts.empty() ? 0.0 : pts.front().t) << ','
        << p_before_update.x() << ',' << p_before_update.y() << ',' << p_before_update.z() << ','
        << state_->pos().x() << ',' << state_->pos().y() << ',' << state_->pos().z() << ','
        << v_before_update.x() << ',' << v_before_update.y() << ',' << v_before_update.z() << ','
        << state_->vel().x() << ',' << state_->vel().y() << ',' << state_->vel().z() << ','
        << (state_->pos()-p_before_update).x() << ',' << (state_->pos()-p_before_update).y() << ','
        << (state_->pos()-p_before_update).z() << ',' << (state_->pos()-p_before_update).norm() << ','
        << dtheta_phys.x() << ',' << dtheta_phys.y() << ',' << dtheta_phys.z() << ','
        << dtheta_phys.norm() << ',' << rank << ',' << lmin << ',' << lmax;
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) ofs << ',' << ekf_.HtH(r,c);
    for (int i = 0; i < 6; ++i) ofs << ',' << ekf_.Htz(i);
    ofs << "\n";
  }

  return avg_res;
}

bool LioProcBase::accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res)
{
  if (voxel_map_->isEmpty()) return false;

  {
    TimedScope ts(profiler_, "lio/ekf/build_residuals");
    buildResiduals(mg.points, residuals_);
  }
  if (residuals_.empty()) return false;

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

void LioProcBase::logEigenspectrum18(int scan_id, double t_abs, const char* mode_label) const
{
  if (!opts_.log_eigenspectrum_en) return;
  if (state_->idxBG() < 0 || state_->idxBA() < 0 || state_->idxG() < 0) return;
  const Eigen::MatrixXd& P = state_->cov();
  if (P.rows() < 18 || P.cols() < 18) return;

  const Eigen::Matrix<double, 18, 18> P18 = P.block<18, 18>(0, 0);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 18, 18>> es(P18);
  const auto& eigvals = es.eigenvalues();     // ascending
  const auto& eigvecs = es.eigenvectors();

  // CQ-74 item 5a: which state block (rot/pos/vel/bg/ba/gravity) the
  // SMALLEST eigenvector loads onto, by largest-component -- StateGroup's
  // own fixed 18-dim layout (idxR=0,idxP=3,idxV=6,idxBG=9,idxBA=12,idxG=15,
  // each a contiguous 3-block).
  static const char* BLOCK_NAMES[6] = {"rot", "pos", "vel", "bg", "ba", "gravity"};
  const Eigen::VectorXd v_min = eigvecs.col(0);
  int max_block = 0;
  double max_block_norm = -1.0;
  for (int b = 0; b < 6; ++b) {
    const double block_norm = v_min.segment<3>(3 * b).squaredNorm();
    if (block_norm > max_block_norm) { max_block_norm = block_norm; max_block = b; }
  }

  static PersistentLogStream log("cq74_eigenspectrum.txt");
  bool first;
  std::ofstream& ofs = log.stream(&first);
  if (first) {
    ofs << "mode,scan_id,t_abs,lambda_min,lambda_max,weak_axis_block";
    for (int i = 0; i < 18; ++i) ofs << ",eig" << i;
    ofs << "\n";
  }
  ofs << mode_label << "," << scan_id << "," << std::fixed << std::setprecision(6) << t_abs
      << std::defaultfloat << "," << eigvals(0) << "," << eigvals(17) << ","
      << BLOCK_NAMES[max_block];
  for (int i = 0; i < 18; ++i) ofs << "," << eigvals(i);
  ofs << "\n";
  ofs.flush();
}

}  // namespace livo_recon
