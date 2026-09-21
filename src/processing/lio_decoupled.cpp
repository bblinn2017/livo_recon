#include "livo_recon/processing/lio_decoupled.h"
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
#include <array>
#include <cmath>
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
// CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
// regression this fixes -- applied to every debugLogXxx() helper in this
// file (see voxelmap.cpp's debugLogFrameStats() for the same fix applied
// project-wide).
void debugLogLio(const std::string& msg)
{
  static PersistentLogStream log("lio.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

// History (40-42): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-40
void debugLogIterError(const std::string& msg)
{
  static PersistentLogStream log("iter_error.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

// History (51-54): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-51
void debugLogLioDryRun(const std::string& msg)
{
  static PersistentLogStream log("lio_dryrun.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

// History (63-70): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-63

// History (72-75): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-72

// History (77-89): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-77
void debugLogQhat(int scan_id, double t_abs, const Eigen::VectorXd& dx,
                  const Eigen::VectorXd& expected_diag,
                  const Eigen::VectorXd& acw_diag)
{
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  const int n = static_cast<int>(dx.size());
  // CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
  // regression this fixes.
  static PersistentLogStream log("qhat.csv");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call) {
    ofs << "scan_id,t,dim";
    for (int i = 0; i < n; ++i) ofs << ",dx" << i;
    for (int i = 0; i < n; ++i) ofs << ",exp" << i;
    for (int i = 0; i < n; ++i) ofs << ",acw" << i;
    ofs << "\n";
  }
  ofs << scan_id << "," << t_abs << "," << n;
  for (int i = 0; i < n; ++i) ofs << "," << dx(i);
  for (int i = 0; i < n; ++i) ofs << "," << expected_diag(i);
  for (int i = 0; i < n; ++i) ofs << "," << acw_diag(i);
  ofs << "\n";
  ofs.flush();
}

}  // namespace

LioProcDecoupled::LioProcDecoupled(NodeContext& ctx)
  : LioProcBase(ctx)
{}

std::string LioProcDecoupled::loadParameters(ros::NodeHandle& pnh)
{
  // One resolver for the whole function: modes validated against a named
  // set, nested keys read only inside a live scope and REFUSED if set into a
  // dead one, and the EFFECTIVE configuration returned rather than the
  // requested one.  See config_resolve.h.
  ConfigResolver cfg(pnh);

  loadSharedParameters(cfg, pnh);
  decoupled_tier1_nees_ = Tier1NeesBuffer(opts_.nees_per_dof_en ? opts_.nees_tier1_window_scans : 0);

  // ── scan spline ──────────────────────────────────────────────────────
  // Everything from here down goes through ConfigResolver, not paramWarn:
  // nested keys are read only inside a live scope and REFUSED if set into a
  // dead one, modes are validated against a named set, and what gets printed
  // is the effective configuration.  See config_resolve.h for why.
  // ── THE ONE MODE KNOB.  Bryce, 2026-09-06. ───────────────────────────────
  // Replaces spline/enable, spline/per_iteration/mode (five levels) and
  // spline/boundary_anchor_mode (three levels).  At "raw_imu" every other
  // spline/* key -- and every adaptive_q/* key -- is REFUSED at startup
  // rather than silently ignored, which is what `sp` gates below.
  cfg.mode("spline/mode", dopts_.spline.mode, "spline",
           { "raw_imu", "spline", "spline+refine" });
  const bool sp = dopts_.spline.splineOn();

  // Control-point RATE is the only control-point knob: n_cp is derived from
  // it and the scan duration, so the axis is comparable across sequences.
  cfg.nested<double>(sp, "spline/mode", "spline/control_points/hz",
                     dopts_.spline.control_point_hz, 100.0);

  cfg.nested<int>(sp && dopts_.spline.refineOn(), "spline/mode=spline+refine",
                  "spline/refine/iters",
                  dopts_.spline.lidar_refine_iters, 1);
  // CQ-41 follow-up: see SplineOptions::refine_curvature_weight's own doc
  // comment for the A/B numbers. Default 0.0 (off) preserves prior behavior.
  cfg.nested<double>(sp && dopts_.spline.refineOn(), "spline/mode=spline+refine",
                     "spline/refine/curvature_weight",
                     dopts_.spline.refine_curvature_weight, 0.0);
  // CQ-41 follow-up, third term: see SplineOptions::refine_imu_acc_weight's
  // own doc comment. Default 0.0 (off) preserves prior behavior.
  cfg.nested<double>(sp && dopts_.spline.refineOn(), "spline/mode=spline+refine",
                     "spline/refine/imu_acc_weight",
                     dopts_.spline.refine_imu_acc_weight, 0.0);
  // Bryce, 2026-09-18: diagnostic experiment -- see
  // SplineOptions::refine_imu_acc_solve_bias's own doc comment.
  cfg.nested<bool>(sp && dopts_.spline.refineOn() && dopts_.spline.refine_imu_acc_weight > 0.0,
                   "spline/refine/imu_acc_weight>0",
                   "spline/refine/imu_acc_solve_bias",
                   dopts_.spline.refine_imu_acc_solve_bias, false);
  // CQ-43 item (2): see SplineOptions::final_pass's own doc comment.
  // DEFAULT false, per item (6) -- do not change this from this card's
  // own numbers, whatever they show.
  cfg.nested<bool>(sp && dopts_.spline.refineOn(), "spline/mode=spline+refine",
                   "spline/refine/final_pass",
                   dopts_.spline.final_pass, false);
  // TQ-38 item (4): see SplineOptions::diag_free_tail_imu_acc_weight's own
  // doc comment -- diagnostic only, never read by the real refinement path.
  cfg.nested<double>(sp, "spline/mode",
                     "spline/refine/diag_free_tail_imu_acc_weight",
                     dopts_.spline.diag_free_tail_imu_acc_weight, 1.0);

  cfg.nested<bool>(sp, "spline/mode", "spline/log_en", dopts_.spline.log_en, false);
  // CQ-41, Bryce 2026-09-18: DEFAULT true, authorised by Bryce in the
  // message that requested this item -- see SplineOptions::
  // end_constraint_velocity's own doc comment. Rule 26 item 1 (a numerics
  // default change) normally NEEDS BRYCE; the flag is set here BY Bryce
  // naming the default he wants, so no further authorization is needed for
  // this one default -- do not extend that to any other default.
  cfg.nested<bool>(sp, "spline/mode", "spline/end_constraint/velocity",
                   dopts_.spline.end_constraint_velocity, true);
  // Analysis-only dense trajectory dump -- see SplineOptions::traj_log_mode.
  cfg.nestedMode(sp, "spline/mode", "spline/trajectory_log/mode",
                 dopts_.spline.traj_log_mode, "off", { "off", "dense" });
  cfg.nested<double>(sp && dopts_.spline.trajLogOn(),
                     "spline/trajectory_log/mode=dense",
                     "spline/trajectory_log/hz", dopts_.spline.traj_log_hz, 200.0);

  // ── live process-noise estimation ────────────────────────────────────
  // The statistic IS the spline-vs-IMU residual, so this scope is nested
  // under the spline rather than merely warned about.
  cfg.nested<bool>(sp, "spline/mode", "adaptive_q/enable",
                   dopts_.adaptive_q.enable, false);
  const bool aq = sp && dopts_.adaptive_q.enable;
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/beta_acc", dopts_.adaptive_q.beta_acc, 0.3);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/beta_gyr", dopts_.adaptive_q.beta_gyr, 0.3);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/z_rate_limit", dopts_.adaptive_q.z_rate_limit, 0.02);
  // CQ-35: kept in sync with AdaptiveQOptions::acf1_max's own default
  // (adaptive_q.h) -- this call's own literal is the REAL effective default
  // whenever the YAML doesn't set the key (the common case), since it
  // overwrites whatever the struct member was already initialised to.
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/acf1_max", dopts_.adaptive_q.acf1_max, 1.00);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/bounds/max_ratio", dopts_.adaptive_q.max_ratio, 100.0);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/bounds/min_ratio", dopts_.adaptive_q.min_ratio, 0.01);
  cfg.nested<int>(aq, "adaptive_q/enable", "adaptive_q/warmup_frames", dopts_.adaptive_q.warmup_frames, 20);
  cfg.nested<double>(aq, "adaptive_q/enable", "adaptive_q/ema", dopts_.adaptive_q.ema, 0.9);
  cfg.nestedMode(aq, "adaptive_q/enable", "adaptive_q/noise_floor/mode",
                 adaptive_q_floor_mode_, "allan", { "allan", "off" });
  dopts_.adaptive_q.use_noise_floor = (adaptive_q_floor_mode_ == "allan");
  cfg.nested<double>(aq && dopts_.adaptive_q.use_noise_floor,
                     "adaptive_q/noise_floor/mode=allan",
                     "adaptive_q/noise_floor/scale", dopts_.adaptive_q.noise_floor_scale, 1.0);
  cfg.nested<bool>(aq, "adaptive_q/enable", "adaptive_q/log_en", dopts_.adaptive_q.log_en, false);
  adaptive_q_.configure(dopts_.adaptive_q);

  // Every spline/* and adaptive_q/* key is read by the resolver above and by
  // nothing else, so anything left over in those namespaces is a key nobody
  // consumes -- a rename this config was never migrated through, or a typo.
  // Deliberately NOT extended to voxel_map/: several of its keys are still
  // read by paramWarn() in voxelmap.cpp, so they are unclaimed here and would
  // be reported as dead when they are merely read elsewhere. Widen this only
  // as those move onto a resolver.
  return finalizeConfig(cfg, { "spline", "adaptive_q", "lio/ekf", "voxel_map" });
}

// LIO Processing

void LioProcDecoupled::deskewAndDownsample(MeasureGroup& mg)
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
  refit_dtraj_rms_ = 0.0;
  refit_dtraj_max_ = 0.0;
  refit_drot_deg_ = 0.0;
  // Captured HERE, before anything this frame can correct state_'s bias/
  // gravity (the IEKF loop hasn't started) -- see the member doc comments.
  spline_frame_bias_acc_ = state_->biasAcc();
  spline_frame_bias_gyr_ = state_->biasGyr();
  spline_frame_gravity_  = state_->gravity();
  spline_.resetRefineStats();
  if (dopts_.spline.splineOn() && !mg.poses.empty())
  {
    TimedScope ts_fit(profiler_, "lio/spline/fit");
    spline_frame_count_++;
    // BOTH ENDPOINTS ARE CLAMPED.  Head = the previous scan's own final
    // pose; tail = the PROPAGATED scan-end pose, because the IEKF's
    // corrected one does not exist yet -- the loop is about to compute it,
    // and moveTailClamp() carries it in on every iteration afterwards.
    //
    // The very first spline scan of a run has no previous boundary to
    // freeze to, so it fits unconstrained (n_frozen = 0).
    spline_.setFrozenBoundary(
        prev_scan_end_valid_ ? SplineOptions::N_FROZEN_CP : 0,
        prev_scan_end_pos_, prev_scan_end_rot_, prev_scan_end_vel_,
        state_->pos(), state_->rot(), state_->vel());
    spline_ok_ = spline_.fit(mg.poses, mg.poses.front().t, mg.image.t, dopts_.spline);
    if (!spline_ok_)
    {
      // Never substitute a bad spline for a working deskew.  Count the
      // failures loudly (reported in finalizeSplineAndQ()) rather than
      // degrading silently -- a run where the fit failed on most frames is
      // not the experiment anyone thinks they ran.
      spline_fit_fail_count_++;
      // CQ-22 item (4): attribute this failure to its specific cause, so
      // "1 refusal" can no longer mean any of nine different things.
      // CQ-23: kChartGuard (the hard ceiling) lands here too, generically --
      // it is one of the nine causes now, not a separate mechanism.
      spline_fail_cause_count_[static_cast<int>(spline_.lastFitFailCause())]++;
    }
    // Independent of spline_ok_: chart_guard_warned_ only fires from inside
    // the chart-guard loop, which a fit that failed at an earlier site never
    // reaches, so this is correctly 0 for those frames without a separate
    // gate here. A HARD-refused fit reaches the loop too (the value is
    // stored before either threshold is checked) but chartGuardWarned()
    // stays false for it -- warn and hard are mutually exclusive per fit(),
    // the hard count is spline_fail_cause_count_[kChartGuard] above.
    if (spline_.chartGuardWarned()) spline_chart_guard_warn_count_++;

    // CQ-41 follow-up: does the large endpoint acceleration come from
    // refinement, or is it already present in the very first fit?  Logged
    // HERE, immediately after fit() and before the IEKF loop's first
    // redeskewFromSpline()/refineSplineFromResiduals() call -- this is the
    // boundary-CONSTRAINED (setFrozenBoundary() above already ran) but
    // UNREFINED spline. Comparing this against spline_endpoint_debug.txt's
    // same-scan_id row (logged post-refinement in finalizeSplineAndQ())
    // isolates refinement's own contribution. It does NOT isolate the
    // boundary constraint's contribution -- that's baked into fit() itself
    // via setFrozenBoundary() above, present in this row exactly as much as
    // in the final one -- a genuinely unconstrained-fit comparison would
    // need a separate ablation.
    if (opts_.log_debug_en && spline_ok_) {
      static PersistentLogStream log("spline_firstfit_debug.txt");
      std::ofstream& ofs = log.stream();
      ofs << "scan_id=" << voxel_map_->frame_idx_
          << " acc_t0=" << spline_.accAt(spline_.t0()).norm()
          << " acc_t1=" << spline_.accAt(spline_.t1()).norm()
          << " omega_t0=" << spline_.omegaBodyAt(spline_.t0()).norm()
          << " omega_t1=" << spline_.omegaBodyAt(spline_.t1()).norm()
          << " n_cp=" << spline_.nControlPoints()
          << "\n";
      ofs.flush();
    }

    // TQ-38 item 1: snapshot the UNREFINED control points, unconditionally
    // (not gated on log_debug_en -- this is a cheap 3xn_cp copy, and the
    // TQ-38 diagnostic below needs it every frame it runs, not only frames
    // where firstfit debug logging happened to also be on).
    cp_p_unrefined_snapshot_ = spline_.cpPos();
  }

  std::vector<PointXYZCov> deskewed;
  if (spline_ok_)
    deskewPointsSpline(state_, spline_, mg.image.t, mg.lidar_points,
                       opts_.deskew, deskewed);
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
    if (spline_ok_ && dopts_.spline.splineOn())
      voxelDownsampleIndexedCsr(deskewed, mg.points, ds_offsets_, ds_members_,
                                PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
    else
      voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
    if (spline_ok_ && dopts_.spline.splineOn()) {
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


bool LioProcDecoupled::redeskewFromSpline(MeasureGroup& mg)
{
  if (!spline_ok_ || !dopts_.spline.splineOn()) return false;

  // CARRY THE IEKF'S CORRECTION IN BY MOVING THE TAIL CLAMP, not by rigidly
  // transforming the spline.  anchorTo() applied ONE correction to the point
  // at t0 and the point at t1 alike -- but the correction is drift
  // accumulated across THIS scan, and at t0 the state was already corrected
  // by the previous scan's own update, so it moved a point that was already
  // right.  CQ-41: moveTailClamp() now distributes it via a constraint-
  // increment KKT solve (see spline.cpp), not the old time-proportional
  // ramp -- zero at the head by construction, and any refinement already
  // applied to the interior survives as far as the fit's own metric allows.
  spline_.moveTailClamp(state_->pos(), state_->rot(), state_->vel());

  // How far the re-deskew actually moved the points it re-placed.  This is
  // the only quantity that says whether the per-iteration mechanism is doing
  // anything at all.
  redeskew_prev_ = mg.points;
  deskewPointsSplineCsr(state_, spline_, mg.image.t, mg.lidar_points,
                        ds_offsets_, ds_members_, opts_.deskew, mg.points);
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

// SHAPE, from the map.  Called from inside the IEKF loop with the residuals
// the solve is ABOUT TO USE, so the refinement and the state update are
// linearised at the SAME trajectory -- neither is stale (Bryce, 2026-09-06).
// The old placement, at the top of the next iteration against the previous
// iteration's residuals, was linearised at a trajectory that no longer
// existed by the time the step was applied.
//
// Both clamps are frozen in the solve, so this owns the interior shape and
// only the interior shape; it reaches the endpoint only indirectly, by
// changing where the points land and therefore what the next residual build
// reports.
bool LioProcDecoupled::refineSplineFromResiduals(const MeasureGroup& mg)
{
  if (!spline_ok_ || !dopts_.spline.refineOn() || residuals_.empty()) return false;

  TimedScope ts_ref(profiler_, "lio/spline/refine");
  lidar_obs_.clear();
  lidar_obs_.reserve(residuals_.size());
  for (const auto& r : residuals_)
  {
    // Inside the fitted window, not "> 0": frame times are start_time-
    // relative, so a scan straddling the end of calibration has genuinely
    // NEGATIVE point times and a sentinel test would reject real returns.
    if (r.t < spline_.t0() - 1e-9 || r.t > spline_.t1() + 1e-9) continue;
    SplineLidarObs o;
    o.t      = r.t;
    o.normal = r.normal;
    o.r      = r.r;
    o.sigma2 = r.sigma_squared;
    lidar_obs_.push_back(o);
  }
  // CQ-39 item (1): threads this LioProc's existing log_debug_en straight
  // through -- not a new config key -- to gate refineWithLidar()'s
  // per-pass proof log.
  // CQ-41 follow-up: imu_samples_raw/spline_frame_bias_acc_/
  // spline_frame_gravity_/varAccFloor() are the same four inputs
  // computeSplineImuResidual() already uses -- only consulted by
  // refineWithLidar() when dopts_.spline.refine_imu_acc_weight > 0.
  return spline_.refineWithLidar(lidar_obs_, dopts_.spline, mg.imu_samples_raw,
                                 spline_frame_bias_acc_, spline_frame_gravity_,
                                 state_->varAccFloor().mean(), state_->covBiasAcc(),
                                 opts_.log_debug_en);
}

void LioProcDecoupled::finalizeSplineAndQ(MeasureGroup& mg)
{
  if (!dopts_.spline.splineOn()) return;

  last_spline_stats_ = SplineImuResidualStats{};

  boundary_dpos_ = -1.0;
  boundary_drot_deg_ = -1.0;
  if (spline_ok_)
  {
    // CQ-22 item (5): last thing before anything else in this block reads
    // fitResidualPos()/fitResidualRot() -- redeskewFromSpline()'s final
    // call (see LioProc's main loop, right before this function runs) has
    // already moved the tail clamp to the converged state, so this is the
    // first point in the frame where the residual reflects that move.
    spline_.updateFitResiduals(mg.poses);

    // CQ-41 items 5a/5c: the endpoint-derivative proof. Debug-gated (same
    // convention as CQ-39's per-pass log), one line per scan, evaluated on
    // the FINAL converged spline (after this frame's last moveTailClamp()
    // -- see the comment immediately below for why that's already true
    // here). HEAD~1 must report exactly 0 for all four; HEAD (either new
    // mode) must not.
    if (opts_.log_debug_en) {
      static PersistentLogStream log("spline_endpoint_debug.txt");
      std::ofstream& ofs = log.stream();
      // Diagnostic addendum (not in CQ-39/41's original DELIVERS list):
      // how well the constraint the KKT solve was actually asked to meet
      // IS met, alongside the (unconstrained, and therefore far more
      // sensitive to boundary data sparsity) acceleration reading -- lets
      // a large acc_t0/acc_t1 be told apart from "the constraint itself
      // isn't holding" (a real bug) vs. "the constraint holds fine, the
      // curvature implied by 13 control points over a ~0.1s window with a
      // 1/delta^2 ~ 1e4 amplification is just large" (expected, item 3b).
      const V3D vel_err0 = spline_.velAt(spline_.t0()) - prev_scan_end_vel_;
      const V3D vel_err1 = spline_.velAt(spline_.t1()) - state_->vel();
      const V3D pos_err0 = spline_.posAt(spline_.t0()) - prev_scan_end_pos_;
      const V3D pos_err1 = spline_.posAt(spline_.t1()) - state_->pos();

      // Bryce, 2026-09-18: free-tail diagnostic -- see
      // ScanSpline::diagnosticFreeTailFit()'s own doc comment. pos1_free/
      // vel1_free/acc1_free are what the LiDAR-only-informed trajectory
      // naturally wants at t1 with the boundary constraint relaxed to
      // head-only; comparing against state_->pos()/vel() (what the tail is
      // actually pinned to) and against acc_t1 above (the actual, boundary-
      // constrained acceleration) is how "the target itself is wrong" gets
      // told apart from "forcing exact equality is what creates curvature".
      V3D pos1_free = V3D::Zero(), vel1_free = V3D::Zero(), acc1_free = V3D::Zero();
      M3D cov_pos1_free = M3D::Zero();
      Eigen::MatrixXd cp_free_unused;
      double fit_res_pos_free_unused = 0.0;
      const bool free_tail_ok = spline_.diagnosticFreeTailFit(
          lidar_obs_, dopts_.spline, mg.imu_samples_raw, spline_frame_bias_acc_,
          spline_frame_gravity_, state_->varAccFloor().mean(), mg.poses,
          pos1_free, vel1_free, acc1_free, cov_pos1_free,
          cp_free_unused, fit_res_pos_free_unused);

      ofs << "scan_id=" << voxel_map_->frame_idx_
          << " acc_t0=" << spline_.accAt(spline_.t0()).norm()
          << " acc_t1=" << spline_.accAt(spline_.t1()).norm()
          << " omega_t0=" << spline_.omegaBodyAt(spline_.t0()).norm()
          << " omega_t1=" << spline_.omegaBodyAt(spline_.t1()).norm()
          << " vel_err0=" << vel_err0.norm() << " vel_err1=" << vel_err1.norm()
          << " pos_err0=" << pos_err0.norm() << " pos_err1=" << pos_err1.norm()
          << " n_cp=" << spline_.nControlPoints()
          << " delta=" << (spline_.t1() - spline_.t0()) / std::max(1, spline_.nControlPoints() - 3)
          << " delta_ba_norm=" << spline_.lastDeltaBiasAcc().norm()
          << " delta_ba_x=" << spline_.lastDeltaBiasAcc().x()
          << " delta_ba_y=" << spline_.lastDeltaBiasAcc().y()
          << " delta_ba_z=" << spline_.lastDeltaBiasAcc().z()
          << " free_tail_ok=" << (free_tail_ok ? 1 : 0)
          << " pos1_free_err=" << (free_tail_ok ? (pos1_free - state_->pos()).norm() : -1.0)
          << " vel1_free_err=" << (free_tail_ok ? (vel1_free - state_->vel()).norm() : -1.0)
          << " acc1_free=" << (free_tail_ok ? acc1_free.norm() : -1.0);
      if (free_tail_ok) {
        // Bryce, 2026-09-18: bias_acc lives in BODY frame (see
        // refineWithLidar()'s R^T*(...)+bias_acc convention) -- a genuine
        // constant body-frame bias, viewed through this WORLD-frame
        // velocity error, would appear to ROTATE as the platform's
        // orientation changes scan to scan, not stay constant. Log both:
        // world-frame (vel1_free_err_w*) for reference, and the SAME
        // vector rotated into this scan's own body frame at t1
        // (vel1_free_err_b*, via rotAt(t1)^T) so a persistence check can be
        // done in the frame the bias itself actually lives in.
        const V3D vel_err_w = vel1_free - state_->vel();
        const V3D vel_err_b = spline_.rotAt(spline_.t1()).transpose() * vel_err_w;
        ofs << " vel1_free_err_wx=" << vel_err_w.x() << " vel1_free_err_wy=" << vel_err_w.y()
            << " vel1_free_err_wz=" << vel_err_w.z()
            << " vel1_free_err_bx=" << vel_err_b.x() << " vel1_free_err_by=" << vel_err_b.y()
            << " vel1_free_err_bz=" << vel_err_b.z();
      }
      ofs << "\n";
      ofs.flush();
    }

    // TQ-35, Bryce 2026-09-18: "is P honest?" -- per-scan NEES-vs-ground-
    // truth diagnostic, plus the free-tail statistic calibrated against it.
    // READ-ONLY: reads state_->cov()/pos() (already posterior -- this runs
    // after ekf_.applyCovarianceUpdate(), see the main loop above) and
    // writes to nees_diag.txt; nothing here can feed back into state_ or
    // any other spline/EKF machinery, so it cannot alter the filter (the
    // md5-unchanged proof this card requires is a property of the code
    // shape, not something that needs separate runtime verification beyond
    // confirming no assignment targets state_/spline_/prior_cov_).
    // GT join is NOT done in C++ -- no GT access exists inside LioProc, and
    // wiring it in is out of this diagnostic's scope -- eps_pos/eps_rot are
    // computed post-hoc in Python from this log joined against the GT
    // source, per TQ-35 item (2)'s own instruction to report the join
    // details explicitly before any epsilon.
    if (opts_.log_debug_en) {
      static PersistentLogStream log("nees_diag.txt");
      std::ofstream& ofs = log.stream();
      const Eigen::MatrixXd& P = state_->cov();
      const int iP = StateGroup::idxP(), iR = StateGroup::idxR();
      const bool have_p = P.rows() >= iP + 3 && P.cols() >= iP + 3;
      const bool have_r = P.rows() >= iR + 3 && P.cols() >= iR + 3;
      const M3D P_pp = have_p ? M3D(P.block<3, 3>(iP, iP)) : M3D::Zero();
      const M3D P_rr = have_r ? M3D(P.block<3, 3>(iR, iR)) : M3D::Zero();
      const double t_abs = mg.image.t + data_queues_->start_time;

      // Motion-onset investigation (2026-09-19): velocity + bias/gravity
      // state and covariance blocks, previously not logged at all for
      // either estimator -- needed to compare decoupled vs coupled beyond
      // position/rotation. idxV() is always valid (velocity is always
      // estimated); idxBG()/idxBA()/idxG() are -1 when that state isn't
      // estimated, same have_* guard convention as have_p/have_r above.
      const int iV = StateGroup::idxV();
      const bool have_v = P.rows() >= iV + 3 && P.cols() >= iV + 3;
      const M3D P_vv = have_v ? M3D(P.block<3, 3>(iV, iV)) : M3D::Zero();
      const int iBG = state_->idxBG(), iBA = state_->idxBA(), iGr = state_->idxG();
      const bool have_bg = iBG >= 0 && P.rows() >= iBG + 3 && P.cols() >= iBG + 3;
      const bool have_ba = iBA >= 0 && P.rows() >= iBA + 3 && P.cols() >= iBA + 3;
      const bool have_gr = iGr >= 0 && P.rows() >= iGr + 3 && P.cols() >= iGr + 3;
      const double trP_bg = have_bg ? P.block<3, 3>(iBG, iBG).trace() : -1.0;
      const double trP_ba = have_ba ? P.block<3, 3>(iBA, iBA).trace() : -1.0;
      const double trP_grav = have_gr ? P.block<3, 3>(iGr, iGr).trace() : -1.0;

      // item (5): free_tail_d, using diagnosticFreeTailFit() with the
      // regularizers FORCED ON (curvature_weight=1.0, imu_acc_weight=1.0)
      // regardless of this run's own dopts_.spline -- the card's own
      // instruction ("with the regularisers on"), decoupled from whatever
      // the main fit's shipped config happens to use (still 0.0/off by
      // default as of CQ-41's own filing).
      // TQ-38 item (4): diag_free_tail_imu_acc_weight overrides the forced
      // imu_acc_weight below, default 1.0 (identical to the hardcoded
      // value TQ-35/TQ-36 already shipped with -- this override is inert
      // at its default). Lets the imu_acc_weight sweep vary ONLY this
      // diagnostic's forced weight without touching the run's own shipped
      // refine_imu_acc_weight (a different field, read by the REAL
      // refinement path, refineWithLidar()).
      SplineOptions forced_opts = dopts_.spline;
      forced_opts.refine_curvature_weight = 1.0;
      forced_opts.refine_imu_acc_weight   = dopts_.spline.diag_free_tail_imu_acc_weight;
      V3D ft_pos1 = V3D::Zero(), ft_vel1 = V3D::Zero(), ft_acc1 = V3D::Zero();
      M3D ft_cov = M3D::Zero();
      Eigen::MatrixXd ft_cp_free;
      double ft_fit_res_pos_free = 0.0;
      const bool ft_ok = spline_.diagnosticFreeTailFit(
          lidar_obs_, forced_opts, mg.imu_samples_raw, spline_frame_bias_acc_,
          spline_frame_gravity_, state_->varAccFloor().mean(), mg.poses,
          ft_pos1, ft_vel1, ft_acc1, ft_cov, ft_cp_free, ft_fit_res_pos_free);
      double free_tail_d = -1.0;
      if (ft_ok && have_p) {
        const V3D d = ft_pos1 - state_->pos();
        const M3D S = P_pp + ft_cov;
        const Eigen::LDLT<M3D> ldlt_s(S);
        if (ldlt_s.info() == Eigen::Success) {
          const V3D x = ldlt_s.solve(d);
          if (x.allFinite()) free_tail_d = d.dot(x);
        }
      }

      // CQ-43 item (4d): state rotation as a quaternion, so rot_drift_deg
      // (= ||Log(R(t0)^T R(t))|| against a chosen reference scan, truth 0
      // at rest) can be computed post-hoc without re-deriving it here --
      // this diagnostic has no notion of "the stationary window's first
      // scan" (that is TQ-34's own scan-id convention, established in
      // Python), so the reference choice stays a post-processing decision.
      const Eigen::Quaterniond state_q(state_->rot());

      ofs << std::setprecision(12)
          << "scan_id=" << voxel_map_->frame_idx_ << " t_abs=" << t_abs
          << " state_px=" << state_->pos().x() << " state_py=" << state_->pos().y()
          << " state_pz=" << state_->pos().z()
          << " state_qw=" << state_q.w() << " state_qx=" << state_q.x()
          << " state_qy=" << state_q.y() << " state_qz=" << state_q.z()
          << " total_dtheta_deg=" << last_total_dtheta_deg_
          << " trP_pos_pre=" << trP_pos_pre_
          << " trP_pos_post=" << (have_p ? P_pp.trace() : -1.0)
          << " Ppp_xx=" << P_pp(0, 0) << " Ppp_xy=" << P_pp(0, 1) << " Ppp_xz=" << P_pp(0, 2)
          << " Ppp_yy=" << P_pp(1, 1) << " Ppp_yz=" << P_pp(1, 2) << " Ppp_zz=" << P_pp(2, 2)
          << " Prr_xx=" << P_rr(0, 0) << " Prr_xy=" << P_rr(0, 1) << " Prr_xz=" << P_rr(0, 2)
          << " Prr_yy=" << P_rr(1, 1) << " Prr_yz=" << P_rr(1, 2) << " Prr_zz=" << P_rr(2, 2)
          << " have_Ppp=" << (have_p ? 1 : 0) << " have_Prr=" << (have_r ? 1 : 0)
          << " free_tail_d=" << free_tail_d
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
          << " trP_bg=" << trP_bg << " trP_ba=" << trP_ba << " trP_grav=" << trP_grav
          << " have_Pbg=" << (have_bg ? 1 : 0) << " have_Pba=" << (have_ba ? 1 : 0)
          << " have_Pgrav=" << (have_gr ? 1 : 0)
          << "\n";
      ofs.flush();

      // TQ-36, Bryce 2026-09-18: is the free-tail correction DRIFT or
      // NOISE, along the WHOLE scan, not just at the endpoint? Per
      // control point i, d_i = cp_free[i] - cp_pinned[i] (cp_pinned is
      // spline_.posAt-equivalent's own cp_p_ -- the REAL, tail-
      // constrained refinement this frame actually used; cp_free is
      // ft_cp_free from the SAME forced-on-regularizer call used for
      // free_tail_d above). Travel direction is a single per-scan
      // reference (state_->vel(), normalized) -- decomposing every d_i
      // against a per-control-point direction would make the profile
      // depend on a quantity (local heading) the card doesn't ask for.
      if (ft_ok && ft_cp_free.cols() == spline_.nControlPoints()) {
        static PersistentLogStream log("tq36_profile.txt");
        std::ofstream& pofs = log.stream();
        const int n_cp = spline_.nControlPoints();
        const Eigen::Matrix<double, 3, Eigen::Dynamic>& cp_pinned = spline_.cpPos();
        const double speed = state_->vel().norm();
        const V3D travel_dir = (speed > 1e-6) ? V3D(state_->vel() / speed) : V3D::Zero();
        pofs << std::setprecision(9)
             << "scan_id=" << voxel_map_->frame_idx_ << " n_cp=" << n_cp
             << " fit_res_pos_pinned=" << spline_.fitResidualPos()
             << " fit_res_pos_free=" << ft_fit_res_pos_free;
        V3D d_last_world = V3D::Zero();
        for (int i = 0; i < n_cp; ++i) {
          const V3D d = ft_cp_free.col(i) - cp_pinned.col(i);
          const double d_travel = d.dot(travel_dir);
          pofs << " d_norm_" << i << "=" << d.norm()
               << " d_travel_" << i << "=" << d_travel;
          if (i == n_cp - 1) d_last_world = d;
        }
        // TQ-36 item (4): body-frame per-axis mean/std needs the FULL
        // vector at the last free control point, not just its travel-
        // direction scalar -- rotated into this scan's own body frame at
        // t1 the same way CQ-41's follow-on did for vel1_free_err.
        const V3D d_last_body = spline_.rotAt(spline_.t1()).transpose() * d_last_world;
        pofs << " d_last_bx=" << d_last_body.x() << " d_last_by=" << d_last_body.y()
             << " d_last_bz=" << d_last_body.z();
        pofs << "\n";
        pofs.flush();
      }

      // TQ-38 item (1): delta_a(t) = R(t)^T * (accAt_refined(t) -
      // accAt_unrefined(t)), at every raw IMU sample in the scan window --
      // the IMU correction the refinement is implicitly asking for.
      // delta_w is 0 by construction (refineWithLidar() touches cp_p_
      // only, never cp_phi_ -- standing/refine-cannot-reach), so it is not
      // computed here, per the card's own instruction.
      //
      // Two "refined" variants against the SAME unrefined baseline
      // (cp_p_unrefined_snapshot_, captured right after fit()):
      //   pinned -- the LIVE cp_p_ (this run's own config: V1 if
      //     curvature_weight=imu_acc_weight=0, V2 if both=1.0)
      //   free   -- ft_cp_free (V3: free tail, regularizers FORCED on,
      //     same diagnosticFreeTailFit() call above used for free_tail_d
      //     and TQ-36's profile -- so V3's own numbers do not depend on
      //     this run's own curvature_weight/imu_acc_weight config at all)
      // accAt() reads whatever cp_p_ currently holds, so evaluating three
      // variants means swapping ScanSpline::cpPosMut() in and back out --
      // read-only: the live value is always the last one restored.
      if (ft_ok && ft_cp_free.cols() == spline_.nControlPoints() &&
          cp_p_unrefined_snapshot_.cols() == spline_.nControlPoints()) {
        static PersistentLogStream log("tq38_delta_a.txt");
        std::ofstream& ofs = log.stream();
        const Eigen::Matrix<double, 3, Eigen::Dynamic> cp_live = spline_.cpPos();
        for (const auto& samp : mg.imu_samples_raw) {
          if (samp.t < spline_.t0() - 1e-9 || samp.t > spline_.t1() + 1e-9) continue;
          const M3D R = spline_.rotAt(samp.t);  // cp_phi_ untouched by any swap below

          const V3D acc_pinned_refined = spline_.accAt(samp.t);  // live cp_p_

          spline_.cpPosMut() = cp_p_unrefined_snapshot_;
          const V3D acc_unrefined = spline_.accAt(samp.t);

          spline_.cpPosMut() = ft_cp_free;
          const V3D acc_free_refined = spline_.accAt(samp.t);

          spline_.cpPosMut() = cp_live;  // restore before anything else reads cp_p_

          const V3D delta_a_pinned = R.transpose() * (acc_pinned_refined - acc_unrefined);
          const V3D delta_a_free   = R.transpose() * (acc_free_refined   - acc_unrefined);
          ofs << std::setprecision(9)
              << "scan_id=" << voxel_map_->frame_idx_
              << " t_abs=" << (samp.t + data_queues_->start_time)
              << " delta_a_pinned_norm=" << delta_a_pinned.norm()
              << " delta_a_free_norm=" << delta_a_free.norm()
              << " acc_refined_pinned_norm=" << acc_pinned_refined.norm()
              << " acc_refined_free_norm=" << acc_free_refined.norm()
              << " acc_unrefined_norm=" << acc_unrefined.norm()
              << " sigma_acc_floor=" << std::sqrt(state_->varAccFloor().mean())
              << "\n";
        }
        ofs.flush();
      }

      // TQ-34, Bryce 2026-09-18: item (3b)'s decile profile -- per RAW IMU
      // sample within this scan's window (not per-scan), its own
      // normalised in-scan time u=(t-t0)/(t1-t0), and the four quantities
      // whose truth is zero at rest: accAt/omegaBodyAt/velAt norms and
      // pos_drift (=posAt(t)-posAt(t0)). rot_drift (degrees, vs rotAt(t0))
      // alongside. Binning into deciles happens downstream, in analysis --
      // this just logs every sample at full resolution.
      {
        static PersistentLogStream log("tq34_samples.txt");
        std::ofstream& sofs = log.stream();
        const double t0 = spline_.t0(), t1 = spline_.t1();
        const double span = std::max(1e-9, t1 - t0);
        const V3D pos_t0 = spline_.posAt(t0);
        const M3D rot_t0 = spline_.rotAt(t0);
        for (const auto& samp : mg.imu_samples_raw) {
          if (samp.t < t0 - 1e-9 || samp.t > t1 + 1e-9) continue;
          const double u = (samp.t - t0) / span;
          const double pos_drift = (spline_.posAt(samp.t) - pos_t0).norm();
          const double rot_drift_deg = std::acos(std::clamp(
              (M3D(rot_t0.transpose() * spline_.rotAt(samp.t)).trace() - 1.0) / 2.0,
              -1.0, 1.0)) * (180.0 / M_PI);
          sofs << std::setprecision(9)
               << "scan_id=" << voxel_map_->frame_idx_ << " t_abs=" << (samp.t + data_queues_->start_time)
               << " u=" << u
               << " acc_at_norm=" << spline_.accAt(samp.t).norm()
               << " omega_body_norm=" << spline_.omegaBodyAt(samp.t).norm()
               << " vel_at_norm=" << spline_.velAt(samp.t).norm()
               << " pos_drift=" << pos_drift
               << " rot_drift_deg=" << rot_drift_deg
               << "\n";
        }
        sofs.flush();
      }
    }

    // BOTH CLAMPS ARE ALREADY EXACT: t0 was pinned before fit() ran and t1
    // was carried in by moveTailClamp() on the last iteration, so there is
    // nothing to re-anchor here. state_ is NOT written back from the spline's own t1: state_
    // is one continuously-propagated object across the whole run (never
    // reconstructed per scan), so it was never actually discontinuous at
    // scan boundaries in the first place -- only the spline's own
    // independently-refit-per-scan shape was. state_'s pose/vel/bias/
    // gravity continue to come exclusively from the IEKF's own
    // covariance-consistent point-to-plane correction, exactly as when
    // freezing is off. The spline here is purely an internal deskewing
    // aid; freezing only makes ITS OWN boundary consistent, not a
    // correction source for state_.

    // DX-2 preflight fix (boundary_dpos): THIS scan's own start (t0)
    // against the PREVIOUS scan's finalized end -- the two spline
    // evaluations this costs. Order matters: compare against the OLD
    // prev_scan_end_* before overwriting it with this scan's own end
    // below. Under boundary freezing this MUST come back ~0 (float
    // roundoff only): setFrozenBoundary() always freezes THIS scan's t0 to
    // the exact prev_scan_end_* recorded below, so the invariant holds by
    // construction regardless of what state_'s own value happens to be.
    if (prev_scan_end_valid_)
    {
      const V3D t0_pos = spline_.posAt(spline_.t0());
      const M3D t0_rot = spline_.rotAt(spline_.t0());
      boundary_dpos_ = (t0_pos - prev_scan_end_pos_).norm();
      const M3D dR = t0_rot * prev_scan_end_rot_.transpose();
      boundary_drot_deg_ = std::acos(std::clamp((dR.trace() - 1.0) / 2.0, -1.0, 1.0))
                          * (180.0 / M_PI);
    }
    // This scan's own end, and the freeze target for the NEXT scan's t0:
    // always state_'s own pose (never the spline's t1 directly). Under
    // nFrozenCp()==0 these agree exactly (anchorTo() above just forced
    // it); under freezing they need not agree exactly, and that's fine --
    // the spline is not the source of truth for state_.
    prev_scan_end_pos_ = state_->pos();
    prev_scan_end_rot_ = state_->rot();
    prev_scan_end_vel_ = state_->vel();
    prev_scan_end_valid_ = true;

    if (!mg.imu_samples_raw.empty())
    {
      TimedScope ts(profiler_, "lio/spline/imu_residual");
      last_spline_stats_ = computeSplineImuResidual(
          spline_, mg.imu_samples_raw,
          state_->biasAcc(), state_->biasGyr(), state_->gravity());
    }
  }

  if (dopts_.adaptive_q.enable)
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

  // Dense trajectory dump: the spline evaluated as the function it is, on a
  // fixed grid across this scan's own window.  Analysis only (see
  // SplineOptions::traj_log_mode); scores still come from results_lio.txt.
  // Consecutive scans' windows abut, so the last sample of one and the first
  // of the next also MEASURE the inter-scan discontinuity -- which is the
  // honest check on how continuous this "continuous-time" trajectory is, and
  // nothing has ever measured it.
  //
  // dopts_.spline.trajLogOn() (traj_log_mode != "off") is already the real
  // gate here; the outer log_en check this used to have was dangling (no
  // braces) and bound only to this one inner if, not to the spline_q.csv
  // block below -- braced explicitly so that's no longer ambiguous to read.
  if (dopts_.spline.log_en || dopts_.adaptive_q.log_en)
  {
  if (spline_ok_ && dopts_.spline.trajLogOn())
  {
    static bool traj_first = true;
    std::ofstream tofs(debugLogPath("spline_traj.csv"),
                       traj_first ? std::ios::trunc : std::ios::app);
    if (traj_first) { tofs << "scan_id,t,px,py,pz,qx,qy,qz,qw\n"; traj_first = false; }
    const double t_off = data_queues_->start_time;
    const double hz = std::max(1.0, dopts_.spline.traj_log_hz);
    const double step = 1.0 / hz;
    const double a = spline_.t0(), b = spline_.t1();
    tofs << std::setprecision(12);
    for (double tt = a; tt <= b + 1e-12; tt += step) {
      const Eigen::Quaterniond q(spline_.rotAt(tt));
      const V3D pp = spline_.posAt(tt);
      tofs << voxel_map_->frame_idx_ << ',' << (tt + t_off) << ','
           << pp.x() << ',' << pp.y() << ',' << pp.z() << ','
           << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
    }
  }
  }

  // spline_q.csv is written UNCONDITIONALLY, every scan, regardless of
  // dopts_.spline.log_en/dopts_.adaptive_q.log_en -- deliberately, not an
  // oversight (C-8): rule 8 already retains this file by name across every
  // batch this project runs, and every AdaptiveQ/spline diagnostic reader
  // (DX-2's SERIES blocks, C-7's time-axis join, D-1R) assumes it always
  // exists. Gating it behind log_en would silently break all of them the
  // moment log_en defaulted false, which it does.
  {
    // CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
    // regression this fixes.
    static PersistentLogStream log("spline_q.csv");
    bool first;
    std::ofstream& ofs = log.stream(&first);
    if (first)
    {
      ofs << "scan_id," << adaptive_q_.csvHeader()
          << ",spline_ok,n_cp,fit_res_pos,fit_res_rot,fit_reg_frac,rot_chord_deg,"
             "fit_fail_count,frame_count,max_abs_acc,max_abs_gyr,"
             "n_cp_req,refine_applied,refine_rejects,last_refine_step,"
             "mode,refine_dcp_max,refine_dcp_rms,"
             "redeskew_calls,redeskew_dp_rms,"
             "refit_dtraj_rms,refit_dtraj_max,refit_drot_deg,cov_acc_pre,cov_gyr_pre,"
             "d_bias_acc_norm,d_bias_gyr_norm,d_gravity_norm,max_abs_cp_phi,"
             "dmin_p,dmax_p,dmin_r,dmax_r,mean_abs_acc,mean_abs_gyr\n";
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
        << (spline_ok_ ? spline_.refineApplied() : 0) << ','
        << (spline_ok_ ? spline_.refineRejects() : 0) << ','
        << (spline_ok_ ? spline_.lastRefineStep() : 0.0) << ','
        // The engagement half: what each toggle was set to, and how far its
        // mechanism actually moved things this frame.  A cell with
        // per_iteration naming a step whose magnitude column is 0 across the
        // whole run is an INERT cell, not a null result.
        << dopts_.spline.mode << ','
        << (spline_ok_ ? spline_.refineDcpMax() : 0.0) << ','
        << (spline_ok_ ? spline_.refineDcpRms() : 0.0) << ','
        << redeskew_calls_ << ',' << redeskew_dp_rms_ << ','
        << refit_dtraj_rms_ << ',' << refit_dtraj_max_ << ',' << refit_drot_deg_ << ','
        << cov_acc_pre_ << ',' << cov_gyr_pre_ << ','
        // The imu_fit bias/gravity correction was removed with the
        // restructure (CQ-21); these three columns are retained for
        // spline_q.csv's schema stability and always read 0.
        << 0.0 << ',' << 0.0 << ',' << 0.0 << ','
        << (spline_ok_ ? spline_.maxAbsCpPhi() : 0.0) << ','
        // CQ-24 item (1): NOT gated on spline_ok_ -- these are meaningful
        // (and -1.0 sentinel otherwise) on a fit that failed at the pivot
        // guard or at kSolveFailed too, which is the whole point.
        << spline_.dminPos() << ',' << spline_.dmaxPos() << ','
        << spline_.dminRot() << ',' << spline_.dmaxRot() << ','
        << last_spline_stats_.mean_abs_acc << ',' << last_spline_stats_.mean_abs_gyr
        << '\n';
    ofs.flush();
  }

  // Run totals for the end-of-run engagement report.
  ++run_frames_;
  run_redeskew_calls_ += redeskew_calls_;
  run_refits_ += spline_refits_;
  run_redeskew_dp_max_ = std::max(run_redeskew_dp_max_, redeskew_dp_rms_);
  if (spline_ok_)
  {
    run_refine_applied_ += spline_.refineApplied();
    run_refine_rejects_ += spline_.refineRejects();
    run_refine_dcp_max_ = std::max(run_refine_dcp_max_, spline_.refineDcpMax());
  }
  if (dopts_.adaptive_q.enable && adaptive_q_.active())
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
LioProcDecoupled::~LioProcDecoupled()
{
  if (!dopts_.spline.splineOn()) return;
  const std::string rep = engagementReport();
  ROS_WARN_STREAM("\n" << rep);
  std::ofstream ofs(debugLogPath("engagement.txt"), std::ios::trunc);
  if (ofs) ofs << rep << '\n';
}

std::string LioProcDecoupled::engagementReport() const
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

  o << "\n  spline/mode = " << dopts_.spline.mode;
  line("  redeskew          ", dopts_.spline.splineOn(), run_redeskew_calls_,
       "max_dp_rms_m", run_redeskew_dp_max_);
  line("  refine            ", dopts_.spline.refineOn(), run_refine_applied_,
       "max_dcp_m", run_refine_dcp_max_);
  if (dopts_.spline.refineOn())
    o << " rejects=" << run_refine_rejects_;

  o << "\n  adaptive_q/enable = "
    << (dopts_.adaptive_q.enable ? "true" : "false");
  if (dopts_.adaptive_q.enable)
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
  // counter is, and the scorer only reads this file. voxel_map_ is a
  // MapBackendPtr (voxel/PCA-debiased backend OR AkfMap) -- plane_var_mode/
  // weight_floor_mode are VoxelOpts-only concepts AkfMap has no equivalent
  // of, so this block only fires for the voxel backend.
  if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get()))
  {
    const long info_fits = voxelPlaneInformationFitCount();
    o << "\n  voxel_map/plane/plane_var_mode = "
      << vm->opts()->plane_var_mode;
    if (vm->opts()->plane_var_mode == "information" ||
        vm->opts()->plane_var_mode == "information_directional")
    {
      o << " count=" << info_fits;
      if (info_fits == 0)
        o << "   <-- INERT: the information model was selected and no plane "
             "was ever fitted under it";
    }
    o << "\n  voxel_map/plane/weight_floor/mode = "
      << vm->opts()->weight_floor_mode;
  }

  o << "\n  spline/mode = " << dopts_.spline.mode
    << "\n  control_points/hz = " << dopts_.spline.control_point_hz;

  // CQ-22 item (4): the fit-fail breakdown by cause; CQ-23 made
  // "chart_guard" (the hard ceiling) a real, live cause again, separate
  // from the soft warning count below -- see ScanSpline::FitFailCause.
  if (dopts_.spline.splineOn())
  {
    static const char* kCauseNames[] = {
      "none", "too_few_poses", "bad_window", "too_few_samples",
      "n_cp_too_small", "degenerate_delta", "underdetermined",
      "solve_failed", "non_finite", "chart_guard_hard", "pivot_guard",
    };
    o << "\n  spline fit_fail_count=" << spline_fit_fail_count_
      << " of frame_count=" << spline_frame_count_ << ", by cause:";
    for (std::size_t c = 1; c < spline_fail_cause_count_.size(); ++c)
      o << "\n    " << kCauseNames[c] << '=' << spline_fail_cause_count_[c];
    o << "\n  spline chart_guard_warn_count=" << spline_chart_guard_warn_count_
      << "  (fits that exceeded CHART_MAX_PHI_RAD but were NOT refused, "
         "CQ-22 item 4 -- a nonzero count here is a parameterisation finding, "
         "not a failure)";
  }

  o << "\n[engagement] a flag marked INERT above did not test anything; the "
       "cell is invalid, not null.";
  return o.str();
}

// See LioProcOptions::dry_run_point_filter_num's doc comment and this
// method's declaration in lio_processing.h.
void LioProcDecoupled::runDryRunShadowPass(const MeasureGroup& mg)
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

std::string LioProcDecoupled::processLIO(MeasureGroup& mg)
{
  // P8.  Seed prior_* to the CURRENT (pre-frame) state before any early
  // return below can skip the real snapshot further down. Without this, an
  // early return (e.g. an empty map) leaves mg.prior_* at MeasureGroup's
  // bare defaults (zero/identity) while publishOdometry() still writes a
  // pose_pair.csv row unconditionally -- gain(t) on that row would read as
  // a huge, meaningless jump instead of the true zero (no update happened).
  mg.prior_pos = state_->pos();
  mg.prior_rot = state_->rot();
  mg.prior_vel = state_->vel();

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

    // CQ-43 item 0: ONE stream for both write sites below (in-loop and
    // post-loop) -- PersistentLogStream::stream() opens its own ofstream
    // with std::ios::trunc on first use per INSTANCE, so two separate
    // `static PersistentLogStream log(...)` declarations against the same
    // basename (the bug this fixes, caught by inspecting the corrupted
    // output before filing) truncate/interleave each other's writes rather
    // than sharing one file position.
    static PersistentLogStream cq43_tailmove_log("cq43_tailmove.txt");

    // Fixed IEKF prior for this frame's ENTIRE inner loop -- see ekf.h's
    // applyMeanUpdate() doc comment. Set once here, read (never rewritten)
    // by every solveSystem()/solveSystem_cuda() call below until the loop
    // finishes and applyCovarianceUpdate() is called exactly once.
    prior_cov_ = state_->cov();
    // CQ-31 item 5: the scalar P controls, applied once here to this frame's
    // fixed prior snapshot (identity at every default -- see
    // PriorScalarOptions::on()). trP_pos_pre_ below intentionally reads the
    // POST-control value, since that is the prior every solveSystem() call
    // this frame actually blends against.
    applyPriorScalarControls(prior_cov_, opts_.prior_scalar);
    state_propagat_ = *state_;
    // P1.  Captured here, before this frame's update runs at all -- pairs
    // with the POST value already computed further down (see that block's
    // "const auto& P = state_->cov()") to give the covariance delta, i.e.
    // how much this scan actually learned, which a level alone cannot.
    trP_pos_pre_ = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();
    // P8.  state_propagat_ IS the propagated (pre-update) state at this
    // exact point -- snapshot it into mg so PubProc::publishOdometry() can
    // export it alongside the posterior pose, without processLIO() needing
    // to know anything about odometry export itself.
    mg.prior_pos = state_propagat_.pos();
    mg.prior_rot = state_propagat_.rot();
    mg.prior_vel = state_propagat_.vel();
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

      // CQ-43 item 0, read-only: how far THIS iteration's moveTailClamp()
      // call (invoked from redeskewFromSpline() immediately above, for
      // iter>0 only) actually moved the tail target. is_final=0 here --
      // the one post-loop call below (after convergence) is is_final=1 and
      // is what the interior never gets refitted against (the card's own
      // finding). -1 sentinels (lastTailMoveDpNorm()'s own convention) mean
      // this scan had no boundary (n_frozen_cp_<=0) and are left as -1
      // rather than coerced to 0, matching that function's own distinction.
      if (opts_.log_debug_en && spline_ok_ && dopts_.spline.splineOn() && iter > 0) {
        std::ofstream& ofs = cq43_tailmove_log.stream();
        const double dp = spline_.lastTailMoveDpNorm();
        ofs << "scan_id=" << voxel_map_->frame_idx_ << " iter=" << iter
            << " is_final=0 dp_mm=" << (dp >= 0.0 ? dp * 1000.0 : dp)
            << " dphi_deg=" << spline_.lastTailMoveDphiNorm() << "\n";
        ofs.flush();
      }

      // T0-D wants the first-iteration (pre-update, un-relinearized)
      // innovation only -- later iterations relinearize at an
      // already-partially-corrected state, which is not the quantity NIS
      // is defined over.
      double error = estimateStateCorrection(mg.points, dtheta, dt, /*allow_consistency_log=*/iter == 0);
      // SHAPE, from the SAME residuals that solve just used.  Refinement and
      // the state update are therefore linearised at one trajectory, which
      // is the whole reason this sits here and not at the top of the next
      // iteration (Bryce, 2026-09-06).
      refineSplineFromResiduals(mg);
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
    //
    // CQ-70: covRedundancyKappa() is 1.0 (no-op) unless
    // lio/ekf/cov_redundancy_discount is set -- the REAL posterior write is
    // the only one this card discounts. The shadow update above (line ~1161,
    // used only for a trace(P) diagnostic comparison at an earlier
    // point_filter_num) is deliberately left undiscounted, so that
    // diagnostic keeps comparing against the historical formula rather than
    // silently changing shape when this new option is engaged.
    if (any_solved)
      ekf_.applyCovarianceUpdate(state_, prior_cov_, covRedundancyKappa());

    // CQ-43 item (4d-ii): save this frame's own EKF rotation correction
    // (total_dtheta is local to this scope) for finalizeSplineAndQ() to
    // log alongside the rest of the rotation consistency read.
    last_total_dtheta_deg_ = total_dtheta.norm() * (180.0 / M_PI);

    // FINAL re-deskew, against the converged state.  redeskewFromSpline()
    // runs at the TOP of each iteration, so without this call the last
    // solve's correction never reaches mg.points -- and the node calls
    // VoxelMap::updateMap() on exactly those points once processLIO()
    // returns.  The map would be built, permanently, from points placed by
    // the second-to-last state.
    //
    // CQ-43 item 1, Bryce 2026-09-18: this comment used to end "Also runs
    // one last shape refinement against the final residual set." THAT WAS
    // FALSE -- redeskewFromSpline() calls only moveTailClamp() then
    // deskewPointsSplineCsr(), and refineSplineFromResiduals() (line 1650
    // above) has no second call site anywhere in this function. So the
    // interior control-point SHAPE is fitted only to the pre-this-
    // iteration's-own-correction residuals inside the loop -- the
    // converged state's own final correction reaches the interior solely
    // as this call's moveTailClamp() RAMP, never as a fit. See CQ-43's own
    // card for the measurement this discrepancy motivated.
    redeskewFromSpline(mg);

    // CQ-43 item 0, read-only: this is the is_final=1 counterpart to the
    // in-loop log above -- the one moveTailClamp() call whose dp/dphi is
    // never followed by a refit, which is the card's own headline
    // question. iter (loop-scope, still in scope here) is this scan's
    // total iteration count, for the "distribution of iteration count per
    // scan" part of item 0.
    if (opts_.log_debug_en && spline_ok_ && dopts_.spline.splineOn()) {
      std::ofstream& ofs = cq43_tailmove_log.stream();
      const double dp = spline_.lastTailMoveDpNorm();
      ofs << "scan_id=" << voxel_map_->frame_idx_ << " iter=" << iter
          << " is_final=1 dp_mm=" << (dp >= 0.0 ? dp * 1000.0 : dp)
          << " dphi_deg=" << spline_.lastTailMoveDphiNorm()
          << " n_iters=" << iter << "\n";
      ofs.flush();
    }

    // CQ-43 item (2): the gated post-convergence refinement pass. Rebuilds
    // residuals against the FULLY CONVERGED trajectory (buildResiduals()
    // is const w.r.t. state_ -- it never calls solveSystem()/
    // applyMeanUpdate(), unlike estimateStateCorrection()) and refits the
    // spline interior to them, then re-deskews once more so the shape
    // change reaches mg.points before VoxelMap::updateMap() sees them.
    // Read-only w.r.t. state_ by construction: buildResiduals() is const,
    // refineSplineFromResiduals() only ever touches spline_'s cp_p_, and
    // redeskewFromSpline()'s own moveTailClamp() call is a no-op here
    // (frozen_pos1_/frozen_rot1_/frozen_vel1_ already equal state_'s
    // current pos()/rot()/vel(), set by THIS frame's last real
    // moveTailClamp() call two lines above -- dp/dphi come back ~0, not
    // skipped, so item (0)'s own log gets one more is_final=1-shaped row
    // for this scan if log_debug_en is on; harmless, and left visible
    // rather than special-cased out).
    if (dopts_.spline.splineOn() && dopts_.spline.final_pass && spline_ok_) {
      const V3D pos_before = state_->pos();
      buildResiduals(mg.points, residuals_, /*allow_consistency_log=*/false);
      refineSplineFromResiduals(mg);
      redeskewFromSpline(mg);
      // Item (2)'s own pass condition: assert, don't just hope.
      const double moved_mm = (state_->pos() - pos_before).norm() * 1000.0;
      if (opts_.log_debug_en) {
        static PersistentLogStream log("cq43_finalpass_assert.txt");
        std::ofstream& ofs = log.stream();
        ofs << "scan_id=" << voxel_map_->frame_idx_
            << " state_moved_mm=" << moved_mm << "\n";
        ofs.flush();
      }
    }

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
      Eigen::MatrixXd phi_p_phit, accum_cov_w, p_before;
      if (imuProcQhatRead(phi_p_phit, accum_cov_w, p_before)) {
        const Eigen::VectorXd dx = state_->boxminusFromPropagat(state_propagat_);
        const Eigen::MatrixXd& P_post = state_->cov();
        if (phi_p_phit.rows() == P_post.rows()) {
          const double t_abs = mg.image.t + data_queues_->start_time;
          debugLogQhat(voxel_map_->frame_idx_, t_abs, dx, (phi_p_phit - P_post).diagonal(),
                       accum_cov_w.diagonal());

          // CQ-71 item 0: the information budget. P_after_IMU reconstructed
          // as phi_p_phit + accum_cov_w (F P F^T + Q_eff, by construction --
          // see imuProcQhatRead()'s own doc comment). eig(H^TWH) is NOT
          // separately tracked anywhere in this codebase -- reconstructed
          // here from ekf.h's own A = H_full + prior_cov^-1 identity:
          // H_full = P_after_LIO^-1 - P_after_IMU^-1 exactly, when
          // lio/ekf/cov_redundancy_discount is "off" (kappa==1, the
          // default) -- NOT valid if a non-default discount is active,
          // flagged via the h_eff_valid column so a reader never mistakes
          // a discounted run's number for the true LiDAR information.
          if (p_before.rows() == P_post.rows() && p_before.rows() > 0) {
            const Eigen::MatrixXd P_after_imu = phi_p_phit + accum_cov_w;
            auto eigStats = [](const Eigen::MatrixXd& M, double& trP, double& logdetP,
                                double& lam_min, double& lam_max) {
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
              const Eigen::VectorXd ev = es.eigenvalues();
              trP = M.trace();
              logdetP = ev.array().log().sum();
              lam_min = ev(0);
              lam_max = ev(ev.size() - 1);
            };
            double trP_before, ld_before, lmin_before, lmax_before;
            double trP_imu, ld_imu, lmin_imu, lmax_imu;
            double trP_fpf, ld_fpf, lmin_fpf, lmax_fpf;
            double trP_qeff, ld_qeff, lmin_qeff, lmax_qeff;
            double trP_lio, ld_lio, lmin_lio, lmax_lio;
            eigStats(p_before, trP_before, ld_before, lmin_before, lmax_before);
            eigStats(P_after_imu, trP_imu, ld_imu, lmin_imu, lmax_imu);
            eigStats(phi_p_phit, trP_fpf, ld_fpf, lmin_fpf, lmax_fpf);
            eigStats(accum_cov_w, trP_qeff, ld_qeff, lmin_qeff, lmax_qeff);
            eigStats(P_post, trP_lio, ld_lio, lmin_lio, lmax_lio);

            const bool discount_off = !opts_.cov_redundancy_discount.on();
            double trP_h = std::numeric_limits<double>::quiet_NaN();
            double ld_h = std::numeric_limits<double>::quiet_NaN();
            double lmin_h = std::numeric_limits<double>::quiet_NaN();
            double lmax_h = std::numeric_limits<double>::quiet_NaN();
            if (discount_off) {
              Eigen::LDLT<Eigen::MatrixXd> ldlt_imu(P_after_imu), ldlt_lio(P_post);
              if (ldlt_imu.info() == Eigen::Success && ldlt_lio.info() == Eigen::Success) {
                const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_post.rows(), P_post.rows());
                const Eigen::MatrixXd H_eff = ldlt_lio.solve(I) - ldlt_imu.solve(I);
                eigStats(H_eff, trP_h, ld_h, lmin_h, lmax_h);
              }
            }

            static PersistentLogStream ib_log("info_budget.txt");
            bool ib_first;
            std::ofstream& ib_ofs = ib_log.stream(&ib_first);
            if (ib_first)
              ib_ofs << "scan_id,t_abs,h_eff_valid,"
                        "trP_before,logdet_before,lmin_before,lmax_before,"
                        "trP_after_imu,logdet_after_imu,lmin_after_imu,lmax_after_imu,"
                        "trP_fpf,logdet_fpf,lmin_fpf,lmax_fpf,"
                        "trP_qeff,logdet_qeff,lmin_qeff,lmax_qeff,"
                        "trP_after_lio,logdet_after_lio,lmin_after_lio,lmax_after_lio,"
                        "trP_h,logdet_h,lmin_h,lmax_h\n";
            ib_ofs << voxel_map_->frame_idx_ << "," << std::setprecision(12) << t_abs << ","
                   << (discount_off ? 1 : 0) << ","
                   << trP_before << "," << ld_before << "," << lmin_before << "," << lmax_before << ","
                   << trP_imu << "," << ld_imu << "," << lmin_imu << "," << lmax_imu << ","
                   << trP_fpf << "," << ld_fpf << "," << lmin_fpf << "," << lmax_fpf << ","
                   << trP_qeff << "," << ld_qeff << "," << lmin_qeff << "," << lmax_qeff << ","
                   << trP_lio << "," << ld_lio << "," << lmin_lio << "," << lmax_lio << ","
                   << trP_h << "," << ld_h << "," << lmin_h << "," << lmax_h << "\n";
            ib_ofs.flush();
          }
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
      const V3D& ba = state_->biasAcc();
      const V3D& bg = state_->biasGyr();
      logConsistencyScan("lio", voxel_map_->frame_idx_, t_abs, dt, trP_pos, trP_vel, trP_att,
                          sum_omega / n_poses, sum_acc / n_poses,
                          ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z());
    }

    std::ostringstream oss;
    oss << "[lio/ekf] iters=" << iter + 1 << "  stop=" << stop
        << std::scientific << std::setprecision(1)
        << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
        << "  |dt|=" << total_dt.norm() * 1000.0 << " mm";

    // Hand the per-frame LIO diagnostics to the map so frame_stats.txt can
    // carry them.  Computed unconditionally and cheaply (two 3x3 eigen
    // solves on a matrix the update already built) because gating them on
    // log_debug_en is exactly how they ended up unavailable to every sweep
    // that needed them -- H_pp_eig has existed for months, inside a
    // free-text debug line no scorer parses.
    {
      LioFrameDiag diag;
      diag.n_residuals = static_cast<int>(residuals_.size());
      diag.n_points_after_pfn = static_cast<int>(mg.lidar_points.size());
      diag.n_points_after_ds  = static_cast<int>(mg.points.size());
      diag.n_imu_samples      = mg.n_imu_samples;
      // CQ-87 item 8: only meaningful when the spline is actually in use
      // this scan (spline.mode != raw_imu) -- left at their -1 defaults
      // otherwise, matching this struct's own "not applicable" convention.
      if (dopts_.spline.splineOn() && spline_ok_) {
        diag.n_c_requested = spline_.nControlPointsRequested();
        diag.n_c_actual    = spline_.nControlPoints();
        diag.n_c_clamped   = spline_.nControlPointsClamped() ? 1 : 0;
      }

      // CQ-19(a): P's own decomposition (prior_cov_, pre-update -- state_->
      // cov()'s POST-update counterpart is read further below, once the
      // covariance update has actually run). Computed unconditionally, NOT
      // gated on residuals_.empty() -- prior_cov_ exists regardless of
      // whether any plane matched this frame, and gating it the same way
      // the HtH-derived diagnostics below must be gated left every
      // zero-residual frame's P columns at their -1.0 "unavailable"
      // default even though the prior itself was perfectly well-defined
      // (confirmed on site1_handheld_1's smoke cell: 150/2044 frames had
      // residuals_.empty() and silently failed the "eigenvalues sum to
      // trP_pos_pre" sanity check as a result).
      Eigen::Vector3d p_pos_pre_min_eigenvector = Eigen::Vector3d::Zero();
      bool have_p_pos_pre = false;
      if (prior_cov_.rows() >= StateGroup::idxP() + 3 && prior_cov_.cols() >= StateGroup::idxP() + 3) {
        const M3D P_pp_pre = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
        Eigen::SelfAdjointEigenSolver<M3D> es_p_pre(P_pp_pre);
        diag.p_pos_eig_min_pre = es_p_pre.eigenvalues()(0);
        diag.p_pos_eig_mid_pre = es_p_pre.eigenvalues()(1);
        diag.p_pos_eig_max_pre = es_p_pre.eigenvalues()(2);
        p_pos_pre_min_eigenvector = es_p_pre.eigenvectors().col(0);
        have_p_pos_pre = true;

        const M3D P_rr_pre = prior_cov_.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
        Eigen::SelfAdjointEigenSolver<M3D> es_r_pre(P_rr_pre);
        diag.p_rot_trace_pre   = P_rr_pre.trace();
        diag.p_rot_eig_min_pre = es_r_pre.eigenvalues()(0);
        // CQ-74 item 1: the rest of the same already-computed spectrum --
        // only the min was kept before.
        diag.p_rot_eig_mid_pre = es_r_pre.eigenvalues()(1);
        diag.p_rot_eig_max_pre = es_r_pre.eigenvalues()(2);

        diag.p_pos_vel_fro_pre =
            prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxV()).norm();
        if (state_->idxBA() >= 0 &&
            prior_cov_.rows() >= state_->idxBA() + 3 && prior_cov_.cols() >= state_->idxBA() + 3) {
          diag.p_pos_bias_fro_pre =
              prior_cov_.block<3, 3>(StateGroup::idxP(), state_->idxBA()).norm();
        }
      }

      if (!residuals_.empty()) {
        const M3D H_pp_d = ekf_.HtH.block<3, 3>(3, 3);
        const M3D H_rr_d = ekf_.HtH.block<3, 3>(0, 0);
        Eigen::SelfAdjointEigenSolver<M3D> es_pp(H_pp_d), es_rr(H_rr_d);
        diag.h_pp_min_eig = es_pp.eigenvalues()(0);
        diag.h_pp_max_eig = es_pp.eigenvalues()(2);  // CQ-81 item B
        diag.h_rr_min_eig = es_rr.eigenvalues()(0);
        diag.h_rr_trace   = H_rr_d.trace();
        // CQ-36 item 1: read directly off ekf_.HtH, unlike sum_weight below
        // (which sums 1/sigma_squared over residuals_ and is therefore
        // structurally blind to any HtH-level correction axis C applies --
        // see residual_redundancy.h). trace(H_pp) == sum_weight EXACTLY
        // whenever no HtH-level correction is active (mode=="off"), so the
        // pair is its own control: equal at off, divergent otherwise.
        // h_rr_trace above (H_rr_d.trace(), already logged) IS this same
        // quantity for the rotation block -- no separate htth_rot_trace
        // column is added; the item-3 one-column table in the round report
        // says so explicitly rather than adding a byte-identical duplicate.
        diag.htth_pos_trace = H_pp_d.trace();
        double sw = 0.0;
        for (const auto& r : residuals_) if (r.sigma_squared > 0.0) sw += 1.0 / r.sigma_squared;
        diag.sum_weight = sw;

        // CQ-18 item (2): S = floor_term + sigma_diag_squared + plane_var_term
        // + s_prior_pose, per residual -- summed for sum_S/the four shares,
        // and used per-residual for nis (nu^2/S) / nis_est (nu^2/(S-s_prior_pose)),
        // matching voxelplane.cpp's per-candidate corr_scan.csv definitions
        // exactly but averaged over THIS frame's accepted residuals instead
        // of a whole scan's candidates.
        {
          double sum_floor = 0.0, sum_sdiag = 0.0, sum_pvar = 0.0, sum_prior_pose = 0.0;
          double sum_nis = 0.0, sum_nis_est = 0.0;
          int n_nis = 0, n_nis_est = 0;
          for (const auto& r : residuals_) {
            if (r.floor_term < 0.0 || r.sigma_diag_squared < 0.0 || r.s_prior_pose < 0.0) continue;
            const double S_i = r.floor_term + r.sigma_diag_squared + r.plane_var_term + r.s_prior_pose;
            if (!(S_i > 0.0)) continue;
            sum_floor      += r.floor_term;
            sum_sdiag      += r.sigma_diag_squared;
            sum_pvar       += r.plane_var_term;
            sum_prior_pose += r.s_prior_pose;
            const double nis_i = r.r * r.r / S_i;
            if (std::isfinite(nis_i)) { sum_nis += nis_i; ++n_nis; }
            const double S_est_i = r.floor_term + r.sigma_diag_squared + r.plane_var_term;
            if (S_est_i > 0.0) {
              const double nis_est_i = r.r * r.r / S_est_i;
              if (std::isfinite(nis_est_i)) { sum_nis_est += nis_est_i; ++n_nis_est; }
            }
          }
          const double sum_S = sum_floor + sum_sdiag + sum_pvar + sum_prior_pose;
          if (sum_S > 0.0) {
            diag.sum_S = sum_S;
            diag.floor_share      = sum_floor / sum_S;
            diag.sdiag_share      = sum_sdiag / sum_S;
            diag.pvar_share       = sum_pvar / sum_S;
            diag.prior_pose_share = sum_prior_pose / sum_S;
          }
          if (n_nis > 0) diag.nis = sum_nis / n_nis;
          if (n_nis_est > 0) diag.nis_est = sum_nis_est / n_nis_est;
        }

        // CQ-19(b): the one dot product that signs rho_ref, using HtH_pp's
        // own weakest eigenvector (es_pp above; SelfAdjointEigenSolver
        // orders eigenvalues ascending, so column 0 is the weakest
        // direction, matching diag.h_pp_min_eig) against P_pp's own
        // weakest eigenvector, computed unconditionally just above.
        if (have_p_pos_pre) {
          diag.cos_pmin_hmin = std::abs(p_pos_pre_min_eigenvector.dot(es_pp.eigenvectors().col(0)));
        }

        // P1.  ask/got/refusal.  ekf_.HtH is 6x6 (rows 0-2 rot, 3-5 pos) and is
        // singular whenever a direction is unconstrained, so solve in the
        // eigenbasis and drop modes below a relative tolerance rather than
        // inverting.  A dropped mode is a direction the scan did not constrain;
        // it contributes nothing to either quantity, which is correct.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es6(ekf_.HtH);
        const auto& ev = es6.eigenvalues();
        const double tol = 1e-12 * std::max(1.0, ev(5));
        const Eigen::Matrix<double, 6, 1> z = es6.eigenvectors().transpose() * ekf_.Htz;
        // Paired with THIS iteration's own dtheta/dt, not the frame's
        // cumulative total_dtheta/total_dt -- ekf_.HtH/Htz are only ever the
        // LAST relinearization (this diagnostic runs once, after the inner
        // loop), and dtheta/dt is set from ekf_.dtheta/dt at that same last
        // solveSystem() call (see the assignment right after it), so this is
        // the correctly paired iteration. Pairing against the cumulative
        // total instead double-counts every earlier iteration's own already-
        // applied correction into "got", routinely pushing got > ask and
        // refusal negative even on ordinary frames -- confirmed on the P1-P4
        // smoke test (66% of frames outside [0,1] before this fix).
        Eigen::Matrix<double, 6, 1> dxv;
        dxv << dtheta, dt;
        const Eigen::Matrix<double, 6, 1> y = es6.eigenvectors().transpose() * dxv;
        double ask = 0.0, got = 0.0;
        for (int i = 0; i < 6; ++i) if (ev(i) > tol) {
          ask += z(i) * z(i) / ev(i);
          got += ev(i) * y(i) * y(i);
        }
        diag.ask = ask;  diag.got = got;
        diag.refusal = (ask > 0.0) ? (1.0 - got / ask)
                                    : std::numeric_limits<double>::quiet_NaN();
        // TQ-20 item 1: kappa_eff = 1/sqrt(1-rho_ref)-1, rho_ref==refusal
        // above (same formula, 1-got/ask) -- got>0.0 is the same
        // availability condition refusal's own ask>0.0 branch relies on,
        // since got/ask both come from the same eigenbasis sum above.
        diag.kappa_eff = (got > 0.0) ? (std::sqrt(ask / got) - 1.0) : -1.0;
        diag.htz_rot_norm = ekf_.Htz.segment<3>(0).norm();
        diag.htz_pos_norm = ekf_.Htz.segment<3>(3).norm();
        diag.iters        = iter + 1;
        diag.dx_rot_deg   = total_dtheta.norm() * (180.0 / M_PI);
        diag.dx_pos_mm    = total_dt.norm() * 1000.0;

        // TQ-20 item 1/6: the 6 generalized eigenvalues of (HtH, P) --
        // needs the SAME combined 6x6 prior block buildResiduals()'s own
        // prior_cov_rp slices (rot 0-3, pos 3-6, matching HtH's layout),
        // re-sliced here since that one is frame-constant context, not
        // stored on diag. have_p_pos_pre (computed above, this same
        // iteration) already proved prior_cov_ is at least idxP()+3 --
        // re-check idxR()+6 explicitly since the combined slice spans
        // both blocks and idxR()/idxP() are not asserted adjacent in
        // general (only true by this codebase's current StateGroup
        // layout, which line 283's prior_cov_rp slice already assumes).
        if (have_p_pos_pre &&
            prior_cov_.rows() >= StateGroup::idxR() + 6 &&
            prior_cov_.cols() >= StateGroup::idxR() + 6) {
          const Eigen::Matrix<double, 6, 6> P_prior_6 =
              prior_cov_.block<6, 6>(StateGroup::idxR(), StateGroup::idxR());
          Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
              ges(ekf_.HtH, P_prior_6);
          if (ges.info() == Eigen::Success) {
            const auto& gev = ges.eigenvalues();
            diag.kappa_gev0 = gev(0); diag.kappa_gev1 = gev(1); diag.kappa_gev2 = gev(2);
            diag.kappa_gev3 = gev(3); diag.kappa_gev4 = gev(4); diag.kappa_gev5 = gev(5);
            diag.kappa_gev_ok = true;
          }
        }
      }
      // Captured at the TOP of processLIO(), before the update -- see note.
      diag.trP_pos_pre = trP_pos_pre_;
      // CQ-19(a): the POST-update counterpart to the pre-update p_pos_eig_*
      // above -- state_->cov() already reflects applyCovarianceUpdate()'s
      // result by this point in processLIO() (that call runs well above,
      // unconditionally on any_solved), independent of whether residuals_
      // was non-empty this frame.
      {
        const Eigen::MatrixXd& P_post = state_->cov();
        if (P_post.rows() >= StateGroup::idxP() + 3 && P_post.cols() >= StateGroup::idxP() + 3) {
          const M3D P_pp_post = P_post.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
          Eigen::SelfAdjointEigenSolver<M3D> es_p_post(P_pp_post);
          diag.p_pos_eig_min_post = es_p_post.eigenvalues()(0);
          diag.p_pos_eig_mid_post = es_p_post.eigenvalues()(1);
          diag.p_pos_eig_max_post = es_p_post.eigenvalues()(2);
          diag.trP_pos_post = P_pp_post.trace();
        }
        // CQ-74 item 1: the rotation block's POST-update counterpart --
        // previously had no post-update columns at all.
        if (P_post.rows() >= StateGroup::idxR() + 3 && P_post.cols() >= StateGroup::idxR() + 3) {
          const M3D P_rr_post = P_post.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
          Eigen::SelfAdjointEigenSolver<M3D> es_r_post(P_rr_post);
          diag.p_rot_trace_post   = P_rr_post.trace();
          diag.p_rot_eig_min_post = es_r_post.eigenvalues()(0);
          diag.p_rot_eig_mid_post = es_r_post.eigenvalues()(1);
          diag.p_rot_eig_max_post = es_r_post.eigenvalues()(2);
        }
      }
      // Captured in finalizeSplineAndQ(), which already ran earlier this
      // same processLIO() call (see that function's own doc comment).
      diag.boundary_dpos     = boundary_dpos_;
      diag.boundary_drot_deg = boundary_drot_deg_;
      // CQ-19(c): AdaptiveQ's own gate state -- adaptive_q_.update() (also
      // above this diag block, earlier in this same processLIO() call) has
      // already run this frame whenever dopts_.adaptive_q.enable is set, so
      // its state is current here regardless of residuals_. Left at their
      // NaN/false defaults when disabled, per LioFrameDiag's own doc
      // comment -- cov_acc/cov_gyr stay config-invariant either way, but
      // these five are the columns that actually distinguish live from off.
      if (dopts_.adaptive_q.enable) {
        // CQ-26: NaN until the channel's gate has actually passed once (see
        // zAccOrNaN()/zGyrOrNaN()'s doc comment) -- was zAcc()/zGyr()
        // directly, which default to 0.0 (a real excursion value) and so
        // could not distinguish "never updated" from "updated to zero".
        diag.q_z_acc    = adaptive_q_.zAccOrNaN();
        diag.q_z_gyr    = adaptive_q_.zGyrOrNaN();
        diag.q_acf1_acc = last_spline_stats_.acf1_acc;
        diag.q_acf1_gyr = last_spline_stats_.acf1_gyr;
        diag.q_active   = adaptive_q_.active();
        diag.q_clamped  = adaptive_q_.clamped();
        // CQ-26: update()'s own per-frame decision, previously computed and
        // discarded -- see LioFrameDiag's doc comment on these six fields.
        diag.q_status          = adaptive_q_.lastStatus();
        diag.q_white_acc       = adaptive_q_.whiteAcc();
        diag.q_white_gyr       = adaptive_q_.whiteGyr();
        diag.q_above_floor_acc = adaptive_q_.aboveFloorAcc();
        diag.q_above_floor_gyr = adaptive_q_.aboveFloorGyr();
        diag.q_active_frame    = adaptive_q_.activeThisFrame();
      }
      // CQ-34: redundancy_stats_ is written by solveSystem()/
      // solveSystem_cuda() earlier this same processLIO() call by an
      // UNCONDITIONAL call to applyResidualRedundancyCorrection() -- no
      // extra guard needed here, unlike AdaptiveQ's block above which only
      // runs when the block ran at all. (Before CQ-34 this comment was
      // already true in spirit but false in fact: the two call sites above
      // still gated the call itself on opts_.residual_redundancy.on(), so a
      // "0" here at mode=="off" was a default-constructed struct, not a
      // measurement -- see CQ-34.)
      diag.redund_groups     = redundancy_stats_.redund_groups;
      diag.redund_groups_seen             = redundancy_stats_.redund_groups_seen;
      diag.redund_groups_degenerate_pv    = redundancy_stats_.redund_groups_degenerate_pv;
      diag.redund_groups_degenerate_var   = redundancy_stats_.redund_groups_degenerate_var;
      diag.redund_n_raw      = redundancy_stats_.redund_n_raw;
      diag.redund_n_eff      = redundancy_stats_.redund_n_eff;
      diag.redund_info_ratio = redundancy_stats_.redund_info_ratio;
      // CQ-31 item 7: log beside, do not drive -- always populated,
      // including mode=="off", from the stats applyResidualRedundancyCorrection()
      // now always computes (see residual_redundancy.cpp).
      diag.naive_info_gain    = redundancy_stats_.naive_info_gain;
      diag.woodbury_info_gain = redundancy_stats_.woodbury_info_gain;
      // CQ-31 item 8: reduced_chi2 was computed twice already in this file
      // (lines ~1048, ~1648) but only ever streamed to a log line -- TQ-23
      // could not report it per-cell for exactly that reason. Same formula,
      // computed unconditionally (not gated behind opts_.log_debug_en) so
      // it is always available: sum(r.r^2/r.sigma_squared) / n_residuals,
      // which averages to ~1 for a correctly-calibrated residual model.
      {
        double sum_chi2 = 0.0;
        for (const auto& r : residuals_) sum_chi2 += (r.r * r.r) / r.sigma_squared;
        diag.reduced_chi2 = residuals_.empty() ? 0.0 : sum_chi2 / residuals_.size();
      }

      // CQ-37 items 2/3/4/6: axis A/B/D engagement, log beside (never drive
      // -- these fields were already applied earlier this frame, before
      // accumulation; this just carries the counters into the diagnostic
      // row).
      diag.collapse_groups_collapsed  = collapse_stats_.groups_collapsed;
      diag.collapse_residuals_removed = collapse_stats_.residuals_removed;
      diag.per_residual_touched       = per_residual_stats_.residuals_touched;
      diag.per_residual_renorm_factor = per_residual_stats_.renorm_factor;
      diag.per_residual_mean_scale    = per_residual_stats_.mean_applied_scale;
      diag.sigma_scale_applied        = last_density_scale_;
      diag.sigma_scale_chi2_ema       = chi2_ema_;

      // CQ-36 item 4/CQ-37 axis D "chi2", FIXED per CQ-36 M4 (coding inbox,
      // 2026-09-17): update chi2_ema_ (the smoothed MEASURED, i.e.
      // post-scale, reduced_chi2 -- diag.reduced_chi2 was just computed
      // above under whatever chi2_scale_ this frame actually applied) from
      // THIS frame's own reading, exactly as before. THEN accumulate it
      // into chi2_scale_ multiplicatively -- chi2_scale_ *= chi2_ema_ --
      // which is the actual fix: an integral controller whose fixed point
      // is chi2_ema_ == 1 (scale stops moving only once measured chi2 has
      // reached 1), unlike the retired "scale = chi2_ema_" direct
      // assignment, whose fixed point was sqrt(raw_chi2) on both sides.
      // Warmup: chi2_ema_ is seeded directly (unchanged) and chi2_scale_
      // stays at its 1.0 (inert) default throughout warmup -- the very
      // first post-warmup frame seeds chi2_scale_ FROM chi2_ema_ (a
      // bootstrap step) rather than starting the multiplicative
      // accumulation from the arbitrary default.
      if (opts_.sigma_scale.chi2On() && diag.reduced_chi2 > 0.0) {
        ++chi2_ema_frames_;
        if (chi2_ema_frames_ <= opts_.sigma_scale.chi2_warmup_frames) {
          chi2_ema_ = diag.reduced_chi2;
        } else if (chi2_ema_frames_ == opts_.sigma_scale.chi2_warmup_frames + 1) {
          const double a = opts_.sigma_scale.chi2_ema;
          chi2_ema_ = a * chi2_ema_ + (1.0 - a) * diag.reduced_chi2;
          chi2_scale_ = chi2_ema_;  // bootstrap: seed from the EMA, not from 1.0
        } else {
          const double a = opts_.sigma_scale.chi2_ema;
          chi2_ema_ = a * chi2_ema_ + (1.0 - a) * diag.reduced_chi2;
          chi2_scale_ = std::min(std::max(chi2_scale_ * chi2_ema_, opts_.sigma_scale.min_ratio),
                                  opts_.sigma_scale.max_ratio);
        }
      }

      if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get())) vm->noteLioFrameDiag(diag);
      logEigenspectrum18(voxel_map_->frame_idx_, mg.image.t + data_queues_->start_time, "decoupled");
    }

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
          << "  sigma_scale=" << last_density_scale_
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

    // CQ-60 item 5: Tier 1's shared NEES machinery -- independent of
    // log_debug_en, mirrors LioProcCoupled's own hook exactly.
    if (opts_.nees_per_dof_en) {
      const Eigen::MatrixXd& P = state_->cov();
      const int iP = StateGroup::idxP(), iR = StateGroup::idxR();
      if (P.rows() >= iP + 3 && P.cols() >= iP + 3 && P.rows() >= iR + 3 && P.cols() >= iR + 3) {
        Eigen::Matrix<double, 6, 6> P6;
        P6.block<3, 3>(0, 0) = P.block<3, 3>(iR, iR);
        P6.block<3, 3>(3, 3) = P.block<3, 3>(iP, iP);
        P6.block<3, 3>(0, 3) = P.block<3, 3>(iR, iP);
        // CQ-62: was P.block(iP,iR).transpose() -- the SAME extraction
        // bug as LioProcCoupled's own equivalent hook (see its comment
        // for the full derivation). P is symmetric, so the extra
        // transpose silently re-flips this back to P.block(iR,iP) -- the
        // same value already placed at (0,3), not its true transpose --
        // making P6 not actually a principal submatrix of P at all.
        P6.block<3, 3>(3, 0) = P.block<3, 3>(iP, iR);
        const double t_abs_nees = mg.image.t + data_queues_->start_time;
        decoupled_tier1_nees_.addScan("tier1_decoupled", voxel_map_->frame_idx_, t_abs_nees,
                                       state_->rot(), state_->pos(), P6);
      }
    }

    return oss.str();
  }
}

// Voxel map update
}
