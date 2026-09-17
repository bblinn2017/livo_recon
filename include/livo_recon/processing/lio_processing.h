#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/cuda/lio_cuda.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/lio/deskew.h"
#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/adaptive_q.h"
#include "livo_recon/lio/residual_redundancy.h"
#include "livo_recon/lio/residual_weighting.h"

#include <array>

namespace livo_recon
{

struct LioProcOptions
{
  int    max_iterations        = 5;
  double min_norm_dtheta       = 0.0;
  double min_norm_dt           = 0.0;
  double min_diff_error        = -1.0;

  // History (24-27): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-24
  bool   log_debug_en          = false;

  // History (30-50): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-30
  int    dry_run_point_filter_num = 0;

  // A per-residual reweighting mechanism (LioProc::applyResidualWeighting(),
  // LioProcOptions::residual_weighting) was investigated (2026-08) as a way
  // to correct for correlated/redundant residuals (many points hitting the
  // same plane, especially at ds_leaf_size=0) and removed entirely after
  // three different, independently-derived schemes were tried:
  //   - "plane_averaged": collapse each plane's residuals to one, at the
  //     centroid. Catastrophically diverged point_filter_num=1+
  //     ds_leaf_size=0 (0.028m -> 3.34m) and regressed the production
  //     baseline (+34%) -- discarded genuine within-plane spatial spread
  //     that constrains rotation, not just correlated noise.
  //   - "count_weighted": keep every residual, scale sigma_squared_i by its
  //     plane's group size (with and without a n_planes/n_residuals
  //     normalization to preserve the aggregate weight budget). Nearly
  //     identical catastrophic failure either way -- root cause was a
  //     dominant, high-point-count plane being the load-bearing source of
  //     POSITION constraint (H_pp_min_eig collapsed ~4-4.6x, H_rr/rotation
  //     untouched) in an otherwise-weak direction; any purely count-based
  //     discount starves that direction regardless of global rescaling.
  //   - "info_gain": direction-aware (Sherman-Morrison running-covariance
  //     downdate, not grouping-based) -- found and fixed a real bug where
  //     its running covariance was seeded from the filter's own
  //     accumulated state_->cov() instead of an uninformative per-frame
  //     prior, causing a runaway cross-frame feedback spiral. Fixing that
  //     (verified via temporary gain-distribution logging) still left it
  //     catastrophic on BOTH the dense config (2.47m) and the previously-
  //     rock-solid production baseline (3.64m, vs 3.73m before the fix --
  //     barely different).
  // All three, done correctly, converge on the same outcome: properly
  // discounting redundant residuals weakens LIO's correction below what's
  // needed to track this dataset's fast-motion segments. Working
  // hypothesis: the unweighted baseline's stability isn't explained by
  // genuinely-sufficient independent geometric information there -- it's
  // propped up by systematically overcounting correlated residuals as
  // independent, which happens to supply enough (statistically
  // unwarranted) extra correction strength to track the maneuver in
  // practice. Residual-correlation-correction was concluded NOT to be the
  // right lever for this problem.
  //
  // Two more variants tried later the same investigation, both also
  // removed after also degrading performance despite being more
  // principled than the three above:
  //   - "info_gain_derived" density_sigma_mode: reused the (corrected)
  //     Sherman-Morrison info_gain machinery, but PURELY to compute one
  //     aggregate per-frame scalar (n_residuals/n_eff, n_eff = sum of
  //     individual Sherman-Morrison information gains) applied UNIFORMLY
  //     to every residual -- unlike info_gain's original per-residual use,
  //     this can't cause direction-specific starvation. Still regressed
  //     (0.0254m -> 0.0316m, eee_01): the adaptive scale it computed
  //     averaged ~164x (vs the fixed density_sigma_ref=3000's ~3.4x),
  //     overshooting pfn3_baseline's own confidence level by ~20x and
  //     overdamping the correction.
  //   - "woodbury_plane_correction": grouped residuals by plane and
  //     properly marginalized each plane's shared plane_var_ fit
  //     uncertainty out of the group's joint noise model via the Woodbury
  //     identity, instead of count_weighted's ad hoc group-size scaling.
  //     Mathematically the most principled of all five schemes tried --
  //     still regressed both pfn1_ds000 (0.0254m -> 0.0287m) and
  //     pfn3_baseline (0.0245m -> 0.0292m), with reduced_chi2 ~0.45-0.52
  //     (overly conservative by the residual-level calibration standard)
  //     in both cases. A fifth independent confirmation of the same
  //     conclusion above.

  // CQ-28: re-lands the "woodbury_plane_correction" mechanism named in the
  // historical comment above, as a standalone config-gated mode -- see
  // residual_redundancy.h for the full derivation. "off" (default) means
  // this mechanism is never invoked at all, so it cannot perturb existing
  // behavior by even one ULP.
  ResidualRedundancyOptions residual_redundancy;

  // CQ-31 item 5: the scalar P controls, independent of residual_redundancy
  // above (these act on the prior directly, not the measurement update) --
  // see residual_redundancy.h's PriorScalarOptions. Identity at every
  // default.
  PriorScalarOptions prior_scalar;

  // CQ-37 axis A/B: residual-set reduction (collapse) and per-residual
  // reweight (per_residual) -- see residual_weighting.h for the full
  // derivation of both. Both default off/identity.
  ResidualWeightingOptions residual_weighting;

  // CQ-37 axis D: ONE global multiplicative scalar applied to every
  // residual's sigma_squared (and, item 1c, plane_var_term by the same
  // factor) BEFORE accumulation. SUBSUMES the former standalone
  // density_sigma_mode/density_sigma_ref (now sigma_scale.mode's three
  // "density_*" levels, unchanged shape -- see below) and CQ-36's proposed
  // standalone sigma_calibration_mode (now sigma_scale.mode's "chi2"
  // level): two separate keys would let two multiplicative scales on the
  // same quantity fight silently, so this rebuild keeps axis D as one
  // enum, exclusive by construction (CQ-37 item 1).
  //
  // Levels: "off" (default, identity). "density_linear"/"density_sqrt"/
  // "density_quadratic" -- x = n_residuals/sigma_scale.density_ref, scale
  // = x / sqrt(x) / x^2 respectively, clamped >= 1 (only ever INCREASES
  // sigma_squared -- this is the former density_sigma_mode's exact shape,
  // renamed). "info_gain_derived" -- one aggregate per-frame scalar
  // n_raw/n_eff from the PREVIOUS frame's residual-redundancy-correction
  // counters (LioProc::redundancy_stats_ -- axis D runs before
  // accumulation, so THIS frame's own n_raw/n_eff do not exist yet; using
  // last frame's is a one-frame lag, not a same-frame reuse, and is
  // flagged as such at its own call site), bounded by
  // [sigma_scale.min_ratio, sigma_scale.max_ratio]. "chi2" -- an EMA of
  // the previous frames' reduced_chi2 (LioProc::chi2_ema_), applied
  // DIRECTLY as the scale (reduced_chi2 < 1 means sigma_squared is too
  // LARGE, so the scale must be able to go BELOW 1 -- CQ-36 item 4b's
  // explicit correction against reusing density's max(1, .) clamp),
  // bounded the same way.
  struct SigmaScaleOptions
  {
    std::string mode = "off";
    double density_ref = 0.0;
    double min_ratio = 0.01;
    double max_ratio = 100.0;
    double chi2_ema = 0.9;
    int    chi2_warmup_frames = 20;

    bool densityOn() const {
      return mode == "density_linear" || mode == "density_sqrt" || mode == "density_quadratic";
    }
    bool infoGainDerivedOn() const { return mode == "info_gain_derived"; }
    bool chi2On() const { return mode == "chi2"; }
    bool on() const { return mode != "off"; }
  } sigma_scale;

  // History (151-159): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-151

  // History (161-165): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-161
  DeskewOptions deskew;

  // History (168-172): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-168
  // See VoxelOpts-style modes: "off" is a value of imu/ds/mode, not a magic
  // ds_leaf_size of 0.0.  dsOn() is the single place the question is asked.
  double ds_leaf_size = 0.15;
  std::string ds_mode = "first";
  // "off" is a value of the mode, so a run with downsampling disabled cannot
  // also carry a mode nothing reads. The two call sites in
  // deskewAndDownsample() ask this instead of testing ds_leaf_size > 0.
  bool dsOn() const { return ds_mode != "off"; }

  // ── Scan-spline trajectory + live process-noise estimation ──────────────
  // See include/livo_recon/lio/spline.h and adaptive_q.h for the full
  // rationale.  Both default OFF; with spline.enable false not one line of
  // either subsystem executes and the frame path is byte-identical to the
  // pre-change build.
  SplineOptions    spline;
  AdaptiveQOptions adaptive_q;

  // Keep the legacy total_dt^3 * var_acc inflation on each point's
  // covariance even when the spline is fitting the intra-scan trajectory.
  // Default false BECAUSE that term puts state->varAcc() -- the same
  // variable cov_w's accelerometer block is built from -- inside the
  // MEASUREMENT covariance, i.e. the same quantity on both sides of
  // S = H P H^T + R.  Leaving it on while AdaptiveQ measures and applies
  // that quantity would close a feedback loop through the estimator's own
  // measurement model.  Exposed so the reasoning can be A/B tested rather
  // than assumed.

  // History (176-191): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-176
  bool log_consistency_scan_en = false;

  // History (194-222): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-194
  bool log_nll_en = false;
};


class LioProc
{
public:
  explicit LioProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  // One line per toggle: what it was set to, how often its mechanism ran, and
  // how far it moved things.  A flag that was ON with a zero counter prints
  // INERT, which is the condition three separate batches had to discover the
  // expensive way -- SP-4a spent 208 jobs on a switch that was parsed,
  // printed and never read.  Written to engagement.txt in the debug log dir
  // and to the log; the sweep scorer reads the INERT lines directly as
  // validity failures rather than reconstructing them from CSV columns.
  std::string engagementReport() const;
  ~LioProc();

  // Stage 1 (see include/livo_recon/lio/deskew.h's module doc comment):
  // deskew + downsample, populating mg.points (PointXYZCov, for
  // buildResiduals()/VoxelMap::updateMap()). Called ONCE per frame, from
  // LivoReconNode::estimateState() right after imu_proc_.processIMU(mg)
  // -- BEFORE the sequential-vs-combined branch, deliberately NOT from
  // inside processLIO() itself, since CombinedProc's primary loop calls
  // accumulateForCombined() directly and never calls processLIO() at all
  // (only as its own fallback) -- both paths need mg.points already
  // populated, mirroring the guarantee ImuProc::undistortLidar()/
  // downsamplePoints() used to provide unconditionally before this move.
  void deskewAndDownsample(MeasureGroup& mg);

  // Re-place this frame's kept points against the spline after the spline
  // has been re-anchored to the corrected state.  Called at the top of
  // every IEKF inner iteration after the first, when
  // spline.redeskew_each_iteration is on.  No-op (returns false) if the
  // spline is not valid for this frame, so a failed fit degrades to the
  // legacy one-shot deskew rather than to garbage.
  bool redeskewFromSpline(MeasureGroup& mg);

  // Refine the spline's interior shape against the residuals the IEKF
  // just accumulated, before the same residuals are used to solve.  Both
  // clamps are frozen in the solve, so it owns the interior only.
  bool refineSplineFromResiduals(const MeasureGroup& mg);

  // Raw IMU + the CURRENT biases/gravity for the fit's optional acc/gyro term.
  // The samples pointer is left null unless a weight is actually non-zero, so
  // with the weights at their 0 default the fit never dereferences anything
  // and is bit-identical to the pose-only one.  Needs imu/keep_raw_samples,
  // the same prerequisite AdaptiveQ has; without it the term is skipped.

  // Fit the spline, and afterwards measure the IMU residual against it and
  // hand the result to AdaptiveQ.  Split from deskewAndDownsample() so the
  // measurement happens AFTER the frame's IEKF has converged and the
  // spline has been anchored to the final state -- measuring against the
  // propagated-only spline would fold the frame's own correction error
  // into the "noise".
  void finalizeSplineAndQ(MeasureGroup& mg);

  // allow_consistency_log: T0-D's corr.csv wants the FIRST-iteration
  // (pre-update, un-relinearized) innovation only -- true only from
  // processLIO()'s iter==0 call; every later re-linearizing iteration,
  // and every call from runDryRunShadowPass()'s throwaway shadow pass,
  // passes false. See VoxelMap::setAllowConsistencyLog()'s doc comment.
  void buildResiduals(
    const std::vector<PointXYZCov>& pts,
    std::vector<Residual>& residuals,
    bool allow_consistency_log = true) const;

  void solveSystem(const std::vector<Residual>& residuals) const;
  void solveSystem_cuda(const std::vector<Residual>& residuals) const;

  double estimateStateCorrection(
    const std::vector<PointXYZCov>& pts,
    V3D &dtheta,
    V3D &dt,
    bool allow_consistency_log = true);

  // Assumes deskewAndDownsample(mg) already ran this frame (mg.points
  // populated) -- see that method's doc comment for why this isn't called
  // internally here.
  std::string processLIO(MeasureGroup& mg);

  // History (270-279): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-270
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res);

private:
  // See LioProcOptions::dry_run_point_filter_num's doc comment. Called at
  // the top of processLIO(), before the real frame's own prior_cov_/
  // state_propagat_ are captured -- runs a full non-committing shadow IEKF
  // convergence pass against mg.dry_run_points (same starting state_, same
  // current voxel_map_ as the real frame is about to use), logs rich
  // per-frame diagnostics (mirroring processLIO()'s own [lio] debug block)
  // as `[lio_dryrun]` lines, then fully discards the result -- state_,
  // prior_cov_, state_propagat_ are all restored to their real values
  // before returning, and voxel_map_ is never written to (buildResiduals()
  // is read-only), so this has zero effect on the real trajectory. No-op if
  // opts_.dry_run_point_filter_num <= 0, !opts_.log_debug_en, or
  // mg.dry_run_points is empty.
  void runDryRunShadowPass(const MeasureGroup& mg);

  StateGroupPtr state_;
  MapBackendPtr voxel_map_;
  ProfilerPtr profiler_;
  DataQueuesPtr data_queues_;  // for start_time -- see debugLogLio()'s absolute timestamps

  LioProcOptions opts_;

  // T0-D scan.csv's dt column -- previous logged scan's t_abs, -1 before
  // the first logged scan (dt written as 0 that first time).
  mutable double last_scan_t_abs_ = -1.0;

  std::vector<Residual> residuals_;

  // ── Scan-spline state, all per-frame ────────────────────────────────────
  ScanSpline spline_;
  bool       spline_ok_ = false;      // fit succeeded for THIS frame
  // Raw-cloud indices of the points that survived the Stage-1 downsample,
  // captured once per frame so the per-iteration re-deskew can go back to
  // each kept point's raw LiDAR-frame coordinates.  Empty only when the
  // spline is off or spline/redeskew_each_iteration is false -- ds_mode no
  // longer matters, which is the whole point of the CSR shape.
  //
  // CSR membership set from the voxel downsample: the raw-point indices
  // behind output point i are ds_members_[ds_offsets_[i] .. ds_offsets_[i+1]).
  // ds_offsets_.size() == mg.points.size() + 1 when it is live.  Populated in
  // BOTH ds_modes -- in FIRST every cell has one member, in AVERAGE the cell
  // is the source set the averaged point was built from.
  std::vector<int> ds_offsets_;
  std::vector<int> ds_members_;
  // Scratch for the control-point refinement, reused across iterations and
  // frames so the per-iteration path allocates nothing.
  std::vector<SplineLidarObs> lidar_obs_;

  // The pose sequence EXACTLY as ImuProc::propagate() produced it, kept only
  // when spline.reintegrate_each_iteration is on.  Every replay starts from
  // this, never from the previous replay: the bias delta is always measured
  // against the propagation-time bias (state_propagat_, whose bias blocks
  // propagate() never touches), so replaying a replay would apply the
  // correction twice.
  std::vector<Pose6D> spline_poses_;    // replay target, reused
  int  spline_refits_ = 0;              // per frame, for spline_q.csv

  // The bias/gravity THIS FRAME's IMU propagation actually started from,
  // captured once at the top of deskewAndDownsample() (before the IEKF
  // loop -- and therefore before anything corrects state_'s bias/gravity
  // this frame) and read by every splineImuFitData() call for the rest of
  // the frame. Same value state_propagat_'s bias/gravity blocks hold once
  // the loop starts (propagate() never touches them either), captured
  // earlier because deskewAndDownsample()'s own fit() call runs before
  // state_propagat_ is assigned for this frame -- see splineImuFitData()'s
  // doc comment for why using state_-> directly here would be wrong.
  V3D spline_frame_bias_acc_ = V3D::Zero();
  V3D spline_frame_bias_gyr_ = V3D::Zero();
  V3D spline_frame_gravity_  = V3D::Zero();

  // ── Engagement counters ─────────────────────────────────────────────────
  // A flag that is on must increment something observable, and where it moves
  // a quantity it must record how far -- an acceptance count alone cannot
  // separate "ran and did nothing" from "ran and mattered".  These are the
  // per-frame magnitudes that were missing when SP-4a"/4b" had to reconstruct
  // its validity gates from two columns by hand.  Reset per frame; the _run_
  // totals below are what the end-of-run INERT report reads.
  int    redeskew_calls_ = 0;           // per frame
  double redeskew_dp_rms_ = 0.0;        // m, RMS point move across a re-deskew
  std::vector<PointXYZCov> redeskew_prev_;   // scratch, for the displacement

  // P3(a).  How far the refit (refine+reintegrate+anchorTo, whatever subset
  // is on) actually moved the trajectory itself, sampled on a fixed 32-point
  // grid across the scan window -- refine_dcp_max/_rms are a control-point
  // proxy for this, not the quantity itself. Reset per frame, max across
  // this frame's iterations, same convention as redeskew_dp_rms_ above.
  double refit_dtraj_rms_ = 0.0;   // m
  double refit_dtraj_max_ = 0.0;   // m
  double refit_drot_deg_  = 0.0;   // deg
  // P3(b).  The IMU-residual variance the PRE-refit spline implies, so the
  // delta against finalizeSplineAndQ()'s post-refit last_spline_stats_ (the
  // existing cov_acc_meas/cov_gyr_meas columns) is readable at all.
  double cov_acc_pre_ = 0.0;
  double cov_gyr_pre_ = 0.0;

  // Run totals, for the engagement report.  A flag that was on with a zero
  // total here is an INERT line and, downstream, an invalid sweep cell.
  long   run_redeskew_calls_ = 0;
  long   run_refine_applied_ = 0;
  long   run_refine_rejects_ = 0;
  long   run_refits_ = 0;
  double run_refine_dcp_max_ = 0.0;
  double run_redeskew_dp_max_ = 0.0;
  long   run_aq_ok_frames_ = 0;
  double run_aq_applied_min_acc_ = 0.0, run_aq_applied_max_acc_ = 0.0;
  double run_aq_applied_min_gyr_ = 0.0, run_aq_applied_max_gyr_ = 0.0;
  long   run_frames_ = 0;

  // adaptive_q/noise_floor/mode, kept as a string so the resolver can
  // validate it; AdaptiveQOptions::use_noise_floor is derived from it.
  std::string adaptive_q_floor_mode_ = "allan";
  AdaptiveQ  adaptive_q_;
  bool       adaptive_q_primed_ = false;
  SplineImuResidualStats last_spline_stats_;
  int        spline_fit_fail_count_ = 0;
  int        spline_frame_count_ = 0;
  // CQ-22 item (4): one counter per ScanSpline::FitFailCause, indexed by
  // static_cast<int>(cause) -- spline_fit_fail_count_ above stays as the
  // pre-existing aggregate (unchanged reader compatibility); this is the
  // per-cause breakdown printed in engagementReport(). kNone's slot (index
  // 0) is never incremented.
  std::array<int, 11> spline_fail_cause_count_{};   // CQ-24: kPivotGuard added, 10 -> 11
  int        spline_chart_guard_warn_count_ = 0;
  mutable std::vector<std::vector<Residual>> build_thread_residuals_;

  // Per-thread miss classification for points where findPlaneResidual()
  // failed entirely (all 3 tiers exhausted) -- see VoxelMap::
  // hasConvergedNeighbor()'s docs. [i][0] = coverage-gap misses (no
  // converged plane anywhere in this point's voxel neighborhood), [i][1] =
  // mismatch misses (a converged plane exists nearby, but no candidate's
  // gate accepted this point -- a geometric/pose-offset problem, not a
  // map-density one). Debug-only, for the fast-motion tier0/1 hit-rate
  // investigation vs FAST-LIVO2.
  mutable std::vector<std::array<int, 2>> build_thread_miss_;
  // Frame totals, summed from build_thread_miss_ after buildResiduals() --
  // see its docs above. Read by lio_processing.cpp's debug log line.
  mutable int n_miss_coverage_ = 0;
  mutable int n_miss_mismatch_ = 0;

  // Tier0 (primary voxel)-specific miss classification -- see VoxelMap::
  // findPlaneResidual()'s tier0_had_plane out-param. [i][0] = tier0 missed,
  // primary voxel had no converged plane (coverage gap at tier0 itself);
  // [i][1] = tier0 missed, primary voxel DID have a converged plane (this
  // point's own projection just failed its gate -- geometric/pose-offset
  // mismatch), counted regardless of whether tier1/tier2 later rescue the
  // point. Unlike n_miss_coverage_/n_miss_mismatch_ above (which only see
  // points that fail ALL tiers), this isolates tier0's own hit-rate drop
  // specifically, since tier2's box search was found to mask most of the
  // signal by the time a point reaches the whole-pipeline classification.
  mutable std::vector<std::array<int, 2>> build_thread_tier0_miss_;
  mutable int n_tier0_miss_coverage_ = 0;
  mutable int n_tier0_miss_mismatch_ = 0;

  // Last-computed axis-D (sigma_scale) scale, for debug logging only -- see
  // LioProcOptions::SigmaScaleOptions docs. Named for its density-mode
  // origin; now shared by every sigma_scale.mode level.
  mutable double last_density_scale_ = 1.0;

  // CQ-37 axis D "chi2" level's cross-frame state.
  //
  // CQ-36 M4 (coding inbox, 2026-09-17): the FIRST version of this made
  // chi2_ema_ the applied scale directly (scale = chi2_ema_). That closes
  // the loop on the wrong variable: chi2_ema_ is an EMA of MEASURED
  // (post-scale) reduced_chi2, and scaling sigma_squared by s makes the
  // next measurement chi2/s -- so "scale = last measured chi2" has its
  // fixed point where scale = raw_chi2/scale, i.e. scale* = sqrt(raw_chi2),
  // NOT raw_chi2 itself, and the measured chi2 at that fixed point is also
  // sqrt(raw_chi2), never 1. Confirmed empirically: both verification
  // cells converged the scale and the reported chi2 to the SAME value
  // (~0.49-0.50) instead of chi2 -> 1.
  //
  // FIX: chi2_scale_ is now a SEPARATE, persistent, multiplicatively-
  // ACCUMULATED state -- an integral controller, not a direct assignment.
  // Each frame: chi2_ema_ (the smoothed MEASURED reduced_chi2 under the
  // CURRENTLY applied chi2_scale_) updates as before; then
  // chi2_scale_ *= chi2_ema_. At the fixed point chi2_ema_ == 1 exactly
  // (scale stops changing only once measured chi2 has reached 1), which is
  // the loop this mode's whole premise requires. chi2_ema_ is seeded
  // directly from measurements during warmup (unchanged); chi2_scale_
  // stays at its identity default (1.0, inert) through warmup and only
  // starts accumulating once warmup ends, seeded from chi2_ema_ at that
  // instant so the first post-warmup step is not a jump from the default.
  mutable double chi2_ema_ = 1.0;
  mutable double chi2_scale_ = 1.0;
  mutable int    chi2_ema_frames_ = 0;

  // Applies opts_.sigma_scale to `residuals` in place (sigma_squared AND,
  // per item 1c, plane_var_term by the same factor) -- called once per
  // frame from both estimateStateCorrection() and accumulateForCombined(),
  // BEFORE accumulation, replacing what used to be two duplicated
  // density-only blocks at those two call sites. No-op when
  // opts_.sigma_scale.mode == "off".
  void applySigmaScale(std::vector<Residual>& residuals) const;

  // Axis A/B (residual_weighting) engagement/magnitude, this frame -- read
  // into LioFrameDiag alongside redundancy_stats_ below.
  mutable CollapseStats collapse_stats_;
  mutable PerResidualStats per_residual_stats_;

  mutable EkfUpdate ekf_;

  // P1.  trP_pos BEFORE this frame's update runs, captured at the top of
  // processLIO() -- paired with the existing POST value (state_->cov() read
  // later in the same call) to give the covariance delta.
  double trP_pos_pre_ = -1.0;

  // DX-2 preflight fix (boundary_dpos).  Consecutive scans' spline windows
  // abut in time, so the last (t1) sample of one scan's finalized spline
  // and the first (t0) sample of the next MEASURE the inter-scan
  // discontinuity -- nothing had ever computed this before (the dense
  // trajLogOn() dump captures it implicitly but is off by default and
  // costly; this is 2 spline evaluations per scan instead). Saved in
  // finalizeSplineAndQ() right after that scan's own anchorTo() call, so
  // t1's pose is exactly (mg.image.t, state_->rot(), state_->pos()) at
  // that instant -- no extra evaluation needed for the outgoing half.
  bool  prev_scan_end_valid_ = false;
  V3D   prev_scan_end_pos_   = V3D::Zero();
  M3D   prev_scan_end_rot_   = M3D::Identity();
  // This scan's own boundary_dpos/drot (vs the PREVIOUS scan's end),
  // -1 if unavailable (first spline scan, or previous scan's spline
  // failed) -- read by debugLogFrameStats() the same frame it's computed.
  double boundary_dpos_ = -1.0;
  double boundary_drot_deg_ = -1.0;

  // Fixed IEKF prior (mean + covariance), snapshotted ONCE per frame (top
  // of processLIO()'s inner iteration loop, before any solveSystem()/
  // solveSystem_cuda() call) and reused, unchanged, by every one of that
  // frame's calls -- see ekf.h's applyMeanUpdate()/applyCovarianceUpdate()
  // doc comments for why re-reading/rewriting state_->cov() every iteration
  // (the old applyUpdate() behavior) was a bug. Only ever read (never
  // written) from solveSystem()/solveSystem_cuda(), both const methods, so
  // these don't need to be mutable.
  Eigen::MatrixXd prior_cov_;
  StateGroup state_propagat_;

  bool cuda_enable_ = false;
  mutable LioCudaBuffers cuda_buf_;

  // CQ-28: this frame's residual-redundancy-correction engagement/magnitude
  // counters, written by solveSystem()/solveSystem_cuda() (both const) right
  // after accumulate[Cuda]() when opts_.residual_redundancy.on() -- same
  // mutable-for-a-const-method precedent as cuda_buf_ above. Read into
  // LioFrameDiag alongside AdaptiveQ's own per-frame state.
  mutable ResidualRedundancyStats redundancy_stats_;
};

}
