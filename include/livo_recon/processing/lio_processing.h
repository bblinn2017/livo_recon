#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/cuda/lio_cuda.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/lio/deskew.h"

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

  // History (115-132): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-115
  double density_sigma_ref = 0.0;

  // density_sigma_mode: shape of the density_sigma_ref scale curve as a
  // function of x = n_residuals/density_sigma_ref (all clamped to >= 1, so
  // this only ever INCREASES sigma_squared, never shrinks it below the unscaled
  // baseline): "linear" (x, the default/original form), "sqrt" (sqrt(x) --
  // gentler growth, damps high-density frames less aggressively than
  // linear, tried to see if it lets rotation's H_rr recover more without
  // needing an extreme (low) density_sigma_ref that risks instability),
  // "quadratic" (x^2 -- steeper growth, tried to see if rotation's lagging
  // recovery just needs stronger damping at the SAME density_sigma_ref
  // rather than a different reference value). An "info_gain_derived" mode
  // (adaptive per-frame scale via the Sherman-Morrison machinery) and a
  // separate woodbury_plane_correction option (per-plane Woodbury
  // marginalization) were also tried and removed -- see the historical
  // doc comment above for both.
  std::string density_sigma_mode = "linear";

  // History (151-159): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-151

  // History (161-165): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-161
  DeskewOptions deskew;

  // History (168-172): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-168
  double ds_leaf_size = 0.15;
  std::string ds_mode = "first";

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

  // Last-computed density_sigma_ref scale, for debug logging only -- see
  // LioProcOptions::density_sigma_ref/mode docs.
  mutable double last_density_scale_ = 1.0;

  mutable EkfUpdate ekf_;

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
};

}
