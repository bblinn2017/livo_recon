#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/cuda/lio_cuda.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/lio/deskew.h"
#include "livo_recon/lio/residual_redundancy.h"
#include "livo_recon/lio/residual_weighting.h"

#include <array>
#include <initializer_list>
#include <string>
#include <vector>

namespace livo_recon
{

class ConfigResolver;

// CQ-49: the options genuinely SHARED between the decoupled and coupled
// estimators -- residual matching, the EKF accumulation, the redundancy/
// weighting/sigma-scale axes, deskew/downsample. spline/adaptive_q moved
// OUT to LioProcDecoupledOptions (lio_decoupled.h): a coupled config may
// carry NO spline/* or adaptive_q/* key at all (requireCombination in
// LioProcCoupled::loadParameters() enforces this), and object identity
// -- which loadParameters() override even LOOKS for those keys -- is what
// makes that enforcement structural rather than a runtime flag (item 3d).
struct LioProcOptions
{
  int    max_iterations        = 5;
  double min_norm_dtheta       = 0.0;
  double min_norm_dt           = 0.0;
  double min_diff_error        = -1.0;

  // History (24-27): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-24
  bool   log_debug_en          = false;

  // CQ-46: dumps raw per-accepted-residual fields (world point, normal,
  // rotation-Jacobian column, plane_id, r, and the four sigma_squared
  // components) to cq46_residuals.txt, and the frame-constant 6x6 prior
  // pose covariance block to cq46_prior.txt, on the FIRST IEKF iteration of
  // every scan only (allow_consistency_log's own gate -- see buildResiduals()'s
  // call sites). Read-only diagnostic: nothing it writes is read back by any
  // estimator path. Default off; the correlation binning/analysis itself is
  // done post-hoc in Python over these two files, not in this build.
  bool   log_pair_corr_en      = false;

  // History (30-50): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-30
  int    dry_run_point_filter_num = 0;

  // CQ-60 item 5: the shared per-dof NEES logger (utils/eval/nees_logger.h)
  // -- report-only, default false, shared by both estimator paths (unlike
  // most of this struct's coupled-vs-decoupled-specific fields, this one
  // key is genuinely common, hence read once here via loadSharedParameters()
  // rather than duplicated into each path's own options struct).
  bool nees_per_dof_en = false;
  // Tier 1's window size in scans -- default 489, matching the bag's own
  // 48.9s stationary prefix (the same window CQ-55 item 10's Allan-
  // deviation figure and this card's own item 2 both already use).
  int nees_tier1_window_scans = 489;

  // A per-residual reweighting mechanism (LioProcBase::applyResidualWeighting(),
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
  // counters (LioProcBase::redundancy_stats_ -- axis D runs before
  // accumulation, so THIS frame's own n_raw/n_eff do not exist yet; using
  // last frame's is a one-frame lag, not a same-frame reuse, and is
  // flagged as such at its own call site), bounded by
  // [sigma_scale.min_ratio, sigma_scale.max_ratio]. "chi2" -- an EMA of
  // the previous frames' reduced_chi2 (LioProcBase::chi2_ema_), applied
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

  // History (176-191): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-176
  bool log_consistency_scan_en = false;

  // History (194-222): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-194
  bool log_nll_en = false;
};


// CQ-49: the abstract interface + the genuinely-shared residual/EKF
// machinery. THIN by design (rule: inheritance for dispatch, composition
// for the shared half would be even safer, but the shared methods below
// are NON-VIRTUAL -- nothing can override buildResiduals()/solveSystem()/
// applySigmaScale()/accumulateForCombined(), so a derived class cannot
// silently break the decoupled path's byte-identity through a well-meaning
// partial override, which is the same protection the card's own
// ResidualEngine-by-composition design gives, without a second indirection
// layer). LioProcDecoupled and LioProcCoupled each own ALL of their own
// estimator-specific state; neither may hold a member the other also
// holds (verified: no coupled_*/spline_* member exists on this class).
class LioProcBase
{
public:
  explicit LioProcBase(NodeContext& ctx);
  virtual ~LioProcBase() = default;

  virtual std::string loadParameters(ros::NodeHandle& pnh) = 0;

  // One line per toggle: what it was set to, how often its mechanism ran, and
  // how far it moved things. Written to engagement.txt in the debug log dir
  // and to the log.
  virtual std::string engagementReport() const = 0;

  // Stage 1 (see include/livo_recon/lio/deskew.h's module doc comment):
  // deskew + downsample, populating mg.points. Called ONCE per frame, from
  // LivoReconNode::estimateState() right after imu_proc_.processIMU(mg).
  // VIRTUAL: this is the OTHER place the spline leaks into coupled mode if
  // left non-virtual on the base (CQ-44 bug 4's second consequence) --
  // LioProcDecoupled fits the spline here and deskews against it when the
  // fit succeeds; LioProcCoupled deskews from the raw IMU chain only, never
  // referencing a spline at all.
  virtual void deskewAndDownsample(MeasureGroup& mg) = 0;

  // Assumes deskewAndDownsample(mg) already ran this frame (mg.points
  // populated) -- see that method's doc comment for why this isn't called
  // internally here.
  virtual std::string processLIO(MeasureGroup& mg) = 0;

  // allow_consistency_log: T0-D's corr.csv wants the FIRST-iteration
  // (pre-update, un-relinearized) innovation only. NON-VIRTUAL: shared
  // machinery, identical for both estimators (item 1 of the split card).
  void buildResiduals(
    const std::vector<PointXYZCov>& pts,
    std::vector<Residual>& residuals,
    bool allow_consistency_log = true) const;

  void solveSystem(const std::vector<Residual>& residuals) const;
  void solveSystem_cuda(const std::vector<Residual>& residuals) const;

  // Used only by the decoupled estimator's own iteration loop, but the
  // residual-build + EKF-accumulate machinery it wraps is shared, so it
  // lives here rather than being duplicated. NOT called by the coupled
  // estimator (item 3's own assertion: estimateCoupledCorrection() is the
  // only update the coupled path ever takes).
  double estimateStateCorrection(
    const std::vector<PointXYZCov>& pts,
    V3D &dtheta,
    V3D &dt,
    bool allow_consistency_log = true);

  // History (270-279): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-270
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res);

protected:
  // Applies opts_.sigma_scale to `residuals` in place (sigma_squared AND,
  // per item 1c, plane_var_term by the same factor) -- called once per
  // frame from both estimateStateCorrection() and accumulateForCombined(),
  // BEFORE accumulation. No-op when opts_.sigma_scale.mode == "off".
  void applySigmaScale(std::vector<Residual>& residuals) const;

  // CQ-49 item 1: the shared half of loadParameters(), extracted once and
  // called from both derived classes' own overrides -- see lio_base.cpp's
  // own doc comment for exactly which keys this reads.
  void loadSharedParameters(ConfigResolver& cfg, ros::NodeHandle& pnh);
  // The shared finish: refuseUnclaimed(allowed_unclaimed_namespaces), the
  // build/accumulation_precision derived line, the ok()-or-throw gate, and
  // returning cfg.report().
  std::string finalizeConfig(ConfigResolver& cfg, std::initializer_list<const char*> allowed_unclaimed_namespaces);

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
  // map-density one).
  mutable std::vector<std::array<int, 2>> build_thread_miss_;
  mutable int n_miss_coverage_ = 0;
  mutable int n_miss_mismatch_ = 0;

  // Tier0 (primary voxel)-specific miss classification -- see VoxelMap::
  // findPlaneResidual()'s tier0_had_plane out-param.
  mutable std::vector<std::array<int, 2>> build_thread_tier0_miss_;
  mutable int n_tier0_miss_coverage_ = 0;
  mutable int n_tier0_miss_mismatch_ = 0;

  // Last-computed axis-D (sigma_scale) scale, for debug logging only.
  mutable double last_density_scale_ = 1.0;

  // CQ-37 axis D "chi2" level's cross-frame state.
  //
  // CQ-36 M4 (coding inbox, 2026-09-17): the FIRST version of this made
  // chi2_ema_ the applied scale directly (scale = chi2_ema_). That closes
  // the loop on the wrong variable: chi2_ema_ is an EMA of MEASURED
  // (post-scale) reduced_chi2, and scaling sigma_squared by s makes the
  // next measurement chi2/s -- so "scale = last measured chi2" has its
  // fixed point where scale = raw_chi2/scale, i.e. scale* = sqrt(raw_chi2),
  // NOT raw_chi2 itself.
  //
  // FIX: chi2_scale_ is a SEPARATE, persistent, multiplicatively-
  // ACCUMULATED state -- an integral controller, not a direct assignment.
  mutable double chi2_ema_ = 1.0;
  mutable double chi2_scale_ = 1.0;
  mutable int    chi2_ema_frames_ = 0;

  // Axis A/B (residual_weighting) engagement/magnitude, this frame.
  mutable CollapseStats collapse_stats_;
  mutable PerResidualStats per_residual_stats_;

  mutable EkfUpdate ekf_;

  // P1. trP_pos BEFORE this frame's update runs, captured at the top of
  // processLIO() -- paired with the POST value (state_->cov(), read later
  // in the same call) to give the covariance delta.
  double trP_pos_pre_ = -1.0;

  // Fixed IEKF prior (mean + covariance), snapshotted ONCE per frame and
  // reused, unchanged, by every one of that frame's calls -- see ekf.h's
  // applyMeanUpdate()/applyCovarianceUpdate() doc comments for why re-
  // reading/rewriting state_->cov() every iteration was a bug.
  Eigen::MatrixXd prior_cov_;
  StateGroup state_propagat_;

  bool cuda_enable_ = false;
  mutable LioCudaBuffers cuda_buf_;

  // CQ-28: this frame's residual-redundancy-correction engagement/magnitude
  // counters, written by solveSystem()/solveSystem_cuda() (both const) right
  // after accumulate[Cuda]() when opts_.residual_redundancy.on().
  mutable ResidualRedundancyStats redundancy_stats_;
};

}  // namespace livo_recon
