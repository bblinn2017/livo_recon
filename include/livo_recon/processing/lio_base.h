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

  // CQ-74 item 5a: full 18-dim eigenspectrum of the POST-update covariance,
  // both estimator paths (shared here rather than duplicated in each --
  // see LioProcBase::logEigenspectrum18()). Default false, report-only,
  // never read back by any estimator path.
  bool   log_eigenspectrum_en  = false;

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

  // R61: raw state + full covariance trace per scan (diagnostics/state_trace.h). Write-only; default ON.
  bool state_trace_en = true;
  std::string state_trace_run_id = "run";

  // R61: open-loop IMU propagation. "off" (default) = normal operation, bit-identical to before.
  // "propagate_only" skips the LiDAR solve entirely (state and covariance are only propagated by the IMU);
  // reset_period_s > 0 resets the state to the stationary reference and the covariance to its first-frame P0
  // every that many seconds, giving several independent drift windows in one run. 0 = never reset.
  struct OpenLoopOptions
  {
    std::string mode = "off";
    double reset_period_s = 0.0;
  } open_loop;

  // The residual model has one independent control and one direct
  // per-VoxelPlane covariance marginalization. Failed collapse, count, and
  // conditioning heuristics were removed; see docs/RESIDUAL_MODE_AUDIT.md.
  ResidualRedundancyOptions residual_redundancy;

  // CQ-31 item 5: the scalar P controls, independent of residual_redundancy
  // above (these act on the prior directly, not the measurement update) --
  // see residual_redundancy.h's PriorScalarOptions. Identity at every
  // default.
  PriorScalarOptions prior_scalar;

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
// accumulateForCombined(), so a derived class cannot
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
    bool allow_consistency_log = true,
    bool include_state_uncertainty_in_gate = true,
    const std::vector<M3D>* pose_cov_overrides = nullptr) const;

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

  void setDiagnosticGnIteration(int iter) { diagnostic_gn_iteration_ = iter; }

protected:
  virtual const char* firstFrameArchitectureName() const { return "unknown"; }

  // One immutable origin shared by every stationary diagnostic. It is
  // captured at the first post-calibration frame that actually queries the
  // map, never reset per frame, and is deliberately outside the estimator.
  void ensureStationaryReference(const MeasureGroup& mg);

  // R61. Writes one state_trace.csv row (no-op when eval/state_trace_en is false).
  void logStateTrace(const char* phase, const MeasureGroup& mg, int residual_count, int completed_iterations);
  // R61. True when lio/open_loop/mode == propagate_only.
  bool openLoopActive() const { return opts_.open_loop.mode == "propagate_only"; }
  // R61. Call right after ensureStationaryReference() and the post_imu trace, only when openLoopActive(): captures
  // P0 on first use, applies the optional reset window, writes the post_lio row (= post_imu state, or the reset state),
  // fills mg.*_after_lio and returns a short report string. The caller must then return WITHOUT solving.
  std::string openLoopFinishScan(MeasureGroup& mg);

public:

  // History (270-279): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-270
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res);

protected:
  // CQ-49 item 1: the shared half of loadParameters(), extracted once and
  // called from both derived classes' own overrides -- see lio_base.cpp's
  // own doc comment for exactly which keys this reads.
  void loadSharedParameters(ConfigResolver& cfg, ros::NodeHandle& pnh);

  // CQ-74 item 5a: full 18x18 POST-update covariance eigenspectrum --
  // shared between both estimator paths so it's implemented once. No-op
  // unless opts_.log_eigenspectrum_en. mode_label distinguishes coupled
  // vs decoupled rows in the one shared log file (cq74_eigenspectrum.txt).
  void logEigenspectrum18(int scan_id, double t_abs, const char* mode_label) const;
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

  // [thread][0] = input points that reached at least one statistical gate;
  // [thread][1] = points rejected overall after at least one statistical
  // gate rejection. These are point counts, not candidate-plane counts.
  mutable std::vector<std::array<int, 2>> build_thread_gate_audit_;
  mutable int n_statistical_gate_candidates_ = 0;
  mutable int n_statistical_gate_rejections_ = 0;

  // R61 open-loop bookkeeping
  bool ol_captured_ = false;
  Eigen::MatrixXd ol_P0_;
  int ol_window_ = 0;
  double ol_window_t0_ = 0.0;
  int ol_reset_now_ = 0;

  bool stationary_reference_valid_ = false;
  M3D stationary_reference_R_ = M3D::Identity();
  V3D stationary_reference_p_ = V3D::Zero();
  V3D stationary_reference_v_ = V3D::Zero();

  // Tier0 (primary voxel)-specific miss classification -- see VoxelMap::
  // findPlaneResidual()'s tier0_had_plane out-param.
  mutable std::vector<std::array<int, 2>> build_thread_tier0_miss_;
  mutable int n_tier0_miss_coverage_ = 0;
  mutable int n_tier0_miss_mismatch_ = 0;

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
  int diagnostic_gn_iteration_ = -1;
  mutable LioCudaBuffers cuda_buf_;

  // CQ-28: this frame's residual-redundancy-correction engagement/magnitude
  // counters, written by solveSystem()/solveSystem_cuda() (both const) right
  // after accumulate[Cuda]() when opts_.residual_redundancy.on().
  mutable ResidualRedundancyStats redundancy_stats_;
};

}  // namespace livo_recon
