#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/cuda/lio_cuda.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/lio/deskew.h"
#include "livo_recon/lio/info_scale.h"
#include "livo_recon/lio/residual_binning.h"

#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace livo_recon
{

class ConfigResolver;

// CQ-49: the options SHARED between the iekf and coupled estimators --
// residual matching, the EKF accumulation, the redundancy/weighting/
// sigma-scale axes, deskew/downsample. The spline/adaptive_q keys no longer
// exist (removed 2026-09-30); finalizeConfig() refuses any left in a config.
struct LioProcOptions
{
  int    max_iterations        = 5;
  // R97A Q2: min_norm_dtheta/min_norm_dt/min_diff_error (the early-stop thresholds) are removed;
  // every scan now runs all max_iterations iterations.

  // History (24-27): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-24
  bool   log_debug_en          = false;

  // R69 (F-300): write-only diagnostics, default off / unchanged.
  // update_diag_en: lio_update_diag.csv, one row per (frame, iteration): HtH upper triangle, Htz, residual variance
  // decomposition, step. live_point_cov_en: live_point_cov.csv, one row per frame. first_frame_dump_frames K:
  // the existing pose_control_first_frame_solve{.csv,_matrices.txt} dump is written for frames 1..K (1 = as before).
  bool   update_diag_en        = false;
  bool   live_point_cov_en     = false;
  int    first_frame_dump_frames = 1;
  // R91C (write-only; empty / false = unchanged and byte-identical): log/lio/first_frame_dump/frame_list is a comma-separated list of further frame indices that get the solve snapshot
  // (pose_control_first_frame_solve{.csv,_matrices.txt}) in addition to 1..first_frame_dump_frames; log/lio/first_frame_dump/slim (refused unless the list is non-empty) writes those frames
  // in the slim form (prior_cov, A_exact, HtH, Htz, the prior and measurement terms, the solution: no H_full, K1, G and no per-residual lines).
  std::vector<int> first_frame_dump_list;
  bool   first_frame_dump_slim = false;

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

  // R97A Q2: dry_run_point_filter_num and iekf_point_stream (the second decimation stream, the dry-run
  // shadow pass and vio/log/dry_run) are removed. The map and the IEKF both use the one primary point
  // set (what "iekf_point_stream = primary" / "dry_run_point_filter_num = 0" always did).

  // CQ-60 item 5: the shared per-dof NEES logger (utils/eval/nees_logger.h)
  // -- report-only, default false, shared by both estimator paths (unlike
  // most of this struct's coupled-vs-iekf-specific fields, this one
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
  // R86B: also write a "pre_imu" state_trace row (the full state and covariance BEFORE the IMU propagation of the scan) for frame indices 1..K.
  // Write-only. 0 (default) writes nothing and is bit-identical. Frame index = voxel_map_->frame_idx_ when estimateState starts, the same value the post_imu row carries.
  int state_trace_pre_imu_frames = 0;
  // R92: also write pose_cov_trace.csv (the slim per-scan trace: post_lio rotation quaternion, position and the 6 x 6 pose covariance). Write-only; default false writes nothing and is bit-identical.
  // Independent of log/state_trace/enable, so the sweep can switch the 18 x 18 trace off and keep this one.
  bool pose_cov_trace_en = false;
  // R92: with pose_cov_trace_en, rows are written only for scans whose time since the bag start is at most this many seconds (the stationary window of the bag; the run continues after it). Negative (default) = every scan.
  // A refused sub-option without log/pose_cov_trace/enable (a dead key, rule 3.4).
  double pose_cov_trace_end_s = -1.0;

  // R97A Q2: open_loop (lio/open_loop/mode, reset_period_s), startup_hold_scans and
  // bootstrap_inflate_scans/factor are removed (all off at the yaml).

  // R93A (Bryce 2026-10-10; key lio/gate_state_uncertainty, default true = the former hard-wired include_state_uncertainty_in_gate argument). true: the admission gate's variance adds the
  // point's pose covariance n^T P_pose n (the residual WEIGHT never does). false: the gate uses the sensor + plane variance only. Experimental when false: the engagement block of the
  // effective config prints the count of gate rows and the mean pose variance they added, and flags a run where the key is on and nothing was added.
  bool gate_state_uncertainty = true;

  // R97A Q2: ResidualRedundancyOptions (lio/residual_redundancy/*) is removed.

  // R70 (F-320): separate rotation / position scales on the LiDAR information (see info_scale.h). mode "off" (default) is
  // bit-identical. iekf estimator only.
  InfoScaleOptions info_scale;

  // R91E / R91F: residual binning by constraint direction (see lio/residual_binning.h): a shared shift per direction bin bounds the information a direction can gain from many rows.
  // mode "off" (default) never enters the function and is bit-identical; "direction" = position bins (R91E), "rotation" = rotation bins, "both" = joint (R91F). iekf estimator only; refused
  // together with info_scale, with combined/enable, and with voxel_map/plane/line_residual/rows = distance (the direction of a distance row is set by the noise).
  ResidualBinningOptions residual_binning;

  // R70 (F-340): write-only per-frame geometry of the admitted residuals (iteration 0 only): normal and lever-arm scatter
  // eigenvalues, plane concentration, distance statistics. Backend independent, so surface and octree can be compared.
  bool   residual_geometry_en  = false;

  // R77 (F-451): write-only per-scan record (line_rows_per_scan.csv, iteration 0 of every frame) of the rows the update used by kind (plane, hard line u0/u1, smooth u0/u1/u2),
  // the points that got line or smooth rows, the sums of fit variance, tilt variance and total variance, and the cumulative R77 engagement counters
  // (voxelPlaneR77Counters()). Default false writes nothing and is bit-identical.
  bool   line_rows_log_en      = false;
  // R95 (Bryce 2026-10-11; key log/lio/zsq, default false = nothing is written, bit-identical): zsq_per_scan.csv, one row per scan for iteration 0 and one for the last executed iteration, holding
  // the sums and counts of z squared = r^2 / sigma_squared of the residuals (diagnostics/zsq_summary.h defines every column). Write-only; experimental: the engagement block of the effective config prints the row count and flags a run where the key is on and no row was written.
  bool   zsq_log_en            = false;

  // R71 (F-363): write-only per-residual dump (residual_dump.csv, residual_dump_state.csv) for the listed frame indices at
  // iteration 0. Empty (default) writes nothing and is bit-identical.
  std::vector<int> residual_dump_frames;
  // R73 (F-417): also dump the residuals of the LAST executed iteration of each listed frame (iteration index recorded in the
  // iter column). Sub-key of log/lio/residual_dump/frames (refused when that list is empty). Default false is bit-identical.
  bool residual_dump_last_iter = false;
  // R86B: dump the residuals of EVERY iteration (iteration index in the iter column) of each listed frame. Sub-key of log/lio/residual_dump/frames (refused when that
  // list is empty). Supersedes residual_dump_last_iter (the last iteration is already one of the dumped iterations). Default false is bit-identical.
  // Row semantics (unchanged, stated here because the R71 comment above was misleading): the residuals of iteration k are evaluated at the state BEFORE the step of
  // iteration k; the state in residual_dump_state.csv is the state AFTER that step (solveSystem applies it before the dump is written).
  bool residual_dump_all_iters = false;
  // R81 (F-485 2B): append voxel_point_count and voxel_distinct_frames (of each row's voxel) to residual_dump.csv. Sub-key of log/lio/residual_dump/frames (refused when that list is empty).
  // Default false leaves the header and rows byte-identical.
  bool residual_dump_voxel_info = false;
  // R91C: append line_axis_x/y/z and line_axial to residual_dump.csv (see diagnostics/residual_dump_format.h). Sub-key of log/lio/residual_dump/frames (refused when that list is empty).
  bool residual_dump_line_geometry = false;
  // R91C: write residual_dump.csv in the 14-column slim form. Sub-key of log/lio/residual_dump/frames (refused when that list is empty).
  bool residual_dump_slim = false;


  // R73 (F-414, F-415): scan_log.csv, one row per map-backed scan: whole-map statistics and digests (voxel count, voxels created,
  // plane nodes, planes changed, exact and 1e-3-rounded digests), residual count, reduced chi2 at iteration 0, the step norm of every
  // iteration, posterior covariance traces. Write-only; default off is bit-identical. iekf estimator only.
  bool scan_log_en = false;

  // R97A Q2: PriorScalarOptions (lio/p_inflate/alpha, lio/p_floor/min_eig, lio/p_fading/lambda) is removed.

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
// silently break the iekf path's byte-identity through a well-meaning
// partial override, which is the same protection the card's own
// ResidualEngine-by-composition design gives, without a second indirection
// layer). LioProcIekf and LioProcCoupled each own ALL of their own
// estimator-specific state; neither may hold a member the other also
// holds (verified: no coupled_* member exists on this class).
class LioProcBase
{
public:
  explicit LioProcBase(NodeContext& ctx);
  virtual ~LioProcBase() = default;

  virtual std::string loadParameters(ros::NodeHandle& pnh) = 0;

  // TEMPORARY (2026-10-27, correction_breakdown_diag scan_id-alignment fix):
  // the SAME physical-frame counter state_trace.csv's "scan_id" column is
  // keyed on (see logStateTrace()) -- used so diagnostics logged from
  // different call sites (combined_processing.cpp, vio_processing.cpp) can
  // tag their rows with a scan_id that's directly comparable across modes,
  // instead of each keeping its own independently-incremented counter
  // (which can drift from this one, and from each other, whenever either
  // path's diagnostic call is skipped on a different subset of frames --
  // confirmed this session to drift by a full frame by scan ~52).
  int frameIdx() const;

  // One line per toggle: what it was set to, how often its mechanism ran, and
  // how far it moved things. Written to engagement.txt in the debug log dir
  // and to the log.
  virtual std::string engagementReport() const = 0;

  // Stage 1 (see include/livo_recon/lio/deskew.h's module doc comment):
  // deskew + downsample, populating mg.points. Called ONCE per frame, from
  // LivoReconNode::estimateState() right after imu_proc_.processIMU(mg).
  // VIRTUAL: LioProcIekf deskews from the IMU poses; LioProcCoupled deskews
  // from the raw IMU chain.
  virtual void deskewAndDownsample(MeasureGroup& mg) = 0;

  // R86B. Writes the state_trace.csv row "pre_imu" for frames 1..log/state_trace/pre_imu_frames. Call it before the IMU propagation of the scan, with the state
  // untouched since the previous scan. No-op by default.
  void logPreImuState(const MeasureGroup& mg);

  // R61. Writes one state_trace.csv row (no-op when log/state_trace/enable is false). R92: for
  // phase post_lio it also writes the pose_cov_trace.csv row when log/pose_cov_trace/enable is
  // true. Made public 2026-10-23: CombinedProc::processCombined() does NOT call processLIO() in
  // its normal (non-rejected) path -- it accumulates LIO's HtH/Htz via lio_proc.
  // accumulateForCombined() instead and applies one joint update -- so without this, combined
  // mode produced NO state_trace.csv rows at all whenever VIO wasn't rejected (confirmed this
  // session: a lio_only/sequential/combined frame 1-6 comparison came back with combined's
  // state_trace.csv simply missing). CombinedProc calls this directly (via its own lio_proc
  // reference) at the same two checkpoints LioProcIekf::processLIO() uses internally -- "post_imu"
  // before its iteration loop, "post_lio" after its one joint correction is applied -- so all
  // three modes produce a comparable post_imu/post_lio row per frame. For combined mode "post_lio"
  // means "after the per-frame correction" (there's no separate LIO-only checkpoint once LIO/VIO
  // are fused -- see LivoReconNode's own identical treatment of mg.pos_after_lio/pos_after_vio).
  void logStateTrace(const char* phase, const MeasureGroup& mg, int residual_count, int completed_iterations);

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
    const std::vector<M3D>* pose_cov_overrides = nullptr) const;

  void solveSystem(const std::vector<Residual>& residuals) const;
  void solveSystem_cuda(const std::vector<Residual>& residuals) const;

  // Used only by the iekf estimator's own iteration loop, but the
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

  // R97A Q2: openLoopActive()/startupHoldActive()/openLoopFinishScan() are removed with
  // lio/open_loop/* and lio/startup_hold/scans.

public:

  // History (270-279): see docs/livo_recon_changelog.md#include-livo_recon-processing-lio_processing.h-270
  // `iteration` (2026-10-28): the combined joint-GN loop's own iteration
  // counter, mirroring VioProcBase::accumulateForCombined()'s existing
  // `iter` parameter -- lets logResidualDump() (log/lio/residual_dump/*)
  // fire in combined mode the same way it already does in LIO-only mode's
  // estimateStateCorrection(). Previously accumulateForCombined() never
  // called logResidualDump() at all (dumpResidualsIfListed() existed but
  // was dead code, never invoked from anywhere), so every combined-mode
  // residual_dump.csv request silently produced nothing.
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res, int iteration = 0);

protected:
  // CQ-49 item 1: the shared half of loadParameters(), extracted once and
  // called from both derived classes' own overrides -- see lio_base.cpp's
  // own doc comment for exactly which keys this reads.
  void loadSharedParameters(ConfigResolver& cfg, ros::NodeHandle& pnh);

  // CQ-74 item 5a: full 18x18 POST-update covariance eigenspectrum --
  // shared between both estimator paths so it's implemented once. No-op
  // unless opts_.log_eigenspectrum_en. mode_label distinguishes coupled
  // vs iekf rows in the one shared log file (cq74_eigenspectrum.txt).
  void logEigenspectrum18(int scan_id, double t_abs, const char* mode_label) const;
  // The shared finish: refuseUnclaimed(allowed_unclaimed_namespaces), the
  // build/accumulation_precision derived line, the ok()-or-throw gate, and
  // returning cfg.report().
  std::string finalizeConfig(ConfigResolver& cfg, std::initializer_list<const char*> allowed_unclaimed_namespaces);

  // R73: per-residual dump of the CURRENT residuals_ (and state_) when the current map frame is in log/lio/residual_dump/frames.
  void dumpResidualsIfListed(int iteration) const;
  // R95: one zsq_per_scan.csv row for the CURRENT residuals_ and map frame (no-op when log/lio/zsq is false). Iteration 0 is written inside estimateStateCorrection(); the IEKF loop calls this once more after the loop for the last executed iteration.
  void logZsqIteration(int iteration) const;
  // R95: rows written to zsq_per_scan.csv since process start (engagement counter).
  static long zsqRowsWritten();

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

  // R97A Q2: open-loop (lio/open_loop/*) is removed; ol_window_/ol_reset_now_ (state_trace.csv's
  // columns) stay at their default 0 always, matching every job's actual behavior.
  int ol_window_ = 0;
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

  // R97A Q2: redundancy_stats_ (ResidualRedundancyStats) is removed.

  // R70 (F-320): engagement counters of the block information scale (written by solveSystem*, which are const).
  // R72 (F-403): step-cap engagement (iterations seen, iterations where the cap bound, sum of applied/raw scale when bound).
  mutable long info_scale_iterations_ = 0;
  mutable long info_scale_discarded_modes_ = 0;
  mutable double info_scale_trace_ratio_sum_ = 0.0;
  // R97A Q2: redundancyAndRenorm()/renorm_* (lio/residual_redundancy/*) are removed.
  void applyInfoScaleToEkf(const std::vector<Residual>& residuals) const;
  // R91E: engagement counters of the residual binning (written by solveSystem*, which are const).
  mutable long residual_binning_iterations_ = 0;
  mutable long residual_binning_rows_sum_ = 0;       // rows in corrected bins, summed over iterations
  mutable long residual_binning_skipped_sum_ = 0;    // rows left out (distance rows, bad variance), summed over iterations
  mutable long residual_binning_bins_sum_ = 0;       // corrected bins, summed over iterations
  mutable double residual_binning_trace_before_sum_ = 0.0, residual_binning_trace_after_sum_ = 0.0;
  mutable long residual_binning_rot_rows_sum_ = 0, residual_binning_rot_bins_sum_ = 0, residual_binning_solve_failed_sum_ = 0;   // R91F (rotation and both modes)
  void applyResidualBinningToEkf(const std::vector<Residual>& residuals) const;
  // 2026-10-27: generalized target of applyResidualBinningToEkf() -- see its own .cpp doc
  // comment. applyResidualBinningToEkf() is now a thin wrapper calling this with ekf_.
  void applyResidualBinningTo(const std::vector<Residual>& residuals, EkfUpdate& target) const;
};

}  // namespace livo_recon
