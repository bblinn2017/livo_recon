#pragma once

#include <memory>

#include "livo_recon/node_context.h"
#include "livo_recon/vio/tracker.h"
#include "livo_recon/vio/vio_accumulator.h"

namespace livo_recon
{

struct VioProcOptions
{
  // Master switch: when false, processVIO() is a no-op (and the node skips
  // ctx_.tracker->updateTracker() entirely too, see LivoReconNode::
  // updateMaps -- no point running the CoTracker backend
  // just to feed a disabled EKF) -- an LIO-only ablation baseline.
  bool   enable           = true;

  int    max_iterations  = 5;

  // History (22-29): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-22
  double min_norm_dtheta = 0.0;
  double min_norm_dt     = 0.0;
  double min_diff_error  = 1e-4;

  // Diagnostic-only (default false, behavior-preserving): when true,
  // solveSystem()'s "reject this iteration if its avg_error isn't better
  // than the previous iteration's" check is skipped -- every frame always
  // runs the full max_iterations, applying each iteration's correction
  // regardless of whether it improved. Exists to remove the selection
  // effect this stop criterion otherwise introduces in per-iteration
  // residual analysis (only "easy" frames that keep improving survive to
  // iteration 4+, biasing a pooled per-iteration average low at high iter
  // counts). NOT meant as a production setting -- the reject-if-worse
  // check exists to prevent committing a worse correction, same rationale
  // as the LIO motion-Q clamp this codebase has separately found matters
  // for real divergence cases; use this only for offline analysis.
  bool force_all_iterations = false;

  // Absolute sanity bound on the frame's final (post-iteration) avg_error:
  // solveSystem only ever rejects an iteration relative to the previous
  // iteration's error, so without this, a frame that lands in a
  // catastrophically bad state (e.g. from a burst of bad correspondences)
  // will happily "converge" to and commit an objectively terrible fit --
  // and since vio has no absolute reference, that contaminates every
  // subsequent frame's fixed "previous pose" permanently. If the best
  // avg_error achieved this frame still exceeds this, the entire frame's
  // correction is rolled back (state left exactly as it was beforehand)
  // rather than committed.
  double max_avg_error = 5.0;

  // History (46-68): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-46
  double pos_epipolar_residual_var = 0.0556;

  // History (71-83): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-71
  double neg_epipolar_residual_var = 0.03125;

  // See VioAccumulateOptions::pos_weight_scale's doc comment (vio_accumulator
  // .h) -- multiplies ONLY the position-column weight of every epipolar
  // residual's HtH/Htz contribution, leaving rotation-column weight
  // untouched. 1.0 (default) reproduces old behavior (no DOF split).
  double pos_weight_scale = 1.0;

  // History (92-108): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-92
  double anchor_split_depth = 0.0;

  // Adds the reverse-direction residual (project the current ray back into
  // the anchor frame, compare against the seed pixel) to the forward one --
  // classical symmetric transfer error. Ported default: true.
  bool bidirectional = true;

  // History (116-133): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-116
  bool distortion_weight_on = false;

  // Assumed per-axis pixel-space measurement noise (std dev, pixels) fed
  // into distortion_weight_on's error propagation -- only consulted when
  // distortion_weight_on is true. Ported default: 0.5.
  double weight_pixel_noise_px = 0.5;

  // See VioAccumulateOptions::residual_mode's doc comment (vio_accumulator.h)
  // -- "epipolar" (default) or "mixed" (ported from FAST-LIVO2's
  // ResidualMode::kMixed: two independently-weighted, DOF-disjoint
  // residuals per point instead of one conflated one).
  std::string residual_mode = "epipolar";
  double pos_ray_distance_residual_var = 1000.0;
  double neg_ray_distance_residual_var = 10.0;

  // History (149-159): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-149

  // One line per ESIKF iteration (see estimateStateCorrection()'s loop) to
  // log_path: residual count/avg_error, the 6x6 HtH normal-equations
  // matrix's rotation/position block traces, this iteration's own dtheta/
  // dt correction, and the resulting cumulative pose -- plus one
  // [vio/frame_stop] summary line per frame. Scoped down from FAST-LIVO2's
  // version (no norm_l-correlation stats, no HtH eigenvalue conditioning,
  // no bg/ba/gravity covariance traces) to what this codebase's own
  // residual computation and StateGroup actually track.
  bool log_iterations = false;

  // Per-point residual/Jacobian diagnostic logging -- one line per tracked
  // point per solveSystem() call. Off by default -- real per-point I/O
  // cost, only meant for a dedicated diagnostic run (see FAST-LIVO2's
  // identical option for the measured I/O-perturbs-timing caveat, which
  // applies here too).
  bool log_point_residuals = false;

  // R97A Q2: vio/log/dry_run is removed.

  // Lightweight alternative to log_point_residuals -- only accumulates
  // accepted/rejected COUNTS (no per-point string formatting/file writes)
  // and logs ONE summary line per solveSystem() call: n_valid, n rejected
  // (degenerate epipolar line -- the only reject reason this residual has,
  // since the cone-deadzone/depth-validity gates were removed, see
  // anchor_split_depth's doc comment), and frac_low_norm_l (fraction of
  // ACCEPTED points whose norm_l falls below log_norm_l_low_frac *
  // baseline_cc, this frame's camera-to-camera translation magnitude --
  // same relative-to-baseline definition FAST-LIVO2 uses).
  bool log_residual_counts = false;
  double log_norm_l_low_frac = 0.3;

  // History (200-203): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-200
  bool log_consistency_corr_en = false;

  // One line per frame, logged in processVIO() from the already-computed
  // TrackedFrame it's handed -- BEFORE estimateStateCorrection() ever
  // touches the values, so this is the tracker's raw output completely
  // undisturbed by anything EKF-side. See FAST-LIVO2's identical option's
  // doc comment for the tracker-determinism-debugging motivation this
  // exists for.
  bool log_raw_tracker_output = false;

  // One row per tracked point per frame: raw (u,v) pixel coordinate, keyed
  // by its anchor index (frame.indices[i]) so the SAME physical point can
  // be followed across frames (until the next reseed changes the anchor
  // set) -- lets a consumer compute true per-point frame-to-frame pixel
  // displacement directly, which log_raw_tracker_output's one-line-per-
  // frame summary (counts only) can't give. Separate flag from
  // log_raw_tracker_output rather than folding into it, so anything
  // already parsing that single summary line's format is unaffected.
  bool log_raw_uv_dump = false;

  // Single opt-in debug-log destination for every option above.
  std::string log_path = "/tmp/livo_recon_myvio_log.txt";

  // History (217-225): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-217
};

// History (228-257): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-228
class VioProc
{
public:
  explicit VioProc(NodeContext& ctx);
  ~VioProc() = default;

  std::string loadParameters(ros::NodeHandle& pnh);

  bool enabled() const { return opts_.enable; }

  // Runs the EKF correction using mg.tracked_frame (task #145) -- already
  // populated by CbkProc::syncMeasures() by the time this frame reaches
  // state estimation at all (see its own waitReadyFor(0ms) gate), so no
  // wait/consume call is needed here.
  std::string processVIO(MeasureGroup& mg);

  // History (274-282): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-274
  // `iter` is CombinedProc's own outer joint-EKF iteration index (0..max_iterations-1),
  // threaded through to the shared accumulate() call purely so log/vio/point_residuals
  // tags each row with which combined-loop iteration produced it -- previously hardcoded
  // to 0 regardless of the caller's actual iteration, which made every combined-mode
  // point-residual row indistinguishable by iteration. Does not affect accumulation math.
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, float& avg_error, int iter);

private:
  // Snapshot of the anchor set TrackedFrame::indices refers into, as
  // consumed from the async tracker's result -- see TrackedFrame's own
  // doc comment for why this can't just be tracker_->anchors() read live
  // (the background thread may have already overwritten it with a NEWER
  // reseed by the time this frame's correction runs).
  std::vector<AnchorPoint> frame_anchors_;
  VioProcOptions opts_;

  EkfUpdate ekf_;

  // Fixed IEKF prior (mean + covariance), snapshotted ONCE at frame entry
  // (top of estimateStateCorrection(), before any iteration runs) and
  // reused, unchanged, by every solveSystem() call this frame -- see
  // ekf.h's applyMeanUpdate()/applyCovarianceUpdate() doc comments for why
  // re-reading/rewriting state_->cov() every iteration (the old
  // applyUpdate() behavior) was a bug.
  Eigen::MatrixXd prior_cov_;
  StateGroup state_propagat_;

  TrackerPtr tracker_;  // ctx_-level (see NodeContext::tracker) -- owns the PointTracker backend, baseline reference, and anchor selection

  StateGroupPtr state_;
  ProfilerPtr   profiler_;
  DataQueuesPtr data_queues_;  // only used for start_time, to log absolute (bag/wall-clock) t_abs matching lio_processing.cpp's own [lio] lines and FAST-LIVO2's [vio/iter] lines -- see processVIO()'s t_abs computation

  int last_n_residuals_ = 0;  // number of residuals actually used in the most recent solveSystem call

  // History (313-317): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-313
  VioAccumulator accumulator_;

  // Builds a VioAccumulateOptions view of opts_ -- called fresh each
  // accumulate() call rather than cached, since options can't change
  // mid-run anyway and this keeps VioAccumulateOptions's field set the
  // single source of truth for what the accumulator actually reads.
  VioAccumulateOptions accumulateOptions() const;

  // Takes the whole TrackedFrame (not just indices/uv_curr) so it can also
  // read uv_curr_uv_jacobian for distortion_weight_on. t_abs/iter identify
  // the caller for log_iterations/log_point_residuals.
  bool solveSystem(const TrackedFrame& frame, float& last_error, double t_abs, int iter);

  // Runs the EKF correction loop (repeated solveSystem calls) to convergence
  // and returns a log string summarizing how it stopped.
  std::string estimateStateCorrection(const TrackedFrame& frame, double t_abs);
};

}  // namespace livo_recon
