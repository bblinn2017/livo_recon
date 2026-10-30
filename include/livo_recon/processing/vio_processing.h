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

  // Absolute sanity bound on the frame's final (post-iteration) fit
  // quality: solveSystem only ever rejects an iteration relative to the
  // previous iteration's error, so without this, a frame that lands in a
  // catastrophically bad state (e.g. from a burst of bad correspondences)
  // will happily "converge" to and commit an objectively terrible fit --
  // and since vio has no absolute reference, that contaminates every
  // subsequent frame's fixed "previous pose" permanently. If the best fit
  // achieved this frame still exceeds this, the entire frame's correction
  // is rolled back (state left exactly as it was beforehand) rather than
  // committed.
  //
  // 2026-10-23: units changed from a weighted/squared/normalized-
  // coordinate quantity (the EKF's own optimization objective, not
  // interpretable in real-world terms) to avg_abs_pixel_residual -- the
  // mean |pixel error| across tracked points (see VioAccumulator::
  // accumulate()'s doc comment). The per-iteration convergence check
  // inside solveSystem() is UNCHANGED, still on the old scale -- only
  // this whole-frame gate switched. The old default (5.0) was calibrated
  // for the old metric and needs re-tuning for this one; still the
  // placeholder value pending that.
  double max_avg_error = 5.0;

  // 2026-10-27: collapsed from pos_/neg_epipolar_residual_var into one
  // value, and renamed from "_var" to "_weight" -- it was always applied
  // DIRECTLY as the weight (weight = epipolar_residual_weight), never as
  // an inverse variance requiring a runtime 1/var division, so "_var" was
  // misleading (larger value = MORE weight/trust, opposite of what a
  // variance name implies). Mixed mode and pos_weight_scale, the other
  // two residual-strength levers, were removed the same day -- see
  // docs/livo_recon_changelog.md for the removal history.
  double epipolar_residual_weight = 0.0556;

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
  // TEMPORARY (2026-10-27, vio_residual_diag investigation -- delete
  // force_run/out_* params + their plumbing in estimateStateCorrection()
  // once done): force_run=true bypasses the `!opts_.enable` early return
  // and actually runs (and APPLIES to state_) a real VIO correction even
  // when VIO is normally disabled -- the caller is responsible for
  // snapshotting/restoring state_ around such a call if it wants a
  // read-only evaluation (see livo_recon_node.h's force_vio_residual_log_
  // doc comment for why). out_avg_error/out_avg_abs_pixel_residual/
  // out_n_valid/out_ok, when non-null, receive the TRUE final values
  // (evaluated AFTER the accept/reject decision, at whatever pose this
  // frame actually ends with) -- not the stale pre-last-update values
  // last_error/last_pixel_residual hold mid-loop. Default args keep every
  // existing call site byte-identical.
  // pre_lio_state_for_ask (2026-10-27, correction_breakdown_diag fix):
  // optional snapshot of state_ taken by the caller BEFORE LioProc::
  // processLIO() ran this frame. In sequential mode, by the time
  // processVIO() is entered, state_ already holds LIO's converged
  // correction for this frame -- so without this, the diagnostic "ask"
  // block in estimateStateCorrection() computes VIO's unweighted ask
  // against a post-LIO pose, not the true frame-entry (post-IMU-
  // propagation, pre-any-correction) pose that combined mode's own ask
  // diagnostic uses. When non-null, estimateStateCorrection() temporarily
  // swaps *state_ to this snapshot for that one read-only accumulate()
  // call, then restores it before the real correction loop runs -- see
  // estimateStateCorrection()'s own comment. Does not affect the real
  // correction in any way.
  // diag_frame_idx (2026-10-27, correction_breakdown_diag scan_id-alignment
  // fix): the caller's voxel_map frame index (LioProcBase::frameIdx()),
  // used ONLY to tag the ask-diagnostic row with the SAME physical-frame
  // counter combined mode's own ask diagnostic uses (see LioProcBase::
  // frameIdx()'s doc comment) -- replaces a separately-incremented counter
  // that could (and, confirmed this session, did) drift by a full frame
  // relative to combined mode's.
  std::string processVIO(MeasureGroup& mg, bool force_run = false,
                         float* out_avg_error = nullptr, float* out_avg_abs_pixel_residual = nullptr,
                         int* out_n_valid = nullptr, bool* out_ok = nullptr,
                         const StateGroup* pre_lio_state_for_ask = nullptr,
                         int diag_frame_idx = -1);

  // History (274-282): see docs/livo_recon_changelog.md#include-livo_recon-processing-vio_processing.h-274
  // `iter` is CombinedProc's own outer joint-EKF iteration index (0..max_iterations-1),
  // threaded through to the shared accumulate() call purely so log/vio/point_residuals
  // tags each row with which combined-loop iteration produced it -- previously hardcoded
  // to 0 regardless of the caller's actual iteration, which made every combined-mode
  // point-residual row indistinguishable by iteration. Does not affect accumulation math.
  // avg_abs_pixel_residual: see VioAccumulator::accumulate()'s doc comment
  // -- the genuinely pixel-space quantity CombinedProcOptions::
  // max_avg_error now compares against, separate from avg_error (which
  // stays on its weighted/squared/normalized scale for anything that
  // reads it as a convergence signal).
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, float& avg_error,
                             float& avg_abs_pixel_residual, int iter);

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

  // TEMPORARY (2026-10-27, correction_breakdown_diag investigation) -- delete alongside it.
  int correction_breakdown_frame_index_ = 0;

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
  // last_pixel_residual is updated in lockstep with last_error, ONLY on an
  // iteration that's actually accepted (mirrors last_error's own update
  // site) -- see VioAccumulator::accumulate()'s doc comment for why this
  // is a separate, genuinely pixel-space quantity.
  bool solveSystem(const TrackedFrame& frame, float& last_error, float& last_pixel_residual,
                   double t_abs, int iter);

  // Runs the EKF correction loop (repeated solveSystem calls) to convergence
  // and returns a log string summarizing how it stopped.
  // TEMPORARY (2026-10-27, vio_residual_diag): out_avg_error/
  // out_avg_abs_pixel_residual/out_n_valid, when non-null, receive the
  // TRUE final residual (see processVIO()'s matching doc comment) --
  // delete alongside the rest of this investigation's plumbing.
  std::string estimateStateCorrection(const TrackedFrame& frame, double t_abs,
                                      float* out_avg_error = nullptr,
                                      float* out_avg_abs_pixel_residual = nullptr,
                                      int* out_n_valid = nullptr,
                                      const StateGroup* pre_lio_state_for_ask = nullptr,
                                      int diag_frame_idx = -1);
};

}  // namespace livo_recon
