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
  // updateMaps -- no point running the CoTracker/TAPNext/Track-On backend
  // just to feed a disabled EKF) -- an LIO-only ablation baseline.
  bool   enable           = true;

  int    max_iterations  = 5;

  // 2026-08-12: defaulted to 0.0 (never activates -- ekf_.dtheta.norm()/
  // dt.norm() are >=0 and essentially never EXACTLY 0.0 in floating point,
  // so this branch of estimateStateCorrection()'s stop condition becomes
  // unreachable dead weight at these defaults) rather than removed
  // outright -- FAST-LIVO2's current myvio has no equivalent convergence
  // criterion at all (min_diff_error alone gates iteration count there),
  // and this codebase's own field is kept only so a caller can still
  // opt back into it explicitly via config if ever wanted.
  double min_norm_dtheta = 0.0;
  double min_norm_dt     = 0.0;
  double min_diff_error  = 1e-4;

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

  // Per-point WEIGHT (2026-08-04, PORTED FROM FAST-LIVO2's livo_vio A/B
  // scaffold -- see that repo's vio_proc.h/.cpp for the full ablation
  // history this default reflects) for the epipolar-line residual on a
  // point whose free (unconstrained) anchor-ray depth s0 is >=
  // anchor_split_depth (a normal, "in front of the camera" point), applied
  // DIRECTLY (weight = pos_epipolar_residual_var, not a variance requiring
  // a runtime 1/var division) unless distortion_weight_on multiplies it by
  // a per-point 1/predicted-variance factor -- see that option's doc
  // comment. Fixed base value -- NO state/pose-covariance term folded in
  // (that was this function's own bug before this port: weight =
  // 1/(J*P66*J^T + epipolar_residual_var), which (a) double-counts the
  // same pose uncertainty already accounted for via prior_cov_.inverse()
  // inside EkfUpdate::applyMeanUpdate()'s A = H_full + prior_cov.inverse()
  // -- exactly the pose_cov_in_sigma double-count LioProc's voxel residual
  // already had fixed on the LIO side, never ported to VIO until now --
  // and (b) creates a self-reinforcing feedback loop, P shrinks -> state_var
  // shrinks -> weight rises -> P shrinks further, plus the per-point
  // information Jr^T*weight*Jr = J^2/(J*P*J^T+var) cancels J's own scale
  // for near-degenerate points instead of vanishing the way a
  // no-information point should. Confirmed this session: running the OLD
  // state_var-coupled weight on eee_01 (pfn3_baseline defaults) diverged to
  // 10.93m ATE; this fixed-weight port is what's being validated to
  // replace it.
  double pos_epipolar_residual_var = 0.0556;

  // Per-point WEIGHT for the epipolar-line residual on a point whose free
  // anchor-ray depth s0 is < anchor_split_depth (it would triangulate
  // behind or too close to the anchor camera) -- kept in the solve anyway,
  // at this separately-tunable (normally looser) weight, instead of being
  // dropped outright: still real epipolar-line information even though its
  // depth is untrustworthy. This fallback is unconditional (2026-08-05 --
  // see anchor_split_depth's doc comment, which replaced the old
  // use_behind_camera_penalty_epipolar toggle): set equal to
  // pos_epipolar_residual_var to make depth-clamped points
  // indistinguishable from normal ones; set to 0.0 to zero out their
  // contribution entirely (equivalent to dropping them, since a zero
  // weight contributes nothing to HtH/Htz either way). Ported default:
  // 1/32.
  double neg_epipolar_residual_var = 0.03125;

  // See VioAccumulateOptions::pos_weight_scale's doc comment (vio_accumulator
  // .h) -- multiplies ONLY the position-column weight of every epipolar
  // residual's HtH/Htz contribution, leaving rotation-column weight
  // untouched. 1.0 (default) reproduces old behavior (no DOF split).
  double pos_weight_scale = 1.0;

  // 2026-08-12: consolidates the old, separately-tunable clamp_depth
  // (pos_/neg_epipolar_residual_var split threshold) and ray_min_depth/
  // use_cone_deadzone_epipolar/cone_radius_per_depth/cone_deadzone_hard
  // (a rejection-cone mechanism around the epipolar line, gated on a
  // SEPARATE min-depth floor) into ONE shared threshold, matching
  // FAST-LIVO2's own consolidation (see that codebase's
  // EpipolarOptions::anchor_split_depth doc comment) -- the cone-deadzone
  // mechanism itself was found strictly worse than the plain clamp_depth
  // fall-through weighting there (NTU_VIRAL production sweep:
  // bidir_cone_only 0.0297m vs bidir_clamp_bce 0.0292m) and removed
  // entirely rather than ported. A point whose free (unconstrained)
  // anchor-ray depth s0 falls below this floor is NOT excluded -- it still
  // contributes the same line residual, just at neg_epipolar_residual_
  // var's weight instead of pos_epipolar_residual_var's (see
  // computeDirectionalEpipolar). Default 0.0 (was 0.5 for the old
  // ray_min_depth) -- matches the request to have this floor alone drive
  // the pos_/neg_ split with no separate depth-validity gate underneath it.
  double anchor_split_depth = 0.0;

  // Adds the reverse-direction residual (project the current ray back into
  // the anchor frame, compare against the seed pixel) to the forward one --
  // classical symmetric transfer error. Ported default: true.
  bool bidirectional = true;

  // 2026-08-12: rigorous per-point distortion-weighting, ported from
  // FAST-LIVO2's EpipolarOptions::distortion_weight_mode -- simplified to
  // a plain on/off toggle here (FAST-LIVO2 also has a "cheap_bz2" middle
  // option this codebase has no equivalent bookkeeping for, so it's not
  // offered). When true, MULTIPLIES the base pos_/neg_epipolar_residual_
  // var weight by 1/Var(r), an error-propagation estimate of how much the
  // epipolar-line residual r is expected to move around from an assumed
  // weight_pixel_noise_px pixel-space Gaussian measurement noise alone
  // (propagated through both the anchor-seed and current-tracked pixels'
  // local undistortion Jacobians, AnchorPoint::uv_jacobian_seed/
  // TrackedFrame::uv_curr_uv_jacobian, into the residual's own analytic
  // Jacobian w.r.t. those two pixels) -- see epipolarLineWeightVar(). This
  // is a genuinely different signal from the depth-based pos_/neg_ split:
  // a point near the epipole (small norm_l) gets a large predicted
  // variance regardless of its depth, so it's downweighted here even when
  // its depth looks perfectly normal. false (default) reproduces the
  // existing fixed-weight behavior exactly (weight = pos_/neg_
  // epipolar_residual_var, unchanged).
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

  // ── Diagnostic logging (2026-08-12, ported from FAST-LIVO2's myvio --
  // see VioProcOptions there for the originals). Replaces the old
  // log_debug_en/debugLogVio mechanism (a compile-time-gated dump to
  // /tmp/vio.txt that was permanently dead -- kDebugVioLoggingEnabled was
  // hardcoded false -- since task #149's cleanup) with properly runtime-
  // toggleable options sharing one log_path. NOT ported:
  // log_parallax_histogram -- livo_recon's epipolar residual has no
  // parallax-gating/parallax-angle computation at all (unlike FAST-LIVO2's
  // ParallaxGateOptions machinery), so there is no parallax angle to
  // histogram here; porting the flag alone with nothing behind it would
  // be a no-op that looks configurable but silently isn't. ──────────────

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

  // When true, estimateStateCorrection() still runs the full per-iteration
  // residual/Jacobian/HtH computation and writes every normal log line,
  // but skips the two calls that actually mutate state_ (ekf_.
  // applyMeanUpdate/applyCovarianceUpdate). Net effect: VIO is computed
  // and logged as usual, but the odometry output ends up recording pure
  // LIO-propagated poses -- for an apples-to-apples ATE comparison ("what
  // if this residual's corrections were never applied at all, at these
  // exact timestamps") against a real run with the same config.
  bool dry_run = false;

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

  // One line per frame, logged in processVIO() from the already-computed
  // TrackedFrame it's handed -- BEFORE estimateStateCorrection() ever
  // touches the values, so this is the tracker's raw output completely
  // undisturbed by anything EKF-side. See FAST-LIVO2's identical option's
  // doc comment for the tracker-determinism-debugging motivation this
  // exists for.
  bool log_raw_tracker_output = false;

  // Single opt-in debug-log destination for every option above.
  std::string log_path = "/tmp/livo_recon_myvio_log.txt";

  // Tried and REMOVED (2026-08-05): a soft, continuous per-point
  // downweighting by 3D ray-consistency (anchor ray through uv_seed vs.
  // current-camera ray through uv_tracked, weight scaled by
  // 1/(1+(ray_gap/ray_gap_ref)^2)). Consistently mildly harmful once every
  // other option was correctly re-ablated post-undistortion-fix, both on
  // single-sequence (eee_01) and full 9-sequence-suite testing (see
  // FAST-LIVO2's NTU_VIRAL.yaml livo_vio: promotion comment for the
  // confirming numbers) -- removed from the code entirely rather than left
  // off by default.
};

// Correspondence-driven VIO update: maintains a persistent set of tracked
// points (see vio/tracker.h, ctx_.tracker) via whichever native backend
// (CoTracker, TAPNext, or Track-On) is configured, anchored to a fixed grid
// of Shi-Tomasi seed pixels (see AnchorPoint) rather than requiring LIDAR
// depth. Every tracked point contributes a 1D epipolar-line residual (see
// computeEpipolarResidualAndJacobian) between its seed-time pixel and its
// currently-tracked pixel, via the same EkfUpdate machinery.
//
// This replaces an earlier design that instead anchored each tracked point
// to a real LIDAR 3D point and ran a metric-scale reprojection residual
// against it -- ported from FAST-LIVO2 (livo_vio/vio_proc.{h,cpp}), where a
// point-to-point reprojection residual against LIDAR-anchored 3D points was
// tried and removed: on eee_01 its per-iteration corrections were
// uncorrelated with (mixed config) or anti-correlated with (LIDAR-only
// config) actually moving the state closer to ground truth, unlike the
// epipolar residual's clearly corrective behavior.
//
// 2026-08-04: the residual/weighting model (everything in VioProcOptions
// above except max_iterations/min_norm_dtheta/min_norm_dt/min_diff_error/
// max_avg_error, which were already correct and unchanged) was replaced
// wholesale, ported from FAST-LIVO2's livo_vio A/B scaffold's own
// extensively-ablated production default -- see each option's doc comment
// above for what specifically changed and why. The EKF core (EkfUpdate,
// ekf.h) was NOT touched -- it's shared with LioProc, already carries this
// session's earlier validated fixes, and was confirmed correct; only the
// per-point residual/weight computation feeding into it was replaced.
// Motivated by: running the pre-port code on eee_01 (pfn3_baseline and
// pfn1_ds000 production defaults, VIO enabled) diverged to 10.93m and
// 1.51m ATE respectively -- see git history/session notes for the pre-port
// version if a reference is ever needed.
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

  // Combined-mode entry point (CombinedProc, 2026-08-14): builds this
  // frame's residuals via the shared accumulator_ (same options/geometry
  // path as the sequential solveSystem() below) and fills `out` -- does
  // NOT call out.applyMeanUpdate()/applyCovarianceUpdate() itself
  // (CombinedProc sums this with LioProc::accumulateForCombined()'s output
  // first). Returns false if there's no image/tracker/valid frame this
  // call, or no valid residuals were found (out/avg_error untouched).
  // avg_error is the same per-frame diagnostic solveSystem() has always
  // computed, for CombinedProc's own VIO-fit-rejection check.
  bool accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, float& avg_error);

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

  // Owns the actual residual/Jacobian/HtH-Htz math (2026-08-14, extracted
  // to vio/vio_accumulator.h so CombinedProc can call the exact same code
  // -- see that header's doc comment). solveSystem()/accumulateForCombined()
  // both just adapt VioProcOptions into a VioAccumulateOptions and call
  // accumulator_.accumulate().
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
