#pragma once

#include <deque>
#include <limits>
#include <vector>
#include <geometry_msgs/PoseStamped.h>

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

struct EvoProcOptions
{
  bool enable = false;

  // Both modes follow the same standardized convention: the ESTIMATE is
  // interpolated to ground truth's own timestamps, never the other way
  // around (interpolating GT would assume straight-line motion across
  // whatever gap separates two GT samples, which is not a safe assumption
  // when they're sparse -- and is simply unnecessary busywork when they're
  // dense, since the estimate is the one being freshly synthesized either
  // way). See processEvoTopicMode()/processEvoFileMode() and BracketCache.
  //
  // "topic": subscribe to gt_topic, a dense/regularly-sampled ground-truth
  // pose stream (e.g. NTU VIRAL's Leica laser-tracker prism pose).
  // processEvoTopicMode() commits one correspondence per buffered GT
  // sample, synthesizing the estimate's pose at that sample's exact
  // timestamp by interpolating between the two bracketing ESTIMATE frames.
  //
  // "file": load a static TUM-format ground-truth file once at startup
  // (gt_file) instead of subscribing to anything -- e.g. HILTI's publicly
  // released groundtruth_2022/2023 sequences, which are a handful (4-22)
  // of sparse surveyed control points spanning an entire multi-minute
  // sequence, not a continuous stream. Because these checkpoints are too
  // widely spaced for any reasonable relative-pose comparison,
  // compute_rte/compute_roe are force-disabled when gt_source == "file"
  // (see loadParameters()) -- matches HILTI's own evaluation.py, which is
  // APE/ATE-only for the same reason. For the same reason, "interp" mode
  // (see max_time_diff's doc comment) is never attempted for gt_source ==
  // "file" -- only "nearest" -- matching HILTI's own evaluation.py, whose
  // sparse-checkpoint path (num_poses <= 100) nearest-matches rather than
  // interpolating.
  std::string gt_source = "topic";

  // Ground-truth pose topic (e.g. a Leica laser-tracker prism pose, as in
  // NTU VIRAL). Orientation is used if the topic provides a meaningful one
  // (needed for ROE); if it's always identity (as NTU VIRAL's Leica topic
  // is), ROE against it is not meaningful and should stay disabled.
  // Only used when gt_source == "topic".
  std::string gt_topic = "/leica/pose/relative";

  // TUM-format ground-truth file path (timestamp tx ty tz qx qy qz qw,
  // '#'-prefixed comment lines ignored) -- e.g. one of HILTI's
  // groundtruth_2022/2023 sequence files. Only used when gt_source ==
  // "file". Timestamps are in the same absolute (bag/wall-clock) epoch as
  // the recorded sensor data, same convention gtCallback() assumes for
  // gt_topic messages.
  std::string gt_file;

  int queue_size = 2000;

  // Body/IMU-frame -> ground-truth-sensor lever arm, added (rotated by the
  // frame's own estimated orientation) to est_pos before comparing against
  // gt_pos. Zero by default (most ground-truth sources track the body
  // frame itself, or close enough not to matter). NTU VIRAL's Leica
  // tracker measures a physical prism mounted off-center from the IMU --
  // see FAST-LIVO2's own Log/result/ntu_viral/evaluate_viral.py
  // (trans_B2prism = [-0.293656, -0.012288, -0.273095]), which performs
  // this exact correction (position-only, no rotation offset needed since
  // the prism has no orientation of its own) before running its own
  // ATE/RPE evaluation. Omitting this doesn't bias a comparison between
  // two systems evaluated the same way (both miss the same term), but it
  // does add an uncorrected, orientation-dependent error of up to the
  // lever arm's own norm (~0.4m here) into the absolute ATE number itself.
  V3D gt_lever_arm = V3D::Zero();

  // Maximum time gap (seconds), same role as FAST-LIVO2's own
  // fastlivo_evo.py --max-time-diff (default there: 0.08; kept at this
  // module's pre-existing 0.05 default here since datasets/tuning already
  // depend on it). Two association MODES are computed against this single
  // threshold, exactly mirroring fastlivo_evo.py's lookup_est()/
  // lookup_nearest() (see processEvo()'s doc comment and MODE_NEAREST/
  // MODE_INTERP below):
  //   nearest: a GT sample is matched to whichever of the two RAW estimate
  //     frames bracketing it (before/after) is closer in time, and dropped
  //     if that gap exceeds max_time_diff.
  //   interp:  the estimate is lerp(position)/slerp(rotation)-interpolated
  //     to the GT sample's exact timestamp, and the sample is dropped if
  //     the two bracketing frames are more than 2*max_time_diff apart (the
  //     interpolation would span too much real motion to be meaningful) --
  //     fastlivo_evo.py's lookup_est() uses this same "2x" bracket-gap
  //     threshold, see its EVO_MAX_TIME_DIFF comment.
  // Both modes are computed unconditionally (whenever gt_source == "topic")
  // so this module's ATE/RTE/ROE are directly comparable to fastlivo_evo.py's
  // own dual-mode output -- there is no yaml toggle to pick just one.
  double max_time_diff = 0.05;

  // RTE/ROE (RPE's translation/rotation parts) compare pairs of matched
  // poses separated by ~rpe_delta_s; ATE is always computed regardless of
  // these flags. Force-disabled (loadParameters() logs a warning if set
  // true in yaml) when gt_source == "file" -- see gt_source's doc comment.
  bool   compute_rte = false;
  bool   compute_roe = false;
  double rpe_delta_s = 1.0;

  // Whether "nearest" mode (see max_time_diff's doc comment) is computed
  // LIVE when gt_source == "topic". Only affects topic mode -- file mode
  // (HILTI-style sparse checkpoints) has no interp alternative and always
  // computes nearest regardless of this flag (see gt_source's doc comment).
  //
  // Default false: "nearest" pushes RAW, unblended pose snapshots into the
  // alignment history (nearest.lio_pos = use_before ? bracket_cache_.
  // lio_pos : mg.pos_after_lio, no lerp/slerp), unlike "interp" which
  // always blends between both bracketing frames. An occasional bad
  // bracket pick -- e.g. bracket_cache_'s "before" candidate still
  // reflecting a near-uninitialized pose very early in a run -- injects
  // that raw bad point directly into the matched-pose history, and
  // computeAte()'s batch Kabsch/Horn SVD re-fit (over the WHOLE history,
  // every call) can be meaningfully biased by even one such outlier.
  // Confirmed this session: a livo_recon LIO-only NTU_VIRAL nya_01 run's
  // LIVE nearest ATE reported 1.35m for a trajectory that was actually
  // ~0.03m (confirmed by re-scoring the same run's exported odometry.txt
  // -- see outputs/odom/export -- through fastlivo_evo.py's own --mode
  // nearest, offline, after the run). "interp" mode did not show this
  // fragility on the identical run/data. With this flag off, "nearest" is
  // computed ONLY offline, via fastlivo_evo.py against the exported
  // odometry.txt, where a bad early point is one of potentially thousands
  // fit simultaneously in a single batch solve -- not live, where the same
  // bug can silently corrupt every subsequently-reported "current" nearest
  // ATE for the rest of the run. "interp" keeps being computed live
  // regardless of this flag (for incremental/expanding-ATE analysis).
  bool compute_nearest_live = false;
};

// Optional trajectory-evaluation module: subscribes to a ground-truth pose
// topic and, after each frame's state estimation, matches it against the
// pipeline's own estimated pose to accumulate a trajectory comparison.
// Always reports ATE (Absolute Trajectory Error, after a rigid Kabsch/Horn
// SE3 alignment between the accumulated estimated and ground-truth
// position trajectories -- no scale, since this is a metric LIO/VIO
// system, not monocular); RTE (relative translation error) and ROE
// (relative rotation error), RPE's two parts, are computed only if their
// respective flags are enabled, since ROE in particular is meaningless
// against a position-only ground truth (e.g. NTU VIRAL's Leica topic,
// whose orientation field is always identity).
class EvoProc
{
public:
  explicit EvoProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  // Advances the ground-truth buffer against mg's estimate and, for every
  // GT sample whose bracketing RAW estimate frames (before/after) are close
  // enough in time, commits a correspondence in one or both ASSOCIATION
  // MODES and recomputes the enabled metrics for each. Returns a log
  // string, or empty if evo is disabled or nothing new was committed this
  // call.
  //
  // Standardized convention for BOTH gt_source modes: of the two ESTIMATE
  // frames bracketing a GT sample (last frame strictly before it, cached;
  // first frame at/after it, this call's mg), a correspondence is built
  // TWICE, exactly mirroring FAST-LIVO2's own post-hoc evaluator
  // (fastlivo_evo.py, which always scores both --mode interp and --mode
  // nearest for a given odometry file -- see run_evo.sh's score_variant()):
  //   "nearest": whichever bracketing frame is closer in time to the GT
  //     sample is used exactly as-is (no interpolation); dropped if that
  //     gap exceeds opts_.max_time_diff. Matches
  //     evo.core.sync.associate_trajectories'/fastlivo_evo.py's
  //     lookup_nearest() own behavior exactly.
  //   "interp": the estimate is lerp(position)/slerp(rotation)-interpolated
  //     to the GT sample's own exact timestamp; dropped if the two
  //     bracketing frames are more than 2*opts_.max_time_diff apart.
  //     Matches fastlivo_evo.py's lookup_est() exactly. Skipped entirely
  //     for gt_source == "file" (see gt_source's doc comment).
  // Each mode's correspondences accumulate into their own independently-
  // aligned trajectory (see StageMatches) and get their own logStage() line
  // in /tmp/evo.txt, tagged mode=nearest/mode=interp -- an earlier version
  // of this module computed nearest-only (having previously computed
  // interp-only before that), reasoning that only one mode could be "the"
  // canonical comparison against evo; the numbers that actually matter for
  // cross-pipeline comparison against FAST-LIVO2 are BOTH of fastlivo_evo.py's
  // own outputs, so both are now computed here, unconditionally.
  //
  // gt_source == "file": see processEvoFileMode() -- sparse checkpoints
  // (bracket_cache_ as the "before" bracket), routed here directly.
  //
  // gt_source == "topic": see processEvoTopicMode() -- a dense, LIVE-
  // growing GT buffer (populated asynchronously by gtCallback()), same
  // bracket-and-commit structure, generalized to possibly commit several
  // GT samples per call (if GT publishes faster than the estimate
  // updates) or none at all (if the estimate updates faster than GT
  // publishes -- the common case, and expected).
  //
  // Either way, a given call usually returns empty (no GT sample newly
  // bracketed this tick) and only returns a fresh string on calls that
  // close at least one bracket in at least one mode.
  std::string processEvo(MeasureGroup& mg);

  // gt_source == "file" only, and now essentially a no-op kept for the
  // LivoReconNode::run() call site (see below): with interpolation,
  // commitMatch() fires as soon as the "after" bracket frame is seen, so
  // there's no longer a "pending, window still open" state to flush at
  // run-end the way the old nearest-single-frame version had. The only
  // thing that can be left uncommitted when the run ends is a checkpoint
  // whose timestamp falls AFTER the very last estimate frame ever
  // produced -- i.e. it has a "before" bracket but will never get an
  // "after" one. That's the same "no bracket -> skip, don't extrapolate"
  // edge case bracket_cache_ handles at the start of the run, just at the
  // opposite end -- there is deliberately no code path here (or anywhere)
  // that extrapolates a checkpoint's pose from only one side. Always
  // returns empty. Intended to be called once, after the pipeline's main
  // processing loop exits -- see LivoReconNode::run()'s existing
  // pub_proc_.exportColmap() / ctx_.profiler->exportReport() calls in that
  // same spot, which this mirrors (no destructor/signal-handler shutdown
  // hook exists in this codebase to piggyback on instead).
  std::string finalizePendingFileMatch();

private:
  struct GtSample
  {
    double t;
    V3D    pos;
    M3D    rot;  // identity if the topic doesn't provide a meaningful orientation
  };

  struct MatchedPose
  {
    double t;
    V3D    est_pos;
    M3D    est_rot;
    V3D    gt_pos;
    M3D    gt_rot;
  };

  // One pipeline stage's estimate at a single instant, used to build one
  // GT sample's correspondence -- see EstimateSnapshot's two flavors below
  // (nearest-picked raw pose vs. interpolated pose), built once per
  // GT-sample-bracket in processEvoFileMode()/processEvoTopicMode() and fed
  // to commitMatch().
  struct EstimateSnapshot
  {
    V3D imu_pos = V3D::Zero(), lio_pos = V3D::Zero(), vio_pos = V3D::Zero();
    M3D imu_rot = M3D::Identity(), lio_rot = M3D::Identity(), vio_rot = M3D::Identity();
  };

  // A pipeline stage's trajectory accumulates separately per association
  // mode (see processEvo()'s doc comment) -- "nearest" and "interp" are
  // scored as two entirely independent trajectories/alignments, exactly as
  // fastlivo_evo.py's two separate --mode invocations would, not as one
  // trajectory with two candidate poses per sample.
  struct StageMatches
  {
    std::vector<MatchedPose> nearest;
    std::vector<MatchedPose> interp;
  };

  // Parses a TUM-format file (see EvoProcOptions::gt_file) into gt_buffer_
  // once at startup, timestamps left in their raw absolute epoch (the same
  // convention gtCallback() shifts by data_queues_->start_time -- but
  // start_time isn't known yet this early, so the shift is applied lazily,
  // once, on the first processEvo() call instead -- see
  // shiftGtFileTimestampsOnce()). Returns false (and logs via ROS_ERROR) if
  // the file can't be opened or contains no parseable rows.
  bool loadGtFile(const std::string& path);

  // gt_source == "file" only: subtracts data_queues_->start_time from every
  // buffered sample's timestamp exactly once, the first time it's safe to
  // (start_time is set once, at the pipeline's first real sensor message --
  // not yet known when loadGtFile() runs during loadParameters()). A no-op
  // on every call after the first.
  void shiftGtFileTimestampsOnce();

  // gt_source == "topic" only: drains DataQueues::popGt() (populated by
  // CbkProc::gtCallback(), live subscriber or runOffline() bag dispatch --
  // see that function's doc comment) into gt_buffer_ each processEvo()
  // tick, replacing what used to be a live ROS-subscriber callback owned
  // directly by this class.
  void drainGtQueue();

  // True once a buffered ground-truth sample's orientation has been seen to
  // differ from identity -- i.e. the gt_topic actually reports a real
  // attitude (not every source does: e.g. NTU VIRAL's Leica prism tracker
  // is position-only, so its pose messages carry an always-identity
  // orientation field as a placeholder). ROE/ARE computed against a
  // never-varying identity "ground truth" don't measure orientation error
  // at all -- they measure how far the (ATE-aligned) estimate's own
  // orientation happens to sit from the identity matrix, which is a
  // meaningless number dressed up as one. Gates computeRoe/computeAre in
  // logStage() regardless of opts_.compute_roe, so a test script enabling
  // that flag against a position-only GT source can't produce a
  // misleading-looking value.
  bool gtOrientationMeaningful() const { return gt_orientation_seen_; }

  // Recomputes the rigid (rotation+translation, no scale) alignment
  // between `matched`'s accumulated estimated and ground-truth position
  // trajectories via Kabsch/Horn SVD alignment, and returns the ATE RMSE
  // (meters) after applying it. Returns -1 if too few matched poses have
  // been accumulated yet to align meaningfully. Takes `matched` as a
  // parameter (rather than an implicit member) so the same math can be
  // run independently per pipeline stage -- see the matched_imu_/
  // matched_lio_/matched_vio_ members and processEvo()'s logStage().
  double computeAte(const std::vector<MatchedPose>& matched, M3D& R_align_out, V3D& t_align_out) const;

  // ARE (Absolute Rotation Error): RMSE (degrees) of the geodesic angle
  // between each matched pose's estimated orientation (rotated by the ATE
  // alignment's R_align, so it shares gt's world frame) and its ground-
  // truth orientation -- the rotation analog of ATE (an absolute,
  // whole-trajectory comparison), as opposed to ROE (a RELATIVE comparison
  // between pairs of poses ~rpe_delta_s apart, where the alignment's
  // rotation cancels out and isn't needed). Meaningless against a
  // position-only ground truth whose orientation field is always identity
  // (e.g. NTU VIRAL's Leica topic) for the same reason ROE is.
  double computeAre(const std::vector<MatchedPose>& matched, const M3D& R_align) const;

  // RTE: RMSE (meters) of evo's PoseRelation.point_distance -- the
  // difference between the estimated and ground-truth relative
  // displacement MAGNITUDES (||pos_j - pos_i||, not a vector difference)
  // over pairs of matched poses delta_frames apart. No alignment input is
  // needed: a common rigid transform applied to both matched[i]/matched[j]
  // cancels out of this magnitude regardless (rotation preserves norm,
  // translation cancels in the difference), unlike ATE/ARE. delta_frames
  // is an INDEX offset, not a time window -- see logStage()'s conversion
  // from opts_.rpe_delta_s, matching evo's own Unit.frames convention
  // (evo's id_pairs_from_delta has no working seconds unit). Returns -1 if
  // delta_frames >= matched.size().
  double computeRte(const std::vector<MatchedPose>& matched, int delta_frames) const;

  // ROE: RMSE (degrees) of the relative-rotation angle between the
  // estimated and ground-truth relative rotation over the same
  // delta_frames-apart pose pairs. The ATE alignment rotation cancels out
  // of a relative-rotation comparison, so it isn't needed here. Returns -1
  // if delta_frames >= matched.size().
  double computeRoe(const std::vector<MatchedPose>& matched, int delta_frames) const;

  struct StageMetrics { double ate = -1.0, rte = -1.0, roe = -1.0, are = -1.0; };

  // Appends (t_abs, est_pos, est_rot, gt_pos, gt_rot) to `matched` (one
  // stage's independently-accumulated trajectory, for ONE association
  // mode), recomputes that stage/mode's own Kabsch alignment and
  // ATE/RTE/ROE/ARE from scratch against it, and appends one line to
  // /tmp/evo.txt (stage name, mode name, absolute timestamp, n, and
  // whichever of ATE/RTE/ROE/ARE are available). Each (stage, mode) pair is
  // aligned independently (not sharing one pipeline-wide alignment) so
  // e.g. raw IMU-propagation drift and VIO's corrected trajectory, or the
  // interp- and nearest-mode trajectories of the same stage, aren't forced
  // through the same rigid transform -- a fair "how good is this
  // stage/mode's trajectory on its own" comparison, matching how
  // fastlivo_evo.py's own separate --mode interp/--mode nearest
  // invocations independently align/score each of their own trajectories.
  StageMetrics logStage(const char* stage_name, const char* mode_name, double t_abs,
                        std::vector<MatchedPose>& matched,
                        const V3D& est_pos, const M3D& est_rot, const V3D& gt_pos, const M3D& gt_rot) const;

  // Given one already-resolved GT sample and BOTH of its candidate estimate
  // snapshots (nearest_valid/nearest: the nearest-RAW-timestamp pick;
  // interp_valid/interp: the lerp/slerp-interpolated pose -- either may be
  // invalid/unused independently, per opts_.max_time_diff's two separate
  // gates, see processEvo()'s doc comment), runs whichever mode(s) are
  // valid through all 3 stages via logStage() and builds the same
  // console/overlay output processEvo() has always built -- shared by both
  // gt_source branches (processEvoTopicMode()'s per-GT-sample commits and
  // processEvoFileMode()'s per-checkpoint commits) so neither duplicates
  // this tail logic. Updates last_stats_nearest_/last_stats_interp_/
  // mg.evo_stats as a side effect. gt_source == "file" always passes
  // interp_valid == false (see gt_source's doc comment) -- `interp` is
  // still taken as a parameter (rather than overloading) so the call sites
  // stay symmetric and the "always both modes, skip whichever gate fails"
  // logic lives in one place.
  std::string commitMatch(MeasureGroup* mg, double t_abs,
                          bool nearest_valid, const EstimateSnapshot& nearest,
                          bool interp_valid,  const EstimateSnapshot& interp,
                          const V3D& gt_pos, const M3D& gt_rot);

  // Shared by both gt_source modes (only one is ever active in a given
  // run -- gt_source is fixed at startup -- so a single cache/cursor pair
  // is reused rather than duplicated per mode).
  //
  // gt_source == "file": sparse checkpoints are 15-40s apart, spanning a
  // window that can contain many estimate frames (one per stage tick).
  // This caches the most recent estimate frame seen so far (the "before"
  // bracket for whichever checkpoint comes next) and, the moment a frame
  // lands AT/AFTER a checkpoint's timestamp (the "after" bracket), builds
  // the "nearest" EstimateSnapshot (whichever of "before"/"after" is
  // closer in time to the checkpoint, its RAW pose, no interpolation) --
  // exactly evo.core.sync.associate_trajectories'/fastlivo_evo.py's own
  // lookup_nearest() behavior. "interp" is never attempted here (see
  // gt_source's doc comment -- HILTI's own evaluation.py nearest-matches
  // its sparse checkpoints too, never interpolates).
  //
  // gt_source == "topic": same structure, generalized to a dense/growing
  // GT buffer -- see processEvoTopicMode(). The cache here plays the exact
  // same "before" bracket role; the difference is purely in how many GT
  // samples get bracketed per call (usually 0 or 1 here as well, since a
  // checkpoint-spaced GT buffer rarely has more than one entry appear
  // between estimate frames, but processEvoTopicMode() doesn't rely on
  // that -- see its own doc comment).
  struct BracketCache
  {
    double t = 0.0;  // mg.image.t (start_time-relative, same basis as gt_buffer_ post-shift)
    V3D imu_pos = V3D::Zero(), lio_pos = V3D::Zero(), vio_pos = V3D::Zero();
    M3D imu_rot = M3D::Identity(), lio_rot = M3D::Identity(), vio_rot = M3D::Identity();
    // False until the first estimate frame this run has ever seen. If a
    // GT sample's timestamp already falls before that first frame is
    // cached (no "before" bracket exists to interpolate from), that GT
    // sample is skipped outright rather than extrapolated backward.
    bool valid = false;
  };

  // gt_source == "file" only: processes one frame against the sparse
  // checkpoint buffer -- for the checkpoint at next_gt_idx_ (and, in the
  // rare case of a big gap in incoming frames, any subsequent checkpoints
  // this same frame also lands at/after), builds the "nearest"
  // EstimateSnapshot from bracket_cache_ (the "before" bracket) and this
  // frame (the "after" bracket) and commits it (interp_valid always
  // false -- see gt_source's doc comment), advancing next_gt_idx_ past
  // each one committed (or skipped, per BracketCache's edge case). Always
  // updates bracket_cache_ to this frame before returning, so it's ready
  // to serve as the "before" bracket for whichever checkpoint comes next.
  // Returns the same kind of log string processEvo() does (empty if
  // nothing was committed this call).
  std::string processEvoFileMode(MeasureGroup& mg);

  // gt_source == "topic" only: same bracket-and-commit structure as
  // processEvoFileMode(), inverted in which side is dense/growing --
  // gt_buffer_ here is populated live and asynchronously by gtCallback()
  // (a subscriber callback firing independently of this function's own
  // calls), rather than loaded once at startup. Advances next_gt_idx_
  // through gt_buffer_ and, for EVERY GT sample whose timestamp falls
  // between bracket_cache_'s cached "before" frame and this call's mg
  // (the "after" frame) -- there can be more than one, if the GT topic
  // publishes faster than the estimate updates -- builds BOTH candidate
  // EstimateSnapshots (nearest: raw before/after pick; interp: lerp
  // position/slerp orientation between those same two bracketing frames
  // to synthesize a pose at the sample's exact timestamp) and commits
  // whichever gate(s) opts_.max_time_diff allows (see processEvo()'s doc
  // comment) via commitMatch(). If the estimate updates faster than GT
  // publishes (the more typical case for a rate like NTU VIRAL's ~15-20Hz
  // Leica topic), most calls find zero new samples to commit and simply
  // update bracket_cache_ to this frame before returning empty -- expected
  // and fine, mirroring file mode's own sparser-than-you'd-think match
  // rate. Ground truth itself needs no interpolation in this scheme --
  // gt_buffer_[idx]'s own stored orientation is used as-is, at its own
  // real timestamp, which is the entire point of interpolating the
  // estimate instead.
  std::string processEvoTopicMode(MeasureGroup& mg);

  // Shared lerp(position)/slerp(rotation) helper for "interp" mode -- see
  // EstimateSnapshot and processEvo()'s doc comment. alpha in [0,1], where
  // alpha=0 reproduces `before` and alpha=1 reproduces `after`.
  static M3D slerpRot(const M3D& before, const M3D& after, double alpha);

  // Formats one association mode's "[evo/mode_name]" console line + rviz-
  // overlay table from that mode's own VIO StageMetrics/matched-count, and
  // updates last_stats_slot (last_stats_nearest_ or last_stats_interp_) as
  // a side effect. Returns empty if vio.ate < 0 (not enough matches yet to
  // align) -- last_stats_slot is still updated to an "accumulating"
  // placeholder in that case. Shared by commitMatch()'s nearest/interp
  // passes so neither duplicates this formatting.
  static std::string formatModeStats(const char* mode_name, const StageMetrics& vio, size_t n,
                                     double rpe_delta_s, bool gt_orientation_meaningful,
                                     std::string& last_stats_slot);

  size_t       next_gt_idx_ = 0;      // see processEvoFileMode()/processEvoTopicMode()
  BracketCache bracket_cache_;        // see BracketCache

  StateGroupPtr    state_;
  ProfilerPtr      profiler_;
  DataQueuesPtr    data_queues_;  // for start_time, and popGt() -- see drainGtQueue()

  EvoProcOptions opts_;

  std::deque<GtSample> gt_buffer_;
  bool                 gt_orientation_seen_ = false;  // see gtOrientationMeaningful()
  bool                 gt_file_ts_shifted_ = false;    // see shiftGtFileTimestampsOnce()

  // One independently-aligned trajectory per pipeline stage PER ASSOCIATION
  // MODE (see StageMatches -- LivoReconNode::estimateState()'s
  // mg.pos_after_imu/lio/vio snapshots feed both .nearest and .interp
  // independently). matched_vio_ is also what feeds the console/overlay
  // ATE/RTE/ROE/ARE (the final refinement step, per explicit instruction --
  // "use the vio ate for right now") -- both modes' VIO numbers are shown.
  StageMatches matched_imu_;
  StageMatches matched_lio_;
  StageMatches matched_vio_;

  // Most recently computed mg.evo_stats block for each mode (see
  // processEvo()/commitMatch()) -- carried forward onto every frame's mg
  // (concatenated) even on frames that don't get a fresh ground-truth match
  // this exact tick, so the rviz overlay updates continuously instead of
  // going blank/stale on the (common) frames that don't happen to produce a
  // new number for either mode.
  std::string last_stats_nearest_ = "mode=nearest  accumulating (need >= 3 matched poses)\n";
  std::string last_stats_interp_  = "mode=interp   accumulating (need >= 3 matched poses)\n";
};

}  // namespace livo_recon
