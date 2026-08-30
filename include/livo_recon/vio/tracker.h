#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>

#include "livo_recon/common_lib.h"
#include "livo_recon/vio/point_tracker.h"
#include "livo_recon/vio/tracked_frame.h"
#include "livo_recon/vio/tracker_cache.h"

namespace livo_recon
{

struct TrackerOptions
{
  // "cotracker", "tapnext", or "trackon" -- which PointTracker backend
  // (see vio/point_tracker.h) to run. Each backend's model file paths
  // are fixed (see tracker.cpp) rather than configurable, since they're
  // paired 1:1 with the backend itself.
  std::string backend = "cotracker";

  std::string device = "cuda";

  // CoTracker-only (ignored by tapnext/trackon): "jit" (default), "aot"
  // (AOTInductor + fp16), or "aot_fp32" (AOTInductor, full precision) --
  // see tracker.cpp's model_path_a/model_path_b selection.
  std::string tracker_format = "jit";

  // Grid-bucketed selection of the persistent tracked-point set's anchors
  // (see Tracker::selectAnchors): each grid_rows x grid_cols cell picks the
  // image's own best Shi-Tomasi corner (cv::cornerMinEigenVal) within that
  // cell as its anchor -- purely image-based, no LIDAR depth is needed (the
  // epipolar residual only ever needs a 2D seed pixel). A cell whose crop
  // lands entirely outside the valid image region contributes no anchor, so
  // the tracked-point count is variable (up to grid_rows*grid_cols), not
  // fixed.
  int    grid_rows            = 10;
  int    grid_cols            = 10;

  // ── Reseed sub-options (tracker/reseed/* in tracker.yaml) ───────────
  // Once fewer than this fraction of the currently seeded grid remain
  // visible, the whole grid is due for a reseed (see PointTracker::
  // needsReseed()). CoTrackerBackend ignores this entirely (its
  // needsReseed() always reports true: it has no persistent state worth
  // preserving across frames, see cotracker_backend.h).
  double reseed_active_fraction = 0.5;

  // Minimum fraction of grid_rows*grid_cols selectAnchors() must find a
  // candidate for before a reseed is actually committed to -- below this,
  // reseeding is skipped entirely and whatever's currently tracked (if
  // anything) is left untouched for another frame, rather than committing
  // to a small/degraded anchor set. NOTE: the fraction of grid cells that
  // get a candidate under normal, healthy conditions drops as
  // grid_rows*grid_cols grows, so this needs to be retuned lower when
  // increasing grid resolution -- a value tuned for a 32x32 grid will
  // likely reject perfectly healthy reseeds at 64x64.
  double min_candidate_fraction = 0.25;

  // Debug viz: draw an arrow from each tracked point's previous to current
  // position (see Tracker::drawUvTracking), on demand via
  // maybeDrawUvTracking() -- called from PubProc, not stored per-frame.
  bool viz_uv_tracking = false;

  // Path to a .rlivtrackcache file produced by the cache_tracker_output_
  // livo_recon tool (src/tools/cache_tracker_output.cpp). Empty (the
  // default) means live tracking, unchanged. When set, asyncTrackingLoop()
  // looks each fed frame's timestamp up in the cache instead of calling
  // backend_->step()/reseed() -- skips the GPU tracker pass entirely, same
  // motivation and design as FAST-LIVO2's LivoVioOptions::track_cache_path
  // (see that option's doc comment): step()/reseed() are a pure function
  // of (bag, grid_rows, grid_cols, tracker_format, backend) with no
  // pose/state dependence (confirmed -- see tracker.cpp, state_ is only
  // read for fixed camera intrinsics/distortion and in recordFramePose(),
  // both outside step()/reseed()'s own call path), so every ablation sweep
  // that only varies downstream (state-dependent) VioProc params can reuse
  // one cached tracking pass instead of re-running the identical, GPU-
  // expensive tracker call on every single run.
  std::string track_cache_path;
};

// Owns the vio point-tracking subsystem as a single ctx_-level object
// (see NodeContext::tracker), mirroring VoxelMap: holds the
// state, the underlying PointTracker backend (CoTracker/TAPNext/Track-On),
// and anchor selection -- everything VioProc needs from "the tracker" that
// isn't itself part of the EKF residual/update logic.
//
// ── Async tracking pipeline (task #145) ──────────────────────────────────
// step()+reseed() (stepAndReseedOnce(), below) run on a dedicated
// background thread, NOT on the caller's thread -- both are genuinely
// expensive (CoTracker/TAPNext/Track-On neural-net inference), and calling
// them synchronously from LivoReconNode::run()'s main loop previously
// blocked that loop's own ros::spinOnce() for however long each call took,
// starving ROS callback processing (message queues) for the whole
// duration.
//
// feedImage() is called from CbkProc::imageCallback() -- the INSTANT an
// image message arrives, before it's even pushed onto DataQueues -- and
// returns immediately; the actual work happens on the background thread.
// CbkProc::syncMeasures() (which decides when a real MeasureGroup is
// formed) checks waitReadyFor(0ms) -- an instant, non-blocking peek, NOT a
// real wait -- right alongside its existing DataQueues::ready() check, and
// returns false (defer to the next call, exactly like "lidar/imu not
// settled yet" already does) if tracking isn't done yet for the image at
// the front of the queue. Since the run() loop's outer iteration already
// re-runs ros::spinOnce() before retrying syncMeasures(), this achieves
// the exact same guarantee as a genuine bounded wait+spinOnce() loop (ROS
// callbacks never stall for the duration of a slow tracker call) with NO
// separate wait loop anywhere and no risk of the busy-poll-vs-blocking
// mistake FAST-LIVO2's own livo_vio_manager.h waitReadyFor() doc comment
// describes a real production incident from (a busy-poll variant pegged
// 6-8 CPU cores and starved rosbag's real-time publisher thread badly
// enough to desync the IMU timestamp stream and diverge the EKF
// catastrophically) -- there's simply no busy-poll to get wrong here,
// since the "poll" is the SAME already-existing, already-correctly-paced
// syncMeasures() retry cadence. Once ready, syncMeasures() calls
// consumeResult() and stores it directly on the MeasureGroup it forms
// (see measures.h's tracked_frame field) -- VioProc::processVIO() just
// reads mg.tracked_frame, no wait/consume call of its own needed at all.
//
// Feed and consume are always 1:1, in strict arrival order (feedImage() is
// called for every image pushed to DataQueues, in the same order
// DataQueues pops them) -- multiple images CAN be in flight at once (e.g.
// if measure groups back up), unlike an earlier version of this design
// that assumed strictly one at a time, but consumeResult()'s simple
// pop-front is still correct regardless, since both queues advance in
// that same shared order.
class Tracker
{
public:
  Tracker(StateGroupPtr state, ProfilerPtr profiler);

  std::string loadParameters(ros::NodeHandle& pnh);

  // True once the underlying backend loaded successfully (live mode) OR
  // the track cache loaded successfully (replay mode -- see
  // TrackerOptions::track_cache_path).
  bool loaded() const { return static_cast<bool>(backend_) || static_cast<bool>(cache_reader_); }
  bool replayMode() const { return static_cast<bool>(cache_reader_); }

  // ── PointTracker pass-throughs ──────────────────────────────────────
  // hasPrev(): whether there's a real previous frame to track against at
  // all (the sole gate on whether stepAndReseedOnce() attempts step() --
  // see task #145, this and needsReseed() below are the ONLY conditions
  // left; no camera-motion/baseline gate).
  bool hasPrev() const { return backend_ && backend_->hasPrev(); }
  bool needsReseed() const { return backend_ && backend_->needsReseed(); }

  // ── Async tracking pipeline ──────────────────────────────────────────
  // Starts the background tracking thread. Call exactly once, after the
  // backend has loaded (loadParameters()) and before any feedImage() call.
  void startAsyncTracking();

  // Stops the background thread cleanly. Call exactly once, at shutdown,
  // before destruction.
  void stopAsyncTracking();

  // True once startAsyncTracking() has actually started the background
  // thread -- lets a caller (CbkProc::syncMeasures()) tell "pipeline
  // active, gate on it" apart from "never started" (VIO disabled or
  // tracker failed to load), where waitReadyFor()/feedImage() should be
  // skipped entirely rather than gating forever on a queue nothing will
  // ever fill.
  bool asyncTrackingActive() const { return tracking_thread_.joinable(); }

  // Enqueues ONE new image for the async pipeline to step()+reseed()
  // against. Returns immediately; the actual work happens on the
  // background thread. Silently no-ops if the pipeline hasn't been
  // started yet (startAsyncTracking() not called, or tracker not loaded).
  void feedImage(const cv::Mat& img, double timestamp);

  // Checks whether at least one tracking result is ready to consume.
  // `timeout` of zero (the only value this codebase actually uses -- see
  // this class's own doc comment) makes this a genuine instant,
  // non-blocking peek: check the queue once, return immediately either
  // way, no sleeping. A nonzero timeout would turn this into a real
  // bounded condition-variable wait (mostly asleep, woken instantly on
  // real readiness) if some future caller ever needed one.
  bool waitReadyFor(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(result_mutex_);
    return result_cv_.wait_for(lock, timeout, [this] {
      return !result_queue_.empty() || stop_requested_.load();
    });
  }

  // Pops the oldest ready tracking result into `out`. Returns false (out
  // left default-constructed) if nothing is ready yet -- callers should
  // have already confirmed readiness via waitReadyFor().
  bool consumeResult(TrackedFrame& out);

  // Pose-history mechanism (2026-08-14, ported from FAST-LIVO2's
  // LivoVioManager::anchor_Rt_) -- replaces the previous caller-driven
  // "commit now" pattern (Tracker used to own mutable rot_anchor_/
  // pos_anchor_, updated via a commitAnchorPose() call every caller had to
  // remember to make at exactly the right time; a NEW caller (CombinedProc)
  // forgetting to do so was a real bug this session). Instead:
  //   - recordFramePose() is called ONCE per frame by LivoReconNode::
  //     estimateState(), AFTER that frame's correction (sequential or
  //     combined, whichever path ran) has fully finished -- pushes
  //     (timestamp, state_->rot(), state_->pos()) onto a FIFO history.
  //   - resolveAnchorPose() is called ONCE per frame, BEFORE either
  //     correction path runs, and looks up frame.prev_timestamp in that
  //     same history (FIFO front -- frames are consumed and recorded in
  //     strict timestamp order) to fill frame.rot_anchor/pos_anchor.
  // No caller-side "did I remember to commit" contract exists anymore --
  // any consumer of a TrackedFrame just reads rot_anchor/pos_anchor,
  // already resolved.
  void recordFramePose(double timestamp);
  void resolveAnchorPose(TrackedFrame& frame);

  // ── Anchor selection ─────────────────────────────────────────────────
  // Grid-bucketed Shi-Tomasi anchor selection (see grid_rows/grid_cols
  // above): per grid cell, the image's own best corner
  // (cv::cornerMinEigenVal) within that cell's crop. Purely image-based --
  // no LIDAR depth is needed, since the epipolar residual only ever needs a
  // 2D seed pixel per anchor (see VioProc::computeEpipolarResidualAndJacobian).
  // A cell whose crop lands entirely outside the valid image region
  // contributes nothing, so the returned vector's length is variable (<=
  // grid_rows*grid_cols), not fixed. Called only from the background
  // tracking thread (via stepAndReseedOnce()) once that thread is running.
  std::vector<cv::Point2f> selectAnchors(const cv::Mat& image,
                                        std::vector<AnchorPoint>& anchors_out) const;

  // Draws an arrow from each uvPrev()[i] to uvCurr()[i] on a copy of
  // `image`, colored by uvPrev()[i]'s position (a polar color wheel around
  // the image center: hue encodes direction from center, value encodes
  // distance).
  cv::Mat drawUvTracking(const cv::Mat& image) const;

  // Returns drawUvTracking(image) if opts_.viz_uv_tracking and there's
  // something to draw, else an empty cv::Mat -- called from
  // PubProc::publishImage. NOTE (task #145): uv_prev_/uv_curr_ are now
  // written by the BACKGROUND tracking thread (inside stepAndReseedOnce());
  // this is still called from the main thread, same as before. In
  // practice this codebase's single-image-in-flight usage means the
  // background thread is idle (already pushed its result and is blocked
  // waiting for the next feedImage()) by the time the main thread gets
  // here, so there's no real contention, but unlike uv_curr in
  // TrackedFrame (a proper snapshot), these two members are NOT
  // synchronized against the background thread by a mutex -- viz-only, not
  // used for anything EKF-correction-related (see solveSystem(), which
  // uses TrackedFrame::uv_curr instead).
  cv::Mat maybeDrawUvTracking(const cv::Mat& image)
  {
    cv::Mat vis;
    if (opts_.viz_uv_tracking && !uv_curr_.empty())
      vis = drawUvTracking(image);
    uv_prev_.clear();
    uv_curr_.clear();
    return vis;
  }

  // The center-cropped region (in raw camera pixel coordinates) the
  // backend's model actually sees for a camera of the given resolution.
  cv::Rect validRegion(int cam_width, int cam_height) const { return backend_->validRegion(cam_width, cam_height); }

private:
  // The actual per-image work (step() using the CURRENT anchor set, then
  // conditionally reseed -- see TrackedFrame's own doc comment for why
  // `anchors`/`indices` in the result are a pre-reseed snapshot). Runs
  // ONLY on the background tracking thread once started. Pose-independent
  // (see task #145): hasPrev()/needsReseed() are the sole gates, no
  // camera-motion/baseline check.
  TrackedFrame stepAndReseedOnce(const cv::Mat& image);

  void asyncTrackingLoop();

  // Maps a RAW (distorted) observed pixel to its distortion-corrected
  // pixel-space equivalent (2026-08-04): (u-cx)/fx, (v-cy)/fy is only the
  // correct inverse-pinhole projection for an IDEAL (distortion-free)
  // pixel -- VioProc::computeEpipolarResidualAndJacobian applies exactly
  // that projection to whatever pixel it's given, so feeding it the raw
  // observed pixel directly (as this codebase always had, a bug shared
  // with FAST-LIVO2's livo_vio scaffold this residual model was ported
  // from) bakes in a real geometric bias that grows with distance from
  // the principal point whenever the lens has non-negligible distortion
  // (NTU_VIRAL's cam_d0 is -0.288, not negligible). Fixed HERE, once per
  // point at the source (seed selection / tracked-position snapshot)
  // rather than repeatedly inside the residual function every EKF
  // iteration -- cv::undistortPoints with P=K (the camera matrix) maps
  // the distorted pixel to its true undistorted position and
  // re-projects it back into pixel units via P, so the OUTPUT stays a
  // plain pixel coordinate and every downstream consumer (the epipolar
  // residual's own (u-cx)/fx, drawUvTracking's visualization, etc.)
  // needs no changes at all. Deliberately NOT applied to uv_out in
  // selectAnchors() or to uv_prev_/uv_curr_ (the backend's raw I/O) --
  // those must stay in true (distorted) image pixel space since that's
  // what the actual image data is and what the tracking backend
  // operates against; only the values that end up feeding the epipolar
  // residual (AnchorPoint::uv_seed, TrackedFrame::uv_curr) are corrected.
  cv::Point2f undistortPixel(const cv::Point2f& p) const;

  // Local 2x2 Jacobian d(normalized_bearing)/d(raw_pixel) at a raw pixel,
  // where normalized_bearing is the SAME ((u-cx)/fx,(v-cy)/fy) quantity
  // VioProc's residual math operates in (x0/x1 in
  // computeEpipolarResidualAndJacobian) -- i.e. the derivative of the full
  // raw-pixel -> undistortPixel() -> (subtract principal point, divide by
  // focal length) chain. Feeds VioProcOptions::distortion_weight_on's
  // rigorous per-point weight (error propagation: pixel-space noise ->
  // normalized-coordinate-space noise). Computed via central finite
  // differences directly on undistortPixel() itself (no analytic camera-
  // model Jacobian exists in this codebase, unlike FAST-LIVO2's
  // AbstractCamera::jacobian_2x2() -- see that codebase's
  // LivoVioStateAdapter::localPixelToUVJacobian() doc comment, which notes
  // finite-differencing its own cam2world() as the fallback path for a
  // singular/degenerate analytic Jacobian; this is that same fallback,
  // used unconditionally here since it's the only distortion model
  // available). Identity (scaled by 1/fx,1/fy) when no distortion
  // coefficients are loaded, matching undistortPixel()'s own identity
  // short-circuit in that case.
  Eigen::Matrix2d localPixelToUVJacobian(const cv::Point2f& px) const;

  TrackerOptions opts_;
  StateGroupPtr  state_;
  ProfilerPtr    profiler_;

  std::unique_ptr<PointTracker> backend_;

  // See TrackerOptions::track_cache_path -- non-null iff replay mode is
  // active. asyncTrackingLoop() looks each fed frame's timestamp up here
  // instead of calling stepAndReseedOnce() when set.
  std::unique_ptr<TrackerCacheReader> cache_reader_;

  std::vector<AnchorPoint> anchors_;

  // See recordFramePose()/resolveAnchorPose()'s doc comment above.
  std::deque<std::tuple<double, M3D, V3D>> anchor_pose_history_;

  // Every active track's position as of the last step()/reseed() call and
  // its newly predicted position -- see uvPrev()/uvCurr() callers' own
  // caveats (viz-only, not synchronized -- EKF correction uses
  // TrackedFrame::uv_curr instead).
  std::vector<cv::Point2f> uv_prev_;
  std::vector<cv::Point2f> uv_curr_;

  // ── Async tracking pipeline state ────────────────────────────────────
  std::thread tracking_thread_;
  std::atomic<bool> stop_requested_{false};

  std::mutex input_mutex_;
  std::condition_variable input_cv_;
  std::deque<std::pair<cv::Mat, double>> input_queue_;

  std::mutex result_mutex_;
  std::condition_variable result_cv_;
  std::deque<TrackedFrame> result_queue_;
};

using TrackerPtr = std::shared_ptr<Tracker>;

}  // namespace livo_recon
