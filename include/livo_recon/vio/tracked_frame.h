#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "livo_recon/utils/algo/math.h"  // V3D/M3D -- lightweight (Eigen/Core only), no cycle risk

// Deliberately minimal includes (no common_lib.h) -- this header is
// included from BOTH vio/tracker.h and utils/data/measures.h (MeasureGroup
// carries a TrackedFrame as of task #145's async tracking pipeline), and
// common_lib.h itself includes measures.h, so pulling IT in here would
// create a #include cycle. Eigen/Dense itself is fine (no livo_recon
// dependency of its own).

namespace livo_recon
{

// A single tracked point's fixed image-space anchor: the pixel it was
// picked at (seed-time image), used as the epipolar residual's reference
// point against its currently-tracked pixel (see
// VioProc::computeEpipolarResidualAndJacobian).
//
// This replaces an earlier design that anchored each tracked point to a
// real LIDAR 3D point (body-frame position + covariance) and ran a
// metric-scale reprojection residual against it -- ported from FAST-LIVO2
// (livo_vio/tracker.h), where a point-to-point reprojection residual
// against LIDAR-anchored 3D points was tried and removed: its per-iteration
// corrections were uncorrelated (mixed LIDAR+visual config) or
// anti-correlated (LIDAR-only config) with actually moving the state closer
// to ground truth, unlike the epipolar residual's clearly corrective
// behavior. See VioProc's own header comment for the residual this anchor
// feeds.
struct AnchorPoint
{
  cv::Point2f uv_seed;  // the pixel this anchor was picked at (seed-time image)

  // Local 2x2 Jacobian d(normalized_bearing)/d(raw_pixel) at uv_seed's RAW
  // (pre-undistortion) pixel location -- i.e. the derivative of the full
  // raw-pixel -> undistorted-pixel -> ((u-cx)/fx,(v-cy)/fy) chain, evaluated
  // via central finite differences on Tracker::undistortPixel() (see
  // Tracker::localPixelToUVJacobian()). Feeds VioProcOptions::
  // distortion_weight_on's rigorous per-point weight (propagates an assumed
  // weight_pixel_noise_px pixel-space Gaussian uncertainty into the
  // normalized-coordinate residual's variance) -- identity when no
  // distortion model is loaded (undistortPixel() is then itself an
  // identity map, scaled by 1/fx,1/fy). Computed unconditionally at seed
  // time regardless of whether distortion_weight_on is set, mirroring
  // FAST-LIVO2's tracker.cpp (localPixelToUVJacobian call sites) -- cheap
  // relative to the tracking backend itself.
  Eigen::Matrix2d uv_jacobian_seed = Eigen::Matrix2d::Identity();
};

// Fully self-contained result of Tracker::stepAndReseedOnce() for ONE
// image -- everything VioProc::processVIO() needs, with no further live
// reads of Tracker's own mutable anchors_/uv_curr_ members. This exists
// because tracking now runs on its own background thread (see task #145:
// decouple tracker step()/reseed() from blocking ros::spinOnce()) --
// `anchors` is a snapshot of the anchor set as of BEFORE this call's own
// reseed (what `indices` refers into), since the async thread overwrites
// Tracker::anchors_ with a NEW candidate set as part of the same call,
// which would otherwise race with (or simply postdate) the main thread's
// use of the OLD set for this frame's EKF correction.
//
// Carried as a plain field on MeasureGroup (see measures.h), populated by
// CbkProc::syncMeasures() the moment a measure group is formed (which
// itself only proceeds once the tracking result for that group's image is
// confirmed ready -- see DataQueues::ready()/Tracker::waitReadyFor() call
// site there) -- so by the time VioProc::processVIO() reads
// mg.tracked_frame, it's simply already there, no wait/consume call
// needed at that point at all.
struct TrackedFrame
{
  // The image timestamp (bag/wall-clock seconds, matching CbkProc::
  // imageCallback's img_time) this result was produced for -- set by
  // Tracker::asyncTrackingLoop() from the same (image, timestamp) pair
  // feedImage() enqueued. Not used by VioProc (which always consumes
  // results strictly in feed order, see Tracker's class doc comment), but
  // needed for TrackerCacheWriter/TrackerCacheReader (task: offline+
  // caching support) to index/match cached records by timestamp the same
  // way FAST-LIVO2's .trackcache format does (findByCurrTimestamp).
  double timestamp = -1.0;

  // Timestamp of the frame whose reseed produced `anchors`/`indices`
  // below -- -1.0 sentinel = the bootstrap seed image. Set by
  // Tracker::asyncTrackingLoop() (a simple chain: each frame's
  // prev_timestamp is the previous fed frame's own timestamp). Used by
  // Tracker::resolveAnchorPose() to look up rot_anchor/pos_anchor from
  // the pose-history recorded by recordFramePose() -- see Tracker's own
  // doc comment for the full design (ported from FAST-LIVO2's
  // LivoVioManager::anchor_Rt_/TrackedFrame::prev_timestamp).
  double prev_timestamp = -1.0;

  bool ok = false;                        // mirrors step()'s own return value; uv_curr/indices only meaningful when true
  std::vector<cv::Point2f> uv_curr;
  // Parallel to uv_curr -- see AnchorPoint::uv_jacobian_seed's doc comment,
  // same quantity but evaluated at each point's CURRENT raw tracked pixel
  // (computed fresh every frame, unlike the anchor's own Jacobian which is
  // fixed at seed time).
  std::vector<Eigen::Matrix2d> uv_curr_uv_jacobian;
  std::vector<int> indices;               // indices into `anchors` below (NOT into any live Tracker state)
  std::vector<AnchorPoint> anchors;       // the anchor set `indices` refers to (as of BEFORE this call's own reseed)

  // The pose the epipolar residual's anchor-frame geometry is relative to
  // -- resolved by Tracker::resolveAnchorPose() (via prev_timestamp above)
  // BEFORE this frame reaches any residual computation, NOT set by
  // stepAndReseedOnce()/asyncTrackingLoop() (the async thread has no
  // access to a pose that doesn't exist yet).
  M3D rot_anchor = M3D::Identity();
  V3D pos_anchor = V3D::Zero();

  // True if this call performed a reseed (a NEW anchor set is now live in
  // Tracker::anchors_, for the NEXT tracked frame to step() against) --
  // diagnostic only (logged in [vio/frame_stop] etc.); no longer drives
  // any pose-resolution logic (see rot_anchor/pos_anchor above).
  bool reseeded = false;
  std::string log;
};

}  // namespace livo_recon
