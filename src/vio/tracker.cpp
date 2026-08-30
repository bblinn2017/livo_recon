#include "livo_recon/vio/tracker.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "livo_recon/vio/cotracker/cotracker_backend.h"
#include "livo_recon/vio/tapnext/tapnext_backend.h"
#include "livo_recon/vio/trackon/trackon_backend.h"

namespace livo_recon
{

namespace
{

// Each backend's exported module paths are fixed 1:1 with the backend
// itself (see co-tracker/export_pairwise_tracker.py,
// tapnext/export_pairwise_tapnext.py), so they're hardcoded here rather
// than independently configurable -- except CoTracker's own encoder/
// tracker, which additionally vary with tracker_format (see below).
constexpr const char* kCoTrackerEncoderPath = "/root/catkin_ws/src/livo_recon/checkpoints/cotracker_frame_encoder.pt";
constexpr const char* kCoTrackerTrackerPath = "/root/catkin_ws/src/livo_recon/checkpoints/cotracker_pair_tracker.pt";
constexpr const char* kCoTrackerCheckpointDir = "/root/catkin_ws/src/livo_recon/checkpoints";
constexpr const char* kTAPNextInitPath      = "/root/catkin_ws/src/livo_recon/checkpoints/tapnext_init.pt";
constexpr const char* kTAPNextStepPath      = "/root/catkin_ws/src/livo_recon/checkpoints/tapnext_step.pt";
constexpr const char* kTrackOnInitPath      = "/root/catkin_ws/src/livo_recon/checkpoints/trackon_init.pt";
constexpr const char* kTrackOnStepPath      = "/root/catkin_ws/src/livo_recon/checkpoints/trackon_step.pt";

}  // namespace

Tracker::Tracker(StateGroupPtr state, ProfilerPtr profiler)
  : state_(state), profiler_(profiler)
{}

std::string Tracker::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<std::string>(pnh, "tracker/backend", opts_.backend, "cotracker");
  paramWarn<std::string>(pnh, "tracker/device",  opts_.device,  "cuda");
  // "jit" (default) -> cotracker_frame_encoder.pt/cotracker_pair_tracker.pt.
  // "aot" -> a genuine full-precision AOTInductor deployment (see
  // export_encoder_fp32_aot_prod.py/export_tracker_fp32_aot_prod.py in the
  // co-tracker repo). Ported from FAST-LIVO2, where it was validated
  // end-to-end against the jit baseline (within ordinary float-precision
  // noise of jit). An earlier fp16 AOT variant (hard-cast fp16 for both
  // encoder and tracker) was removed: fp32 AOT was added specifically to
  // isolate whether fp16 itself (vs AOTInductor compilation in general)
  // was responsible for any given behavior difference from jit, and fp32
  // AOT is what every actual test/ablation this project has run used.
  paramWarn<std::string>(pnh, "tracker/tracker_format", opts_.tracker_format, "jit");
  paramWarn<int>(pnh, "tracker/grid_selection/grid_rows",              opts_.grid_rows,           10);
  paramWarn<int>(pnh, "tracker/grid_selection/grid_cols",              opts_.grid_cols,           10);
  paramWarn<double>(pnh, "tracker/reseed/active_fraction", opts_.reseed_active_fraction, 0.5);
  paramWarn<double>(pnh, "tracker/reseed/min_candidate_fraction", opts_.min_candidate_fraction, 0.25);
  paramWarn<bool>(pnh, "viz/uv_tracking", opts_.viz_uv_tracking, false);
  // See TrackerOptions::track_cache_path's doc comment -- empty (default)
  // means live tracking, unchanged below.
  paramWarn<std::string>(pnh, "tracker/track_cache_path", opts_.track_cache_path, "");

  std::ostringstream oss;
  oss << "[params/tracker]"
      << "\n  backend:              " << opts_.backend
      << "\n  device:               " << opts_.device
      << "\n  tracker_format:       " << opts_.tracker_format
      << "\n  grid_selection/grid_rows:           " << opts_.grid_rows
      << "\n  grid_selection/grid_cols:           " << opts_.grid_cols
      << "\n  reseed/active_fraction:        " << opts_.reseed_active_fraction
      << "\n  reseed/min_candidate_fraction: " << opts_.min_candidate_fraction
      << "\n  viz/uv_tracking:        " << (opts_.viz_uv_tracking ? "true" : "false")
      << "\n  track_cache_path:     " << (opts_.track_cache_path.empty() ? "(live tracking)" : opts_.track_cache_path);

  if (!opts_.track_cache_path.empty())
  {
    // Replay mode -- skip backend construction/model loading entirely (the
    // whole point: no GPU tracker pass at all). loaded() reports true once
    // cache_reader_ is populated (see the .h's loaded() doc comment).
    cache_reader_ = std::make_unique<TrackerCacheReader>();
    std::string err;
    if (!cache_reader_->load(opts_.track_cache_path, err))
    {
      oss << "\n  FAILED to load track cache '" << opts_.track_cache_path << "': " << err;
      cache_reader_.reset();
      return oss.str();
    }
    const TrackerCacheHeader& h = cache_reader_->header();
    const std::string this_run_format = opts_.backend + "/" + opts_.tracker_format;
    if (h.grid_rows != opts_.grid_rows || h.grid_cols != opts_.grid_cols || h.tracker_format != this_run_format)
    {
      oss << "\n  FAILED: cache '" << opts_.track_cache_path << "' was built for grid="
          << h.grid_rows << "x" << h.grid_cols << " tracker_format=" << h.tracker_format
          << ", but this run is configured for grid=" << opts_.grid_rows << "x" << opts_.grid_cols
          << " tracker_format=" << this_run_format << " -- refusing to replay a mismatched cache.";
      cache_reader_.reset();
      return oss.str();
    }
    oss << "\n  replay mode: loaded " << cache_reader_->size() << " frames from "
        << opts_.track_cache_path << " (bag=" << h.bag_basename
        << ") -- GPU tracker pass will be skipped entirely.";
    return oss.str();
  }

  PointTrackerOptions topts;
  topts.device = opts_.device;
  topts.profiler = profiler_;
  topts.reseed_active_fraction = opts_.reseed_active_fraction;

  std::string model_path_a, model_path_b;
  if (opts_.backend == "tapnext")
  {
    backend_ = std::make_unique<TAPNextBackend>(topts);
    model_path_a = kTAPNextInitPath;
    model_path_b = kTAPNextStepPath;
  }
  else if (opts_.backend == "trackon")
  {
    backend_ = std::make_unique<TrackOnBackend>(topts);
    model_path_a = kTrackOnInitPath;
    model_path_b = kTrackOnStepPath;
  }
  else
  {
    backend_ = std::make_unique<CoTrackerBackend>(topts);
    const std::string dir = std::string(kCoTrackerCheckpointDir) + "/";
    // Tracker's AOTInductor package is compiled for a FIXED query count
    // (grid_rows*grid_cols at export time), so its filename encodes the
    // exact grid shape (cotracker_pair_tracker_aot_<rows>x<cols>.pt2) --
    // changing grid_rows/grid_cols with tracker_format=aot makes this look
    // for a differently-named file automatically. If no matching export
    // exists, AOTIModelPackageLoader's constructor throws (file not
    // found) and the try/catch below turns that into a clear "FAILED to
    // load" line instead of silently loading a wrong-shaped model.
    if (opts_.tracker_format == "aot")
    {
      model_path_a = dir + "cotracker_frame_encoder_aot.pt2";
      model_path_b = dir + "cotracker_pair_tracker_aot_" + std::to_string(opts_.grid_rows) + "x"
                   + std::to_string(opts_.grid_cols) + ".pt2";
    }
    else
    {
      model_path_a = kCoTrackerEncoderPath;
      model_path_b = kCoTrackerTrackerPath;
    }
  }

  try
  {
    backend_->load(model_path_a, model_path_b);
    oss << "\n  loaded " << opts_.backend << " modules on " << opts_.device;
  }
  catch (const std::exception& e)
  {
    backend_.reset();
    oss << "\n  FAILED to load " << opts_.backend << " modules: " << e.what();
  }

  return oss.str();
}

// ─────────────────────────────────────────────
//  Distortion correction (see the .h doc comment)
// ─────────────────────────────────────────────

cv::Point2f Tracker::undistortPixel(const cv::Point2f& p) const
{
  const cv::Matx33d K(state_->cameraFx(), 0.0,                state_->cameraCx(),
                      0.0,                state_->cameraFy(), state_->cameraCy(),
                      0.0,                0.0,                1.0);
  const std::vector<double>& d = state_->cameraDistortionCoeffs();
  if (d.empty()) return p;  // no distortion model loaded -- nothing to correct

  std::vector<cv::Point2f> src{p}, dst;
  // P=K re-projects the undistorted result back into pixel units (instead
  // of cv::undistortPoints' default normalized-camera-coordinate output),
  // so every downstream consumer keeps working with plain pixel
  // coordinates unchanged.
  cv::undistortPoints(src, dst, K, d, cv::noArray(), K);
  return dst[0];
}

Eigen::Matrix2d Tracker::localPixelToUVJacobian(const cv::Point2f& px) const
{
  const double fx = state_->cameraFx(), fy = state_->cameraFy();
  Eigen::Matrix2d scale = Eigen::Matrix2d::Zero();
  scale(0, 0) = 1.0 / fx;
  scale(1, 1) = 1.0 / fy;

  if (state_->cameraDistortionCoeffs().empty())
    return scale;  // undistortPixel() is identity -- normalized = (px-c)/f directly

  // Central finite difference, one pixel step -- see the .h doc comment for
  // why this (not an analytic camera-model Jacobian) is the only available
  // path here.
  constexpr float kEps = 1.0f;
  const cv::Point2f u_xp = undistortPixel(px + cv::Point2f(kEps, 0.0f));
  const cv::Point2f u_xm = undistortPixel(px - cv::Point2f(kEps, 0.0f));
  const cv::Point2f u_yp = undistortPixel(px + cv::Point2f(0.0f, kEps));
  const cv::Point2f u_ym = undistortPixel(px - cv::Point2f(0.0f, kEps));

  Eigen::Matrix2d J_pix;  // d(undistorted_pixel)/d(raw_pixel)
  J_pix(0, 0) = (u_xp.x - u_xm.x) / (2.0 * kEps);
  J_pix(1, 0) = (u_xp.y - u_xm.y) / (2.0 * kEps);
  J_pix(0, 1) = (u_yp.x - u_ym.x) / (2.0 * kEps);
  J_pix(1, 1) = (u_yp.y - u_ym.y) / (2.0 * kEps);

  return scale * J_pix;
}

// ─────────────────────────────────────────────
//  Shi-Tomasi anchor selection
// ─────────────────────────────────────────────

std::vector<cv::Point2f> Tracker::selectAnchors(const cv::Mat& image,
                                                std::vector<AnchorPoint>& anchors_out) const
{
  const int cam_w = state_->cameraWidth();
  const int cam_h = state_->cameraHeight();
  const cv::Rect valid = validRegion(cam_w, cam_h);

  const int rows = std::max(opts_.grid_rows, 1);
  const int cols = std::max(opts_.grid_cols, 1);
  const double cell_w = static_cast<double>(valid.width) / cols;
  const double cell_h = static_cast<double>(valid.height) / rows;
  const int n_cells = rows * cols;

  std::vector<cv::Point2f> uv_out;
  anchors_out.clear();
  uv_out.reserve(n_cells);
  anchors_out.reserve(n_cells);

  if (image.empty()) return uv_out;

  cv::Mat gray;
  if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else                       gray = image;
  cv::Mat shi_tomasi_scores;
  cv::cornerMinEigenVal(gray, shi_tomasi_scores, /*blockSize=*/3, /*ksize=*/3);

  // Purely image-based: the epipolar residual only ever needs a 2D seed
  // pixel per anchor (see VioProc::computeEpipolarResidualAndJacobian), not
  // LIDAR depth, so every grid cell picks its own best Shi-Tomasi corner
  // directly from the image rather than requiring (or preferring) a LIDAR
  // point to land in that cell.
  for (int cell = 0; cell < n_cells; ++cell)
  {
    const int row = cell / cols;
    const int col = cell % cols;
    const int x0 = static_cast<int>(std::round(valid.x + col * cell_w));
    const int x1 = static_cast<int>(std::round(valid.x + (col + 1) * cell_w));
    const int y0 = static_cast<int>(std::round(valid.y + row * cell_h));
    const int y1 = static_cast<int>(std::round(valid.y + (row + 1) * cell_h));
    const cv::Rect cell_rect(x0, y0, std::max(x1 - x0, 1), std::max(y1 - y0, 1));
    const cv::Rect clipped = cell_rect & cv::Rect(0, 0, shi_tomasi_scores.cols, shi_tomasi_scores.rows);
    if (clipped.width <= 0 || clipped.height <= 0) continue;

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(shi_tomasi_scores(clipped), nullptr, &max_val, nullptr, &max_loc);
    const cv::Point2f uv_pixel(static_cast<float>(clipped.x + max_loc.x), static_cast<float>(clipped.y + max_loc.y));

    uv_out.push_back(uv_pixel);  // raw pixel -- fed to the tracking backend, must stay in true image space
    AnchorPoint a;
    a.uv_seed = undistortPixel(uv_pixel);  // distortion-corrected -- feeds the epipolar residual instead
    a.uv_jacobian_seed = localPixelToUVJacobian(uv_pixel);  // evaluated at the RAW pixel, see doc comment
    anchors_out.push_back(a);
  }

  return uv_out;
}

cv::Mat Tracker::drawUvTracking(const cv::Mat& image) const
{
  cv::Mat vis = image.clone();
  const double cx = (image.cols - 1) / 2.0;
  const double cy = (image.rows - 1) / 2.0;

  for (size_t i = 0; i < uv_prev_.size() && i < uv_curr_.size(); ++i)
  {
    const cv::Point2f& p0 = uv_prev_[i];
    const cv::Point2f& p1 = uv_curr_[i];

    const double dx = (p0.x - cx) / std::max(cx, 1.0);
    const double dy = (p0.y - cy) / std::max(cy, 1.0);
    const double hue = std::fmod(std::atan2(dy, dx) / (2.0 * M_PI) + 1.0, 1.0) * 360.0;
    const double radius = std::min(std::sqrt(dx * dx + dy * dy), 1.0);
    const double value = 0.5 + 0.5 * radius;

    cv::Mat hsv_px(1, 1, CV_32FC3, cv::Scalar(hue, 1.0, value));
    cv::Mat bgr_px;
    cv::cvtColor(hsv_px, bgr_px, cv::COLOR_HSV2BGR);
    const cv::Vec3f c = bgr_px.at<cv::Vec3f>(0, 0);
    const cv::Scalar color(c[0] * 255.0, c[1] * 255.0, c[2] * 255.0);

    cv::arrowedLine(vis, p0, p1, color, 2, cv::LINE_AA, 0, 0.5);
  }

  return vis;
}

// ─────────────────────────────────────────────
//  Async tracking pipeline (task #145)
// ─────────────────────────────────────────────

TrackedFrame Tracker::stepAndReseedOnce(const cv::Mat& image)
{
  TimedScope ts(profiler_, "vio/step_and_reseed");

  TrackedFrame out;
  if (!loaded() || image.empty()) return out;

  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

  std::vector<int> indices;
  out.ok = hasPrev() && backend_->step(rgb, uv_prev_, uv_curr_, indices);

  // Snapshot BEFORE the reseed below overwrites anchors_ -- `indices` (from
  // step(), just above) are indices into the anchor set as of the LAST
  // reseed, not the one about to happen here. See TrackedFrame's own doc
  // comment for why this must be a copy, not a reference into live
  // Tracker state.
  if (out.ok)
  {
    // Distortion-corrected (see undistortPixel()'s doc comment) -- unlike
    // uv_prev_/uv_curr_ themselves (left untouched, still true image-space
    // pixels for the backend/visualization), TrackedFrame::uv_curr feeds
    // directly into the epipolar residual.
    out.uv_curr.reserve(uv_curr_.size());
    out.uv_curr_uv_jacobian.reserve(uv_curr_.size());
    for (const cv::Point2f& p : uv_curr_)
    {
      out.uv_curr.push_back(undistortPixel(p));
      out.uv_curr_uv_jacobian.push_back(localPixelToUVJacobian(p));  // evaluated at the RAW pixel, see doc comment
    }
    out.indices = std::move(indices);
    out.anchors = anchors_;
  }

  // Pose-independent (see task #145): reseed whenever there's no prior
  // tracking to build on, or the backend's own needsReseed() signal says
  // coverage has degraded -- no camera-motion/baseline gate. Uses THIS
  // SAME image, exactly like the old synchronous updateTracker() did.
  std::ostringstream log;
  if (!hasPrev() || needsReseed())
  {
    std::vector<AnchorPoint> candidate_anchors;
    const std::vector<cv::Point2f> query_points = selectAnchors(image, candidate_anchors);

    const int n_candidates = static_cast<int>(candidate_anchors.size());
    const int n_cells = std::max(opts_.grid_rows, 1) * std::max(opts_.grid_cols, 1);
    const int min_candidates = static_cast<int>(opts_.min_candidate_fraction * n_cells);
    if (n_candidates < min_candidates)
    {
      log << "[vio/reseed]  skipped (too few candidates: " << n_candidates << " < " << min_candidates << ")";
    }
    else
    {
      anchors_ = std::move(candidate_anchors);
      backend_->reseed(rgb, query_points);
      out.reseeded = true;
      log << "[vio/reseed]  n_points=" << query_points.size();
    }
  }
  out.log = log.str();

  return out;
}

void Tracker::startAsyncTracking()
{
  if (tracking_thread_.joinable()) return;  // already started
  stop_requested_.store(false);
  tracking_thread_ = std::thread(&Tracker::asyncTrackingLoop, this);
}

void Tracker::stopAsyncTracking()
{
  if (!tracking_thread_.joinable()) return;
  stop_requested_.store(true);
  input_cv_.notify_all();
  tracking_thread_.join();
}

void Tracker::feedImage(const cv::Mat& img, double timestamp)
{
  if (!tracking_thread_.joinable()) return;  // pipeline not started -- drop silently
  {
    std::lock_guard<std::mutex> lock(input_mutex_);
    input_queue_.emplace_back(img, timestamp);
  }
  input_cv_.notify_one();
}

void Tracker::asyncTrackingLoop()
{
  // Chains every fed image's own timestamp into the NEXT frame's
  // prev_timestamp (mirrors FAST-LIVO2's LivoVioManager::asyncTrackingLoop
  // local `prev_timestamp`) -- unconditional, regardless of whether a given
  // frame's own stepAndReseedOnce() call happened to reseed: this is the
  // anchor-pose-HISTORY chain (see Tracker's class doc comment), a
  // different thing from `anchors_`/`indices`' own "as of the last
  // successful reseed" semantics.
  double prev_timestamp_chain = -1.0;

  while (true)
  {
    cv::Mat img;
    double timestamp = 0.0;
    {
      std::unique_lock<std::mutex> lock(input_mutex_);
      input_cv_.wait(lock, [this] { return !input_queue_.empty() || stop_requested_.load(); });
      if (stop_requested_.load() && input_queue_.empty()) return;
      img = input_queue_.front().first;
      timestamp = input_queue_.front().second;
      input_queue_.pop_front();
    }

    // See TrackerOptions::track_cache_path's doc comment -- replay mode
    // looks the frame up by timestamp instead of running the real
    // step()/reseed() pass, skipping the GPU tracker entirely. A miss (no
    // record within tolerance -- e.g. this frame was never fed to the
    // cache-generation tool, or timestamps drifted) yields a default-
    // constructed TrackedFrame (ok=false), the same "nothing to correct
    // against this frame" signal a live tracking failure would produce,
    // rather than crashing or silently reusing a wrong frame's result.
    TrackedFrame frame;
    if (cache_reader_)
    {
      const size_t idx = cache_reader_->findByTimestamp(timestamp);
      if (idx < cache_reader_->size())
        frame = cache_reader_->at(idx);
    }
    else
    {
      frame = stepAndReseedOnce(img);
    }
    frame.timestamp = timestamp;
    // Always THIS run's own live chain, not whatever the cache record (if
    // any) stored -- correct regardless of a live run's bootstrap offset
    // relative to however the cache was originally generated.
    frame.prev_timestamp = prev_timestamp_chain;
    prev_timestamp_chain = timestamp;

    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      result_queue_.push_back(std::move(frame));
    }
    result_cv_.notify_all();
  }
}

bool Tracker::consumeResult(TrackedFrame& out)
{
  std::lock_guard<std::mutex> lock(result_mutex_);
  if (result_queue_.empty()) return false;
  out = std::move(result_queue_.front());
  result_queue_.pop_front();
  return true;
}

void Tracker::recordFramePose(double timestamp)
{
  anchor_pose_history_.emplace_back(timestamp, state_->rot(), state_->pos());
}

void Tracker::resolveAnchorPose(TrackedFrame& frame)
{
  if (!anchor_pose_history_.empty() &&
      std::fabs(std::get<0>(anchor_pose_history_.front()) - frame.prev_timestamp) < 1e-6)
  {
    frame.rot_anchor = std::get<1>(anchor_pose_history_.front());
    frame.pos_anchor = std::get<2>(anchor_pose_history_.front());
    anchor_pose_history_.pop_front();
  }
  else if (frame.prev_timestamp >= 0.0)
  {
    // Bootstrap frames (prev_timestamp == -1.0 sentinel) are EXPECTED to
    // miss -- no prior frame's pose was ever recorded for them to find.
    // Any other miss should not happen (every real frame's predecessor
    // gets exactly one history entry, consumed exactly once, in strict
    // order) -- falls back to identity/zero (matches TrackedFrame's own
    // default) rather than use a mismatched entry.
    ROS_ERROR_STREAM("[vio/tracker] anchor pose history lookup miss for prev_timestamp="
        << std::fixed << frame.prev_timestamp << " -- falling back to identity/zero anchor pose");
  }
}

}  // namespace livo_recon
