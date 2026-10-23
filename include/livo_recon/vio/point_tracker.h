#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

// Options common to every PointTracker backend, passed in at construction
// time (not load()) and held by the base class as a member so a backend's
// own methods can just read opts_ rather than needing every option
// threaded through load()'s argument list.
struct PointTrackerOptions
{
  std::string device = "cuda";

  // For TimedScope-instrumenting step()/reseed()'s internal work under
  // "vio/select_and_track". Must be non-null.
  ProfilerPtr profiler;

  // Resolution the backend's traced TorchScript graph(s) were exported at
  // (see co-tracker/export_pairwise_tracker.py) -- validRegion()/encoding
  // center-crop to this aspect ratio before resizing.
  int interp_width  = 512;
  int interp_height = 384;
};

// Interface wrapping the native (LibTorch) point-tracking backend vio
// runs (CoTrackerBackend -- the only PointTracker implementation; TAPNext
// and Track-On backends existed here previously and were removed, since
// no config ever used them and CoTracker has no persistent state for
// their coverage-triggered reseed design to apply to anyway).
//
// PointTracker itself only ever deals in raw 2D pixel positions -- it has
// no notion of what a query point "means" (e.g. vio's LIDAR-anchored
// 3D points with covariance). Query point *selection* is the caller's
// responsibility (VioProc::updateTracker, called once per measure group
// after this frame's state has been fully corrected -- see
// livo_recon_node.cpp's updateMaps): step() only ever advances the
// existing tracked set by one frame, and reseed() only ever (re-)seeds on
// exactly the image handed to it -- the caller (Tracker::
// stepAndReseedOnce()) calls both every single frame, unconditionally.
class PointTracker
{
public:
  explicit PointTracker(const PointTrackerOptions& opts) : opts_(opts) {}
  virtual ~PointTracker() = default;

  // Backend-specific model file paths (CoTracker: frame_encoder.pt +
  // pair_tracker.pt).
  virtual void load(const std::string& model_path_a, const std::string& model_path_b) = 0;

  // True once this tracker has been seeded at least once -- i.e. whether
  // there is a valid persistent track state for step() to advance from
  // (and, for the caller, a valid previous pose to compare a baseline
  // displacement against).
  bool hasPrev() const { return have_prev_; }

  // (Re-)initializes the persistent track set from scratch at
  // `query_points` (raw camera pixels, guaranteed within validRegion()),
  // discarding whatever was being tracked before. query_points[i]'s index
  // i is what a later step()'s indices_out will report back once that slot
  // is tracked. Marks hasPrev() true.
  void reseed(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points);

  // Advances every active track by one frame (from whatever image the last
  // reseed()/step() call saw to `rgb`). Returns false (nothing to advance)
  // if hasPrev() is false. Otherwise fills uv_prev_out[i]/uv_curr_out[i]/
  // indices_out[i] with each active track's position as of the last call,
  // its newly predicted position in `rgb`, and its query index (into
  // whatever reseed() was last called with).
  bool step(const cv::Mat& rgb,
           std::vector<cv::Point2f>& uv_prev_out,
           std::vector<cv::Point2f>& uv_curr_out,
           std::vector<int>& indices_out);

  // The center-cropped region (in raw camera pixel coordinates) the model
  // actually sees for a camera of the given resolution.
  virtual cv::Rect validRegion(int cam_width, int cam_height) const = 0;

  // Number of currently-active persistent tracks (i.e. what the next
  // step() call will report, before further visibility-based attrition).
  virtual size_t numActive() const = 0;

protected:
  virtual void seedImpl(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points) = 0;

  virtual bool stepImpl(const cv::Mat& rgb,
                        std::vector<cv::Point2f>& uv_prev_out,
                        std::vector<cv::Point2f>& uv_curr_out,
                        std::vector<int>& indices_out) = 0;

  PointTrackerOptions opts_;

private:
  bool have_prev_ = false;
};

}  // namespace livo_recon
