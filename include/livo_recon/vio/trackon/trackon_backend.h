#pragma once

#include <torch/script.h>

#include "livo_recon/vio/point_tracker.h"

namespace livo_recon
{

// Runs Track-On (Track-On2/Track-On-R) natively (via TorchScript modules
// exported by trackon/export_pairwise_trackon.py) as a PointTracker
// backend. Like TAPNextBackend, Track-On genuinely is a causal, per-frame
// tracker with persistent state: seed() (trackon_init.pt) commits a query
// set into a compact transformer memory, and each step() (trackon_step.pt)
// advances that same memory by exactly one frame.
//
// Unlike TAPNext, Track-On's persistent state is just three plain tensors
// (q_init, point_memory, temporal_mask) rather than a per-layer SSM cache,
// and its coordinate output isn't quantized into a small fixed bin count
// (it's a patch-grid argmax plus a continuous offset, at full float
// precision), so -- unlike TAPNext -- opts.interp_width/interp_height are
// genuinely configurable here, not forced to a fixed native resolution.
//
// The query batch size is fixed for the lifetime of a state (it's baked
// into every cached tensor's shape via the traced graph), so -- exactly
// like the other backends -- a track the model reports as no longer
// visible is excluded from step()'s output but not removed from the
// state; only seed() changes the query count.
class TrackOnBackend : public PointTracker
{
public:
  explicit TrackOnBackend(const PointTrackerOptions& opts) : PointTracker(opts) {}

  void load(const std::string& init_path, const std::string& step_path) override;

  cv::Rect validRegion(int cam_width, int cam_height) const override;

  size_t numActive() const override { return n_visible_; }

protected:
  bool needsReseed() const override
  {
    return last_uv_.empty() || n_visible_ < opts_.reseed_active_fraction * last_uv_.size();
  }

  void seedImpl(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points) override;

  bool stepImpl(const cv::Mat& rgb,
               std::vector<cv::Point2f>& uv_prev_out,
               std::vector<cv::Point2f>& uv_curr_out,
               std::vector<int>& indices_out) override;

private:
  torch::Tensor toFrameTensor(const cv::Mat& rgb, cv::Rect& crop_out) const;

  // See CoTrackerBackend::camHw() -- identical rationale: crop dims are
  // fixed for the tracker's lifetime, so cache the device upload instead of
  // reallocating + re-uploading every frame.
  const torch::Tensor& camHw(const cv::Rect& crop);

  torch::jit::script::Module init_;
  torch::jit::script::Module step_;
  torch::Device device_ = torch::kCPU;

  // Persistent state, opaque to everything but trackon_step.pt: q_init is
  // each query's committed feature descriptor (unchanged across step()
  // calls); point_memory/temporal_mask are the compact transformer memory,
  // updated every call via Track-On's own roll-and-append rule (baked into
  // the traced graph).
  torch::Tensor q_init_;
  torch::Tensor point_memory_;
  torch::Tensor temporal_mask_;

  std::vector<cv::Point2f> last_uv_;  // every slot's position as of the last seed()/step() call
  size_t                   n_visible_ = 0;

  torch::Tensor cam_hw_cache_;
  int cam_hw_cache_h_ = -1, cam_hw_cache_w_ = -1;
};

}  // namespace livo_recon
