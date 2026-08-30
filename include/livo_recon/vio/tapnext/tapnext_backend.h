#pragma once

#include <torch/script.h>

#include "livo_recon/vio/point_tracker.h"

namespace livo_recon
{

// Runs TAPNext natively (via TorchScript modules exported by
// tapnext/export_pairwise_tapnext.py) as a PointTracker backend. Unlike
// CoTrackerBackend, TAPNext genuinely is a causal, per-frame recurrent
// tracker: seed() (tapnext_init.pt) commits a query set into a persistent
// 12-layer SSM cache, and each step() (tapnext_step.pt) advances that same
// cache by exactly one frame in O(1) time -- no frame history replay, and
// no re-anchoring trick needed the way CoTrackerBackend needs one.
//
// The query batch size is fixed for the lifetime of a cache (it's baked
// into every cached tensor's shape via the traced graph's token layout, see
// export_pairwise_tapnext.py), so -- exactly like CoTrackerBackend -- a
// track the model reports as no longer visible is excluded from step()'s
// output but not removed from the cache; only seed() changes the query
// count.
//
// Unlike CoTracker/FlowSeek, the working resolution here is NOT a tunable
// option: TAPNext's coordinate head is a fixed-width Linear(256, 512),
// i.e. its output resolution is hardcoded to 256 discrete position bins
// per axis regardless of image_size. That's a 1:1 bin-to-pixel mapping
// only at the native 256x256 training resolution used by
// export_pairwise_tapnext.py's default; anything else silently truncates
// (any position past pixel 255 can never be expressed). So opts.interp_width/
// interp_height are overridden to 256x256 here no matter what's passed in.
class TAPNextBackend : public PointTracker
{
public:
  explicit TAPNextBackend(const PointTrackerOptions& opts) : PointTracker(opts)
  {
    opts_.interp_width  = 256;
    opts_.interp_height = 256;
  }

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

  // Persistent recurrent state, opaque to everything but tapnext_step.pt:
  // query_points_ is the original (t=0, y, x) query spec (unchanged across
  // step() calls); cache_ is the 12-layer SSM cache (rg_lru_state,
  // conv1d_state per layer), updated every call.
  torch::Tensor              query_points_;
  std::vector<torch::Tensor> cache_;

  std::vector<cv::Point2f> last_uv_;  // every slot's position as of the last seed()/step() call
  size_t                   n_visible_ = 0;

  torch::Tensor cam_hw_cache_;
  int cam_hw_cache_h_ = -1, cam_hw_cache_w_ = -1;
};

}  // namespace livo_recon
