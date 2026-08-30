#pragma once

#include <memory>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>

#include "livo_recon/vio/point_tracker.h"

namespace livo_recon
{

// Runs CoTracker natively (via TorchScript modules exported by
// co-tracker/export_pairwise_tracker.py, or AOTInductor .pt2 packages
// exported by export_encoder_fp16.py/export_tracker_fp16.py and
// export_encoder_fp32_aot_prod.py/export_tracker_fp32_aot_prod.py) as a
// PointTracker backend.
//
// AOTInductor support ported from FAST-LIVO2's copy of this backend
// (livo_vio/cotracker/cotracker_backend.{h,cpp}), which validated both an
// fp16 and a full-precision AOTInductor deployment end-to-end (~2.4x
// full-pipeline speedup for fp16, 0.63px max track disagreement / 100%
// visibility agreement vs the fp32 TorchScript reference). An earlier
// attempt at this backend's own AOT export was tried and reverted (the
// export only compiled/ran correctly under torch 2.13.0, mismatched with
// the 2.7.x libtorch this links) -- that was a limitation of the export
// script used at the time, not of AOTInductor itself; the newer export
// scripts referenced above don't have that problem.
//
// This port only carries over the encoder/tracker execution path (jit vs
// AOT selection, fp16 casting, contiguous-layout fix) -- NOT FAST-LIVO2's
// separately-added confidence_out/observeFrame plumbing, which is a
// distinct, not-yet-ported feature. The AOT tracker package's 3rd output
// (a continuous confidence score) is therefore read but discarded here,
// same as it would be if using the JIT tracker export that predates it.
//
// CoTracker3's own exported pair_tracker.pt only ever computes a direct
// two-frame correspondence -- it has no real persistent state of its own
// (unlike TAPNextBackend's genuine recurrent cache), so there's no benefit
// to carrying the same query points across many frames the way TAPNext
// needs to: needsReseed() always reports true here, so PointTracker::track()
// requests a brand new query grid from the caller on every single call
// (seeded on the previous frame, then immediately tracked forward one step
// to the current one -- see track()'s docs) rather than maintaining a
// persistent set of tracks between coverage-triggered reseeds.
//
// opts.interp_width/interp_height default to CoTracker3's own
// model_resolution (512x384) if not overridden.
class CoTrackerBackend : public PointTracker
{
public:
  explicit CoTrackerBackend(const PointTrackerOptions& opts) : PointTracker(opts) {}

  void load(const std::string& encoder_path, const std::string& tracker_path) override;

  cv::Rect validRegion(int cam_width, int cam_height) const override;

  // Number of tracks the most recent step reported visible. The
  // underlying query count itself (active_uv_.size()) never shrinks
  // between seeds -- lost tracks are simply excluded from stepImpl()'s
  // output, not removed from the model's query batch, since the traced
  // graph's shapes were only ever exercised at a fixed query count.
  size_t numActive() const override { return n_visible_; }

protected:
  // Always reseed: unlike TAPNext, CoTracker has no persistent state worth
  // preserving across frames, so a fresh grid every call (rather than a
  // coverage-triggered reseed of a maintained one) is simplest and cheap.
  bool needsReseed() const override { return true; }

  void seedImpl(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points) override;

  bool stepImpl(const cv::Mat& rgb,
               std::vector<cv::Point2f>& uv_prev_out,
               std::vector<cv::Point2f>& uv_curr_out,
               std::vector<int>& indices_out) override;

private:
  // Runs encoder_ (TorchScript) or encoder_aot_ (AOTInductor), whichever
  // load() selected.
  std::vector<torch::Tensor> encodeFrame(const cv::Mat& rgb);

  // Runs tracker_ (TorchScript) or tracker_aot_ (AOTInductor), whichever
  // load() selected. out_conf is populated (AOT packages export it) but
  // unused by any caller here -- see the class-level comment.
  void runTracker(const std::vector<torch::Tensor>& anchor_pyramid,
                   const std::vector<torch::Tensor>& curr_pyramid,
                   const torch::Tensor& query_tensor, const torch::Tensor& cam_hw,
                   torch::Tensor& out_tracks, torch::Tensor& out_vis, torch::Tensor& out_conf);

  // {crop.height, crop.width} uploaded to device_, rebuilt only when the
  // crop dims actually change -- in practice that's never after the first
  // call, since interp_width/interp_height and the camera resolution are
  // both fixed for the tracker's lifetime, so this turns a fresh small GPU
  // allocation + H2D upload every single frame into a one-time cost.
  const torch::Tensor& camHw(const cv::Rect& crop);

  // queries_cpu_/queries_gpu_ are reused across step() calls (only
  // reallocated if n changes, which happens at most once per seed()) --
  // avoids a fresh CPU alloc + fresh GPU alloc + H2D upload every frame for
  // what is otherwise the exact same tensor shape/dtype every time.
  torch::Tensor queries(int n, const cv::Rect& crop);

  // Exactly one of each pair is populated after load(), selected by the
  // checkpoint's file extension: ".pt" -> TorchScript (encoder_/tracker_),
  // ".pt2" -> AOTInductor (encoder_aot_/tracker_aot_).
  torch::jit::script::Module encoder_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> encoder_aot_;
  bool encoder_is_aot_ = false;

  torch::jit::script::Module tracker_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> tracker_aot_;
  bool tracker_is_aot_ = false;

  // Whether the AOT encoder/tracker packages specifically expect
  // half-precision input, determined from the filename ("fp16" present or
  // not) independently of encoder_is_aot_/tracker_is_aot_ -- see
  // FAST-LIVO2's cotracker_backend.h for the full rationale (decouples
  // "using the AOT path" from "using fp16" so a genuine fp32 AOTInductor
  // export, tracker_format="aot_fp32", runs without a cast it was never
  // compiled for).
  bool encoder_use_fp16_ = false;
  bool tracker_use_fp16_ = false;

  // Query count the loaded .pt2 tracker was compiled for, parsed from its
  // "..._<rows>x<cols>.pt2" filename at load() time. Every AOT call in
  // runTracker() asserts the actual query tensor's point count against
  // this before invoking tracker_aot_, so a grid-size/model mismatch fails
  // clearly instead of running AOTInductor against an uncompiled shape
  // (undefined behavior, not just a wrong answer).
  int tracker_aot_expected_queries_ = -1;

  torch::Device device_ = torch::kCPU;

  std::vector<torch::Tensor> anchor_pyramid_;  // encodeFrame() output for the frame active_uv_ is defined on
  std::vector<cv::Point2f>   active_uv_;       // every tracked slot's position as of anchor_pyramid_'s frame (fixed count between seed() calls)
  size_t                     n_visible_ = 0;   // how many of active_uv_ the most recent step() reported visible

  torch::Tensor cam_hw_cache_;
  int cam_hw_cache_h_ = -1, cam_hw_cache_w_ = -1;

  torch::Tensor queries_cpu_;
  torch::Tensor queries_gpu_;
  int queries_cache_n_ = -1;
};

}  // namespace livo_recon
