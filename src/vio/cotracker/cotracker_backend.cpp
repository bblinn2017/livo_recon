#include "livo_recon/vio/cotracker/cotracker_backend.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <regex>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "livo_recon/vio/crop_utils.h"

namespace livo_recon
{

namespace
{
bool hasSuffix(const std::string& s, const std::string& suffix)
{
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Parses the required "..._<rows>x<cols>.pt2" filename convention and
// returns rows*cols. Throws std::runtime_error (caught by Tracker::
// loadParameters()'s existing try/catch, surfaced as a "FAILED to load"
// log line) if the filename doesn't match -- an AOTInductor tracker
// package compiled for one query count silently invokes undefined
// behavior if run against a different one, so this must be a hard
// failure, not a warning.
int parseAotExpectedQueries(const std::string& path)
{
  static const std::regex kPattern(R"(_(\d+)x(\d+)\.pt2$)");
  std::smatch m;
  if (!std::regex_search(path, m, kPattern))
  {
    throw std::runtime_error(
        "AOTInductor tracker path '" + path + "' does not match the required "
        "'..._<rows>x<cols>.pt2' naming convention (e.g. cotracker_pair_tracker_16x16.pt2) "
        "-- refusing to load a query-count-fixed AOTInductor package without a filename-"
        "encoded shape to validate against tracker/grid_selection/grid_rows * grid_cols at runtime.");
  }
  const int rows = std::stoi(m[1].str());
  const int cols = std::stoi(m[2].str());
  return rows * cols;
}
}  // namespace

void CoTrackerBackend::load(const std::string& encoder_path, const std::string& tracker_path)
{
  device_ = torch::Device(opts_.device);

  encoder_is_aot_ = hasSuffix(encoder_path, ".pt2");
  if (encoder_is_aot_)
  {
    encoder_aot_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(encoder_path);
  }
  else
  {
    encoder_ = torch::jit::load(encoder_path, device_);
    encoder_.eval();
  }

  tracker_is_aot_ = hasSuffix(tracker_path, ".pt2");
  if (tracker_is_aot_)
  {
    // Must succeed (throws otherwise) BEFORE constructing the loader --
    // no point paying AOTInductor's load cost for a package we already
    // know we can't validate at runtime.
    tracker_aot_expected_queries_ = parseAotExpectedQueries(tracker_path);
    tracker_aot_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(tracker_path);
  }
  else
  {
    tracker_ = torch::jit::load(tracker_path, device_);
    tracker_.eval();
  }

  encoder_use_fp16_ = encoder_is_aot_ && (encoder_path.find("fp16") != std::string::npos);
  tracker_use_fp16_ = tracker_is_aot_ && (tracker_path.find("fp16") != std::string::npos);

  anchor_pyramid_.clear();
  active_uv_.clear();
  n_visible_ = 0;
}

cv::Rect CoTrackerBackend::validRegion(int cam_width, int cam_height) const
{
  return centerCropRect(cam_width, cam_height, opts_.interp_width, opts_.interp_height);
}

const torch::Tensor& CoTrackerBackend::camHw(const cv::Rect& crop)
{
  if (crop.height != cam_hw_cache_h_ || crop.width != cam_hw_cache_w_)
  {
    cam_hw_cache_ = torch::tensor(
        {static_cast<float>(crop.height), static_cast<float>(crop.width)}, torch::kFloat32).to(device_);
    cam_hw_cache_h_ = crop.height;
    cam_hw_cache_w_ = crop.width;
  }
  return cam_hw_cache_;
}

torch::Tensor CoTrackerBackend::queries(int n, const cv::Rect& crop)
{
  // queries_cpu_/queries_gpu_ are reused across step() calls (only
  // reallocated if n changes, which happens at most once per seed()) --
  // avoids a fresh CPU alloc + fresh GPU alloc + H2D upload every frame for
  // what is otherwise the exact same tensor shape/dtype every time.
  if (n != queries_cache_n_)
  {
    queries_cpu_ = torch::empty({1, n, 2}, torch::kFloat32);
    queries_gpu_ = torch::empty({1, n, 2}, torch::kFloat32).to(device_);
    queries_cache_n_ = n;
  }

  auto qacc = queries_cpu_.accessor<float, 3>();
  #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
  for (int i = 0; i < n; ++i)
  {
    // Query points are guaranteed (by validRegion(), enforced at seed()/
    // reseed time) to fall inside `crop`; re-express relative to its
    // top-left corner, the coordinate frame the traced graph's own cam_hw
    // scaling operates in.
    qacc[0][i][0] = active_uv_[i].x - crop.x;
    qacc[0][i][1] = active_uv_[i].y - crop.y;
  }
  // In-place copy into the already-allocated GPU buffer -- no fresh cudaMalloc
  // every frame, just the H2D transfer itself.
  queries_gpu_.copy_(queries_cpu_);
  return queries_gpu_;
}

std::vector<torch::Tensor> CoTrackerBackend::encodeFrame(const cv::Mat& rgb)
{
  torch::NoGradGuard no_grad;

  torch::Tensor frame;
  {
    TimedScope ts(opts_.profiler, "vio/cotracker/preprocess");
    const cv::Rect crop = centerCropRect(rgb.cols, rgb.rows, opts_.interp_width, opts_.interp_height);
    cv::Mat resized;
    cv::resize(rgb(crop), resized, cv::Size(opts_.interp_width, opts_.interp_height), 0, 0, cv::INTER_LINEAR);

    frame = torch::from_blob(
        const_cast<uchar*>(resized.data), {resized.rows, resized.cols, 3}, torch::kUInt8).clone();
    // .contiguous() is required here, not cosmetic: .permute({2,0,1}) only
    // rewrites strides (HWC -> CHW view), it doesn't copy memory. TorchScript
    // (encoder_) tolerates an arbitrarily-strided input fine via its generic
    // op dispatch -- but an AOTInductor package (encoder_aot_) is compiled/
    // specialized against the exact contiguous NCHW layout it was traced
    // with, and silently misinterprets a differently-strided-but-same-shape
    // tensor rather than erroring (confirmed in FAST-LIVO2's own port of
    // this backend: gross corruption without .contiguous(), ~0.003 max diff
    // vs eager with it).
    frame = frame.to(device_).to(torch::kFloat32).permute({2, 0, 1}).contiguous().unsqueeze(0);
    if (encoder_use_fp16_) frame = frame.to(torch::kHalf);
  }

  std::vector<torch::Tensor> pyramid;
  {
    TimedScope ts(opts_.profiler, "vio/cotracker/encode_forward");
    if (encoder_is_aot_)
    {
      const std::vector<at::Tensor> outputs = encoder_aot_->run({frame});
      pyramid.assign(outputs.begin(), outputs.end());
    }
    else
    {
      auto outputs = encoder_.forward({frame}).toTuple();
      pyramid.reserve(4);
      for (int i = 0; i < 4; ++i)
        pyramid.push_back(outputs->elements()[i].toTensor());
    }
  }
  return pyramid;
}

void CoTrackerBackend::runTracker(const std::vector<torch::Tensor>& anchor_pyramid,
                                   const std::vector<torch::Tensor>& curr_pyramid,
                                   const torch::Tensor& query_tensor, const torch::Tensor& cam_hw,
                                   torch::Tensor& out_tracks, torch::Tensor& out_vis, torch::Tensor& out_conf)
{
  if (tracker_is_aot_)
  {
    const int64_t n_queries = query_tensor.size(1);
    if (n_queries != tracker_aot_expected_queries_)
    {
      throw std::runtime_error(
          "CoTrackerBackend: AOTInductor tracker was compiled for " +
          std::to_string(tracker_aot_expected_queries_) + " query points (from its filename) "
          "but this call has " + std::to_string(n_queries) + " -- tracker/grid_selection/grid_rows * "
          "grid_cols must match the .pt2 export's --max_queries, or load a TorchScript (.pt) tracker instead.");
    }

    const torch::Tensor query_tensor_in = tracker_use_fp16_ ? query_tensor.to(torch::kHalf) : query_tensor;
    const torch::Tensor cam_hw_in = tracker_use_fp16_ ? cam_hw.to(torch::kHalf) : cam_hw;

    std::vector<at::Tensor> inputs;
    inputs.reserve(anchor_pyramid.size() + curr_pyramid.size() + 2);
    for (const auto& t : anchor_pyramid) inputs.push_back(t);
    for (const auto& t : curr_pyramid)   inputs.push_back(t);
    inputs.push_back(query_tensor_in);
    inputs.push_back(cam_hw_in);

    const std::vector<at::Tensor> outputs = tracker_aot_->run(inputs);
    out_tracks = outputs[0];
    out_vis    = outputs[1];
    out_conf   = outputs[2];
    return;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(anchor_pyramid.size() + curr_pyramid.size() + 2);
  for (const auto& t : anchor_pyramid) inputs.emplace_back(t);
  for (const auto& t : curr_pyramid)   inputs.emplace_back(t);
  inputs.emplace_back(query_tensor);
  inputs.emplace_back(cam_hw);

  auto outputs = tracker_.forward(inputs).toTuple();
  out_tracks = outputs->elements()[0].toTensor();
  out_vis    = outputs->elements()[1].toTensor();
  // Pre-AOT tracker exports (the JIT path here) only ever had 2 outputs --
  // no confidence tensor to read. out_conf is left as whatever the caller
  // default-constructed it to; callers here never read it either way.
}

void CoTrackerBackend::seedImpl(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points)
{
  anchor_pyramid_ = encodeFrame(rgb);
  active_uv_ = query_points;
  n_visible_ = query_points.size();
}

bool CoTrackerBackend::stepImpl(const cv::Mat& rgb,
                               std::vector<cv::Point2f>& uv_prev_out,
                               std::vector<cv::Point2f>& uv_curr_out,
                               std::vector<int>& indices_out)
{
  torch::NoGradGuard no_grad;

  const std::vector<torch::Tensor> curr_pyramid = encodeFrame(rgb);

  if (active_uv_.empty() || anchor_pyramid_.empty())
  {
    anchor_pyramid_ = curr_pyramid;
    n_visible_ = 0;
    return false;
  }

  const cv::Rect crop = centerCropRect(rgb.cols, rgb.rows, opts_.interp_width, opts_.interp_height);
  const int n = static_cast<int>(active_uv_.size());

  torch::Tensor query_tensor;
  {
    TimedScope ts(opts_.profiler, "vio/cotracker/preprocess");
    query_tensor = queries(n, crop);
  }
  const torch::Tensor& cam_hw = camHw(crop);

  torch::Tensor outputs_tracks, outputs_vis, outputs_conf;
  {
    TimedScope ts(opts_.profiler, "vio/cotracker/forward");
    runTracker(anchor_pyramid_, curr_pyramid, query_tensor, cam_hw,
               outputs_tracks, outputs_vis, outputs_conf);
  }

  std::vector<cv::Point2f> new_active_uv(n);
  std::vector<uint8_t> visible(n, 0);
  {
    TimedScope ts(opts_.profiler, "vio/cotracker/postprocess");
    // .to(kFloat32) is a no-op for the JIT path (already fp32); required
    // for the AOT path, whose outputs are fp16 -- accessor<float,...>()
    // below requires actual float32 storage, not just a value convertible
    // to float.
    const torch::Tensor tracks = outputs_tracks.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const torch::Tensor vis    = outputs_vis.to(torch::kCPU).to(torch::kFloat32).contiguous();
    // tracks: (1, 2, N, 2) -- index 0 is the anchor frame, index 1 is curr.
    // vis:    (1, 2, N)    -- already thresholded (0/1) inside the traced model.

    // The query batch itself (active_uv_, size n) never shrinks between
    // seed() calls -- the traced graph's shapes were only ever exercised at
    // a fixed query count, so a track the model loses is simply excluded
    // from this step's reported output (not removed from the batch); it
    // keeps being carried forward (re-anchored on its latest, possibly
    // unreliable, predicted position) until the next seed().
    auto tacc = tracks.accessor<float, 4>();
    auto vacc = vis.accessor<float, 3>();

    // History (287-293): see docs/livo_recon_changelog.md#src-vio-cotracker-cotracker_backend.cpp-287
    const bool have_conf = outputs_conf.defined();
    const torch::Tensor conf = have_conf
        ? outputs_conf.to(torch::kCPU).to(torch::kFloat32).contiguous()
        : torch::Tensor();
    const auto cacc = have_conf ? conf.accessor<float, 3>() : vacc;  // unused when !have_conf
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      new_active_uv[i] = cv::Point2f(tacc[0][1][i][0] + crop.x, tacc[0][1][i][1] + crop.y);
      const bool ok_vis  = vacc[0][1][i] > 0.5f;
      const bool ok_conf = !have_conf || cacc[0][1][i] > 0.5f;
      visible[i] = (ok_vis && ok_conf) ? 1 : 0;
    }

    // Filtering into the dynamically-sized output vectors is inherently
    // sequential (push_back), but it's now just a cheap scan over
    // already-computed per-index results, not per-index tensor-accessor work.
    n_visible_ = 0;
    for (int i = 0; i < n; ++i)
    {
      if (visible[i])
      {
        uv_prev_out.push_back(active_uv_[i]);
        uv_curr_out.push_back(new_active_uv[i]);
        indices_out.push_back(i);
        ++n_visible_;
      }
    }
  }

  active_uv_ = std::move(new_active_uv);
  anchor_pyramid_ = curr_pyramid;
  return true;
}

}  // namespace livo_recon
