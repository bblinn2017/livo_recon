#include "livo_recon/vio/trackon/trackon_backend.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <opencv2/imgproc.hpp>

#include "livo_recon/vio/crop_utils.h"

namespace livo_recon
{

void TrackOnBackend::load(const std::string& init_path, const std::string& step_path)
{
  device_ = torch::Device(opts_.device);
  init_ = torch::jit::load(init_path, device_);
  step_ = torch::jit::load(step_path, device_);
  init_.eval();
  step_.eval();
  point_memory_ = torch::Tensor();
  temporal_mask_ = torch::Tensor();
  last_uv_.clear();
  n_visible_ = 0;
}

cv::Rect TrackOnBackend::validRegion(int cam_width, int cam_height) const
{
  return centerCropRect(cam_width, cam_height, opts_.interp_width, opts_.interp_height);
}

torch::Tensor TrackOnBackend::toFrameTensor(const cv::Mat& rgb, cv::Rect& crop_out) const
{
  crop_out = centerCropRect(rgb.cols, rgb.rows, opts_.interp_width, opts_.interp_height);
  cv::Mat resized;
  cv::resize(rgb(crop_out), resized, cv::Size(opts_.interp_width, opts_.interp_height), 0, 0, cv::INTER_LINEAR);

  torch::Tensor frame = torch::from_blob(
      const_cast<uchar*>(resized.data), {resized.rows, resized.cols, 3}, torch::kUInt8).clone();
  return frame.to(device_).to(torch::kFloat32).permute({2, 0, 1}).unsqueeze(0);
}

const torch::Tensor& TrackOnBackend::camHw(const cv::Rect& crop)
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

void TrackOnBackend::seedImpl(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points)
{
  torch::NoGradGuard no_grad;

  cv::Rect crop;
  torch::Tensor frame, queries;
  {
    TimedScope ts(opts_.profiler, "vio/trackon/preprocess");
    frame = toFrameTensor(rgb, crop);

    const int n = static_cast<int>(query_points.size());
    torch::Tensor queries_cpu = torch::empty({n, 2}, torch::kFloat32);
    auto qacc = queries_cpu.accessor<float, 2>();
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      qacc[i][0] = query_points[i].x - crop.x;
      qacc[i][1] = query_points[i].y - crop.y;
    }
    queries = queries_cpu.to(device_);
  }
  const torch::Tensor& cam_hw = camHw(crop);

  torch::Tensor tracks, vis;
  {
    TimedScope ts(opts_.profiler, "vio/trackon/forward");
    auto outputs = init_.forward({frame, queries, cam_hw}).toTuple();
    const auto& elems = outputs->elements();
    tracks = elems[0].toTensor().to(torch::kCPU).contiguous();
    vis    = elems[1].toTensor().to(torch::kCPU).contiguous();
    q_init_        = elems[2].toTensor();  // stays on device_, unchanged across step() calls
    point_memory_  = elems[3].toTensor();
    temporal_mask_ = elems[4].toTensor();
  }

  {
    TimedScope ts(opts_.profiler, "vio/trackon/postprocess");
    const int n = static_cast<int>(query_points.size());
    last_uv_.assign(query_points.begin(), query_points.end());
    n_visible_ = 0;
    auto tacc = tracks.accessor<float, 2>();  // (N, 2)
    auto vacc = vis.accessor<bool, 1>();      // (N,)
    std::vector<uint8_t> visible(n, 0);
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      last_uv_[i] = cv::Point2f(tacc[i][0] + crop.x, tacc[i][1] + crop.y);
      visible[i] = vacc[i] ? 1 : 0;
    }
    for (int i = 0; i < n; ++i)
      if (visible[i]) ++n_visible_;
  }
}

bool TrackOnBackend::stepImpl(const cv::Mat& rgb,
                              std::vector<cv::Point2f>& uv_prev_out,
                              std::vector<cv::Point2f>& uv_curr_out,
                              std::vector<int>& indices_out)
{
  torch::NoGradGuard no_grad;

  if (last_uv_.empty() || !point_memory_.defined())
  {
    n_visible_ = 0;
    return false;
  }

  cv::Rect crop;
  torch::Tensor frame;
  {
    TimedScope ts(opts_.profiler, "vio/trackon/preprocess");
    frame = toFrameTensor(rgb, crop);
  }
  const torch::Tensor& cam_hw = camHw(crop);

  torch::Tensor tracks, vis;
  {
    TimedScope ts(opts_.profiler, "vio/trackon/forward");
    auto outputs = step_.forward({frame, q_init_, point_memory_, temporal_mask_, cam_hw}).toTuple();
    const auto& elems = outputs->elements();
    tracks = elems[0].toTensor().to(torch::kCPU).contiguous();
    vis    = elems[1].toTensor().to(torch::kCPU).contiguous();
    q_init_        = elems[2].toTensor();
    point_memory_  = elems[3].toTensor();
    temporal_mask_ = elems[4].toTensor();
  }

  const int n = static_cast<int>(last_uv_.size());
  std::vector<cv::Point2f> new_uv(n);
  {
    TimedScope ts(opts_.profiler, "vio/trackon/postprocess");
    std::vector<uint8_t> visible(n, 0);
    auto tacc = tracks.accessor<float, 2>();
    auto vacc = vis.accessor<bool, 1>();
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      new_uv[i] = cv::Point2f(tacc[i][0] + crop.x, tacc[i][1] + crop.y);
      visible[i] = vacc[i] ? 1 : 0;
    }

    n_visible_ = 0;
    for (int i = 0; i < n; ++i)
    {
      if (visible[i])
      {
        uv_prev_out.push_back(last_uv_[i]);
        uv_curr_out.push_back(new_uv[i]);
        indices_out.push_back(i);
        ++n_visible_;
      }
    }
  }

  last_uv_ = std::move(new_uv);
  return true;
}

}  // namespace livo_recon
