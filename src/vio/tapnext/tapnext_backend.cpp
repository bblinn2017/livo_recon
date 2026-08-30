#include "livo_recon/vio/tapnext/tapnext_backend.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <opencv2/imgproc.hpp>

#include "livo_recon/vio/crop_utils.h"

namespace livo_recon
{

void TAPNextBackend::load(const std::string& init_path, const std::string& step_path)
{
  device_ = torch::Device(opts_.device);
  init_ = torch::jit::load(init_path, device_);
  step_ = torch::jit::load(step_path, device_);
  init_.eval();
  step_.eval();
  cache_.clear();
  last_uv_.clear();
  n_visible_ = 0;
}

cv::Rect TAPNextBackend::validRegion(int cam_width, int cam_height) const
{
  return centerCropRect(cam_width, cam_height, opts_.interp_width, opts_.interp_height);
}

torch::Tensor TAPNextBackend::toFrameTensor(const cv::Mat& rgb, cv::Rect& crop_out) const
{
  crop_out = centerCropRect(rgb.cols, rgb.rows, opts_.interp_width, opts_.interp_height);
  cv::Mat resized;
  cv::resize(rgb(crop_out), resized, cv::Size(opts_.interp_width, opts_.interp_height), 0, 0, cv::INTER_LINEAR);

  torch::Tensor frame = torch::from_blob(
      const_cast<uchar*>(resized.data), {resized.rows, resized.cols, 3}, torch::kUInt8).clone();
  return frame.to(device_).to(torch::kFloat32).permute({2, 0, 1}).unsqueeze(0);
}

const torch::Tensor& TAPNextBackend::camHw(const cv::Rect& crop)
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

void TAPNextBackend::seedImpl(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points)
{
  torch::NoGradGuard no_grad;

  cv::Rect crop;
  torch::Tensor frame, queries;
  {
    TimedScope ts(opts_.profiler, "vio/tapnext/preprocess");
    frame = toFrameTensor(rgb, crop);

    const int n = static_cast<int>(query_points.size());
    torch::Tensor queries_cpu = torch::empty({1, n, 2}, torch::kFloat32);
    auto qacc = queries_cpu.accessor<float, 3>();
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      qacc[0][i][0] = query_points[i].x - crop.x;
      qacc[0][i][1] = query_points[i].y - crop.y;
    }
    queries = queries_cpu.to(device_);
  }
  const torch::Tensor& cam_hw = camHw(crop);

  torch::Tensor tracks, vis;
  {
    TimedScope ts(opts_.profiler, "vio/tapnext/forward");
    auto outputs = init_.forward({frame, queries, cam_hw}).toTuple();
    const auto& elems = outputs->elements();
    tracks = elems[0].toTensor().to(torch::kCPU).contiguous();
    vis    = elems[1].toTensor().to(torch::kCPU).contiguous();
    query_points_ = elems[2].toTensor();  // stays on device_, unchanged across step() calls

    cache_.clear();
    cache_.reserve(elems.size() - 3);
    for (size_t i = 3; i < elems.size(); ++i)
      cache_.push_back(elems[i].toTensor());
  }

  {
    TimedScope ts(opts_.profiler, "vio/tapnext/postprocess");
    const int n = static_cast<int>(query_points.size());
    last_uv_.assign(query_points.begin(), query_points.end());
    n_visible_ = 0;
    auto tacc = tracks.accessor<float, 3>();  // (1, Q, 2)
    auto vacc = vis.accessor<bool, 3>();      // (1, Q, 1)
    std::vector<uint8_t> visible(n, 0);
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      last_uv_[i] = cv::Point2f(tacc[0][i][0] + crop.x, tacc[0][i][1] + crop.y);
      visible[i] = vacc[0][i][0] ? 1 : 0;
    }
    for (int i = 0; i < n; ++i)
      if (visible[i]) ++n_visible_;
  }
}

bool TAPNextBackend::stepImpl(const cv::Mat& rgb,
                             std::vector<cv::Point2f>& uv_prev_out,
                             std::vector<cv::Point2f>& uv_curr_out,
                             std::vector<int>& indices_out)
{
  torch::NoGradGuard no_grad;

  if (last_uv_.empty() || cache_.empty())
  {
    n_visible_ = 0;
    return false;
  }

  cv::Rect crop;
  torch::Tensor frame;
  {
    TimedScope ts(opts_.profiler, "vio/tapnext/preprocess");
    frame = toFrameTensor(rgb, crop);
  }
  const torch::Tensor& cam_hw = camHw(crop);

  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(3 + cache_.size());
  inputs.emplace_back(frame);
  inputs.emplace_back(query_points_);
  inputs.emplace_back(cam_hw);
  for (const auto& t : cache_) inputs.emplace_back(t);

  torch::Tensor tracks, vis;
  {
    TimedScope ts(opts_.profiler, "vio/tapnext/forward");
    auto outputs = step_.forward(inputs).toTuple();
    const auto& elems = outputs->elements();
    tracks = elems[0].toTensor().to(torch::kCPU).contiguous();
    vis    = elems[1].toTensor().to(torch::kCPU).contiguous();

    cache_.clear();
    cache_.reserve(elems.size() - 2);
    for (size_t i = 2; i < elems.size(); ++i)
      cache_.push_back(elems[i].toTensor());
  }

  const int n = static_cast<int>(last_uv_.size());
  std::vector<cv::Point2f> new_uv(n);
  {
    TimedScope ts(opts_.profiler, "vio/tapnext/postprocess");
    std::vector<uint8_t> visible(n, 0);
    auto tacc = tracks.accessor<float, 3>();  // (1, Q, 2)
    auto vacc = vis.accessor<bool, 3>();      // (1, Q, 1)
    #pragma omp parallel for schedule(static) num_threads(cappedOmpThreads())
    for (int i = 0; i < n; ++i)
    {
      new_uv[i] = cv::Point2f(tacc[0][i][0] + crop.x, tacc[0][i][1] + crop.y);
      visible[i] = vacc[0][i][0] ? 1 : 0;
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
