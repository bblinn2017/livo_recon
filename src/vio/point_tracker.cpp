#include "livo_recon/vio/point_tracker.h"

namespace livo_recon
{

void PointTracker::reseed(const cv::Mat& rgb, const std::vector<cv::Point2f>& query_points)
{
  TimedScope ts(opts_.profiler, "vio/select_and_track");
  seedImpl(rgb, query_points);
  have_prev_ = true;
}

bool PointTracker::step(const cv::Mat& rgb,
                        std::vector<cv::Point2f>& uv_prev_out,
                        std::vector<cv::Point2f>& uv_curr_out,
                        std::vector<int>& indices_out)
{
  TimedScope ts(opts_.profiler, "vio/select_and_track");

  uv_prev_out.clear();
  uv_curr_out.clear();
  indices_out.clear();

  if (!have_prev_) return false;
  return stepImpl(rgb, uv_prev_out, uv_curr_out, indices_out);
}

}  // namespace livo_recon
