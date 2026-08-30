#include "livo_recon/vio/crop_utils.h"

#include <algorithm>
#include <cmath>

namespace livo_recon
{

cv::Rect centerCropRect(int cam_width, int cam_height, int target_width, int target_height)
{
  const double input_ar  = static_cast<double>(cam_width) / cam_height;
  const double target_ar = static_cast<double>(target_width) / target_height;

  int crop_w, crop_h;
  if (input_ar > target_ar)
  {
    crop_h = cam_height;
    crop_w = static_cast<int>(std::round(cam_height * target_ar));
  }
  else
  {
    crop_w = cam_width;
    crop_h = static_cast<int>(std::round(cam_width / target_ar));
  }
  crop_w = std::min(std::max(crop_w, 1), cam_width);
  crop_h = std::min(std::max(crop_h, 1), cam_height);

  const int left = (cam_width - crop_w) / 2;
  const int top  = (cam_height - crop_h) / 2;
  return cv::Rect(left, top, crop_w, crop_h);
}

}  // namespace livo_recon
