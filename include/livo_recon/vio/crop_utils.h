#pragma once

#include <opencv2/core.hpp>

namespace livo_recon
{

// Largest same-aspect-ratio-as-target crop, centered in a cam_width x
// cam_height image -- shared by every native point-tracking backend (see
// vio/point_tracker.h), all of which center-crop to their model's own
// aspect ratio before resizing rather than anisotropically squishing,
// since the underlying models are trained on aspect-preserving crops.
cv::Rect centerCropRect(int cam_width, int cam_height, int target_width, int target_height);

}  // namespace livo_recon
