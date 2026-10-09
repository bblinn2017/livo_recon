#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace livo_recon
{

// R66 item 0b: one row per MeasureGroup (frame), written after deskew + downsample.
// Columns: frame_index, t_abs_s, frame_dt_s, n_lidar_raw, n_points_kept, n_imu_samples.
// frame_dt_s is the difference to the previous written frame (0 for the first row).
void writeFrameWindowRow(std::size_t frame_index, double t_abs_s, std::size_t n_lidar_raw,
                         std::size_t n_points_kept, int n_imu_samples);

}  // namespace livo_recon
