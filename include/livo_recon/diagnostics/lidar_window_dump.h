#pragma once
// TEMPORARY diagnostic (2026-10-27, independent LiDAR-registration
// investigation). Dumps each frame's deskewed/downsampled LiDAR points
// (mg.points, still in BODY/IMU frame -- NOT transformed to world here;
// the standalone offline registration script does that itself using each
// frame's already-logged pose from odometry.txt/state_trace.csv) to a flat
// CSV, for frames whose scan_id (voxel_map's frame_idx_) falls within
// [1, diag/dump_lidar_window_end_scan]. Default 0 = disabled, no file
// written, zero behavior change. Gated separately from force_vio_residual_
// log since this is a different, narrower investigation.
//
// Delete this file + its call site (livo_recon_node.cpp) + the
// dump_lidar_window_end_scan_ option once done.
#include <vector>

namespace livo_recon
{
struct PointXYZCov;

void writeLidarWindowPoints(int scan_id, double t_abs, const std::vector<PointXYZCov>& points);

}  // namespace livo_recon
