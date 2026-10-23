#pragma once

namespace livo_recon {

// log/lidar_image_bounds (default false): write-only per-frame row counting
// how many of this frame's mg.points land in front of the camera AND inside
// the image bounds when projected with the CURRENT state's imuToCamera()/
// camera intrinsics. Measured right after deskewAndDownsample() populates
// mg.points (same point in LivoReconNode::estimateState() as the existing
// log/frame_window row) -- i.e. with the state as it enters this frame,
// BEFORE this frame's own LIO/VIO correction, not the post-correction pose.
void writeLidarImageBoundsRow(int scan_id, double t_abs, int n_points,
                              int n_behind_camera, int n_out_of_bounds,
                              int n_visible);

}  // namespace livo_recon
