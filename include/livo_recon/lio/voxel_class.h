#pragma once
// R82 (D4, F-500, Bryce 2026-10-06): one place that decides, at FIT time and for both fits (pca VoxelPlane::update and debiased refitDebiased), whether a voxel is rejected, a plane or a line.
// ROS-free and pure so tests/test_voxel_class_r82.cpp can check it. Used only when voxel_map/plane/line_residual/classify_at_fit is true; the default (off) never calls it and every fit is bit-identical to R81.
//
// Input: the FINAL eigenvalues e0 <= e1 <= e2 of the fit (for the debiased fit after the noise correction, the clamp at 0, the sensor floor and the e0 <= e1 clamp), the number of points, the threshold the
// voxel's e0 is tested against (plane_threshold, with the hysteresis factor on the debiased fit), and the line rule (LineOptions of line_residual).
// Order of the decision:
//   1. non-finite eigenvalue, or e2 < degenerate_floor (a cluster with no direction): Reject (kDegenerate).
//      With the line model OFF the old test is kept exactly: e1 < degenerate_floor also rejects. With it ON (a hard form) a voxel with e1 < degenerate_floor <= e2 is a perfect line and is kept.
//   2. e0 >= plane_threshold (a blob): Reject (kBlob).
//   3. hard form and the line rule says line: Line.
//   4. otherwise Plane. (Smooth form has no classes: every accepted voxel is a Plane and takes its three rows later.)
#include <cmath>

#include "livo_recon/lio/line_residual.h"

namespace livo_recon
{
enum class VoxelClass { Reject = 0, Plane = 1, Line = 2 };
enum class VoxelRejectWhy { None = 0, NonFinite = 1, Degenerate = 2, Blob = 3 };

struct VoxelClassResult
{
  VoxelClass cls = VoxelClass::Reject;
  VoxelRejectWhy why = VoxelRejectWhy::None;
  bool e1_tiny_kept = false;   // a line kept although e1 < degenerate_floor (the old test would have rejected it)
};

inline VoxelClassResult classifyVoxel(const LineOptions& line, double e0, double e1, double e2, int n_points, double plane_threshold_eff,
                                      double degenerate_floor = 1e-8)
{
  VoxelClassResult r;
  if (!std::isfinite(e0) || !std::isfinite(e1) || !std::isfinite(e2)) { r.why = VoxelRejectWhy::NonFinite; return r; }
  const bool hard = (line.mode != LineMode::Off && line.mode != LineMode::Smooth);
  if (hard ? (e2 < degenerate_floor) : (e1 < degenerate_floor || e2 < degenerate_floor)) { r.why = VoxelRejectWhy::Degenerate; return r; }
  if (!(e0 < plane_threshold_eff)) { r.why = VoxelRejectWhy::Blob; return r; }
  if (hard && isLineVoxel(line, e0, e1, n_points, e2)) {
    r.cls = VoxelClass::Line;
    r.e1_tiny_kept = e1 < degenerate_floor;
    return r;
  }
  r.cls = VoxelClass::Plane;
  return r;
}
}  // namespace livo_recon
