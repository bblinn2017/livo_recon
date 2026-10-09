#pragma once
// R81 (F-468, F-485 Part 3): the two plane-variance keys, as pure arithmetic (no Eigen, no ROS) so the order and the refusal rules are testable on their own
// (tests/test_plane_var_r81.cpp). VoxelPlane::gate() calls planeVarR81Apply() ONLY when a key is on; at the defaults (both 0.0) the call is skipped and the
// plane variance term is the value the R80 code produced, bit for bit.
//
//   voxel_map/plane/plane_var_scan_s2 (s2, default 0.0 = off): plane_var_term := plane_var_term + s2 / n_scans, n_scans = max(1, distinct frames of the voxel)
//   voxel_map/plane/plane_var_min     (v_min, default 0.0 = off): plane_var_term := max(plane_var_term, v_min)
//   Order: the scan term is added first, then the clamp. Both keys must be finite and >= 0 (a negative, NaN or infinite value is refused at startup).
#include <algorithm>
#include <cmath>
#include <string>

namespace livo_recon
{

// Empty string = valid; otherwise the refusal text. The comparisons are written so that NaN fails them.
inline std::string planeVarR81Error(double v_min, double s2)
{
  if (!(v_min >= 0.0) || !std::isfinite(v_min))
    return "voxel_map/plane/plane_var_min must be finite and >= 0 (0 = off); got " + std::to_string(v_min);
  if (!(s2 >= 0.0) || !std::isfinite(s2))
    return "voxel_map/plane/plane_var_scan_s2 must be finite and >= 0 (0 = off); got " + std::to_string(s2);
  return std::string();
}

// plane_var_fit: the plane variance term the fit gave (J_nq plane_var J_nq^T). distinct_frames: the voxel's distinct-frame count (< 1 is treated as 1).
// scan_added / clamped (optional): set to whether each step changed the value's inputs (scan term added with s2 > 0; clamp raised the value).
inline double planeVarR81Apply(double plane_var_fit, double s2, double v_min, int distinct_frames, bool* scan_added = nullptr, bool* clamped = nullptr)
{
  double v = plane_var_fit;
  if (scan_added) *scan_added = false;
  if (clamped) *clamped = false;
  if (s2 > 0.0)
  {
    v += s2 / static_cast<double>(std::max(1, distinct_frames));
    if (scan_added) *scan_added = true;
  }
  if (v_min > 0.0 && v < v_min)
  {
    v = v_min;
    if (clamped) *clamped = true;
  }
  return v;
}

}  // namespace livo_recon
