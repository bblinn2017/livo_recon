#pragma once

#include <vector>

// ============================================================================
// R78 (Bryce 2026-10-04: "use all points for map building but only use filtered points for the iekf"):
// which deskewed, downsampled point set the IEKF residuals are built from.
//
//   primary (default): mg.points, the same set the voxel map is built from (bit-identical to R77).
//   second:            mg.dry_run_points, the second decimation stream (lio/dry_run_point_filter_num, deskewed and downsampled exactly like
//                      mg.points, from the same raw scan). With cbk/lidar/point_filter_num = 1 the map receives every point and the IEKF receives the
//                      every-k-th set a point-filter-k run would give it. The map update always reads mg.points.
//
// A frame whose second-stream set is empty falls back to the primary set and is counted (fell_back), so a run that silently did not split is visible.
// Header-only, ROS-free (unit-tested in tests/test_iekf_stream_r78.cpp).
// ============================================================================

namespace livo_recon
{

struct IekfStreamPick
{
  bool second_used = false;   // the IEKF used the second-stream set this frame
  bool fell_back = false;     // second was requested, its set was empty, the primary set was used
};

template <class PointVec>
inline const PointVec& pickIekfPoints(bool want_second, const PointVec& primary, const PointVec& second, IekfStreamPick& out)
{
  out = IekfStreamPick();
  if (!want_second) return primary;
  if (second.empty())
  {
    out.fell_back = true;
    return primary;
  }
  out.second_used = true;
  return second;
}

}  // namespace livo_recon
