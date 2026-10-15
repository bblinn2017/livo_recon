#pragma once

// R87. The point-count admission rule of a voxel fit, in one ROS-free header so both fit paths (plain and binned pca, src/lio/voxelnode.cpp) use the same decision and a
// unit test can show it fires (tests/test_admission_rule_r87.cpp).
//
//   fitWanted(admitted, since_fit, min_init, min_update, bootstrap):
//     admitted == false (the voxel has had no admitted fit yet): fit once `since_fit` >= min_init (voxel_map/points/min_init). `since_fit` is the number of raw points
//       received since the last fit, i.e. all points received so far for a voxel that has never been fitted.
//     admitted == true: refit once `since_fit` >= min_update (voxel_map/points/min_update), EXCEPT in the bootstrap frame (global frame index 0), where min_update does not
//       gate: every batch refits (the bootstrap map is built from several stationary scans and must not wait on update batches).
//   min_init is NOT relaxed in the bootstrap frame: a voxel with fewer than min_init raw points gets no fit there either.
//   Points are counted raw (one per return). Counting distinct beams is a different rule (the binned voxel class) and is not done here.
//
// Before R87 only the pca path applied min_init / min_update, and it applied min_update in the bootstrap frame as well.

namespace livo_recon
{
namespace admission
{
inline bool fitWanted(bool admitted, long since_fit, int min_init, int min_update, bool bootstrap)
{
  if (!admitted) return since_fit >= min_init;
  return bootstrap || since_fit >= min_update;
}
}  // namespace admission
}  // namespace livo_recon
