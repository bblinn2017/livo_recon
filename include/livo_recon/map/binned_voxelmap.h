#pragma once
// R88: the "binned" map type (voxel_map/backend = binned). A VoxelMap (same grid, gates, residual query, line rows, diagnostics and tiered search) whose debiased nodes keep their returns in bins
// (map/binned/binned_bins.h) and fit from the occupied bins. Everything that is specific to binning lives in this class, the bin accumulator and VoxelPlane::refitFromBins(); the default map is untouched when this
// backend is not selected.
//
// Config keys (all under voxel_map/binned/):
//   bin_size_fraction  (default 0.2)       bin edge = fraction x the node's own edge, 0 < fraction <= 1
//   cov_mode           average | shrunk    covariance carried by a bin's representative point (C / k, or C / k^2)
//   weight_mode        count | uniform     weight of a bin in the fit sums (k, or 1)
// Required of the rest of the config (refused otherwise, at start-up): plane_fit_mode = debiased, plane_fit_pose_cov_mode = sensor_only, use_bins = false.
// voxel_map/points/min_init and min_update count BINS in this map type (min_init: occupied bins before the first fit; min_update: bins opened since the last fit), with the same bootstrap exemption as the default map.
#include "livo_recon/map/voxelmap.h"

namespace livo_recon
{

class BinnedVoxelMap : public VoxelMap
{
public:
  using VoxelMap::VoxelMap;

  std::string loadParameters(ros::NodeHandle& pnh) override;
};

}  // namespace livo_recon
