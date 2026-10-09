#include "livo_recon/map/binned_voxelmap.h"

#include <sstream>
#include <stdexcept>

#include "livo_recon/map/binned/binned_bins.h"
#include "livo_recon/utils/log/param_warn.h"

namespace livo_recon
{

std::string BinnedVoxelMap::loadParameters(ros::NodeHandle& pnh)
{
  std::string base = VoxelMap::loadParameters(pnh);
  VoxelOpts& o = *opts();

  double fraction = 0.2;
  std::string cov = "average", weight = "count";
  paramWarn<double>(pnh, "voxel_map/binned/bin_size_fraction", fraction, 0.2);
  paramWarn<std::string>(pnh, "voxel_map/binned/cov_mode", cov, "average");
  paramWarn<std::string>(pnh, "voxel_map/binned/weight_mode", weight, "count");

  if (!(fraction > 0.0 && fraction <= 1.0))
    throw std::runtime_error("[config] refused: voxel_map/binned/bin_size_fraction=" + std::to_string(fraction) + " -- must be in (0, 1]");
  binned::CovMode cm;
  binned::WeightMode wm;
  if (!binned::parseCovMode(cov, cm))
    throw std::runtime_error("[config] refused: voxel_map/binned/cov_mode=" + cov + " -- must be average or shrunk");
  if (!binned::parseWeightMode(weight, wm))
    throw std::runtime_error("[config] refused: voxel_map/binned/weight_mode=" + weight + " -- must be count or uniform");
  if (o.plane_fit_mode != "debiased")
    throw std::runtime_error("[config] refused: voxel_map/backend=binned needs voxel_map/plane/plane_fit_mode=debiased (the bins feed the debiased accumulators; the pca path has its own use_bins)");
  if (o.plane_fit_pose_cov_mode != "sensor_only")
    throw std::runtime_error("[config] refused: voxel_map/backend=binned needs voxel_map/plane/plane_fit_pose_cov_mode=sensor_only (bins keep the sensor covariance only)");
  if (o.use_bins)
    throw std::runtime_error("[config] refused: voxel_map/backend=binned and voxel_map/plane/use_bins=true are two different binnings; set use_bins=false");

  o.binned_enabled = true;
  o.binned_bin_fraction = fraction;
  o.binned_cov_mode = (cm == binned::CovMode::Shrunk) ? 1 : 0;
  o.binned_weight_mode = (wm == binned::WeightMode::Uniform) ? 1 : 0;

  std::ostringstream oss;
  oss << base
      << "\n  binned/bin_size_fraction:        " << fraction << "   (bin edge " << fraction * o.voxel_size << " m at the root voxel)"
      << "\n  binned/cov_mode:                 " << binned::covModeName(cm)
      << "\n  binned/weight_mode:              " << binned::weightModeName(wm)
      << "\n  binned/min_init, min_update:     " << o.min_init_points << ", " << o.min_update_points << "   (counted in bins)";
  return oss.str();
}

}  // namespace livo_recon
