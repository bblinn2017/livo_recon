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
  std::string cov = "average";
  paramWarn<double>(pnh, "voxel_map/binned/bin_size_fraction", fraction, 0.2);
  paramWarn<std::string>(pnh, "voxel_map/binned/cov_mode", cov, "average");

  if (!(fraction > 0.0 && fraction <= 1.0))
    throw std::runtime_error("[config] refused: voxel_map/binned/bin_size_fraction=" + std::to_string(fraction) + " -- must be in (0, 1]");
  binned::CovMode cm;
  if (!binned::parseCovMode(cov, cm))
    throw std::runtime_error("[config] refused: voxel_map/binned/cov_mode=" + cov + " -- must be average or shrunk");

  o.binned_enabled = true;
  o.binned_bin_fraction = fraction;
  o.binned_cov_mode = (cm == binned::CovMode::Shrunk) ? 1 : 0;

  std::ostringstream oss;
  oss << base
      << "\n  binned/bin_size_fraction:        " << fraction << "   (bin edge " << fraction * o.voxel_size << " m at the root voxel)"
      << "\n  binned/cov_mode:                 " << binned::covModeName(cm)
      << "\n  binned/weight:                   count (w = k; the only weight since R95)"
      << "\n  binned/fit:                      pca, one weighted point per occupied bin"
      << "\n  binned/min_init, min_update:     " << o.min_init_points << ", " << o.min_update_points << "   (counted in unique bins touched since the last fit)";
  return oss.str();
}

}  // namespace livo_recon
