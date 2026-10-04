#pragma once
// R74 (F-428): the test values of the plane and eigengap decisions of one octree node, as plain numbers, so the offline analysis never has
// to re-derive them. ROS-free and pure (tests/test_plane_test_values.cpp).
//   e0 <= e1 <= e2 are the node's eigenvalues (VoxelPlane::eigen_values_, the values the decisions used).
//   edge_m                       voxel_size / 2^layer
//   planarity_value              e0, the quantity compared with plane_threshold (pca: is_plane = e0 < plane_threshold)
//   planarity_pass               is_plane as stored
//   second_eig_below_threshold   e1 < plane_threshold (the F-430 line test: two small eigenvalues = the normal is not determined)
//   eigengap_denom1, denom2      e0 - e1 and e0 - e2 (the denominators of the plane-covariance Jacobian; they appear in the code exactly so)
//   ratio_e1_over_e0, ratio_e2_over_e1   e1 / e0 and e2 / e1 (nan when the divisor is 0)
//   rel_gap_threshold            1 + 3 sqrt(2 / (N - 1)) with N = point_count (nan when N < 2): the F-432 relative rule calls two neighbouring
//                                eigenvalues distinguishable when their ratio exceeds it.
#include <cmath>
#include <limits>
#include <ostream>

namespace livo_recon
{
struct PlaneTestValues
{
  double edge_m = 0.0, plane_threshold = 0.0, planarity_value = 0.0;
  int planarity_pass = 0, second_eig_below_threshold = 0;
  double eigengap_denom1 = 0.0, eigengap_denom2 = 0.0, ratio_e1_over_e0 = 0.0, ratio_e2_over_e1 = 0.0, rel_gap_threshold = 0.0;
};

inline double relativeGapThreshold(int n_points, double n_sigma = 3.0)
{
  if (n_points < 2) return std::numeric_limits<double>::quiet_NaN();
  return 1.0 + n_sigma * std::sqrt(2.0 / static_cast<double>(n_points - 1));
}

inline PlaneTestValues planeTestValues(double e0, double e1, double e2, int layer, double voxel_size, double plane_threshold,
                                       bool is_plane, int point_count)
{
  const double nanv = std::numeric_limits<double>::quiet_NaN();
  PlaneTestValues v;
  v.edge_m = voxel_size / static_cast<double>(1 << layer);
  v.plane_threshold = plane_threshold;
  v.planarity_value = e0;
  v.planarity_pass = is_plane ? 1 : 0;
  v.second_eig_below_threshold = e1 < plane_threshold ? 1 : 0;
  v.eigengap_denom1 = e0 - e1;
  v.eigengap_denom2 = e0 - e2;
  v.ratio_e1_over_e0 = e0 != 0.0 ? e1 / e0 : nanv;
  v.ratio_e2_over_e1 = e1 != 0.0 ? e2 / e1 : nanv;
  v.rel_gap_threshold = relativeGapThreshold(point_count);
  return v;
}

inline const char* planeTestValuesHeader()
{
  return "edge_m,plane_threshold,planarity_value,planarity_pass,second_eig_below_threshold,eigengap_denom1,eigengap_denom2,"
         "ratio_e1_over_e0,ratio_e2_over_e1,rel_gap_threshold";
}

inline void writePlaneTestValues(std::ostream& o, const PlaneTestValues& v)
{
  o << v.edge_m << ',' << v.plane_threshold << ',' << v.planarity_value << ',' << v.planarity_pass << ',' << v.second_eig_below_threshold
    << ',' << v.eigengap_denom1 << ',' << v.eigengap_denom2 << ',' << v.ratio_e1_over_e0 << ',' << v.ratio_e2_over_e1 << ',' << v.rel_gap_threshold;
}
}  // namespace livo_recon
