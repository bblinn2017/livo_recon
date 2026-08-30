#pragma once

#include <memory>
#include <string>

#include "livo_recon/utils/map/voxelmap_utils.h"

// Shared structs/options for AkfMap (Phase 3 of the plan at
// ~/.claude/plans/imperative-tumbling-karp.md) -- mirrors
// voxelmap_utils.h's role for VoxelMap. AkfMap is a faithful port of
// AKF-LIO's map (https://github.com/xpxie/AKF-LIO, cloned at build time
// to /tmp/akf_lio_clone -- see that repo's include/ivox3d/ivox3d.h,
// ivox3d_node.hpp, and src/laser_mapping.cc's ObsModel()/MapIncremental()
// for the design this ports). Default values below are the actual
// RUNTIME defaults AKF-LIO's own LoadParams() passes to nh.param() as the
// fallback -- NOT always the same as the class member initializers seen
// in laser_mapping.h (confirmed two literal mismatches while porting:
// t_mal's member initializer is 3 but LoadParams' nh.param default is
// 11.28; t_stop_pseudo_merge_'s member initializer is 1 but LoadParams'
// nh.param default is ALSO 11.28 -- both member initializers are
// pre-load placeholders, never what actually runs).
namespace livo_recon
{

// One Gaussian summary: a fused point (mean + covariance) that may
// represent one or many raw scan returns (pt_num tracks how many).
// Distinct from PointXYZCov (sensor_cov/pos_cov kept separate) -- AkfMap
// combines them into one `cov` once at insertion (see mergeTwoGaussians()
// callers), since AKF-LIO's own fusion math operates on one combined
// covariance throughout, not a sensor/pose split.
struct AkfPoint
{
  V3D    point       = V3D::Zero();
  M3D    cov         = M3D::Identity();
  int    pt_num      = 1;   // how many raw returns this Gaussian summarizes (position/cov fusion weight)
  int    use_num     = 1;   // how many times this point has been matched as a residual (uncertainty-blend weight)
  double uncertainty = 0.0; // post-EKF residual uncertainty, fed back via AkfMap::updateUncertainty()
  double time        = 0.0; // frame time this point was last touched, for LRU/time-based pruning
};

enum class AkfNearbyType { CENTER, NEARBY6, NEARBY18, NEARBY26 };

struct AkfMapOpts
{
  // Spatial hash cell size (ivox_options_.resolution_ in AKF-LIO, default
  // 0.5 -- "ivox_grid_resolution" nh.param default).
  double voxel_size = 0.5;

  // Which neighbor cells a query/insertion searches around the point's
  // own cell (ivox_nearby_type, default 26 -- "ivox_nearby_type" nh.param
  // default in AKF-LIO's LoadParams(), NOT NEARBY6 despite NEARBY6 being
  // IVox::Options' own compile-time default -- confirm this file's own
  // comment above before assuming otherwise).
  AkfNearbyType nearby_type = AkfNearbyType::NEARBY26;

  // Mahalanobis-distance gate for insertion-time fusion (AddPoints()'s
  // KNNPointMAL call) -- a new incoming point only merges into an
  // existing map point if within this threshold; otherwise it seeds a
  // new Gaussian. "t_mal" nh.param default.
  double t_mal = 11.28;

  // Query-time pseudo-merge gates (ObsModel()'s p_sum accumulation loop):
  // merge unconditionally while the running Gaussian's pt_num is below
  // this ("options::MIN_NUM_MATCH_POINTS", a fixed constant in AKF-LIO,
  // not itself a runtime param there -- kept configurable here).
  int min_num_match_points = 5;
  // Once pt_num >= min_num_match_points, each further candidate must pass
  // BOTH: (a) the already-merged covariance's middle eigenvalue >= this
  // (flatness -- literal 0.04 in ObsModel(), not a named/configurable
  // AKF-LIO param, kept configurable here), and (b) the candidate's
  // eigenvalue-weighted in-plane projection distance <= t_stop_pseudo_merge^2.
  double flatness_eig_floor = 0.04;
  double t_stop_pseudo_merge = 11.28;  // "t_stop_pseudo_merge" nh.param default

  // K-NN search bounds for the query-time neighbor pull (ObsModel()'s
  // ivox_->GetClosestPoint(..., NUM_MATCH_POINTS, 3) call).
  int    num_match_points = 100;  // options::NUM_MATCH_POINTS (fixed constant in AKF-LIO)
  double query_max_range  = 3.0;  // literal `3` argument at that call site

  // Range-adaptive point-to-plane residual gate (ObsModel()'s final
  // |p2pl| > (1/9)*sqrt(range) check) -- range is BODY-frame point norm in
  // AKF-LIO (p_body.norm()), not world-frame; ported as-is (see
  // AkfMap::findPlaneResidual()'s doc comment for why this is intentional
  // to preserve, not "fixed" to world-frame).
  double range_gate_coeff = 1.0 / 9.0;

  // Residual weight scaling: sigma_squared = 1/(exp(t_ratio_b*uncertainty)*thickness).
  // t_ratio_b=0 (AKF-LIO's own "t_ratio_b" nh.param default) makes the
  // exp() term inert (=1), i.e. weight is thickness-only by default --
  // t_ratio_b is an opt-in extra scaling AKF-LIO itself ships off.
  double t_ratio_b = 0.0;

  // New map point seed covariance/uncertainty (AddPoints()'s point_world.uncertainty
  // = init_uncertainty_, and VoxelGridDownsample()'s tp.cov = LIDAR_COV*I).
  double init_uncertainty = 0.01;    // "init_uncertainty" nh.param default
  double init_lidar_cov   = 0.001;   // "mapping/lidar_cov" nh.param default -- seed AkfPoint::cov = init_lidar_cov*I

  // Caps/pruning (ivox3d.h's ErasePoints()/Merge2()).
  int    max_fea_num             = 10000;    // options::MAX_FEA_NUM's actual runtime default (options.cc)
  double time_to_delete_local_map = 1000.0;  // options::TIME_TO_DELETE_LOCAL_MAP's actual runtime default
  double capacity                 = 10000000.0;  // IVox::Options::capacity_'s compile-time default (never overridden via nh.param in AKF-LIO)
};
using AkfMapOptsPtr = std::shared_ptr<AkfMapOpts>;

// Closed-form two-Gaussian moment merge -- ports ivox3d_node.hpp's/
// laser_mapping.cc's Merge2() exactly (both are identical). A convex
// combination of two already-valid Gaussians' second moments: additive,
// PSD by construction, unlike VoxelPlane's debiased fit's subtractive
// correction. `p2` is treated as the "base" whose non-fusion fields
// (time handling aside) seed the output, matching AKF-LIO's own
// `pt3 = pt2` convention.
inline AkfPoint mergeTwoGaussians(const AkfPoint& p1, const AkfPoint& p2, int max_fea_num)
{
  AkfPoint p3 = p2;
  const double ratio1 = static_cast<double>(p1.pt_num) / static_cast<double>(p1.pt_num + p2.pt_num);
  const double ratio2 = static_cast<double>(p2.pt_num) / static_cast<double>(p1.pt_num + p2.pt_num);

  p3.point = ratio1 * p1.point + ratio2 * p2.point;
  p3.cov = ratio1 * (p1.cov + p1.point * p1.point.transpose()) +
           ratio2 * (p2.cov + p2.point * p2.point.transpose()) -
           p3.point * p3.point.transpose();
  p3.pt_num = p1.pt_num + p2.pt_num;
  if (p3.pt_num > max_fea_num) p3.pt_num = max_fea_num;
  p3.time = std::max(p1.time, p2.time);

  const int use_sum = p1.use_num + p2.use_num;
  if (use_sum != 0) {
    const double ratio3 = static_cast<double>(p1.use_num) / static_cast<double>(use_sum);
    const double ratio4 = static_cast<double>(p2.use_num) / static_cast<double>(use_sum);
    p3.uncertainty = ratio3 * p1.uncertainty + ratio4 * p2.uncertainty;
    p3.use_num = use_sum;
    if (p3.use_num > max_fea_num) p3.use_num = max_fea_num;
  }
  return p3;
}

// Uncertainty-only merge (ivox3d_node.hpp's UpdateResidualOnly) -- used by
// AkfMap::updateUncertainty() to fold a post-EKF-converged residual's
// uncertainty into its matched map point without touching position/cov.
inline AkfPoint updateResidualOnly(const AkfPoint& p1, const AkfPoint& p2, int max_fea_num)
{
  AkfPoint p3 = p2;
  const int use_sum = p1.use_num + p2.use_num;
  if (use_sum != 0) {
    const double ratio3 = static_cast<double>(p1.use_num) / static_cast<double>(use_sum);
    const double ratio4 = static_cast<double>(p2.use_num) / static_cast<double>(use_sum);
    p3.uncertainty = ratio3 * p1.uncertainty + ratio4 * p2.uncertainty;
    p3.use_num = use_sum;
    if (p3.use_num > max_fea_num) p3.use_num = max_fea_num;
  }
  return p3;
}

// Mahalanobis distance between two Gaussians' means, using their SUMMED
// covariance (ivox3d_node.hpp's MalDistance: d^T*(cov1+cov2)^-1*d).
inline double malDistance(const V3D& p1, const M3D& cov1, const V3D& p2, const M3D& cov2)
{
  const V3D d = p1 - p2;
  return d.transpose() * (cov1 + cov2).inverse() * d;
}

}  // namespace livo_recon
