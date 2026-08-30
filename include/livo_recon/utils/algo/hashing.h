#pragma once

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/algo/robin_hood.h"
#include <vector>

namespace livo_recon
{

struct VoxelKey
{
  int x, y, z;

  bool operator==(const VoxelKey& other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey& k) const
  {
    static constexpr std::size_t P1 = 73856093, P2 = 19349669, P3 = 83492791;
    return (std::size_t)k.x * P1 ^ (std::size_t)k.y * P2 ^ (std::size_t)k.z * P3;
  }
};

// Two per-cell reduction strategies, A/B-tested this session on eee_01
// (see each mode's own results in test_results/):
//   FIRST (default, best result so far) -- a plain hash-map emplace keeps
//     one arbitrary raw point per cell (whichever the map happens to see
//     first) and discards the rest. Arbitrary, but never fabricates a
//     position/covariance that doesn't correspond to a real return.
//   AVERAGE -- literal mean position and literal mean covariance (plain
//     elementwise average of the per-point covariances already computed
//     upstream by getBodyCov(), no position-spread term added). Mirrors
//     FAST-LIVO2's downSizeFilterSurf position-averaging step, but stops
//     short of its full approach: FAST-LIVO2 discards the old per-point
//     covariances entirely and recomputes a fresh one from its sensor
//     noise model at the averaged position, which this mode does not do
//     (still an open confound -- see hashing.h git history/PRECISION_WEIGHTED
//     removal for the earlier "law of total variance" variant, which added
//     a position-spread term and regressed even worse than this one).
//
// MEDOID and PRECISION_WEIGHTED variants were also tried and removed after
// tying (MEDOID) or regressing (PRECISION_WEIGHTED, ATE 1.55m) relative to
// FIRST -- both still blend/select based on position, so neither avoided
// the "ghost point across a real depth discontinuity" failure mode any
// better than plain AVERAGE once a cell straddles two real surfaces.
enum class DsMode { FIRST, AVERAGE };

template <typename PointType, typename KeyFunc>
void voxelDownsample(
    const std::vector<PointType>& input,
    std::vector<PointType>& output,
    KeyFunc key_func,
    DsMode mode = DsMode::FIRST)
{
  output.clear();

  if (mode == DsMode::FIRST) {
    robin_hood::unordered_flat_map<VoxelKey, PointType, VoxelKeyHash> voxel_map;
    voxel_map.reserve(input.size());
    for (const auto& pt : input)
      voxel_map.emplace(key_func(pt), pt);
    output.reserve(voxel_map.size());
    for (const auto& kv : voxel_map)
      output.push_back(kv.second);
    return;
  }

  robin_hood::unordered_flat_map<VoxelKey, std::vector<const PointType*>, VoxelKeyHash> voxel_map;
  voxel_map.reserve(input.size());
  for (const auto& pt : input)
    voxel_map[key_func(pt)].push_back(&pt);

  output.reserve(voxel_map.size());
  for (const auto& kv : voxel_map) {
    const auto& pts = kv.second;
    if (pts.size() == 1) { output.push_back(*pts[0]); continue; }  // nothing to combine

    // AVERAGE: literal mean position and literal mean covariance
    // components (sensor/pose averaged separately, not combined -- see
    // PointXYZCov's docs).
    V3D sum_point = V3D::Zero();
    M3D sum_sensor_cov = M3D::Zero();
    M3D sum_pos_cov    = M3D::Zero();
    for (const auto* p : pts) {
      sum_point += p->point;
      sum_sensor_cov += p->sensor_cov;
      sum_pos_cov    += p->pos_cov;
    }
    const double n = static_cast<double>(pts.size());
    PointType out_pt{};
    out_pt.point = sum_point / n;
    out_pt.sensor_cov = sum_sensor_cov / n;
    out_pt.pos_cov    = sum_pos_cov / n;
    output.push_back(out_pt);
  }
}

inline VoxelKey worldToKeyFn(const V3D& p, double voxel_size)
{
  VoxelKey key;
  key.x = static_cast<int>(std::floor(p.x() / voxel_size));
  key.y = static_cast<int>(std::floor(p.y() / voxel_size));
  key.z = static_cast<int>(std::floor(p.z() / voxel_size));
  return key;
}

struct PointXYZCovKeyFn
{
  double voxel_size;

  VoxelKey operator()(const PointXYZCov& pt) const
  {
    return worldToKeyFn(pt.point, voxel_size);
  }
};

struct V3DKeyFn
{
  double voxel_size;

  VoxelKey operator()(const V3D& pt) const
  {
    return worldToKeyFn(pt, voxel_size);
  }
};

}
