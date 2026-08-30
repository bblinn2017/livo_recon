#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include "livo_recon/utils/map/voxelmap_utils.h"

namespace livo_recon
{

struct MeasureGroup;

// Abstract interface both livo_recon map backends (VoxelMap's persistent
// per-voxel plane fits, and AkfMap's AKF-LIO-style raw-point Gaussian
// merge map) implement, so LioProc/PubProc/CombinedProc can drive either
// one without knowing which is active. See the plan at
// ~/.claude/plans/imperative-tumbling-karp.md for the full design
// rationale and Phase 3's AkfMap port -- this header covers Phase 1 only.
//
// Residual/WorldPointCov (voxelmap_utils.h) are the shared currency
// between a backend and LioProc's residual/Jacobian math -- deliberately
// unchanged here, since both are already backend-agnostic ("a plane match
// result with weight" / "a point plus its two noise sources").
class MapBackend
{
public:
  virtual ~MapBackend() = default;

  virtual std::string loadParameters(ros::NodeHandle& pnh) = 0;

  // Called once per LIO frame: insert this frame's points into the map.
  // Implementations own their own internal parallelization and must not
  // introduce non-deterministic reduction order (see utils/algo/
  // omp_utils.h's pinOmpThreadsForDeterminism() doc comment for why).
  virtual void updateMap(MeasureGroup& mg) = 0;

  // Per-point residual query, called from LioProc::buildResiduals()'s
  // existing per-point OMP loop (many thousands of calls/frame) -- must
  // be const/thread-safe with no shared mutable state touched across
  // concurrent calls.
  virtual bool findPlaneResidual(const WorldPointCov& pt, Residual& res,
                                 bool* tier0_had_plane = nullptr) const = 0;

  // Diagnostic-only (see VoxelMap::hasConvergedNeighbor()'s own doc
  // comment for what this classifies). Default "false" reads naturally as
  // "no coverage signal available" for a backend with no converged/locked
  // plane concept (e.g. AkfMap) rather than forcing every backend to
  // implement a concept only VoxelMap actually has.
  virtual bool hasConvergedNeighbor(const V3D&) const { return false; }

  virtual bool isEmpty() const = 0;
  virtual std::string statsString() const = 0;

  virtual bool vizDirty() const { return false; }
  virtual visualization_msgs::MarkerArray buildVizMarkers(const std::string& /*frame_id*/,
                                                           const ros::Time& /*stamp*/)
  {
    return {};
  }

  // Per-frame diagnostic counters (see VoxelMap's own doc comment on
  // these two) -- plain fields, not virtual accessors, so existing
  // lio_processing.cpp logging needs no changes beyond the pointer type.
  int last_n_map_pts_ = 0;
  int last_n_active_voxels_ = 0;
  int frame_idx_ = 0;
};
using MapBackendPtr = std::shared_ptr<MapBackend>;

}  // namespace livo_recon
