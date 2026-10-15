#pragma once

#include "livo_recon/lio/voxelplane.h"
#include "livo_recon/map/binned/binned_bins.h"
#include <memory>
#include <tuple>

namespace livo_recon
{

// Sets the monotonic frame index of the frame currently being processed,
// called once per frame by VoxelMap::updateMap() -- used by VoxelNode::
// insertPoints() to count how many DISTINCT frames have contributed points
// to a plane between init and (would-be) convergence (see VoxelNode::
// distinct_frames_), which convergence_mode=="frame_gated" requires reach
// min_frames_to_converge before locking in.
void setCurrentFrame(int idx);
// During stationary bootstrap, fit/split incrementally but do not permanently
// lock nodes before all calibration observations have been seen.
void setBootstrapInsertion(bool active);

struct BootstrapObservationStatsRow {
  int32_t node_id = -1;
  VoxelKey root_key;
  int layer = 0;
  int observation_id = -1;
  int point_count = 0;
  V3D sum_p = V3D::Zero();
  M3D sum_pp = M3D::Zero();
};

class VoxelNode
{
public:
  VoxelNode(VoxelOptsPtr opts, VoxelStatsPtr stats, int layer, const V3D& center);
  ~VoxelNode();

  void insertPoints(const std::vector<PointXYZCov>& points_world,
                    std::vector<PlaneUpdate>& updates);
  // scan_id: threaded straight through to VoxelPlane::computeResidual()'s
  // own scan_id param (see its doc comment) -- only VoxelMap::
  // findPlaneResidual() ever supplies one (its own frame_idx_ member).
  bool findPlaneResidual(const WorldPointCov& pt, Residual& res, int scan_id = -1) const;

  int32_t nodeId() const { return node_id_; }

  // Diagnostic-only (see investigation into livo_recon's fast-motion
  // tier0/1 hit-rate drop vs FAST-LIVO2): true iff this node itself
  // (top-level only, does NOT recurse into PARENT children) has a usable
  // plane fit -- NOT gated on status_==CONVERGED, since VoxelPlane::
  // computeResidual() only checks is_plane_ (see voxelplane.cpp), so an
  // OPEN node (still accumulating points toward isFull()/max_points) with
  // a valid-but-not-yet-full plane can already match residuals fine.
  // Gating this on CONVERGED status undercounts real "mismatch" misses as
  // "coverage" ones (a bug caught and fixed after initial results looked
  // suspiciously coverage-dominated). PARENT/DISABLED nodes already set
  // plane_retired_ (see insertPoints()), so that check alone correctly
  // excludes them without needing to check status_.
  bool hasConvergedPlane() const { return !plane_retired_ && plane_.isPlane(); }

  // Recursively exports every live octree node. Diagnostic-only and called
  // once after calibration/map seeding when explicitly enabled.
  void appendPlaneSnapshots(const VoxelKey& root_key,
                            std::vector<VoxelPlaneSnapshot>& out) const;
  void appendBootstrapObservationSupport(const VoxelKey& root_key,
      std::vector<BootstrapObservationStatsRow>& out) const;

private:
  void passToChildren(const std::vector<PointXYZCov>& points_world,
                      std::vector<PlaneUpdate>& updates);
  int childIndex(const V3D& p) const;

  VoxelOptsPtr  opts_;
  VoxelStatsPtr stats_;
  // Perf audit finding #7: inlined (was a separately heap-allocated
  // VoxelPlane*) to remove one level of pointer-chasing on every
  // findPlaneResidual() hit -- this and the VoxelKeyMap hash lookup that
  // finds the owning VoxelNode were the two heap dereferences the per-point
  // hot loop paid on every match attempt. VoxelPlane is default-copyable/
  // -movable (no custom dtor) and constructed once, in this class's own
  // constructor initializer list, exactly as the old `new VoxelPlane(opts)`
  // was. plane_retired_ replaces the old null-pointer sentinel for "this
  // node transitioned to PARENT/DISABLED and its plane data is stale" --
  // set (never cleared) at the same two call sites that used to `delete
  // plane_ptr_; plane_ptr_ = nullptr;`. The retired VoxelPlane object
  // itself is simply left inert (not destroyed/freed) for the rest of this
  // node's lifetime -- a small, bounded amount of memory (one VoxelPlane's
  // worth) versus the alternative of adding a reset()/clear() method,
  // which only a minority of nodes (ones that get subdivided or give up)
  // ever pay at all.
  VoxelPlane    plane_;
  bool          plane_retired_ = false;
  int           layer_;
  V3D           voxel_center_;
  VoxelStatus   status_;

  int32_t   node_id_;
  VoxelNode *leaves_[8];
  int update_count_ = 0;   // raw points received since the last fit (both paths)
  // R88 (binned map): this node's bins, created on the first insertion when opts_->binned_enabled. In that mode update_count_ is the number of UNIQUE bins that received a return since the last fit (BinSet::dirtyCount(); before the first fit
  // that is the number of occupied bins; R92: the only count, the R88 to R90 `opened` count was removed), and the admission rule reads it exactly as it reads raw points elsewhere.
  std::unique_ptr<binned::BinSet> binset_;
  std::vector<PointXYZCov> points_;
  // R65 (F-158): true when the most recent binned insertion appended its batch to points_.
  bool batch_stored_in_points_ = false;

  // Running sufficient statistics paralleling points_ (see RunningMoments' docs) -- added to whenever a point is appended to points_, reset (with ref rebased to voxel_center_) whenever points_ is cleared
  // (subdivision/lock-in). Consulted by the default (raw-point) pca path only; the binned map keeps its returns in binset_.
  RunningMoments running_moments_;

  // Frame-diversity tracking (see setCurrentFrame()'s docs):
  // last_frame_idx_seen_ starts at -1 so the first ever insertPoints() call
  // with non-empty points always counts as one new distinct frame.
  int last_frame_idx_seen_ = -1;
  int distinct_frames_     = 0;
  struct BootstrapObservationAccumulator {
    int count = 0;
    V3D sum_p = V3D::Zero();
    M3D sum_pp = M3D::Zero();
  };
  robin_hood::unordered_flat_map<int, BootstrapObservationAccumulator> bootstrap_observation_support_;

  // distinct_frames_'s value as of this node's last VoxelPlane::update()
  // call (init or refit) -- lets insertPoints() measure how many NEW
  // distinct frames elapsed SINCE that last fit (distinct_frames_ - this),
  // rather than distinct_frames_'s own cumulative since-creation total
  // (which conflates "how long has this voxel existed" with "how often
  // does it actually get refit").
  int last_update_distinct_frames_ = 0;
};

using VoxelKeyMap = robin_hood::unordered_flat_map<VoxelKey, VoxelNode*, VoxelKeyHash>;

}  // namespace livo_recon
