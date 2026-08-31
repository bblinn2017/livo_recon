#pragma once

#include "livo_recon/common_lib.h"
#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/map/map_backend.h"

namespace livo_recon
{

using BucketMap = robin_hood::unordered_flat_map<VoxelKey, BucketEntry, VoxelKeyHash>;

class VoxelMap : public MapBackend
{
public:
  VoxelMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues);

  std::string loadParameters(ros::NodeHandle& pnh) override;

  void updateMap(MeasureGroup& mg) override;

  void ensureNode(const VoxelKey &key);
  // tier0_had_plane (if non-null) is set whenever the PRIMARY voxel itself
  // missed (regardless of whether tier1/tier2 subsequently rescue the
  // point or the whole call ultimately returns false): true if that
  // primary voxel already had a converged plane (this specific point's
  // projection just didn't pass its gate -- a geometric/pose-offset
  // mismatch), false if the primary voxel had no converged plane at all
  // (a real tier0-level coverage gap). Left unset if the primary voxel hit
  // directly (no miss to classify). Debug-only, for isolating why tier0's
  // hit-rate specifically degrades during fast motion (see
  // hasConvergedNeighbor()'s broader, whole-pipeline counterpart).
  bool findPlaneResidual(const WorldPointCov& pt, Residual &res, bool* tier0_had_plane = nullptr) const override;

  // T0-D (2026-08-31): gates whether findPlaneResidual() passes a real
  // scan_id (enabling corr.csv logging, see VoxelOpts::
  // log_consistency_corr_en) or -1 (no-op) down to VoxelPlane::
  // computeResidual() -- NOT part of the MapBackend virtual interface,
  // since only LioProc's own IEKF loop knows which call is the frame's
  // FIRST buildResiduals() (the pre-update, un-relinearized innovation
  // the register's nu/S spec wants) versus a later re-linearizing
  // iteration or a throwaway runDryRunShadowPass() call, neither of which
  // should be logged. Defaults true (log allowed) so any call site that
  // doesn't explicitly manage this (there are none as of this writing,
  // but a future one would silently get the safe default rather than
  // silently losing all logging) still works.
  void setAllowConsistencyLog(bool v) { allow_consistency_log_ = v; }

  // Diagnostic-only (see investigation into livo_recon's fast-motion
  // tier0/1 hit-rate drop vs FAST-LIVO2): true iff any voxel within
  // neighborhood_size of p_world's key has a converged plane -- called only
  // for points where findPlaneResidual() already failed, to classify that
  // miss as a real local map-coverage gap (false here) vs a geometric/
  // pose-offset mismatch (true here -- planes exist nearby, none matched).
  bool hasConvergedNeighbor(const V3D& p_world) const override;

  std::string statsString() const override;
  VoxelStatsPtr stats() const { return stats_; }
  bool isEmpty() const override;

  // Per-frame map-insertion density, set at the end of updateMap() -- see
  // the investigation into livo_recon's per-point coverage-miss rate vs
  // FAST-LIVO2's: last_n_map_pts_ is how many points this frame actually
  // got inserted (mg.points.size()), last_n_active_voxels_ is how many
  // distinct voxels received at least one of them -- their ratio is the
  // average points/voxel/frame, directly comparable to FAST-LIVO2's
  // UpdateVoxelMap() (see last_n_map_pts_fastlivo_-equivalent logging
  // added there). frame_idx_ is a monotonic per-frame counter, incremented
  // once per updateMap() call, fed to setCurrentFrame() for VoxelNode's
  // distinct-frame-count tracking (see voxelnode.h). All three are plain
  // fields inherited from MapBackend now, not redeclared here.
  VoxelKey worldToKey(const V3D &p_world) const;
  bool vizDirty() const override { return !viz_dirty_.empty(); }
  visualization_msgs::MarkerArray buildVizMarkers(const std::string& frame_id,
                                                  const ros::Time& stamp) override;

private:
  // See setAllowConsistencyLog()'s doc comment above.
  bool allow_consistency_log_ = true;

  VoxelOptsPtr  opts_;
  VoxelStatsPtr stats_;
  StateGroupPtr state_;
  VoxelKeyMap   voxel_map_;
  ProfilerPtr   profiler_;
  DataQueuesPtr data_queues_;

  // persistent scratch buffers reused each frame to avoid heap churn
  BucketMap bucket_map_;
  std::vector<VoxelKey> active_keys_;
  std::vector<std::pair<const VoxelKey*, BucketEntry*>> bucket_flat_;
  std::vector<PointXYZCov> pts_world_;
  std::vector<std::vector<std::pair<VoxelKey, int>>> thread_keys_;
  std::vector<std::vector<PlaneUpdate>> thread_plane_updates_;

  // plane info cache: node_id → PlaneVizInfo (ground truth for each active plane)
  robin_hood::unordered_flat_map<int32_t, PlaneVizInfo> viz_cache_;
  // dirty set accumulated since last buildVizMarkers: node_id → deleted?
  robin_hood::unordered_flat_map<int32_t, bool>         viz_dirty_;

  // ── visualization geometry lists ─────────────────────────────────────────
  // All plane discs in one TRIANGLE_LIST marker; all dots in one SPHERE_LIST.
  // Each plane patch occupies DISC_SEGS consecutive triangles (3 vertices each).

  static constexpr int DISC_SEGS = 2;   // triangles per plane patch (square = 2 triangles)

  visualization_msgs::Marker disc_list_;  // TRIANGLE_LIST — all plane discs
  visualization_msgs::Marker dot_list_;   // SPHERE_LIST   — converged plane centres

  // disc_seg_owner_[seg] = node_id that owns segment seg (each seg = DISC_SEGS*3 points)
  std::vector<int32_t> disc_seg_owner_;
  // dot_pt_owner_[i] = node_id that owns dot point i
  std::vector<int32_t> dot_pt_owner_;

  // node_id → segment index in disc_list_ (NOT point index)
  robin_hood::unordered_flat_map<int32_t, uint32_t> disc_seg_idx_;
  // node_id → point index in dot_list_
  robin_hood::unordered_flat_map<int32_t, uint32_t> dot_pt_idx_;

  // scan_id: passed straight through from findPlaneResidual() (which
  // supplies its own frame_idx_ member) down to VoxelNode::
  // findPlaneResidual()/VoxelPlane::computeResidual() -- see the latter's
  // doc comment. Kept as an explicit param rather than reading frame_idx_
  // directly so these two stay pure/testable in isolation.
  bool findPlaneResidualDirectional(const WorldPointCov& pt, const VoxelKey& base, Residual &res,
                                    VoxelKey* tried_key = nullptr, int scan_id = -1) const;
  bool findPlaneResidualNeighborhood(const WorldPointCov& pt, const VoxelKey& base, Residual &res,
                                     const VoxelKey* exclude = nullptr, int scan_id = -1) const;

  void eraseDisc(int32_t id);
  void eraseDot(int32_t id);
  void appendDisc(int32_t id, const PlaneVizInfo& p);
  void appendDot(int32_t id, const PlaneVizInfo& p);
};

}
