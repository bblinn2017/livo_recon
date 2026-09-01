#pragma once

#include "livo_recon/lio/voxelplane.h"

namespace livo_recon
{

// Sets the monotonic frame index of the frame currently being processed,
// called once per frame by VoxelMap::updateMap() -- used by VoxelNode::
// insertPoints() to count how many DISTINCT frames have contributed points
// to a plane between init and (would-be) convergence (see VoxelNode::
// distinct_frames_), which convergence_mode=="frame_gated" requires reach
// min_frames_to_converge before locking in.
void setCurrentFrame(int idx);

// Running sufficient statistics for one fine spatial bin (a fixed fraction
// of the node's own voxel extent, see VoxelNode's density_weight_leaf_),
// used by VoxelNode::insertPoints()/bins_ whenever convergence_mode==
// "always_update" -- to bound memory/refit cost, avoiding re-binning the
// whole accumulated point set from scratch on every single refit call.
// Without this, a plane that never locks into CONVERGED keeps appending to
// an ever-growing raw points_ vector, and VoxelPlane::update()'s O(N)
// refit on every min_update_points new points compounds into a real
// pipeline stall (confirmed empirically: one voxel's points_ reached
// 229,514 points under always_update with undecimated map insertion,
// stalling the whole node's real-time processing and producing a genuine ATE
// regression despite the resulting plane fits themselves scoring fine on
// acceptance rate/coverage). Each new point is merged into its bin in O(1)
// (count++, sum_point += point, sum_cov += cov) instead of being stored
// raw, bounding the number of live entries to however many distinct fine
// cells the voxel's volume actually contains, not how many points/frames
// have ever contributed.
//
// Passing each bin's mean (sum_point/count, sum_cov/count) into
// VoxelPlane::update() with weight = count gives an APPROXIMATE
// reconstruction of the true unweighted PCA over every raw point: exact
// for the mean, and exact for the covariance's between-bin term, but
// missing each bin's own within-bin positional scatter -- the smaller
// density_weight_leaf_ is relative to real surface curvature, the smaller
// that omitted term, but it is not exactly zero.
struct PointBin
{
  int count = 0;
  V3D sum_point = V3D::Zero();
  // Sensor/pose covariance summed SEPARATELY, matching PointXYZCov's own
  // direct-storage convention (no combined field, no subtraction) -- a
  // consumer combines sensor_cov+pos_cov (or uses sensor_cov alone,
  // see plane_fit_pose_cov_mode) at the point of use, same as for any
  // other PointXYZCov.
  M3D sum_sensor_cov = M3D::Zero();
  M3D sum_pos_cov    = M3D::Zero();
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
  // suspiciously coverage-dominated). PARENT/DISABLED nodes already null
  // out plane_ptr_ (see insertPoints()), so the plane_ptr_ null check
  // alone correctly excludes them without needing to check status_.
  bool hasConvergedPlane() const { return plane_ptr_ != nullptr && plane_ptr_->isPlane(); }

private:
  void passToChildren(const std::vector<PointXYZCov>& points_world,
                      std::vector<PlaneUpdate>& updates);
  int childIndex(const V3D& p) const;

  // Rebuilds (representative point, weight) pairs from bins_ -- used both
  // when refitting and (rarely) when subdividing a non-planar always_update
  // node, so children/VoxelPlane::update() see bins_'s current contents
  // either way. See PointBin's docs for the weight choice.
  // fit_weights: opts_->bin_weight_mode_fit. var_weights: opts_->
  // bin_weight_mode_var (see T3-0c -- separate weightings for the plane
  // fit vs. its uncertainty scaling).
  void buildBinReps(std::vector<PointXYZCov>& reps, std::vector<double>& fit_weights,
                     std::vector<double>& var_weights) const;

  VoxelOptsPtr  opts_;
  VoxelStatsPtr stats_;
  VoxelPlane   *plane_ptr_;
  int           layer_;
  V3D           voxel_center_;
  VoxelStatus   status_;

  int32_t   node_id_;
  VoxelNode *leaves_[8];
  int update_count_ = 0;
  std::vector<PointXYZCov> points_;

  // Running sufficient statistics paralleling points_ (see RunningMoments'
  // docs) -- added to whenever a point is appended to points_, reset
  // (with ref rebased to voxel_center_) whenever points_ is cleared
  // (subdivision/lock-in). Only meaningful/consulted when useBins() is
  // false -- the weighted bin path doesn't use this.
  RunningMoments running_moments_;

  // Bounded alternative to points_ (see PointBin's docs) -- populated/
  // consulted instead of points_ whenever useBins() is true.
  robin_hood::unordered_flat_map<VoxelKey, PointBin, VoxelKeyHash> bins_;

  int total_count_ = 0;

  // History (123-130): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelnode.h-123
  bool useBins() const { return opts_->use_bins; }

  // This node's own bin size for the PointBin accumulator, in world
  // units -- a fraction (opts_->bin_size_fraction) of this node's OWN
  // voxel extent (opts_->voxel_size / 2^layer_), computed once at
  // construction so subdivided (smaller, deeper) nodes get proportionally
  // finer bins instead of a single fixed-meters size that would be
  // misscaled for most layers.
  const double density_weight_leaf_;

  // Frame-diversity tracking (see setCurrentFrame()'s docs):
  // last_frame_idx_seen_ starts at -1 so the first ever insertPoints() call
  // with non-empty points always counts as one new distinct frame.
  int last_frame_idx_seen_ = -1;
  int distinct_frames_     = 0;

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
