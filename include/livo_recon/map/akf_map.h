#pragma once

#include <mutex>

#include "livo_recon/common_lib.h"
#include "livo_recon/map/map_backend.h"
#include "livo_recon/utils/map/akfmap_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/algo/robin_hood.h"

namespace livo_recon
{

// Faithful port of AKF-LIO's map (https://github.com/xpxie/AKF-LIO,
// ivox3d.h/ivox3d_node.hpp + laser_mapping.cc's ObsModel()/
// MapIncremental()) behind livo_recon's MapBackend interface -- see the
// plan at ~/.claude/plans/imperative-tumbling-karp.md for the full design
// rationale (VoxelMap's debiased plane fit is a subtractive correction
// that isn't guaranteed PSD; this backend's fusion is additive/convex,
// PSD by construction).
//
// Storage: a spatial hash of small std::vector<AkfPoint> cells. Each
// AkfPoint is a GAUSSIAN SUMMARY (mean + covariance + point count), not a
// single raw scan return -- insertion progressively FUSES new points into
// existing nearby summaries (Mahalanobis-gated), so a cell holds a bounded
// number of distinct local blobs, not an ever-growing raw point list.
//
// Two accepted deviations from a byte-literal port, both documented at
// their point of use below: (1) insertion is single-threaded (AKF-LIO's
// own insertion can touch NEIGHBOR cells via cross-cell Mahalanobis search
// and re-key a merged point into a different cell than either input
// started in -- unlike VoxelMap's per-key-independent design, this isn't
// safely parallelizable by simply sharding over distinct keys without
// real cross-cell locking, not attempted in this first port); (2)
// query-time K-NN uses a plain hash map with no LRU list threading, so
// time/capacity-based pruning is a scan rather than an early-break over an
// LRU-ordered list -- the SET of evicted cells is identical either way,
// only the scan cost differs.
class AkfMap : public MapBackend
{
public:
  AkfMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues);

  std::string loadParameters(ros::NodeHandle& pnh) override;

  void updateMap(MeasureGroup& mg) override;

  // Query-time plane derivation (AKF-LIO's ObsModel() core loop): K-NN via
  // Mahalanobis distance, then a sequential pairwise pseudo-merge gated on
  // (a) the running Gaussian's flatness (2nd eigenvalue floor) and (b) the
  // candidate's in-plane Mahalanobis-style consistency -- a candidate
  // failing either gate truncates the remaining candidate list rather than
  // being skipped individually. Every candidate that survives (not
  // truncated) is recorded for this frame's uncertainty feedback (see
  // updateMap()'s doc comment on the "across all of this frame's
  // iterations" simplification).
  bool findPlaneResidual(const WorldPointCov& pt, Residual& res,
                         bool* tier0_had_plane = nullptr) const override;

  bool isEmpty() const override;
  std::string statsString() const override;

private:
  void generateNearbyGrids();

  // Post-EKF-converged uncertainty feedback (ivox3d.h's UpdateUncertainty,
  // called from MapIncremental() before AddPoints()) -- refreshes every
  // point matched_this_period_ recorded during findPlaneResidual() calls
  // back toward opts_->init_uncertainty, blended by use_num ratio.
  // ACCEPTED SIMPLIFICATION: AKF-LIO captures matches from only the FINAL
  // converged IEKF iteration; this port has no hook into LioProc's
  // iteration boundaries, so matched_this_period_ accumulates matches
  // across ALL of a frame's iterations (cleared once consumed here, right
  // before that frame's own insertion pass) -- a superset of AKF-LIO's
  // exact behavior, not expected to matter since later iterations'
  // matches dominate (tighter gating as the state converges), but flagged
  // as a real, documented deviation rather than silently assumed harmless.
  void updateUncertaintyFeedback();

  // Single-threaded insertion for one world-frame point (see class doc
  // comment on why this isn't parallelized). Mirrors ivox3d.h's
  // AddPoints(): search the neighbor pattern for the single
  // Mahalanobis-nearest existing point gated at opts_->t_mal; merge if
  // found (re-keying if the merged mean moved cells), else insert fresh.
  void insertPoint(const V3D& p_world, const M3D& cov, double time_now,
                   robin_hood::unordered_flat_set<VoxelKey, VoxelKeyHash>& touched_this_frame);

  // Port of ivox3d.h's ErasePoints() -- see class doc comment on the
  // scan-based (non-LRU-ordered) simplification.
  void erasePoints(double cur_time);

  AkfMapOptsPtr opts_;
  StateGroupPtr state_;
  ProfilerPtr   profiler_;
  DataQueuesPtr data_queues_;

  using AkfCell = std::vector<AkfPoint>;
  // mutable: findPlaneResidual() is logically a query (const on the
  // interface, called from LioProc's per-point OMP loop) but AKF-LIO's own
  // query-time gating never mutates map POSITIONS -- only records which
  // points were touched, via matched_this_period_ below, for later
  // feedback. cells_ itself is genuinely read-only during
  // findPlaneResidual(); marked mutable only so its non-const iterators
  // can hand back non-const AkfPoint* for matched_this_period_ to store
  // (the same pointers updateUncertaintyFeedback() later mutates), without
  // a const_cast.
  mutable robin_hood::unordered_flat_map<VoxelKey, AkfCell, VoxelKeyHash> cells_;
  std::vector<VoxelKey> nearby_grids_;

  mutable std::mutex match_mutex_;
  mutable std::vector<AkfPoint*> matched_this_period_;
};

}  // namespace livo_recon
