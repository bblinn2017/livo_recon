#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include <fstream>
#include <sstream>

namespace livo_recon
{

namespace
{

int g_current_frame_idx = -1;

// Diagnostic-only (see the point_filter_num=1 single-frame-init failure
// mode investigation): one line per voxel's FIRST plane fit, comparing raw
// PCA quality (eigenvalues, in-plane radius) and fit-uncertainty
// (plane_var_'s trace) against how many distinct frames actually
// contributed. Remove once done debugging.
void debugLogPlaneInit(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("plane_init.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

}  // namespace

void setCurrentFrame(int idx) { g_current_frame_idx = idx; }

VoxelNode::VoxelNode(VoxelOptsPtr opts, VoxelStatsPtr stats, int layer, const V3D& center)
  : opts_(opts), stats_(stats), plane_ptr_(new VoxelPlane(opts)), layer_(layer),
    voxel_center_(center), status_(VoxelStatus::OPEN),
    node_id_(stats->next_node_id.fetch_add(1, std::memory_order_relaxed)),
    density_weight_leaf_(opts->bin_size_fraction * opts->voxel_size / (1 << layer))
{
  for (int i = 0; i < 8; i++)
    leaves_[i] = nullptr;
  running_moments_.reset(voxel_center_);
  stats_->increment(VoxelStatus::OPEN);
}

VoxelNode::~VoxelNode()
{
  for (int i = 0; i < 8; i++)
    if (leaves_[i])
      delete leaves_[i];
  if (plane_ptr_)
    delete plane_ptr_;
}

int VoxelNode::childIndex(const V3D& p) const
{
  return (p.x() > voxel_center_.x() ? 4 : 0)
       | (p.y() > voxel_center_.y() ? 2 : 0)
       | (p.z() > voxel_center_.z() ? 1 : 0);
}

void VoxelNode::buildBinReps(std::vector<PointXYZCov>& reps, std::vector<double>& fit_weights,
                              std::vector<double>& var_weights) const
{
  reps.reserve(bins_.size());
  fit_weights.reserve(bins_.size());
  var_weights.reserve(bins_.size());
  for (const auto& kv : bins_) {
    const auto& bin = kv.second;
    const double inv_count = 1.0 / bin.count;
    PointXYZCov rep;
    rep.point       = bin.sum_point * inv_count;
    rep.sensor_cov  = bin.sum_sensor_cov * inv_count;
    rep.pos_cov     = bin.sum_pos_cov * inv_count;
    reps.push_back(rep);
    fit_weights.push_back(opts_->bin_weight_mode_fit == "uniform" ? 1.0 : static_cast<double>(bin.count));
    var_weights.push_back(opts_->bin_weight_mode_var == "uniform" ? 1.0 : static_cast<double>(bin.count));
  }
}

void VoxelNode::passToChildren(const std::vector<PointXYZCov>& points_world,
                               std::vector<PlaneUpdate>& updates)
{
  const double offset = opts_->voxel_size / (4.0 * (1 << layer_));

  std::vector<PointXYZCov> child_pts[8];
  for (const auto& pt : points_world)
    child_pts[childIndex(pt.point)].push_back(pt);

  for (int i = 0; i < 8; i++) {
    if (child_pts[i].empty()) continue;
    if (!leaves_[i]) {
      V3D child_center(
        voxel_center_.x() + (i & 4 ? offset : -offset),
        voxel_center_.y() + (i & 2 ? offset : -offset),
        voxel_center_.z() + (i & 1 ? offset : -offset));
      leaves_[i] = new VoxelNode(opts_, stats_, layer_ + 1, child_center);
    }
    leaves_[i]->insertPoints(child_pts[i], updates);
  }
}

void VoxelNode::insertPoints(const std::vector<PointXYZCov>& points_world,
                             std::vector<PlaneUpdate>& updates)
{
  if (status_ == VoxelStatus::CONVERGED || status_ == VoxelStatus::DISABLED) return;

  if (status_ == VoxelStatus::PARENT) {
    passToChildren(points_world, updates);
    return;
  }

  if (!points_world.empty() && g_current_frame_idx != last_frame_idx_seen_) {
    last_frame_idx_seen_ = g_current_frame_idx;
    ++distinct_frames_;
  }

  const bool was_init   = plane_ptr_->isInit();
  bool       was_plane  = plane_ptr_->isPlane();
  const bool debiased   = opts_->plane_fit_mode == "debiased";

  bool now_plane;
  {
    if (debiased) {
      // O(1) forever -- no min_update_points batching threshold needed
      // (see plane_fit_mode's docs), but the FIRST fit still needs real
      // view diversity, same risk/reasoning as the pca path below (a fit
      // off one frame's correlated-noise points, see
      // sensor_noise_floor_eig0's docs). Accumulate unconditionally
      // either way (cheap) so the eventual first fit already reflects
      // every point seen so far, not just the frame that finally
      // satisfies min_frames_to_init.
      // g_current_frame_idx == 0 is the calibration/bootstrap window (many
      // scans from a STATIONARY sensor, same vantage point repeated -- see
      // VoxelPlane::addPoints()'s trust_sensor_noise docs); don't trust
      // those points' sensor_cov for the debiasing correction.
      plane_ptr_->addPoints(points_world, -1, distinct_frames_, g_current_frame_idx >= 1, g_current_frame_idx);
      // History (135-153): see docs/livo_recon_changelog.md#src-lio-voxelnode.cpp-135
      //
      // n_planes accounting fix.  addPoints() (via refitDebiased()) can
      // flip is_plane_ right above, but the two early returns just below
      // can exit before the shared was_plane/now_plane bookkeeping at the
      // bottom of this function ever runs -- so a voxel that becomes a
      // plane during the bootstrap window (before isInit()/
      // min_frames_to_init are satisfied) never got its stats_->planes
      // increment counted, while its eventual LOSS of plane status still
      // fired the decrement later, with no matching increment behind it.
      // Confirmed this was the entire cause of stats_->planes (frame_
      // stats.txt's n_planes) reading deeply negative on every debiased-
      // arm DX-1R cell (down to -49204) -- residual generation itself
      // gates on is_plane_ directly, not this counter, so this was a
      // diagnostic-only bug, not a correctness one. Fixed by recording
      // the transition immediately, using was_plane's value AT THIS
      // POINT (updated in place) so the shared bookkeeping below sees no
      // further transition to double-count.
      {
        const bool now_plane_immediate = plane_ptr_->isPlane();
        if (!was_plane && now_plane_immediate)  stats_->planes.fetch_add(1, std::memory_order_relaxed);
        if (was_plane  && !now_plane_immediate) stats_->planes.fetch_sub(1, std::memory_order_relaxed);
        was_plane = now_plane_immediate;
      }
      if (!plane_ptr_->isInit()) return;
      if (!was_init && g_current_frame_idx >= 1 &&
          distinct_frames_ < opts_->min_frames_to_init)
      {
        return;
      }
      now_plane = plane_ptr_->isPlane();
    } else {
    const bool use_bins = useBins();

    update_count_ += points_world.size();
    if (use_bins) {
      for (const auto& pt : points_world) {
        auto& bin = bins_[worldToKeyFn(pt.point, density_weight_leaf_)];
        ++bin.count;
        bin.sum_point += pt.point;
        bin.sum_sensor_cov += pt.sensor_cov;
        bin.sum_pos_cov    += pt.pos_cov;
      }
      total_count_ += static_cast<int>(points_world.size());
    } else {
      for (const auto& pt : points_world) {
        points_.push_back(pt);
        running_moments_.add(pt.point);
      }
    }

    if ((was_init && update_count_ < opts_->min_update_points) ||
        (!was_init && update_count_ < opts_->min_init_points))
    {
      return;
    }

    // Require real view diversity before this voxel's FIRST plane fit, not
    // just point count (see min_frames_to_init's docs) -- skipped for the
    // calibration/bootstrap frame (global frame 0) so LIO still gets usable
    // planes immediately after calibration.
    if (!was_init && g_current_frame_idx >= 1 &&
        distinct_frames_ < opts_->min_frames_to_init)
    {
      return;
    }

    // The information model's design-effect correction needs to know how
    // many DISTINCT frames these points came from -- returns from one sweep
    // share that instant's pose error and are not independent observations.
    // The debiased path receives this through addPoints(); the pca path has
    // no other channel for it.
    plane_ptr_->noteFrames(distinct_frames_);

    if (use_bins) {
      std::vector<PointXYZCov> bin_reps;
      std::vector<double> fit_weights, var_weights;
      buildBinReps(bin_reps, fit_weights, var_weights);
      plane_ptr_->update(bin_reps, total_count_, &fit_weights, nullptr, &var_weights, g_current_frame_idx);
    } else {
      plane_ptr_->update(points_, -1, nullptr, &running_moments_, nullptr, g_current_frame_idx);
    }
    update_count_ = 0;
    now_plane = plane_ptr_->isPlane();
    }

    if (!was_init) {
      if (opts_->log_debug_en) {
        const V3D& ev = plane_ptr_->eigenValues();
        std::ostringstream dbg;
        // voxel_center_ is the SAME quantized key across separate runs on
        // the same bag/config as long as trajectories haven't yet diverged
        // (voxel keys are derived purely from world-frame point position at
        // insertion time) -- logging it lets a pca-mode run and a
        // debiased-mode run be joined on voxel_center_ post-hoc to compare
        // the SAME voxel's first fit directly, isolating fit differences
        // from any already-accumulated trajectory drift. See the
        // plane_fit_mode divergence investigation
        // (docs/debiased_voxel_plane_fit_2026aug24.md).
        PlaneVizInfo viz{};
        const bool have_viz = plane_ptr_->getVizInfo(viz);
        dbg << "mode=" << opts_->plane_fit_mode
            << " voxel=[" << voxel_center_.transpose() << "]"
            << " frame=" << g_current_frame_idx
            << " distinct_frames=" << distinct_frames_
            << " n_points=" << plane_ptr_->pointsSize()
            << " eig0=" << ev(0) << " eig1=" << ev(1) << " eig2=" << ev(2)
            << " radius=" << plane_ptr_->radius()
            << " plane_var_trace=" << plane_ptr_->planeVar().trace()
            << " is_plane=" << (now_plane ? 1 : 0);
        if (have_viz)
          dbg << " normal=[" << viz.normal.transpose() << "]"
              << " center=[" << viz.center.transpose() << "]";
        debugLogPlaneInit(dbg.str());
      }

      stats_->init_frames_sum.fetch_add(distinct_frames_, std::memory_order_relaxed);
      stats_->init_count.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats_->update_frames_sum.fetch_add(distinct_frames_ - last_update_distinct_frames_, std::memory_order_relaxed);
      stats_->update_count.fetch_add(1, std::memory_order_relaxed);
    }
    last_update_distinct_frames_ = distinct_frames_;
  }

  if (!was_plane && now_plane)  stats_->planes.fetch_add(1, std::memory_order_relaxed);
  if (was_plane  && !now_plane) stats_->planes.fetch_sub(1, std::memory_order_relaxed);

  if (!now_plane && layer_ < opts_->max_layer) {
    stats_->transition(status_, VoxelStatus::PARENT);
    status_ = VoxelStatus::PARENT;
    if (was_plane) updates.push_back({node_id_, true, {}});
    delete plane_ptr_;
    plane_ptr_ = nullptr;
    if (debiased) {
      // No raw-point history to hand off (VoxelPlane::addPoints() keeps
      // only O(1) accumulators, by design) -- fall back to just this
      // batch, same accepted-approximation spirit as the bins_ path below
      // for this already-rare non-planar/give-up case.
      passToChildren(points_world, updates);
    } else if (useBins()) {
      // points_ stays empty whenever useBins() is true (see insertPoints()
      // above) -- children need bins_'s current contents instead, collapsed to one
      // representative point per bin (loses each bin's own internal
      // weighting once handed to a child, an accepted approximation for
      // this rare non-planar/give-up path -- see PointBin's docs).
      std::vector<PointXYZCov> bin_reps;
      std::vector<double> fit_weights_unused, var_weights_unused;
      buildBinReps(bin_reps, fit_weights_unused, var_weights_unused);
      passToChildren(bin_reps, updates);
      bins_.clear();
      total_count_ = 0;
    } else {
      passToChildren(points_, updates);
      points_.clear();
      running_moments_.reset(voxel_center_);
    }
    return;
  }

  // "normal": lock in as soon as isFull() (points_size_ >= max_points).
  // "frame_gated": also require enough distinct frames' worth of view
  // diversity before locking in. "always_update": a valid plane never
  // locks in (points_ keeps growing, update() keeps refitting from
  // scratch every min_update_points new points) -- the non-planar
  // dead-end path above (subdivide, or fall through here to DISABLED at
  // max_layer) is unaffected either way.
  bool should_lock;
  if (debiased) {
    // Never freeze a GOOD (now_plane) converged plane -- no O(N) cost to
    // keep refitting, so no reason to ever stop incorporating new points
    // (see plane_fit_mode's docs). max_points/convergence_mode/
    // min_frames_to_converge are pca-mode-only under this setting.
    // STILL allow locking a genuinely non-planar voxel to DISABLED once
    // it's exhausted subdivision (layer_ >= max_layer, so the earlier
    // PARENT/subdivide branch above no longer applies) -- otherwise such a
    // voxel refits from scratch forever with no way out. Residual
    // generation gates on is_plane_, not VoxelStatus, so this was a
    // wasted-CPU/stats-accounting gap, not a correctness bug, but fix it
    // anyway for parity with pca mode's give-up behavior.
    should_lock = !now_plane && (layer_ >= opts_->max_layer);
  } else {
    should_lock = plane_ptr_->isFull();
    if (opts_->convergence_mode == "frame_gated")
      should_lock = should_lock && (distinct_frames_ >= opts_->min_frames_to_converge);
    else if (opts_->convergence_mode == "always_update" && now_plane)
      should_lock = false;
  }

  if (!should_lock) {
    PlaneVizInfo info;
    info.is_converged = false;
    if (now_plane && plane_ptr_->getVizInfo(info))
      updates.push_back({node_id_, false, info});
    else if (was_plane && !now_plane)
      updates.push_back({node_id_, true, {}});
    return;
  }

  points_.clear();
  bins_.clear();
  running_moments_.reset(voxel_center_);
  total_count_ = 0;
  VoxelStatus new_status = now_plane ? VoxelStatus::CONVERGED : VoxelStatus::DISABLED;
  stats_->transition(status_, new_status);
  status_ = new_status;

  if (now_plane) {
    PlaneVizInfo info;
    info.is_converged = true;
    if (plane_ptr_->getVizInfo(info))
      updates.push_back({node_id_, false, info});
  } else {
    if (was_plane) updates.push_back({node_id_, true, {}});
    delete plane_ptr_;
    plane_ptr_ = nullptr;
  }
}

bool VoxelNode::findPlaneResidual(const WorldPointCov& pt, Residual& res, int scan_id) const
{
  if (status_ == VoxelStatus::DISABLED) return false;

  if (status_ == VoxelStatus::PARENT) {
    const VoxelNode* child = leaves_[childIndex(pt.point)];
    if (!child) return false;
    return child->findPlaneResidual(pt, res, scan_id);
  }

  if (!plane_ptr_) return false;
  return plane_ptr_->computeResidual(pt, res, scan_id);
}

}  // namespace livo_recon
