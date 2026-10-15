#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/lio/admission_rule.h"
#include <fstream>
#include <sstream>

namespace livo_recon
{

namespace
{

int g_current_frame_idx = -1;
bool g_bootstrap_insertion = false;

// Diagnostic-only (see the point_filter_num=1 single-frame-init failure
// mode investigation): one line per voxel's FIRST plane fit, comparing raw
// PCA quality (eigenvalues, in-plane radius) and fit-uncertainty
// (plane_var_'s trace) against how many distinct frames actually
// contributed. Remove once done debugging.
// CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
// regression this fixes.
void debugLogPlaneInit(const std::string& msg)
{
  static PersistentLogStream log("plane_init.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

}  // namespace

void setCurrentFrame(int idx) { g_current_frame_idx = idx; }
void setBootstrapInsertion(bool active) { g_bootstrap_insertion = active; }

VoxelNode::VoxelNode(VoxelOptsPtr opts, VoxelStatsPtr stats, int layer, const V3D& center)
  : opts_(opts), stats_(stats), plane_(opts), layer_(layer),
    voxel_center_(center), status_(VoxelStatus::OPEN),
    node_id_(stats->next_node_id.fetch_add(1, std::memory_order_relaxed))
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
}

int VoxelNode::childIndex(const V3D& p) const
{
  return (p.x() > voxel_center_.x() ? 4 : 0)
       | (p.y() > voxel_center_.y() ? 2 : 0)
       | (p.z() > voxel_center_.z() ? 1 : 0);
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

  for (const auto& p : points_world) {
    if (p.bootstrap_observation_id < 0) continue;
    auto& st = bootstrap_observation_support_[p.bootstrap_observation_id];
    ++st.count;
    st.sum_p += p.point;
    st.sum_pp.noalias() += p.point * p.point.transpose();
  }

  if (!points_world.empty() && g_current_frame_idx != last_frame_idx_seen_) {
    last_frame_idx_seen_ = g_current_frame_idx;
    ++distinct_frames_;
  }

  const bool was_init   = plane_.isInit();
  bool       was_plane  = plane_.isPlane();

  bool now_plane;
  {
    if (opts_->binned_enabled) {
      // R90 (binned map, pca fit): returns go into bins; min_init / min_update count the unique bins touched since the last fit (R92); the fit is the pca fit of one weighted point per occupied bin
      // (VoxelPlane::update() with the weights of binned::BinSet::representatives()). The admission test reads plane_.isInit() as the pca path always did; pointsSize() is the number of occupied bins at the last
      // fit (so isFull() / points/max also count bins, and a voxel stays open until it holds points/max occupied bins).
      if (!binset_) binset_.reset(new binned::BinSet(opts_->voxel_size / static_cast<double>(1 << layer_) * opts_->binned_bin_fraction));
      long long opened = 0;
      for (const auto& pt : points_world)
        if (binset_->add(pt.point, pt.sensor_cov)) ++opened;
      update_count_ = static_cast<int>(binset_->dirtyCount());   // R92: unique bins hit since the last fit (the only count; `opened` is kept for the binned_new_bins counter)
      stats_->binned_points.fetch_add(static_cast<long long>(points_world.size()), std::memory_order_relaxed);
      stats_->binned_new_bins.fetch_add(opened, std::memory_order_relaxed);
      // History for a later subdivision, bounded by max_points.
      batch_stored_in_points_ = false;
      if (static_cast<int>(points_.size()) < opts_->max_points) {
        points_.insert(points_.end(), points_world.begin(), points_world.end());
        batch_stored_in_points_ = true;
      }
      if (!admission::fitWanted(was_init, update_count_, opts_->min_init_points, opts_->min_update_points, g_current_frame_idx == 0)) return;
      if (!was_init && g_current_frame_idx >= 1 && distinct_frames_ < opts_->min_frames_to_init) return;
      plane_.noteFrames(distinct_frames_);
      binned::Options bo;
      bo.cov_mode = opts_->binned_cov_mode == 1 ? binned::CovMode::Shrunk : binned::CovMode::Average;
      const binned::Reps reps = binset_->representatives(bo);
      std::vector<PointXYZCov> bin_reps(reps.p.size());
      for (std::size_t i = 0; i < reps.p.size(); ++i) {
        bin_reps[i].point = reps.p[i];
        bin_reps[i].sensor_cov = reps.cov[i];
        bin_reps[i].pos_cov = M3D::Zero();   // the binned map keeps the sensor covariance only (every fit uses the sensor covariance only)
      }
      plane_.update(bin_reps, static_cast<int>(binset_->numBins()), &reps.w, nullptr, &reps.w, g_current_frame_idx, layer_);
      stats_->binned_fits.fetch_add(1, std::memory_order_relaxed);
      stats_->binned_touched_at_fit.fetch_add(binset_->dirtyCount(), std::memory_order_relaxed);
      binset_->markFitted();
      update_count_ = 0;
      now_plane = plane_.isPlane();
    } else {
    update_count_ += points_world.size();
    for (const auto& pt : points_world) {
      points_.push_back(pt);
      running_moments_.add(pt.point);
    }

    // R87: the decision is the shared admission rule; the only change to the pca path is that the bootstrap frame (global frame 0) is no longer gated by min_update.
    if (!admission::fitWanted(was_init, update_count_, opts_->min_init_points, opts_->min_update_points, g_current_frame_idx == 0))
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
    plane_.noteFrames(distinct_frames_);

    plane_.update(points_, -1, nullptr, &running_moments_, nullptr, g_current_frame_idx, layer_);
    update_count_ = 0;
    now_plane = plane_.isPlane();
    }

    if (!was_init) {
      if (opts_->log_debug_en) {
        const V3D& ev = plane_.eigenValues();
        std::ostringstream dbg;
        // voxel_center_ is the SAME quantized key across separate runs on
        // the same bag/config as long as trajectories haven't yet diverged
        // (voxel keys are derived purely from world-frame point position at
        // insertion time), so runs can be joined on it post-hoc.
        PlaneVizInfo viz{};
        const bool have_viz = plane_.getVizInfo(viz);
        dbg << "mode=pca"
            << " voxel=[" << voxel_center_.transpose() << "]"
            << " frame=" << g_current_frame_idx
            << " distinct_frames=" << distinct_frames_
            << " n_points=" << plane_.pointsSize()
            << " eig0=" << ev(0) << " eig1=" << ev(1) << " eig2=" << ev(2)
            << " radius=" << plane_.radius()
            << " plane_var_trace=" << plane_.planeVar().trace()
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
  if (was_plane  && !now_plane) {
    stats_->planes.fetch_sub(1, std::memory_order_relaxed);
    voxelPlaneFitEvent(*opts_, g_current_frame_idx, kEvRetire);   // R64 diagnostics
  }

  if (!now_plane && layer_ < opts_->max_layer) {
    voxelPlaneFitEvent(*opts_, g_current_frame_idx, kEvSubdivide);   // R64 diagnostics
    stats_->transition(status_, VoxelStatus::PARENT);
    status_ = VoxelStatus::PARENT;
    if (was_plane) updates.push_back({node_id_, true, {}});
    plane_retired_ = true;
    if (opts_->binned_enabled) {
      // R65 (F-158): children receive the stored history (R90: the binned pca branch keeps the same history) (points_, bounded by max_points as in the pca
      // path). If this batch was not stored (points_ already at the bound) it is added so the newest
      // points are never lost. No refit from scratch: the parent's O(1) sums are simply retired.
      if (batch_stored_in_points_) {
        passToChildren(points_, updates);
      } else {
        std::vector<PointXYZCov> hist(points_);
        hist.insert(hist.end(), points_world.begin(), points_world.end());
        passToChildren(hist, updates);
      }
      points_.clear();
      running_moments_.reset(voxel_center_);
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
  should_lock = plane_.isFull();
  if (opts_->convergence_mode == "frame_gated")
    should_lock = should_lock && (distinct_frames_ >= opts_->min_frames_to_converge);
  else if (opts_->convergence_mode == "always_update" && now_plane)
    should_lock = false;

  if (g_bootstrap_insertion) should_lock = false;

  if (!should_lock) {
    PlaneVizInfo info;
    info.is_converged = false;
    if (now_plane && plane_.getVizInfo(info))
      updates.push_back({node_id_, false, info});
    else if (was_plane && !now_plane)
      updates.push_back({node_id_, true, {}});
    return;
  }

  points_.clear();
  binset_.reset();   // R90: the binned pca branch locks in like the pca path (no more fits)
  running_moments_.reset(voxel_center_);
  VoxelStatus new_status = now_plane ? VoxelStatus::CONVERGED : VoxelStatus::DISABLED;
  stats_->transition(status_, new_status);
  status_ = new_status;

  if (now_plane) {
    PlaneVizInfo info;
    info.is_converged = true;
    if (plane_.getVizInfo(info))
      updates.push_back({node_id_, false, info});
  } else {
    voxelPlaneFitEvent(*opts_, g_current_frame_idx, kEvDisabled);   // R64 diagnostics
    if (was_plane) updates.push_back({node_id_, true, {}});
    plane_retired_ = true;
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

  if (plane_retired_) return false;
  if (!plane_.computeResidual(pt, res, scan_id)) return false;
  res.plane_node_id = node_id_;
  res.voxel_layer = layer_;   // R83: write-only (residual dump under lio/residual_dump_voxel_info)
  return true;
}

void VoxelNode::appendPlaneSnapshots(
    const VoxelKey& root_key, std::vector<VoxelPlaneSnapshot>& out) const
{
  VoxelPlaneSnapshot row;
  row.node_id = node_id_;
  row.root_key = root_key;
  row.layer = layer_;
  row.cell_center = voxel_center_;
  const double half_extent = 0.5 * opts_->voxel_size /
      static_cast<double>(1 << layer_);
  row.cell_min = voxel_center_ - V3D::Constant(half_extent);
  row.cell_max = voxel_center_ + V3D::Constant(half_extent);
  row.status = static_cast<int>(status_);
  row.retired = plane_retired_;
  plane_.fillSnapshot(row);
  out.push_back(row);
  if (status_ == VoxelStatus::PARENT)
    for (const VoxelNode* child : leaves_)
      if (child) child->appendPlaneSnapshots(root_key, out);
}

void VoxelNode::appendBootstrapObservationSupport(
    const VoxelKey& root_key,
    std::vector<BootstrapObservationStatsRow>& out) const
{
  for (const auto& kv : bootstrap_observation_support_) {
    BootstrapObservationStatsRow row;
    row.node_id = node_id_;
    row.root_key = root_key;
    row.layer = layer_;
    row.observation_id = kv.first;
    row.point_count = kv.second.count;
    row.sum_p = kv.second.sum_p;
    row.sum_pp = kv.second.sum_pp;
    out.push_back(row);
  }
  if (status_ == VoxelStatus::PARENT)
    for (const VoxelNode* child : leaves_)
      if (child) child->appendBootstrapObservationSupport(root_key, out);
}

}  // namespace livo_recon
