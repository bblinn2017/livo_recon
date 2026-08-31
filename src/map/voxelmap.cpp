#include "livo_recon/map/voxelmap.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/algo/omp_utils.h"

namespace livo_recon
{

VoxelMap::VoxelMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues)
  : opts_(std::make_shared<VoxelOpts>())
  , stats_(std::make_shared<VoxelStats>())
  , state_(state)
  , profiler_(profiler)
  , data_queues_(data_queues)
{}

std::string VoxelMap::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<double>(pnh, "voxel_map/map/voxel_size",   opts_->voxel_size,  0.5);
  paramWarn<int>(pnh, "voxel_map/map/max_layer",       opts_->max_layer,    2);
  paramWarn<bool>(pnh, "voxel_map/plane/sensor_noise_floor_eig0", opts_->sensor_noise_floor_eig0, false);
  paramWarn<std::string>(pnh, "voxel_map/plane/convergence_mode", opts_->convergence_mode, "normal");
  paramWarn<int>(pnh, "voxel_map/plane/min_frames_to_converge", opts_->min_frames_to_converge, 5);
  paramWarn<int>(pnh, "voxel_map/plane/min_frames_to_init", opts_->min_frames_to_init, 1);

  paramWarn<double>(pnh, "voxel_map/plane/plane_threshold",           opts_->plane_threshold,           0.01);

  paramWarn<int>(pnh, "voxel_map/points/min_init",   opts_->min_init_points,  3);
  paramWarn<int>(pnh, "voxel_map/points/max",        opts_->max_points,       50);
  paramWarn<int>(pnh, "voxel_map/points/min_update", opts_->min_update_points,5);

  { double sigma_num; paramWarn<double>(pnh, "voxel_map/residual/sigma_num", sigma_num, 3.); opts_->sigma_num_squared = sigma_num * sigma_num; }
  paramWarn<double>(pnh, "voxel_map/residual/max_radius",  opts_->max_radius,          3.);
  paramWarn<bool>(pnh, "voxel_map/residual/pose_cov_in_sigma", opts_->pose_cov_in_sigma, false);
  paramWarn<bool>(pnh, "voxel_map/plane/log_debug_en", opts_->log_debug_en, false);
  paramWarn<bool>(pnh, "voxel_map/plane/log_variance_shares_en", opts_->log_variance_shares_en, false);
  paramWarn<int>(pnh, "voxel_map/search/neighborhood_size", opts_->neighborhood_size,   1);
  paramWarn<std::string>(pnh, "voxel_map/plane/plane_fit_mode", opts_->plane_fit_mode, "pca");
  paramWarn<std::string>(pnh, "voxel_map/plane/plane_fit_pose_cov_mode", opts_->plane_fit_pose_cov_mode, "combined");
  paramWarn<std::string>(pnh, "voxel_map/plane/bin_weight_mode_fit", opts_->bin_weight_mode_fit, "count");
  paramWarn<std::string>(pnh, "voxel_map/plane/bin_weight_mode_var", opts_->bin_weight_mode_var, "count");
  paramWarn<double>(pnh, "voxel_map/plane/bin_size_fraction", opts_->bin_size_fraction, 0.2);
  paramWarn<bool>(pnh, "voxel_map/plane/use_bins", opts_->use_bins, false);
  paramWarn<bool>(pnh, "voxel_map/plane/log_consistency_corr_en", opts_->log_consistency_corr_en, false);
  paramWarn<bool>(pnh, "voxel_map/plane/log_consistency_covariates_en", opts_->log_consistency_covariates_en, false);

  std::ostringstream oss;
  oss << "[params/voxel_map]"
      << "\n  map/voxel_size:                  " << opts_->voxel_size
      << "\n  map/max_layer:                   " << opts_->max_layer
      << "\n  plane/sensor_noise_floor_eig0:   " << (opts_->sensor_noise_floor_eig0 ? "true" : "false")
      << "\n  plane/convergence_mode:          " << opts_->convergence_mode
      << "\n  plane/min_frames_to_converge:    " << opts_->min_frames_to_converge
      << "\n  plane/min_frames_to_init:        " << opts_->min_frames_to_init
      << "\n  plane/plane_threshold:           " << opts_->plane_threshold
      << "\n  plane/plane_fit_mode:            " << opts_->plane_fit_mode
      << "\n  plane/plane_fit_pose_cov_mode:   " << opts_->plane_fit_pose_cov_mode
      << "\n  plane/bin_weight_mode_fit:       " << opts_->bin_weight_mode_fit
      << "\n  plane/bin_weight_mode_var:       " << opts_->bin_weight_mode_var
      << "\n  plane/bin_size_fraction:         " << opts_->bin_size_fraction
      << "\n  plane/use_bins:                  " << (opts_->use_bins ? "true" : "false")
      << "\n  plane/log_variance_shares_en:    " << (opts_->log_variance_shares_en ? "true" : "false")
      << "\n  plane/log_consistency_corr_en:   " << (opts_->log_consistency_corr_en ? "true" : "false")
      << "\n  plane/log_consistency_covariates_en: " << (opts_->log_consistency_covariates_en ? "true" : "false")
      << "\n  points/min_init:  " << opts_->min_init_points
      << "\n  points/max:       " << opts_->max_points
      << "\n  points/min_update:" << opts_->min_update_points
      << "\n  residual/sigma_num_squared:  " << opts_->sigma_num_squared
      << "\n  residual/max_radius:         " << opts_->max_radius
      << "\n  residual/pose_cov_in_sigma:  " << (opts_->pose_cov_in_sigma ? "true" : "false")
      << "\n  search/neighborhood_size:    " << opts_->neighborhood_size;
  return oss.str();
}

bool VoxelMap::isEmpty() const { return voxel_map_.empty(); }

VoxelKey VoxelMap::worldToKey(const V3D& p_world) const
{
  return worldToKeyFn(p_world, opts_->voxel_size);
}

void VoxelMap::updateMap(MeasureGroup& mg) {
  TimedScope ts_total(profiler_, "voxelmap");
  setCurrentFrame(frame_idx_++);
  // The single point set (point_filter_num + ds_leaf_size, see measures.h's
  // docs on MeasureGroup::points) used for both voxel-map insertion and
  // LIO's own residual matching -- no separate full-resolution stream.
  const std::vector<PointXYZCov>& map_pts = mg.points;
  {
    TimedScope ts(profiler_, "voxelmap/bucket_build");
    const int np = static_cast<int>(map_pts.size());
    pts_world_.resize(np);

    {
      TimedScope ts(profiler_, "voxelmap/bucket_build/clear");
      for (const auto& key : active_keys_) {
        auto it = bucket_map_.find(key);
        it->second.world.clear();
        it->second.body.clear();
      }
      active_keys_.clear();
    }

    {
      TimedScope ts(profiler_, "voxelmap/bucket_build/transform");
      const int threads = cappedOmpThreads();
      thread_keys_.resize(threads);
      for (auto& v : thread_keys_) v.clear();

      #pragma omp parallel for schedule(static) num_threads(threads)
      for (int i = 0; i < np; ++i) {
        pts_world_[i] = state_->toWorld(map_pts[i]);
        // sensor_cov (see PointXYZCov's docs) is always sensor-only, never
        // mutated downstream -- a consumer needing the isotropic proxy
        // derives it inline (sensor_cov.trace()/3.0), no separate stored
        // scalar. pos_cov is captured directly here too (NOT derived by
        // subtracting from any combined value) -- a consumer that needs
        // the combined sensor+pose covariance (e.g. the PCA plane fit's
        // own residual-weighting propagation) adds `sensor_cov + pos_cov`
        // explicitly at the point of use. Matches gate()'s own
        // WorldPointCov, which already kept these separate.
        pts_world_[i].pos_cov = state_->poseCovAt(map_pts[i].point);
        thread_keys_[omp_get_thread_num()].emplace_back(
          worldToKey(pts_world_[i].point), i);
      }
    }

    {
      TimedScope ts(profiler_, "voxelmap/bucket_build/merge");
      for (const auto& local : thread_keys_)
        for (auto [key, i] : local) {
          auto& entry = bucket_map_[key];
          if (entry.world.empty()) active_keys_.push_back(key);
          entry.world.push_back(pts_world_[i]);
          entry.body.push_back(map_pts[i].point);
        }
    }
  }

  {
    // No insertions happen here — pointers into bucket_map_ are stable
    TimedScope ts(profiler_, "voxelmap/ensure_nodes");
    bucket_flat_.clear();
    bucket_flat_.reserve(active_keys_.size());
    for (const auto& key : active_keys_) {
      auto it = bucket_map_.find(key);
      ensureNode(key);
      bucket_flat_.emplace_back(&it->first, &it->second);
    }
  }

  last_n_map_pts_ = static_cast<int>(map_pts.size());
  last_n_active_voxels_ = static_cast<int>(active_keys_.size());

  const int n = static_cast<int>(bucket_flat_.size());

  {
    TimedScope ts(profiler_, "voxelmap/insert");
    const int threads = cappedOmpThreads();
    thread_plane_updates_.resize(threads);
    for (auto& v : thread_plane_updates_) v.clear();

    #pragma omp parallel for schedule(guided) num_threads(threads)
    for (int i = 0; i < n; ++i) {
      const auto& [key, entry] = bucket_flat_[i];
      auto it = voxel_map_.find(*key);
      assert(it != voxel_map_.end() && it->second != nullptr);
      it->second->insertPoints(entry->world, thread_plane_updates_[omp_get_thread_num()]);
    }

    for (const auto& thread_updates : thread_plane_updates_) {
      for (const auto& u : thread_updates) {
        viz_dirty_[u.node_id] = u.deleted;
        if (u.deleted)
          viz_cache_.erase(u.node_id);
        else
          viz_cache_[u.node_id] = u.info;
      }
    }
  }
}

// ─────────────────────────────────────────────
//  VoxelPlane
// ─────────────────────────────────────────────

std::string VoxelMap::statsString() const
{
  std::ostringstream oss;
  oss << "[VoxelMap] total=" << stats_->total()
      << "  open="      << stats_->open
      << "  converged=" << stats_->converged
      << "  disabled="  << stats_->disabled
      << "  planes="    << stats_->planes;
  return oss.str();
}

void VoxelMap::ensureNode(const VoxelKey& key)
{
  auto [it, inserted] = voxel_map_.try_emplace(key, nullptr);
  if (inserted) {
    const double vs = opts_->voxel_size;
    const V3D center((key.x + 0.5) * vs, (key.y + 0.5) * vs, (key.z + 0.5) * vs);
    it->second = new VoxelNode(opts_, stats_, 0, center);
  }
}

bool VoxelMap::findPlaneResidualDirectional(const WorldPointCov& pt, const VoxelKey& base, Residual &res,
                                            VoxelKey* tried_key, int scan_id) const
{
  // FAST-LIVO2-style single-neighbor step: the primary voxel missed, so
  // step to the one neighbor the point actually leans toward -- per axis,
  // compare the point's world position against that axis's voxel center
  // +/- a quarter voxel width, and step +1/-1 only on the axis(es) where
  // the point sits past that boundary. Cheap (at most one extra hash
  // lookup), but only ever tries that single candidate -- see
  // findPlaneResidual() for the tiered fallback this feeds into.
  const double vs = opts_->voxel_size;
  const double quarter = vs / 4.0;
  const V3D center((base.x + 0.5) * vs, (base.y + 0.5) * vs, (base.z + 0.5) * vs);

  VoxelKey near_key = base;
  bool has_neighbor = false;
  for (int axis = 0; axis < 3; ++axis) {
    const double d = pt.point[axis] - center[axis];
    int* comp = (axis == 0) ? &near_key.x : (axis == 1) ? &near_key.y : &near_key.z;
    if (d > quarter)       { *comp += 1; has_neighbor = true; }
    else if (d < -quarter) { *comp -= 1; has_neighbor = true; }
  }
  if (tried_key) *tried_key = near_key;
  if (!has_neighbor) return false;

  auto it_near = voxel_map_.find(near_key);
  if (it_near == voxel_map_.end() || !it_near->second) return false;
  if (!it_near->second->findPlaneResidual(pt, res, scan_id)) return false;
  res.match_tier = 1;
  return true;
}

bool VoxelMap::findPlaneResidualNeighborhood(const WorldPointCov& pt, const VoxelKey& base, Residual &res,
                                             const VoxelKey* exclude, int scan_id) const
{
  // Exhaustive box search (original livo_recon approach): score every
  // candidate voxel within neighborhood_size in each direction and keep
  // the best-scoring match, rather than committing to a single candidate.
  // `exclude` (if non-null) skips one key already tried by an earlier tier
  // (the primary voxel and/or the directional neighbor), to avoid a
  // redundant findPlaneResidual() call against it.
  const int n = opts_->neighborhood_size;
  double best_score = -1.0;
  bool found = false;

  for (int dx = -n; dx <= n; ++dx)
  for (int dy = -n; dy <= n; ++dy)
  for (int dz = -n; dz <= n; ++dz)
  {
    VoxelKey key = base;
    key.x += dx; key.y += dy; key.z += dz;

    if (key == base) continue;
    if (exclude && key == *exclude) continue;

    auto it = voxel_map_.find(key);
    if (it == voxel_map_.end() || !it->second) continue;

    Residual cand;
    if (!it->second->findPlaneResidual(pt, cand, scan_id)) continue;

    const double score = (1.0 / std::sqrt(cand.sigma_squared)) *
                         std::exp(-0.5 * cand.r * cand.r / cand.sigma_squared);
    if (score > best_score) {
      best_score = score;
      res = cand;
      res.match_tier = 2;
      found = true;
    }
  }

  return found;
}

bool VoxelMap::hasConvergedNeighbor(const V3D& p_world) const
{
  const VoxelKey base = worldToKey(p_world);
  const int n = opts_->neighborhood_size;
  for (int dx = -n; dx <= n; ++dx)
  for (int dy = -n; dy <= n; ++dy)
  for (int dz = -n; dz <= n; ++dz) {
    VoxelKey key = base;
    key.x += dx; key.y += dy; key.z += dz;
    auto it = voxel_map_.find(key);
    if (it != voxel_map_.end() && it->second && it->second->hasConvergedPlane())
      return true;
  }
  return false;
}

bool VoxelMap::findPlaneResidual(const WorldPointCov& pt, Residual &res, bool* tier0_had_plane) const
{
  const VoxelKey base = worldToKey(pt.point);

  auto it = voxel_map_.find(base);
  if (it != voxel_map_.end() && it->second && it->second->findPlaneResidual(pt, res, allow_consistency_log_ ? frame_idx_ : -1))
    return true;

  if (tier0_had_plane)
    *tier0_had_plane = (it != voxel_map_.end() && it->second && it->second->hasConvergedPlane());

  // Tiered fallback: (1) primary voxel [above] -- (2) the single directional
  // neighbor the point actually leans toward (cheap, catches the common
  // case) -- (3) only if that also misses, the full neighborhood_size box
  // search (expensive, but a real safety net -- see the residual-count
  // collapse this was reinstated to fix: the directional-only fallback let
  // matching thin out badly during dynamic motion, since a wrong single
  // guess had no second chance). frame_idx_ (this frame's scan_id, for
  // T0-D's corr.csv) is threaded through every tier the same way.
  VoxelKey directional_key = base;
  if (findPlaneResidualDirectional(pt, base, res, &directional_key, allow_consistency_log_ ? frame_idx_ : -1))
    return true;

  return findPlaneResidualNeighborhood(pt, base, res, &directional_key, allow_consistency_log_ ? frame_idx_ : -1);
}

// ── Visualization list helpers ────────────────────────────────────────────────

void VoxelMap::eraseDisc(int32_t id)
{
  auto it = disc_seg_idx_.find(id);
  if (it == disc_seg_idx_.end()) return;

  const uint32_t seg    = it->second;
  const uint32_t stride = DISC_SEGS * 3;
  const uint32_t p0     = seg * stride;
  auto& pts  = disc_list_.points;
  auto& cols = disc_list_.colors;
  const uint32_t last_seg = static_cast<uint32_t>(disc_seg_owner_.size()) - 1;

  if (seg != last_seg) {
    const uint32_t tail = last_seg * stride;
    std::copy(pts.begin()  + tail, pts.begin()  + tail + stride, pts.begin()  + p0);
    std::copy(cols.begin() + tail, cols.begin() + tail + stride, cols.begin() + p0);
    disc_seg_owner_[seg] = disc_seg_owner_[last_seg];
    disc_seg_idx_[disc_seg_owner_[seg]] = seg;
  }

  pts.resize(pts.size() - stride);
  cols.resize(cols.size() - stride);
  disc_seg_owner_.pop_back();
  disc_seg_idx_.erase(it);
}

void VoxelMap::eraseDot(int32_t id)
{
  auto it = dot_pt_idx_.find(id);
  if (it == dot_pt_idx_.end()) return;

  const uint32_t idx  = it->second;
  const uint32_t last = static_cast<uint32_t>(dot_pt_owner_.size()) - 1;
  auto& pts  = dot_list_.points;
  auto& cols = dot_list_.colors;

  if (idx != last) {
    pts[idx]  = pts[last];
    cols[idx] = cols[last];
    dot_pt_owner_[idx] = dot_pt_owner_[last];
    dot_pt_idx_[dot_pt_owner_[idx]] = idx;
  }

  pts.pop_back();
  cols.pop_back();
  dot_pt_owner_.pop_back();
  dot_pt_idx_.erase(it);
}

void VoxelMap::appendDisc(int32_t id, const PlaneVizInfo& p)
{
  // colour by world-space azimuth (HSV: S=0.85, V=0.9)
  const double hue = std::fmod(std::atan2(p.center.y(), p.center.x()) / (2.0 * M_PI) + 1.0, 1.0);
  const double h6  = hue * 6.0;
  const int    hi  = static_cast<int>(h6) % 6;
  const double f   = h6 - std::floor(h6);
  const double sv = 0.9, s = 0.85;
  const double pv = sv*(1-s), qv = sv*(1-s*f), tv = sv*(1-s*(1-f));
  float r, g, b;
  switch (hi) {
    case 0: r=sv; g=tv; b=pv; break;
    case 1: r=qv; g=sv; b=pv; break;
    case 2: r=pv; g=sv; b=tv; break;
    case 3: r=pv; g=qv; b=sv; break;
    case 4: r=tv; g=pv; b=sv; break;
    default:r=sv; g=pv; b=qv; break;
  }

  std_msgs::ColorRGBA col;
  col.r = r; col.g = g; col.b = b; col.a = 0.7f;

  // Build orthonormal basis in the plane
  const V3D& n = p.normal;
  const V3D  t = (std::abs(n.z()) < 0.9) ? V3D(0, 0, 1) : V3D(1, 0, 0);
  const V3D  xa = n.cross(t).normalized();
  const V3D  ya = n.cross(xa);

  // Square with side = 2*radius, built from 2 triangles
  const V3D v0 = p.center + p.radius * ( xa + ya);
  const V3D v1 = p.center + p.radius * (-xa + ya);
  const V3D v2 = p.center + p.radius * (-xa - ya);
  const V3D v3 = p.center + p.radius * ( xa - ya);

  const auto toPoint = [](const V3D& v) {
    geometry_msgs::Point pt;
    pt.x = v.x(); pt.y = v.y(); pt.z = v.z();
    return pt;
  };

  disc_list_.points.push_back(toPoint(v0));
  disc_list_.points.push_back(toPoint(v1));
  disc_list_.points.push_back(toPoint(v2));
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);
  disc_list_.points.push_back(toPoint(v0));
  disc_list_.points.push_back(toPoint(v2));
  disc_list_.points.push_back(toPoint(v3));
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);

  disc_seg_idx_[id] = static_cast<uint32_t>(disc_seg_owner_.size());
  disc_seg_owner_.push_back(id);
}

void VoxelMap::appendDot(int32_t id, const PlaneVizInfo& p)
{
  geometry_msgs::Point pt;
  pt.x = p.center.x(); pt.y = p.center.y(); pt.z = p.center.z();
  dot_list_.points.push_back(pt);

  std_msgs::ColorRGBA col;
  col.r = 1.f; col.g = 1.f; col.b = 1.f; col.a = 1.f;
  dot_list_.colors.push_back(col);

  dot_pt_idx_[id] = static_cast<uint32_t>(dot_pt_owner_.size());
  dot_pt_owner_.push_back(id);
}

visualization_msgs::MarkerArray VoxelMap::buildVizMarkers(const std::string& frame_id,
                                                          const ros::Time& stamp)
{
  TimedScope ts(profiler_, "voxelmap/viz_markers");

  // Lazy-initialise the two list markers
  if (disc_list_.type == 0) {
    disc_list_.header.frame_id = frame_id;
    disc_list_.ns              = "voxel_planes";
    disc_list_.id              = 0;
    disc_list_.type            = visualization_msgs::Marker::TRIANGLE_LIST;
    disc_list_.action          = visualization_msgs::Marker::ADD;
    disc_list_.scale.x         = 1.0;
    disc_list_.scale.y         = 1.0;
    disc_list_.scale.z         = 1.0;
    disc_list_.pose.orientation.w = 1.0;
    disc_list_.color.a         = 1.0f;  // RViz gates TRIANGLE_LIST rendering on color.a > 0

    dot_list_.header.frame_id  = frame_id;
    dot_list_.ns               = "voxel_planes_converged";
    dot_list_.id               = 0;
    dot_list_.type             = visualization_msgs::Marker::SPHERE_LIST;
    dot_list_.action           = visualization_msgs::Marker::ADD;
    dot_list_.scale.x          = 0.04;
    dot_list_.scale.y          = 0.04;
    dot_list_.scale.z          = 0.04;
    dot_list_.pose.orientation.w = 1.0;
  }

  disc_list_.header.stamp = stamp;
  dot_list_.header.stamp  = stamp;

  for (const auto& [id, deleted] : viz_dirty_) {
    eraseDisc(id);
    eraseDot(id);
    if (!deleted) {
      const PlaneVizInfo& p = viz_cache_.at(id);
      appendDisc(id, p);
      if (p.is_converged) appendDot(id, p);
    }
  }

  viz_dirty_.clear();

  visualization_msgs::MarkerArray arr;
  arr.markers = {disc_list_, dot_list_};
  return arr;
}

}
