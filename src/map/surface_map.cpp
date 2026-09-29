#include "livo_recon/map/surface_map.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <visualization_msgs/Marker.h>

#include "livo_recon/diagnostics/bootstrap_diagnostic_writer.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

namespace
{
using Clock = std::chrono::steady_clock;
double msSince(const Clock::time_point& t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Deterministic colour from a surface id (golden-ratio hue walk).
void hueToRgb(std::uint64_t id, float& r, float& g, float& b)
{
  const double h = std::fmod(static_cast<double>(id) * 0.61803398875, 1.0) * 6.0;
  const double s = 0.75, v = 0.95;
  const int i = static_cast<int>(std::floor(h));
  const double f = h - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  double rr = v, gg = t, bb = p;
  switch (i % 6) {
    case 0: rr = v; gg = t; bb = p; break;
    case 1: rr = q; gg = v; bb = p; break;
    case 2: rr = p; gg = v; bb = t; break;
    case 3: rr = p; gg = q; bb = v; break;
    case 4: rr = t; gg = p; bb = v; break;
    default: rr = v; gg = p; bb = q; break;
  }
  r = static_cast<float>(rr); g = static_cast<float>(gg); b = static_cast<float>(bb);
}
}  // namespace

SurfaceMap::SurfaceMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues)
  : layer_(new surface::SurfaceLayer(surface::Options())),
    state_(state), profiler_(profiler), data_queues_(data_queues)
{}

std::string SurfaceMap::loadParameters(ros::NodeHandle& pnh)
{
  ConfigResolver cfg(pnh);
  // --- surface-specific keys (voxel_map/surface/*) ---
  cfg.get<double>("voxel_map/surface/cell_size", opts_.core.cell_size, 0.25);
  cfg.get<int>("voxel_map/surface/min_points", opts_.core.min_points, 5);
  cfg.get<double>("voxel_map/surface/merge_angle_deg", opts_.core.merge_angle_deg, 5.0);
  cfg.get<double>("voxel_map/surface/merge_offset", opts_.core.merge_offset, 0.05);
  cfg.get<double>("voxel_map/surface/merge_gap_cells", opts_.merge_gap_cells, 3.0);
  cfg.get<bool>("voxel_map/surface/support_test", opts_.core.support_test, true);
  cfg.nested<double>(opts_.core.support_test, "voxel_map/surface/support_test=true",
                     "voxel_map/surface/support_gap_cells", opts_.support_gap_cells, 1.0);
  cfg.get<double>("voxel_map/surface/split_planarity_max", opts_.core.split_planarity_max, 0.20);
  cfg.get<bool>("voxel_map/surface/frame_log_en", opts_.frame_log_en, true);
  cfg.get<bool>("voxel_map/surface/snapshot_post_calibration", opts_.snapshot_post_calibration, false);
  cfg.get<int>("voxel_map/surface/snapshot_period_frames", opts_.snapshot_period_frames, 0);
  cfg.get<int>("voxel_map/surface/viz_period_frames", opts_.viz_period_frames, 20);
  // Enumerated with ONE allowed value each: the surface backend implements only these (rule 3.4: anything else is
  // refused, not silently replaced). plane_fit_pose_cov_mode's compiled default in the octree is "combined"; here it is
  // sensor_only by design.
  {
    std::string pose_mode, var_mode;
    cfg.mode("voxel_map/plane/plane_fit_pose_cov_mode", pose_mode, "sensor_only", {"sensor_only"});
    cfg.mode("voxel_map/plane/plane_var_mode", var_mode, "eigengap", {"eigengap"});
  }
  // --- keys shared with the octree backend (same physical meaning) ---
  cfg.get<double>("voxel_map/plane/plane_threshold", opts_.core.plane_threshold, 0.01);
  { double sigma_num = 3.0; cfg.get<double>("voxel_map/residual/sigma_num", sigma_num, 3.0);
    opts_.sigma_num_squared = sigma_num * sigma_num; }
  cfg.get<double>("voxel_map/residual/max_radius", opts_.max_radius, 3.0);
  cfg.get<bool>("voxel_map/residual/pose_cov_in_sigma", opts_.pose_cov_in_sigma, false);
  { double range_err = 0.05; cfg.get<double>("imu/sensor/range_err", range_err, 0.05);
    opts_.weight_sigma_r2 = range_err * range_err; }
  paramWarn<std::string>(pnh, "voxel_map/map/bootstrap_mode", bootstrap_mode_, "aggregate");
  if (bootstrap_mode_ != "aggregate" && bootstrap_mode_ != "sequential")
    throw std::runtime_error("voxel_map/map/bootstrap_mode must be aggregate or sequential");
  cfg.refuseUnclaimed({"voxel_map/surface"});

  if (!cfg.ok()) {
    ROS_FATAL_STREAM("\n" << cfg.report());
    throw std::runtime_error("[config] refused: voxel_map option(s) outside the surface backend's allowed set -- see "
                             "the [config/REFUSED] block above");
  }
  ROS_INFO_STREAM("\n" << cfg.report());

  if (!(opts_.core.cell_size > 1e-3)) throw std::runtime_error("voxel_map/surface/cell_size must be > 0.001");
  opts_.core.merge_gap = opts_.merge_gap_cells * opts_.core.cell_size;
  opts_.core.support_gap = opts_.support_gap_cells * opts_.core.cell_size;
  layer_.reset(new surface::SurfaceLayer(opts_.core));

  std::ostringstream oss;
  oss << "[params/voxel_map(surface)]"
      << "\n  surface/cell_size:             " << opts_.core.cell_size
      << "\n  surface/min_points:            " << opts_.core.min_points
      << "\n  surface/merge_angle_deg:       " << opts_.core.merge_angle_deg
      << "\n  surface/merge_offset:          " << opts_.core.merge_offset
      << "\n  surface/merge_gap (m):         " << opts_.core.merge_gap << " (" << opts_.merge_gap_cells << " cells)"
      << "\n  surface/support_test:          " << (opts_.core.support_test ? "true" : "false")
      << "\n  surface/support_gap (m):       " << opts_.core.support_gap << " (" << opts_.support_gap_cells << " cells)"
      << "\n  surface/split_planarity_max:   " << opts_.core.split_planarity_max
      << "\n  plane/plane_threshold:         " << opts_.core.plane_threshold
      << "\n  residual/sigma_num_squared:    " << opts_.sigma_num_squared
      << "\n  residual/max_radius:           " << opts_.max_radius
      << "\n  residual/pose_cov_in_sigma:    " << (opts_.pose_cov_in_sigma ? "true" : "false")
      << "\n  weight_floor sigma_r2:         " << opts_.weight_sigma_r2
      << "\n  map/bootstrap_mode:            " << bootstrap_mode_;
  return oss.str();
}

bool SurfaceMap::isEmpty() const { return layer_->empty(); }

std::string SurfaceMap::statsString() const
{
  const surface::Summary m = layer_->summary();
  std::ostringstream oss;
  oss << "[surface_map] cells=" << m.cells << " valid=" << m.cells_valid << " surfaces=" << m.surfaces
      << " multi=" << m.surfaces_multi << " largest=" << m.largest << " merges=" << layer_->totalMerges()
      << " splits=" << layer_->totalSplits() << "  last_n_map_pts=" << last_n_map_pts_
      << "  last_n_active_cells=" << last_n_active_voxels_;
  return oss.str();
}

// ---------------------------------------------------------------------------------------------------------------------
// Insertion
// ---------------------------------------------------------------------------------------------------------------------
void SurfaceMap::updateMap(MeasureGroup& mg) { updateMapInternal(mg.points, true, false); }

void SurfaceMap::updateMapInternal(const std::vector<PointXYZCov>& pts_body, bool advance_live_frame,
                                   bool bootstrap_phase)
{
  TimedScope ts_total(profiler_, "surfacemap");
  const int insertion_frame = frame_idx_;
  if (advance_live_frame) ++frame_idx_;
  const int np = static_cast<int>(pts_body.size());

  auto t0 = Clock::now();
  std::vector<PointXYZCov> pw(np);
  {
    TimedScope ts(profiler_, "surfacemap/transform");
    const int threads = cappedOmpThreads();
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < np; ++i) pw[i] = state_->toWorld(pts_body[i]);
  }
  const double ms_transform = msSince(t0);

  t0 = Clock::now();
  {
    TimedScope ts(profiler_, "surfacemap/insert");
    // Serial on purpose: O(1) work per point and a fixed insertion order keeps the result independent of thread count.
    for (int i = 0; i < np; ++i) layer_->addPoint(pw[i].point, pw[i].sensor_cov);
  }
  const double ms_insert = msSince(t0);

  t0 = Clock::now();
  surface::UpdateStats us;
  {
    TimedScope ts(profiler_, "surfacemap/update");
    us = layer_->update();
  }
  const double ms_update = msSince(t0);

  last_n_map_pts_ = np;
  last_n_active_voxels_ = static_cast<int>(us.dirty_cells);
  viz_pending_ = true;

  if (opts_.frame_log_en) writeFrameRow(insertion_frame, bootstrap_phase, static_cast<std::size_t>(np), us,
                                        ms_transform, ms_insert, ms_update);
  if (advance_live_frame && opts_.snapshot_period_frames > 0 && frame_idx_ % opts_.snapshot_period_frames == 0)
    writeSnapshot("f" + std::to_string(frame_idx_));
}

void SurfaceMap::bootstrap(const std::vector<std::vector<PointXYZCov>>& observations)
{
  if (!isEmpty() || frame_idx_ != 0) throw std::logic_error("SurfaceMap bootstrap requires a fresh map");

  const int frame_idx_before = frame_idx_;
  const StateGroup state_before = *state_;
  size_t flattened_count = 0;
  for (const auto& obs : observations) flattened_count += obs.size();
  std::vector<PointXYZCov> flattened;
  flattened.reserve(flattened_count);
  for (const auto& obs : observations) flattened.insert(flattened.end(), obs.begin(), obs.end());
  const uint64_t flattened_hash = hashBootstrapPopulation(flattened);

  if (bootstrap_mode_ == "aggregate") {
    // one insertion of the whole flattened population, one merge pass
    updateMapInternal(flattened, /*advance_live_frame=*/false, /*bootstrap_phase=*/true);
  } else {
    // one insertion + merge pass per stationary observation, in order
    for (const auto& obs : observations) updateMapInternal(obs, false, true);
  }

  ++frame_idx_;   // one initialisation epoch; the first real map-backed LIO query uses frame_idx_ == 1
  if (opts_.snapshot_post_calibration) writeSnapshot("post_calibration");
  writeBootstrapMapSeedDiagnostic(bootstrap_mode_, frame_idx_before, frame_idx_, state_before, *state_,
                                  flattened_hash, flattened_count);
}

// ---------------------------------------------------------------------------------------------------------------------
// Residual query
// ---------------------------------------------------------------------------------------------------------------------
bool SurfaceMap::residualAgainst(const WorldPointCov& pt, const surface::Fit& fit, const void* owner,
                                 std::uint64_t node_id, Residual& res) const
{
  // VoxelPlane::gate() with plane_gate_mode=disc, weight_floor/mode=sensor_range; VoxelPlane::computeResidual()'s
  // accepted-path fields.
  const V3D n = fit.normal;
  const double r = n.dot(pt.point) + fit.d;
  if (!std::isfinite(r)) return false;
  const V3D d_center = pt.point - fit.center;
  const double range_dis = std::sqrt(std::max(0.0, d_center.squaredNorm() - r * r));
  if (range_dis > opts_.max_radius * fit.radius) return false;

  Eigen::Matrix<double, 1, 3> J_nq;
  J_nq(0, 0) = d_center.dot(fit.y_axis);
  J_nq(0, 1) = d_center.dot(fit.x_axis);
  J_nq(0, 2) = 1.0;

  double sigma_diag_squared =
      pointPlaneGateMeasurementVariance(n, pt.sensor_cov, pt.pose_cov, opts_.pose_cov_in_sigma);
  if (!std::isfinite(sigma_diag_squared) || sigma_diag_squared <= 0.0) return false;
  if (sigma_diag_squared < 1e-6) sigma_diag_squared = 1e-6;
  double sigma_gate_diag_squared = pointPlaneGateMeasurementVariance(
      n, pt.sensor_cov, pt.pose_cov, opts_.pose_cov_in_sigma || pt.include_pose_cov_in_gate);
  if (!std::isfinite(sigma_gate_diag_squared) || sigma_gate_diag_squared <= 0.0) return false;
  if (sigma_gate_diag_squared < 1e-6) sigma_gate_diag_squared = 1e-6;

  const double plane_var_term = (J_nq * fit.plane_var * J_nq.transpose()).value();
  const double floor_term = opts_.weight_sigma_r2;
  const double sigma_gate_squared = floor_term + sigma_gate_diag_squared + plane_var_term;
  if (!std::isfinite(sigma_gate_squared) || sigma_gate_squared <= 0.0) return false;
  if (!(r * r <= opts_.sigma_num_squared * sigma_gate_squared)) return false;

  res.r = r;
  res.normal = n;
  res.floor_term = floor_term;
  res.sigma_diag_squared = sigma_diag_squared;
  res.sigma_squared = floor_term + sigma_diag_squared;
  res.plane_id = owner;
  res.plane_node_id = static_cast<int32_t>(node_id);
  res.plane_var_term = plane_var_term;
  res.plane_jacobian = J_nq.transpose();
  res.plane_covariance = fit.plane_var;
  res.vis_state = -1;
  {
    const V3D point_cross_normal = pt.body_point.cross(pt.rot_transpose * n);
    Eigen::Matrix<double, 1, 6> H_i;
    H_i << point_cross_normal.transpose(), n.transpose();
    res.s_prior_pose = (H_i * pt.prior_cov_rp * H_i.transpose()).value();
  }
  return true;
}

bool SurfaceMap::tryCell(const WorldPointCov& pt, const surface::Cell& c, Residual& res, bool& plane_exists,
                         bool& used_surface) const
{
  const void* owner = nullptr;
  const surface::Fit* f = layer_->planeFor(c, &owner);
  plane_exists = (f != nullptr);
  if (!f) return false;
  used_surface = (owner != static_cast<const void*>(&c));
  const std::uint64_t node_id = used_surface ? c.surface_id : c.id;
  return residualAgainst(pt, *f, owner, node_id, res);
}

bool SurfaceMap::findPlaneResidual(const WorldPointCov& pt, Residual& res, bool* tier0_had_plane,
                                   bool* had_converged_neighbor) const
{
  q_total_.fetch_add(1, std::memory_order_relaxed);
  const surface::CellKey base = layer_->keyOf(pt.point);
  bool base_plane = false;
  if (const surface::Cell* c = layer_->findCell(base)) {
    bool used_surface = false;
    if (tryCell(pt, *c, res, base_plane, used_surface)) {
      res.match_tier = 0;
      (used_surface ? q_hit_surface0_ : q_hit_cell0_).fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  if (tier0_had_plane) *tier0_had_plane = base_plane;
  if (had_converged_neighbor && base_plane) *had_converged_neighbor = true;

  // Single directional neighbour: step toward the side the point leans to (past a quarter cell from the centre).
  const double cs = opts_.core.cell_size;
  const double quarter = cs / 4.0;
  surface::CellKey nk = base;
  bool has_neighbor = false;
  const double centre[3] = {(base.x + 0.5) * cs, (base.y + 0.5) * cs, (base.z + 0.5) * cs};
  int* comp[3] = {&nk.x, &nk.y, &nk.z};
  for (int axis = 0; axis < 3; ++axis) {
    const double d = pt.point[axis] - centre[axis];
    if (d > quarter) { *comp[axis] += 1; has_neighbor = true; }
    else if (d < -quarter) { *comp[axis] -= 1; has_neighbor = true; }
  }
  if (has_neighbor) {
    if (const surface::Cell* nc = layer_->findCell(nk)) {
      bool plane_exists = false, used_surface = false;
      const bool hit = tryCell(pt, *nc, res, plane_exists, used_surface);
      if (had_converged_neighbor && plane_exists) *had_converged_neighbor = true;
      if (hit) {
        res.match_tier = 1;
        (used_surface ? q_hit_surface1_ : q_hit_cell1_).fetch_add(1, std::memory_order_relaxed);
        return true;
      }
    }
  }
  q_miss_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool SurfaceMap::hasConvergedNeighbor(const V3D& p_world) const
{
  const surface::CellKey base = layer_->keyOf(p_world);
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = -1; dz <= 1; ++dz) {
        surface::CellKey k;
        k.x = base.x + dx; k.y = base.y + dy; k.z = base.z + dz;
        const surface::Cell* c = layer_->findCell(k);
        if (c && c->fit.valid) return true;
      }
  return false;
}

// ---------------------------------------------------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------------------------------------------------
void SurfaceMap::writeFrameRow(int frame, bool bootstrap_phase, std::size_t n_points, const surface::UpdateStats& us,
                               double ms_transform, double ms_insert, double ms_update)
{
  if (!frame_log_open_) {
    frame_log_.open(debugLogPath("surface_map_frames.csv"));
    frame_log_ << "frame,phase,points,dirty_cells,dirty_valid,cells,cells_valid,surfaces,surfaces_multi,largest,"
                  "patches_in_multi,points_valid_frac,merges,splits,merges_total,splits_total,pair_tests,gate_rejects,"
                  "support_rejects,combined_rejects,denom_rejected,ms_transform,ms_insert,ms_update,"
                  "resid_queries,resid_hit_surface0,resid_hit_cell0,resid_hit_surface1,resid_hit_cell1,resid_miss\n";
    frame_log_open_ = true;
  }
  const surface::Summary m = layer_->summary();
  frame_log_ << frame << ',' << (bootstrap_phase ? "bootstrap" : "live") << ',' << n_points << ',' << us.dirty_cells
             << ',' << us.cells_valid_dirty << ',' << m.cells << ',' << m.cells_valid << ',' << m.surfaces << ','
             << m.surfaces_multi << ',' << m.largest << ',' << m.patches_in_multi << ','
             << (m.points_total > 0 ? m.points_valid / m.points_total : 0.0) << ',' << us.merges << ',' << us.splits
             << ',' << layer_->totalMerges() << ',' << layer_->totalSplits() << ',' << us.pair_tests << ','
             << us.gate_rejects << ',' << us.support_rejects << ',' << us.combined_rejects << ',' << us.denom_rejected
             << ',' << ms_transform << ',' << ms_insert << ',' << ms_update << ',' << q_total_.exchange(0) << ','
             << q_hit_surface0_.exchange(0) << ',' << q_hit_cell0_.exchange(0) << ',' << q_hit_surface1_.exchange(0)
             << ',' << q_hit_cell1_.exchange(0) << ',' << q_miss_.exchange(0) << '\n';
  frame_log_.flush();
}

void SurfaceMap::writeSnapshot(const std::string& tag) const
{
  // Same column layout as the stationary harness's patches / surfaces CSVs so the existing analysis tools read them.
  // Coordinates are WORLD frame (the harness dumps are body frame; rotate before comparing).
  std::ofstream pf(debugLogPath("surface_map_patches_" + tag + ".csv"));
  pf << std::setprecision(17);
  pf << "checkpoint,patch_id,surface_id,n_children,child_ids,point_count,center_x,center_y,center_z,normal_x,normal_y,"
        "normal_z,d,eig0,eig1,eig2,planarity,bb_min_x,bb_min_y,bb_min_z,bb_max_x,bb_max_y,bb_max_z,rejected_points,"
        "reservoir_pending,rank2\n";
  layer_->forEachCell([&](const surface::Cell& c) {
    if (!c.fit.valid) return;
    const surface::Surface* s = layer_->findSurface(c.surface_id);
    std::ostringstream ids;
    std::size_t nch = 1;
    if (s) {
      nch = s->children.size();
      for (std::size_t i = 0; i < s->children.size(); ++i) ids << (i ? ";" : "") << s->children[i];
    } else {
      ids << c.id;
    }
    pf << frame_idx_ << ',' << c.id << ',' << c.surface_id << ',' << nch << ',' << ids.str() << ',' << c.st.n << ','
       << c.fit.center.x() << ',' << c.fit.center.y() << ',' << c.fit.center.z() << ',' << c.fit.normal.x() << ','
       << c.fit.normal.y() << ',' << c.fit.normal.z() << ',' << c.fit.d << ',' << c.fit.eig(0) << ',' << c.fit.eig(1)
       << ',' << c.fit.eig(2) << ',' << c.fit.planarity << ',' << c.bb_min.x() << ',' << c.bb_min.y() << ','
       << c.bb_min.z() << ',' << c.bb_max.x() << ',' << c.bb_max.y() << ',' << c.bb_max.z() << ",0,0,1\n";
  });

  std::ofstream sf(debugLogPath("surface_map_surfaces_" + tag + ".csv"));
  sf << std::setprecision(17);
  sf << "checkpoint,surface_id,n_patches,n_d2,points,max_angle_deg,max_offset_m,max_d2,mean_d2\n";
  for (std::uint64_t sid : layer_->surfaceIds()) {
    const surface::Surface* s = layer_->findSurface(sid);
    if (!s || !s->fit.valid) continue;
    std::size_t nvalid = 0;
    double pts = 0.0, max_ang = 0.0, max_off = 0.0;
    for (std::uint64_t cid : s->children) {
      const surface::Cell& c = layer_->cellById(cid);
      if (!c.fit.valid) continue;
      ++nvalid;
      pts += c.st.n;
      max_ang = std::max(max_ang, surface::planeAngleDeg(c.fit.normal, s->fit.normal));
      max_off = std::max(max_off, std::abs(s->fit.normal.dot(c.fit.center - s->fit.center)));
    }
    if (nvalid < 2) continue;
    sf << frame_idx_ << ',' << sid << ',' << nvalid << ",0," << pts << ',' << max_ang << ',' << max_off << ",0,0\n";
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Viz: one TRIANGLE_LIST of cell squares in each valid cell's own plane, coloured by surface id (single cells grey).
// ---------------------------------------------------------------------------------------------------------------------
bool SurfaceMap::vizDirty() const
{
  return opts_.viz_period_frames > 0 && viz_pending_ && (frame_idx_ - viz_last_frame_) >= opts_.viz_period_frames;
}

visualization_msgs::MarkerArray SurfaceMap::buildVizMarkers(const std::string& frame_id, const ros::Time& stamp)
{
  TimedScope ts(profiler_, "surfacemap/viz_markers");
  visualization_msgs::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "surface_cells";
  m.id = 0;
  m.type = visualization_msgs::Marker::TRIANGLE_LIST;
  m.action = visualization_msgs::Marker::ADD;
  m.scale.x = m.scale.y = m.scale.z = 1.0;
  m.pose.orientation.w = 1.0;
  m.color.a = 1.0f;
  const double half = 0.5 * opts_.core.cell_size;
  layer_->forEachCell([&](const surface::Cell& c) {
    if (!c.fit.valid) return;
    const surface::Surface* s = layer_->findSurface(c.surface_id);
    const bool multi = s && s->children.size() > 1;
    std_msgs::ColorRGBA col;
    col.a = 1.0f;
    if (multi) hueToRgb(c.surface_id, col.r, col.g, col.b);
    else { col.r = col.g = col.b = 0.6f; }
    const V3D u = c.fit.x_axis * half, v = c.fit.y_axis * half;
    const V3D corners[4] = {c.fit.center - u - v, c.fit.center + u - v, c.fit.center + u + v, c.fit.center - u + v};
    const int tri[6] = {0, 1, 2, 0, 2, 3};
    for (int k : tri) {
      geometry_msgs::Point p;
      p.x = corners[k].x(); p.y = corners[k].y(); p.z = corners[k].z();
      m.points.push_back(p);
      m.colors.push_back(col);
    }
  });
  viz_pending_ = false;
  viz_last_frame_ = frame_idx_;
  visualization_msgs::MarkerArray arr;
  arr.markers = {m};
  return arr;
}

}  // namespace livo_recon
