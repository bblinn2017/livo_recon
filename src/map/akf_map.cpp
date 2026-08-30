#include "livo_recon/map/akf_map.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/data/measures.h"

namespace livo_recon
{

AkfMap::AkfMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues)
  : opts_(std::make_shared<AkfMapOpts>())
  , state_(state)
  , profiler_(profiler)
  , data_queues_(data_queues)
{
  generateNearbyGrids();
}

std::string AkfMap::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<double>(pnh, "akf_map/voxel_size", opts_->voxel_size, 0.5);
  { int nt; paramWarn<int>(pnh, "akf_map/nearby_type", nt, 26);
    opts_->nearby_type = nt == 0 ? AkfNearbyType::CENTER : nt == 6 ? AkfNearbyType::NEARBY6 :
                         nt == 18 ? AkfNearbyType::NEARBY18 : AkfNearbyType::NEARBY26; }
  paramWarn<double>(pnh, "akf_map/t_mal", opts_->t_mal, 11.28);
  paramWarn<int>(pnh, "akf_map/min_num_match_points", opts_->min_num_match_points, 5);
  paramWarn<double>(pnh, "akf_map/flatness_eig_floor", opts_->flatness_eig_floor, 0.04);
  paramWarn<double>(pnh, "akf_map/t_stop_pseudo_merge", opts_->t_stop_pseudo_merge, 11.28);
  paramWarn<int>(pnh, "akf_map/num_match_points", opts_->num_match_points, 100);
  paramWarn<double>(pnh, "akf_map/query_max_range", opts_->query_max_range, 3.0);
  paramWarn<double>(pnh, "akf_map/range_gate_coeff", opts_->range_gate_coeff, 1.0 / 9.0);
  paramWarn<double>(pnh, "akf_map/t_ratio_b", opts_->t_ratio_b, 0.0);
  paramWarn<double>(pnh, "akf_map/init_uncertainty", opts_->init_uncertainty, 0.01);
  paramWarn<double>(pnh, "akf_map/init_lidar_cov", opts_->init_lidar_cov, 0.001);
  paramWarn<int>(pnh, "akf_map/max_fea_num", opts_->max_fea_num, 10000);
  paramWarn<double>(pnh, "akf_map/time_to_delete_local_map", opts_->time_to_delete_local_map, 1000.0);
  paramWarn<double>(pnh, "akf_map/capacity", opts_->capacity, 10000000.0);

  generateNearbyGrids();

  std::ostringstream oss;
  oss << "[params/akf_map]"
      << "\n  voxel_size:              " << opts_->voxel_size
      << "\n  nearby_type:             " << static_cast<int>(opts_->nearby_type)
      << "\n  t_mal:                   " << opts_->t_mal
      << "\n  min_num_match_points:    " << opts_->min_num_match_points
      << "\n  flatness_eig_floor:      " << opts_->flatness_eig_floor
      << "\n  t_stop_pseudo_merge:     " << opts_->t_stop_pseudo_merge
      << "\n  num_match_points:        " << opts_->num_match_points
      << "\n  query_max_range:         " << opts_->query_max_range
      << "\n  range_gate_coeff:        " << opts_->range_gate_coeff
      << "\n  t_ratio_b:               " << opts_->t_ratio_b
      << "\n  init_uncertainty:        " << opts_->init_uncertainty
      << "\n  init_lidar_cov:          " << opts_->init_lidar_cov
      << "\n  max_fea_num:             " << opts_->max_fea_num
      << "\n  time_to_delete_local_map:" << opts_->time_to_delete_local_map
      << "\n  capacity:                " << opts_->capacity;
  return oss.str();
}

void AkfMap::generateNearbyGrids()
{
  nearby_grids_.clear();
  switch (opts_->nearby_type) {
    case AkfNearbyType::CENTER:
      nearby_grids_ = {{0,0,0}};
      break;
    case AkfNearbyType::NEARBY6:
      nearby_grids_ = {{0,0,0},{-1,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,-1},{0,0,1}};
      break;
    case AkfNearbyType::NEARBY18:
      nearby_grids_ = {{0,0,0},{-1,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,-1},{0,0,1},
                       {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},
                       {1,0,-1},{-1,0,-1},{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}};
      break;
    case AkfNearbyType::NEARBY26:
    default:
      nearby_grids_ = {{0,0,0},{-1,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,-1},{0,0,1},
                       {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},
                       {1,0,-1},{-1,0,-1},{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},
                       {1,1,1},{-1,1,1},{1,-1,1},{1,1,-1},{-1,-1,1},{-1,1,-1},
                       {1,-1,-1},{-1,-1,-1}};
      break;
  }
}

bool AkfMap::isEmpty() const { return cells_.empty(); }

std::string AkfMap::statsString() const
{
  size_t n_points = 0;
  for (const auto& kv : cells_) n_points += kv.second.size();
  std::ostringstream oss;
  oss << "[akf_map] n_cells=" << cells_.size() << "  n_gaussians=" << n_points
      << "  last_n_map_pts=" << last_n_map_pts_ << "  last_n_active_voxels=" << last_n_active_voxels_;
  return oss.str();
}

void AkfMap::updateUncertaintyFeedback()
{
  AkfPoint fresh;
  fresh.uncertainty = opts_->init_uncertainty;
  fresh.use_num = 1;
  fresh.pt_num = 1;
  for (AkfPoint* mp : matched_this_period_) {
    *mp = updateResidualOnly(fresh, *mp, opts_->max_fea_num);
  }
  matched_this_period_.clear();
}

void AkfMap::insertPoint(const V3D& p_world, const M3D& cov, double time_now,
                         robin_hood::unordered_flat_set<VoxelKey, VoxelKeyHash>& touched_this_frame)
{
  AkfPoint incoming;
  incoming.point = p_world;
  incoming.cov = cov;
  incoming.pt_num = 1;
  incoming.use_num = 1;
  incoming.uncertainty = opts_->init_uncertainty;
  incoming.time = time_now;

  const VoxelKey key = worldToKeyFn(p_world, opts_->voxel_size);

  double best_d = -1.0;
  VoxelKey best_key{};
  int best_idx = -1;
  for (const auto& delta : nearby_grids_) {
    const VoxelKey nk{key.x + delta.x, key.y + delta.y, key.z + delta.z};
    auto it = cells_.find(nk);
    if (it == cells_.end()) continue;
    auto& cell = it->second;
    for (int j = 0; j < static_cast<int>(cell.size()); ++j) {
      const double d = malDistance(incoming.point, incoming.cov, cell[j].point, cell[j].cov);
      if (d < opts_->t_mal && (best_idx == -1 || d < best_d)) {
        best_d = d;
        best_key = nk;
        best_idx = j;
      }
    }
  }

  if (best_idx >= 0) {
    auto& src_cell = cells_[best_key];
    const AkfPoint merged = mergeTwoGaussians(incoming, src_cell[best_idx], opts_->max_fea_num);
    src_cell.erase(src_cell.begin() + best_idx);
    const VoxelKey merged_key = worldToKeyFn(merged.point, opts_->voxel_size);
    cells_[merged_key].push_back(merged);
    touched_this_frame.insert(merged_key);
  } else {
    cells_[key].push_back(incoming);
    touched_this_frame.insert(key);
  }
}

void AkfMap::erasePoints(double cur_time)
{
  for (auto it = cells_.begin(); it != cells_.end(); ) {
    bool erase_cell = false;
    if (!it->second.empty()) {
      double newest = it->second.front().time;
      for (const auto& p : it->second) newest = std::max(newest, p.time);
      if (cur_time - newest >= opts_->time_to_delete_local_map) erase_cell = true;
    }
    it = erase_cell ? cells_.erase(it) : std::next(it);
  }

  const double max_capacity = opts_->capacity / (opts_->voxel_size * opts_->voxel_size);
  while (static_cast<double>(cells_.size()) > max_capacity && !cells_.empty()) {
    cells_.erase(cells_.begin());
  }
}

void AkfMap::updateMap(MeasureGroup& mg)
{
  TimedScope ts_total(profiler_, "akfmap");
  frame_idx_++;
  const double cur_time = mg.image.t;

  // Refresh uncertainty on every point matched during this frame's
  // (just-finished) residual-building phase, BEFORE this frame's own
  // points get inserted -- mirrors MapIncremental()'s UpdateUncertainty()
  // -> AddPoints() ordering exactly.
  updateUncertaintyFeedback();

  const std::vector<PointXYZCov>& map_pts = mg.points;
  const int np = static_cast<int>(map_pts.size());
  std::vector<V3D> pts_world(np);
  std::vector<M3D> pts_cov(np);

  {
    TimedScope ts(profiler_, "akfmap/transform");
    const int threads = cappedOmpThreads();
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < np; ++i) {
      const PointXYZCov sensor_world = state_->toWorld(map_pts[i]);
      pts_world[i] = sensor_world.point;
      pts_cov[i] = sensor_world.sensor_cov + state_->poseCovAt(map_pts[i].point);
    }
  }

  // Single-threaded insertion -- see class doc comment on why (Mahalanobis
  // search crosses cell boundaries, so parallelizing by distinct key
  // isn't safe the way VoxelMap's per-key-independent insertion is).
  robin_hood::unordered_flat_set<VoxelKey, VoxelKeyHash> touched_this_frame;
  {
    TimedScope ts(profiler_, "akfmap/insert");
    for (int i = 0; i < np; ++i) {
      insertPoint(pts_world[i], pts_cov[i], cur_time, touched_this_frame);
    }
  }

  last_n_map_pts_ = np;
  last_n_active_voxels_ = static_cast<int>(touched_this_frame.size());

  {
    TimedScope ts(profiler_, "akfmap/erase");
    erasePoints(cur_time);
  }
}

bool AkfMap::findPlaneResidual(const WorldPointCov& pt, Residual& res, bool* /*tier0_had_plane*/) const
{
  const VoxelKey center = worldToKeyFn(pt.point, opts_->voxel_size);
  const M3D pt_cov = pt.sensor_cov + pt.pose_cov;

  // K-NN via Mahalanobis distance, within query_max_range (AKF-LIO's own
  // GetClosestPoint(..., max_range) argument is a dead parameter -- its
  // KNNPointByCondition never actually uses it, confirmed by reading
  // ivox3d_node.hpp. This port deliberately enforces it instead of
  // faithfully replicating that oversight, since doing so is strictly
  // safer and serves no purpose to leave dead.
  std::vector<std::pair<double, AkfPoint*>> candidates;
  for (const auto& delta : nearby_grids_) {
    const VoxelKey nk{center.x + delta.x, center.y + delta.y, center.z + delta.z};
    auto it = cells_.find(nk);
    if (it == cells_.end()) continue;
    for (auto& mp : it->second) {
      if ((mp.point - pt.point).norm() > opts_->query_max_range) continue;
      candidates.emplace_back(malDistance(pt.point, pt_cov, mp.point, mp.cov), &mp);
    }
  }
  if (candidates.empty()) return false;
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  if (static_cast<int>(candidates.size()) > opts_->num_match_points) {
    candidates.resize(opts_->num_match_points);
  }

  // Sequential pseudo-merge (ObsModel()'s p_sum accumulation loop).
  AkfPoint p_sum = *candidates[0].second;
  size_t survive_count = 1;
  for (size_t j = 1; j < candidates.size(); ++j) {
    const AkfPoint& cand = *candidates[j].second;
    if (p_sum.pt_num < opts_->min_num_match_points) {
      p_sum = mergeTwoGaussians(cand, p_sum, opts_->max_fea_num);
      survive_count = j + 1;
      continue;
    }
    Eigen::SelfAdjointEigenSolver<M3D> es(p_sum.cov);
    const V3D normal_y = es.eigenvectors().col(1);
    const V3D normal_z = es.eigenvectors().col(2);
    const V3D pq = p_sum.point - pt.point;
    const double e1 = es.eigenvalues()(1);
    const double e2 = es.eigenvalues()(2);
    const double p2q = std::pow(normal_y.dot(pq), 2) / e1 + std::pow(normal_z.dot(pq), 2) / e2;
    if (e1 < opts_->flatness_eig_floor ||
        p2q > opts_->t_stop_pseudo_merge * opts_->t_stop_pseudo_merge) {
      p_sum = mergeTwoGaussians(cand, p_sum, opts_->max_fea_num);
      survive_count = j + 1;
      continue;
    }
    break;  // gate failed -- truncate remaining candidates
  }
  if (p_sum.pt_num < opts_->min_num_match_points) return false;

  Eigen::SelfAdjointEigenSolver<M3D> es_final(p_sum.cov);
  const V3D normal = es_final.eigenvectors().col(0);  // smallest-eigenvalue direction
  const double thickness_raw = normal.transpose() * p_sum.cov * normal;
  const double thickness = std::max(thickness_raw, 1e-12);

  // Range-adaptive gate. AKF-LIO uses BODY-frame range (p_body.norm());
  // WorldPointCov carries no body-frame coordinate, so this port uses
  // world-frame distance from the current sensor position as the
  // equivalent physical proxy (same intent: farther points get a looser
  // gate) -- a deliberate, documented adaptation, not an oversight.
  const double range = (pt.point - state_->pos()).norm();
  const double p2pl = normal.dot(p_sum.point - pt.point);  // AKF-LIO's own sign: normal . (map - query)
  if (std::abs(p2pl) > opts_->range_gate_coeff * std::sqrt(range)) return false;

  // Fill Residual using THIS codebase's sign convention (r = n.(p-center),
  // see VoxelPlane::gate()) rather than AKF-LIO's own p2pl sign.
  res.r = -p2pl;
  res.normal = normal;
  res.sigma_squared = std::exp(opts_->t_ratio_b * p_sum.uncertainty) * thickness;
  // No persistent plane identity for this backend -- the local plane is
  // re-derived fresh per query, unlike VoxelMap's voxel-lifetime-stable
  // plane_id (see Residual::plane_id's doc comment).
  res.plane_id = nullptr;
  res.plane_var_term = 0.0;

  {
    std::lock_guard<std::mutex> lock(match_mutex_);
    for (size_t k = 0; k < survive_count; ++k) matched_this_period_.push_back(candidates[k].second);
  }
  return true;
}

}  // namespace livo_recon
