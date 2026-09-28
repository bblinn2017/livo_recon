#include "livo_recon/processing/map_bootstrapper.h"

#include "livo_recon/map/map_backend.h"
#include "livo_recon/processing/lio_base.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iomanip>

namespace livo_recon
{
std::string MapBootstrapper::seedSequentialStationary(
    std::vector<CalibrationLidarObservation> observations,
    LioProcBase& lio_proc)
{
  std::vector<PointXYZT> aggregate;
  std::vector<int> source_observation;
  size_t raw_count = 0;
  for (const auto& obs : observations) raw_count += obs.points.size();
  aggregate.reserve(raw_count);
  source_observation.reserve(raw_count);

  for (size_t obs_id = 0; obs_id < observations.size(); ++obs_id) {
    for (auto p : observations[obs_id].points) {
      p.t = 0.0;  // one stationary reference, matching the incumbent aggregate path
      aggregate.push_back(std::move(p));
      source_observation.push_back(static_cast<int>(obs_id));
    }
  }

  std::vector<PointXYZCov> processed;
  std::vector<int> processed_observation;
  lio_proc.preprocessStationaryBootstrap(
      aggregate, source_observation, processed, processed_observation);

  std::vector<std::vector<PointXYZCov>> groups(observations.size());
  for (size_t i = 0; i < processed.size(); ++i)
    groups.at(static_cast<size_t>(processed_observation.at(i))).push_back(std::move(processed[i]));

  const V3D p_before = ctx_.state->pos();
  const M3D R_before = ctx_.state->rot();
  const V3D v_before = ctx_.state->vel();
  const Eigen::MatrixXd P_before = ctx_.state->cov();
  const int live_frame_before = ctx_.voxel_map->frame_idx_;

  std::vector<size_t> group_counts(groups.size());
  for (size_t i = 0; i < groups.size(); ++i) group_counts[i] = groups[i].size();

  std::ofstream diag(debugLogPath("map_bootstrap.csv"));
  diag << "row_type,mode,observation_id,viewpoint_id,raw_points,processed_points,live_frame_idx\n";

  ctx_.voxel_map->beginBootstrap();
  for (size_t obs_id = 0; obs_id < groups.size(); ++obs_id) {
    if (groups[obs_id].empty()) continue;
    MeasureGroup mg;
    mg.image.t = 0.0;
    mg.points = std::move(groups[obs_id]);
    ctx_.voxel_map->insertBootstrapObservation(
        mg, static_cast<int>(obs_id), 0 /* one stationary viewpoint */);
    diag << "observation,sequential_stationary," << obs_id << ",0,"
         << observations[obs_id].points.size() << ',' << group_counts[obs_id] << ','
         << ctx_.voxel_map->frame_idx_ << "\n";
  }
  ctx_.voxel_map->finishBootstrap();

  const double state_delta = (ctx_.state->pos() - p_before).norm()
      + (ctx_.state->rot() - R_before).norm()
      + (ctx_.state->vel() - v_before).norm()
      + (ctx_.state->cov() - P_before).norm();
  if (state_delta != 0.0)
    throw std::logic_error("stationary map bootstrap mutated estimator state");
  if (ctx_.voxel_map->frame_idx_ != live_frame_before + 1)
    throw std::logic_error("stationary map bootstrap did not preserve the single legacy calibration frame boundary");

  diag << "summary,sequential_stationary,-1,0," << raw_count << ','
       << processed_observation.size() << ',' << ctx_.voxel_map->frame_idx_ << "\n";
  diag.flush();

  std::ostringstream out;
  out << "[map bootstrap] sequential_stationary observations=" << observations.size()
      << " raw_points=" << raw_count
      << " processed_points=" << processed_observation.size()
      << " inserted_points=";
  out << processed_observation.size()
      << " viewpoint_id=0 live_frame_idx=" << ctx_.voxel_map->frame_idx_;
  return out.str();
}
}  // namespace livo_recon
