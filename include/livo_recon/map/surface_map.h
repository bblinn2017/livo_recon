#pragma once

#include <atomic>
#include <fstream>
#include <memory>

#include "livo_recon/common_lib.h"
#include "livo_recon/map/map_backend.h"
#include "livo_recon/map/surface/surface_core.h"

namespace livo_recon
{

// "surface" MapBackend: a flat grid of cells with O(1) additive (debiased-path) plane statistics, plus a dirty-set
// merge/split layer that groups neighbouring cell planes into surfaces (the stationary harness's gaussian_surface
// family). See map/surface/surface_core.h for the statistics and the merge logic; this class owns the ROS-facing
// parts: parameters, world-frame insertion, the residual query, bootstrap, viz and diagnostics.
//
// Residual model: a point in a cell that belongs to a valid multi-cell surface is judged against the SURFACE plane,
// otherwise against the cell's own plane; on a miss the single directional neighbour cell (the cell the point leans
// toward, same rule as VoxelMap's tier 1) is tried the same way. There is no tier-2 box search.
// The gate and the variance terms are VoxelPlane::gate()/computeResidual() for plane_gate_mode=disc,
// weight_floor/mode=sensor_range, pose_cov_in_sigma as configured; occupancy/visibility/plane-confidence machinery is
// NOT part of this backend (vis_state stays -1).
class SurfaceMap : public MapBackend
{
public:
  SurfaceMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues);

  std::string loadParameters(ros::NodeHandle& pnh) override;

  void updateMap(MeasureGroup& mg) override;
  void bootstrap(const std::vector<std::vector<PointXYZCov>>& observations) override;

  bool findPlaneResidual(const WorldPointCov& pt, Residual& res, bool* tier0_had_plane = nullptr,
                         bool* had_converged_neighbor = nullptr) const override;
  bool hasConvergedNeighbor(const V3D& p_world) const override;

  bool isEmpty() const override;
  std::string statsString() const override;

  bool vizDirty() const override;
  visualization_msgs::MarkerArray buildVizMarkers(const std::string& frame_id, const ros::Time& stamp) override;

private:
  struct Opts
  {
    surface::Options core;
    double merge_gap_cells = 3.0;
    double support_gap_cells = 1.0;
    // residual side (shared voxel_map/residual/* keys)
    double sigma_num_squared = 9.0;
    double max_radius = 3.0;
    bool pose_cov_in_sigma = false;
    double weight_sigma_r2 = 0.0025;   // imu/sensor/range_err^2, the sensor_range weight floor
    // diagnostics
    bool frame_log_en = true;
    bool snapshot_post_calibration = false;
    int snapshot_period_frames = 0;    // 0 = off
    int viz_period_frames = 20;        // 0 = viz off
  };

  void updateMapInternal(const std::vector<PointXYZCov>& pts_body, bool advance_live_frame, bool bootstrap_phase);
  bool residualAgainst(const WorldPointCov& pt, const surface::Fit& fit, const void* owner, std::uint64_t node_id,
                       Residual& res) const;
  // Tries the plane of cell `c` (surface plane if it belongs to a valid multi-cell surface, else its own plane).
  // `plane_exists` is set true if the cell had a plane at all.
  bool tryCell(const WorldPointCov& pt, const surface::Cell& c, Residual& res, bool& plane_exists,
               bool& used_surface) const;
  void writeSnapshot(const std::string& tag) const;
  void writeFrameRow(int frame, bool bootstrap_phase, std::size_t n_points, const surface::UpdateStats& us,
                     double ms_transform, double ms_insert, double ms_update);

  Opts opts_;
  std::unique_ptr<surface::SurfaceLayer> layer_;
  StateGroupPtr state_;
  ProfilerPtr profiler_;
  DataQueuesPtr data_queues_;
  std::string bootstrap_mode_ = "aggregate";

  // Residual-query counters since the last frame row (relaxed atomics; queries run under OMP).
  mutable std::atomic<std::uint64_t> q_total_{0}, q_hit_surface0_{0}, q_hit_cell0_{0}, q_hit_surface1_{0},
      q_hit_cell1_{0}, q_miss_{0};

  std::ofstream frame_log_;
  bool frame_log_open_ = false;

  // viz
  bool viz_pending_ = false;
  int viz_last_frame_ = -1000000;
};

}  // namespace livo_recon
