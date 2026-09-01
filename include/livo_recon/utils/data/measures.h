#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "livo_recon/utils/data/data_queues.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/vio/tracked_frame.h"

namespace livo_recon
{

struct PlanePoints
{
  std::vector<V3D> body_points;
  V3D normal_w;
};

struct MeasureGroup
{
  // Raw Data
  ImageData image;
  std::vector<PointXYZT> lidar_points;
  std::deque<ImuSample> imu_samples;

  // IMU Processing
  std::vector<Pose6D> poses;

  // A copy of this frame's raw IMU samples, kept ONLY when the scan-spline
  // mechanism is on (SplineOptions::enable).  ImuProc::propagate() clears
  // imu_samples once it has consumed them, but the spline-vs-IMU residual
  // -- which is the whole basis of the live Q estimate -- needs the raw,
  // unaveraged, un-bias-corrected stream afterwards.  Empty (and never
  // populated, zero cost) with the feature off.  Includes the interpolated
  // final sample at mg.image.t so the residual covers the full scan.
  std::vector<ImuSample> imu_samples_raw;

  // LIO Processing -- the single point set used for both residual matching
  // and voxel-map insertion (VoxelMap::updateMap()). Built from
  // lidar_points via point_filter_num (keep-every-Nth, applied at
  // ingestion -- CbkProc) then ds_leaf_size (voxel-grid downsample,
  // applied in LioProc::processLIO() after deskewing -- moved from
  // ImuProc 2026-08-18) -- both no-ops at their extreme settings
  // (point_filter_num=1, ds_leaf_size=0), so there's one point-processing
  // pipeline, not a separate decimated/full-resolution pair. Set once
  // (LioProc::deskewAndDownsample()) and never touched again this frame.
  std::vector<PointXYZCov> points;

  // 2026-08-24: REMOVED points_orig/points_orig_offset/spline_anchor --
  // state for the removed iterative-deskew Hermite-spline mechanism. See
  // docs/removed_livo_recon_spline_deskew_2026aug24.md.

  // 2026-08-24: opt-in shadow dry-run diagnostic (LioProcOptions::
  // dry_run_point_filter_num) -- a SECOND raw point set, decimated at a
  // different point_filter_num than the primary `points` above (see
  // CbkProc's ingestion), used ONLY by LioProc::runDryRunShadowPass() to
  // run a non-committing shadow IEKF pass for direct comparison against
  // the real frame's residual structure. Empty (no-op, zero cost) unless
  // dry_run_point_filter_num > 0. Never touched by anything else --
  // never inserted into the voxel map, never affects `points`/state_.
  std::vector<PointXYZT> dry_run_lidar_points;
  std::vector<PointXYZCov> dry_run_points;

  // State Snapshots -- pos()/rot() captured right after each of
  // LivoReconNode::estimateState()'s three refinement stages, so EvoProc
  // can compare ATE/ARE per stage (IMU propagation alone, then LIO's
  // correction, then VIO's correction) instead of only the final result --
  // see EvoProc::processEvo()'s per-stage debug logging.
  V3D pos_after_imu = V3D::Zero();
  M3D rot_after_imu = M3D::Identity();
  V3D pos_after_lio = V3D::Zero();
  M3D rot_after_lio = M3D::Identity();
  V3D pos_after_vio = V3D::Zero();
  M3D rot_after_vio = M3D::Identity();

  // Visualization Outputs
  cv::Mat grid_vis_image;
  std::string evo_stats;  // latest "n=.. ATE=.. RTE=.. ROE=.." line, set by EvoProc::processEvo() -- empty until enough poses have been matched to align

  // VIO tracking result for this group's image (task #145) -- populated by
  // CbkProc::syncMeasures() (consumeResult(), guaranteed ready by the time
  // a MeasureGroup is formed at all -- see its own waitReadyFor(0ms) gate)
  // directly from the async tracker's background thread, so
  // VioProc::processVIO() just reads this field, no wait/consume call of
  // its own needed. Default-constructed (tracked_frame.ok == false) when
  // the async pipeline was never started (VIO disabled or tracker failed
  // to load) -- same as the old "tracker not ready" skip path.
  TrackedFrame tracked_frame;

  // Set true by LivoReconNode::estimateState() when common/insert_map_
  // after_lio has already inserted this frame's points into the voxel map
  // (right after LIO, before VIO runs) -- tells the later
  // LivoReconNode::updateMaps() call (in drainOnce(), after estimateState()
  // returns) to skip re-inserting the same points a second time. Always
  // false (default insertion timing: after VIO, in updateMaps()) unless
  // that option is on.
  bool map_inserted = false;

  MeasureGroup() = default;

  MeasureGroup(ImageData&& image,
               std::vector<PointXYZT>&& lidar_points,
               std::deque<ImuSample>&& imu_samples)
    : image(std::move(image)),
      lidar_points(std::move(lidar_points)),
      imu_samples(std::move(imu_samples))
      {}
};

struct Measures
{
  LockedValue<bool>   calib_done{false};
  LockedValue<double> curr_time{0.0};

  std::mutex measure_mutex;
  std::deque<MeasureGroup> measure_groups;

  bool isValid(double t) const;
  void pushMeasureGroup(MeasureGroup&& mg);
  bool popMeasureGroup(MeasureGroup& mg);
};
using MeasuresPtr = std::shared_ptr<Measures>;

}  // namespace livo_recon
