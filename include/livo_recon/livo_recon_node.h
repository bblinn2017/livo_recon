#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/processing/cbk_processing.h"
#include "livo_recon/processing/pub_processing.h"
#include "livo_recon/processing/calib_processing.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/processing/lio_iekf.h"
#include "livo_recon/processing/vio_processing.h"
#include "livo_recon/processing/combined_processing.h"
#include "livo_recon/processing/evo_processing.h"

#include <memory>

namespace livo_recon
{

class LivoReconNode
{
public:
  LivoReconNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);

  void run();

  // Reproducible-eval alternative to run(): reads bag_path directly via
  // rosbag::View (all subscribed topics merged into one strictly time-
  // ordered iterator) and dispatches each message straight to
  // cbk_proc_'s imageCallback()/lidarCallbackMsg()/lidarCallbackPcl()/
  // imuCallback() synchronously, in-process, as fast as possible -- no
  // rosbag play subprocess, no ROS transport/callback-queue timing
  // dependence at all. Mirrors FAST-LIVO2's LIVMapper::runOffline() (see
  // its doc comment) as closely as this codebase's structure allows.
  //
  // Unlike FAST-LIVO2's version, no manual "wait for async tracker, retry
  // feeding" loop is needed here: livo_recon's DataQueues::ready()/
  // popMeasureGroup() are already fully non-blocking (a measure group
  // simply isn't produced yet if the tracker isn't ready -- see
  // TrackedFrame's doc comment), so the ordinary run()-loop pattern of
  // "feed one input, then drain whatever's ready" already handles the
  // async tracker correctly without any offline-specific gating logic.
  // The tail end (bag exhausted, but the async tracker thread may still
  // be finishing in-flight work in real wall-clock time) is handled by
  // the SAME auto_terminate_on_idle mechanism run() already has -- forced
  // on for this mode (see runOffline()'s own comment) since there's no
  // rosbag-play process left to signal external completion.
  //
  // Only called from main() when common/offline_bag_path is set; run()
  // is untouched otherwise.
  void runOffline(const std::string& bag_path);

  // See loadParameters()'s common/offline_bag_path read -- empty (the
  // default) means main() calls run() unchanged.
  std::string offline_bag_path_;
  // Caps runOffline() to the first N seconds (bag-relative, from the
  // earliest message in the requested topics) of offline_bag_path_. <= 0
  // (default) means no cap -- process the whole bag. Mirrors FAST-LIVO2's
  // offline_duration_secs_.
  double offline_duration_secs_ = -1.0;

private:
  void loadParameters();
  void completeCalibrationAndBootstrap();
  void estimateState(MeasureGroup& mg);
  void updateMaps(MeasureGroup& mg);
  // One iteration of run()'s post-CALIB drain loop (sync + pop + estimate +
  // publish), factored out so runOffline() can share it exactly instead of
  // duplicating the loop body. Returns true if at least one measure group
  // was popped and processed this call.
  bool drainOnce();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;

  NodeContext ctx_;

  CbkProc   cbk_proc_;
  PubProc   pub_proc_;
  CalibProc calib_proc_;
  ImuProc   imu_proc_;
  // R97: the estimator is always the IEKF (the estimator/mode key and makeLioProc() were removed); the base-class pointer type is kept.
  std::unique_ptr<LioProcBase> lio_proc_;
  VioProc      vio_proc_;
  CombinedProc combined_proc_;
  EvoProc      evo_proc_;

  // Opt-in (run/auto_terminate_on_idle, default false) self-shutdown
  // for headless/scripted test runs: once popMeasureGroup() has come back
  // empty for auto_terminate_idle_secs_ real (wall-clock) seconds straight
  // -- i.e. every input queue has been empty that whole time, meaning the
  // node has genuinely finished processing everything published so far --
  // run() breaks its own loop and lets the normal end-of-run export
  // (exportColmap/exportReport) execute. See FAST-LIVO2's identical
  // LIVMapper::run() addition for the full rationale: this replaces
  // external heuristics (console-log silence, CPU-time polling) that
  // proved unreliable for GPU-bound VIO frontends, which can leave the
  // process legitimately CPU-idle for a few seconds at a time while
  // blocked on a GPU kernel, even mid-frame.
  bool auto_terminate_on_idle_ = false;
  double auto_terminate_idle_secs_ = 5.0;

  // common/insert_map_after_lio (default false, sequential mode only --
  // ignored when combined/enable is true, since combined mode has no
  // separate "after LIO, before VIO" point at all). When true,
  // estimateState() inserts this frame's points into the voxel map
  // immediately after LIO's correction (and BEFORE VIO's), so the map that
  // LIO/VIO residuals compute against on every SUBSEQUENT frame reflects
  // LIO's own globally-consistent estimate rather than VIO's frame-local
  // one. Default false preserves the original behavior: insertion happens
  // once, in updateMaps() (called after estimateState() fully returns, so
  // after BOTH LIO and VIO have corrected the state).
  bool insert_map_after_lio_ = false;
  // R66 item 0b: per-frame window log (log/frame_window, default false).
  bool frame_window_log_ = false;
  std::size_t frame_window_index_ = 0;
  // log/vio_correction/enable (default false): sequential-path only, see
  // diagnostics/vio_correction.h's doc comment for why combined mode has
  // no equivalent "post-LIO, pre-VIO" checkpoint to log this way.
  bool vio_correction_log_ = false;
  std::size_t vio_correction_frame_index_ = 0;
  // log/lidar_image_bounds (default false): see diagnostics/lidar_image_bounds.h's doc comment.
  bool lidar_image_bounds_log_ = false;
  std::size_t lidar_image_bounds_frame_index_ = 0;
  // log/state_correction_trace (default false): see diagnostics/state_correction_trace.h's doc comment.
  bool state_correction_trace_log_ = false;
  std::size_t state_correction_trace_frame_index_ = 0;

  // TEMPORARY (2026-10-27, mid360 lio/sequential/combined VIO-error
  // comparison investigation -- delete this flag + its call sites +
  // diagnostics/vio_residual_diag.* once done). diag/force_vio_residual_log,
  // default false (no behavior change for any existing config). When true:
  // (1) forces the async tracker to start even when vio/enable and
  // combined/enable are both false (normally the tracker never starts in
  // that case, so lio_only mode has no tracked features at all to compute
  // any VIO residual against -- see loadParameters()'s tracker-startup
  // gate), and (2) after every frame's pose is finalized (regardless of
  // which mode produced it), re-evaluates VioProc::accumulateForCombined()
  // read-only (no state mutation -- see its own doc comment) at that final
  // pose and logs one row to vio_residual_diag.csv. This gives a uniform,
  // mode-agnostic "post-refinement avg VIO pixel-reprojection error" series
  // even for lio_only, which otherwise never computes any such quantity.
  bool force_vio_residual_log_ = false;
  std::size_t vio_residual_diag_frame_index_ = 0;

  // TEMPORARY (2026-10-27, independent LiDAR-registration investigation --
  // delete alongside diagnostics/lidar_window_dump.* once done).
  // diag/dump_lidar_window_end_scan, default 0 (disabled, no behavior
  // change). When > 0, every frame's deskewed body-frame points are
  // appended to lidar_window_dump.csv for scan_id in [1, this value] --
  // meant for a small pre-motion window (tens of frames), not the whole
  // run, so keep this set to just past the frame of interest.
  int dump_lidar_window_end_scan_ = 0;

  // TEMPORARY (2026-10-27, apples-to-apples ask-comparison investigation --
  // delete alongside diagnostics/pin_state_diag.* once done). diag/
  // pin_state_before_scan (default -1, disabled) + diag/pin_state_file +
  // diag/pin_state_phase (default "post_lio"): when pin_state_before_scan_
  // >= 0, the frame whose scan_id (ctx_.voxel_map->frame_idx_) equals it
  // has its state_ OVERWRITTEN from the matching row of pin_state_file_
  // (a state_trace.csv from a reference run, e.g. a lio_only baseline),
  // right before that frame's IMU propagation runs -- see estimateState()'s
  // call site. Lets two differently-configured runs (e.g. sequential vs.
  // combined) be forced to enter the SAME pivot frame from an IDENTICAL
  // prior state, isolating that one frame's own correction from the two
  // runs' otherwise-independently-diverged trajectories. No effect on any
  // other frame; no effect at all when pin_state_before_scan_ < 0 (default).
  int pin_state_before_scan_ = -1;
  std::string pin_state_file_;
  std::string pin_state_phase_ = "post_lio";
};

}  // namespace livo_recon
