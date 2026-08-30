#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/processing/cbk_processing.h"
#include "livo_recon/processing/pub_processing.h"
#include "livo_recon/processing/calib_processing.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/processing/lio_processing.h"
#include "livo_recon/processing/vio_processing.h"
#include "livo_recon/processing/combined_processing.h"
#include "livo_recon/processing/evo_processing.h"

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
  LioProc      lio_proc_;
  VioProc      vio_proc_;
  CombinedProc combined_proc_;
  EvoProc      evo_proc_;

  // Opt-in (outputs/auto_terminate_on_idle, default false) self-shutdown
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
};

}  // namespace livo_recon
