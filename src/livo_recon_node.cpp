#include "livo_recon/livo_recon_node.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <stdexcept>

#include <rosbag/bag.h>
#include <rosbag/view.h>

namespace livo_recon
{

LivoReconNode::LivoReconNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh), it_(nh_),
    ctx_(nh_, pnh_, it_),
    cbk_proc_(ctx_), pub_proc_(ctx_), calib_proc_(ctx_),
    imu_proc_(ctx_), lio_proc_(ctx_), vio_proc_(ctx_), combined_proc_(ctx_), evo_proc_(ctx_)
{
  loadParameters();
  ctx_.printer->print(PrintCategory::PARAMS, "[node] initialized");
}

void LivoReconNode::loadParameters()
{
  // Must be set before any other loadParameters() call below -- several of
  // them (vio_proc_, cbk_proc_, and every per-file debugLogXxx() helper in
  // lio_processing.cpp/imu_processing.cpp/evo_processing.cpp/voxelnode.cpp/
  // voxelplane.cpp) resolve their own debug-log path through debugLogPath()
  // at load/first-write time. Empty (default) preserves the old hardcoded
  // "/tmp/<name>.txt" behavior for ad hoc/manual runs.
  std::string debug_log_dir;
  pnh_.param<std::string>("outputs/debug_log_dir", debug_log_dir, "");
  setDebugLogDir(debug_log_dir);
  ctx_.printer->print(PrintCategory::PARAMS,
      "[params/common]  debug_log_dir: " + (debug_log_dir.empty() ? std::string("(unset, using /tmp)") : debug_log_dir));

  std::string profiler_export;
  pnh_.param<std::string>("outputs/profiler/export_path", profiler_export, "");
  ctx_.profiler->setExportPath(profiler_export);

  ctx_.printer->print(PrintCategory::PARAMS, ctx_.printer->loadParameters(pnh_));
  ctx_.printer->print(PrintCategory::PARAMS, ctx_.state->loadParameters(pnh_));

  ctx_.printer->print(PrintCategory::PARAMS, ctx_.voxel_map->loadParameters(pnh_));

  // vio_proc_ (which owns the vio/enable switch) must load before the
  // tracker so we know whether to load the tracker's backend at all -- with
  // VIO disabled (an LIO-only ablation), no CoTracker/TAPNext/Track-On
  // model should ever be constructed, not just skipped per-frame.
  // combined_proc_ also needs loading here (not down with the other procs
  // below) since CombinedProc::processCombined() always calls VioProc::
  // accumulateForCombined() unconditionally (it doesn't check vio/enable at
  // all) -- so the tracker must load whenever EITHER flag wants VIO
  // residuals, or combined mode would silently degrade to LIO-only.
  ctx_.printer->print(PrintCategory::PARAMS, vio_proc_.loadParameters(pnh_));
  ctx_.printer->print(PrintCategory::PARAMS, combined_proc_.loadParameters(pnh_));
  if (vio_proc_.enabled() || combined_proc_.enabled())
  {
    ctx_.printer->print(PrintCategory::PARAMS, ctx_.tracker->loadParameters(pnh_));
    if (!ctx_.tracker->loaded())
    {
      // A silently-failed tracker load previously meant every frame's VIO
      // correction was skipped for the rest of the run -- output looked
      // plausible (identical to a no-VIO run) but was never real VIO, and
      // nothing surfaced the difference short of manually diffing vio.txt
      // against lio.txt. Confirmed this exact failure mode in the FAST-LIVO2
      // livo_vio A/B scaffold: a grid_rows/grid_cols value with no matching
      // AOT-compiled checkpoint on disk silently ran an entire 9-sequence
      // suite as LIO-only. Fail loudly and immediately instead.
      throw std::runtime_error(
          "livo_recon: tracker backend failed to load, refusing to run "
          "with VIO silently disabled (see FAILED-to-load log line above)");
    }
    // Async tracking pipeline (task #145) -- safe to start now that the
    // backend is confirmed loaded.
    ctx_.tracker->startAsyncTracking();
  }
  else
  {
    ctx_.printer->print(PrintCategory::PARAMS, "[params/tracker]  skipped (vio/enable=false)");
  }

  // pub_proc_ must come after state_->loadParameters so cameras.bin can use intrinsics
  ctx_.printer->print(PrintCategory::PARAMS, pub_proc_.loadParameters(pnh_));

  ctx_.printer->print(PrintCategory::PARAMS, cbk_proc_.loadParameters(pnh_));
  ctx_.printer->print(PrintCategory::PARAMS, calib_proc_.loadParameters(pnh_));
  ctx_.printer->print(PrintCategory::PARAMS, imu_proc_.loadParameters(pnh_));
  ctx_.printer->print(PrintCategory::PARAMS, lio_proc_.loadParameters(pnh_));
  ctx_.printer->print(PrintCategory::PARAMS, evo_proc_.loadParameters(pnh_));

  pnh_.param<bool>("outputs/auto_terminate_on_idle", auto_terminate_on_idle_, false);
  pnh_.param<double>("outputs/auto_terminate_idle_secs", auto_terminate_idle_secs_, 5.0);

  pnh_.param<bool>("common/insert_map_after_lio", insert_map_after_lio_, false);
  ctx_.printer->print(PrintCategory::PARAMS,
      std::string("[params/common]  insert_map_after_lio: ") +
      (insert_map_after_lio_ ? "true" : "false"));

  // See offline_bag_path_'s doc comment -- reproducible-eval alternative
  // to run(), mirroring FAST-LIVO2's common/offline_bag_path. Read via
  // pnh_ (not nh_, unlike FAST-LIVO2's nh-scoped "common/" params) --
  // every launch file in this codebase loads its <rosparam> blocks INSIDE
  // the <node> tag (see livo_recon_slam_2022.launch), which lands them in
  // the node's PRIVATE namespace regardless of the "common/" key prefix,
  // so nh_.param() here would never see anything set via config/defaults/
  // common.yaml or a per-run overrides_file.
  pnh_.param<std::string>("common/offline_bag_path", offline_bag_path_, "");
  pnh_.param<double>("common/offline_duration_secs", offline_duration_secs_, -1.0);
  if (!offline_bag_path_.empty() && !auto_terminate_on_idle_)
  {
    // Without a rosbag-play subprocess, nothing external ever signals
    // "this run is done" -- runOffline() depends on the SAME idle-based
    // termination run() already has (see its own doc comment) to exit
    // once the bag is exhausted and the async tracker/EKF pipeline has
    // genuinely drained, not just paused mid-frame. Force it on rather
    // than hanging forever if the config forgot to set it.
    ctx_.printer->print(PrintCategory::PARAMS,
        "[node] common/offline_bag_path set but outputs/auto_terminate_on_idle "
        "was false -- forcing it on (offline mode has no other way to detect "
        "completion)");
    auto_terminate_on_idle_ = true;
  }
}

void LivoReconNode::estimateState(MeasureGroup& mg) {
  TimedScope ts_total(ctx_.profiler, "state_estimation");

  { TimedScope ts(ctx_.profiler, "imu"); imu_proc_.processIMU(mg); }
  mg.pos_after_imu = ctx_.state->pos();
  mg.rot_after_imu = ctx_.state->rot();

  // Deskew + downsample (see LioProc::deskewAndDownsample()'s own doc
  // comment for why this is called unconditionally here, before the
  // sequential-vs-combined branch, rather than from inside processLIO()
  // itself) -- populates mg.points, which BOTH paths below assume is
  // already populated (mirrors what ImuProc::undistortLidar()/
  // downsamplePoints() used to guarantee before this deskewing logic
  // moved out of ImuProc).
  lio_proc_.deskewAndDownsample(mg);

  // mg.tracked_frame was already populated by CbkProc::syncMeasures()
  // before this measure group ever reached popMeasureGroup() (task #145)
  // -- no wait needed here, whichever path below reads it. Resolve its
  // anchor pose (rot_anchor/pos_anchor) BEFORE either path can read it --
  // single, mode-agnostic call site (see Tracker::resolveAnchorPose()'s
  // doc comment for the design this replaces: a caller-driven "commit"
  // pattern that caused a real bug when a new caller, CombinedProc, didn't
  // know to call it). Guarded on asyncTrackingActive() so this is a no-op
  // (and the pose history never grows) when VIO/combined are both off.
  if (ctx_.tracker->asyncTrackingActive())
    ctx_.tracker->resolveAnchorPose(mg.tracked_frame);

  if (combined_proc_.enabled())
  {
    // Opt-in joint LIO+VIO step (combined/enable, default false) -- see
    // combined_processing.h's doc comment. mg.pos_after_lio/pos_after_vio
    // (and rot_ counterparts) are set to the SAME combined result so
    // EvoProc's existing per-stage ATE table keeps working unmodified:
    // there's no meaningful separate "LIO-only" checkpoint once the two
    // are fused, so both stages correctly read identically here.
    std::string combined_log;
    { TimedScope ts(ctx_.profiler, "combined"); combined_log = combined_proc_.processCombined(mg, lio_proc_, vio_proc_); }
    mg.pos_after_lio = mg.pos_after_vio = ctx_.state->pos();
    mg.rot_after_lio = mg.rot_after_vio = ctx_.state->rot();
    ctx_.printer->print(PrintCategory::LIO, combined_log);
  }
  else
  {
    // Original sequential path, unchanged.
    std::string lio_log, vio_log;
    { TimedScope ts(ctx_.profiler, "lio"); lio_log = lio_proc_.processLIO(mg); }
    mg.pos_after_lio = ctx_.state->pos();
    mg.rot_after_lio = ctx_.state->rot();

    // Opt-in (common/insert_map_after_lio, default false): insert this
    // frame's points using LIO's own corrected pose, before VIO gets a
    // chance to move the state -- see insert_map_after_lio_'s doc comment
    // in livo_recon_node.h for the motivation (VIO's frame-local correction
    // never gets baked into the map LIO/VIO themselves match against on
    // future frames). updateMaps() (called after estimateState() returns)
    // checks mg.map_inserted and skips its own call when this already ran.
    if (insert_map_after_lio_)
    {
      ctx_.voxel_map->updateMap(mg);  // times itself internally ("voxelmap" scope)
      mg.map_inserted = true;
    }

    { TimedScope ts(ctx_.profiler, "vio"); vio_log = vio_proc_.processVIO(mg); }
    mg.pos_after_vio = ctx_.state->pos();
    mg.rot_after_vio = ctx_.state->rot();

    ctx_.printer->print(PrintCategory::LIO, lio_log);
    ctx_.printer->print(PrintCategory::VIO, vio_log);
  }

  // Record THIS frame's own final, fully-corrected pose (whichever path
  // above produced it, including any CombinedProc rollback) for whatever
  // FUTURE TrackedFrame's prev_timestamp matches this one -- see
  // Tracker::recordFramePose()'s doc comment. Must happen last, after
  // state_ reflects the truly final pose.
  if (ctx_.tracker->asyncTrackingActive())
    ctx_.tracker->recordFramePose(mg.tracked_frame.timestamp);
}

void LivoReconNode::updateMaps(MeasureGroup& mg) {
  // Skip if estimateState() already inserted this frame's points (see
  // common/insert_map_after_lio / mg.map_inserted's doc comments) --
  // otherwise every frame would be inserted into the map twice.
  if (!mg.map_inserted)
    ctx_.voxel_map->updateMap(mg);
  // Tracker reseed decisions now happen INSIDE the async pipeline itself
  // (see Tracker::stepAndReseedOnce(), consumed by VioProc::processVIO()
  // in estimateState() above) -- no separate call needed here anymore
  // (task #145 removed the old synchronous ctx_.tracker->updateTracker(mg)
  // call that used to live in this exact spot).
}

bool LivoReconNode::drainOnce()
{
  while (cbk_proc_.syncMeasures()) continue;

  MeasureGroup mg;
  bool popped_any = false;
  while (ctx_.measures->popMeasureGroup(mg))
  {
    popped_any = true;
    estimateState(mg);
    ctx_.printer->print(PrintCategory::EVO, evo_proc_.processEvo(mg));
    updateMaps(mg);
    pub_proc_.publishResults(mg);
    // Export periodically (every measure group), not just on a clean
    // shutdown (below) -- a killed/crashed run (e.g. a test harness's
    // --terminate, or any non-SIGINT kill) previously left
    // outputs/profiler/export_path never written at all.
    ctx_.profiler->exportReport();
  }
  return popped_any;
}

// Shared tail sequence for run() and runOffline() -- both loops end with
// the same "stop tracking, export, finalize" steps, just reached via
// different feed mechanisms above.
namespace
{
void finishRun(NodeContext& ctx, PubProc& pub_proc, EvoProc& evo_proc)
{
  // Stop the async tracking thread cleanly before shutdown (task #145) --
  // no-op if it was never started (VIO disabled or tracker failed to load).
  ctx.tracker->stopAsyncTracking();

  pub_proc.exportColmap();
  ctx.profiler->exportReport();
  // gt_source=="file" only, no-op otherwise: commits the last sparse
  // checkpoint's pending match if the run ends while its window is still
  // open (no later frame ever arrived to close it via processEvo()) --
  // see EvoProc::finalizePendingFileMatch()'s doc comment. Mirrors the
  // pattern of the two calls just above (the only "run finished" hook
  // point in this codebase -- no destructor/signal handler exists).
  ctx.printer->print(PrintCategory::EVO, evo_proc.finalizePendingFileMatch());
}
}  // namespace

void LivoReconNode::run()
{
  ros::Rate rate(5000);

  // CALIB
  while (ros::ok() && !ctx_.measures->calib_done.get()) {
    ros::spinOnce();

    ctx_.printer->print(PrintCategory::CALIB, calib_proc_.estimateFromBuffer());
    ctx_.printer->print(PrintCategory::PROFILER, ctx_.profiler->report());

    rate.sleep();
  }

  ros::Time last_pop_time = ros::Time::now();
  while (ros::ok())
  {
    ros::spinOnce();
    bool popped_any = drainOnce();

    if (popped_any)
      last_pop_time = ros::Time::now();
    else if (auto_terminate_on_idle_ &&
             (ros::Time::now() - last_pop_time).toSec() >= auto_terminate_idle_secs_)
    {
      ctx_.printer->print(PrintCategory::PARAMS,
          "[node] auto_terminate_on_idle: no new measure group for " +
          std::to_string(auto_terminate_idle_secs_) + "s -- shutting down");
      break;
    }

    rate.sleep();
  }

  finishRun(ctx_, pub_proc_, evo_proc_);
}

void LivoReconNode::runOffline(const std::string& bag_path)
{
  ctx_.printer->print(PrintCategory::PARAMS,
      "[node] runOffline: reading " + bag_path + " directly (no rosbag play, "
      "no ROS transport/callback queues in the loop) for reproducible evaluation");

  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);

  const CbkProcOptions& cbk_opts = cbk_proc_.opts();
  std::vector<std::string> topics = {cbk_opts.image_topic, cbk_opts.lidar_topic, cbk_opts.imu_topic};

  // Ground-truth topic dispatch -- see CbkProcOptions::evo_enable's doc
  // comment and CbkProc::gtCallback(). Only relevant for gt_source=="topic"
  // (e.g. NTU_VIRAL's live Leica stream): runOffline() never runs rosbag
  // play or any real ROS transport, so a plain topic subscriber would
  // otherwise NEVER fire in offline mode -- confirmed 2026-08-12: all 9
  // NTU_VIRAL jobs in a livo_recon offline-mode sweep failed (0 "[evo"
  // lines each) for exactly this reason, while file-GT (HILTI) jobs worked
  // fine since gt_source=="file" reads a static file directly, no topic
  // delivery involved -- doesn't need gt_topic added to the bag view.
  const bool feed_gt_topic = cbk_opts.evo_enable && cbk_opts.evo_gt_source == "topic";
  if (feed_gt_topic) topics.push_back(cbk_opts.evo_gt_topic);

  rosbag::View view(bag, rosbag::TopicQuery(topics));
  rosbag::View::iterator bag_it = view.begin();
  const rosbag::View::iterator bag_end = view.end();
  const ros::Time bag_start_time = view.getBeginTime();
  if (offline_duration_secs_ > 0.0)
    ctx_.printer->print(PrintCategory::PARAMS,
        "[node] runOffline: capping to first " + std::to_string(offline_duration_secs_) + "s (bag-relative)");

  // Feeds exactly one message from the bag view (topics merged into a
  // single strictly-timestamp-ordered stream -- see rosbag::View's own
  // docs) directly into the same callback the live ROS subscriber would
  // have invoked. No queueing, no threads, no wall-clock pacing: this call
  // either dispatches synchronously right now, on this thread, or the bag
  // is exhausted. Also treats hitting offline_duration_secs_ (if set) as
  // "exhausted" -- same early-stop effect as reaching bag_end.
  auto feedNextMessage = [&]() -> bool {
    if (bag_it == bag_end) return false;
    if (offline_duration_secs_ > 0.0 &&
        (bag_it->getTime() - bag_start_time).toSec() > offline_duration_secs_)
      return false;
    const rosbag::MessageInstance& m = *bag_it;
    if (feed_gt_topic && m.getTopic() == cbk_opts.evo_gt_topic)
    {
      geometry_msgs::PoseStamped::ConstPtr msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (msg) cbk_proc_.gtCallback(msg);
    }
    else if (m.getTopic() == cbk_opts.imu_topic)
    {
      sensor_msgs::Imu::ConstPtr msg = m.instantiate<sensor_msgs::Imu>();
      if (msg) cbk_proc_.imuCallback(msg);
    }
    else if (m.getTopic() == cbk_opts.image_topic)
    {
      sensor_msgs::Image::ConstPtr msg = m.instantiate<sensor_msgs::Image>();
      if (msg) cbk_proc_.imageCallback(msg);
    }
    else if (m.getTopic() == cbk_opts.lidar_topic)
    {
      if (cbk_opts.lidar_type == LidarType::AVIA)
      {
        livox_ros_driver::CustomMsg::ConstPtr msg = m.instantiate<livox_ros_driver::CustomMsg>();
        if (msg) cbk_proc_.lidarCallbackMsg(msg);
      }
      else
      {
        sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (msg) cbk_proc_.lidarCallbackPcl(msg);
      }
    }
    ++bag_it;
    return true;
  };

  // CALIB -- same structure as run()'s own CALIB loop, feedNextMessage()
  // replacing ros::spinOnce(). Stops once calib_done fires OR the bag runs
  // out before calibration ever settles (a config/data problem, not
  // something to spin forever on).
  while (ros::ok() && !ctx_.measures->calib_done.get())
  {
    if (!feedNextMessage()) break;
    ctx_.printer->print(PrintCategory::CALIB, calib_proc_.estimateFromBuffer());
    ctx_.printer->print(PrintCategory::PROFILER, ctx_.profiler->report());
  }

  // Main loop -- unlike FAST-LIVO2's runOffline(), no manual "vio_round_
  // pending, wait for async tracker, retry feeding" branch is needed: see
  // this method's doc comment (livo_recon_node.h) for why livo_recon's
  // non-blocking DataQueues::ready()/popMeasureGroup() design already
  // handles that correctly with the same feed-then-drain pattern run()
  // uses. Idle-based termination (forced on in loadParameters() for this
  // mode) covers the tail: once the bag is exhausted, feedNextMessage()
  // returns false forever, and the loop just keeps draining/sleeping until
  // the async tracker's last in-flight frame(s) finish and
  // auto_terminate_idle_secs_ of true idle time has passed.
  ros::Rate rate(5000);
  ros::Time last_pop_time = ros::Time::now();
  while (ros::ok())
  {
    feedNextMessage();
    bool popped_any = drainOnce();

    if (popped_any)
      last_pop_time = ros::Time::now();
    else if ((ros::Time::now() - last_pop_time).toSec() >= auto_terminate_idle_secs_)
    {
      ctx_.printer->print(PrintCategory::PARAMS,
          "[node] runOffline: no new measure group for " +
          std::to_string(auto_terminate_idle_secs_) + "s -- shutting down");
      break;
    }

    rate.sleep();
  }
  bag.close();

  finishRun(ctx_, pub_proc_, evo_proc_);
}

}  // namespace livo_recon
