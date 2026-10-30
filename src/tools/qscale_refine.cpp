// qscale_refine -- standalone LiDAR-only trajectory estimator for IMU Q-scale
// calibration, built on livo_recon's own LIO/VoxelMap pipeline instead of
// LiDAR_IMU_Init's li_init + ct_refine. See
// docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md for the full rationale: li_init's
// EKF uses a flat, hardcoded per-point measurement weight (R_inv=1000 for
// every point regardless of range/incidence/plane-fit quality) and
// unvalidated process-noise constants, which this investigation traced to a
// persistent ~9.9-10Hz covariance-ringing artifact in its own trajectory
// output -- present even against a frozen, independently-built map, in both
// offline and live-replay modes, and only partially damped (not removed) by
// ct_refine's own (also flat-weighted) batch spline smoothing.
//
// This tool is PHASE 1 of the restructure plan: produce a seed trajectory
// using livo_recon's own preprocessing/deskew/VoxelMap machinery, weighted
// by VoxelPlane::gate()'s adaptive sigma_diag_squared + plane_var_term
// (not a flat constant like li_init's). Originally this ran livo_recon's
// own LioProcIekf (a full iterated EKF) for the per-scan correction step --
// REMOVED (2026-10-29, same session, per explicit user objection raised
// twice): an EKF's correction step blends the LiDAR residual with a
// Kalman gain computed from the IMU's PROPAGATED COVARIANCE, which depends
// on imu/process_noise/motion/{acc,gyro}_scale -- the very constants this
// whole pipeline exists to calibrate. Using an EKF here would make the
// calibration circular in exactly the same way li_init's own EKF-based
// reference trajectory was. Replaced with calib::icpRegister() (src/calib/
// icp_odometry.cpp): plain point-to-plane Gauss-Newton against the
// VoxelMap, no Kalman gain, no prior covariance of any kind -- seeded from
// the IMU's own MEAN-propagated pose (ImuProc::processIMU()'s own
// state_->setPropagatedState() call), which is safe to use since the mean
// integration itself does not depend on acc_scale/gyro_scale (only their
// COVARIANCE does -- see icp_odometry.h's header comment for the full
// argument). Driven by a minimal standalone main() instead of the full ROS
// node -- per explicit instruction, this is its own executable, not a mode
// of livo_recon_node. Writes a TUM-format pose trajectory.
//
// PHASE 2: after Phase 1's LIO pass finishes (map built, every scan's
// deskewed/downsampled points collected along the way), runs a batch
// B-spline refinement of the trajectory against that now-frozen VoxelMap
// (src/calib/trajectory_refine.cpp -- ported from LiDAR_IMU_Init's
// ct_refine.cpp, but weighted by VoxelMap::findPlaneResidual()'s real
// per-point sigma_squared instead of ct_refine's single flat scalar, and
// associated against livo_recon's own map instead of a from-scratch
// kd-tree/PCA plane fit). Writes a second TUM-format pose file. This is
// the piece expected to actually remove the ringing artifact this whole
// investigation started from, rather than just damp it.
//
// Usage: rosrun livo_recon qscale_refine_livo_recon _common/offline_bag_path:=<bag>
//        _tools/qscale_refine/out_path:=<trajectory.txt> [standard livo_recon params]

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <unordered_map>

#include "livo_recon/node_context.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/data/data_queues.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/processing/cbk_processing.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/processing/lio_iekf.h"
#include "livo_recon/processing/calib_processing.h"
#include "livo_recon/processing/vio_processing.h"
#include "livo_recon/processing/combined_processing.h"
#include "livo_recon/calib/trajectory_refine.h"
#include "livo_recon/calib/icp_odometry.h"

using namespace livo_recon;

namespace {

// Writes one TUM-format row (t x y z qx qy qz qw) per scan -- the same
// convention qscale_fit.cpp's Poses::writePoses() consumer expects
// (adapted; li_init's own lidar_odom.txt uses a 3x3-rotation-matrix form
// instead, this tool uses quaternion TUM since that's livo_recon's own
// evo/GT convention elsewhere in this codebase -- a format-conversion step
// is needed before plugging into the as-is qscale_fit.cpp, noted in the
// restructure plan's migration checklist).
void writeTumPose(std::ofstream& out, double t, const V3D& pos, const M3D& rot) {
  Eigen::Quaterniond q(rot);
  out << std::setprecision(17) << t << " " << pos.x() << " " << pos.y() << " " << pos.z() << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}

// One point per voxel (random pick, deterministic seed) up to a cap --
// ported from LiDAR_IMU_Init's ct_refine.cpp selectPoints(), applied per
// scan so the Phase 2 batch solve's per-outer-iteration association pass
// (one findPlaneResidual() call per selected point) stays tractable across
// a long bag, rather than feeding it literally every deskewed point.
void selectPointsForRefine(const std::vector<PointXYZCov>& scan, double voxel, int cap, std::mt19937& rng,
                            std::vector<calib::TrajPoint>& out) {
  std::vector<size_t> order(scan.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);
  std::unordered_map<int64_t, size_t> vox;
  std::vector<size_t> pick;
  for (size_t i : order) {
    const V3D& q = scan[i].raw_body_point;
    const int64_t kx = static_cast<int64_t>(std::floor(q.x() / voxel)) + (1 << 20);
    const int64_t ky = static_cast<int64_t>(std::floor(q.y() / voxel)) + (1 << 20);
    const int64_t kz = static_cast<int64_t>(std::floor(q.z() / voxel)) + (1 << 20);
    const int64_t key = (kx << 42) | (ky << 21) | kz;
    if (vox.emplace(key, i).second) pick.push_back(i);
  }
  if (static_cast<int>(pick.size()) > cap) pick.resize(cap);
  for (size_t i : pick) {
    out.push_back(calib::TrajPoint{scan[i].raw_body_point, scan[i].sensor_cov, scan[i].t});
  }
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "qscale_refine");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  image_transport::ImageTransport it(nh);

  std::string bag_path, out_path;
  paramWarn<std::string>(pnh, "common/offline_bag_path", bag_path, "");
  paramWarn<std::string>(pnh, "tools/qscale_refine/out_path", out_path, "");
  double duration_secs = -1.0;
  paramWarn<double>(pnh, "tools/qscale_refine/duration_secs", duration_secs, -1.0);
  // Optional: dump the within-scan, pre-correction IMU-propagated poses
  // (MeasureGroup::poses, populated by ImuProc::processIMU() at the IMU's
  // native sample rate, ~200Hz) rather than just one post-correction pose
  // per scan (~10Hz, the mid360's native scan rate -- too low to resolve
  // an 8-12Hz ripple at all, unlike li_init's 40Hz split-scan output).
  // This is a supplementary diagnostic, not the tool's primary output.
  std::string highrate_out_path;
  paramWarn<std::string>(pnh, "tools/qscale_refine/highrate_out_path", highrate_out_path, "");

  // PHASE 2 (batch spline refinement, see this file's header comment).
  // Empty refine_out_path (default) skips Phase 2 entirely -- Phase 1's
  // own out_path is unaffected either way.
  std::string refine_out_path;
  paramWarn<std::string>(pnh, "tools/qscale_refine/refine_out_path", refine_out_path, "");
  // 2026-10-29: 0.05 (the original default) let the spline track the
  // acceleration-domain noise floor (real per-point LiDAR sensor noise,
  // massively amplified by double-differentiation) almost as freely as it
  // tracked genuine motion, leaving the acc Q-scale channel essentially
  // unresolved (LiDAR-noise floor share 88% of the fit's own variance) no
  // matter how the voxel map/point covariance were tuned. 0.1 was found
  // this session to be the knot spacing that denies the spline that
  // freedom while barely touching genuine (lower-frequency) motion: per-T
  // joint-fit consistency stays close to 1.0 there, but visibly degrades
  // at 0.2 (a sign of real motion attenuation starting, not just noise
  // removal) -- see docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md for the full
  // knot-spacing sweep. See mid360.yaml's imu/process_noise/motion doc
  // comment for the resulting acc_scale/gyro_scale values.
  double refine_knot = 0.1, refine_prior_rot = 0.05, refine_prior_pos = 0.10, refine_window_s = 25.0;
  double refine_sample_rate = 40.0, refine_voxel = 0.15;
  double refine_still_t0 = -1.0, refine_still_t1 = -1.0;
  int refine_outer = 6, refine_per_scan_cap = 1000;
  paramWarn<double>(pnh, "tools/qscale_refine/refine/knot", refine_knot, 0.1);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/prior_rot", refine_prior_rot, 0.05);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/prior_pos", refine_prior_pos, 0.10);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/window_s", refine_window_s, 25.0);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/sample_rate", refine_sample_rate, 40.0);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/voxel", refine_voxel, 0.15);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/still_t0", refine_still_t0, -1.0);
  paramWarn<double>(pnh, "tools/qscale_refine/refine/still_t1", refine_still_t1, -1.0);
  paramWarn<int>(pnh, "tools/qscale_refine/refine/outer", refine_outer, 6);
  paramWarn<int>(pnh, "tools/qscale_refine/refine/per_scan_cap", refine_per_scan_cap, 1000);
  std::string refine_debug_dump_path;
  paramWarn<std::string>(pnh, "tools/qscale_refine/refine/debug_dump_path", refine_debug_dump_path, "");
  // 2026-10-29 (acceleration-smoothing investigation): raw spline control
  // points, in LiDAR_IMU_Init's own pose-text format (t, 3x3 rotation row-
  // major, position) at uniform knot spacing refine_knot -- NOT resampled
  // poses. Lets qscale_fit's existing --control-points/--control-points-dt
  // flags (ct_refine's own splineAccel()) evaluate the spline's EXACT
  // analytic second derivative instead of finite-differencing the
  // resampled --poses output, which adds its own discretization noise on
  // top of whatever the spline fit's real uncertainty is. Same cumulative
  // cubic B-spline convention both tools already use, so no format
  // adaptation needed. Empty (default): no export, no behavior change.
  std::string refine_control_points_out_path;
  paramWarn<std::string>(pnh, "tools/qscale_refine/refine/control_points_out_path",
                         refine_control_points_out_path, "");

  // Visual (epipolar) correction, see trajectory_refine.h's
  // TrajectoryRefineOptions::Camera doc comment. Intrinsics/extrinsics are
  // NOT re-specified here -- read directly from ctx.state (the same
  // camera/* params a live run loads), so this can never drift from the
  // calibration a live run actually uses.
  bool refine_camera_enable = false;
  std::string refine_camera_track_cache_path;
  double refine_camera_weight_px = 1.0;
  int refine_camera_max_matches_per_pair = 64;
  paramWarn<bool>(pnh, "tools/qscale_refine/refine/camera/enable", refine_camera_enable, false);
  paramWarn<std::string>(pnh, "tools/qscale_refine/refine/camera/track_cache_path",
                         refine_camera_track_cache_path, "");
  paramWarn<double>(pnh, "tools/qscale_refine/refine/camera/weight_px", refine_camera_weight_px, 1.0);
  paramWarn<int>(pnh, "tools/qscale_refine/refine/camera/max_matches_per_pair",
                 refine_camera_max_matches_per_pair, 64);

  if (bag_path.empty() || out_path.empty()) {
    ROS_ERROR("qscale_refine: both common/offline_bag_path and "
              "tools/qscale_refine/out_path are required");
    return 1;
  }

  NodeContext ctx(nh, pnh, it);
  CbkProc cbk_proc(ctx);
  ImuProc imu_proc(ctx);
  LioProcIekf lio_proc(ctx);
  CalibProc calib_proc(ctx);
  // Instantiated and loadParameters()'d purely to CLAIM their config keys
  // (livo_recon's strict unclaimed-param scanner aborts the run otherwise,
  // see mid360.yaml's voxel_map/vio/run/log namespaces) -- never used for
  // actual processing below (this tool is LIO-only by construction, not by
  // runtime flag). Mirrors LivoReconNode::loadParameters()'s own call
  // order exactly (printer/state -> voxel_map -> vio_proc_ -> combined_proc_).
  VioProc vio_proc(ctx);
  CombinedProc combined_proc(ctx);
  bool auto_terminate_on_idle = false;
  double auto_terminate_idle_secs = 5.0;
  paramWarn<bool>(pnh, "run/auto_terminate_on_idle", auto_terminate_on_idle, false);
  paramWarn<double>(pnh, "run/auto_terminate_idle_secs", auto_terminate_idle_secs, 5.0);
  std::string profiler_export;
  paramWarn<std::string>(pnh, "log/profiler_path", profiler_export, "");
  ctx.profiler->setExportPath(profiler_export);

  ROS_INFO_STREAM(ctx.printer->loadParameters(pnh));
  ROS_INFO_STREAM(ctx.state->loadParameters(pnh));
  ROS_INFO_STREAM(ctx.voxel_map->loadParameters(pnh));
  ROS_INFO_STREAM(vio_proc.loadParameters(pnh));
  ROS_INFO_STREAM(combined_proc.loadParameters(pnh));
  ROS_INFO_STREAM(cbk_proc.loadParameters(pnh));
  ROS_INFO_STREAM(imu_proc.loadParameters(pnh));
  ROS_INFO_STREAM(lio_proc.loadParameters(pnh));
  // Root cause of an earlier NaN-pose bug: without this call,
  // CalibProcOptions::num_frames stays at its default-constructed 0, so
  // collectSamples()'s loop condition (groups_qualifying < num_frames) is
  // false on the very first call and it returns true vacuously (0 >= 0)
  // without ever consuming data -- calibration then "succeeds" on an
  // empty buffer (NaN bias, zeroed P0) instead of actually waiting.
  ROS_INFO_STREAM(calib_proc.loadParameters(pnh));

  const auto& cbk_opts = cbk_proc.opts();

  std::ofstream out(out_path);
  if (!out) {
    ROS_ERROR_STREAM("qscale_refine: cannot open output path " << out_path);
    return 1;
  }
  out << "# t x y z qx qy qz qw  (TUM format, livo_recon world frame, LIO-only)\n";

  std::ofstream highrate_out;
  if (!highrate_out_path.empty()) {
    highrate_out.open(highrate_out_path);
    if (!highrate_out) {
      ROS_ERROR_STREAM("qscale_refine: cannot open highrate_out_path " << highrate_out_path);
      return 1;
    }
    highrate_out << "# t pos_x pos_y pos_z vel_x vel_y vel_z  (pre-correction, IMU-propagated, native IMU rate)\n";
  }

  ROS_INFO_STREAM("qscale_refine: reading " << bag_path << " directly (no rosbag play, no real-time pacing)");

  // Required before any feedNextMessage()/DataQueues::ready() call: switches
  // ready()'s quiet-stream check from wall-clock to bag-time (see
  // LivoReconNode::runOffline()'s identical call and DataQueues::
  // lookahead_margin_s's doc comment) -- without this, ready() never
  // settles when messages are fed with no real-time pacing, and
  // CalibProc::collectSamples() silently calibrates on an empty/garbage
  // buffer instead of actually waiting for data.
  ctx.data_queues->setOfflineMode(true);

  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (const rosbag::BagException& e) {
    ROS_ERROR_STREAM("qscale_refine: cannot open " << bag_path << ": " << e.what());
    return 1;
  }

  // Must include the image topic too, even though this tool never uses VIO:
  // with vio/enable=true (mid360.yaml's default), CbkProc::syncMeasures()
  // waits for synchronized image data before completing a measure group --
  // omitting the image topic here made every popMeasureGroup() call starve
  // forever (0 poses written, confirmed empirically), not just skip VIO.
  std::vector<std::string> topics{cbk_opts.imu_topic, cbk_opts.lidar_topic, cbk_opts.image_topic};
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  rosbag::View::iterator bag_it = view.begin();
  const rosbag::View::iterator bag_end = view.end();
  const ros::Time bag_start_time = bag_it != bag_end ? bag_it->getTime() : ros::Time(0);

  auto feedNextMessage = [&]() -> bool {
    if (bag_it == bag_end) return false;
    if (duration_secs > 0.0 && (bag_it->getTime() - bag_start_time).toSec() > duration_secs) return false;
    const rosbag::MessageInstance& m = *bag_it;
    if (m.getTopic() == cbk_opts.imu_topic) {
      sensor_msgs::Imu::ConstPtr msg = m.instantiate<sensor_msgs::Imu>();
      if (msg) cbk_proc.imuCallback(msg);
    } else if (m.getTopic() == cbk_opts.image_topic) {
      sensor_msgs::Image::ConstPtr msg = m.instantiate<sensor_msgs::Image>();
      if (msg) cbk_proc.imageCallback(msg);
    } else if (m.getTopic() == cbk_opts.lidar_topic) {
      // mid360 (and every non-PCL lidar_type) uses the raw CustomMsg
      // callback -- see CbkProc::loadParameters()'s own live-subscription
      // dispatch and LivoReconNode::runOffline()'s identical branch (that
      // branch's comment documents a real bug this mirroring avoids: a
      // MID360 bag's /livox/lidar previously silently failed every single
      // m.instantiate<PointCloud2>() call when special-cased wrong).
      if (cbk_opts.lidar_type != LidarType::PCL) {
        livox_ros_driver::CustomMsg::ConstPtr msg = m.instantiate<livox_ros_driver::CustomMsg>();
        if (msg) cbk_proc.lidarCallbackMsg(msg);
      } else {
        sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (msg) cbk_proc.lidarCallbackPcl(msg);
      }
    }
    ++bag_it;
    return true;
  };

  // CALIB phase (gravity alignment / stationary bootstrap) -- same
  // structure as LivoReconNode::runOffline()'s own CALIB loop.
  while (ros::ok() && !ctx.measures->calib_done.get()) {
    if (!feedNextMessage()) break;
    // Must be called every iteration, not throttled: estimateFromBuffer()
    // is what actually drains DataQueues via collectSamples() -- a
    // logging throttle here (ROS_INFO_STREAM_THROTTLE) skips evaluating
    // its own argument outside the throttle window, which skipped calling
    // this function almost entirely and let feedNextMessage() silently
    // exhaust the whole bag before calibration ever ran.
    const std::string calib_msg = calib_proc.estimateFromBuffer();
    ROS_INFO_STREAM_THROTTLE(2.0, calib_msg);
  }
  if (!ctx.measures->calib_done.get()) {
    ROS_ERROR("qscale_refine: bag exhausted before calibration (gravity alignment/bootstrap) completed");
    return 1;
  }
  if (ctx.voxel_map->isEmpty()) ctx.voxel_map->bootstrap(calib_proc.getBootstrapObservations());

  // Main loop. No EKF correction, no VIO/combined, no publishing, no
  // tracker -- syncMeasures -> IMU propagate (mean only) + deskew ->
  // icpRegister() (pure point-to-plane Gauss-Newton, no Kalman gain/prior
  // -- see icp_odometry.h's header comment for why this replaced
  // LioProcIekf::processLIO() here) -> VoxelMap insert -> write this
  // scan's pose.
  const bool do_refine = !refine_out_path.empty();
  std::vector<calib::TrajPoint> refine_points;
  std::vector<calib::TrajPose> refine_init_poses;
  std::mt19937 refine_rng(2);
  bool have_prev_pose = false;
  double prev_abs_t = 0.0;
  V3D prev_pos = V3D::Zero();

  long n_scans = 0;
  while (ros::ok()) {
    while (cbk_proc.syncMeasures()) continue;
    MeasureGroup mg;
    bool popped_any = false;
    while (ctx.measures->popMeasureGroup(mg)) {
      popped_any = true;
      imu_proc.processIMU(mg);
      if (highrate_out.is_open()) {
        for (const auto& pose : mg.poses) {
          highrate_out << std::setprecision(17) << (pose.t + ctx.data_queues->start_time) << " "
                       << pose.pos.x() << " " << pose.pos.y() << " " << pose.pos.z() << " "
                       << pose.vel.x() << " " << pose.vel.y() << " " << pose.vel.z() << "\n";
        }
      }
      lio_proc.deskewAndDownsample(mg);
      // Phase 2 input collection: mg.points' raw_body_point/t/sensor_cov
      // are exactly what trajectory_refine.h's TrajPoint needs, pose-
      // independent (see PointXYZCov::raw_body_point's own doc comment).
      // selectPointsForRefine() appends TrajPoints with mg.points' own
      // frame-relative `t`; fix up only the tail it just added to the
      // tool's absolute time base (same conversion as abs_t below) --
      // earlier scans' already-converted entries must not shift twice.
      if (do_refine) {
        const size_t before = refine_points.size();
        selectPointsForRefine(mg.points, refine_voxel, refine_per_scan_cap, refine_rng, refine_points);
        for (size_t k = before; k < refine_points.size(); k++)
          refine_points[k].t += ctx.data_queues->start_time;
      }
      // ICP correction: seed from the IMU's own mean-propagated pose
      // (already written into ctx.state by processIMU()'s own
      // setPropagatedState() call above -- safe to use, see icp_odometry.h),
      // refine in place against the VoxelMap, then write the result back
      // as this scan's "current" state. A degenerate match (too few
      // points, e.g. right after bootstrap) leaves R/p at the IMU-only
      // guess, exactly the right fallback.
      M3D R = ctx.state->rot();
      V3D p = ctx.state->pos();
      calib::icpRegister(mg.points, *ctx.voxel_map, R, p);
      const double abs_t = mg.image.t + ctx.data_queues->start_time;
      // Velocity for the NEXT scan's IMU propagation: finite-difference of
      // consecutive ICP-corrected positions, not the raw IMU-integrated
      // velocity (which would drift unboundedly over 188s with no
      // correction feeding back into it, now that there is no EKF update
      // step doing that implicitly).
      V3D new_vel = ctx.state->vel();
      if (have_prev_pose) {
        const double dt = abs_t - prev_abs_t;
        if (dt > 1e-6) new_vel = (p - prev_pos) / dt;
      }
      ctx.state->setPropagatedState(R, p, new_vel);
      prev_pos = p;
      prev_abs_t = abs_t;
      have_prev_pose = true;

      if (!mg.map_inserted) ctx.voxel_map->updateMap(mg);
      writeTumPose(out, abs_t, ctx.state->pos(), ctx.state->rot());
      if (do_refine) refine_init_poses.push_back(calib::TrajPose{abs_t, ctx.state->rot(), ctx.state->pos()});
      ++n_scans;
    }
    if (!popped_any && !feedNextMessage()) break;
  }
  bag.close();
  out.close();
  if (highrate_out.is_open()) highrate_out.close();

  ROS_INFO_STREAM("qscale_refine: wrote " << n_scans << " poses to " << out_path
                  << " (LIO-only, Phase 1 -- see docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md for Phase 2,"
                     " the batch spline refinement against this run's VoxelMap)");

  if (!do_refine) return 0;

  if (refine_init_poses.size() < 4) {
    ROS_ERROR_STREAM("qscale_refine: Phase 2 needs at least 4 scans to seed the spline, got "
                      << refine_init_poses.size());
    return 1;
  }
  ROS_INFO_STREAM("qscale_refine: Phase 2 starting -- " << refine_points.size() << " points collected, "
                  << refine_init_poses.size() << " seed poses, map stats: " << ctx.voxel_map->statsString());

  calib::SplineTrajectory traj;
  traj.init(refine_init_poses, refine_init_poses.front().t, refine_init_poses.back().t, refine_knot);

  calib::TrajectoryRefineOptions ropt;
  ropt.knot = refine_knot;
  ropt.prior_rot = refine_prior_rot;
  ropt.prior_pos = refine_prior_pos;
  ropt.outer = refine_outer;
  ropt.window_s = refine_window_s;
  ropt.t0_still = refine_still_t0;
  ropt.t1_still = refine_still_t1;
  ropt.debug_dump_path = refine_debug_dump_path;
  ropt.verbose = true;
  ropt.camera.enable = refine_camera_enable;
  ropt.camera.track_cache_path = refine_camera_track_cache_path;
  ropt.camera.weight_px = refine_camera_weight_px;
  ropt.camera.max_matches_per_pair = refine_camera_max_matches_per_pair;
  ropt.camera.fx = ctx.state->cameraFx();
  ropt.camera.fy = ctx.state->cameraFy();
  ropt.camera.cx = ctx.state->cameraCx();
  ropt.camera.cy = ctx.state->cameraCy();
  ropt.camera.R_l2c = ctx.state->R_l2c();
  ropt.camera.t_l2c = ctx.state->t_l2c();

  const calib::TrajectoryRefineReport report = calib::refineTrajectory(refine_points, refine_init_poses,
                                                                         *ctx.voxel_map, traj, ropt);
  ROS_INFO_STREAM("qscale_refine: Phase 2 done, " << report.outer_done << " outer iterations, "
                  << report.seconds << "s");

  if (!refine_control_points_out_path.empty()) {
    std::ofstream cp_out(refine_control_points_out_path);
    if (!cp_out) {
      ROS_ERROR_STREAM("qscale_refine: cannot open refine_control_points_out_path "
                       << refine_control_points_out_path);
      return 1;
    }
    cp_out << std::setprecision(17);
    for (size_t k = 0; k < traj.size(); k++) {
      const double tk = traj.tStart() + (static_cast<double>(k) - 1.0) * traj.dt();
      const Eigen::Quaterniond q(traj.q_[k][0], traj.q_[k][1], traj.q_[k][2], traj.q_[k][3]);
      const M3D R = q.toRotationMatrix();
      cp_out << tk;
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) cp_out << " " << R(r, c);
      cp_out << " " << traj.p_[k][0] << " " << traj.p_[k][1] << " " << traj.p_[k][2] << "\n";
    }
    ROS_INFO_STREAM("qscale_refine: wrote " << traj.size() << " raw control points to "
                    << refine_control_points_out_path << " (knot dt=" << traj.dt() << "s)");
  }

  const std::vector<calib::TrajPose> refined = calib::sampleTrajectory(traj, refine_sample_rate);
  std::ofstream refine_out(refine_out_path);
  if (!refine_out) {
    ROS_ERROR_STREAM("qscale_refine: cannot open refine_out_path " << refine_out_path);
    return 1;
  }
  refine_out << "# t x y z qx qy qz qw  (TUM format, livo_recon world frame, Phase 2 batch spline refinement)\n";
  for (const auto& pose : refined) writeTumPose(refine_out, pose.t, pose.pos, pose.rot);
  refine_out.close();
  ROS_INFO_STREAM("qscale_refine: wrote " << refined.size() << " refined poses to " << refine_out_path);
  return 0;
}
