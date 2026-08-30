#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/common_ros.h"
#include "livo_recon/utils/lidar/lidar.h"

namespace livo_recon
{

enum class LidarType { AVIA, MID360, PCL };

struct CbkProcOptions
{
  std::string image_topic  = "/camera/image_raw";
  std::string lidar_topic  = "/points_raw";
  std::string imu_topic    = "/imu/data";
  std::string camera_frame = "camera_link";
  std::string lidar_frame  = "lidar_link";

  // Different IMU devices/drivers publish sensor_msgs/Imu's linear_acceleration
  // in different units despite the message's own convention being m/s^2 --
  // "g" (needs *G_m_s2 to become m/s^2, the long-standing assumption here)
  // or "m_s2" (already correct, no scaling). Similarly angular_velocity:
  // "rad_s" (message's own convention, no scaling) or "deg_s" (needs
  // *M_PI/180). A mismatch here doesn't fail loudly -- it just means the
  // propagated acceleration/rotation is off by a large constant factor,
  // which imu_processing.cpp's EKF then tries to explain away via a huge,
  // otherwise-implausible bias estimate (see /tmp/imu.txt's bias_acc/
  // bias_gyr traces if this is suspected).
  std::string imu_acc_type = "g";       // "g" | "m_s2"
  std::string imu_gyr_type = "rad_s";   // "rad_s" | "deg_s"
  int    queue_size  = 10;
  double blind_sqr   = 0.25;
  // Added to the raw lidar scan header timestamp before it's used as the
  // scan's reference time (and therefore before every point's absolute
  // timestamp = scan_time + per-point offset is derived) -- corrects a
  // fixed clock/latency skew between the lidar driver and the IMU/image
  // streams, if the dataset has one. Mirrors FAST-LIVO2's LIVMapper.cpp
  // (lidar_time_offset, applied identically: cur_head_time = header.stamp
  // + lidar_time_offset) -- e.g. its NTU_VIRAL.yaml uses -0.1s for this
  // exact dataset. Left at 0 here previously; no equivalent existed in
  // livo_recon at all until this was added specifically to test whether a
  // missing lidar/IMU time-sync correction explains a velocity-correlated
  // (not fixed-frame) position bias found diverging from FAST-LIVO2's own
  // trajectory during sustained fast motion (see git history/session notes
  // around the eee_01 t=73-85s Y-axis bias investigation).
  double lidar_time_offset = 0.0;
  // Keep only every point_filter_num-th point of each raw scan, applied at
  // ingestion before undistortion/voxel downsampling -- mirrors
  // FAST-LIVO2's preprocess/point_filter_num (Preprocess::process's
  // `i % point_filter_num != 0` skip). 1 = keep everything (no decimation).
  int point_filter_num = 1;

  // 2026-08-24: opt-in LIO shadow dry-run diagnostic (see LioProcOptions::
  // dry_run_point_filter_num, lio_processing.h, and MeasureGroup::
  // dry_run_lidar_points/dry_run_points, measures.h). 0 (default) = no-op,
  // zero cost -- the raw scan is decimated only once (at point_filter_num
  // above), exactly as before this feature existed. > 0 = decimate the
  // SAME raw scan a SECOND time at this point_filter_num and store it in
  // MeasureGroup::dry_run_lidar_points, purely for LioProc's own shadow
  // IEKF pass to consume -- never inserted into the voxel map, never
  // affects the real trajectory. Loaded from the SAME rosparam key as
  // LioProcOptions::dry_run_point_filter_num ("lio/dry_run_point_filter_
  // num") -- two Options structs reading one key, to avoid a
  // cbk_processing<->lio_processing header dependency for a single int.
  int dry_run_point_filter_num = 0;

  LidarType lidar_type = LidarType::MID360;
  // tag filter: reject point when (tag & tag_mask) >= tag_max_keep
  // (i.e. keep the point only when masked value is strictly less than tag_max_keep)
  uint8_t tag_mask     = 0x0C;  // MID360 default: confidence bits 3:2
  uint8_t tag_max_keep = 0x08;  // keep high (0x00) and medium (0x04); reject low/invalid

  // Diagnostic-only measurement-group/first-arrival fingerprint logging --
  // mirrors FAST-LIVO2's LIVMapper::sync_packages() [sync_debug] logging
  // (common/log_sync_debug), ported here to verify DataQueues::ready()/
  // streamSettled()'s margin-based acceptance logic is actually producing
  // deterministic measure groups across repeated runs of the same input,
  // not just assuming it. OFF by default -- zero behavioral effect, purely
  // an ofstream write on the callback/syncMeasures() path when disabled
  // (the check itself; no other logic changes). See debugLogSyncStage() /
  // logSyncDebugStage() in cbk_processing.cpp for the log format.
  bool log_sync_debug = false;
  std::string sync_debug_log_path = "/tmp/livo_recon_sync_debug.txt";

  // Generalized frame-rate subsampling -- ported from FAST-LIVO2's
  // preprocess/image_subsample_n (LIVMapper::img_cbk()). 1 (default)
  // processes every image frame, unchanged from prior behavior. N > 1
  // processes only every Nth frame (frame_counter % N == 0), dropping the
  // rest before any other work (tracker feed, queue push) happens on them.
  // Motivated by the 2026-08-12 sync investigation: a rotating lidar
  // publishing full scans at ~10Hz alongside a 40Hz camera means 3 of
  // every 4 image frames can never pair with a lidar scan anyway (see
  // DataQueues::ready()'s doc comment) -- image_subsample_n=4 on such a
  // dataset (e.g. HILTI slam_2022/2023) matches the camera to the lidar's
  // own natural cadence instead of feeding the tracker 3x more frames than
  // syncMeasures() can ever actually use. NTU_VIRAL's camera/lidar rates
  // don't have this mismatch, so it stays at 1 (no skipping) there.
  int image_subsample_n = 1;

  // Ground-truth topic ingestion (evo/gt_source=="topic", e.g. NTU_VIRAL's
  // live Leica stream) -- read here (duplicating EvoProc's own reads of the
  // same evo/* keys, same "cheaper than a public accessor" call as
  // Tracker's opts()-less duplicated reads, see cache_tracker_output.cpp)
  // so CbkProc can own the actual ROS subscription/bag dispatch alongside
  // image/lidar/imu, feeding DataQueues::pushGt() -- see gtCallback()'s
  // doc comment for why this couldn't stay owned by EvoProc once
  // runOffline() needed to dispatch it directly.
  bool        evo_enable    = false;
  std::string evo_gt_source = "topic";
  std::string evo_gt_topic  = "/leica/pose/relative";
};

class CbkProc
{
public:
  explicit CbkProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  void lidarCallbackMsg(const livox_ros_driver::CustomMsg::ConstPtr& msg);
  void lidarCallbackPcl(const sensor_msgs::PointCloud2::ConstPtr& msg);
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);
  // Public (not just the live subscriber's own callback) so
  // LivoReconNode::runOffline() can dispatch matching GT-topic bag
  // messages directly, the same way imageCallback() etc. are public for
  // that exact purpose -- runOffline() never runs a real ROS topic/
  // subscriber pipeline, so evo_gt_source=="topic" datasets would
  // otherwise never receive any GT data in offline mode. Only meaningful
  // when opts_.evo_enable && evo_gt_source=="topic"; a silent no-op
  // otherwise (mirrors feedImage()'s "pipeline not started, drop
  // silently" convention) -- just pushes into DataQueues::pushGt(), no
  // EvoProc-specific logic here at all.
  void gtCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  bool syncMeasures();

  const CbkProcOptions& opts() const;

private:
  DataQueuesPtr data_queues_;
  MeasuresPtr   measures_;
  ProfilerPtr   profiler_;
  TrackerPtr    tracker_;
  ros::NodeHandle&                    nh_;
  image_transport::ImageTransport&    it_;

  CbkProcOptions opts_;

  ros::Subscriber image_sub_;
  ros::Subscriber lidar_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber gt_sub_;

  // First-arrival diagnostic logging is one-shot per topic per process --
  // see opts_.log_sync_debug's docs.
  bool logged_first_lidar_arrival_ = false;
  bool logged_first_imu_arrival_   = false;
  bool logged_first_img_arrival_   = false;
};

}  // namespace livo_recon
