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
  // R97: the LiDAR scan time is the header stamp. The former lidar time offset (cbk/lidar/time_offset) and the R80C/R80D overlap policies (cbk/lidar/overlap_policy
  // discard | retime | retime_trim | restamp, with retime_t0_s, retime_period_s, retime_seq0, retime_ref_offset_s) were removed: the bags are rewritten once (scripts/r97_rewrite_bags.py) so
  // that the header is the start of the scan and no message overlaps the previous one. pushLidar() still discards an overlapping message (an overlap is now an anomaly; lidar_msg_log.csv shows it).
  // R82 (F-504, Bryce 2026-10-06). Which PointCloud2 field carries the per-point time and how it is read (see PointTimeSpec in utils/lidar/lidar.h). Defaults: field "t", mode offset_ns_u32 (u32
  // nanoseconds since scan start: the R81 behaviour bit for bit). Hilti 2022 (Hesai) carries "timestamp" as FLOAT64 absolute epoch seconds: point_time_mode = absolute_s_f64 with
  // point_time_field = timestamp. point_time_ref (header | min_point) is a sub-option of absolute_s_f64 only: the value that is subtracted (message header stamp, or the smallest finite value).
  // point_time_explicit is true when ANY of the three keys was set: then a missing field or a datatype that does not match the mode is an error. PointCloud2 callback only.
  std::string point_time_field = "t";
  std::string point_time_mode = "offset_ns_u32";
  std::string point_time_ref = "header";
  bool point_time_explicit = false;
  // Keep only every point_filter_num-th point of each raw scan, applied at
  // ingestion before undistortion/voxel downsampling -- mirrors
  // FAST-LIVO2's preprocess/point_filter_num (Preprocess::process's
  // `i % point_filter_num != 0` skip). 1 = keep everything (no decimation).
  int point_filter_num = 1;

  // R97A Q2: dry_run_point_filter_num is removed.

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

  // VoxelMap diagnostic campaign (2026-09-22, user request, Part 2/3):
  // n_points_raw genuinely does not exist anywhere in the pipeline today
  // -- point_filter_num's skip loop (lidar.cpp) discards the raw count
  // before it's ever recorded. Writes (scan_time, n_raw) to a debug log
  // directly in the callback -- a raw msg->width*msg->height read, no
  // queue/MeasureGroup plumbing touched (lower blast radius than adding
  // an accumulator into DataQueues, which is shared by every estimator
  // mode's timing-critical consolidation path). Joined to frame_stats.txt/
  // pose_pair.csv by nearest scan_time in post-processing, same pattern
  // already used for GT alignment elsewhere in this campaign. OFF by
  // default -- zero behavioral effect, purely an ofstream write.
  bool log_raw_point_count = false;

  // R65 (F-164/F-173). Write-only log sensor_stamps.csv (kind,stamp,n_raw,n_kept): one row per LiDAR message (n_raw = points in
  // the message, n_kept = points after the blind/tag/point_filter_num filters) and one row per image message (n_raw = n_kept
  // = 0). Off by default; no effect on the estimator.
  bool log_sensor_stamps = false;
  // R80 (F-389 / F-311). Write-only log lidar_msg_log.csv: one row per LiDAR message seen by the PointCloud2 callback (the NTU VIRAL path; the custom-message callback is not instrumented), with the header stamp, the
  // time of the first and of the last point, the last point time of the previously accepted message and the verdict of DataQueues::pushLidar() (include/livo_recon/utils/data/lidar_msg_log.h). Off by default;
  // no effect on the estimator (it reads the message and the queue's latest_lidar_time, and uses pushLidar()'s return value, which callers ignored before).
  bool log_lidar_msgs = false;
  // R65 (F-170 repeats). Sub-noise seeded perturbation of every kept LiDAR point (isotropic Gaussian, std in metres, per-scan
  // deterministic stream from perturb_seed and the scan stamp). 0 (default) = off, bit-identical. Used only to create
  // distinct-but-equivalent inputs for repeat runs, because identical inputs give bit-identical runs.
  double perturb_std_m = 0.0;
  int    perturb_seed  = 0;

  // History (86-98): see docs/livo_recon_changelog.md#include-livo_recon-processing-cbk_processing.h-86
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
