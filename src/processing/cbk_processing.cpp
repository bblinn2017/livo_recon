#include "livo_recon/processing/cbk_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include "livo_recon/vio/tracker.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace livo_recon
{

namespace
{

// Diagnostic-only measurement-group/first-arrival fingerprint log -- mirrors
// FAST-LIVO2's debugLogStage() (LIVMapper.cpp) / livo_recon's own
// debugLogEvo() (evo_processing.cpp): truncated on first call (per process),
// appended thereafter. Only ever called when opts_.log_sync_debug is true,
// so this has zero effect on a default/production run.
void debugLogSyncStage(const std::string& path, const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(path, first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// Wall-clock "now", seconds since epoch, matching FAST-LIVO2's [sync_debug]
// wall_now= field (there taken from ros::WallTime::now(); here
// system_clock, same epoch/units) -- used only for cross-run comparison of
// *when* things happened in real time, never fed into any actual logic.
double wallNowSec()
{
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

CbkProc::CbkProc(NodeContext& ctx)
  : data_queues_(ctx.data_queues), measures_(ctx.measures), profiler_(ctx.profiler),
    tracker_(ctx.tracker), nh_(ctx.nh), it_(ctx.it)
{}

std::string CbkProc::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<std::string>(pnh, "topics/image",      opts_.image_topic,  "/camera/image_raw");
  paramWarn<std::string>(pnh, "topics/lidar",      opts_.lidar_topic,  "/points_raw");
  paramWarn<std::string>(pnh, "topics/imu",        opts_.imu_topic,    "/imu/data");
  paramWarn<std::string>(pnh, "frames/camera",     opts_.camera_frame, "camera_link");
  paramWarn<std::string>(pnh, "frames/lidar",      opts_.lidar_frame,  "lidar_link");
  paramWarn<int>(pnh, "topics/queue_size", opts_.queue_size,   10);
  paramWarn<double>(pnh, "topics/lookahead_margin_s", data_queues_->lookahead_margin_s, 0.1);
  paramWarn<double>(pnh, "topics/quiet_margin_s",     data_queues_->quiet_margin_s,     0.05);

  double blind;
  paramWarn<double>(pnh, "cbk/lidar/blind", blind, 0.5);
  opts_.blind_sqr = blind * blind;

  paramWarn<int>(pnh, "cbk/lidar/point_filter_num", opts_.point_filter_num, 1);
  paramWarn<double>(pnh, "cbk/lidar/time_offset", opts_.lidar_time_offset, 0.0);
  // See CbkProcOptions::dry_run_point_filter_num's doc comment -- SAME
  // rosparam key LioProcOptions::dry_run_point_filter_num reads
  // ("lio/dry_run_point_filter_num"), loaded into both Options structs.
  paramWarn<int>(pnh, "lio/dry_run_point_filter_num", opts_.dry_run_point_filter_num, 0);

  std::string lidar_type_str;
  paramWarn<std::string>(pnh, "cbk/lidar/type", lidar_type_str, "mid360");
  if (lidar_type_str == "avia")
  {
    opts_.lidar_type   = LidarType::AVIA;
    opts_.tag_mask     = 0x30;  // bits 5:4: return type
    opts_.tag_max_keep = 0x20;  // keep single (0x00) and first (0x10); reject second (0x20)
  }
  else if (lidar_type_str == "mid360")
  {
    opts_.lidar_type   = LidarType::MID360;
    opts_.tag_mask     = 0x0C;  // bits 3:2: confidence level
    opts_.tag_max_keep = 0x08;  // keep high (0x00) and medium (0x04); reject low/invalid
  }
  else
  {
    opts_.lidar_type = LidarType::PCL;
  }
  const bool lidar_is_pcl = (opts_.lidar_type == LidarType::PCL);

  paramWarn<std::string>(pnh, "cbk/imu/acc_type", opts_.imu_acc_type, "g");
  paramWarn<std::string>(pnh, "cbk/imu/gyr_type", opts_.imu_gyr_type, "rad_s");

  paramWarn<bool>(pnh, "data_queues/log_sync_debug", opts_.log_sync_debug, false);
  paramWarn<std::string>(pnh, "data_queues/sync_debug_log_path", opts_.sync_debug_log_path,
                          debugLogPath("livo_recon_sync_debug.txt"));
  paramWarn<int>(pnh, "preprocess/image_subsample_n", opts_.image_subsample_n, 1);

  // Ground-truth topic ingestion -- see CbkProcOptions::evo_enable's doc
  // comment. Duplicated reads of evo/* keys EvoProc::loadParameters() also
  // reads independently; harmless since both just read the same rosparam
  // server, not a source of truth conflict.
  paramWarn<bool>(pnh, "evo/enable",           opts_.evo_enable,    false);
  paramWarn<std::string>(pnh, "evo/gt_source", opts_.evo_gt_source, "topic");
  paramWarn<std::string>(pnh, "evo/gt_topic",  opts_.evo_gt_topic,  "/leica/pose/relative");

  std::ostringstream oss;
  oss << "[params/cbk]"
      << "\n  topics/image:     " << opts_.image_topic
      << "\n  topics/lidar:     " << opts_.lidar_topic
      << "\n  topics/imu:       " << opts_.imu_topic
      << "\n  topics/queue_size:" << opts_.queue_size
      << "\n  topics/lookahead_margin_s: " << data_queues_->lookahead_margin_s
      << "\n  topics/quiet_margin_s:     " << data_queues_->quiet_margin_s
      << "\n  frames/camera:    " << opts_.camera_frame
      << "\n  frames/lidar:     " << opts_.lidar_frame
      << "\n  cbk/lidar/blind:  " << blind
      << "\n  cbk/lidar/point_filter_num: " << opts_.point_filter_num
      << "\n  cbk/lidar/time_offset: " << opts_.lidar_time_offset
      << "\n  cbk/lidar/type:   " << lidar_type_str
      << "\n  cbk/imu/acc_type: " << opts_.imu_acc_type
      << "\n  cbk/imu/gyr_type: " << opts_.imu_gyr_type
      << "\n  data_queues/log_sync_debug: " << (opts_.log_sync_debug ? "true" : "false")
      << "\n  data_queues/sync_debug_log_path: " << opts_.sync_debug_log_path
      << "\n  preprocess/image_subsample_n: " << opts_.image_subsample_n
      << "\n  evo/enable (gt ingestion): " << (opts_.evo_enable ? "true" : "false")
      << "\n  evo/gt_source: " << opts_.evo_gt_source
      << "\n  evo/gt_topic: " << opts_.evo_gt_topic;
  image_sub_ = nh_.subscribe(opts_.image_topic, opts_.queue_size, &CbkProc::imageCallback, this);
  if (lidar_is_pcl)
    lidar_sub_ = nh_.subscribe(opts_.lidar_topic, opts_.queue_size, &CbkProc::lidarCallbackPcl, this);
  else
    lidar_sub_ = nh_.subscribe(opts_.lidar_topic, opts_.queue_size, &CbkProc::lidarCallbackMsg, this);
  imu_sub_   = nh_.subscribe(opts_.imu_topic,   opts_.queue_size, &CbkProc::imuCallback,   this);
  if (opts_.evo_enable && opts_.evo_gt_source == "topic")
    gt_sub_ = nh_.subscribe(opts_.evo_gt_topic, opts_.queue_size, &CbkProc::gtCallback, this);
  return oss.str();
}

const CbkProcOptions& CbkProc::opts() const { return opts_; }

void CbkProc::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  // Generalized frame-rate subsampling -- ported from FAST-LIVO2's
  // preprocess/image_subsample_n (see CbkProcOptions::image_subsample_n's
  // doc comment). Checked first, before the TimedScope/profiler entry
  // below, so a skipped frame costs nothing beyond the counter increment.
  if (opts_.image_subsample_n > 1)
  {
    static int frame_counter = 0;
    if (++frame_counter % opts_.image_subsample_n != 0) return;
  }

  TimedScope ts(profiler_, "cbk/image");
  const double img_time = msg->header.stamp.toSec();
  if (!measures_->isValid(img_time))
  {
    ROS_WARN_STREAM("Received an old image. Ignoring. stamp=" << msg->header.stamp);
    return;
  }

  if (opts_.log_sync_debug && !logged_first_img_arrival_)
  {
    logged_first_img_arrival_ = true;
    std::ostringstream oss;
    oss << "FIRST_IMG_ARRIVAL bag_time=" << std::fixed << std::setprecision(9) << img_time
        << " wall_now=" << wallNowSec();
    debugLogSyncStage(opts_.sync_debug_log_path, oss.str());
  }

  const cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  if (cv_ptr->image.empty()) return;
  ImageData img_data{cv_ptr->image.clone(), img_time};

  // Feed the async tracker (see task #145 and Tracker's own doc comment)
  // the INSTANT this image arrives -- the earliest possible point, well
  // before it's even pushed onto data_queues_, let alone synced into a
  // real MeasureGroup. Silently no-ops if the pipeline was never started
  // (VIO disabled or tracker failed to load). Always fed 1:1 with what
  // gets pushed below, in the same order -- syncMeasures()'s consumeResult()
  // depends on that correspondence.
  tracker_->feedImage(img_data.image, img_data.t);

  data_queues_->pushImage(img_data);
}

void CbkProc::lidarCallbackMsg(const livox_ros_driver::CustomMsg::ConstPtr& msg)
{
  TimedScope ts(profiler_, "cbk/lidar");
  const double scan_time = msg->header.stamp.toSec() + opts_.lidar_time_offset;
  const double end_time = scan_time + static_cast<double>(msg->points.back().offset_time) * 1e-9;
  if (!measures_->isValid(end_time))
  {
    ROS_WARN_STREAM("Received an old LiDAR scan. Ignoring. stamp=" << msg->header.stamp);
    return;
  }

  if (opts_.log_sync_debug && !logged_first_lidar_arrival_)
  {
    logged_first_lidar_arrival_ = true;
    std::ostringstream oss;
    oss << "FIRST_LIDAR_ARRIVAL bag_time=" << std::fixed << std::setprecision(9) << scan_time
        << " wall_now=" << wallNowSec();
    debugLogSyncStage(opts_.sync_debug_log_path, oss.str());
  }

  std::vector<PointXYZT> points = pointsFromLivoxMsg(
      *msg, scan_time, opts_.tag_mask, opts_.tag_max_keep, opts_.blind_sqr, measures_,
      opts_.point_filter_num);

  // See CbkProcOptions::dry_run_point_filter_num's doc comment -- reuses
  // the SAME raw msg, decimated a second time at a different rate.
  // pointsFromLivoxMsg has no shared/global side effects (its per-call
  // cumulative-offset state is local to this invocation), so calling it
  // twice on the same msg is safe.
  if (opts_.dry_run_point_filter_num > 0)
  {
    std::vector<PointXYZT> dry_run_points = pointsFromLivoxMsg(
        *msg, scan_time, opts_.tag_mask, opts_.tag_max_keep, opts_.blind_sqr, measures_,
        opts_.dry_run_point_filter_num);
    if (!dry_run_points.empty())
      data_queues_->pushDryRunLidar(std::move(dry_run_points));
  }

  if (points.empty()) return;
  data_queues_->pushLidar(std::move(points));
}

void CbkProc::lidarCallbackPcl(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  TimedScope ts(profiler_, "cbk/lidar");
  const double scan_time = msg->header.stamp.toSec() + opts_.lidar_time_offset;
  if (!measures_->isValid(scan_time))
  {
    ROS_WARN_STREAM("Received an old LiDAR scan. Ignoring. stamp=" << msg->header.stamp);
    return;
  }

  if (opts_.log_sync_debug && !logged_first_lidar_arrival_)
  {
    logged_first_lidar_arrival_ = true;
    std::ostringstream oss;
    oss << "FIRST_LIDAR_ARRIVAL bag_time=" << std::fixed << std::setprecision(9) << scan_time
        << " wall_now=" << wallNowSec();
    debugLogSyncStage(opts_.sync_debug_log_path, oss.str());
  }

  std::vector<PointXYZT> points = pointsFromPointCloud2(
      *msg, scan_time, opts_.blind_sqr, measures_, opts_.point_filter_num);

  // See lidarCallbackMsg()'s matching comment.
  if (opts_.dry_run_point_filter_num > 0)
  {
    std::vector<PointXYZT> dry_run_points = pointsFromPointCloud2(
        *msg, scan_time, opts_.blind_sqr, measures_, opts_.dry_run_point_filter_num);
    if (!dry_run_points.empty())
      data_queues_->pushDryRunLidar(std::move(dry_run_points));
  }

  if (points.empty()) return;
  data_queues_->pushLidar(std::move(points));
}

void CbkProc::imuCallback(const sensor_msgs::ImuConstPtr& msg)
{
  TimedScope ts(profiler_, "cbk/imu");
  double imu_time = msg->header.stamp.toSec();
  if (!measures_->isValid(imu_time))
  {
    ROS_WARN_STREAM("Received an old IMU message. Ignoring. stamp=" << msg->header.stamp);
    return;
  }

  if (opts_.log_sync_debug && !logged_first_imu_arrival_)
  {
    logged_first_imu_arrival_ = true;
    std::ostringstream oss;
    oss << "FIRST_IMU_ARRIVAL bag_time=" << std::fixed << std::setprecision(9) << imu_time
        << " wall_now=" << wallNowSec();
    debugLogSyncStage(opts_.sync_debug_log_path, oss.str());
  }

  ImuSample sample;
  sample.t = imu_time;
  sample.acc = Eigen::Vector3d(
      msg->linear_acceleration.x,
      msg->linear_acceleration.y,
      msg->linear_acceleration.z);
  // "g": raw units are multiples of gravity, scale up to m/s^2 (the
  // long-standing assumption here, still correct for some devices, e.g.
  // this codebase's mid360). "m_s2": already correct (the sensor_msgs/Imu
  // message's own nominal convention, e.g. NTU VIRAL's IMU) -- no scaling.
  if (opts_.imu_acc_type != "m_s2")
    sample.acc *= G_m_s2;

  sample.gyro = Eigen::Vector3d(
      msg->angular_velocity.x,
      msg->angular_velocity.y,
      msg->angular_velocity.z);
  if (opts_.imu_gyr_type == "deg_s")
    sample.gyro *= M_PI / 180.0;

  data_queues_->pushImu(sample);
}

void CbkProc::gtCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  TimedScope ts(profiler_, "cbk/gt");
  // No isValid()/measures_ gate here (unlike image/lidar/imu) -- ground
  // truth doesn't participate in the calib/sync-measures pipeline at all,
  // it's independently timestamped and consumed by EvoProc via
  // DataQueues::popGt(). start_time-relative shift happens inside
  // DataQueues::pushGt() itself (same convention as pushImu/pushLidar/
  // pushImage), not here.
  GtPoseSample s;
  s.t = msg->header.stamp.toSec();
  s.pos = V3D(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  s.rot = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                            msg->pose.orientation.y, msg->pose.orientation.z)
              .toRotationMatrix();
  data_queues_->pushGt(s);
}

bool CbkProc::syncMeasures()
{
  TimedScope ts(profiler_, "sync_measures");
  if (!data_queues_->ready()) return false;

  // See task #145 and Tracker's own doc comment: the front image was
  // already fed to the async tracker the instant it arrived (see
  // imageCallback()), well before this call. Treat "tracking not done yet"
  // exactly like "lidar/imu not settled yet" above -- return false and let
  // the caller (LivoReconNode::run()) retry on its next outer iteration,
  // which already re-runs ros::spinOnce() first. No separate wait loop
  // needed anywhere. waitReadyFor(0ms) is a genuine instant, non-blocking
  // peek here (see its own doc comment), not a real wait. Skipped
  // entirely (never gates) when the pipeline was never started (VIO
  // disabled or tracker failed to load).
  if (tracker_->asyncTrackingActive() && !tracker_->waitReadyFor(std::chrono::milliseconds(0)))
    return false;

  ImageData img = data_queues_->popImage();

  bool success = true;

  std::vector<PointXYZT> points;
  success &= data_queues_->popLidar(points, img.t);

  // See CbkProcOptions::dry_run_point_filter_num's doc comment --
  // deliberately NOT folded into `success`: a missing/short dry-run scan
  // must never cause a real frame to be skipped, it only means this one
  // frame's shadow diagnostic pass has nothing to run on (LioProc checks
  // mg.dry_run_points.empty() itself before attempting it).
  std::vector<PointXYZT> dry_run_points;
  if (opts_.dry_run_point_filter_num > 0)
    data_queues_->popDryRunLidar(dry_run_points, img.t);

  std::deque<ImuSample> imu_samples;
  success &= data_queues_->popImu(imu_samples, img.t);

  measures_->curr_time.set(img.t);

  if (!success)
  {
    ROS_WARN_STREAM("Failed to sync measures at time " << img.t
      << ", skipping image frame.");
    if (opts_.log_sync_debug)
    {
      std::ostringstream oss;
      oss << "SYNC_FAILED img_time=" << std::fixed << std::setprecision(9) << img.t
          << " n_lidar=" << points.size() << " n_imu=" << imu_samples.size()
          << " wall_now=" << wallNowSec();
      debugLogSyncStage(opts_.sync_debug_log_path, oss.str());
    }
    return false;
  }

  if (opts_.log_sync_debug)
  {
    std::ostringstream oss;
    oss << "MEASURE_GROUP img_time=" << std::fixed << std::setprecision(9) << img.t
        << " n_imu=" << imu_samples.size()
        << " n_lidar=" << points.size()
        << " imu_t_first=" << (imu_samples.empty() ? 0.0 : imu_samples.front().t)
        << " imu_t_last=" << (imu_samples.empty() ? 0.0 : imu_samples.back().t)
        << " wall_now=" << wallNowSec();
    debugLogSyncStage(opts_.sync_debug_log_path, oss.str());
  }

  MeasureGroup mg{std::move(img), std::move(points), std::move(imu_samples)};
  mg.dry_run_lidar_points = std::move(dry_run_points);
  // Guaranteed ready (see the waitReadyFor(0ms) gate above) -- stored
  // directly on the MeasureGroup (task #145) so VioProc::processVIO()
  // just reads mg.tracked_frame later, no wait/consume call of its own.
  if (tracker_->asyncTrackingActive())
    tracker_->consumeResult(mg.tracked_frame);
  measures_->pushMeasureGroup(std::move(mg));

  return true;
}

}  // namespace livo_recon
