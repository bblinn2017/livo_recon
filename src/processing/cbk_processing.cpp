#include "livo_recon/processing/cbk_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/data/lidar_msg_log.h"
#include "livo_recon/vio/tracker.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <random>
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
  // CQ-35/36: opened once (truncating), kept open for the process lifetime --
  // every call site passes the same opts_.sync_debug_log_path (a config-
  // loaded, stable string, unlike debugLogPath()'s mutable global -- see
  // that function's PersistentLogStream doc comment for the regression this
  // avoided here), so a single function-local static is correct. Flushed
  // after every write so a tail -f during a run sees data promptly.
  static std::ofstream ofs(path, std::ios::trunc);
  ofs << msg << "\n";
  ofs.flush();
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
  paramWarn<int>(pnh, "topics/queue_size", opts_.queue_size,   10);
  paramWarn<double>(pnh, "topics/lookahead_margin_s", data_queues_->lookahead_margin_s, 0.1);
  paramWarn<double>(pnh, "topics/quiet_margin_s",     data_queues_->quiet_margin_s,     0.05);

  double blind;
  paramWarn<double>(pnh, "cbk/lidar/blind", blind, 0.5);
  opts_.blind_sqr = blind * blind;

  paramWarn<int>(pnh, "cbk/lidar/point_filter_num", opts_.point_filter_num, 1);
  paramWarn<double>(pnh, "cbk/lidar/time_offset", opts_.lidar_time_offset, 0.0);
  // R80C (F-484): lidar overlap policy and its grid (see CbkProcOptions::lidar_overlap_policy). Sub-options are refused unless the mode needs them.
  pnh.param<std::string>("cbk/lidar/overlap_policy", opts_.lidar_overlap_policy, "discard");   // plain param, not paramWarn: the default path prints exactly what R80 printed
  markParamConsumed("cbk/lidar/overlap_policy");   // R80C agent fix: plain pnh.param() does not register with checkAllParamsConsumed() the way paramWarn() does
  if (opts_.lidar_overlap_policy != "discard" && opts_.lidar_overlap_policy != "retime" && opts_.lidar_overlap_policy != "retime_trim" && opts_.lidar_overlap_policy != "restamp")   // R80D: restamp added
    throw std::invalid_argument("cbk/lidar/overlap_policy must be discard, retime, retime_trim or restamp (got '" + opts_.lidar_overlap_policy + "')");
  const bool retime_on = (opts_.lidar_overlap_policy != "discard");
  const bool retime_keys_given = pnh.hasParam("cbk/lidar/retime_t0_s") || pnh.hasParam("cbk/lidar/retime_period_s") || pnh.hasParam("cbk/lidar/retime_seq0") || pnh.hasParam("cbk/lidar/retime_ref_offset_s");   // R80D: retime_ref_offset_s added
  if (!retime_on && retime_keys_given)
    throw std::invalid_argument("cbk/lidar/retime_t0_s, retime_period_s, retime_seq0 and retime_ref_offset_s are only valid with cbk/lidar/overlap_policy = retime, retime_trim or restamp");
  opts_.lidar_retime_ref_offset_s = opts_.lidar_time_offset;   // R80D: default, so the offset term is exactly 0.0 unless the key says otherwise
  if (retime_on)
  {
    pnh.param<double>("cbk/lidar/retime_t0_s", opts_.lidar_retime_t0_s, 0.0);
    pnh.param<double>("cbk/lidar/retime_period_s", opts_.lidar_retime_period_s, 0.0);
    pnh.param<int>("cbk/lidar/retime_seq0", opts_.lidar_retime_seq0, -1);
    markParamConsumed("cbk/lidar/retime_t0_s");       // R80C agent fix: see overlap_policy's markParamConsumed above
    markParamConsumed("cbk/lidar/retime_period_s");
    markParamConsumed("cbk/lidar/retime_seq0");
    pnh.param<double>("cbk/lidar/retime_ref_offset_s", opts_.lidar_retime_ref_offset_s, opts_.lidar_time_offset);   // R80D
    markParamConsumed("cbk/lidar/retime_ref_offset_s");
    if (!(opts_.lidar_retime_t0_s > 1.0e8) || !(opts_.lidar_retime_period_s > 0.09 && opts_.lidar_retime_period_s < 0.11) || opts_.lidar_retime_seq0 < 0)
      throw std::invalid_argument("cbk/lidar/overlap_policy = " + opts_.lidar_overlap_policy +
                                  " needs cbk/lidar/retime_t0_s (> 1e8 s), cbk/lidar/retime_period_s (0.09 to 0.11 s) and cbk/lidar/retime_seq0 (>= 0)");
  }
  // R82 (F-504): per-point time field. Plain params (not paramWarn) so the default path prints exactly what R81 printed; the engagement line below appears only when a key is set.
  {
    opts_.point_time_explicit = pnh.hasParam("cbk/lidar/point_time_field") || pnh.hasParam("cbk/lidar/point_time_mode") || pnh.hasParam("cbk/lidar/point_time_ref");
    pnh.param<std::string>("cbk/lidar/point_time_field", opts_.point_time_field, "t");
    pnh.param<std::string>("cbk/lidar/point_time_mode", opts_.point_time_mode, "offset_ns_u32");
    pnh.param<std::string>("cbk/lidar/point_time_ref", opts_.point_time_ref, "header");
    markParamConsumed("cbk/lidar/point_time_field");
    markParamConsumed("cbk/lidar/point_time_mode");
    markParamConsumed("cbk/lidar/point_time_ref");
    if (opts_.point_time_mode != "offset_ns_u32" && opts_.point_time_mode != "absolute_s_f64")
      throw std::invalid_argument("cbk/lidar/point_time_mode must be offset_ns_u32 or absolute_s_f64 (got '" + opts_.point_time_mode + "')");
    if (opts_.point_time_ref != "header" && opts_.point_time_ref != "min_point")
      throw std::invalid_argument("cbk/lidar/point_time_ref must be header or min_point (got '" + opts_.point_time_ref + "')");
    if (pnh.hasParam("cbk/lidar/point_time_ref") && opts_.point_time_mode != "absolute_s_f64")
      throw std::invalid_argument("cbk/lidar/point_time_ref is only valid with cbk/lidar/point_time_mode = absolute_s_f64");
    if (opts_.point_time_field.empty())
      throw std::invalid_argument("cbk/lidar/point_time_field must not be empty");
  }
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
  if (retime_on && !lidar_is_pcl)
    throw std::invalid_argument("cbk/lidar/overlap_policy = " + opts_.lidar_overlap_policy + " is implemented for the PointCloud2 callback only (cbk/lidar/type pcl)");

  if (opts_.point_time_explicit && !lidar_is_pcl)
    throw std::invalid_argument("cbk/lidar/point_time_field, point_time_mode and point_time_ref are implemented for the PointCloud2 callback only (cbk/lidar/type pcl)");
  paramWarn<std::string>(pnh, "cbk/imu/acc_type", opts_.imu_acc_type, "g");
  paramWarn<std::string>(pnh, "cbk/imu/gyr_type", opts_.imu_gyr_type, "rad_s");

  paramWarn<bool>(pnh, "data_queues/log_sync_debug", opts_.log_sync_debug, false);
  paramWarn<std::string>(pnh, "data_queues/sync_debug_log_path", opts_.sync_debug_log_path,
                          debugLogPath("livo_recon_sync_debug.txt"));
  paramWarn<bool>(pnh, "data_queues/log_raw_point_count", opts_.log_raw_point_count, false);
  paramWarn<bool>(pnh, "data_queues/log_sensor_stamps", opts_.log_sensor_stamps, false);   // R65
  paramWarn<bool>(pnh, "data_queues/log_lidar_msgs", opts_.log_lidar_msgs, false);           // R80 (F-389): lidar_msg_log.csv, write-only
  paramWarn<double>(pnh, "cbk/lidar/perturb_std_m", opts_.perturb_std_m, 0.0);               // R65
  paramWarn<int>(pnh, "cbk/lidar/perturb_seed", opts_.perturb_seed, 0);                       // R65
  if (!(opts_.perturb_std_m >= 0.0))
    throw std::invalid_argument("cbk/lidar/perturb_std_m must be >= 0");
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
      << "\n  cbk/lidar/blind:  " << blind
      << "\n  cbk/lidar/point_filter_num: " << opts_.point_filter_num
      << "\n  cbk/lidar/time_offset: " << opts_.lidar_time_offset
      << "\n  cbk/lidar/overlap_policy: " << opts_.lidar_overlap_policy                                   // R80C (F-484)
      << "\n  cbk/lidar/retime_t0_s: " << std::fixed << std::setprecision(9) << opts_.lidar_retime_t0_s
      << "\n  cbk/lidar/retime_period_s: " << opts_.lidar_retime_period_s << std::defaultfloat << std::setprecision(6)
      << "\n  cbk/lidar/retime_seq0: " << opts_.lidar_retime_seq0
      << "\n  cbk/lidar/retime_ref_offset_s: " << opts_.lidar_retime_ref_offset_s                         // R80D
      << (opts_.point_time_explicit ? "\n  cbk/lidar/point_time: field=" + opts_.point_time_field + " mode=" + opts_.point_time_mode + (opts_.point_time_mode == "absolute_s_f64" ? " ref=" + opts_.point_time_ref : std::string()) : std::string())   // R82 (F-504): printed only when a key is set
      << "\n  cbk/lidar/type:   " << lidar_type_str
      << "\n  cbk/imu/acc_type: " << opts_.imu_acc_type
      << "\n  cbk/imu/gyr_type: " << opts_.imu_gyr_type
      << "\n  data_queues/log_sync_debug: " << (opts_.log_sync_debug ? "true" : "false")
      << "\n  data_queues/sync_debug_log_path: " << opts_.sync_debug_log_path
      << "\n  data_queues/log_lidar_msgs: " << (opts_.log_lidar_msgs ? "true" : "false")   // R80 (F-389)
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

namespace
{
// R65: write-only stamp/count log (see CbkProcOptions::log_sensor_stamps).
void logSensorStamp(char kind, double stamp, std::size_t n_raw, std::size_t n_kept)
{
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  static PersistentLogStream log("sensor_stamps.csv");
  bool first;
  std::ofstream& ofs = log.stream(&first);
  if (first) ofs << "kind,stamp,n_raw,n_kept\n";
  ofs << kind << ',' << std::setprecision(17) << stamp << ',' << n_raw << ',' << n_kept << "\n";
  ofs.flush();
}

// R80 (F-389 / F-311): one row of lidar_msg_log.csv (see CbkProcOptions::log_lidar_msgs and lidar_msg_log.h). Called only from the PointCloud2 callback, which runs on one thread;
// the mutex is for the static state (message counter, previous header stamp), not for concurrency.
void logLidarMsg(unsigned seq, double header_stamp, double scan_time, std::size_t n_raw, std::size_t n_kept, double t_first, double t_last,
                 double prev_accepted_last_t, LidarMsgVerdict verdict, double retime_shift_s = 0.0, std::size_t n_trimmed = 0)
{
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  static PersistentLogStream log("lidar_msg_log.csv");
  static long long msg_index = 0;
  static double prev_header = std::numeric_limits<double>::quiet_NaN();
  bool first;
  std::ofstream& ofs = log.stream(&first);
  if (first) ofs << lidarMsgLogHeader();
  LidarMsgRow r;
  r.msg_index = msg_index++; r.seq = seq; r.header_stamp = header_stamp; r.scan_time = scan_time; r.prev_header_stamp = prev_header;
  r.n_raw = static_cast<long long>(n_raw); r.n_kept = static_cast<long long>(n_kept); r.t_first = t_first; r.t_last = t_last;
  r.prev_accepted_last_t = prev_accepted_last_t; r.verdict = verdict;
  r.retime_shift_s = retime_shift_s; r.n_trimmed = static_cast<long long>(n_trimmed);   // R80C
  prev_header = header_stamp;
  ofs << formatLidarMsgRow(r);
  ofs.flush();
}

// R65: seeded sub-noise perturbation of kept points (see CbkProcOptions::perturb_std_m).
void perturbPoints(std::vector<PointXYZT>& points, double scan_time, double std_m, int seed)
{
  if (!(std_m > 0.0)) return;
  const std::size_t stamp_bits = std::hash<double>()(scan_time);
  std::mt19937_64 rng(static_cast<std::uint64_t>(seed) * 0x9E3779B97F4A7C15ULL ^ static_cast<std::uint64_t>(stamp_bits));
  std::normal_distribution<double> nd(0.0, std_m);
  for (auto& pt : points)
    pt.p += V3D(nd(rng), nd(rng), nd(rng));
}
}  // namespace

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

  if (opts_.log_sensor_stamps) logSensorStamp('I', img_time, 0, 0);   // R65
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
  if (opts_.log_sensor_stamps) logSensorStamp('L', scan_time, static_cast<std::size_t>(msg->point_num), points.size());   // R65
  perturbPoints(points, scan_time, opts_.perturb_std_m, opts_.perturb_seed);                                               // R65

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
  const double header_scan_time = msg->header.stamp.toSec() + opts_.lidar_time_offset;
  // R80C (F-484): under retime / retime_trim / restamp (R80D) the scan time is the grid time of the message's seq (plus the R80D offset term, exactly 0.0 unless cbk/lidar/time_offset differs from retime_ref_offset_s), not its own header; the discard policy is exactly the R80 line above.
  const bool retime = (opts_.lidar_overlap_policy != "discard");
  const double scan_time = retime ? retimeScanTimeWithOffset(msg->header.seq, opts_.lidar_retime_seq0, opts_.lidar_retime_t0_s, opts_.lidar_retime_period_s, opts_.lidar_time_offset, opts_.lidar_retime_ref_offset_s) : header_scan_time;
  const double retime_shift = scan_time - header_scan_time;   // exactly 0.0 under discard
  {
    // engagement and magnitude (rule 3.3): counted per message, printed on the first message, every 500th, and an error if the policy is on and never moved a scan time
    static long long n_msgs = 0;
    static double sum_shift = 0.0, max_abs_shift = 0.0;
    ++n_msgs; sum_shift += retime_shift; max_abs_shift = std::max(max_abs_shift, std::fabs(retime_shift));
    if (retime && (n_msgs == 1 || n_msgs % 500 == 0))
      ROS_INFO_STREAM("[lidar_overlap_policy] policy=" << opts_.lidar_overlap_policy << " msgs=" << n_msgs << " mean_shift_ms=" << 1e3 * sum_shift / static_cast<double>(n_msgs)
                      << " max_abs_shift_ms=" << 1e3 * max_abs_shift);
    if (retime && n_msgs == 100 && max_abs_shift == 0.0)
      ROS_ERROR_STREAM("[lidar_overlap_policy] policy=" << opts_.lidar_overlap_policy << " is on but no scan time moved in 100 messages (inert when on)");
  }
  if (opts_.log_raw_point_count)
  {
    static PersistentLogStream raw_count_log("lidar_raw_point_count.txt");
    bool raw_count_first;
    std::ofstream& raw_count_ofs = raw_count_log.stream(&raw_count_first);
    if (raw_count_first) raw_count_ofs << "scan_time,n_points_raw\n";
    raw_count_ofs << std::setprecision(17) << scan_time << ","
                  << (static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height)) << "\n";
    raw_count_ofs.flush();
  }
  if (!measures_->isValid(scan_time))
  {
    ROS_WARN_STREAM("Received an old LiDAR scan. Ignoring. stamp=" << msg->header.stamp);
    if (opts_.log_lidar_msgs)   // R80 (F-389): write-only
      logLidarMsg(msg->header.seq, msg->header.stamp.toSec(), scan_time, static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height), 0,
                  std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), data_queues_->latestLidarTime(), LidarMsgVerdict::DroppedOld, retime_shift);
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

  PointTimeSpec pt_spec;   // R82 (F-504): the default spec is the R81 read
  pt_spec.field = opts_.point_time_field;
  pt_spec.mode = opts_.point_time_mode == "absolute_s_f64" ? PointTimeSpec::Mode::AbsoluteSecF64 : PointTimeSpec::Mode::OffsetNsU32;
  pt_spec.ref = opts_.point_time_ref == "min_point" ? PointTimeSpec::Ref::MinPoint : PointTimeSpec::Ref::Header;
  pt_spec.explicit_spec = opts_.point_time_explicit;
  PointTimeStats pt_stats;
  std::vector<PointXYZT> points = pointsFromPointCloud2(
      *msg, scan_time, opts_.blind_sqr, measures_, opts_.point_filter_num, pt_spec, &pt_stats);
  if (opts_.point_time_explicit)
  {
    static long long n_pt_msgs = 0;
    static double pt_off_min = std::numeric_limits<double>::infinity(), pt_off_max = -std::numeric_limits<double>::infinity();
    static long long pt_nonfinite = 0;
    ++n_pt_msgs; pt_nonfinite += static_cast<long long>(pt_stats.n_nonfinite);
    pt_off_min = std::min(pt_off_min, pt_stats.offset_min); pt_off_max = std::max(pt_off_max, pt_stats.offset_max);
    if (n_pt_msgs == 1 || n_pt_msgs % 500 == 0)
      ROS_INFO_STREAM("[lidar_point_time] field=" << opts_.point_time_field << " mode=" << opts_.point_time_mode << " msgs=" << n_pt_msgs << " offset_min_s=" << pt_off_min
                      << " offset_max_s=" << pt_off_max << " nonfinite_points=" << pt_nonfinite);
  }
  if (opts_.log_sensor_stamps)
    logSensorStamp('L', scan_time, static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height), points.size());   // R65
  perturbPoints(points, scan_time, opts_.perturb_std_m, opts_.perturb_seed);                                                     // R65

  // See lidarCallbackMsg()'s matching comment.
  // R86A (F-532): under retime_trim the second stream is trimmed against ITS OWN last queued point time, exactly as the primary stream is below. Before R86A it was pushed untrimmed, and
  // pushDryRunLidar then rejected the whole message wherever the primary stream only trimmed a few points (R86: 40 of 595 frames received a fraction of the second-stream points).
  if (opts_.dry_run_point_filter_num > 0)
  {
    std::vector<PointXYZT> dry_run_points = pointsFromPointCloud2(
        *msg, scan_time, opts_.blind_sqr, measures_, opts_.dry_run_point_filter_num, pt_spec);
    std::size_t dry_trimmed = 0;
    if (opts_.lidar_overlap_policy == "retime_trim" && !dry_run_points.empty())
      dry_trimmed = trimStreamOverlap(dry_run_points, data_queues_->latestDryRunLidarTime());
    const bool dry_nonempty = !dry_run_points.empty();
    const bool dry_pushed = dry_nonempty && data_queues_->pushDryRunLidar(std::move(dry_run_points));
    static long long dr_seen = 0, dr_trim_msgs = 0, dr_trim_pts = 0, dr_pushed = 0, dr_rejected = 0;
    ++dr_seen; if (dry_trimmed > 0) { ++dr_trim_msgs; dr_trim_pts += static_cast<long long>(dry_trimmed); }
    if (dry_pushed) ++dr_pushed; else if (dry_nonempty) ++dr_rejected;
    if (dr_seen == 1 || dr_seen % 100 == 0)
      ROS_INFO_STREAM("[dry_run_stream] msgs=" << dr_seen << " trimmed_msgs=" << dr_trim_msgs << " trimmed_points=" << dr_trim_pts << " pushed_msgs=" << dr_pushed << " rejected_msgs=" << dr_rejected);
  }

  // R80C (F-484), retime_trim only: the points earlier than the last point of the last accepted message are removed here, so the unchanged pushLidar() rule cannot fire. `points` is sorted
  // ascending by time (pointsFromPointCloud2 sorts it). A point at exactly the previous end stays (the pushLidar boundary: strictly earlier is the overlap).
  std::size_t n_trimmed = 0;
  if (opts_.lidar_overlap_policy == "retime_trim" && !points.empty())
  {
    n_trimmed = countPointsBefore(points, data_queues_->latestLidarTime());
    if (n_trimmed > 0) points.erase(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(n_trimmed));
    static long long n_trim_msgs = 0, n_trim_pts = 0, n_trim_seen = 0;
    ++n_trim_seen; if (n_trimmed > 0) { ++n_trim_msgs; n_trim_pts += static_cast<long long>(n_trimmed); }
    if (n_trim_seen == 1 || n_trim_seen % 500 == 0)
      ROS_INFO_STREAM("[lidar_overlap_policy] retime_trim msgs=" << n_trim_seen << " trimmed_msgs=" << n_trim_msgs << " trimmed_points=" << n_trim_pts);
  }

  if (points.empty())
  {
    if (opts_.log_lidar_msgs)   // R80 (F-389): write-only
      logLidarMsg(msg->header.seq, msg->header.stamp.toSec(), scan_time, static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height), 0,
                  std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), data_queues_->latestLidarTime(), LidarMsgVerdict::NoPoints, retime_shift, n_trimmed);
    return;
  }
  // R80 (F-389): the first / last point time and the queue's state are read BEFORE the push (the push moves the points and rewrites the times); the push itself is unchanged.
  const bool log_msg = opts_.log_lidar_msgs;
  const double msg_t_first = log_msg ? points.front().t : 0.0, msg_t_last = log_msg ? points.back().t : 0.0;
  const std::size_t msg_n_kept = log_msg ? points.size() : 0;
  const double prev_accepted_last_t = log_msg ? data_queues_->latestLidarTime() : 0.0;
  const bool accepted = data_queues_->pushLidar(std::move(points));
  if (log_msg)
    logLidarMsg(msg->header.seq, msg->header.stamp.toSec(), scan_time, static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height), msg_n_kept,
                msg_t_first, msg_t_last, prev_accepted_last_t, accepted ? LidarMsgVerdict::Accepted : LidarMsgVerdict::DroppedOverlap, retime_shift, n_trimmed);
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
