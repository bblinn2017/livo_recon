#include "livo_recon/utils/data/data_queues.h"

#include <algorithm>

#include "livo_recon/utils/data/lidar_msg_log.h"
#include "livo_recon/utils/log/bag_clock.h"

namespace livo_recon
{

void DataQueues::setStartTime(const double timestamp)
{
  start_time = timestamp;
  setBagStart(timestamp);   // R74 (F-422): t_rel_s = t_abs - start_time in every per-scan file
  {
    std::lock_guard<std::mutex> lock(imu_mutex);
    for (auto& imu : imu_queue) imu.t -= start_time;
  }
  {
    std::lock_guard<std::mutex> lock(lidar_mutex);
    for (auto& scan : lidar_queue)
      for (auto& pt : scan) pt.t -= start_time;
  }
  {
    std::lock_guard<std::mutex> lock(image_mutex);
    for (auto& img : image_queue) img.t -= start_time;
  }
  {
    std::lock_guard<std::mutex> lock(gt_mutex);
    for (auto& gt : gt_queue) gt.t -= start_time;
  }
}

namespace
{
int64_t nowNs()
{
  return std::chrono::steady_clock::now().time_since_epoch().count();
}
}  // namespace

// CQ-37: CAS loop matching voxelplane.cpp's g_max_plane_var_trace pattern --
// latest_fed_time only ever needs to track the MAXIMUM timestamp seen so
// far across all three streams, and multiple push{Imu,Lidar,Image} calls
// can race here under live playback's independent subscriber callbacks
// just as they already can on last_{lidar,imu}_arrival_ns.
void DataQueues::updateLatestFedTime(double t)
{
  double cur = latest_fed_time.load(std::memory_order_relaxed);
  while (t > cur &&
         !latest_fed_time.compare_exchange_weak(cur, t, std::memory_order_relaxed)) {}
}

void DataQueues::pushImu(const ImuSample& msg)
{
  last_imu_arrival_ns.store(nowNs(), std::memory_order_relaxed);
  updateLatestFedTime(msg.t);
  std::lock_guard<std::mutex> lock(imu_mutex);
  if (msg.t < latest_imu_time) return;
  latest_imu_time = msg.t;
  auto msg_cpy = msg;
  msg_cpy.t -= start_time;
  imu_queue.push_back(msg_cpy);
}

bool DataQueues::pushLidar(std::vector<PointXYZT>&& msg)
{
  last_lidar_arrival_ns.store(nowNs(), std::memory_order_relaxed);
  updateLatestFedTime(msg.back().t);
  std::lock_guard<std::mutex> lock(lidar_mutex);
  if (lidarMessageOverlapsPrevious(msg.front().t, latest_lidar_time)) return false;   // R80: the same comparison as before (msg.front().t < latest_lidar_time), named so a test can check it
  latest_lidar_time = msg.back().t;
  for (auto& pt : msg) pt.t -= start_time;
  lidar_queue.push_back(std::move(msg));
  return true;
}

double DataQueues::latestLidarTime()
{
  std::lock_guard<std::mutex> lock(lidar_mutex);
  return latest_lidar_time;
}

// R97A Q2: pushDryRunLidar()/latestDryRunLidarTime() are removed.

void DataQueues::pushImage(const ImageData& msg)
{
  updateLatestFedTime(msg.t);
  std::lock_guard<std::mutex> lock(image_mutex);
  if (msg.t < latest_image_time) return;
  latest_image_time = msg.t;
  auto msg_cpy = msg;
  msg_cpy.t -= start_time;
  image_queue.push_back(msg_cpy);
}

void DataQueues::pushGt(const GtPoseSample& msg)
{
  std::lock_guard<std::mutex> lock(gt_mutex);
  auto msg_cpy = msg;
  msg_cpy.t -= start_time;
  gt_queue.push_back(msg_cpy);
}

namespace
{
// Per-stream acceptance test against a candidate image timestamp `t` --
// see lookahead_margin_s/quiet_margin_s's docs on DataQueues for the
// rationale of the two branches. CQ-37: the quiet check (B) takes EITHER
// a real-time duration (live mode: nowNs() - last_arrival_ns) OR a
// bag-time duration (offline mode: latest_fed_time - latest_stream_time)
// depending on offline_mode -- see DataQueues::lookahead_margin_s's doc
// comment for why these are not interchangeable.
bool streamSettled(double t, double latest_stream_time, int64_t last_arrival_ns,
                   double lookahead_margin_s, double quiet_margin_s,
                   bool offline_mode, double latest_fed_time)
{
  if (latest_stream_time >= t + lookahead_margin_s) return true;
  if (latest_stream_time < t) return false;
  const double quiet_for = offline_mode
      ? (latest_fed_time - latest_stream_time)
      : (nowNs() - last_arrival_ns) * 1e-9;
  return quiet_for >= quiet_margin_s;
}
}  // namespace

bool DataQueues::ready()
{
  std::lock_guard<std::mutex> lock(image_mutex);
  if (image_queue.empty()) return false;
  const double timestamp = image_queue.front().t + start_time;
  const double fed_time = latest_fed_time.load(std::memory_order_relaxed);
  const bool lidar_settled = streamSettled(timestamp, latest_lidar_time,
      last_lidar_arrival_ns.load(std::memory_order_relaxed), lookahead_margin_s, quiet_margin_s,
      offline_mode, fed_time);
  const bool imu_settled = streamSettled(timestamp, latest_imu_time,
      last_imu_arrival_ns.load(std::memory_order_relaxed), lookahead_margin_s, quiet_margin_s,
      offline_mode, fed_time);
  if (!lidar_settled) return false;
  if (!imu_settled) return false;
  return true;
}

ImageData DataQueues::popImage()
{
  std::lock_guard<std::mutex> lock(image_mutex);
  auto out = image_queue.front();
  image_queue.pop_front();
  return out;
}

bool DataQueues::popLidar(std::vector<PointXYZT>& out, double max_time)
{
  std::lock_guard<std::mutex> lock(lidar_mutex);
  while (!lidar_queue.empty())
  {
    auto& scan = lidar_queue.front();
    if (scan.front().t > max_time) break;
    if (scan.back().t <= max_time)
    {
      out.insert(out.end(),
                 std::make_move_iterator(scan.begin()),
                 std::make_move_iterator(scan.end()));
      lidar_queue.pop_front();
      continue;
    }
    auto it = std::upper_bound(
      scan.begin(), scan.end(), max_time,
      [](double t, const PointXYZT& p) { return t < p.t; });
    out.insert(out.end(),
               std::make_move_iterator(scan.begin()),
               std::make_move_iterator(it));
    scan.erase(scan.begin(), it);
    break;
  }
  return !out.empty();
}

// R97A Q2: popDryRunLidar() is removed.

bool DataQueues::popImu(std::deque<ImuSample>& out, double max_time)
{
  std::lock_guard<std::mutex> lock(imu_mutex);
  while (!imu_queue.empty())
  {
    out.push_back(imu_queue.front());
    if (out.back().t >= max_time) break;
    imu_queue.pop_front();
  }
  return !out.empty();
}

std::vector<GtPoseSample> DataQueues::popGt()
{
  std::lock_guard<std::mutex> lock(gt_mutex);
  std::vector<GtPoseSample> out(std::make_move_iterator(gt_queue.begin()),
                                 std::make_move_iterator(gt_queue.end()));
  gt_queue.clear();
  return out;
}

}  // namespace livo_recon
