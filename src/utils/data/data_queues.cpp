#include "livo_recon/utils/data/data_queues.h"

#include <algorithm>

namespace livo_recon
{

void DataQueues::setStartTime(const double timestamp)
{
  start_time = timestamp;
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
    // History (21-32): see docs/livo_recon_changelog.md#src-utils-data-data_queues.cpp-21
    std::lock_guard<std::mutex> lock(dry_run_lidar_mutex);
    for (auto& scan : dry_run_lidar_queue)
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

void DataQueues::pushImu(const ImuSample& msg)
{
  last_imu_arrival_ns.store(nowNs(), std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(imu_mutex);
  if (msg.t < latest_imu_time) return;
  latest_imu_time = msg.t;
  auto msg_cpy = msg;
  msg_cpy.t -= start_time;
  imu_queue.push_back(msg_cpy);
}

void DataQueues::pushLidar(std::vector<PointXYZT>&& msg)
{
  last_lidar_arrival_ns.store(nowNs(), std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(lidar_mutex);
  if (msg.front().t < latest_lidar_time) return;
  latest_lidar_time = msg.back().t;
  for (auto& pt : msg) pt.t -= start_time;
  lidar_queue.push_back(std::move(msg));
}

// See dry_run_lidar_queue's doc comment (data_queues.h) -- deliberately
// independent bookkeeping from pushLidar()/lidar_mutex above; does NOT
// touch last_lidar_arrival_ns (ready()'s margin logic is keyed on the
// PRIMARY stream only, this is a diagnostic-only side channel).
void DataQueues::pushDryRunLidar(std::vector<PointXYZT>&& msg)
{
  if (msg.empty()) return;
  std::lock_guard<std::mutex> lock(dry_run_lidar_mutex);
  if (msg.front().t < latest_dry_run_lidar_time) return;
  latest_dry_run_lidar_time = msg.back().t;
  for (auto& pt : msg) pt.t -= start_time;
  dry_run_lidar_queue.push_back(std::move(msg));
}

void DataQueues::pushImage(const ImageData& msg)
{
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
// rationale of the two branches.
bool streamSettled(double t, double latest_stream_time, int64_t last_arrival_ns,
                   double lookahead_margin_s, double quiet_margin_s)
{
  if (latest_stream_time >= t + lookahead_margin_s) return true;
  if (latest_stream_time < t) return false;
  const double quiet_for = (nowNs() - last_arrival_ns) * 1e-9;
  return quiet_for >= quiet_margin_s;
}
}  // namespace

bool DataQueues::ready()
{
  std::lock_guard<std::mutex> lock(image_mutex);
  if (image_queue.empty()) return false;
  const double timestamp = image_queue.front().t + start_time;

  if (!streamSettled(timestamp, latest_lidar_time, last_lidar_arrival_ns.load(std::memory_order_relaxed),
                     lookahead_margin_s, quiet_margin_s))
    return false;
  if (!streamSettled(timestamp, latest_imu_time, last_imu_arrival_ns.load(std::memory_order_relaxed),
                     lookahead_margin_s, quiet_margin_s))
    return false;
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

// Mirrors popLidar() exactly, against dry_run_lidar_queue/dry_run_lidar_
// mutex instead -- see that queue's doc comment (data_queues.h).
bool DataQueues::popDryRunLidar(std::vector<PointXYZT>& out, double max_time)
{
  std::lock_guard<std::mutex> lock(dry_run_lidar_mutex);
  while (!dry_run_lidar_queue.empty())
  {
    auto& scan = dry_run_lidar_queue.front();
    if (scan.front().t > max_time) break;
    if (scan.back().t <= max_time)
    {
      out.insert(out.end(),
                 std::make_move_iterator(scan.begin()),
                 std::make_move_iterator(scan.end()));
      dry_run_lidar_queue.pop_front();
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
