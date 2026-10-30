#include "livo_recon/utils/lidar/lidar.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace livo_recon
{

namespace
{

// Shared per-point filter/build loop for both lidar message types: for each
// of `n` points, `access(i)` returns {position, offset_sec, keep}; a point
// survives if `i % point_filter_num == 0`, `keep` is true, its position is
// finite, it clears the blind radius, and its absolute timestamp
// (scan_time + offset_sec) is valid.
template <typename AccessFn>
std::vector<PointXYZT> buildLidarPoints(size_t n, double scan_time, double blind_sqr,
                                        const MeasuresPtr& measures, int point_filter_num,
                                        AccessFn&& access)
{
  std::vector<PointXYZT> points;
  points.reserve(n / std::max(point_filter_num, 1) + 1);

  for (size_t i = 0; i < n; ++i)
  {
    // access(i) always runs, even on a decimated-out point -- some
    // callers (pointsFromLivoxMsg) carry per-iteration state across calls
    // (smoothed cumulative offset_time), so skipping the call itself would
    // desync that state from the raw scan rather than just thinning the
    // output.
    const auto [pos, offset_sec, keep] = access(i);
    if (point_filter_num > 1 && (i % point_filter_num) != 0) continue;
    if (!keep) continue;
    if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) continue;
    if (pos.squaredNorm() < blind_sqr) continue;

    const double timestamp = scan_time + offset_sec;
    if (!measures->isValid(timestamp)) continue;

    points.emplace_back(pos, timestamp);
  }

  return points;
}

}  // namespace

std::vector<PointXYZT> pointsFromLivoxMsg(
    const livox_ros_driver::CustomMsg& msg, double scan_time,
    uint8_t tag_mask, uint8_t tag_max_keep,
    double blind_sqr, const MeasuresPtr& measures,
    int point_filter_num, double point_rate_hz)
{
  const double point_period_s = 1.0 / point_rate_hz;
  double prev_offset = 0.;
  return buildLidarPoints(
      msg.point_num, scan_time, blind_sqr, measures, point_filter_num,
      [&](size_t i) {
        const auto& pt = msg.points[i];

        const double raw_offset = static_cast<double>(pt.offset_time) * 1e-9;
        double diff = raw_offset - prev_offset;
        if (diff > point_period_s) diff = point_period_s;
        const double offset = (prev_offset += diff);

        const bool keep = (pt.tag & tag_mask) < tag_max_keep;
        return std::make_tuple(V3D(pt.x, pt.y, pt.z), offset, keep);
      });
}

std::vector<PointXYZT> pointsFromPointCloud2(
    const sensor_msgs::PointCloud2& msg, double scan_time,
    double blind_sqr, const MeasuresPtr& measures,
    int point_filter_num, const PointTimeSpec& spec, PointTimeStats* stats)
{
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::fromROSMsg(msg, cloud);

  int t_offset = -1;
  uint8_t t_datatype = 0;
  for (const auto& field : msg.fields)
    if (field.name == spec.field) { t_offset = static_cast<int>(field.offset); t_datatype = field.datatype; break; }

  std::vector<double> offsets;
  std::vector<char> keep_pt;
  PointTimeStats local;
  readPointTimeOffsets(msg.data.data(), cloud.size(), msg.point_step, t_offset, t_datatype, spec, msg.header.stamp.toSec(), offsets, keep_pt, local);
  if (stats) *stats = local;

  std::vector<PointXYZT> points = buildLidarPoints(
      cloud.size(), scan_time, blind_sqr, measures, point_filter_num,
      [&](size_t i) {
        const auto& pt = cloud[i];
        return std::make_tuple(V3D(pt.x, pt.y, pt.z), offsets[i], keep_pt[i] != 0);
      });

  // An organized Ouster cloud is laid out ring-major (each of the 16 rings
  // repeats the same azimuth/time sweep in raw buffer order), so the "t"
  // field is not monotonic in iteration order even though it is per-point
  // accurate. Downstream consumers (DataQueues::popLidar's binary-search
  // scan split) require points sorted ascending by timestamp.
  if (t_offset >= 0)
    std::sort(points.begin(), points.end(),
              [](const PointXYZT& a, const PointXYZT& b) { return a.t < b.t; });

  return points;
}

}  // namespace livo_recon
