#pragma once
// R82 (F-504): the per-point time read of a PointCloud2, as ROS-free code so tests/test_point_time_r82.cpp can check it without PCL or ROS.
// PointTimeSpec selects the field and how to read it. The default spec (field "t", mode OffsetNsU32, explicit_spec false) is the R81 behaviour bit for
// bit, including the silent zero offset when no "t" field exists and the read of a field named "t" as u32 nanoseconds whatever its declared datatype.
// With explicit_spec true a missing field or a datatype that does not match the mode throws (rule 3.4: a key that was set must act).
// AbsoluteSecF64 reads a FLOAT64 field of absolute (epoch) seconds, as the Hilti 2022 Hesai cloud carries ("timestamp"): the offset of a point is its value
// minus the reference (the message header stamp, or the smallest finite value of the message). The caller adds the offset to the scan time as before.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace livo_recon
{

struct PointTimeSpec
{
  enum class Mode { OffsetNsU32, AbsoluteSecF64 };
  enum class Ref { Header, MinPoint };
  std::string field = "t";
  Mode mode = Mode::OffsetNsU32;
  Ref ref = Ref::Header;
  bool explicit_spec = false;
};

// Per-call statistics of the point-time read (engagement line and per-message log).
struct PointTimeStats
{
  std::size_t n_points = 0, n_nonfinite = 0;
  double offset_min = 0.0, offset_max = 0.0;   // seconds, over finite offsets (both 0 when no point had one)
  bool field_found = false;
};

// sensor_msgs::PointField datatype codes (UINT32 = 6, FLOAT64 = 8 -- see
// sensor_msgs/msg/PointField.msg: INT8=1, UINT8=2, INT16=3, UINT16=4,
// INT32=5, UINT32=6, FLOAT32=7, FLOAT64=8. Confirmed against
// /opt/ros/*/share/sensor_msgs/msg/PointField.msg directly; the Hesai
// "timestamp" field genuinely reports datatype=8 (checked by reading an
// actual exp04 message's field list), so this was an off-by-one that
// made every cbk/lidar/point_time_mode=absolute_s_f64 run refuse at
// startup, which is exactly the R82 HIL "pt...on" arms.
constexpr uint8_t kPointFieldUint32 = 6;
constexpr uint8_t kPointFieldFloat64 = 8;

// `t_offset` < 0 means the field was not found. On return offsets[i] is the offset (s) of point i (0 when the field is absent or the value is not finite) and keep[i] is 0 only for a
// non-finite value. Throws std::runtime_error as described above.
inline void readPointTimeOffsets(const uint8_t* data, std::size_t n_points, std::size_t point_step, int t_offset, uint8_t datatype, const PointTimeSpec& spec,
                                 double header_stamp_s, std::vector<double>& offsets, std::vector<char>& keep, PointTimeStats& stats)
{
  const bool absolute = spec.mode == PointTimeSpec::Mode::AbsoluteSecF64;
  if (spec.explicit_spec)
  {
    if (t_offset < 0) throw std::runtime_error("cbk/lidar/point_time_field '" + spec.field + "' is not a field of the PointCloud2 message");
    const uint8_t want = absolute ? kPointFieldFloat64 : kPointFieldUint32;
    if (datatype != want)
      throw std::runtime_error("cbk/lidar/point_time_field '" + spec.field + "' has PointField datatype " + std::to_string(static_cast<int>(datatype)) + ", but point_time_mode " +
                               (absolute ? "absolute_s_f64 needs FLOAT64 (8)" : "offset_ns_u32 needs UINT32 (6)"));
  }
  offsets.assign(n_points, 0.0);
  keep.assign(n_points, 1);
  stats = PointTimeStats();
  stats.field_found = t_offset >= 0;
  if (t_offset < 0) return;

  double ref_value = header_stamp_s;
  if (absolute && spec.ref == PointTimeSpec::Ref::MinPoint)
  {
    double mn = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n_points; ++i)
    {
      double v = 0.0;
      std::memcpy(&v, data + i * point_step + t_offset, sizeof(v));
      if (std::isfinite(v)) mn = std::min(mn, v);
    }
    if (std::isfinite(mn)) ref_value = mn;
  }
  double lo = std::numeric_limits<double>::infinity(), hi = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n_points; ++i)
  {
    double off;
    if (absolute)
    {
      double v = 0.0;
      std::memcpy(&v, data + i * point_step + t_offset, sizeof(v));
      off = v - ref_value;
    }
    else
    {
      uint32_t t_ns = 0;
      std::memcpy(&t_ns, data + i * point_step + t_offset, sizeof(t_ns));
      off = static_cast<double>(t_ns) * 1e-9;
    }
    ++stats.n_points;
    if (!std::isfinite(off)) { ++stats.n_nonfinite; keep[i] = 0; continue; }
    offsets[i] = off;
    lo = std::min(lo, off); hi = std::max(hi, off);
  }
  if (std::isfinite(lo)) { stats.offset_min = lo; stats.offset_max = hi; }
}

}  // namespace livo_recon
