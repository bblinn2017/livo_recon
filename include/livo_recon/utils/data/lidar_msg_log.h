#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>

// ============================================================================
// R80 (Bryce 2026-10-05, F-389 / F-311: "I'd like to be entirely certain about why we're missing scans before we change our code"):
// the per-message LiDAR log. WRITE-ONLY: it reads the message and the queue's state, writes one CSV row, changes nothing.
//
// Why it exists. DataQueues::pushLidar() discards a whole LiDAR message when the time of its first point is earlier than the time of the last point of the
// previously ACCEPTED message (data_queues.cpp, `if (msg.front().t < latest_lidar_time) return;`). R79 inferred from header stamps alone that this removes about
// 9.7 percent of the eee_01 messages; it ASSUMED a sweep span of 0.0999 s and a first point at the header time. This log measures both, per message, and records
// the verdict of the real code (pushLidar's return value), so the inference can be checked row by row:
//   off_first_s  = time of the first point minus scan_time (scan_time = the header stamp); the first point may be delayed
//   off_last_s   = time of the last point minus scan_time (the header-to-last-point time Bryce asked for)
//   span_s       = off_last_s - off_first_s
//   prev_accepted_last_t = DataQueues::latest_lidar_time just before this push (the last point time of the last accepted message)
//   overlap_s    = prev_accepted_last_t - t_first (positive = this message starts before the previous accepted one ended)
//   verdict      = 0 accepted, 1 dropped by pushLidar (overlap), 2 dropped by the callback (scan older than the bag start / the measure-group check: Measures::isValid), 3 no points left after the
//                  blind / tag / point_filter_num filters (nothing pushed)
// Header-only, ROS-free (unit-tested in tests/test_lidar_msg_log_r80.cpp). The rule below is the SAME comparison pushLidar() makes; pushLidar() calls it, so
// the test checks the rule the queue really uses.
// ============================================================================

namespace livo_recon
{

enum class LidarMsgVerdict : int { Accepted = 0, DroppedOverlap = 1, DroppedOld = 2, NoPoints = 3 };

// True when a message whose first point is at `first_point_t` must be discarded because the last accepted message ended at `latest_accepted_last_t` (seconds, same clock).
inline bool lidarMessageOverlapsPrevious(double first_point_t, double latest_accepted_last_t)
{
  return first_point_t < latest_accepted_last_t;
}

struct LidarMsgRow
{
  long long msg_index = 0;              // 0-based count of messages seen by the callback
  unsigned seq = 0;                     // header.seq as published by the driver (gaps here are messages the bag never had)
  double header_stamp = 0.0;            // header.stamp.toSec()
  double scan_time = 0.0;               // the header stamp (R97)
  double prev_header_stamp = std::numeric_limits<double>::quiet_NaN();   // header stamp of the previous message seen (accepted or not)
  long long n_raw = 0;                  // width * height of the message
  long long n_kept = 0;                 // points after the filters (0 when not built)
  double t_first = std::numeric_limits<double>::quiet_NaN();   // absolute time of the first point (NaN when not built)
  double t_last = std::numeric_limits<double>::quiet_NaN();    // absolute time of the last point
  double prev_accepted_last_t = std::numeric_limits<double>::quiet_NaN();   // queue's latest_lidar_time before this push (0 before the first accepted message)
  LidarMsgVerdict verdict = LidarMsgVerdict::Accepted;
};

inline const char* lidarMsgLogHeader()
{
  return "msg_index,seq,header_stamp,scan_time,gap_header_s,n_raw,n_kept,t_first,t_last,off_first_s,off_last_s,span_s,prev_accepted_last_t,overlap_s,verdict\n";
}

inline std::string lidarMsgFmt(double v, const char* f)
{
  if (!std::isfinite(v)) return "nan";
  char b[64];
  std::snprintf(b, sizeof(b), f, v);
  return b;
}

// One CSV row (with the newline). Absolute times with 6 decimals (microseconds; the clock is about 1.6e9 s so a double keeps ~1e-7 s), offsets and gaps with 9.
inline std::string formatLidarMsgRow(const LidarMsgRow& r)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const bool built = std::isfinite(r.t_first) && std::isfinite(r.t_last);
  const double off_first = built ? r.t_first - r.scan_time : nan;
  const double off_last = built ? r.t_last - r.scan_time : nan;
  const double span = built ? r.t_last - r.t_first : nan;
  const bool has_prev = std::isfinite(r.prev_accepted_last_t) && r.prev_accepted_last_t > 0.0;   // latest_lidar_time starts at 0.0: no accepted message yet
  const double overlap = (built && has_prev) ? r.prev_accepted_last_t - r.t_first : nan;
  const double gap = std::isfinite(r.prev_header_stamp) ? r.header_stamp - r.prev_header_stamp : nan;
  std::string s;
  s += std::to_string(r.msg_index) + "," + std::to_string(r.seq) + ",";
  s += lidarMsgFmt(r.header_stamp, "%.6f") + "," + lidarMsgFmt(r.scan_time, "%.6f") + "," + lidarMsgFmt(gap, "%.9f") + ",";
  s += std::to_string(r.n_raw) + "," + std::to_string(r.n_kept) + ",";
  s += lidarMsgFmt(r.t_first, "%.6f") + "," + lidarMsgFmt(r.t_last, "%.6f") + ",";
  s += lidarMsgFmt(off_first, "%.9f") + "," + lidarMsgFmt(off_last, "%.9f") + "," + lidarMsgFmt(span, "%.9f") + ",";
  s += lidarMsgFmt(has_prev ? r.prev_accepted_last_t : nan, "%.6f") + "," + lidarMsgFmt(overlap, "%.9f") + ",";
  s += std::to_string(static_cast<int>(r.verdict)) + "\n";
  return s;
}

}  // namespace livo_recon
