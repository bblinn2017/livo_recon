#pragma once
// R74 (F-422): time since bag start. `t_abs` (seconds since the epoch, about 1.6e9) loses resolution when it is printed with few
// significant digits; `t_rel_s` = scan time minus the bag start time (DataQueues::start_time) is small and is written with 6 decimals.
// ROS-free so tests/test_bag_clock.cpp can check it. The bag start is set once, from DataQueues::setStartTime().
#include <cmath>
#include <cstdio>
#include <string>

namespace livo_recon
{
struct BagClockState
{
  double start = 0.0;
  bool set = false;
};
inline BagClockState& bagClockState()
{
  static BagClockState s;
  return s;
}
inline void setBagStart(double t_abs_start)
{
  bagClockState().start = t_abs_start;
  bagClockState().set = true;
}
inline bool bagStartIsSet() { return bagClockState().set; }
inline double bagStart() { return bagClockState().start; }

// Pure formatting: t_rel_s with exactly 6 decimals (fixed). Not finite -> "nan".
inline std::string formatTRel(double t_abs, double bag_start)
{
  const double v = t_abs - bag_start;
  if (!std::isfinite(v)) return "nan";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", v);
  return std::string(buf);
}
// Uses the global bag start; "nan" while it is not set (a visible failure, never a silently wrong 0).
inline std::string tRelString(double t_abs) { return bagStartIsSet() ? formatTRel(t_abs, bagStart()) : std::string("nan"); }
}  // namespace livo_recon
