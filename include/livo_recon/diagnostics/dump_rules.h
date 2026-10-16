#pragma once

// R86B. The three decisions of the R86B write-only dumps, kept in one ROS-free header so a unit test can show each fires (tests/test_dump_rules_r86b.cpp).
//
//   preImuTraceWanted(frames, frame_idx): a "pre_imu" state_trace row is written for frame indices 1..frames. frames == 0 (default) writes none.
//   residualDumpIterationWanted(all_iters, iteration): the residual dump of a listed frame is written at iteration 0 only (default, as before R86B), or at every
//     iteration >= 0 when log/lio/residual_dump/all_iters is on. A negative iteration (no iteration context) is never dumped.
//   lastIterDumpWanted(last_iter, all_iters): the separate dump after the iteration loop (log/lio/residual_dump/last_iter) is suppressed when every iteration is dumped,
//     otherwise the last iteration would be written twice.

// R91C. The solve-snapshot decision and the frame-list parser of the R91C write-only dumps (tests/test_dump_rules_r91c.cpp).
//
//   firstFrameSolveDumpMode(frame_idx, k, list, slim): 0 = no solve snapshot for this frame, 1 = the full snapshot (frames 1..k, as before R91C), 2 = the slim snapshot (frame is in
//     log/lio/first_frame_dump/frame_list, not in 1..k, and log/lio/first_frame_dump/slim is on). A listed frame with slim off gets the full snapshot (mode 1).
//   parseFrameList(text, key): comma-separated positive integers, no repeats, at most kMaxDumpListFrames of them; anything else throws std::invalid_argument naming the key (rule 3.4: never silently ignored).

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace livo_recon
{
namespace dump_rules
{
constexpr int kMaxDumpListFrames = 200;

inline std::vector<int> parseFrameList(const std::string& text, const std::string& key)
{
  std::vector<int> out;
  std::string tok;
  auto flush = [&]() {
    if (tok.empty()) return;
    char* end = nullptr;
    const long v = std::strtol(tok.c_str(), &end, 10);
    if (end == tok.c_str() || *end != '\0' || v < 1 || v > 100000000L)
      throw std::invalid_argument(key + ": '" + tok + "' is not a positive integer frame index (comma-separated list, e.g. \"100,101,150\")");
    if (std::find(out.begin(), out.end(), static_cast<int>(v)) != out.end())
      throw std::invalid_argument(key + ": frame " + tok + " is listed twice");
    out.push_back(static_cast<int>(v));
    tok.clear();
  };
  for (char c : text) {
    if (c == ',') { flush(); tok.clear(); }
    else if (c != ' ') tok.push_back(c);
  }
  flush();
  if (static_cast<int>(out.size()) > kMaxDumpListFrames)
    throw std::invalid_argument(key + ": more than " + std::to_string(kMaxDumpListFrames) + " frames listed");
  return out;
}

inline int firstFrameSolveDumpMode(long frame_idx, int k, const std::vector<int>& list, bool slim)
{
  if (frame_idx >= 1 && frame_idx <= k) return 1;
  if (frame_idx >= 1 && std::find(list.begin(), list.end(), static_cast<int>(frame_idx)) != list.end()) return slim ? 2 : 1;
  return 0;
}

inline bool preImuTraceWanted(int frames, long frame_idx) { return frames > 0 && frame_idx >= 1 && frame_idx <= frames; }
inline bool residualDumpIterationWanted(bool all_iters, int iteration) { return all_iters ? iteration >= 0 : iteration == 0; }
inline bool lastIterDumpWanted(bool last_iter, bool all_iters) { return last_iter && !all_iters; }
}  // namespace dump_rules
}  // namespace livo_recon
