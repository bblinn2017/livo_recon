#pragma once

// R86B. The three decisions of the R86B write-only dumps, kept in one ROS-free header so a unit test can show each fires (tests/test_dump_rules_r86b.cpp).
//
//   preImuTraceWanted(frames, frame_idx): a "pre_imu" state_trace row is written for frame indices 1..frames. frames == 0 (default) writes none.
//   residualDumpIterationWanted(all_iters, iteration): the residual dump of a listed frame is written at iteration 0 only (default, as before R86B), or at every
//     iteration >= 0 when lio/residual_dump_all_iters is on. A negative iteration (no iteration context) is never dumped.
//   lastIterDumpWanted(last_iter, all_iters): the separate dump after the iteration loop (lio/residual_dump_last_iter) is suppressed when every iteration is dumped,
//     otherwise the last iteration would be written twice.

namespace livo_recon
{
namespace dump_rules
{
inline bool preImuTraceWanted(int frames, long frame_idx) { return frames > 0 && frame_idx >= 1 && frame_idx <= frames; }
inline bool residualDumpIterationWanted(bool all_iters, int iteration) { return all_iters ? iteration >= 0 : iteration == 0; }
inline bool lastIterDumpWanted(bool last_iter, bool all_iters) { return last_iter && !all_iters; }
}  // namespace dump_rules
}  // namespace livo_recon
