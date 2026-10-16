#pragma once

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// R79 (Bryce 2026-10-05, F-464 / F-441 / F-440: "a voxel-map snapshot at every one of frames 1 to 5"): which frames write a voxel-map snapshot.
//
//   log/voxel_map/snapshot_post_calibration (R47..R78, unchanged): post_calibration_voxelmap.csv, the map frame 1 queries (written after the
//   bootstrap, when frame_idx_ has just become 1).
//   log/voxel_map/snapshot_frames (R79, new, default empty = off): a comma-separated list of frame indices N >= 2. voxelmap_snapshot_f<N>.csv is the map
//   frame N QUERIES: it is written at the end of the live map insertion after which frame_idx_ == N (so f2 is the map after the insertion of the
//   first live frame, and so on). The residual dump of frame N (log/lio/residual_dump/frames) joins to it by node_id.
//
// Refused, never silently ignored (standing rule 3.4): a non-integer token, a frame below 2 (frame 1 is the post-calibration snapshot), a repeated frame,
// more than kMaxSnapshotFrames frames, a non-empty list while snapshot_post_calibration is off (the frame-1 snapshot is the first member of the series
// every reducer expects).
// Header-only, ROS-free (unit-tested in tests/test_snapshot_frames_r79.cpp).
// ============================================================================

namespace livo_recon
{

constexpr int kMaxSnapshotFrames = 20;

inline std::vector<int> parseSnapshotFrames(const std::string& list, bool post_calibration_on)
{
  std::vector<int> out;
  std::string tok;
  auto flush = [&]() {
    if (tok.empty()) return;
    char* end = nullptr;
    const long v = std::strtol(tok.c_str(), &end, 10);
    if (end == tok.c_str() || *end != '\0')
      throw std::invalid_argument("log/voxel_map/snapshot_frames: '" + tok + "' is not an integer frame index (comma-separated list, e.g. \"2,3,4,5\")");
    if (v < 2)
      throw std::invalid_argument("log/voxel_map/snapshot_frames: frame " + tok + " is below 2 (frame 1 is log/voxel_map/snapshot_post_calibration)");
    if (std::find(out.begin(), out.end(), static_cast<int>(v)) != out.end())
      throw std::invalid_argument("log/voxel_map/snapshot_frames: frame " + tok + " is listed twice");
    out.push_back(static_cast<int>(v));
    tok.clear();
  };
  for (char c : list)
  {
    if (c == ',') { flush(); tok.clear(); }
    else if (c != ' ') tok.push_back(c);
  }
  flush();
  if (static_cast<int>(out.size()) > kMaxSnapshotFrames)
    throw std::invalid_argument("log/voxel_map/snapshot_frames: more than " + std::to_string(kMaxSnapshotFrames) + " frames");
  if (!out.empty() && !post_calibration_on)
    throw std::invalid_argument("log/voxel_map/snapshot_frames needs log/voxel_map/snapshot_post_calibration = true (the frame-1 snapshot is the first of the series)");
  return out;
}

inline bool snapshotFrameWanted(const std::vector<int>& frames, int frame_idx)
{
  return std::find(frames.begin(), frames.end(), frame_idx) != frames.end();
}

inline std::string snapshotFrameFileName(int frame)
{
  return "voxelmap_snapshot_f" + std::to_string(frame) + ".csv";
}

}  // namespace livo_recon
