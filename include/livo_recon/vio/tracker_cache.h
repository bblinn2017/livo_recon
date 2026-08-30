#pragma once

// Reader for the .rlivtrackcache binary format written by
// src/tools/cache_tracker_output.cpp -- see that file's header comment for
// the full format spec. Loads an entire cache eagerly into memory at
// construction (a few hundred MB per bag, well within budget for a single
// test run) and hands out TrackedFrame records by index -- Tracker's
// replay mode (TrackerOptions::track_cache_path) pulls from this instead
// of running Tracker::stepAndReseedOnce() live, skipping the GPU tracker
// pass entirely.
//
// Deliberately mirrors FAST-LIVO2's livo_vio/tracker_cache.h field-for-
// field where the two TrackedFrame layouts overlap (see that file for the
// original design this was ported from) -- the format tag differs
// (.rlivtrackcache vs .trackcache) since the two TrackedFrame structs are
// NOT wire-compatible (livo_recon's is missing FAST-LIVO2's per-point
// bearing_z/rad_per_px/uv_jacobian rigorous-distortion-weight fields,
// which have no livo_recon equivalent).

#include <string>
#include <vector>

#include "livo_recon/vio/tracked_frame.h"

namespace livo_recon
{

struct TrackerCacheHeader
{
  int32_t grid_rows = 0;
  int32_t grid_cols = 0;
  std::string tracker_format;  // backend + tracker_format, e.g. "cotracker/aot"
  std::string bag_basename;
  uint32_t n_frames = 0;
};

class TrackerCacheReader
{
public:
  // Reads the whole file into `frames_`. Returns true on success; on
  // failure, `err` is set to a human-readable message and the reader is
  // left empty (size() == 0).
  bool load(const std::string& path, std::string& err);

  const TrackerCacheHeader& header() const { return header_; }
  size_t size() const { return frames_.size(); }

  // Caller must ensure i < size() -- no bounds check (matches the
  // performance-sensitive access pattern in asyncTrackingLoop()).
  const TrackedFrame& at(size_t i) const { return frames_[i]; }

  // Finds the record whose timestamp matches `ts` within 1e-6 -- NOT a
  // plain sequential/index-based lookup. See FAST-LIVO2's
  // findByCurrTimestamp doc comment for the underlying reason (a live
  // run's real feed sequence is a contiguous SUFFIX of the cache's
  // sequence, not the whole thing from index 0, once IMU
  // initialization/calibration eats the first several frames) -- the same
  // reasoning applies here since livo_recon's CALIB phase similarly
  // consumes frames before any MeasureGroup/tracking result is actually
  // needed. Binary search (frames_ is strictly increasing in timestamp by
  // construction: the cache was built from one forward pass over the
  // bag). Returns frames_.size() if no match is found within tolerance.
  size_t findByTimestamp(double ts) const;

private:
  TrackerCacheHeader header_;
  std::vector<TrackedFrame> frames_;
};

}  // namespace livo_recon
