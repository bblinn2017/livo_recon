// R79 unit test: which frames write a voxel-map snapshot (include/livo_recon/map/snapshot_frames.h). Pure parsing / selection logic, ROS-free, header only.
//   SF1 "2,3,4,5" parses to {2,3,4,5} in order; spaces are ignored; an empty string is off (empty list) even with snapshot_post_calibration off.
//   SF2 the file name is voxelmap_snapshot_f<N>.csv and snapshotFrameWanted() is true for the listed frames only (frame 1, frame 6 and 0 are not wanted).
//   SF3 every refusal throws and says why: a non-integer token, 0, 1 (the post-calibration snapshot), a negative number, a repeated frame, more than kMaxSnapshotFrames frames,
//       and a non-empty list while snapshot_post_calibration is off.
//   SF4 the checks can fail: a parser that accepted frame 1 would fail the "1" refusal; a parser that ignored the post-calibration flag would fail the flag refusal;
//       an off-by-one in snapshotFrameWanted() (frame_idx_ - 1) would fail SF2 on frame 2 and on frame 6.
#include "livo_recon/map/snapshot_frames.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

static bool refused(const std::string& s, bool post_cal, const std::string& needle)
{
  try { parseSnapshotFrames(s, post_cal); }
  catch (const std::invalid_argument& e) { return std::string(e.what()).find(needle) != std::string::npos; }
  return false;
}

int main()
{
  const std::vector<int> f = parseSnapshotFrames("2,3,4,5", true);
  CHECK(f.size() == 4 && f[0] == 2 && f[1] == 3 && f[2] == 4 && f[3] == 5, "SF1 \"2,3,4,5\" parses to {2,3,4,5}");
  const std::vector<int> g = parseSnapshotFrames(" 2 , 7 ,", true);
  CHECK(g.size() == 2 && g[0] == 2 && g[1] == 7, "SF1 spaces and a trailing comma are ignored");
  CHECK(parseSnapshotFrames("", true).empty() && parseSnapshotFrames("", false).empty(), "SF1 an empty string is off, with or without snapshot_post_calibration");

  CHECK(snapshotFrameFileName(2) == "voxelmap_snapshot_f2.csv" && snapshotFrameFileName(15) == "voxelmap_snapshot_f15.csv", "SF2 file name voxelmap_snapshot_f<N>.csv");
  CHECK(snapshotFrameWanted(f, 2) && snapshotFrameWanted(f, 5), "SF2 frames 2 and 5 are wanted");
  CHECK(!snapshotFrameWanted(f, 1) && !snapshotFrameWanted(f, 6) && !snapshotFrameWanted(f, 0), "SF2 frames 0, 1 and 6 are not wanted (off-by-one would fail here)");

  CHECK(refused("2,x", true, "not an integer"), "SF3 a non-integer token is refused");
  CHECK(refused("0", true, "below 2"), "SF3 frame 0 is refused");
  CHECK(refused("1", true, "snapshot_post_calibration"), "SF3 frame 1 is refused and the message names snapshot_post_calibration");
  CHECK(refused("-3", true, "below 2"), "SF3 a negative frame is refused");
  CHECK(refused("2,3,2", true, "twice"), "SF3 a repeated frame is refused");
  std::string many;
  for (int i = 2; i < 2 + kMaxSnapshotFrames + 1; ++i) many += (many.empty() ? "" : ",") + std::to_string(i);
  CHECK(refused(many, true, "more than"), "SF3 more than kMaxSnapshotFrames frames is refused");
  CHECK(refused("2,3", false, "needs voxel_map/map/snapshot_post_calibration"), "SF3 a non-empty list without snapshot_post_calibration is refused");
  CHECK(!refused("2,3", true, "x") , "SF4 the refusal check itself can say no: a valid list is not refused");
  std::printf("ALL PASS\n");
  return 0;
}
