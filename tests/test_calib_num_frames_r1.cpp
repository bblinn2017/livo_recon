// R97A R1 unit test: calib/stationary/num_frames stop-condition policy (CalibFrameStopState,
// calibFrameStopConsumeGroup()), tested as a pure, ROS-free function against a synthetic stream
// of group sizes -- no live DataQueues/ROS stack needed.
//   F1 num_frames = 3 against groups [1, 5, 40, 40, 40, 41] stops after the third group of at
//      least two samples (the 4th group overall) and reports samples = 85 (5 + 40 + 40).
//   F2 num_frames = 0 (the default) never stops early on its own -- the caller decides the stop
//      condition from num_samples instead, so this policy should accept every group that qualifies
//      without ever being told to stop (checked indirectly: the loop below never needs to break).
#include "livo_recon/processing/calib_processing.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  {
    const std::vector<int> groups = {1, 5, 40, 40, 40, 41};
    const int num_frames = 3;
    CalibFrameStopState s;
    int groups_consumed = 0;
    for (int n : groups)
    {
      if (s.groups_qualifying >= num_frames) break;
      calibFrameStopConsumeGroup(s, n);
      ++groups_consumed;
    }
    CHECK(groups_consumed == 4, "F1 stops after the 4th group overall (3rd qualifying group)");
    CHECK(s.groups_qualifying == 3, "F1 qualifying count is 3");
    CHECK(s.groups_short_imu == 1, "F1 exactly one group (size 1) was dropped for <2 samples");
    CHECK(s.raw_samples_qualifying == 85, "F1 reports samples = 85 (5 + 40 + 40)");
  }
  {
    const std::vector<int> groups = {1, 5, 40, 40, 40, 41};
    CalibFrameStopState s;
    for (int n : groups) calibFrameStopConsumeGroup(s, n);
    CHECK(s.groups_seen == 6, "F2 num_frames=0: every group is consumed, none skipped by this policy");
    CHECK(s.groups_qualifying == 5, "F2 5 of 6 groups qualify (only the size-1 group is dropped)");
    CHECK(s.raw_samples_qualifying == 5 + 40 + 40 + 40 + 41, "F2 raw sample sum over all qualifying groups");
  }
  std::printf("ALL PASS\n");
  return 0;
}
