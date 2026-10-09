// R86B: the three dump decisions (include/livo_recon/diagnostics/dump_rules.h). Header-only; no ROS.
#include "livo_recon/diagnostics/dump_rules.h"
#include <cstdio>

using namespace livo_recon::dump_rules;

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main()
{
  // pre_imu rows: none by default, frames 1..K otherwise, never frame 0 (the bootstrap scan) and never beyond K.
  CK(!preImuTraceWanted(0, 1));
  CK(!preImuTraceWanted(0, 0));
  CK(preImuTraceWanted(1, 1));
  CK(!preImuTraceWanted(1, 0));
  CK(!preImuTraceWanted(1, 2));
  CK(preImuTraceWanted(3, 3));
  CK(!preImuTraceWanted(3, 4));
  CK(!preImuTraceWanted(-1, 1));
  // residual dump iterations: default = iteration 0 only (unchanged behaviour); all_iters = every iteration >= 0.
  CK(residualDumpIterationWanted(false, 0));
  for (int i = 1; i < 8; ++i) CK(!residualDumpIterationWanted(false, i));
  CK(!residualDumpIterationWanted(false, -1));
  for (int i = 0; i < 8; ++i) CK(residualDumpIterationWanted(true, i));
  CK(!residualDumpIterationWanted(true, -1));
  // last-iteration dump: only without all_iters.
  CK(lastIterDumpWanted(true, false));
  CK(!lastIterDumpWanted(true, true));
  CK(!lastIterDumpWanted(false, false));
  CK(!lastIterDumpWanted(false, true));
  std::printf("test_dump_rules_r86b %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
