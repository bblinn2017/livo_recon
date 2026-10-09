// R87: the admission rule (include/livo_recon/lio/admission_rule.h). Header-only; no ROS. Own check macro (does not depend on NDEBUG).
#include "livo_recon/lio/admission_rule.h"
#include <cstdio>

using livo_recon::admission::fitWanted;

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main()
{
  // never fitted: needs min_init raw points, in the bootstrap frame too
  CK(!fitWanted(false, 0, 5, 5, false));
  CK(!fitWanted(false, 4, 5, 5, false));
  CK(fitWanted(false, 5, 5, 5, false));
  CK(!fitWanted(false, 3, 5, 5, true));      // bootstrap does NOT relax min_init
  CK(!fitWanted(false, 4, 5, 5, true));
  CK(fitWanted(false, 5, 5, 5, true));
  CK(fitWanted(false, 3, 3, 5, false));      // min_init 3 reproduces the pre-R87 debiased first fit
  CK(!fitWanted(false, 2, 3, 5, false));
  // already fitted: min_update points since the last fit, except in the bootstrap frame
  CK(!fitWanted(true, 0, 5, 5, false));
  CK(!fitWanted(true, 4, 5, 5, false));
  CK(fitWanted(true, 5, 5, 5, false));
  CK(fitWanted(true, 1, 5, 5, true));        // bootstrap exempt from min_update
  CK(fitWanted(true, 0, 5, 5, true));
  CK(fitWanted(true, 1, 3, 1, false));       // min_update 1 reproduces the pre-R87 debiased refit on every batch
  CK(fitWanted(true, 7, 3, 1, false));
  // the two thresholds are independent
  CK(fitWanted(false, 4, 4, 100, false));
  CK(!fitWanted(true, 50, 3, 100, false));
  std::printf("test_admission_rule_r87 %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
