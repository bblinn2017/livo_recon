// R73 unit test (F-414): validation of the shadow options (include/livo_recon/lio/shadow_options.h).
//   SO1 mode off is always valid; window and model are not examined.
//   SO2 mode on needs window_start >= 1 and window_end >= window_start.
//   SO3 process_noise_model must be same, fixed or isotropic; isotropic validates beta in [0,1), scales >= 0, caps > 0.
//   SO4 inWindow: inclusive on both ends, false when mode is off.
#include "livo_recon/lio/shadow_options.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  ShadowOptions o;
  CHECK(shadowOptionsError(o).empty() && !o.on() && !o.inWindow(5), "SO1 default is off and valid");
  o.mode = "on";
  CHECK(!shadowOptionsError(o).empty(), "SO2 on with window 0,0 is refused (frame 0 is the bootstrap scan)");
  o.window_start = 330; o.window_end = 329;
  CHECK(!shadowOptionsError(o).empty(), "SO2 end < start refused");
  o.window_end = 420;
  CHECK(shadowOptionsError(o).empty(), "SO2 valid window accepted");
  CHECK(o.inWindow(330) && o.inWindow(420) && !o.inWindow(329) && !o.inWindow(421), "SO4 window is inclusive on both ends");
  o.process_noise_model = "bogus";
  CHECK(!shadowOptionsError(o).empty(), "SO3 unknown model refused");
  o.process_noise_model = "isotropic";
  CHECK(shadowOptionsError(o).empty(), "SO3 isotropic with the default motion-Q values accepted");
  o.motion_beta = 1.0;
  CHECK(!shadowOptionsError(o).empty(), "SO3 beta = 1 refused");
  o.motion_beta = 0.9; o.motion_acc_cap = 0.0;
  CHECK(!shadowOptionsError(o).empty(), "SO3 cap 0 refused");
  o.motion_acc_cap = 1e6; o.motion_gyr_scale = -1.0;
  CHECK(!shadowOptionsError(o).empty(), "SO3 negative scale refused");
  o.mode = "off";
  CHECK(shadowOptionsError(o).empty() && !o.inWindow(350), "SO1 off is valid whatever the rest holds and never in window");
  std::printf("ALL PASS\n");
  return 0;
}
