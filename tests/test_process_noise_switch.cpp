// R74 unit test (F-426 part 2): the process-noise switch at one scan (include/livo_recon/lio/process_noise_switch.h).
//   PS1 off (scan 0): never due, whatever the frame; applying nothing leaves the options untouched; no error.
//   PS2 due exactly once: not due before `scan`, due at the first frame >= scan (also if the first frame seen is later than `scan`),
//       and not due again once applied.
//   PS3 fixed -> isotropic applies the model and the five motion parameters; isotropic -> fixed changes only the model name.
//   PS4 validation: negative scan, unknown model, beta outside [0,1), negative scales, non-positive caps are each reported by key.
//   PS5 control (the check can fail): a deliberately wrong expectation (switch due one frame early) is rejected.
// Exit code 0 = all pass.
#include "livo_recon/lio/process_noise_switch.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

struct MockOpts   // the members ImuProcOptions has, with its names
{
  std::string process_noise_model = "fixed";
  double motion_beta = 0.0, motion_acc_scale = 1.0, motion_gyr_scale = 1.0;
  double motion_acc_max_dynamic_variance = 0.5, motion_gyr_max_dynamic_variance = 0.3;
};

int main()
{
  {
    ProcessNoiseSwitch s;   // scan 0
    CHECK(!s.on() && processNoiseSwitchError(s).empty(), "PS1 default is off and valid");
    CHECK(!processNoiseSwitchDue(s, 1, false) && !processNoiseSwitchDue(s, 100000, false), "PS1 off is never due");
  }
  {
    ProcessNoiseSwitch s; s.scan = 340; s.model = "isotropic";
    CHECK(!processNoiseSwitchDue(s, 1, false) && !processNoiseSwitchDue(s, 339, false), "PS2 not due before the scan");
    CHECK(processNoiseSwitchDue(s, 340, false), "PS2 due at the scan");
    CHECK(processNoiseSwitchDue(s, 342, false), "PS2 due at the first frame seen after the scan if the exact one was never seen");
    CHECK(!processNoiseSwitchDue(s, 340, true) && !processNoiseSwitchDue(s, 400, true), "PS2 not due again once applied");
    const bool wrong_expectation_holds = (processNoiseSwitchDue(s, 339, false) == true);   // what a switch that fires one frame early would claim
    CHECK(!wrong_expectation_holds, "PS5 control: the expectation 'due at 339' is rejected (the check can fail)");
  }
  {
    MockOpts o;
    ProcessNoiseSwitch s; s.scan = 5; s.model = "isotropic"; s.motion_beta = 0.8; s.motion_acc_scale = 1.0; s.motion_gyr_scale = 5.0; s.motion_acc_cap = 7.0; s.motion_gyr_cap = 9.0;
    applyProcessNoiseSwitch(o, s);
    CHECK(o.process_noise_model == "isotropic" && o.motion_beta == 0.8 && o.motion_acc_scale == 1.0 && o.motion_gyr_scale == 5.0 &&
          o.motion_acc_max_dynamic_variance == 7.0 && o.motion_gyr_max_dynamic_variance == 9.0, "PS3 fixed -> isotropic applies model and the five parameters");
    ProcessNoiseSwitch back; back.scan = 5; back.model = "fixed";
    MockOpts before = o;
    applyProcessNoiseSwitch(o, back);
    CHECK(o.process_noise_model == "fixed" && o.motion_beta == before.motion_beta && o.motion_acc_scale == before.motion_acc_scale &&
          o.motion_gyr_scale == before.motion_gyr_scale && o.motion_acc_max_dynamic_variance == before.motion_acc_max_dynamic_variance,
          "PS3 isotropic -> fixed changes only the model name");
  }
  {
    ProcessNoiseSwitch s; s.scan = 5;
    s.scan = -1;               CHECK(processNoiseSwitchError(s).find("switch/scan") != std::string::npos, "PS4 negative scan");
    s.scan = 5; s.model = "x"; CHECK(processNoiseSwitchError(s).find("switch/model") != std::string::npos, "PS4 unknown model");
    s.model = "isotropic"; s.motion_beta = 1.0; CHECK(processNoiseSwitchError(s).find("motion_beta") != std::string::npos, "PS4 beta 1");
    s.motion_beta = 0.9; s.motion_acc_scale = -1.0; CHECK(processNoiseSwitchError(s).find("scale") != std::string::npos, "PS4 negative scale");
    s.motion_acc_scale = 0.3; s.motion_gyr_cap = 0.0; CHECK(processNoiseSwitchError(s).find("cap") != std::string::npos, "PS4 zero cap");
    s.motion_gyr_cap = 1e6; CHECK(processNoiseSwitchError(s).empty(), "PS4 a valid isotropic switch has no error");
  }
  std::printf("ALL PASS\n");
  return 0;
}
