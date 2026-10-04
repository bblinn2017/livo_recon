// R76 unit test (F-445, Bryce 2026-10-02): resolveLineConfig of include/livo_recon/lio/line_residual.h, the (enabled, form, rule) -> LineMode map.
//   SA disabled -> Off for every form/rule text (the caller refuses a key set while the flag is off; this function never sees it).
//   SB enabled defaults (form hard, rule absolute) -> HardAbsolute (the default Bryce chose).
//   SC every rule under form hard -> its LineMode (absolute 1, relative 2, relative_noise 4, shape 5); smooth -> 3.
//   SD refusals: unknown form, unknown rule, smooth together with a non-default rule; each leaves a non-empty reason.
//   SE the resolved mode behaves as chosen: HardAbsolute is a line at e1 just under plane_threshold and not at it (strict <), the same constant that tests e0.
//   SF the R74/R75 pair parser is unchanged (the internal pair vocabulary the glue test still uses).
// Counts failures and continues; exits 1 at the end if any check failed.
#include "livo_recon/lio/line_residual.h"

#include <cstdio>
#include <string>

static int g_fail = 0, g_pass = 0;
#define CHECK(c, msg)                                                                                      \
  do {                                                                                                     \
    if (!(c)) { ++g_fail; std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); } \
    else { ++g_pass; std::printf("[PASS] %s\n", msg); }                                                    \
  } while (0)

using namespace livo_recon;

int main()
{
  LineMode m = LineMode::HardShape;
  std::string why;
  {  // SA
    bool all_off = true;
    for (const char* f : {"hard", "smooth", "bogus", ""}) for (const char* r : {"absolute", "shape", "bogus", ""})
    { m = LineMode::HardShape; all_off = all_off && resolveLineConfig(false, f, r, m) && m == LineMode::Off; }
    CHECK(all_off, "SA enabled=false resolves to Off whatever form and rule say");
  }
  {  // SB
    CHECK(resolveLineConfig(true, "hard", "absolute", m) && m == LineMode::HardAbsolute, "SB enabled with the defaults (hard, absolute) is HardAbsolute");
  }
  {  // SC
    CHECK(resolveLineConfig(true, "hard", "relative", m) && static_cast<int>(m) == 2, "SC hard + relative = 2");
    CHECK(resolveLineConfig(true, "hard", "relative_noise", m) && static_cast<int>(m) == 4, "SC hard + relative_noise = 4");
    CHECK(resolveLineConfig(true, "hard", "shape", m) && static_cast<int>(m) == 5, "SC hard + shape = 5");
    CHECK(resolveLineConfig(true, "hard", "absolute", m) && static_cast<int>(m) == 1, "SC hard + absolute = 1");
    CHECK(resolveLineConfig(true, "smooth", "absolute", m) && static_cast<int>(m) == 3, "SC smooth (rule left at its default) = 3");
    CHECK(resolveLineConfig(true, "smooth", "", m) && static_cast<int>(m) == 3, "SC smooth with no rule text = 3");
  }
  {  // SD
    why.clear();
    CHECK(!resolveLineConfig(true, "bogus", "absolute", m, &why) && !why.empty(), "SD unknown form is refused with a reason");
    why.clear();
    CHECK(!resolveLineConfig(true, "hard", "bogus", m, &why) && !why.empty(), "SD unknown rule is refused with a reason");
    why.clear();
    CHECK(!resolveLineConfig(true, "hard", "", m, &why) && !why.empty(), "SD empty rule under hard is refused");
    why.clear();
    CHECK(!resolveLineConfig(true, "smooth", "shape", m, &why) && !why.empty(), "SD smooth together with rule shape is refused");
    CHECK(!resolveLineConfig(true, "smooth", "relative", m), "SD smooth together with rule relative is refused");
  }
  {  // SE
    LineOptions o; resolveLineConfig(true, "hard", "absolute", m); o.mode = m;
    const double thr = o.plane_threshold;
    CHECK(isLineVoxel(o, 1e-9, thr * 0.999, 20) && !isLineVoxel(o, 1e-9, thr, 20) && !isLineVoxel(o, 1e-9, thr * 1.001, 20),
          "SE the default rule is e1 < plane_threshold (strict), the constant that tests e0");
    CHECK(thr == 2.5e-3, "SE the default plane_threshold the default rule reads is 2.5e-3");
  }
  {  // SF
    LineMode p = LineMode::Off;
    CHECK(parseLineMode("hard", "absolute", p) && p == LineMode::HardAbsolute && parseLineMode("smooth", "", p) && p == LineMode::Smooth &&
          parseLineMode("off", "", p) && p == LineMode::Off && !parseLineMode("hard", "bogus", p), "SF the R74/R75 pair parser is unchanged");
  }
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  if (g_fail) { std::fprintf(stderr, "FAILED: %d check(s)\n", g_fail); return 1; }
  std::printf("ALL PASS\n");
  return 0;
}
