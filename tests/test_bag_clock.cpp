// R74 unit test (F-422): t_rel_s = scan time minus bag start time, 6 decimals (include/livo_recon/utils/log/bag_clock.h).
//   BC1 a scan 0.1 s after the bag start prints exactly "0.100000" for t_abs about 1.6e9 (the resolution that 9 significant digits lost).
//   BC2 two scans 0.1 s apart at t_abs about 1.6e9 give t_rel_s values that differ by 0.100000 (+-1e-6): the key a reducer needs.
//   BC3 exactly 6 decimals for a range of values (0, 1e-7 rounds to 0.000000, 12.3456789 -> 12.345679, a negative offset keeps its sign).
//   BC4 control: printing t_abs with 9 significant digits (the R73 defect) collapses scans 0.1 s apart to the same string - the check BC2 can tell the two apart.
//   BC5 the global form prints "nan" until setBagStart() is called, then equals formatTRel with that start; a non finite input prints "nan".
#include "livo_recon/utils/log/bag_clock.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  const double start = 1609056025.123456;
  CHECK(formatTRel(start + 0.1, start) == "0.100000", "BC1 0.1 s after the start prints 0.100000");
  const double a = std::atof(formatTRel(start + 3.0, start).c_str());
  const double b = std::atof(formatTRel(start + 3.1, start).c_str());
  CHECK(std::abs((b - a) - 0.1) < 1.0e-6, "BC2 consecutive scans differ by 0.1 s in t_rel_s");
  CHECK(formatTRel(5.0, 5.0) == "0.000000", "BC3 zero");
  CHECK(formatTRel(5.0000001, 5.0) == "0.000000", "BC3 1e-7 rounds to 0.000000");
  CHECK(formatTRel(12.3456789, 0.0) == "12.345679", "BC3 6 decimals, rounded");
  CHECK(formatTRel(1.0, 3.5) == "-2.500000", "BC3 negative offset keeps its sign");
  char s1[64], s2[64];
  std::snprintf(s1, sizeof(s1), "%.9g", start + 3.0);
  std::snprintf(s2, sizeof(s2), "%.9g", start + 3.1);
  CHECK(std::strcmp(s1, s2) == 0, "BC4 control: 9 significant digits cannot tell scans 0.1 s apart (the R73 defect)");
  CHECK(tRelString(100.0) == "nan" || bagStartIsSet(), "BC5 prints nan while the bag start is unset");
  setBagStart(start);
  CHECK(tRelString(start + 0.1) == formatTRel(start + 0.1, start), "BC5 global form equals the pure form once set");
  CHECK(formatTRel(std::nan(""), 0.0) == "nan", "BC5 non finite input prints nan");
  std::printf("ALL PASS\n");
  return 0;
}
