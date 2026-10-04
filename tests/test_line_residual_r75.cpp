// R75 unit test (Bryce 2026-10-02): the two new hard-line classifications of include/livo_recon/lio/line_residual.h.
//   RA parseLineMode: hard + relative_noise and hard + shape parse to LineMode 4 and 5; the R74 pairs still parse to 1, 2, 3; unknown thresholds are refused.
//   RB HardRelativeNoise: line when e1 <= (1 + k sqrt(2/(N-1))) max(e0, sigma^2). At the degenerate voxel of R74 (e0 about 0, e1 about 5e-4, N = 20, sigma^2 = 4e-4)
//       it IS a line while HardRelative (R74) is NOT (control: the two disagree, which is the reason for the variant). Exactly at the bound it is a line (<=), just above not.
//       With e0 above sigma^2 it equals HardRelative. sigma^2 = 0 makes it equal HardRelative for every tested (e0, e1, N). N < 2 is never a line. rel_sigmas is real.
//   RC HardShape: line when e1 <= tau e2. 0.099 e2 is a line, 0.101 e2 is not (tau 0.1); tau is a real parameter (0.15 e2: line at tau 0.2, not at 0.1); e2 <= 0 and the
//       default argument (no e2) are never lines; N and sigma^2 play no role.
//   RD rows: both new variants give exactly 2 rows (normals u0, u1, residuals u_i . d, fit variances e_i / N as the other hard variants) for a line voxel and 0 rows for a plane voxel.
//   RE R74 variants unchanged: HardAbsolute and HardRelative give the same answer with and without the e2 argument (it is ignored), Off is never a line.
// This test counts failures and continues; it exits 1 at the end if any check failed (the first-failure exit of earlier tests hid later checks).
#include "livo_recon/lio/line_residual.h"

#include <cstdio>
#include <initializer_list>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(c, msg)                                                                                      \
  do {                                                                                                     \
    if (!(c)) { ++g_fail; std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); } \
    else { ++g_pass; std::printf("[PASS] %s\n", msg); }                                                    \
  } while (0)

using namespace livo_recon;
using V3 = Eigen::Vector3d;

int main()
{
  LineOptions rel; rel.mode = LineMode::HardRelative;
  LineOptions rn;  rn.mode = LineMode::HardRelativeNoise; rn.sensor_noise_var = 4e-4;
  LineOptions sh;  sh.mode = LineMode::HardShape;
  LineOptions abs_o; abs_o.mode = LineMode::HardAbsolute;
  LineOptions off_o;

  {  // RA
    LineMode m = LineMode::Off;
    CHECK(parseLineMode("hard", "relative_noise", m) && m == LineMode::HardRelativeNoise && static_cast<int>(m) == 4, "RA hard + relative_noise parses to 4");
    CHECK(parseLineMode("hard", "shape", m) && m == LineMode::HardShape && static_cast<int>(m) == 5, "RA hard + shape parses to 5");
    CHECK(parseLineMode("hard", "absolute", m) && static_cast<int>(m) == 1 && parseLineMode("hard", "relative", m) && static_cast<int>(m) == 2 &&
          parseLineMode("smooth", "", m) && static_cast<int>(m) == 3, "RA the R74 combinations still parse to 1, 2, 3");
    CHECK(!parseLineMode("hard", "relative_noisy", m) && !parseLineMode("smooth", "shape", m) && !parseLineMode("hard", "", m), "RA unknown or misplaced thresholds are refused");
  }
  {  // RB
    const double e0 = 1e-9, e1 = 5e-4; const int N = 20;
    CHECK(!isLineVoxel(rel, e0, e1, N), "RB control: HardRelative (R74) does NOT call the degenerate voxel (e0 near 0) a line");
    CHECK(isLineVoxel(rn, e0, e1, N), "RB HardRelativeNoise calls the same voxel a line (e1 5e-4 within the band around sigma^2 4e-4)");
    CHECK(!isLineVoxel(rn, e0, 2e-3, N), "RB a voxel with e1 = 2e-3 is not a line (above the band)");
    const double thr = relativeGapThreshold(N, 3.0);
    CHECK(isLineVoxel(rn, e0, thr * 4e-4, N) && !isLineVoxel(rn, e0, thr * 4e-4 * 1.0001, N), "RB exactly at (1 + 3 sqrt(2/(N-1))) sigma^2 is a line (<=), just above is not");
    CHECK(isLineVoxel(rn, 1e-3, thr * 1e-3, N) && !isLineVoxel(rn, 1e-3, thr * 1e-3 * 1.0001, N), "RB with e0 above sigma^2 the bound is thr * e0");
    bool same = true;
    LineOptions z = rn; z.sensor_noise_var = 0.0;
    for (double a : {1e-9, 1e-5, 1e-4, 1e-3}) for (double b : {1e-9, 1e-4, 3e-4, 1e-3, 5e-3}) for (int n : {2, 5, 20, 200})
      same = same && (isLineVoxel(z, a, b, n) == isLineVoxel(rel, a, b, n));
    CHECK(same, "RB sigma^2 = 0 reduces to HardRelative on every tested (e0, e1, N)");
    CHECK(!isLineVoxel(rn, e0, e1, 1) && !isLineVoxel(rn, e0, e1, 0), "RB N < 2 is never a line");
    LineOptions k5 = rn; k5.rel_sigmas = 5.0;
    const double b = relativeGapThreshold(20, 4.0) * 4e-4;   // between the 3-sigma and the 5-sigma bound
    CHECK(!isLineVoxel(rn, e0, b, N) && isLineVoxel(k5, e0, b, N), "RB rel_sigmas is a real parameter (a value between the 3 and 5 sigma bounds flips)");
  }
  {  // RC
    CHECK(isLineVoxel(sh, 1e-9, 0.099 * 0.05, 20, 0.05) && !isLineVoxel(sh, 1e-9, 0.101 * 0.05, 20, 0.05), "RC e1 = 0.099 e2 is a line, 0.101 e2 is not (tau 0.1)");
    LineOptions t2 = sh; t2.shape_ratio = 0.2;
    CHECK(!isLineVoxel(sh, 1e-9, 0.15 * 0.05, 20, 0.05) && isLineVoxel(t2, 1e-9, 0.15 * 0.05, 20, 0.05), "RC shape_ratio is a real parameter (0.15 e2: line at 0.2, not at 0.1)");
    CHECK(!isLineVoxel(sh, 0.0, 0.0, 20, 0.0) && !isLineVoxel(sh, 0.0, 0.0, 20, -1.0) && !isLineVoxel(sh, 1e-9, 1e-4, 20), "RC e2 <= 0 and the default (no e2) are never lines");
    CHECK(isLineVoxel(sh, 1e-9, 1e-3, 1, 0.05) == isLineVoxel(sh, 1e-9, 1e-3, 500, 0.05), "RC the point count plays no role");
    LineOptions s2 = sh; s2.sensor_noise_var = 1.0;
    CHECK(isLineVoxel(s2, 1e-9, 1e-3, 20, 0.05) == isLineVoxel(sh, 1e-9, 1e-3, 20, 0.05), "RC the sensor noise variance plays no role");
  }
  {  // RD
    const V3 u0(1, 0, 0), u1(0, 1, 0), u2(0, 0, 1);
    const V3 line_e(1e-4, 3e-4, 5e-2), plane_e(1e-5, 1.3e-2, 5e-2), d(0.003, -0.002, 0.04);
    const int N = 200;
    LineRow rows[3];
    for (const LineOptions* o : {&rn, &sh}) {
      const int n = lineRows(*o, u0, u1, u2, line_e, d, N, rows);
      CHECK(n == 2, "RD a line voxel gives exactly 2 rows");
      CHECK(n == 2 && (rows[0].normal - u0).norm() == 0.0 && (rows[1].normal - u1).norm() == 0.0, "RD the rows are along u0 and u1");
      CHECK(n == 2 && rows[0].r == 0.003 && rows[1].r == -0.002, "RD the residuals are u_i . d");
      CHECK(n == 2 && rows[0].fit_var == 1e-4 / N && rows[1].fit_var == 3e-4 / N, "RD the fit variances are e_i / N (unchanged from the other hard variants)");
      CHECK(lineRows(*o, u0, u1, u2, plane_e, d, N, rows) == 0, "RD a plane voxel gives 0 rows (the caller keeps the single ordinary row)");
    }
  }
  {  // RE
    bool same = true;
    for (double a : {1e-9, 1e-4, 1e-3}) for (double b : {1e-4, 1e-3, 5e-3}) for (int n : {5, 200}) {
      same = same && (isLineVoxel(abs_o, a, b, n) == isLineVoxel(abs_o, a, b, n, 0.05)) && (isLineVoxel(rel, a, b, n) == isLineVoxel(rel, a, b, n, 0.05));
    }
    CHECK(same, "RE HardAbsolute and HardRelative ignore e2 (same answer with and without it)");
    CHECK(!isLineVoxel(off_o, 1e-9, 1e-9, 20, 0.05), "RE mode Off is never a line");
  }
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  if (g_fail) { std::fprintf(stderr, "FAILED: %d check(s)\n", g_fail); return 1; }
  std::printf("ALL PASS\n");
  return 0;
}
