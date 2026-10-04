// R74 unit test (F-430, F-431, F-432): residual rows of a voxel that may be a line (include/livo_recon/lio/line_residual.h).
// Fixtures are synthetic single-ring voxels (points along a 3-D segment, noise sigma 0.01 m across it) and synthetic planes.
//   LR1 classification, hard absolute: the single-ring line (e0, e1 about 1e-4 < 2.5e-3) is a line; the plane (e1 about 1.3e-2) is not; e1 exactly equal
//       to plane_threshold is NOT a line (strict <); mode Off never is.
//   LR2 classification, hard relative: exactly at the formula: e1 == thr e0 (thr = 1 + 3 sqrt(2/(N-1))) is a line (<=), e1 just above is not; N < 2 and
//       e0 <= 0 are never lines (no division by zero); rel_sigmas is a real parameter (at N = 200, e1 = 1.5 e0: 3 sigma says plane, 5 sigma says line).
//   LR3 the accepted constant is not a calibrated 3-sigma test of the ORDERED ratio: over 2000 synthetic single-ring voxels of 200 points the relative rule
//       calls between 70 and 95 percent of them lines (numpy reference 83 percent), the absolute rule 100 percent. Printed; the bounds only guard
//       against an implementation error. (Finding for Bryce: the miss rate is real; rel_sigmas is a config value.)
//   LR4 hard rows: a line gives exactly 2 rows with normals u0 and u1; the rows' residuals are u_i . d; a displacement ALONG the ring (u2) gives
//       residuals 0 in both rows (no constraint along the ring) and a displacement along u0 gives r0 = the displacement, r1 = 0.
//   LR5 information: sum_i u_i u_i^T / var over the two line rows has rank 2 with its null vector along the ring direction (|u2 . null| > 0.999);
//       control: the single point-to-plane row (u0 only) has rank 1, so the second perpendicular direction is unconstrained - the difference
//       is what F-430 adds.
//   LR6 a plane gives 0 rows in both hard variants (the caller keeps the ordinary row, bit for bit).
//   LR7 smooth: always 3 rows (u0, u1, u2), residuals u_i . d, fit variances max(e_i, floor)(1 + 1/N) exactly; e0 = 0 is floored, never 0 or nan.
//   LR8 smooth, plane voxel: the weight along the normal is more than 20 times the weight along each in-plane direction (it reduces to point-to-plane
//       without a class switch). smooth, line voxel: the two small-eigenvalue directions get weights within a factor 3 of each other and each more than 20
//       times the weight along the ring.
//   LR9 parseLineMode: the three accepted combinations parse; hard without a threshold, relative without hard, and unknown names do not.
#include "livo_recon/lio/line_residual.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;
using V3 = Eigen::Vector3d;

struct Fit { V3 c, e; Eigen::Matrix3d U; };

static Fit fitPoints(const std::vector<V3>& P)
{
  V3 c = V3::Zero();
  for (const auto& p : P) c += p;
  c /= static_cast<double>(P.size());
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (const auto& p : P) C += (p - c) * (p - c).transpose();
  C /= static_cast<double>(P.size());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
  return Fit{c, es.eigenvalues(), es.eigenvectors()};
}

static std::vector<V3> ring(std::mt19937& rng, int n)
{
  std::uniform_real_distribution<double> U(-0.2, 0.2);
  std::normal_distribution<double> N(0.0, 0.01);
  std::vector<V3> v;
  for (int i = 0; i < n; ++i) v.push_back(V3(U(rng), N(rng), N(rng)));
  return v;
}

static std::vector<V3> plane(std::mt19937& rng, int n)
{
  std::uniform_real_distribution<double> U(-0.2, 0.2);
  std::normal_distribution<double> N(0.0, 0.005);
  std::vector<V3> v;
  for (int i = 0; i < n; ++i) v.push_back(V3(U(rng), U(rng), N(rng)));
  return v;
}

static Eigen::Matrix3d info(const LineRow* rows, int n, double point_var)
{
  Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
  for (int i = 0; i < n; ++i) I += rows[i].normal * rows[i].normal.transpose() / (point_var + rows[i].fit_var);
  return I;
}

int main()
{
  std::mt19937 rng(430);
  const int N = 60;
  const Fit L = fitPoints(ring(rng, N));
  const Fit P = fitPoints(plane(rng, N));

  LineOptions abs_o; abs_o.mode = LineMode::HardAbsolute;
  LineOptions rel_o; rel_o.mode = LineMode::HardRelative;
  LineOptions off_o;
  CHECK(isLineVoxel(abs_o, L.e(0), L.e(1), N), "LR1 single-ring voxel is a line (absolute)");
  CHECK(!isLineVoxel(abs_o, P.e(0), P.e(1), N), "LR1 plane voxel is not a line (absolute)");
  CHECK(!isLineVoxel(abs_o, 1e-4, 2.5e-3, N) && isLineVoxel(abs_o, 1e-4, 2.4999e-3, N), "LR1 e1 equal to plane_threshold is not a line (strict <)");
  CHECK(!isLineVoxel(off_o, L.e(0), L.e(1), N), "LR1 mode Off is never a line");

  {
    const double thr = relativeGapThreshold(200, 3.0);
    CHECK(isLineVoxel(rel_o, 1e-4, thr * 1e-4, 200), "LR2 e1 equal to threshold * e0 is a line (<=)");
    CHECK(!isLineVoxel(rel_o, 1e-4, thr * 1e-4 * 1.0001, 200), "LR2 e1 just above threshold * e0 is not a line");
    CHECK(!isLineVoxel(rel_o, 1e-4, 1e-4, 1) && !isLineVoxel(rel_o, 1e-4, 1e-4, 0), "LR2 N < 2 is never a line (threshold is nan)");
    CHECK(!isLineVoxel(rel_o, 0.0, 1e-4, 50) && !isLineVoxel(rel_o, -1e-9, 1e-4, 50), "LR2 e0 <= 0 is never a line (no division by zero)");
    LineOptions k5 = rel_o; k5.rel_sigmas = 5.0;
    CHECK(isLineVoxel(k5, 1e-4, 1.5e-4, 200) && !isLineVoxel(rel_o, 1e-4, 1.5e-4, 200), "LR2 rel_sigmas is a real parameter (5 calls e1 = 1.5 e0 at N = 200 a line, 3 does not)");
  }

  {
    int rel_lines = 0, abs_lines = 0;
    const int trials = 2000, n_pts = 200;
    for (int t = 0; t < trials; ++t) {
      const Fit f = fitPoints(ring(rng, n_pts));
      rel_lines += isLineVoxel(rel_o, f.e(0), f.e(1), n_pts) ? 1 : 0;
      abs_lines += isLineVoxel(abs_o, f.e(0), f.e(1), n_pts) ? 1 : 0;
    }
    const double fr = static_cast<double>(rel_lines) / trials;
    std::printf("[INFO] LR3 single-ring voxels called lines: relative rule %.3f, absolute rule %.3f (N = %d)\n", fr, static_cast<double>(abs_lines) / trials, n_pts);
    CHECK(abs_lines == trials, "LR3 the absolute rule calls every synthetic single-ring voxel a line");
    CHECK(fr > 0.70 && fr < 0.95, "LR3 the relative rule (3 sigma) calls between 70 and 95 percent of them lines");
  }

  LineRow rows[3];
  const V3 u0 = L.U.col(0), u1 = L.U.col(1), u2 = L.U.col(2);
  {
    const int n = lineRows(abs_o, u0, u1, u2, L.e, V3::Zero(), N, rows);
    CHECK(n == 2, "LR4 a line gives exactly 2 rows");
    CHECK((rows[0].normal - u0).norm() == 0.0 && (rows[1].normal - u1).norm() == 0.0, "LR4 normals are u0 and u1");
    const V3 along = 0.05 * u2;
    lineRows(abs_o, u0, u1, u2, L.e, along, N, rows);
    CHECK(std::abs(rows[0].r) < 1e-15 && std::abs(rows[1].r) < 1e-15, "LR4 a displacement along the ring gives residual 0 in both rows");
    const V3 across = 0.03 * u0 - 0.02 * u1;
    lineRows(abs_o, u0, u1, u2, L.e, across, N, rows);
    CHECK(std::abs(rows[0].r - 0.03) < 1e-15 && std::abs(rows[1].r + 0.02) < 1e-15, "LR4 a perpendicular displacement appears in the two rows with its components");
    CHECK(rows[0].fit_var == std::max(L.e(0), 0.0) / N && rows[1].fit_var == std::max(L.e(1), 0.0) / N, "LR4 fit variances are e_i / N");
  }

  {
    const int n = lineRows(rel_o, u0, u1, u2, V3(1e-4, 1.2e-4, 1e-2), V3::Zero(), 200, rows);
    CHECK(n == 2, "LR4 the relative variant also gives 2 rows for an indistinguishable pair");
    const double point_var = 1e-4;
    lineRows(abs_o, u0, u1, u2, L.e, V3::Zero(), N, rows);
    const Eigen::Matrix3d I2 = info(rows, 2, point_var);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es2(I2);
    CHECK(es2.eigenvalues()(0) < 1e-9 * es2.eigenvalues()(2) && es2.eigenvalues()(1) > 1e-3 * es2.eigenvalues()(2), "LR5 line rows: information has rank 2");
    CHECK(std::abs(es2.eigenvectors().col(0).dot(u2)) > 0.999, "LR5 the null direction is along the ring");
    const Eigen::Matrix3d I1 = info(rows, 1, point_var);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es1(I1);
    CHECK(es1.eigenvalues()(1) < 1e-9 * es1.eigenvalues()(2), "LR5 control: the single point-to-plane row has rank 1 (the second direction is unconstrained)");
  }

  CHECK(lineRows(abs_o, P.U.col(0), P.U.col(1), P.U.col(2), P.e, V3::Zero(), N, rows) == 0, "LR6 a plane gives 0 rows (absolute)");
  CHECK(lineRows(rel_o, P.U.col(0), P.U.col(1), P.U.col(2), P.e, V3::Zero(), N, rows) == 0, "LR6 a plane gives 0 rows (relative)");
  CHECK(lineRows(off_o, u0, u1, u2, L.e, V3::Zero(), N, rows) == 0, "LR6 mode Off gives 0 rows");

  {
    LineOptions s; s.mode = LineMode::Smooth; s.sensor_noise_var = 1e-4;
    const V3 d(0.01, -0.02, 0.03);
    const int n = lineRows(s, u0, u1, u2, L.e, d, N, rows);
    CHECK(n == 3, "LR7 smooth always gives 3 rows");
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
      const V3 ui = i == 0 ? u0 : (i == 1 ? u1 : u2);
      ok = ok && (rows[i].normal - ui).norm() == 0.0 && std::abs(rows[i].r - ui.dot(d)) < 1e-15 &&
           rows[i].fit_var == std::max(L.e(i), 1e-4) * (1.0 + 1.0 / N);
    }
    CHECK(ok, "LR7 normals, residuals u_i . d and fit variances max(e_i, floor)(1 + 1/N)");
    lineRows(s, u0, u1, u2, V3(0.0, 0.0, 1.0), d, N, rows);
    CHECK(std::isfinite(rows[0].fit_var) && rows[0].fit_var == 1e-4 * (1.0 + 1.0 / N), "LR7 e0 = 0 is floored at the sensor noise variance");

    const double point_var = 1e-4;
    lineRows(s, P.U.col(0), P.U.col(1), P.U.col(2), P.e, V3::Zero(), N, rows);
    const double w0 = 1.0 / (point_var + rows[0].fit_var), w1 = 1.0 / (point_var + rows[1].fit_var), w2 = 1.0 / (point_var + rows[2].fit_var);
    CHECK(w0 > 20.0 * w1 && w0 > 20.0 * w2, "LR8 smooth on a plane: the normal carries more than 20 times the weight of each in-plane direction");
    lineRows(s, u0, u1, u2, L.e, V3::Zero(), N, rows);
    const double a0 = 1.0 / (point_var + rows[0].fit_var), a1 = 1.0 / (point_var + rows[1].fit_var), a2 = 1.0 / (point_var + rows[2].fit_var);
    CHECK(a0 < 3.0 * a1 && a1 < 3.0 * a0, "LR8 smooth on a line: the two small-eigenvalue directions have weights within a factor 3");
    CHECK(a0 > 20.0 * a2 && a1 > 20.0 * a2, "LR8 smooth on a line: each carries more than 20 times the weight along the ring");
  }

  {
    LineMode m = LineMode::Off;
    CHECK(parseLineMode("hard", "absolute", m) && m == LineMode::HardAbsolute, "LR9 hard + absolute");
    CHECK(parseLineMode("hard", "relative", m) && m == LineMode::HardRelative, "LR9 hard + relative");
    CHECK(parseLineMode("smooth", "", m) && m == LineMode::Smooth, "LR9 smooth");
    CHECK(parseLineMode("off", "", m) && m == LineMode::Off, "LR9 off");
    CHECK(!parseLineMode("hard", "", m) && !parseLineMode("hard", "bogus", m) && !parseLineMode("line", "absolute", m), "LR9 hard without a valid threshold and unknown modes are refused");
  }
  std::printf("ALL PASS\n");
  return 0;
}
