// R77 unit test (F-453, F-433): the tilt term of the line / smooth row variance (VoxelOpts::line_variance = plane_equivalent) and the eigengap floor of the plane variance
// (VoxelOpts::plane_gap_floor_en). Planning agent's test, NOT built or run by me. Same private-access pattern as test_line_residual_glue.cpp. No random numbers.
//
//   BOX fixture   : 17 x 3 grid, x in [-0.4, 0.4] step 0.05, y in {-0.03, 0, 0.03}, a tiny deterministic z wiggle (amplitude 2e-5, no symmetry): three well separated eigenvalues
//                   e2 ~ 0.05, e1 ~ 6e-4 (< plane_threshold 2.5e-3: a LINE under hard/absolute), e0 ~ 4e-10, so every tilt gap is large and a central finite difference of the
//                   eigenvectors is well conditioned.
//   ELLIPSE       : the R74 line fixture (80 points on a thin ellipse, e1 / e0 = 1.21, e0 - e1 = -1.05e-7): a near-degenerate eigengap on purpose.
//   GRID          : the R74 plane fixture (9 x 9 grid, e0 tiny, e1 = e2 = 0.0667): every gap far above any floor.
//   V0  preconditions (the fixtures give the planes and eigenvalue orders the other checks rely on).
//   V1  defaults (line_var_plane_equivalent false, plane_gap_floor_en false): no tilt matrix is made, no R77 counter moves, plane_var_ is what update() gave.
//   V2  hard rows, finite-difference check: for rows 0 and 1, lineTiltVar(row, d, hard) equals (u2 . d)^2 Var(theta_i2) with Var(theta_i2) computed INDEPENDENTLY by central
//       finite differences of the covariance eigenvectors of the perturbed points, propagated with the same per-point covariance (relative 1e-3).
//   V3  smooth rows, same finite-difference check for rows 0, 1, 2 with both tilt axes (floor made inactive with weight_sigma_r2 = 1e-12).
//   V4  floor: with a floor above a tilt gap (weight_sigma_r2 = 1e-3 > 6e-4) the floored axis variance of smooth row 0 scales by (gap / floor)^2 exactly, the unfloored one is unchanged, tilt_floored_axes moved.
//   V5  residual glue: on a line voxel, hard/absolute, plane_equivalent adds exactly tilt_var_term to the u0 fit variance and extra_tilt_var[0] to the u1 fit variance of the
//       fit_only residual of the same point (and tilt is > 0); fit_only reports tilt 0.
//   V6  F-433 floor, near-degenerate fit: ELLIPSE plane_var_ trace with the floor is finite and smaller than without it, gap_d1 moved; GRID: plane_var_ with the floor is
//       bit-for-bit the plane_var_ without it and no gap counter moved.
//   V7  controls (the checks can fail): a deliberately wrong expected value (the tilt variance doubled) is rejected; a deliberately wrong floor expectation (floor not engaged) is rejected.
// Exit code 0 = all pass.
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#define private public
#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/lio/voxelplane.h"
#undef private

#include "livo_recon/lio/line_residual.h"

static int g_failed = 0;
#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); ++g_failed; } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

namespace {

const double kSensorVar = 1e-6, kPoseVar = 1e-7;

PointXYZCov mkPoint(const V3D& p)
{
  PointXYZCov q;
  q.point = p;
  q.sensor_cov = kSensorVar * M3D::Identity();
  q.pos_cov = kPoseVar * M3D::Identity();
  return q;
}

std::vector<PointXYZCov> boxFixture()
{
  std::vector<PointXYZCov> v;
  int n = 0;
  for (int i = -8; i <= 8; ++i)
    for (int j = -1; j <= 1; ++j, ++n)
      v.push_back(mkPoint(V3D(0.05 * i, 0.03 * j, 2e-5 * (((i + 8) * 3 + (j + 1) * 5 + n) % 7 - 3) / 3.0)));
  return v;
}

std::vector<PointXYZCov> ellipseFixture()
{
  const int m = 40;
  const double a = 1e-3, b = 1.1e-3;
  std::vector<PointXYZCov> v;
  for (int k = 0; k < m; ++k) {
    const double th = 2.0 * M_PI * k / m;
    const double x = 0.4 * (k + 1) / m;
    v.push_back(mkPoint(V3D(x, a * std::cos(th), b * std::sin(th))));
    v.push_back(mkPoint(V3D(-x, a * std::cos(th), b * std::sin(th))));
  }
  return v;
}

std::vector<PointXYZCov> gridFixture()
{
  std::vector<PointXYZCov> v;
  for (int i = -4; i <= 4; ++i)
    for (int j = -4; j <= 4; ++j)
      v.push_back(mkPoint(V3D(0.1 * i, 0.1 * j, 1e-4 * (((i + 4) * 3 + (j + 4) * 5) % 7 - 3))));
  return v;
}

std::shared_ptr<VoxelOpts> mkOpts(const std::string& mode, bool plane_equivalent, bool gap_floor, double sigma_r2)
{
  auto o = std::make_shared<VoxelOpts>();
  o->max_points = 1000;
  o->max_layer = 1;
  o->voxel_size = 1.0;
  o->min_init_points = 5;
  o->min_update_points = 5;
  o->weight_sigma_r2 = sigma_r2;
  o->line_enabled = (mode != "off");
  o->line_form = (mode == "smooth") ? "smooth" : "hard";
  o->line_rule = "absolute";
  LineMode lm = LineMode::Off;
  parseLineMode(mode, "absolute", lm);
  o->line_residual_enum = static_cast<int>(lm);
  o->line_variance = plane_equivalent ? "plane_equivalent" : "fit_only";
  o->line_var_plane_equivalent = o->line_enabled && plane_equivalent;
  o->plane_gap_floor_en = gap_floor;
  o->plane_fit_pose_cov_mode = "sensor_only";   // the production default; the finite differences below use the SAME covariance
  return o;
}

// a VoxelPlane fitted directly (no node, no binning): uniform weights, exactly the points given
std::unique_ptr<VoxelPlane> fit(const std::shared_ptr<VoxelOpts>& o, const std::vector<PointXYZCov>& pts)
{
  std::unique_ptr<VoxelPlane> p(new VoxelPlane(o));
  setCurrentFrame(1);
  p->update(pts, -1, nullptr, nullptr, nullptr, 1);
  return p;
}

// ---- independent finite-difference variance of the tilts ----------------------------------------------------------------------------------
struct Eig { V3D e; M3D U; };
Eig eigOf(const std::vector<PointXYZCov>& P)
{
  V3D c = V3D::Zero();
  for (const auto& p : P) c += p.point;
  c /= static_cast<double>(P.size());
  M3D C = M3D::Zero();
  for (const auto& p : P) C += (p.point - c) * (p.point - c).transpose();
  C /= static_cast<double>(P.size());
  Eigen::SelfAdjointEigenSolver<M3D> s(C);
  return {s.eigenvalues(), s.eigenvectors()};
}

// theta_ia = u_a . u_i' (the perturbed eigenvector i projected on the BASE axis a), sign aligned with the base eigenvector
double thetaOf(const std::vector<PointXYZCov>& P, const Eig& base, int i, int a)
{
  const Eig q = eigOf(P);
  V3D ui = q.U.col(i);
  if (ui.dot(base.U.col(i)) < 0.0) ui = -ui;
  return base.U.col(a).dot(ui);
}

// L(a_idx, b_idx) for row i, axes (a, b): sum_k J_a,k^T Cov_k J_b,k with Cov_k = sensor covariance (the test's pose-cov mode is sensor_only)
Eigen::Matrix2d numericL(const std::vector<PointXYZCov>& P, int i, int a, int b)
{
  const Eig base = eigOf(P);
  const double eps = 1e-6;
  Eigen::Matrix2d L = Eigen::Matrix2d::Zero();
  for (size_t k = 0; k < P.size(); ++k) {
    V3D Ja = V3D::Zero(), Jb = V3D::Zero();
    for (int c = 0; c < 3; ++c) {
      std::vector<PointXYZCov> Pp = P, Pm = P;
      Pp[k].point(c) += eps;
      Pm[k].point(c) -= eps;
      Ja(c) = (thetaOf(Pp, base, i, a) - thetaOf(Pm, base, i, a)) / (2.0 * eps);
      Jb(c) = (thetaOf(Pp, base, i, b) - thetaOf(Pm, base, i, b)) / (2.0 * eps);
    }
    const M3D C = P[k].sensor_cov;
    L(0, 0) += Ja.dot(C * Ja);
    L(1, 1) += Jb.dot(C * Jb);
    L(0, 1) += Ja.dot(C * Jb);
  }
  L(1, 0) = L(0, 1);
  return L;
}

bool relNear(double a, double b, double rel) { return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)) + 1e-300; }
bool near_equal(double a, double b) { return std::abs(a - b) <= 1e-15 + 1e-9 * std::max(std::abs(a), std::abs(b)); }

}  // namespace

int main()
{
  const auto box = boxFixture();
  const auto ell = ellipseFixture();
  const auto grd = gridFixture();

  // ---- V0 -------------------------------------------------------------------------------------------------------------------------------------
  {
    auto p = fit(mkOpts("hard", false, false, 1e-12), box);
    const V3D e = p->eigen_values_;
    CHECK(p->isPlane() && e(0) < e(1) && e(1) < e(2) && e(1) < p->opts_->plane_threshold, "V0 box fixture: a plane with e0 < e1 < e2 and e1 < plane_threshold (a line under hard/absolute)");
    CHECK(e(1) - e(0) > 1e-4 && e(2) - e(1) > 1e-2, "V0 box fixture: all eigenvalue gaps are large (e1 - e0 > 1e-4, e2 - e1 > 1e-2)");
    auto q = fit(mkOpts("hard", false, false, 1e-12), ell);
    CHECK(q->isPlane() && std::abs(q->eigen_values_(0) - q->eigen_values_(1)) < 2e-7, "V0 ellipse fixture: a plane with |e0 - e1| < 2e-7 (near-degenerate)");
    auto g = fit(mkOpts("hard", false, false, 1e-12), grd);
    CHECK(g->isPlane() && g->eigen_values_(1) - g->eigen_values_(0) > 1e-2, "V0 grid fixture: a plane with a large e1 - e0");
  }

  // ---- V1 defaults ----------------------------------------------------------------------------------------------------------------------------
  {
    const VoxelPlaneR77Counters c0 = voxelPlaneR77Counters();
    auto p = fit(mkOpts("hard", false, false, 4e-4), box);
    auto q = fit(mkOpts("off", false, false, 4e-4), ell);
    const VoxelPlaneR77Counters c1 = voxelPlaneR77Counters();
    CHECK(!p->line_tilt_valid_[0] && !p->line_tilt_valid_[1] && !p->line_tilt_valid_[2], "V1 defaults: no tilt matrix is made (fit_only)");
    CHECK(c1.gap_fits == c0.gap_fits && c1.tilt_voxels == c0.tilt_voxels && c1.tilt_floored_axes == c0.tilt_floored_axes && c1.tilt_fallbacks == c0.tilt_fallbacks,
          "V1 defaults: no R77 counter moved");
    double t = -1.0;
    CHECK(!p->lineTiltVar(0, V3D(0.01, 0.0, 0.0), false, t) && t == -1.0, "V1 defaults: lineTiltVar reports no matrix and leaves the output untouched");
    CHECK(std::isfinite(q->plane_var_(0, 0)), "V1 defaults: plane_var_ finite on the ellipse (not exact tie)");
  }

  // ---- V2 hard rows against finite differences -----------------------------------------------------------------------------------------------
  {
    auto p = fit(mkOpts("hard", true, false, 1e-12), box);
    CHECK(p->line_tilt_valid_[0] && p->line_tilt_valid_[1] && !p->line_tilt_valid_[2], "V2 hard: tilt matrices for rows 0 and 1 only");
    const V3D u[3] = {p->plane_.normal, p->y_normal_, p->x_normal_};
    const V3D d = 0.17 * u[2] + 0.004 * u[1] + 0.001 * u[0];
    for (int row = 0; row < 2; ++row) {
      // hard rows tilt toward u2 only: axis index b = 2; (a unused: pass the other small axis, only L(1,1) is read)
      const int a = row == 0 ? 1 : 0;
      const Eigen::Matrix2d Ln = numericL(box, row, a, 2);
      const double expect = std::pow(u[2].dot(d), 2) * Ln(1, 1);
      double got = -1.0;
      CHECK(p->lineTiltVar(row, d, false, got), "V2 hard: lineTiltVar returns a value");
      const std::string tag = "V2 hard row " + std::to_string(row) + ": tilt variance equals (u2 . d)^2 Var(theta_i2) from finite differences (rel 1e-3)";
      CHECK(expect > 0.0 && relNear(got, expect, 1e-3), tag.c_str());
      CHECK(!relNear(got, 2.0 * expect, 1e-3), "V7 control: a doubled expected tilt variance is rejected");
    }
  }

  // ---- V3 smooth rows against finite differences ---------------------------------------------------------------------------------------------
  {
    auto p = fit(mkOpts("smooth", true, false, 1e-12), box);
    CHECK(p->line_tilt_valid_[0] && p->line_tilt_valid_[1] && p->line_tilt_valid_[2], "V3 smooth: tilt matrices for all three rows");
    const V3D u[3] = {p->plane_.normal, p->y_normal_, p->x_normal_};
    const V3D d = 0.17 * u[2] + 0.004 * u[1] + 0.001 * u[0];
    for (int row = 0; row < 3; ++row) {
      const int a = (row == 0) ? 1 : 0, b = (row == 2) ? 1 : 2;
      const Eigen::Matrix2d Ln = numericL(box, row, a, b);
      const double la = u[a].dot(d), lb = u[b].dot(d);
      const double expect = la * la * Ln(0, 0) + 2.0 * la * lb * Ln(0, 1) + lb * lb * Ln(1, 1);
      double got = -1.0;
      CHECK(p->lineTiltVar(row, d, true, got), "V3 smooth: lineTiltVar returns a value");
      const std::string tag = "V3 smooth row " + std::to_string(row) + ": tilt variance equals the finite-difference value (rel 1e-3)";
      CHECK(expect > 0.0 && relNear(got, expect, 1e-3), tag.c_str());
    }
  }

  // ---- V4 floor scales the floored axis exactly and is counted ---------------------------------------------------------------------------------
  {
    auto lo = fit(mkOpts("smooth", true, false, 1e-12), box);
    const VoxelPlaneR77Counters c0 = voxelPlaneR77Counters();
    auto hi = fit(mkOpts("smooth", true, false, 1e-3), box);   // floor 1e-3 > |e0 - e1| ~ 6e-4: row 0's axis-a gap (to u1) is floored, its axis-b gap (to u2) is not
    const VoxelPlaneR77Counters c1 = voxelPlaneR77Counters();
    const double gap = std::abs(lo->eigen_values_(0) - lo->eigen_values_(1));
    CHECK(gap < 1e-3 && gap > 1e-4, "V4 precondition: the e0 - e1 gap lies between 1e-4 and the 1e-3 floor");
    const double ratio = hi->line_tilt_L_[0](0, 0) / lo->line_tilt_L_[0](0, 0);
    CHECK(relNear(ratio, std::pow(gap / 1e-3, 2), 1e-6), "V4 row 0: the floored axis variance L(0,0) scales by (gap / floor)^2 exactly");
    CHECK(hi->line_tilt_L_[0](1, 1) == lo->line_tilt_L_[0](1, 1), "V4 row 0: the unfloored axis variance L(1,1) is unchanged bit for bit");
    CHECK(c1.tilt_floored_axes > c0.tilt_floored_axes, "V4 tilt_floored_axes moved");
    CHECK(!(c1.tilt_floored_axes == c0.tilt_floored_axes), "V7 control: 'the floor was not engaged' is rejected");
  }

  // ---- V5 residual glue -----------------------------------------------------------------------------------------------------------------------
  {
    auto build = [&](bool pe) {
      auto o = mkOpts("hard", pe, false, 4e-4);
      std::shared_ptr<VoxelStats> st = std::make_shared<VoxelStats>();
      std::unique_ptr<VoxelNode> n(new VoxelNode(o, st, 0, V3D::Zero()));
      std::vector<PlaneUpdate> upd;
      setCurrentFrame(1);
      n->insertPoints(box, upd);
      return std::make_pair(o, std::move(n));
    };
    auto fo = build(false);
    auto pe = build(true);
    const auto& pf = fo.second->plane_;
    const auto& pp = pe.second->plane_;
    const V3D d = 0.02 * pf.x_normal_ + 0.002 * pf.y_normal_ + 0.0005 * pf.plane_.normal;
    WorldPointCov w;
    w.point = pf.plane_.center + d;
    w.sensor_cov = 1e-6 * M3D::Identity();
    w.pose_cov = M3D::Zero();
    w.include_pose_cov_in_gate = false;
    w.body_point = w.point;
    w.rot_transpose = M3D::Identity();
    w.prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();
    Residual rf, rp;
    CHECK(pf.computeResidual(w, rf, -1) && pp.computeResidual(w, rp, -1), "V5 both forms admit the point");
    CHECK(rf.row_kind == kRowLineU0 && rp.row_kind == kRowLineU0 && rf.n_extra == 1 && rp.n_extra == 1, "V5 both forms give the hard line rows (u0, u1)");
    CHECK(rf.tilt_var_term == 0.0 && rf.extra_tilt_var[0] == 0.0, "V5 fit_only reports tilt 0 on both rows");
    CHECK(rp.tilt_var_term > 0.0 && rp.extra_tilt_var[0] > 0.0, "V5 plane_equivalent reports tilt > 0 on both rows");
    CHECK(near_equal(rp.plane_var_term - rf.plane_var_term, rp.tilt_var_term) && near_equal(rp.extra_fit_var[0] - rf.extra_fit_var[0], rp.extra_tilt_var[0]),
          "V5 plane_equivalent fit variance = fit_only fit variance + tilt, on both rows");
  }

  // ---- V6 F-433 gap floor ---------------------------------------------------------------------------------------------------------------------
  {
    const VoxelPlaneR77Counters c0 = voxelPlaneR77Counters();
    auto off = fit(mkOpts("off", false, false, 4e-4), ell);
    auto on = fit(mkOpts("off", false, true, 4e-4), ell);
    const VoxelPlaneR77Counters c1 = voxelPlaneR77Counters();
    CHECK(std::isfinite(on->plane_var_.trace()) && on->plane_var_.trace() < off->plane_var_.trace(), "V6 ellipse: the floor gives a finite, smaller plane_var_ trace");
    CHECK(c1.gap_fits == c0.gap_fits + 1 && c1.gap_d1 == c0.gap_d1 + 1, "V6 ellipse: one gap fit counted, the e0 - e1 gap floored");
    auto goff = fit(mkOpts("off", false, false, 4e-4), grd);
    const VoxelPlaneR77Counters c2 = voxelPlaneR77Counters();
    auto gon = fit(mkOpts("off", false, true, 4e-4), grd);
    const VoxelPlaneR77Counters c3 = voxelPlaneR77Counters();
    bool same = true;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) same = same && (goff->plane_var_(a, b) == gon->plane_var_(a, b));
    CHECK(same, "V6 grid: plane_var_ with the floor is bit-for-bit plane_var_ without it (gaps far above the floor)");
    CHECK(c3.gap_fits == c2.gap_fits + 1 && c3.gap_d1 == c2.gap_d1 && c3.gap_d2 == c2.gap_d2, "V6 grid: the fit is counted, no gap floored");
  }

  if (g_failed) { std::fprintf(stderr, "%d check(s) FAILED\n", g_failed); return 1; }
  std::printf("ALL PASS\n");
  return 0;
}
