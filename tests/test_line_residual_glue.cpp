// R74 unit test (F-430, F-431, F-432): the glue in VoxelPlane::computeResidual() that fills the extra rows of a line / smooth voxel.
// Same private-access pattern as test_plane_var_symmetry.cpp. No random numbers: the two fixtures are exact constructions, so the
// eigenstructure is known by construction and every expected value below is computed from the plane's own members.
//   LINE fixture  : 40 pairs (+x, -x) on a thin ELLIPSE with semi-axes a = 1e-3 (y) and b = 1.1e-3 (z): e0 = a^2/2 = 5e-7, e1 = b^2/2 = 6.05e-7,
//                   e1/e0 = 1.21 (a line under both thresholds: 1.21 < 1 + 3 sqrt(2/79) = 1.477), e2 ~ 0.05. NOT exactly degenerate: an exact tie
//                   e0 == e1 makes the R73 eigengap code divide by 0 and leave NaN in plane_var_, so the gate rejects every point (F-433).
//   PLANE fixture : a 9 x 9 grid in (x, y) of spacing 0.1 with a tiny deterministic z wiggle: e0 tiny, e1 = e2 = 0.0667.
//   LG0  preconditions: both fixtures give is_plane_; line e1 < plane_threshold; plane e1 >= plane_threshold.
//   LG1  mode off: row_kind 0, n_extra 0 on both fixtures (the values are kept for LG2).
//   LG2  planar voxel under hard/absolute, hard/relative: bit-for-bit the same Residual values as mode off (r, sigma_squared, plane_var_term,
//        plane_jacobian, floor_term, row_kind 0, n_extra 0). Planes keep the ordinary row.
//   LG3  line voxel, hard/absolute and hard/relative: row_kind 1 (u0), n_extra 1 with extra_kind 2 (u1); u1 perpendicular to u0; extra_r = u1.d;
//        plane_var_term = e0/N, extra_fit_var = e1/N; extra_sigma = floor + the sensor variance along u1.
//   LG4  hard line: a point 0.45 m away along u1 passes the plane gate (r = 0) but fails the u1 gate: the u0 row is produced, n_extra = 0.
//   LG5  smooth on both fixtures: kinds 3 / 4 / 5, n_extra 2, floor_term 0, sigma_squared = sigma_diag_squared, fit variances
//        max(e_i, weight_sigma_r2)(1 + 1/N), the three directions orthonormal, extra rows are NOT gated (the 0.45 m point still gets 3 rows).
//   LG6  control (the check can fail): a deliberately wrong expected fit variance (e0 instead of e0/N) is rejected.
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
#undef private

#include "livo_recon/lio/line_residual.h"

// R74 addendum 2: a failing check is counted and the run continues, so one run lists EVERY failing check (the first version exited at the first
// failure, which hid LG2 to LG6 behind LG1). The exit code is still non-zero if any check failed.
static int g_failed = 0;
#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); ++g_failed; } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

namespace {

PointXYZCov mkPoint(const V3D& p)
{
  PointXYZCov q;
  q.point = p;
  q.sensor_cov = 1e-6 * M3D::Identity();
  q.pos_cov = 1e-7 * M3D::Identity();
  return q;
}

std::vector<PointXYZCov> lineFixture()
{
  const int m = 40;
  const double a = 1e-3, b = 1.1e-3;   // R74 addendum 2: an ellipse, not a circle, so e0 != e1 (see the header comment and F-433)
  std::vector<PointXYZCov> v;
  for (int k = 0; k < m; ++k) {
    const double th = 2.0 * M_PI * k / m;
    const double x = 0.4 * (k + 1) / m;
    v.push_back(mkPoint(V3D(x, a * std::cos(th), b * std::sin(th))));
    v.push_back(mkPoint(V3D(-x, a * std::cos(th), b * std::sin(th))));
  }
  return v;
}

std::vector<PointXYZCov> planeFixture()
{
  std::vector<PointXYZCov> v;
  for (int i = -4; i <= 4; ++i)
    for (int j = -4; j <= 4; ++j)
      v.push_back(mkPoint(V3D(0.1 * i, 0.1 * j, 1e-4 * (((i + 4) * 3 + (j + 4) * 5) % 7 - 3))));
  return v;
}

struct Fixture
{
  std::shared_ptr<VoxelOpts> opts;
  std::shared_ptr<VoxelStats> stats;
  std::unique_ptr<VoxelNode> node;
};

Fixture build(const std::vector<PointXYZCov>& pts, const std::string& mode, const std::string& threshold)
{
  Fixture f;
  f.opts = std::make_shared<VoxelOpts>();
  f.opts->max_points = 1000;
  f.opts->max_layer = 1;
  f.opts->voxel_size = 1.0;
  f.opts->min_init_points = 5;
  f.opts->min_update_points = 5;
  // R76: the test still speaks the R74 (mode, threshold) pairs; map them onto the R76 flag/form/rule fields.
  f.opts->line_enabled = (mode != "off");
  f.opts->line_form = (mode == "smooth") ? "smooth" : "hard";
  f.opts->line_rule = threshold;
  LineMode lm = LineMode::Off;
  parseLineMode(mode, threshold, lm);
  f.opts->line_residual_enum = static_cast<int>(lm);
  f.stats = std::make_shared<VoxelStats>();
  f.node.reset(new VoxelNode(f.opts, f.stats, 0, V3D::Zero()));
  std::vector<PlaneUpdate> updates;
  setCurrentFrame(1);
  f.node->insertPoints(pts, updates);
  return f;
}

WorldPointCov query(const V3D& p)
{
  WorldPointCov w;
  w.point = p;
  w.sensor_cov = 1e-6 * M3D::Identity();
  w.pose_cov = M3D::Zero();
  w.include_pose_cov_in_gate = false;
  w.body_point = p;
  w.rot_transpose = M3D::Identity();
  w.prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();
  return w;
}

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

}  // namespace

int main()
{
  Fixture line_off = build(lineFixture(), "off", "absolute");
  Fixture plane_off = build(planeFixture(), "off", "absolute");
  const double thr = line_off.opts->plane_threshold;
  CHECK(line_off.node->plane_.isPlane(), "LG0 line fixture gives is_plane_ (e0 < plane_threshold)");
  CHECK(plane_off.node->plane_.isPlane(), "LG0 plane fixture gives is_plane_");
  CHECK(line_off.node->plane_.eigen_values_(1) < thr, "LG0 line fixture: e1 < plane_threshold");
  CHECK(plane_off.node->plane_.eigen_values_(1) >= thr, "LG0 plane fixture: e1 >= plane_threshold");
  {
    const double e0 = line_off.node->plane_.eigen_values_(0), e1 = line_off.node->plane_.eigen_values_(1);
    CHECK(near(e1 / e0, 1.21, 1e-6), "LG0 line fixture: e1 / e0 = 1.21 (near-degenerate by construction)");
    CHECK(e0 != e1 && std::isfinite(line_off.node->plane_.plane_var_(0, 0)) && std::isfinite(line_off.node->plane_.plane_var_(2, 2)),
          "LG0 line fixture: e0 != e1 and plane_var_ is finite (no exact-tie NaN, F-433)");
  }

  // ---- LG1: off --------------------------------------------------------------------------------------------------------------
  const V3D q_plane_off = plane_off.node->plane_.plane_.center + 0.02 * plane_off.node->plane_.x_normal_ + 0.01 * plane_off.node->plane_.y_normal_ +
                          0.001 * plane_off.node->plane_.plane_.normal;
  Residual ref_plane;
  {
    WorldPointCov w = query(q_plane_off);
    CHECK(plane_off.node->plane_.computeResidual(w, ref_plane, -1), "LG1 off: planar voxel admits the point");
    CHECK(ref_plane.row_kind == 0 && ref_plane.n_extra == 0, "LG1 off: planar voxel gives the ordinary single row");
  }
  {
    const auto& pl = line_off.node->plane_;
    const V3D q = pl.plane_.center + 0.02 * pl.x_normal_ + 0.002 * pl.y_normal_;
    WorldPointCov w = query(q);
    Residual r;
    CHECK(line_off.node->plane_.computeResidual(w, r, -1), "LG1 off: line voxel admits the point");
    CHECK(r.row_kind == 0 && r.n_extra == 0, "LG1 off: line voxel still gives the ordinary single row");
  }

  // ---- LG2: planar voxel unchanged under the hard modes ---------------------------------------------------------------------
  for (const char* thrmode : {"absolute", "relative"}) {
    Fixture pf = build(planeFixture(), "hard", thrmode);
    WorldPointCov w = query(q_plane_off);
    Residual r;
    const std::string tag = std::string("hard/") + thrmode;
    CHECK(pf.node->plane_.computeResidual(w, r, -1), ("LG2 " + tag + ": planar voxel admits the point").c_str());
    CHECK(r.row_kind == 0 && r.n_extra == 0, ("LG2 " + tag + ": planar voxel keeps the single row").c_str());
    CHECK(r.r == ref_plane.r && r.sigma_squared == ref_plane.sigma_squared && r.plane_var_term == ref_plane.plane_var_term &&
          r.floor_term == ref_plane.floor_term && r.sigma_diag_squared == ref_plane.sigma_diag_squared,
          ("LG2 " + tag + ": r, sigma_squared, plane_var_term, floor_term, sigma_diag_squared equal mode off exactly").c_str());
    CHECK((r.plane_jacobian - ref_plane.plane_jacobian).norm() == 0.0 && (r.plane_covariance - ref_plane.plane_covariance).norm() == 0.0 &&
          (r.normal - ref_plane.normal).norm() == 0.0,
          ("LG2 " + tag + ": normal, plane_jacobian, plane_covariance equal mode off exactly").c_str());
  }

  // ---- LG3 / LG4 / LG6: line voxel, hard ----------------------------------------------------------------------------------------
  for (const char* thrmode : {"absolute", "relative"}) {
    const std::string tag = std::string("hard/") + thrmode;
    Fixture lf = build(lineFixture(), "hard", thrmode);
    const auto& pl = lf.node->plane_;
    const double N = static_cast<double>(pl.points_size_);
    const double e0 = pl.eigen_values_(0), e1 = pl.eigen_values_(1);
    const V3D d = 0.02 * pl.x_normal_ + 0.002 * pl.y_normal_ + 0.001 * pl.plane_.normal;
    WorldPointCov w = query(pl.plane_.center + d);
    Residual r;
    CHECK(pl.computeResidual(w, r, -1), ("LG3 " + tag + ": line voxel admits the point").c_str());
    CHECK(r.row_kind == kRowLineU0 && r.n_extra == 1 && r.extra_kind[0] == kRowLineU1,
          ("LG3 " + tag + ": u0 row plus one u1 row, with their kinds").c_str());
    CHECK(std::abs(r.extra_normal[0].dot(pl.plane_.normal)) < 1e-12 && near(r.extra_normal[0].norm(), 1.0, 1e-12),
          ("LG3 " + tag + ": u1 is a unit vector perpendicular to u0").c_str());
    CHECK(near(r.extra_r[0], pl.y_normal_.dot(d), 1e-15), ("LG3 " + tag + ": extra_r = u1 . d").c_str());
    CHECK(near(r.plane_var_term, e0 / N, 1e-18), ("LG3 " + tag + ": u0 fit variance = e0 / N").c_str());
    CHECK(near(r.extra_fit_var[0], e1 / N, 1e-18), ("LG3 " + tag + ": u1 fit variance = e1 / N").c_str());
    CHECK(near(r.extra_sigma[0], line_off.opts->weight_sigma_r2 + 1e-6, 1e-12),
          ("LG3 " + tag + ": u1 base variance = floor + sensor variance along u1").c_str());
    CHECK(r.plane_jacobian.norm() == 0.0 && r.plane_covariance.norm() == 0.0,
          ("LG3 " + tag + ": the exact plane Jacobian/covariance are zeroed on a modified row").c_str());
    CHECK(!near(r.plane_var_term, e0, 1e-18), ("LG6 control " + tag + ": a wrong fit variance (e0, not e0/N) is rejected").c_str());

    WorldPointCov far = query(pl.plane_.center + 0.45 * pl.y_normal_);
    Residual rf;
    CHECK(pl.computeResidual(far, rf, -1), ("LG4 " + tag + ": the far point passes the plane gate").c_str());
    CHECK(rf.row_kind == kRowLineU0 && rf.n_extra == 0, ("LG4 " + tag + ": the u1 row is dropped by its own sigma_num gate").c_str());
  }

  // ---- LG5: smooth ---------------------------------------------------------------------------------------------------------------
  for (int which = 0; which < 2; ++which) {
    const std::string tag = which == 0 ? "smooth/line" : "smooth/plane";
    Fixture sf = build(which == 0 ? lineFixture() : planeFixture(), "smooth", "absolute");
    const auto& pl = sf.node->plane_;
    const double N = static_cast<double>(pl.points_size_);
    const double floor_var = sf.opts->weight_sigma_r2;
    const V3D d = 0.02 * pl.x_normal_ + 0.002 * pl.y_normal_ + 0.001 * pl.plane_.normal;
    WorldPointCov w = query(pl.plane_.center + d);
    Residual r;
    CHECK(pl.computeResidual(w, r, -1), ("LG5 " + tag + ": admits the point").c_str());
    CHECK(r.row_kind == kRowSmoothU0 && r.n_extra == 2 && r.extra_kind[0] == kRowSmoothU1 && r.extra_kind[1] == kRowSmoothU2,
          ("LG5 " + tag + ": three rows with kinds 3, 4, 5").c_str());
    CHECK(r.floor_term == 0.0 && r.sigma_squared == r.sigma_diag_squared, ("LG5 " + tag + ": the weight floor is not added a second time").c_str());
    const V3D e = pl.eigen_values_;
    CHECK(near(r.plane_var_term, std::max(e(0), floor_var) * (1.0 + 1.0 / N), 1e-15) &&
          near(r.extra_fit_var[0], std::max(e(1), floor_var) * (1.0 + 1.0 / N), 1e-15) &&
          near(r.extra_fit_var[1], std::max(e(2), floor_var) * (1.0 + 1.0 / N), 1e-15),
          ("LG5 " + tag + ": fit variances max(e_i, sensor noise)(1 + 1/N)").c_str());
    const V3D u0 = pl.plane_.normal, u1 = r.extra_normal[0], u2 = r.extra_normal[1];
    CHECK(std::abs(u0.dot(u1)) < 1e-12 && std::abs(u0.dot(u2)) < 1e-12 && std::abs(u1.dot(u2)) < 1e-12 &&
          near(u1.norm(), 1.0, 1e-12) && near(u2.norm(), 1.0, 1e-12),
          ("LG5 " + tag + ": the three directions are orthonormal").c_str());
    CHECK(near(r.extra_r[0], u1.dot(d), 1e-15) && near(r.extra_r[1], u2.dot(d), 1e-15), ("LG5 " + tag + ": extra residuals are u_i . d").c_str());
    WorldPointCov far = query(pl.plane_.center + 0.45 * pl.y_normal_);
    Residual rf;
    CHECK(pl.computeResidual(far, rf, -1) && rf.n_extra == 2, ("LG5 " + tag + ": the extra rows are not gated (far point still gets 3 rows)").c_str());
  }

  if (g_failed) { std::fprintf(stderr, "%d CHECK(S) FAILED\n", g_failed); return 1; }
  std::printf("ALL PASS\n");
  return 0;
}
