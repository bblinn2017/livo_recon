// R80 unit test (F-453, Bryce 2026-10-05): rows = distance. A hard line voxel gives ONE row, the shortest distance from the point to the line (axis u2 through the plane centre),
// along the unit direction q from the line to the point. Planning agent's test, WRITTEN BUT NOT BUILT OR RUN by me. Same private-access pattern and fixture as test_line_variance_r77.cpp.
//
//   BOX fixture (the R77 fixture): 17 x 3 grid, x in [-0.4, 0.4], y in {-0.03, 0, 0.03}, a tiny deterministic z wiggle: e0 ~ 4e-10, e1 ~ 6e-4 (< plane_threshold: a LINE under hard/absolute),
//                   e2 ~ 0.05; every tilt gap (e_i - e2) is large, so the eigenvectors are well conditioned and the first-order propagation and its finite differences agree.
//   D0  preconditions (box is a line voxel; the two query offsets pass the plane gate and the distance gate under per_normal AND distance).
//   D1  geometry, rows = distance, variance = fit_only: r = |d - (u2.d) u2| (1e-12); normal = unit vector from the line to the point (1e-12, perpendicular to u2, along d_perp);
//       row_kind 6 (kRowLineDist), n_extra 0; plane_var_term = c0^2 e0/N + c1^2 e1/N (c_i = q.u_i); tilt_var_term 0; plane_jacobian / plane_covariance zeroed;
//       sigma_squared = floor_term + sigma_diag_squared, sigma_diag_squared = q.C.q (the point's sensor variance along q). Negative control: a wrong variance (e0, e1 unweighted) is rejected.
//   D2  variance = plane_equivalent, three independent references for the tilt term t = c0^2 l'L0 l + c1^2 l'L1 l + 2 c0 c1 l'X l, l = (0, u2.d, 1):
//       (i)  explicit per-point gradients written in this file (h_i,k = -w u_i + s w/(e_i - e2) (z_k,2 u_i + z_k,i u2), H_k = c0 h_0,k + c1 h_1,k, t = sum_k H_k' C_k H_k), relative 1e-9;
//       (ii) central finite differences of the distance |d' - (u2'.d') u2'| of the REFIT eigen-decomposition with respect to every coordinate of every point, propagated with the same C, relative 1e-3;
//       (iii) a Monte-Carlo (std::mt19937, fixed seed) of the same refit distance under the points' sensor noise, variance ratio MC / t in [0.85, 1.15].
//       The residual's plane_var_term equals the fit_only value plus tilt_var_term, and lineDistanceTiltVar returns the same t. No tilt fallback is counted. Negative control: 1.2 t is rejected by (i).
//   D3  per_normal (the default) is unchanged: the same voxel and point give kind 1 (u0) plus one extra row of kind 2 (u1), n_extra 1, r = u0.d, extra_r = u1.d, fit variances e0/N and e1/N.
//       Also: the flag is inert when the line feature is off (ordinary row), and a forced flag under form smooth still gives the three smooth rows (no row of kind 6).
//   D4  refusal: rows = distance together with form = smooth is refused by VoxelMap::loadParameters() (ConfigResolver::requireCombination). No existing test builds a ConfigResolver (it needs a live
//       ros::NodeHandle and parameter server), so this is a SOURCE-CONTRACT check: the guard expression and the message text 'rows = distance needs form = hard' are present in src/map/voxelmap.cpp
//       (path from the compile definition LIVO_R80_VOXELMAP_SRC). It does not execute the refusal.
//   D5  admission: a point on the plane gate's pass side (u0 offset 0) that is 0.45 m from the line along u1 fails the distance row's own gate (returns false, no row), where per_normal still returns
//       the u0 row (n_extra 0, as LG4); a point ON the line (|d_perp| <= 1e-9) is not admitted either.
// Failing checks are counted and the run continues; exit code 0 = all pass.
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
#include <random>
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

// rows: "per_normal" | "distance"; form: "hard" | "smooth"; plane_equivalent selects the variance; line_on = false switches the feature off.
// line_rows_distance is set directly, as VoxelMap::loadParameters() resolves it (line_enabled && rows == "distance"), except in the forced cases of D3.
std::shared_ptr<VoxelOpts> mkOpts(bool distance, bool plane_equivalent, const std::string& form = "hard", bool line_on = true, bool force_flag = false)
{
  auto o = std::make_shared<VoxelOpts>();
  o->max_points = 1000;
  o->max_layer = 1;
  o->voxel_size = 1.0;
  o->min_init_points = 5;
  o->min_update_points = 5;
  o->weight_sigma_r2 = 4e-4;   // the production floor; every tilt gap of the box fixture (~0.05) is far above it, so it never engages
  o->line_enabled = line_on;
  o->line_form = form;
  o->line_rule = "absolute";
  LineMode lm = LineMode::Off;
  std::string why;
  resolveLineConfig(line_on, form, "absolute", lm, &why);
  o->line_residual_enum = static_cast<int>(lm);
  o->line_variance = plane_equivalent ? "plane_equivalent" : "fit_only";
  o->line_var_plane_equivalent = o->line_enabled && plane_equivalent;
  o->line_rows = distance ? "distance" : "per_normal";
  o->line_rows_distance = force_flag ? true : (o->line_enabled && distance);
  // R93A: every fit uses the sensor covariance only, which is what the references below use
  return o;
}

std::unique_ptr<VoxelPlane> fit(const std::shared_ptr<VoxelOpts>& o, const std::vector<PointXYZCov>& pts)
{
  std::unique_ptr<VoxelPlane> p(new VoxelPlane(o));
  setCurrentFrame(1);
  p->update(pts, -1, nullptr, nullptr, nullptr, 1);
  return p;
}

WorldPointCov query(const V3D& p)
{
  WorldPointCov w;
  w.point = p;
  w.sensor_cov = kSensorVar * M3D::Identity();
  w.pose_cov = M3D::Zero();
  w.include_pose_cov_in_gate = false;
  w.body_point = p;
  w.rot_transpose = M3D::Identity();
  w.prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();
  return w;
}

// ---- independent references (nothing below reads a VoxelPlane member) ------------------------------------------------------------------------------
struct Eig { V3D c; V3D e; M3D U; };
Eig eigOf(const std::vector<PointXYZCov>& P)
{
  V3D c = V3D::Zero();
  for (const auto& p : P) c += p.point;
  c /= static_cast<double>(P.size());
  M3D C = M3D::Zero();
  for (const auto& p : P) C += (p.point - c) * (p.point - c).transpose();
  C /= static_cast<double>(P.size());
  Eigen::SelfAdjointEigenSolver<M3D> s(C);
  return {c, s.eigenvalues(), s.eigenvectors()};
}

// distance of the fixed world point x from the line (axis u2 through the centre) of the fit of P
double distOf(const std::vector<PointXYZCov>& P, const V3D& x)
{
  const Eig g = eigOf(P);
  const V3D u2 = g.U.col(2);
  const V3D d = x - g.c;
  return (d - u2.dot(d) * u2).norm();
}

// (i) the explicit per-point first-order gradients, written out here
double tiltRefExplicit(const std::vector<PointXYZCov>& P, const V3D& x)
{
  const Eig g = eigOf(P);
  const double w = 1.0 / static_cast<double>(P.size());
  const V3D u[3] = {g.U.col(0), g.U.col(1), g.U.col(2)};
  const V3D d = x - g.c;
  const double s = u[2].dot(d);
  const V3D dperp = d - s * u[2];
  const V3D q = dperp / dperp.norm();
  const double c0 = q.dot(u[0]), c1 = q.dot(u[1]);
  double t = 0.0;
  for (const auto& p : P) {
    const V3D z = p.point - g.c;
    V3D h[2];
    for (int i = 0; i < 2; ++i)
      h[i] = -w * u[i] + s * (w / (g.e(i) - g.e(2))) * (u[2].dot(z) * u[i] + u[i].dot(z) * u[2]);
    const V3D H = c0 * h[0] + c1 * h[1];
    t += H.dot(p.sensor_cov * H);
  }
  return t;
}

// (ii) central finite differences of distOf with respect to every coordinate of every point
double tiltRefFD(const std::vector<PointXYZCov>& P, const V3D& x)
{
  const double eps = 1e-6;
  double t = 0.0;
  for (size_t k = 0; k < P.size(); ++k) {
    V3D J = V3D::Zero();
    for (int c = 0; c < 3; ++c) {
      std::vector<PointXYZCov> Pp = P, Pm = P;
      Pp[k].point(c) += eps;
      Pm[k].point(c) -= eps;
      J(c) = (distOf(Pp, x) - distOf(Pm, x)) / (2.0 * eps);
    }
    t += J.dot(P[k].sensor_cov * J);
  }
  return t;
}

// (iii) Monte-Carlo variance of the refit distance under the points' sensor noise (fixed seed)
double tiltRefMC(const std::vector<PointXYZCov>& P, const V3D& x, int trials, unsigned seed)
{
  std::mt19937 gen(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  const double sd = std::sqrt(kSensorVar);
  double sum = 0.0, sum2 = 0.0;
  std::vector<PointXYZCov> Q = P;
  for (int it = 0; it < trials; ++it) {
    for (size_t k = 0; k < P.size(); ++k)
      Q[k].point = P[k].point + sd * V3D(nd(gen), nd(gen), nd(gen));
    const double r = distOf(Q, x);
    sum += r;
    sum2 += r * r;
  }
  const double mean = sum / trials;
  return (sum2 - trials * mean * mean) / (trials - 1);
}

bool relNear(double a, double b, double rel) { return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)) + 1e-300; }
bool absNear(double a, double b, double tol) { return std::abs(a - b) <= tol; }

}  // namespace

int main()
{
  const auto box = boxFixture();

  // two query offsets in the voxel's own frame (components along u0, u1, u2); both have a non-zero u0 and u1 part so that c0, c1 are both non-trivial
  struct Q { double a0, a1, a2; const char* name; };
  const Q queries[2] = {{0.002, 0.004, 0.17, "Qa (+u0, +u1, +u2)"}, {0.003, -0.0005, -0.2, "Qb (+u0, -u1, -u2)"}};

  auto base_fd = fit(mkOpts(true, false), box);   // fit_only, distance
  auto base_pe = fit(mkOpts(true, true), box);    // plane_equivalent, distance
  const double N = static_cast<double>(base_fd->points_size_);
  const V3D e = base_fd->eigen_values_;

  // ---- D0 -------------------------------------------------------------------------------------------------------------------------------------
  {
    CHECK(base_fd->isPlane() && e(0) < e(1) && e(1) < e(2) && e(1) < base_fd->opts_->plane_threshold, "D0 box fixture: a plane with e0 < e1 < e2 and e1 < plane_threshold (a line under hard/absolute)");
    CHECK(e(2) - e(1) > 1e-2 && e(1) - e(0) > 1e-4, "D0 box fixture: the tilt gaps are large (e2 - e1 > 1e-2, e1 - e0 > 1e-4)");
    CHECK(base_fd->opts_->line_rows_distance && base_fd->opts_->line_residual_enum == static_cast<int>(LineMode::HardAbsolute), "D0 the options give hard/absolute with the distance flag set");
    CHECK(base_pe->line_tilt_valid_[0] && base_pe->line_tilt_valid_[1] && base_pe->line_tilt_X_valid_, "D0 plane_equivalent fit has the tilt matrices L0, L1 and the cross matrix X");
    CHECK(!base_fd->line_tilt_X_valid_, "D0 fit_only fit makes no cross matrix");
    CHECK(kRowLineDist == 6, "D0 kRowLineDist is 6");
  }

  for (const Q& qq : queries) {
    const std::string tag = std::string(qq.name) + ": ";
    const VoxelPlane& pf = *base_fd;
    const V3D u0 = pf.plane_.normal, u1 = pf.y_normal_, u2 = pf.x_normal_;
    const V3D d = qq.a0 * u0 + qq.a1 * u1 + qq.a2 * u2;
    const V3D x = pf.plane_.center + d;
    const WorldPointCov w = query(x);

    // ---- D1 geometry, fit_only --------------------------------------------------------------------------------------------------------------
    Residual r;
    CHECK(pf.computeResidual(w, r, -1), (tag + "D1 distance row admitted (fit_only)").c_str());
    const double s_ax = u2.dot(d);
    const V3D dperp = d - s_ax * u2;
    const double rd = dperp.norm();
    const V3D qdir = dperp / rd;
    const double c0 = qdir.dot(u0), c1 = qdir.dot(u1);
    CHECK(absNear(r.r, rd, 1e-12), (tag + "D1 r = |d - (u2.d) u2| (1e-12)").c_str());
    CHECK(absNear(r.normal.norm(), 1.0, 1e-12) && (r.normal - qdir).norm() < 1e-12, (tag + "D1 row direction is the unit vector from the line to the point (1e-12)").c_str());
    CHECK(std::abs(r.normal.dot(u2)) < 1e-12 && r.normal.dot(dperp) > 0.0, (tag + "D1 row direction is perpendicular to the line and points to the point").c_str());
    CHECK(absNear(r.normal.dot(d), r.r, 1e-12), (tag + "D1 q.d = r (the row is the projection of d on its own direction)").c_str());
    CHECK(r.row_kind == kRowLineDist && r.row_kind == 6 && r.n_extra == 0, (tag + "D1 row_kind 6, n_extra 0").c_str());
    const double fit_expect = c0 * c0 * e(0) / N + c1 * c1 * e(1) / N;
    CHECK(relNear(r.plane_var_term, fit_expect, 1e-12), (tag + "D1 fit_only variance = c0^2 e0/N + c1^2 e1/N").c_str());
    CHECK(!relNear(r.plane_var_term, c0 * c0 * e(0) + c1 * c1 * e(1), 1e-3), (tag + "D1 control: a variance without the 1/N is rejected").c_str());
    CHECK(r.tilt_var_term == 0.0, (tag + "D1 fit_only reports tilt 0").c_str());
    CHECK(r.plane_jacobian.norm() == 0.0 && r.plane_covariance.norm() == 0.0, (tag + "D1 the exact plane Jacobian / covariance are zeroed on the modified row").c_str());
    CHECK(absNear(r.sigma_diag_squared, kSensorVar, 1e-12) && relNear(r.sigma_squared, r.floor_term + r.sigma_diag_squared, 1e-14) && absNear(r.floor_term, pf.opts_->weight_sigma_r2, 1e-12),
          (tag + "D1 sigma_squared = floor_term + sigma_diag_squared, sigma_diag_squared = q.C.q = 1e-6, floor = weight_sigma_r2").c_str());

    // ---- D2 plane_equivalent ------------------------------------------------------------------------------------------------------------------
    const VoxelPlane& pp = *base_pe;
    const VoxelPlaneR77Counters k0 = voxelPlaneR77Counters();
    Residual rp;
    CHECK(pp.computeResidual(w, rp, -1), (tag + "D2 distance row admitted (plane_equivalent)").c_str());
    const VoxelPlaneR77Counters k1 = voxelPlaneR77Counters();
    CHECK(rp.row_kind == kRowLineDist && rp.n_extra == 0 && absNear(rp.r, r.r, 1e-12), (tag + "D2 same row (kind 6, n_extra 0, same r) as fit_only").c_str());
    CHECK(k1.tilt_fallbacks == k0.tilt_fallbacks, (tag + "D2 no tilt fallback counted").c_str());
    double t_fn = -1.0;
    CHECK(pp.lineDistanceTiltVar(c0, c1, s_ax, t_fn) && t_fn > 0.0, (tag + "D2 lineDistanceTiltVar returns a positive value").c_str());
    CHECK(relNear(rp.tilt_var_term, t_fn, 1e-12) && relNear(rp.plane_var_term, fit_expect + t_fn, 1e-12),
          (tag + "D2 residual: tilt_var_term = lineDistanceTiltVar and plane_var_term = fit_only variance + tilt").c_str());

    const double t_exp = tiltRefExplicit(box, x);
    CHECK(relNear(t_fn, t_exp, 1e-9), (tag + "D2 (i) tilt variance equals the explicit per-point gradient reference (rel 1e-9)").c_str());
    CHECK(!relNear(t_fn, 1.2 * t_exp, 1e-9), (tag + "D2 control: 1.2 x the reference is rejected").c_str());
    const double t_fd = tiltRefFD(box, x);
    CHECK(relNear(t_fn, t_fd, 1e-3), (tag + "D2 (ii) tilt variance equals the finite-difference variance of the refit distance (rel 1e-3)").c_str());
    const double t_mc = tiltRefMC(box, x, 20000, 20261005u);
    const double ratio = t_mc / t_fn;
    std::printf("[INFO] %s MC variance / formula = %.4f (MC %.6e, formula %.6e, FD %.6e, explicit %.6e)\n", qq.name, ratio, t_mc, t_fn, t_fd, t_exp);
    CHECK(ratio >= 0.85 && ratio <= 1.15, (tag + "D2 (iii) Monte-Carlo variance of the refit distance / formula within [0.85, 1.15]").c_str());
  }

  // ---- D3 per_normal unchanged; inert / forced flags ------------------------------------------------------------------------------------------
  {
    auto pn = fit(mkOpts(false, false), box);
    const V3D u0 = pn->plane_.normal, u1 = pn->y_normal_;
    const V3D d = 0.002 * u0 + 0.004 * u1 + 0.17 * pn->x_normal_;
    const WorldPointCov w = query(pn->plane_.center + d);
    Residual r;
    CHECK(pn->computeResidual(w, r, -1), "D3 per_normal admits the point");
    CHECK(!pn->opts_->line_rows_distance && r.row_kind == kRowLineU0 && r.n_extra == 1 && r.extra_kind[0] == kRowLineU1, "D3 per_normal: kinds 1 (u0) and 2 (u1), n_extra 1");
    CHECK(absNear(r.r, u0.dot(d), 1e-15) && absNear(r.extra_r[0], u1.dot(d), 1e-15) && (r.normal - u0).norm() == 0.0 && (r.extra_normal[0] - u1).norm() == 0.0,
          "D3 per_normal: the rows are u0.d and u1.d along u0 and u1");
    CHECK(relNear(r.plane_var_term, pn->eigen_values_(0) / pn->points_size_, 1e-12) && relNear(r.extra_fit_var[0], pn->eigen_values_(1) / pn->points_size_, 1e-12),
          "D3 per_normal: fit variances e0/N and e1/N");

    auto off = fit(mkOpts(true, false, "hard", false, true), box);   // feature off, flag forced on
    Residual ro;
    CHECK(off->computeResidual(query(off->plane_.center + 0.002 * off->plane_.normal + 0.1 * off->x_normal_), ro, -1) && ro.row_kind == kRowPlane && ro.n_extra == 0,
          "D3 line feature off: the distance flag is inert (ordinary single plane row)");

    auto sm = fit(mkOpts(true, false, "smooth", true, true), box);   // form smooth, flag forced on (the resolver refuses this; here only the glue's own behaviour is checked)
    Residual rs;
    const V3D ds = 0.002 * sm->plane_.normal + 0.004 * sm->y_normal_ + 0.17 * sm->x_normal_;
    CHECK(sm->computeResidual(query(sm->plane_.center + ds), rs, -1) && rs.row_kind == kRowSmoothU0 && rs.n_extra == 2 && rs.extra_kind[0] == kRowSmoothU1 && rs.extra_kind[1] == kRowSmoothU2,
          "D3 smooth with the flag forced: the three smooth rows, no distance row (kind 6 never produced)");
  }

  // ---- D4 refusal (source contract, see the header) ------------------------------------------------------------------------------------------------
  {
#ifdef LIVO_R80_VOXELMAP_SRC
    std::ifstream f(LIVO_R80_VOXELMAP_SRC);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    CHECK(f.is_open() && !src.empty(), "D4 src/map/voxelmap.cpp is readable");
    CHECK(src.find("rows = distance needs form = hard") != std::string::npos, "D4 the refusal message contains 'rows = distance needs form = hard'");
    const size_t g = src.find("opts_->line_rows_distance && opts_->line_form != \"hard\"");
    const size_t m = src.find("rows = distance needs form = hard");
    CHECK(g != std::string::npos && m != std::string::npos && m > g && m - g < 200, "D4 the message follows the guard 'line_rows_distance && line_form != \"hard\"' and is passed to requireCombination");
    CHECK(src.find("cfg.requireCombination", g) < m, "D4 the guard calls cfg.requireCombination (an error that aborts the run, not a warning)");
#else
    CHECK(false, "D4 LIVO_R80_VOXELMAP_SRC is not defined: register the test through CMakeLists.txt");
#endif
  }

  // ---- D5 admission ----------------------------------------------------------------------------------------------------------------------------
  {
    auto dist = fit(mkOpts(true, false), box);
    auto pn = fit(mkOpts(false, false), box);
    const V3D far_d = 0.45 * dist->y_normal_;   // u0 offset 0: the plane gate passes; 0.45 m from the line
    Residual r, rn;
    CHECK(!dist->computeResidual(query(dist->plane_.center + far_d), r, -1), "D5 distance: a point 0.45 m off the line fails the distance row's gate (no row)");
    CHECK(pn->computeResidual(query(pn->plane_.center + 0.45 * pn->y_normal_), rn, -1) && rn.row_kind == kRowLineU0 && rn.n_extra == 0,
          "D5 control: per_normal admits the same point as the u0 row only (the plane gate passes, the u1 row is dropped)");
    Residual ro;
    CHECK(!dist->computeResidual(query(dist->plane_.center + 0.1 * dist->x_normal_), ro, -1), "D5 distance: a point ON the line (|d_perp| <= 1e-9) gets no row");
    Residual ra;
    CHECK(dist->computeResidual(query(dist->plane_.center + 0.002 * dist->plane_.normal + 0.004 * dist->y_normal_ + 0.17 * dist->x_normal_), ra, -1) && ra.row_kind == kRowLineDist,
          "D5 control: a point 4.5 mm off the line is admitted");
  }

  if (g_failed) { std::fprintf(stderr, "%d check(s) FAILED\n", g_failed); return 1; }
  std::printf("ALL PASS\n");
  return 0;
}
