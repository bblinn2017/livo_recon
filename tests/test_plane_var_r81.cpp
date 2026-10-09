// R81 unit test (F-468, F-485 Part 3): voxel_map/plane/plane_var_scan_s2 and voxel_map/plane/plane_var_min, and the residual's voxel_point_count / voxel_distinct_frames.
// Planning agent's test, NOT built or run by the planner on the node. No random numbers.
//
//   Part A (pure arithmetic, lio/plane_var_r81.h only; compiles with g++ alone, define R81_PURE_ONLY to build only this part):
//     P1  defaults (0, 0): the value comes back bit for bit and neither flag is set.
//     P2  clamp: max(v, v_min) (raised and not raised cases), flag set only when raised.
//     P3  scan term: v + s2 / n, n = max(1, frames) (frames 0 and negative act as 1), flag set.
//     P4  ORDER: scan term first, then the clamp. Values where the order matters: v + s2/n < v_min < v + s2 (clamp first then add would give v_min + s2/n); and
//         v < v_min < v + s2/n (result is v + s2/n, not v_min).
//     P5  refusal: negative, NaN, +inf, -inf for either key is refused with a message naming the key; 0 and positive finite values are accepted.
//     P6  control (F2: the check can fire): the wrong order (clamp, then add) is NOT equal to the right one on the P4 values.
//   Part B (VoxelPlane::gate() and computeResidual(); needs the project sources, same private-access pattern as test_line_variance_r77.cpp):
//     G1  defaults: plane_var_term equals J P J^T computed in the test, bit for bit; the R81 engagement counters do not move.
//     G2  s2 only: gate()'s plane_var_term = J P J^T + s2 / F with F = the plane's distinct_frames_ (F = 4 set directly, F = 0 acts as 1); counters count the rows.
//     G3  clamp only: raised to v_min when above the fit, unchanged when below; clamped counter moves only when raised.
//     G4  both, order: the P4 values through gate().
//     G5  the gate uses the same value: sigma_gate_squared moves by exactly (new term - fit term); a point just outside the default gate is rejected by default and admitted
//         with a clamp that lifts the variance (the admission and the weight see the same term).
//     G6  computeResidual(): plane_var_term is the modified term, sigma_squared is floor + sigma_diag (the caller adds plane_var_term), voxel_point_count = pointsSize(),
//         voxel_distinct_frames = distinct_frames_; at the defaults plane_var_term is the unmodified value.
//     G7  control: a plane variance key with line_residual on is not tested here (the refusal lives in VoxelMap::loadParameters(), read in the source check below);
//         the source check greps src/map/voxelmap.cpp for the key names, the refusal text and the params dump line (LIVO_R81_VOXELMAP_SRC, when defined).
// Exit code 0 = all pass.
#include "livo_recon/lio/plane_var_r81.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

static int g_failed = 0;
#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); ++g_failed; } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using livo_recon::planeVarR81Apply;
using livo_recon::planeVarR81Error;

namespace {
bool bitEqual(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }
}  // namespace

#ifndef R81_PURE_ONLY
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#define private public
#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/lio/voxelplane.h"
#undef private
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

std::vector<PointXYZCov> gridFixture()
{
  std::vector<PointXYZCov> v;
  for (int i = -4; i <= 4; ++i)
    for (int j = -4; j <= 4; ++j)
      v.push_back(mkPoint(V3D(0.1 * i, 0.1 * j, 1e-4 * (((i + 4) * 3 + (j + 4) * 5) % 7 - 3))));
  return v;
}

std::shared_ptr<VoxelOpts> mkOpts()
{
  auto o = std::make_shared<VoxelOpts>();
  o->max_points = 1000;
  o->max_layer = 1;
  o->voxel_size = 1.0;
  o->min_init_points = 5;
  o->min_update_points = 5;
  o->weight_sigma_r2 = 4e-4;
  o->plane_fit_pose_cov_mode = "sensor_only";
  return o;   // plane_var_min = plane_var_scan_s2 = 0.0 (the defaults)
}

struct GateOut { bool ok; double r, sd, pv, floor_term, sg; Eigen::Matrix<double, 1, 3> J; };

GateOut callGate(const VoxelPlane& p, const V3D& pt)
{
  GateOut g{};
  g.J.setZero();
  bool cand = false, abl = false, reached = false;
  g.floor_term = std::numeric_limits<double>::quiet_NaN();
  g.sg = -1.0;
  const V3D body_normal = p.plane_.normal;   // identity rotation
  g.ok = p.gate(pt, 1e-6 * M3D::Identity(), M3D::Zero(), false, pt, body_normal, g.r, g.sd, g.pv, g.J, &cand, &abl, &g.floor_term, &reached, &g.sg);
  return g;
}

WorldPointCov mkWorld(const V3D& pt)
{
  WorldPointCov w;
  w.point = pt;
  w.sensor_cov = 1e-6 * M3D::Identity();
  w.pose_cov = M3D::Zero();
  w.include_pose_cov_in_gate = false;
  w.body_point = pt;
  w.rot_transpose = M3D::Identity();
  w.prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();
  return w;
}
}  // namespace
#endif

int main()
{
  // ---------------------------------------------------------------- Part A
  {   // P1 defaults
    bool sa = true, cl = true;
    const double v = 3.14159e-4;
    CHECK(bitEqual(planeVarR81Apply(v, 0.0, 0.0, 7, &sa, &cl), v) && !sa && !cl, "P1 defaults (0, 0): the value comes back bit for bit, no flag set");
    CHECK(bitEqual(planeVarR81Apply(0.0, 0.0, 0.0, 1), 0.0), "P1 defaults: zero stays zero");
  }
  {   // P2 clamp
    bool sa = true, cl = false;
    CHECK(bitEqual(planeVarR81Apply(1e-4, 0.0, 4e-4, 5, &sa, &cl), 4e-4) && !sa && cl, "P2 clamp raises 1e-4 to 4e-4 and sets 'clamped'");
    CHECK(bitEqual(planeVarR81Apply(9e-4, 0.0, 4e-4, 5, &sa, &cl), 9e-4) && !cl, "P2 clamp leaves 9e-4 alone, 'clamped' not set");
    CHECK(bitEqual(planeVarR81Apply(4e-4, 0.0, 4e-4, 5, &sa, &cl), 4e-4) && !cl, "P2 value equal to the clamp: unchanged, not counted as raised");
  }
  {   // P3 scan term
    bool sa = false, cl = true;
    CHECK(bitEqual(planeVarR81Apply(1e-4, 2e-4, 0.0, 4, &sa, &cl), 1e-4 + 2e-4 / 4.0) && sa && !cl, "P3 v + s2 / 4 with 4 frames");
    CHECK(bitEqual(planeVarR81Apply(1e-4, 2e-4, 0.0, 0, nullptr, nullptr), 1e-4 + 2e-4), "P3 frames 0 acts as 1");
    CHECK(bitEqual(planeVarR81Apply(1e-4, 2e-4, 0.0, -3, nullptr, nullptr), 1e-4 + 2e-4), "P3 negative frames acts as 1");
    CHECK(planeVarR81Apply(1e-4, 2e-4, 0.0, 8, nullptr, nullptr) < planeVarR81Apply(1e-4, 2e-4, 0.0, 2, nullptr, nullptr), "P3 more frames, smaller scan term");
  }
  {   // P4 order
    const double v = 1e-4, s2 = 4e-4;
    const int n = 4;   // v + s2/n = 2e-4, v + s2 = 5e-4
    bool sa = false, cl = false;
    double got = planeVarR81Apply(v, s2, 3e-4, n, &sa, &cl);   // v + s2/n = 2e-4 < 3e-4: the clamp raises the sum
    CHECK(bitEqual(got, 3e-4) && sa && cl, "P4 sum below the clamp: result is the clamp");
    got = planeVarR81Apply(v, s2, 1.5e-4, n, &sa, &cl);        // v < 1.5e-4 < v + s2/n: the clamp does not bind AFTER the scan term
    CHECK(bitEqual(got, v + s2 / 4.0) && sa && !cl, "P4 clamp between the fit and the sum: result is the SUM (scan term first, then the clamp)");
    // P6 control: clamp first, then add, would give a different number on the same inputs
    const double wrong = std::max(v, 1.5e-4) + s2 / 4.0;
    CHECK(!bitEqual(wrong, got), "P6 control: clamp-then-add gives a different value, so the order check can fail");
    CHECK(planeVarR81Apply(v, s2, 1.5e-4, n, nullptr, nullptr) != wrong, "P6 control: the implementation is not the wrong order");
  }
  {   // P5 refusal
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    CHECK(planeVarR81Error(0.0, 0.0).empty() && planeVarR81Error(4e-4, 2e-4).empty() && planeVarR81Error(0.0, 4e-4).empty() && planeVarR81Error(4.5e-4, 0.0).empty(), "P5 0 and positive finite values are accepted (s2 alone is allowed)");
    CHECK(contains(planeVarR81Error(-1e-4, 0.0), "plane_var_min") && contains(planeVarR81Error(0.0, -1e-4), "plane_var_scan_s2"), "P5 negative refused, naming the key");
    CHECK(contains(planeVarR81Error(nan, 0.0), "plane_var_min") && contains(planeVarR81Error(0.0, nan), "plane_var_scan_s2"), "P5 NaN refused, naming the key");
    CHECK(contains(planeVarR81Error(inf, 0.0), "plane_var_min") && contains(planeVarR81Error(0.0, inf), "plane_var_scan_s2"), "P5 +inf refused, naming the key");
    CHECK(contains(planeVarR81Error(-inf, 0.0), "plane_var_min") && contains(planeVarR81Error(0.0, -inf), "plane_var_scan_s2"), "P5 -inf refused, naming the key");
    CHECK(!planeVarR81Error(1e-4, -1.0).empty() && !planeVarR81Error(-1.0, 1e-4).empty(), "P5 one bad key refuses even when the other is fine");
  }

#ifndef R81_PURE_ONLY
  // ---------------------------------------------------------------- Part B
  {
    auto o = mkOpts();
    std::unique_ptr<VoxelPlane> pp(new VoxelPlane(o));
    setCurrentFrame(1);
    pp->update(gridFixture(), -1, nullptr, nullptr, nullptr, 1);
    VoxelPlane& P = *pp;
    CHECK(P.isPlane(), "G0 the grid fixture fits a plane");
    const V3D c = P.plane_.center;
    const V3D n = P.plane_.normal;
    const V3D pt = c + 0.05 * P.x_normal_ + 0.03 * P.y_normal_ + 1e-4 * n;
    P.distinct_frames_ = 4;

    // G1 defaults
    const VoxelPlaneR81Counters c0 = voxelPlaneR81Counters();
    const GateOut base = callGate(P, pt);
    const double fit = (base.J * P.plane_var_ * base.J.transpose()).value();
    CHECK(base.ok && bitEqual(base.pv, fit) && fit > 0.0, "G1 defaults: plane_var_term is J P J^T bit for bit");
    const VoxelPlaneR81Counters c1 = voxelPlaneR81Counters();
    CHECK(c1.rows == c0.rows && c1.scan_rows == c0.scan_rows && c1.clamped == c0.clamped, "G1 defaults: the R81 engagement counters do not move");

    // G2 s2 only
    const double s2 = 3.0 * fit;
    o->plane_var_scan_s2 = s2;
    GateOut g = callGate(P, pt);
    CHECK(bitEqual(g.pv, fit + s2 / 4.0), "G2 s2 only: plane_var_term = fit + s2 / 4 (4 distinct frames)");
    P.distinct_frames_ = 0;
    g = callGate(P, pt);
    CHECK(bitEqual(g.pv, fit + s2), "G2 distinct_frames_ = 0 acts as 1");
    P.distinct_frames_ = 4;
    VoxelPlaneR81Counters c2 = voxelPlaneR81Counters();
    CHECK(c2.rows == c1.rows + 2 && c2.scan_rows == c1.scan_rows + 2 && c2.clamped == c1.clamped, "G2 counters: two rows, two scan rows, none clamped");
    CHECK((c2.sum_n_scans - c1.sum_n_scans) == 4 + 1, "G2 counters: sum of n_scans = 4 + 1");

    // G3 clamp only
    o->plane_var_scan_s2 = 0.0;
    o->plane_var_min = 10.0 * fit;
    g = callGate(P, pt);
    CHECK(bitEqual(g.pv, 10.0 * fit), "G3 clamp above the fit: plane_var_term = v_min");
    VoxelPlaneR81Counters c3 = voxelPlaneR81Counters();
    CHECK(c3.clamped == c2.clamped + 1 && c3.scan_rows == c2.scan_rows, "G3 the clamped counter moved by 1, scan_rows did not");
    o->plane_var_min = 0.5 * fit;
    g = callGate(P, pt);
    CHECK(bitEqual(g.pv, fit), "G3 clamp below the fit: plane_var_term unchanged");
    CHECK(voxelPlaneR81Counters().clamped == c3.clamped, "G3 clamp not binding: clamped counter did not move");

    // G4 order through gate()
    o->plane_var_scan_s2 = 4.0 * fit;       // + s2/4 = + fit
    o->plane_var_min = 1.5 * fit;           // fit < 1.5 fit < 2 fit
    g = callGate(P, pt);
    CHECK(bitEqual(g.pv, fit + (4.0 * fit) / 4.0), "G4 both keys: the scan term is added first, the clamp (between fit and fit + s2/n) does not bind");
    o->plane_var_min = 3.0 * fit;           // above fit + s2/4 = 2 fit
    g = callGate(P, pt);
    CHECK(bitEqual(g.pv, 3.0 * fit), "G4 both keys: a clamp above the sum wins");

    // G5 gate and weight see the same term
    o->plane_var_scan_s2 = 0.0; o->plane_var_min = 0.0;
    const GateOut b = callGate(P, pt);
    o->plane_var_scan_s2 = 4.0 * fit; o->plane_var_min = 3.0 * fit;
    const GateOut a = callGate(P, pt);
    CHECK(std::abs((a.sg - b.sg) - (a.pv - b.pv)) <= 1e-15 * std::max(1.0, std::abs(a.sg)), "G5 sigma_gate_squared moves by exactly the change of plane_var_term");
    {
      // a point just outside the default admission, inside after the lift
      o->plane_var_scan_s2 = 0.0; o->plane_var_min = 0.0;
      const double thr = o->sigma_num_squared * b.sg;
      const double r_out = std::sqrt(1.5 * thr);
      const V3D pt_out = c + r_out * n;   // on the normal through the centre: the geometric gate passes
      const GateOut d0 = callGate(P, pt_out);
      o->plane_var_min = 1.0e-2;          // a clamp far above the fit
      const GateOut d1 = callGate(P, pt_out);
      CHECK(!d0.ok && d1.ok, "G5 a point outside the default gate is admitted when the clamp lifts the variance (admission uses the changed term)");
    }

    // G6 computeResidual
    o->plane_var_scan_s2 = 0.0; o->plane_var_min = 0.0;
    Residual r0{}, r1{};
    const WorldPointCov w = mkWorld(pt);
    CHECK(P.computeResidual(w, r0, -1), "G6 residual admitted at the defaults");
    CHECK(bitEqual(r0.plane_var_term, fit) && r0.voxel_point_count == P.pointsSize() && r0.voxel_distinct_frames == 4,
          "G6 defaults: plane_var_term unmodified; voxel_point_count = pointsSize(), voxel_distinct_frames = 4");
    CHECK(r0.voxel_point_count == 81, "G6 voxel_point_count is the 81 fixture points");
    o->plane_var_scan_s2 = 4.0 * fit; o->plane_var_min = 3.0 * fit;
    CHECK(P.computeResidual(w, r1, -1), "G6 residual admitted with the keys on");
    CHECK(bitEqual(r1.plane_var_term, 3.0 * fit) && bitEqual(r1.sigma_squared, r0.sigma_squared) && r1.sigma_gate_squared > r0.sigma_gate_squared,
          "G6 keys on: plane_var_term = the clamped sum; sigma_squared (floor + sigma_diag) unchanged; the caller adds plane_var_term to it, so the weight sees the term");
    CHECK(r1.voxel_point_count == 81 && r1.voxel_distinct_frames == 4, "G6 keys on: voxel columns unchanged by the variance keys");
  }
  {   // G7 source contract of the loader (the refusal for lines on and the params line); only when the path is given
#ifdef LIVO_R81_VOXELMAP_SRC
    std::ifstream f(LIVO_R81_VOXELMAP_SRC);
    std::stringstream ss; ss << f.rdbuf();
    const std::string src = ss.str();
    CHECK(!src.empty(), "G7 voxelmap.cpp readable");
    CHECK(contains(src, "\"voxel_map/plane/plane_var_min\"") && contains(src, "\"voxel_map/plane/plane_var_scan_s2\""), "G7 both keys are read through cfg.get (claimed, so the unknown-key check passes)");
    CHECK(contains(src, "planeVarR81Error(") && contains(src, "line_residual/enabled = true: line, distance and smooth rows"), "G7 negative/NaN/inf refusal and the line-residual refusal are in the loader");
    CHECK(contains(src, "plane/plane_var_min:") && contains(src, "plane/plane_var_scan_s2:"), "G7 the [params/voxel_map] dump states both values");
#endif
  }
#endif

  if (g_failed) { std::fprintf(stderr, "%d check(s) FAILED\n", g_failed); return 1; }
  std::printf("all checks passed\n");
  return 0;
}
