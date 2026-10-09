// R82 (D4, F-500): integration test of the fit-time line classification (both fits), the one gap floor and the debiased plane-equivalent line variance, driving VoxelPlane directly (links
// src/lio/voxelplane.cpp only; the sandbox build stubs opencv2/opencv.hpp). Planner sandbox on synthetic points: NOT evidence about any bag. Identities and refusals only (rule 3.1).
//   DL1 classify off: a thin line cloud is rejected by the debiased fit (denominator guard), no R82 counter moves, isLine() is false (the default path is the R81 path).
//   DL2 classify on + gap floor, debiased: the same cloud is kept and isLine(); line_classified, line_accepted and gaps_floored each +1; a plane cloud is untouched (bit identical eigenvalues, isLine false).
//   DL3 plane_equivalent line variance on the debiased fit, form moment, equals the pca path's computeLineTilt output for the same points: rows 0 and 1 and the distance-row form agree to 1e-3 relative.
//   DL4 form mean_cov equals form moment when every point covariance is equal (to 1e-8), and differs from it when they are not.
//   DL5 a plane cloud has no tilt matrices on the debiased fit.
//   DL6 gap floor alone (classify off), debiased: the guard is not applied, the line cloud is kept as a PLANE (isLine false), gaps_floored +1.
//   DL7 pca fit with classify on: the line cloud isLine(), line_classified +1; off: isLine false; a plane cloud is never a line.
//   DL8 a perfectly thin line (e1 < 1e-8 <= e2): pca fit classify off rejects it, classify on keeps it as a line and e1_tiny_kept +1.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "livo_recon/lio/line_residual.h"
#include "livo_recon/lio/voxelplane.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static std::vector<PointXYZCov> cloud(std::mt19937& rng, double sx, double sy, double sz, double cov_scale, bool unequal)
{
  std::normal_distribution<double> N01(0, 1);
  const M3D R = Eigen::AngleAxisd(0.8, V3D(1, -2, 0.5).normalized()).toRotationMatrix();
  const V3D origin(3.0, -2.0, 1.0);
  std::vector<PointXYZCov> v;
  for (int k = 0; k < 60; ++k)
  {
    PointXYZCov p;
    p.point = origin + R * V3D(sx * N01(rng), sy * N01(rng), sz * N01(rng));
    const double s = cov_scale * (unequal ? (0.5 + (k % 7)) : 1.0);
    M3D B; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) B(i, j) = N01(rng);
    p.sensor_cov = unequal ? M3D(s * (B * B.transpose() + 0.2 * M3D::Identity())) : M3D(s * M3D::Identity());
    p.pos_cov = M3D::Zero();
    v.push_back(p);
  }
  return v;
}

static VoxelOptsPtr makeOpts(bool debiased, bool classify, int form, bool plane_equiv, bool gap_floor_alone = false)
{
  auto o = std::make_shared<VoxelOpts>();
  o->plane_threshold = 0.01;
  o->plane_fit_mode = debiased ? "debiased" : "pca";
  o->plane_var_mode = "eigengap";
  o->plane_fit_pose_cov_mode = "sensor_only";
  o->weight_sigma_r2 = 4e-4;
  o->line_enabled = true;
  o->line_residual_enum = static_cast<int>(LineMode::HardAbsolute);
  o->line_var_plane_equivalent = plane_equiv;
  o->line_classify_at_fit = classify;
  o->plane_gap_floor_en = classify || gap_floor_alone;
  o->debiased_line_form = form;
  o->debiased_line_equiv = form == 2 ? "moment" : (form == 1 ? "mean_cov" : "off");
  return o;
}

static double tiltVar(const VoxelPlane& vp, int row, const V3D& d, bool& ok) { double v = -1.0; ok = vp.lineTiltVar(row, d, false, v); return v; }

int main()
{
  std::mt19937 rng(5);
  // a thin line: 0.003 m thick in two directions, 0.8 m long; tiny sensor noise
  const auto line = cloud(rng, 0.003, 0.004, 0.8, 1e-9, true);
  const auto plane = cloud(rng, 0.004, 0.5, 0.5, 1e-9, true);
  const V3D d(0.05, -0.1, 0.3);

  // DL1: classify off
  {
    const VoxelPlaneR82Counters c0 = voxelPlaneR82Counters();
    VoxelPlane vp(makeOpts(true, false, 0, false));
    vp.addPoints(line, -1, 1, true, 0);
    CHECK(!vp.isPlane() && !vp.isLine());
    const VoxelPlaneR82Counters c1 = voxelPlaneR82Counters();
    CHECK(c1.line_classified == c0.line_classified && c1.gaps_floored == c0.gaps_floored && c1.line_accepted == c0.line_accepted);
  }
  // DL2: classify on (with the gap floor, as the config requires)
  {
    const VoxelPlaneR82Counters c0 = voxelPlaneR82Counters();
    VoxelPlane vp(makeOpts(true, true, 0, false));
    vp.addPoints(line, -1, 1, true, 0);
    CHECK(vp.isPlane() && vp.isLine());
    const VoxelPlaneR82Counters c1 = voxelPlaneR82Counters();
    CHECK(c1.line_classified == c0.line_classified + 1 && c1.line_accepted == c0.line_accepted + 1 && c1.gaps_floored == c0.gaps_floored + 1);
    VoxelPlane vq(makeOpts(true, true, 0, false)), vr(makeOpts(true, false, 0, false));
    vq.addPoints(plane, -1, 1, true, 0); vr.addPoints(plane, -1, 1, true, 0);
    const VoxelPlaneR82Counters c2 = voxelPlaneR82Counters();
    CHECK(vq.isPlane() == vr.isPlane() && !vq.isLine() && c2.line_classified == c1.line_classified);
    CHECK((vq.eigenValues() - vr.eigenValues()).norm() == 0.0);   // a plane cloud: bit identical with and without the key
  }
  // DL3: moment form on the debiased fit == pca computeLineTilt (same points, same covariances)
  {
    VoxelPlane deb(makeOpts(true, true, 2, true));
    deb.addPoints(line, -1, 1, true, 0);
    VoxelPlane pca(makeOpts(false, false, 0, true));
    pca.update(line);
    CHECK(deb.isPlane() && pca.isPlane());
    for (int row = 0; row < 2; ++row)
    {
      bool okd = false, okp = false;
      const double vd = tiltVar(deb, row, d, okd), vp = tiltVar(pca, row, d, okp);
      CHECK(okd && okp && vd > 0.0 && vp > 0.0);
      std::printf("  DL3 row %d: debiased moment %.6e  pca %.6e\n", row, vd, vp);
      if (!(std::fabs(vd - vp) < 1e-3 * std::max(vd, vp))) std::printf("  row %d: debiased %.6e pca %.6e\n", row, vd, vp);
      CHECK(std::fabs(vd - vp) < 1e-3 * std::max(vd, vp));
    }
    double a = 0, b = 0;
    const bool okda = deb.lineDistanceTiltVar(0.6, 0.8, 0.2, a), okpa = pca.lineDistanceTiltVar(0.6, 0.8, 0.2, b);
    CHECK(okda && okpa && std::fabs(a - b) < 1e-3 * std::max(a, b));
  }
  // DL4: mean_cov == moment with equal covariances; differs with unequal ones
  {
    const auto lineEq = cloud(rng, 0.003, 0.004, 0.8, 1e-9, false);
    VoxelPlane m(makeOpts(true, true, 2, true)), mc(makeOpts(true, true, 1, true));
    m.addPoints(lineEq, -1, 1, true, 0); mc.addPoints(lineEq, -1, 1, true, 0);
    for (int row = 0; row < 2; ++row) { bool o1, o2; const double x = tiltVar(m, row, d, o1), y = tiltVar(mc, row, d, o2); CHECK(o1 && o2 && std::fabs(x - y) < 1e-8 * std::max(x, y)); }
    VoxelPlane mu(makeOpts(true, true, 2, true)), muc(makeOpts(true, true, 1, true));
    mu.addPoints(line, -1, 1, true, 0); muc.addPoints(line, -1, 1, true, 0);
    bool o1, o2; const double x = tiltVar(mu, 0, d, o1), y = tiltVar(muc, 0, d, o2);
    CHECK(o1 && o2 && std::fabs(x - y) > 1e-4 * std::max(x, y));
  }
  // DL5: a plane cloud is not a line: no tilt matrices
  {
    VoxelPlane vp(makeOpts(true, true, 2, true));
    vp.addPoints(plane, -1, 1, true, 0);
    bool ok = true; tiltVar(vp, 0, d, ok);
    CHECK(vp.isPlane() && !ok);
  }
  // DL6: gap floor alone
  {
    const VoxelPlaneR82Counters c0 = voxelPlaneR82Counters();
    VoxelPlane vp(makeOpts(true, false, 0, false, true));
    vp.addPoints(line, -1, 1, true, 0);
    const VoxelPlaneR82Counters c1 = voxelPlaneR82Counters();
    CHECK(vp.isPlane() && !vp.isLine() && c1.gaps_floored == c0.gaps_floored + 1);
  }
  // DL7: pca fit
  {
    const VoxelPlaneR82Counters c0 = voxelPlaneR82Counters();
    VoxelPlane on(makeOpts(false, true, 0, false)), off(makeOpts(false, false, 0, false)), pl(makeOpts(false, true, 0, false));
    on.update(line); off.update(line); pl.update(plane);
    const VoxelPlaneR82Counters c1 = voxelPlaneR82Counters();
    CHECK(on.isPlane() && on.isLine() && !off.isLine() && !pl.isLine() && c1.line_classified == c0.line_classified + 1);
  }
  // DL8: a perfectly thin line
  {
    std::vector<PointXYZCov> thin;
    for (int k = 0; k < 40; ++k) { PointXYZCov p; p.point = V3D(0.02 * k, 0.0, 0.0); p.sensor_cov = M3D::Zero(); p.pos_cov = M3D::Zero(); thin.push_back(p); }
    const VoxelPlaneR82Counters c0 = voxelPlaneR82Counters();
    VoxelPlane off(makeOpts(false, false, 0, false)), on(makeOpts(false, true, 0, false));
    off.update(thin); on.update(thin);
    const VoxelPlaneR82Counters c1 = voxelPlaneR82Counters();
    CHECK(!off.isPlane());
    CHECK(on.isPlane() && on.isLine() && c1.e1_tiny_kept == c0.e1_tiny_kept + 1);
  }
  std::printf(g_fail ? "FAILED %d\n" : "ALL PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
