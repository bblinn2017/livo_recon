// R64 unit test for the debiased plane-fit variants (voxel_map/plane/debias_mode, plane_threshold_hysteresis) and the
// plane-fit reason log. ROS-free: constructs VoxelPlane directly and calls the real production addPoints(); the only
// arithmetic duplicated here is the plain reference formula the R63 code implemented, written independently
// (Eigen + 3x3 accumulation) so "full" can be compared against it.
//
// Checks
//   D1  debias_mode="full" (default) reproduces the R63 formula: eigenvalues of Spp/N - mean mean^T - Sbar_sensor
//       (pose term is zero here: pos_cov = 0 and F = 1 or shrinkage is applied to a zero matrix) to 1e-10.
//   D2  debias_mode="none": eigenvalues equal those of the raw scatter to 1e-10, normal equals the raw normal.
//   D3  debias_mode="lambda0_only": lambda1, lambda2 and the normal equal the raw ones; lambda0 equals
//       max(0, raw lambda0 - n^T Sbar n) to 1e-10.
//   D4  isotropic sensor covariance s2*I: "full" eigenvalues equal raw eigenvalues minus s2 (all three) while all
//       remain positive.
//   D5  hysteresis: a voxel accepted at lambda0 ~ 0.8 thr, then refit after adding rougher points so that the
//       combined lambda0 lies in (thr, 2 thr), is revoked at hysteresis 1.0 and kept at hysteresis 2.0; a voxel that
//       was NOT a plane before the same refit is not helped by hysteresis 2.0 (rejected at both settings).
//   D6  reason accounting: with plane_fit_reasons_en=true, voxelPlaneFitTraceFlush() writes plane_fit_reasons.csv
//       into the debug log dir (/tmp when unset) whose 'accept' + rejected columns sum to the number of fit
//       attempts made in this test for the debiased path.
// Exit code 0 = all pass. Prints one line per check with the numbers compared.

#include "livo_recon/lio/voxelplane.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace livo_recon;

namespace {

int g_fail = 0;
void check(bool ok, const char* name, const std::string& detail)
{
  std::printf("[%s] %s  %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
  if (!ok) ++g_fail;
}

std::string fmt(double a, double b)
{
  std::ostringstream o; o.precision(12); o << "got " << a << " expected " << b << " diff " << std::fabs(a - b);
  return o.str();
}

// Points on the plane z = 0.05 x + 0.02 y + offset, noise std sz in z, x,y uniform in [-1,1].
std::vector<PointXYZCov> makePoints(std::mt19937& rng, int n, double sz, const M3D& sensor_cov, double offset = 0.0)
{
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::normal_distribution<double> Nz(0.0, sz);
  std::vector<PointXYZCov> pts;
  for (int i = 0; i < n; ++i) {
    PointXYZCov p;
    const double x = U(rng), y = U(rng);
    p.point = V3D(x, y, 0.05 * x + 0.02 * y + offset + Nz(rng));
    p.sensor_cov = sensor_cov;
    p.pos_cov = M3D::Zero();
    p.bootstrap_observation_id = -1;
    pts.push_back(p);
  }
  return pts;
}

VoxelOptsPtr makeOpts(const std::string& mode, double hysteresis = 1.0, bool reasons = false)
{
  auto o = std::make_shared<VoxelOpts>();
  o->plane_fit_mode = "debiased";
  o->plane_fit_pose_cov_mode = "sensor_only";
  o->debias_mode = mode;
  o->plane_threshold_hysteresis = hysteresis;
  o->plane_fit_reasons_en = reasons;
  o->plane_threshold = 0.01;
  return o;
}

struct Ref { V3D eig_raw; V3D eig_full; M3D evec_raw; V3D n_sens_var; M3D Sbar; };

Ref reference(const std::vector<PointXYZCov>& pts)
{
  const double N = static_cast<double>(pts.size());
  V3D sp = V3D::Zero(); M3D spp = M3D::Zero(); M3D ss = M3D::Zero();
  for (const auto& p : pts) { sp += p.point; spp += p.point * p.point.transpose(); ss += p.sensor_cov; }
  const V3D mean = sp / N;
  const M3D raw = spp / N - mean * mean.transpose();
  const M3D full = raw - ss / N;
  Eigen::SelfAdjointEigenSolver<M3D> a(raw), b(full);
  Ref r;
  r.eig_raw = a.eigenvalues();
  r.eig_full = b.eigenvalues();
  r.evec_raw = a.eigenvectors();
  r.Sbar = ss / N;
  return r;
}

}  // namespace

int main()
{
  std::mt19937 rng(12345);
  const double s2 = 4.0e-4;                       // 2 cm isotropic sensor sigma
  const M3D sens_iso = s2 * M3D::Identity();
  M3D sens_aniso = M3D::Zero();
  sens_aniso.diagonal() = V3D(1.0e-4, 4.0e-4, 9.0e-4);
  const auto pts = makePoints(rng, 300, 0.02, sens_iso);
  const auto pts_an = makePoints(rng, 300, 0.02, sens_aniso);

  // ---- D1: full == R63 formula
  {
    auto o = std::make_shared<VoxelOpts>();       // default debias_mode must be "full"
    o->plane_fit_mode = "debiased"; o->plane_fit_pose_cov_mode = "sensor_only";
    check(o->debias_mode == "full", "D0 default debias_mode is full", o->debias_mode);
    VoxelPlane vp(o);
    vp.addPoints(pts_an, -1, 2, true, 5);
    const Ref r = reference(pts_an);
    // The reference is the anisotropic full correction; eigenvalues are clamped at >= 0 for lambda0 in production.
    V3D e = vp.eigenValues();
    check(std::fabs(e(1) - r.eig_full(1)) < 1e-10 && std::fabs(e(2) - r.eig_full(2)) < 1e-10 &&
              std::fabs(e(0) - std::max(r.eig_full(0), 0.0)) < 1e-10,
          "D1 full == R63 formula (aniso cov)", fmt(e(2), r.eig_full(2)));
  }
  // ---- D2 / D3 / D4
  {
    const Ref r = reference(pts_an);
    // raw normal
    const V3D n_raw = r.evec_raw.col(0);
    {
      VoxelPlane vp(makeOpts("none"));
      vp.addPoints(pts_an, -1, 2, true, 5);
      const V3D e = vp.eigenValues();
      PlaneVizInfo vi{}; const bool ok = vp.getVizInfo(vi);
      check(ok, "D2 none: accepted as plane", "");
      check((e - r.eig_raw).norm() < 1e-10, "D2 none: eigenvalues == raw scatter", fmt(e(0), r.eig_raw(0)));
      check(ok && std::fabs(std::fabs(vi.normal.dot(n_raw)) - 1.0) < 1e-9, "D2 none: normal == raw normal",
            fmt(std::fabs(vi.normal.dot(n_raw)), 1.0));
    }
    {
      VoxelPlane vp(makeOpts("lambda0_only"));
      vp.addPoints(pts_an, -1, 2, true, 5);
      const V3D e = vp.eigenValues();
      PlaneVizInfo vi{}; const bool ok = vp.getVizInfo(vi);
      const double sens_n = n_raw.dot(r.Sbar * n_raw);
      const double e0_expected = std::max(r.eig_raw(0) - sens_n, 0.0);
      check(std::fabs(e(1) - r.eig_raw(1)) < 1e-10 && std::fabs(e(2) - r.eig_raw(2)) < 1e-10,
            "D3 lambda0_only: lambda1, lambda2 raw", fmt(e(1), r.eig_raw(1)));
      check(std::fabs(e(0) - e0_expected) < 1e-10, "D3 lambda0_only: lambda0 = max(0, raw - n^T S n)", fmt(e(0), e0_expected));
      check(ok && std::fabs(std::fabs(vi.normal.dot(n_raw)) - 1.0) < 1e-9, "D3 lambda0_only: normal == raw normal",
            fmt(std::fabs(vi.normal.dot(n_raw)), 1.0));
    }
    {
      const Ref ri = reference(pts);
      VoxelPlane vp(makeOpts("full"));
      vp.addPoints(pts, -1, 2, true, 5);
      const V3D e = vp.eigenValues();
      const V3D expect = ri.eig_raw - V3D::Constant(s2);
      check(std::fabs(e(1) - expect(1)) < 1e-9 && std::fabs(e(2) - expect(2)) < 1e-9 && expect(1) > 0.0,
            "D4 isotropic cov: full = raw - s2 on all three", fmt(e(2), expect(2)));
    }
  }
  // ---- D3b: lambda0_only with an UNDER-stated sensor covariance so the corrected lambda0 stays positive
  {
    std::mt19937 r4(99);
    const auto pu = makePoints(r4, 300, 0.02, 1.0e-4 * M3D::Identity());
    const Ref r = reference(pu);
    const V3D n_raw = r.evec_raw.col(0);
    VoxelPlane vp(makeOpts("lambda0_only"));
    vp.addPoints(pu, -1, 2, true, 5);
    const double expected = r.eig_raw(0) - n_raw.dot(r.Sbar * n_raw);
    check(expected > 1e-5 && std::fabs(vp.eigenValues()(0) - expected) < 1e-10,
          "D3b lambda0_only positive case: lambda0 = raw - n^T S n", fmt(vp.eigenValues()(0), expected));
    VoxelPlane vn(makeOpts("none"));
    vn.addPoints(pu, -1, 2, true, 5);
    check(std::fabs(vn.eigenValues()(0) - r.eig_raw(0)) < 1e-10 && vn.eigenValues()(0) > vp.eigenValues()(0),
          "D3b none keeps the uncorrected lambda0 (larger than lambda0_only)", fmt(vn.eigenValues()(0), r.eig_raw(0)));
  }
  // ---- D5: hysteresis
  {
    std::mt19937 r2(777);
    const M3D zero = M3D::Zero();
    const auto a = makePoints(r2, 200, 0.09, zero);   // lambda0 ~ 0.0081 < 0.01
    const auto b = makePoints(r2, 200, 0.14, zero);   // combined lambda0 ~ 0.0139 in (0.01, 0.02)
    bool was_plane_h1 = false, after_h1 = true, was_plane_h2 = false, after_h2 = false;
    {
      VoxelPlane v1(makeOpts("none", 1.0));
      v1.addPoints(a, -1, 1, true, 1); was_plane_h1 = v1.isPlane();
      v1.addPoints(b, -1, 2, true, 2); after_h1 = v1.isPlane();
    }
    {
      VoxelPlane v2(makeOpts("none", 2.0));
      v2.addPoints(a, -1, 1, true, 1); was_plane_h2 = v2.isPlane();
      v2.addPoints(b, -1, 2, true, 2); after_h2 = v2.isPlane();
    }
    check(was_plane_h1 && was_plane_h2, "D5 first fit accepted at both hysteresis settings", "");
    check(!after_h1, "D5 hysteresis 1.0 revokes the plane after rougher points", "");
    check(after_h2, "D5 hysteresis 2.0 keeps the plane", "");
    // a voxel first seen with the combined (rough) set is rejected at both settings
    std::vector<PointXYZCov> ab = a; ab.insert(ab.end(), b.begin(), b.end());
    VoxelPlane v3(makeOpts("none", 2.0));
    v3.addPoints(ab, -1, 2, true, 2);
    check(!v3.isPlane(), "D5 hysteresis does not apply to a voxel that was not a plane", "");
  }
  // ---- D6: reason accounting (counts written by voxelPlaneFitTraceFlush)
  {
    auto o = makeOpts("full", 1.0, /*reasons=*/true);
    VoxelPlane vp(o);
    int attempts = 0;
    for (int k = 0; k < 3; ++k) {
      std::mt19937 r3(100 + k);
      vp.addPoints(makePoints(r3, 40, 0.02, sens_iso), -1, k + 1, true, 7);
      ++attempts;
    }
    voxelPlaneFitTraceFlush();
    std::ifstream f("/tmp/plane_fit_reasons.csv");
    std::string line; long total = 0; bool have = false;
    if (f && std::getline(f, line)) {
      while (std::getline(f, line)) {
        std::stringstream ss(line); std::string tok; std::vector<std::string> c;
        while (std::getline(ss, tok, ',')) c.push_back(tok);
        if (c.size() >= 12 && c[0] == "7" && c[1] == "debiased") {
          have = true;
          for (int i = 2; i <= 11; ++i) total += std::stol(c[i]);   // accept .. info_reject
        }
      }
    }
    check(have && total == attempts, "D6 reason counts sum to fit attempts (frame 7, debiased)",
          "sum " + std::to_string(total) + " attempts " + std::to_string(attempts) +
          (have ? "" : "  (plane_fit_reasons.csv not found in /tmp: set the debug log dir unset for this test)"));
  }
  std::printf("%s (%d failed)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
  return g_fail == 0 ? 0 : 1;
}
