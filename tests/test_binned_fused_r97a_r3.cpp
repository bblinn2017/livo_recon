// R97A R3: voxel_map/binned/cov_mode = fused | fused_pose (include/livo_recon/map/binned/binned_bins.h).
// Eigen + std only; own check macro (does not depend on NDEBUG), same pattern as test_binned_bins_r88.cpp.
//   F1 one return per bin: fused equals the return and its covariance.
//   F2 k identical returns with equal Sigma: p* equals the arithmetic mean, covariance equals Sigma/k.
//   F3 two returns with different anisotropic covariances: p* and covariance equal the closed-form
//      precision-weighted values (computed independently here, not via the production code).
//   F4 fused_pose covariance minus fused covariance equals s^2 I.
//   F5 (identity) average/shrunk are not exercised here at all -- test_binned_bins_r88.cpp's own
//      existing checks are untouched by this patch and remain the identity evidence for those two modes.
#include "livo_recon/map/binned/binned_bins.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace livo_recon::binned;

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static double maxAbs(const M3& a, const M3& b) { return (a - b).cwiseAbs().maxCoeff(); }

int main()
{
  const double bs = 0.05;

  // F1: one return per bin -- fused equals the return and its covariance exactly (a bin of one
  // return is the return: L = Sigma^-1, Lp = Sigma^-1 p, p* = L^-1 Lp = p, cov = L^-1 = Sigma).
  {
    Options o; o.cov_mode = CovMode::Fused;
    BinSet bins(bs, o);
    M3 Sigma = M3::Identity() * 1e-4; Sigma(0, 1) = Sigma(1, 0) = 3e-5;
    const V3 p(1.01, 2.02, 0.51);
    bins.add(p, Sigma);
    const Reps r = bins.representatives(o);
    CK(r.p.size() == 1);
    CK((r.p[0] - p).norm() < 1e-9);
    CK(maxAbs(r.cov[0], Sigma) < 1e-9);
    CK(std::fabs(r.w[0] - 1.0) < 1e-12);
  }

  // F2: k identical returns with equal Sigma -- p* = arithmetic mean = the point, cov = Sigma/k.
  {
    Options o; o.cov_mode = CovMode::Fused;
    BinSet bins(bs, o);
    M3 Sigma = M3::Identity() * 2e-4; Sigma(0, 2) = Sigma(2, 0) = -4e-5;
    const V3 p(0.3, -0.2, 1.1);
    const int k = 5;
    for (int i = 0; i < k; ++i) bins.add(p, Sigma);
    const Reps r = bins.representatives(o);
    CK(r.p.size() == 1);
    CK((r.p[0] - p).norm() < 1e-9);
    CK(maxAbs(r.cov[0], Sigma / static_cast<double>(k)) < 1e-9);
    CK(std::fabs(r.w[0] - static_cast<double>(k)) < 1e-12);
  }

  // F3: two returns with different anisotropic covariances -- closed-form precision-weighted
  // fusion, computed independently of the production code.
  {
    Options o; o.cov_mode = CovMode::Fused;
    BinSet bins(bs, o);
    M3 S1 = M3::Identity(); S1(0, 0) = 1e-3; S1(1, 1) = 5e-5; S1(2, 2) = 2e-4;
    M3 S2 = M3::Identity(); S2(0, 0) = 2e-5; S2(1, 1) = 8e-4; S2(2, 2) = 3e-4; S2(0, 2) = S2(2, 0) = 5e-6;
    // Both points must floor to the same bin at bin size bs=0.05 -- stay well inside [1.0, 1.05) x
    // [0.0, 0.05) x [0.0, 0.05) (a point like z=-0.01 would floor into the bin below z=0's).
    const V3 p1(1.01, 0.01, 0.01), p2(1.02, 0.03, 0.02);
    bins.add(p1, S1);
    bins.add(p2, S2);
    const Reps r = bins.representatives(o);
    CK(r.p.size() == 1);

    const M3 L1 = S1.inverse(), L2 = S2.inverse();
    const M3 L = L1 + L2;
    const V3 Lp = L1 * p1 + L2 * p2;
    const M3 L_inv = L.inverse();
    const V3 p_star = L_inv * Lp;

    CK((r.p[0] - p_star).norm() / std::max(1e-12, p_star.norm()) < 1e-9);
    CK(maxAbs(r.cov[0], L_inv) / std::max(1e-12, L_inv.norm()) < 1e-9);
    CK(std::fabs(r.w[0] - 2.0) < 1e-12);
  }

  // F4: fused_pose covariance minus fused covariance equals s^2 I.
  {
    Options o_fused; o_fused.cov_mode = CovMode::Fused;
    Options o_pose; o_pose.cov_mode = CovMode::FusedPose; o_pose.fused_pose_sigma_m = 0.07;

    // One BinSet constructed with cov_mode = Fused (add()'s accumulation is identical for Fused
    // and FusedPose -- only representatives() differs, by the added s^2*I term), so the same L/Lp
    // sums can be read back through both Options values.
    BinSet bins(bs, o_fused);
    M3 S1 = M3::Identity() * 4e-4; S1(0, 1) = S1(1, 0) = 1e-4;
    M3 S2 = M3::Identity() * 9e-4;
    // Both points must floor to the same bin at bin size bs=0.05 -- see the F3 comment above.
    bins.add(V3(0.51, 0.51, 0.51), S1);
    bins.add(V3(0.52, 0.53, 0.54), S2);

    const Reps r_fused = bins.representatives(o_fused);
    const Reps r_pose = bins.representatives(o_pose);
    CK(r_fused.p.size() == 1 && r_pose.p.size() == 1);
    CK((r_fused.p[0] - r_pose.p[0]).norm() < 1e-12);   // the representative point itself is unchanged

    const M3 diff = r_pose.cov[0] - r_fused.cov[0];
    const M3 expect = (o_pose.fused_pose_sigma_m * o_pose.fused_pose_sigma_m) * M3::Identity();
    CK(maxAbs(diff, expect) < 1e-12);
  }

  if (fails == 0) std::printf("ALL PASS (R97A R3: fused/fused_pose)\n");
  else std::printf("%d CHECK(S) FAILED\n", fails);
  return fails == 0 ? 0 : 1;
}
