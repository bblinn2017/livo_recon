// R82 (F-506): unit test of the first-pose alignment (header only, ROS-free). Algebraic identities (rule 3.1). Planner sandbox, not evidence.
#include <cstdio>
#include <random>
#include <vector>

#include "livo_recon/utils/algo/first_pose_align.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

struct MP { double t; Eigen::Vector3d est_pos; Eigen::Matrix3d est_rot; Eigen::Vector3d gt_pos; Eigen::Matrix3d gt_rot; };

static Eigen::Matrix3d rotz(double a) { return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix(); }

int main()
{
  std::mt19937 rng(7); std::normal_distribution<double> nd(0.0, 1.0);
  // A rigid transform T between the two frames: gt = T est exactly. The first-pose alignment must recover T and give zero error.
  const Eigen::Matrix3d Rt = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  const Eigen::Vector3d tt(3.0, -2.0, 0.5);
  std::vector<MP> v;
  for (int i = 0; i < 50; ++i)
  {
    MP m; m.t = i * 0.1;
    m.est_pos = Eigen::Vector3d(0.1 * i, std::sin(0.1 * i), 0.01 * i * i);
    m.est_rot = rotz(0.02 * i) * Eigen::AngleAxisd(0.01 * i, Eigen::Vector3d::UnitX()).toRotationMatrix();
    m.gt_pos = Rt * m.est_pos + tt; m.gt_rot = Rt * m.est_rot;
    v.push_back(m);
  }
  double a, r; Eigen::Matrix3d R0; Eigen::Vector3d t0;
  CHECK(firstPoseAlignedErrors(v, a, r, R0, t0));
  CHECK(a < 1e-12 && r < 1e-6);
  CHECK((R0 - Rt).norm() < 1e-12 && (t0 - tt).norm() < 1e-12);
  // The first pose has zero error by construction, whatever the later errors are.
  std::vector<MP> w = v; for (size_t i = 1; i < w.size(); ++i) { w[i].gt_pos += 0.01 * Eigen::Vector3d(nd(rng), nd(rng), nd(rng)); w[i].gt_rot = rotz(0.003 * nd(rng)) * w[i].gt_rot; }
  CHECK(firstPoseAlignedErrors(w, a, r, R0, t0));
  CHECK(((R0 * w[0].est_pos + t0) - w[0].gt_pos).norm() < 1e-12);
  CHECK(a > 1e-4 && r > 1e-3);
  // A constant position offset of the estimate in the GT frame after the first pose is the RMS offset (the first pose has 0): sqrt((n-1)/n) * d.
  std::vector<MP> q = v; const double d = 0.05; for (size_t i = 1; i < q.size(); ++i) q[i].gt_pos += Eigen::Vector3d(d, 0, 0);
  firstPoseAlignedErrors(q, a, r, R0, t0);
  CHECK(std::fabs(a - d * std::sqrt(49.0 / 50.0)) < 1e-12);
  // A constant rotation offset: RMS angle sqrt((n-1)/n) * theta.
  std::vector<MP> z = v; const double th = 0.02; for (size_t i = 1; i < z.size(); ++i) z[i].gt_rot = rotz(th) * z[i].gt_rot;
  firstPoseAlignedErrors(z, a, r, R0, t0);
  CHECK(std::fabs(r - th * std::sqrt(49.0 / 50.0) * 180.0 / M_PI) < 1e-9);
  // Fewer than 3 poses: false.
  std::vector<MP> two(v.begin(), v.begin() + 2);
  CHECK(!firstPoseAlignedErrors(two, a, r, R0, t0));
  std::printf(g_fail ? "FAILED %d\n" : "ALL PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
