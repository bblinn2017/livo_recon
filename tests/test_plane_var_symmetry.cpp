// R74 unit test (F-419 extension): the plane covariance plane_var_ of a fitted voxel plane is exactly symmetric.
// Same private-access pattern as test_voxel_history.cpp (standard headers first, then #define private public).
//   PV1 precondition: the planar batch gives a plane (debiased and pca) and plane_var_ is not the zero matrix (so the check is not vacuous).
//   PV2 asymmetryMax(plane_.planeVar()) == 0 exactly, with full (non-diagonal) random sensor and pose covariances on every point, for
//       plane_fit_mode debiased and pca (R93A: the `combined` pose mode was removed), and plane_var_mode default (eigengap) and information, plus pca + information_directional (the H.inverse() path).
//   PV3 control: a copy of the matrix with one off-diagonal pair perturbed by 1e-12 is reported as asymmetric (the check can fail).
//   PV4 the symmetrisation does not change the value beyond rounding: |planeVar - planeVar^T| is 0 and planeVar is positive semidefinite
//       to 1e-12 relative (smallest eigenvalue >= -1e-12 trace).
// Exit code 0 = all pass.
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
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
#undef private

#include "livo_recon/utils/algo/symmetrize.h"

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

namespace {

M3D randomSpd(std::mt19937& rng, double scale)
{
  std::normal_distribution<double> N(0, 1);
  M3D B;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) B(i, j) = N(rng);
  return scale * (B * B.transpose() + 0.1 * M3D::Identity());
}

std::vector<PointXYZCov> planarWithCov(std::mt19937& rng, int n)
{
  std::uniform_real_distribution<double> U(-0.2, 0.2);
  std::normal_distribution<double> Nz(0, 1e-3);
  std::vector<PointXYZCov> v;
  for (int i = 0; i < n; ++i) {
    PointXYZCov p;
    p.point = V3D(U(rng), U(rng), Nz(rng));
    p.sensor_cov = randomSpd(rng, 1e-6);
    p.pos_cov = randomSpd(rng, 1e-7);
    v.push_back(p);
  }
  return v;
}

void run(const std::string& fit_mode, const std::string& var_mode)
{
  std::mt19937 rng(74);
  auto opts = std::make_shared<VoxelOpts>();
  opts->plane_fit_mode = fit_mode;
  if (!var_mode.empty()) opts->plane_var_mode = var_mode;
  opts->max_points = 1000;
  opts->max_layer = 1;
  opts->voxel_size = 0.5;
  opts->min_init_points = 5;
  opts->min_update_points = 5;
  auto stats = std::make_shared<VoxelStats>();
  VoxelNode node(opts, stats, 0, V3D::Zero());
  std::vector<PlaneUpdate> updates;
  setCurrentFrame(1);
  node.insertPoints(planarWithCov(rng, 80), updates);
  const std::string tag = fit_mode + "/" + (var_mode.empty() ? "default" : var_mode);
  CHECK(node.plane_.isPlane(), ("PV1 planar batch gives a plane (" + tag + ")").c_str());
  const M3D V = node.plane_.planeVar();
  CHECK(V.cwiseAbs().maxCoeff() > 0.0, ("PV1 plane_var_ is not the zero matrix (" + tag + ")").c_str());
  CHECK(asymmetryMax(V) == 0.0, ("PV2 plane_var_ exactly symmetric (" + tag + ")").c_str());
  Eigen::SelfAdjointEigenSolver<M3D> es(V);
  CHECK(es.eigenvalues().minCoeff() >= -1e-12 * V.trace(), ("PV4 plane_var_ positive semidefinite to 1e-12 trace (" + tag + ")").c_str());
  M3D bump = V;
  bump(0, 1) += 1e-12;
  CHECK(asymmetryMax(bump) > 0.0, ("PV3 control: a perturbed pair is detected (" + tag + ")").c_str());
}

}  // namespace

int main()
{
  for (const char* fit : {"debiased", "pca"})
    for (const char* var : {"", "information"})
      run(fit, var);
  run("pca", "information_directional");   // the only combination that reaches plane_var_ = H.inverse() (debiased + directional cannot engage, voxelmap.cpp P7)
  std::printf("ALL PASS\n");
  return 0;
}
