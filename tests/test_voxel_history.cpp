// R65 unit test (F-158): a debiased voxel keeps its raw history and hands ALL of it to its children on subdivision.
// Same private-access pattern as test_gravity_s2.cpp (standard headers first, then #define private public).
//   VH1 precondition: after batch A (planar points, frame 1) the debiased voxel is a plane.
//   VH2 after batch B (off-plane scatter, frame 2) the voxel has subdivided (status PARENT, plane retired).
//   VH3 the children's accumulated point counts sum to |A|+|B| (before F-158 they summed to |B| only), for both
//       max_points=50 (points_ already at the bound when B arrives, so the batch-not-stored branch) and max_points=1000
//       (batch stored branch).
//   VH4 the parent's points_ is empty after the hand-off.
//   VH5 pca control: the same scenario with plane_fit_mode=pca; children hold at least the last batch (not asserted to be all of it).
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

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

namespace {

std::vector<PointXYZCov> planar(std::mt19937& rng, int n)
{
  std::uniform_real_distribution<double> U(-0.2, 0.2);
  std::vector<PointXYZCov> v;
  for (int i = 0; i < n; ++i) {
    PointXYZCov p; p.point = V3D(U(rng), U(rng), 0.0); p.sensor_cov = 1e-8 * M3D::Identity(); v.push_back(p);
  }
  return v;
}
std::vector<PointXYZCov> scatter(std::mt19937& rng, int n)
{
  std::uniform_real_distribution<double> U(-0.2, 0.2);
  std::vector<PointXYZCov> v;
  for (int i = 0; i < n; ++i) {
    PointXYZCov p; p.point = V3D(U(rng), U(rng), U(rng)); p.sensor_cov = 1e-8 * M3D::Identity(); v.push_back(p);
  }
  return v;
}

void scenario(const std::string& mode, int max_points, bool expect_history)
{
  std::mt19937 rng(11);
  auto opts = std::make_shared<VoxelOpts>();
  opts->plane_fit_mode = mode;
  opts->max_points = max_points;
  opts->max_layer = 1;
  opts->voxel_size = 0.5;
  opts->min_init_points = 5;
  opts->min_update_points = 5;
  auto stats = std::make_shared<VoxelStats>();
  VoxelNode node(opts, stats, 0, V3D::Zero());
  std::vector<PlaneUpdate> updates;
  const int nA = 60, nB = 60;
  setCurrentFrame(1);
  node.insertPoints(planar(rng, nA), updates);
  const std::string tag = mode + " max_points=" + std::to_string(max_points);
  if (mode == "debiased") CHECK(node.plane_.isPlane(), ("VH1 planar batch gives a plane (" + tag + ")").c_str());
  setCurrentFrame(2);
  node.insertPoints(scatter(rng, nB), updates);
  CHECK(node.status_ == VoxelStatus::PARENT, ("VH2 voxel subdivided (" + tag + ")").c_str());
  int sum = 0;
  for (int i = 0; i < 8; ++i)
    if (node.leaves_[i]) sum += node.leaves_[i]->plane_.pointsSize();
  if (expect_history)
    CHECK(sum == nA + nB, ("VH3 children hold all history " + std::to_string(sum) + " == 120 (" + tag + ")").c_str());
  else
    CHECK(sum >= nB, ("VH5 control children hold at least the last batch, got " + std::to_string(sum) + " (" + tag + ")").c_str());
  CHECK(node.points_.empty(), ("VH4 parent points_ cleared (" + tag + ")").c_str());
}

}  // namespace

int main()
{
  scenario("debiased", 50, true);
  scenario("debiased", 1000, true);
  scenario("pca", 1000, false);
  std::printf("ALL PASS\n");
  return 0;
}
