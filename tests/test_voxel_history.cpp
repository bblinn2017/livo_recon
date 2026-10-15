// R65 unit test (F-158), rewritten in R97 (the debiased path is removed): a binned voxel keeps its raw history and hands ALL of it to its children on subdivision.
// Same private-access pattern as test_gravity_s2.cpp (standard headers first, then #define private public).
//   VH1 precondition: after batch A (planar points, frame 1) the binned voxel is a plane.
//   VH2 after batch B (off-plane scatter, frame 2) the voxel has subdivided (status PARENT, plane retired).
//   VH3 the children's bin accumulators together hold |A|+|B| returns (before F-158 they held |B| only), with
//       max_points=1000 (batch stored branch). The batch-not-stored branch (points_ already at max_points when B arrives) is not covered here: with a small max_points the children lock
//       once full and drop their bins, so their returns cannot be counted. Counted as returns in the children's BinSet (numPoints()), not as occupied bins.
//   VH4 the parent's points_ is empty after the hand-off.
//   VH5 plain (unbinned) pca control: the same scenario with binned_enabled=false; children hold at least the last batch (not asserted to be all of it).
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
  opts->binned_enabled = (mode == "binned");
  opts->binned_bin_fraction = 0.1;
  opts->binned_cov_mode = 0;
  opts->max_points = max_points;
  opts->max_layer = 1;
  opts->voxel_size = 0.5;
  opts->min_init_points = 5;
  opts->min_update_points = 5;
  auto stats = std::make_shared<VoxelStats>();
  VoxelNode node(opts, stats, 0, V3D::Zero());
  std::vector<PlaneUpdate> updates;
  // R66: batch B enlarged (was 60) so the planar share of the pooled points is 60/460 and the off-plane scatter is
  // unambiguous; VH2 printed no state when it failed in R65, so the node state is now printed before the check.
  const int nA = 60, nB = 400;
  setCurrentFrame(1);
  node.insertPoints(planar(rng, nA), updates);
  const std::string tag = mode + " max_points=" + std::to_string(max_points);
  if (mode == "binned") CHECK(node.plane_.isPlane(), ("VH1 planar batch gives a plane (" + tag + ")").c_str());
  setCurrentFrame(2);
  node.insertPoints(scatter(rng, nB), updates);
  std::printf("[INFO] after batch B (%s): status=%d is_plane=%d pointsSize=%d\n", tag.c_str(),
              static_cast<int>(node.status_), node.plane_.isPlane() ? 1 : 0, static_cast<int>(node.plane_.pointsSize()));
  CHECK(node.status_ == VoxelStatus::PARENT, ("VH2 voxel subdivided (" + tag + ")").c_str());
  int sum = 0;
  for (int i = 0; i < 8; ++i) {
    if (!node.leaves_[i]) continue;
    if (expect_history) { if (node.leaves_[i]->binset_) sum += static_cast<int>(node.leaves_[i]->binset_->numPoints()); }
    else sum += node.leaves_[i]->plane_.pointsSize();
  }
  if (expect_history)
    CHECK(sum == nA + nB, ("VH3 children hold all history " + std::to_string(sum) + " == " + std::to_string(nA + nB) + " (" + tag + ")").c_str());
  else
    CHECK(sum >= nB, ("VH5 control children hold at least the last batch, got " + std::to_string(sum) + " (" + tag + ")").c_str());
  CHECK(node.points_.empty(), ("VH4 parent points_ cleared (" + tag + ")").c_str());
}

}  // namespace

int main()
{
  scenario("binned", 1000, true);
  scenario("pca", 1000, false);
  std::printf("ALL PASS\n");
  return 0;
}
