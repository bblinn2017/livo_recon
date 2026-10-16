// R91H unit test, reduced by R97A Q3: imu/undistort/rot_warp_scale and rot_warp_last_frame (and DeskewRotStats,
// the scaled-rotation branch of deskewPoints()) are removed -- the rotation warp now always applies the full,
// unscaled rotation Exp(gyr, dt) (what "scale 1" used to mean; the only path left). This file keeps that one
// remaining path's own check (RW1) and drops RW2-RW7, which tested the now-removed scale != 1 branch.
// Same pattern as test_point_cov.cpp (ros first, then #define private public) so it stays ROS-runtime-free.
// An isolated mathematical/identity test on fixed numbers; no number from it is evidence about the node (rule 3.1).
//   RW1 the applied rotation at point time t equals Exp(gyr, t) exactly (independent AngleAxis reference), the
//       translation part is identical between two independent calls (determinism), and raw_body_point is the
//       untransformed body-frame point.
#include <ros/ros.h>
#define private public
#include "livo_recon/utils/state/state.h"
#undef private
#include "livo_recon/lio/deskew.h"
#include "livo_recon/utils/algo/math.h"

#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

namespace {

struct Scene
{
  StateGroupPtr state;
  std::vector<Pose6D> poses;
  std::vector<PointXYZT> pts;
  double t_end = 0.1;
  V3D gyr;
};

Scene makeScene(const V3D& vel)
{
  Scene s;
  s.state = std::make_shared<StateGroup>();   // rot = I, pos = 0, R_li = I, t_li = 0
  s.gyr = V3D(0.20, -0.30, 0.10);
  // two IMU heads: t = 0.00 and 0.05 (each lasts 0.05 s); constant-axis rotation Exp(gyr * t); zero acceleration
  for (double t0 : {0.0, 0.05})
    s.poses.emplace_back(t0, V3D::Zero(), V3D::Zero(), s.gyr, vel, vel * t0, Exp(s.gyr, t0), 0.05);
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> U(-30.0, 30.0);
  for (int i = 0; i < 60; ++i)
  {
    PointXYZT p;
    p.p = V3D(U(rng), U(rng), U(rng));
    p.t = 0.0016 * (i + 1);   // ascending, all in (0, 0.0976]
    s.pts.push_back(p);
  }
  return s;
}

std::vector<PointXYZCov> run(const Scene& s)
{
  DeskewOptions o;
  o.time_based_process_noise = "none";
  std::vector<PointXYZCov> out;
  deskewPoints(s.state, s.poses, s.t_end, s.pts, o, out);
  return out;
}

// the rotation the warp applies at point time t for the constant-axis scene (R_end = I): Exp(gyr * t)
M3D rrel(const Scene& s, double t) { return Exp(s.gyr, t); }

}  // namespace

int main()
{
  const Scene s0 = makeScene(V3D::Zero());
  const size_t n = s0.pts.size();

  {  // RW1
    const auto a = run(s0), b = run(s0);
    double worst_det = 0.0, worst_rot = 0.0, minrot = 1e9;
    for (size_t i = 0; i < n; ++i)
    {
      worst_det = std::max(worst_det, (a[i].point - b[i].point).norm());
      const M3D R_full = rrel(s0, s0.pts[i].t);
      worst_rot = std::max(worst_rot, (a[i].point - R_full * s0.pts[i].p).norm());
      minrot = std::min(minrot, (a[i].point - s0.pts[i].p).norm());
      CHECK((a[i].raw_body_point - s0.pts[i].p).norm() == 0.0, "RW1 raw_body_point is the untransformed body point");
    }
    CHECK(worst_det == 0.0, "RW1 two independent calls are bit-identical (deterministic)");
    CHECK(worst_rot < 1e-11, "RW1 the applied rotation equals Exp(gyr, t) exactly (independent AngleAxis reference)");
    CHECK(minrot > 1e-3, "RW1 the warp really rotates (the check above is not vacuous)");
  }
  std::printf("ALL PASS\n");
  return 0;
}
