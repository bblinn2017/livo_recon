// R91H unit test: the rotation-warp scale of deskewPoints() (DeskewOptions::rot_warp_scale, DeskewRotStats).
// Same pattern as test_point_cov.cpp (ros first, then #define private public) so it stays ROS-runtime-free. Uses its own CHECK macro (not assert),
// so NDEBUG does not matter. An isolated mathematical/identity test on fixed numbers; no number from it is evidence about the node (rule 3.1).
//   RW1 scale 1.0 with stats is bit-identical to the default options without stats (the baseline path), on every output field.
//   RW2 scale 0: every point equals the raw point plus the translation part of the warp (zero here), and equals the scale-1 point minus the rotation part.
//   RW3 scale 0.5: the applied rotation is the scale-1 rotation with half the angle about the same axis (independent AngleAxis reference).
//   RW4 counters: points, points_rescaled, mean theta (= |gyr| * t for the constant-axis poses used) and applied = 0.5 * raw.
//   RW5 the point covariance is rotated by the same applied rotation (scale 0: equals the symmetrized sensor model in the IMU frame).
//   RW6 the translation part is untouched by the scale (non-zero velocity): (out_scale0 - p) equals (out_scale1 - R_rel p).
//   RW7 raw_body_point is the same for every scale.
// Build the variant with -DR91H_BREAK to see the test fail once (RW2's expectation is then deliberately wrong): the CHECK macro exits non-zero.
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

std::vector<PointXYZCov> run(const Scene& s, double scale, DeskewRotStats* st)
{
  DeskewOptions o;
  o.time_based_process_noise = "none";
  o.rot_warp_scale = scale;
  std::vector<PointXYZCov> out;
  deskewPoints(s.state, s.poses, s.t_end, s.pts, o, out, st);
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
    DeskewOptions o; o.time_based_process_noise = "none";
    std::vector<PointXYZCov> base; deskewPoints(s0.state, s0.poses, s0.t_end, s0.pts, o, base);
    DeskewRotStats st; const auto with = run(s0, 1.0, &st);
    bool same = base.size() == with.size();
    for (size_t i = 0; same && i < n; ++i)
      same = (base[i].point - with[i].point).norm() == 0.0 && (base[i].sensor_cov - with[i].sensor_cov).norm() == 0.0 &&
             (base[i].raw_body_point - with[i].raw_body_point).norm() == 0.0 && base[i].t == with[i].t;
    CHECK(same, "RW1 scale 1 with counters is bit-identical to the default call (point, covariance, raw point, time)");
    CHECK(st.points == static_cast<long>(n) && st.points_rescaled == 0, "RW1 counters at scale 1: all points counted, none rescaled");
  }
  {  // RW2
    const auto o0 = run(s0, 0.0, nullptr);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) worst = std::max(worst, (o0[i].point - s0.pts[i].p).norm());
#ifdef R91H_BREAK
    CHECK(worst > 1e-3, "RW2 (BROKEN EXPECTATION, must fail) scale 0 point differs from the raw point");
#else
    CHECK(worst == 0.0, "RW2 scale 0: point equals the raw point exactly when the translation part of the warp is zero");
#endif
  }
  {  // RW3
    const auto o1 = run(s0, 1.0, nullptr), oh = run(s0, 0.5, nullptr);
    double worst = 0.0, minrot = 1e9;
    for (size_t i = 0; i < n; ++i)
    {
      Eigen::AngleAxisd aa(rrel(s0, s0.pts[i].t));
      const M3D Rh = Eigen::AngleAxisd(0.5 * aa.angle(), aa.axis()).toRotationMatrix();
      worst = std::max(worst, (oh[i].point - Rh * s0.pts[i].p).norm());
      minrot = std::min(minrot, (o1[i].point - s0.pts[i].p).norm());
    }
    CHECK(worst < 1e-11, "RW3 scale 0.5 equals the half-angle rotation about the same axis (independent AngleAxis reference)");
    CHECK(minrot > 1e-3, "RW3 the scale-1 warp really rotates (the check above is not vacuous)");
  }
  {  // RW4
    DeskewRotStats st; run(s0, 0.5, &st);
    double th = 0.0; for (const auto& p : s0.pts) th += s0.gyr.norm() * p.t;
    CHECK(st.points == static_cast<long>(n) && st.points_rescaled == static_cast<long>(n), "RW4 every point counted and rescaled at scale 0.5");
    CHECK(std::abs(st.sum_theta_raw - th) < 1e-9, "RW4 sum of theta equals |gyr| * t summed over the points");
    CHECK(std::abs(st.sum_theta_applied - 0.5 * st.sum_theta_raw) < 1e-12, "RW4 applied theta is half the raw theta at scale 0.5");
    CHECK(st.max_theta_raw > 0.0 && st.max_theta_raw <= s0.gyr.norm() * s0.t_end + 1e-12, "RW4 max theta within the geometric bound");
  }
  {  // RW5
    DeskewOptions o; o.time_based_process_noise = "none"; o.rot_warp_scale = 0.0;
    std::vector<PointXYZCov> out; deskewPoints(s0.state, s0.poses, s0.t_end, s0.pts, o, out);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
      const M3D ref = 0.5 * (getBodyCov(s0.pts[i].p, o.sigma_r2, o.sigma_a2) + getBodyCov(s0.pts[i].p, o.sigma_r2, o.sigma_a2).transpose());
      worst = std::max(worst, (out[i].sensor_cov - ref).norm());
    }
    CHECK(worst < 1e-12, "RW5 scale 0: covariance equals the sensor model (no rotation applied to it)");
  }
  {  // RW6
    const Scene sv = makeScene(V3D(0.4, -0.2, 0.1));
    const auto a = run(sv, 0.0, nullptr), b = run(sv, 1.0, nullptr);
    double worst = 0.0, size = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
      const V3D tr0 = a[i].point - sv.pts[i].p;
      const V3D tr1 = b[i].point - rrel(sv, sv.pts[i].t) * sv.pts[i].p;
      worst = std::max(worst, (tr0 - tr1).norm());
      size = std::max(size, tr0.norm());
    }
    CHECK(size > 1e-4, "RW6 the translation part is non-zero in this scene");
    CHECK(worst < 1e-12, "RW6 the translation part of the warp is identical at scale 0 and scale 1");
  }
  {  // RW7
    const auto a = run(s0, 0.0, nullptr), b = run(s0, 1.0, nullptr);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) worst = std::max(worst, (a[i].raw_body_point - b[i].raw_body_point).norm());
    CHECK(worst == 0.0, "RW7 raw_body_point does not depend on the scale");
  }
  std::printf("ALL PASS\n");
  return 0;
}
