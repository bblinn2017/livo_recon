// R72 unit test (F-403): hard cap on the iekf update step (include/livo_recon/lio/step_cap.h). Isolated algebraic identities only.
//   SC1 step inside both caps: scale == 1 exactly, bound == false, s * solution is bit-identical to solution.
//   SC2 rotation binding: ||s * solution_rot|| equals the rotation cap (1e-12 relative).
//   SC3 direction preserved: s * solution is parallel to solution (cosine == 1 to 1e-14) over random vectors.
//   SC4 both over: the smaller scale wins, neither applied norm exceeds its cap, and exactly one equals its cap.
//   SC5 a step with norm exactly equal to the cap is not bound (the cap is not strict).
//   SC6 the check fires: a per-component clamp (the tempting alternative) changes the direction of an anisotropic step by a
//       wide margin, so SC3 would fail for that wrong implementation.
//   SC7 mode=off reports on() == false (the integration guard) and mode=hard reports true.
#include "livo_recon/lio/step_cap.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  const int IR = 0, IP = 3, DIM = 18;
  StepCapOptions o;
  o.mode = "hard";
  o.rotation_rad = 1e-2;
  o.position_m = 2e-2;

  {   // SC1
    Eigen::VectorXd x = Eigen::VectorXd::Zero(DIM);
    x.segment<3>(IR) << 3e-3, -4e-3, 1e-3;
    x.segment<3>(IP) << 1e-2, 5e-3, -5e-3;
    x(10) = 7.0;   // outside the capped blocks, must not matter
    const StepCapResult r = stepCapScale(o, x, IR, IP);
    const Eigen::VectorXd y = r.scale * x;
    CHECK(r.scale == 1.0 && !r.bound, "SC1 inside both caps: scale == 1, not bound");
    CHECK((y - x).cwiseAbs().maxCoeff() == 0.0, "SC1 inside both caps: step bit-identical");
  }
  {   // SC2
    Eigen::VectorXd x = Eigen::VectorXd::Zero(DIM);
    x.segment<3>(IR) << 0.03, -0.04, 0.0;    // norm 0.05, cap 0.01
    x.segment<3>(IP) << 1e-3, 0.0, 0.0;
    const StepCapResult r = stepCapScale(o, x, IR, IP);
    const Eigen::VectorXd y = r.scale * x;
    CHECK(r.bound, "SC2 rotation over cap: bound");
    CHECK(std::abs(y.segment<3>(IR).norm() - o.rotation_rad) <= 1e-12 * o.rotation_rad, "SC2 applied rotation norm equals the cap");
  }
  {   // SC3, SC4
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0, 1);
    double worst_cos = 1.0;
    int n_both = 0;
    for (int t = 0; t < 2000; ++t) {
      Eigen::VectorXd x = Eigen::VectorXd::Zero(DIM);
      for (int k = 0; k < 3; ++k) { x(IR + k) = 0.02 * N(rng); x(IP + k) = 0.04 * N(rng); }
      const StepCapResult r = stepCapScale(o, x, IR, IP);
      if (r.scale <= 0.0) continue;
      const Eigen::VectorXd y = r.scale * x;
      const double c = x.dot(y) / (x.norm() * y.norm());
      worst_cos = std::min(worst_cos, c);
      const double ar = y.segment<3>(IR).norm(), ap = y.segment<3>(IP).norm();
      if (ar > o.rotation_rad * (1 + 1e-12) || ap > o.position_m * (1 + 1e-12)) { CHECK(false, "SC4 applied norm exceeds a cap"); }
      if (r.rot_norm > o.rotation_rad && r.pos_norm > o.position_m) {
        ++n_both;
        const bool rot_at = std::abs(ar - o.rotation_rad) <= 1e-12 * o.rotation_rad;
        const bool pos_at = std::abs(ap - o.position_m) <= 1e-12 * o.position_m;
        if (rot_at == pos_at && !(rot_at && pos_at)) { CHECK(false, "SC4 exactly one cap must be tight"); }
        if (rot_at && pos_at) { /* both tight only when the ratios coincide */ }
        if (!rot_at && !pos_at) { CHECK(false, "SC4 one cap must be tight"); }
      }
    }
    CHECK(worst_cos >= 1.0 - 1e-14, "SC3 direction preserved over 2000 random steps");
    CHECK(n_both > 50, "SC4 the both-over case was exercised (more than 50 draws)");
  }
  {   // SC5
    Eigen::VectorXd x = Eigen::VectorXd::Zero(DIM);
    x(IR) = o.rotation_rad;
    const StepCapResult r = stepCapScale(o, x, IR, IP);
    CHECK(!r.bound && r.scale == 1.0, "SC5 norm exactly at the cap is not bound");
  }
  {   // SC6
    Eigen::VectorXd x = Eigen::VectorXd::Zero(DIM);
    x.segment<3>(IR) << 0.10, 0.01, 0.0;   // anisotropic, norm ~0.1005
    const StepCapResult r = stepCapScale(o, x, IR, IP);
    const Eigen::VectorXd good = r.scale * x;
    Eigen::VectorXd naive = x;
    for (int k = 0; k < 3; ++k) naive(IR + k) = std::max(-o.rotation_rad, std::min(o.rotation_rad, x(IR + k)));
    const double c_good = x.dot(good) / (x.norm() * good.norm());
    const double c_naive = x.dot(naive) / (x.norm() * naive.norm());
    CHECK(c_good >= 1.0 - 1e-14, "SC6 scalar cap keeps direction");
    CHECK(c_naive < 1.0 - 1e-3, "SC6 per-component clamp would rotate the step (the check fires)");
  }
  {   // SC7
    StepCapOptions off;
    CHECK(!off.on(), "SC7 default mode is off");
    CHECK(o.on(), "SC7 mode=hard is on");
  }
  std::printf("ALL PASS\n");
  return 0;
}
