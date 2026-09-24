#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/diagnostics/pose_control/pose_control_physical_diagnostics.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon;

namespace {
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-58s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

PoseControlSpline stationarySpline(int N, double t0, double t1)
{
  PoseControlSpline s;
  s.init(N, t0, t1);
  s.R_anchor = M3D::Identity();
  for (int k = 0; k < N; ++k) {
    s.cp_p.col(k) = V3D::Zero();
    s.cp_phi.col(k) = V3D::Zero();
  }
  return s;
}

std::vector<ImuSample> perfectImu(const PoseControlSpline& s, const V3D& gravity, double dt)
{
  std::vector<ImuSample> out;
  for (double t = s.t0(); t <= s.t1() + 1e-12; t += dt) {
    ImuSample m;
    m.t = std::min(t, s.t1());
    m.acc = s.rotAt(m.t).transpose() * (s.accAt(m.t) - gravity);
    m.gyro = s.omegaBodyAt(m.t);
    out.push_back(m);
    if (m.t >= s.t1()) break;
  }
  return out;
}

double directCost(const PoseControlSpline& s, const std::vector<ImuSample>& imu,
                  const V3D& ba, const V3D& bg, const V3D& gravity,
                  const V3D& varA, const V3D& varW)
{
  double cost = 0.0;
  for (const auto& m : imu) {
    if (m.t < s.t0() || m.t > s.t1()) continue;
    const V3D ea = s.rotAt(m.t).transpose() * (s.accAt(m.t) - gravity) + ba - m.acc;
    const V3D ew = s.omegaBodyAt(m.t) + bg - m.gyro;
    cost += 0.5 * (ea.array().square() / varA.array()).sum();
    cost += 0.5 * (ew.array().square() / varW.array()).sum();
  }
  return cost;
}

void testQuadraticMatchesRawCost()
{
  const int N = 7;
  const double t0 = 0.0, t1 = 0.1;
  const V3D g(0.0, 0.0, -9.81);
  const V3D ba = V3D::Zero(), bg = V3D::Zero();
  const V3D varA = V3D::Constant(0.02 * 0.02);
  const V3D varW = V3D::Constant(0.005 * 0.005);

  // The IMU is generated from a TRUTH spline; the gradient/Hessian check
  // below linearizes at a DIFFERENT point s0 (truth + a small offset), so
  // the residual at the linearization point is genuinely nonzero. Checking
  // the gradient exactly AT the residual's own zero (truth==linearization
  // point) is an ill-posed finite-difference target: the true cost is then
  // an EXACT global minimum (cost==0 identically), so central-difference
  // gradient estimates there measure pure floating-point roundoff
  // amplified by 1/(2*eps), not signal -- confirmed directly (f0==0.0
  // bit-exact) when this test used a single shared point.
  PoseControlSpline s_truth = stationarySpline(N, t0, t1);
  s_truth.cp_p.col(3) = V3D(0.002, -0.001, 0.001);
  s_truth.cp_phi.col(3) = V3D(0.001, -0.002, 0.001);
  PoseControlFreeLayout layout;
  layout.N = N;
  layout.has_bg = layout.has_ba = layout.has_g = false;

  const auto imu = perfectImu(s_truth, g, 0.001);

  PoseControlSpline s0 = s_truth;
  s0.cp_p.col(2) += V3D(3e-4, -2e-4, 1e-4);
  s0.cp_phi.col(4) += V3D(-2e-4, 3e-4, -1e-4);
  const int d = layout.dim();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(d, d);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(d);
  buildPoseControlContinuousImuPrior(s0, layout, imu, ba, bg, g, varA, varW, A, b, nullptr, nullptr);

  // Independent finite-difference gradient/Hessian of the raw sample cost.
  const double eps = 1e-6;
  const double f0 = directCost(s0, imu, ba, bg, g, varA, varW);
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(d);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d, d);
  const auto base = s0;
  auto perturb = [&](const Eigen::VectorXd& dx) {
    PoseControlSpline sp = base;
    const int half = 3 * N;
    for (int i = 0; i < dx.size(); ++i) {
      if (std::abs(dx[i]) < 1e-30) continue;
      if (i < half) sp.cp_p.col(i / 3)(i % 3) += dx[i];
      else sp.cp_phi.col((i - half) / 3)((i - half) % 3) += dx[i];
    }
    return sp;
  };
  for (int i = 0; i < d; ++i) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(d); e[i] = eps;
    const double fp = directCost(perturb(e), imu, ba, bg, g, varA, varW);
    const double fm = directCost(perturb(-e), imu, ba, bg, g, varA, varW);
    grad[i] = (fp - fm) / (2.0 * eps);
    H(i, i) = (fp - 2.0 * f0 + fm) / (eps * eps);
    for (int j = i + 1; j < d; ++j) {
      Eigen::VectorXd f = e; f[j] += eps;
      Eigen::VectorXd gvec = e; gvec[j] -= eps;
      Eigen::VectorXd h = -e; h[j] += eps;
      Eigen::VectorXd k = -e; k[j] -= eps;
      const double fpp = directCost(perturb(f), imu, ba, bg, g, varA, varW);
      const double fpm = directCost(perturb(gvec), imu, ba, bg, g, varA, varW);
      const double fmp = directCost(perturb(h), imu, ba, bg, g, varA, varW);
      const double fmm = directCost(perturb(k), imu, ba, bg, g, varA, varW);
      H(i, j) = H(j, i) = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps);
    }
  }

  const double grad_err = (grad + b).norm() / std::max(1.0, grad.norm());
  const double hess_err = (H - A).norm() / std::max(1.0, H.norm());
  check(grad_err < 5e-4, "raw-cost gradient matches -b", grad_err, 5e-4);
  check(hess_err < 5e-3, "raw-cost Hessian matches A", hess_err, 5e-3);
}

void testHighFrequencyModeIsPhysicallyExpensive()
{
  const V3D g(0.0, 0.0, -9.81);
  const V3D varA = V3D::Constant(0.02 * 0.02);
  const V3D varW = V3D::Constant(0.005 * 0.005);
  const double t0 = 0.0, t1 = 0.1;

  PoseControlSpline base = stationarySpline(13, t0, t1);
  const auto imu = perfectImu(base, g, 0.001);
  const double base_cost = directCost(base, imu, V3D::Zero(), V3D::Zero(), g, varA, varW);

  PoseControlSpline alternating = base;
  const double amp = 1e-3;
  for (int k = 3; k < alternating.N(); ++k)
    alternating.cp_p.col(k).x() = ((k & 1) ? -amp : amp);

  PoseControlSpline rigid = base;
  for (int k = 0; k < rigid.N(); ++k)
    rigid.cp_p.col(k).x() = amp;

  const double hf_acc = 0.0 + alternating.accAt(0.05).norm();
  const double rigid_acc = rigid.accAt(0.05).norm();
  const double hf_cost = directCost(alternating, imu, V3D::Zero(), V3D::Zero(), g, varA, varW) - base_cost;
  const double rigid_cost = directCost(rigid, imu, V3D::Zero(), V3D::Zero(), g, varA, varW) - base_cost;

  check(hf_acc > std::max(10.0, 10.0 * rigid_acc), "alternating mode creates much larger physical acceleration", hf_acc, std::max(10.0, 10.0 * rigid_acc));
  check(hf_cost > 10.0 * std::max(rigid_cost, 1e-12), "IMU prior cost strongly penalizes high-frequency knot mode", hf_cost, 10.0 * std::max(rigid_cost, 1e-12));
}

void testPhysicalCovarianceIsFiniteAndPSD()
{
  const V3D g(0.0, 0.0, -9.81);
  const V3D varA = V3D::Constant(0.02 * 0.02);
  const V3D varW = V3D::Constant(0.005 * 0.005);
  double worst_relative_negativity = 0.0;
  for (const int N : {4, 7, 13}) {
    PoseControlSpline s = stationarySpline(N, 0.0, 0.1);
    const auto imu = perfectImu(s, g, 0.001);
    PoseControlFreeLayout layout;
    layout.N = N;
    layout.has_bg = layout.has_ba = layout.has_g = false;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
    Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
    buildPoseControlContinuousImuPrior(s, layout, imu, V3D::Zero(), V3D::Zero(), g, varA, varW, A, b, nullptr, nullptr);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
    check(es.info() == Eigen::Success, "IMU information eigensolve succeeds", static_cast<double>(N));
    const double this_min = es.eigenvalues().minCoeff();
    const double this_max = es.eigenvalues().maxCoeff();
    // PSD-ness is checked RELATIVE to this matrix's own dynamic range
    // (max|eig|), not an absolute threshold: A's own condition number
    // spans many orders of magnitude by construction (dAccDcp ~ 1/dt^2),
    // so a fixed absolute tolerance is meaningless here -- a negative
    // eigenvalue at the level of max_eig*machine_epsilon is exactly the
    // expected symmetrization/eigensolve roundoff floor, not a real PSD
    // violation.
    if (this_max > 0.0) worst_relative_negativity = std::max(worst_relative_negativity, -this_min / this_max);
    check(this_max < 1e18, "IMU information remains finite", this_max, 1e18);
  }
  check(worst_relative_negativity < 1e-9, "IMU information is PSD relative to its own dynamic range (min_eig/max_eig)", worst_relative_negativity, 1e-9);
}
}  // namespace

int main()
{
  testQuadraticMatchesRawCost();
  testHighFrequencyModeIsPhysicallyExpensive();
  testPhysicalCovarianceIsFiniteAndPSD();
  return failures ? 1 : 0;
}
