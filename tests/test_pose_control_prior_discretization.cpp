#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_spline.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon;

namespace
{
int failures = 0;

void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-64s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

PoseControlSpline makeSpline()
{
  PoseControlSpline s;
  s.init(7, 0.0, 0.1);
  s.R_anchor = M3D::Identity();
  for (int k = 0; k < s.N(); ++k) {
    s.cp_p.col(k) = V3D(0.01 * k + 1e-4 * k * k,
                        -0.002 * k + 5e-5 * k * k,
                        0.001 * k - 3e-5 * k * k);
    s.cp_phi.col(k) = V3D(0.001 * k, -0.0005 * k, 0.0002 * k);
  }
  return s;
}

std::vector<ImuSample> makeImu(const PoseControlSpline& s, double dt)
{
  const V3D g(0, 0, -9.81);
  std::vector<ImuSample> samples;
  for (double t = s.t0(); t < s.t1() - 0.25 * dt; t += dt) {
    ImuSample m;
    m.t = t;
    m.acc = s.rotAt(t).transpose() * (s.accAt(t) - g);
    m.gyro = s.omegaBodyAt(t);
    samples.push_back(m);
  }
  return samples;
}

double directionalInformation(const PoseControlSpline& spline,
                               const PoseControlFreeLayout& layout,
                               const std::vector<ImuSample>& imu,
                               const V3D& var_acc,
                               const V3D& var_gyr,
                               const Eigen::VectorXd& d)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
  Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
  PoseControlPriorHeadBlock hb;
  buildPoseControlContinuousImuPrior(
      spline, layout, imu, V3D::Zero(), V3D::Zero(), V3D(0, 0, -9.81),
      var_acc, var_gyr, A, b, &hb, nullptr);
  return d.dot(A * d);
}

Eigen::VectorXd makeDirection(const PoseControlFreeLayout& layout)
{
  Eigen::VectorXd d = Eigen::VectorXd::Zero(layout.dim());
  for (int k = 0; k < layout.N; ++k) {
    const double sign = (k % 2 == 0) ? 1.0 : -1.0;
    d.segment<3>(layout.colPos(k)) = sign * V3D(1.0, -0.5, 0.25);
  }
  const double n = d.norm();
  return d / n;
}

void testPerSampleVarianceScaling()
{
  const PoseControlSpline spline = makeSpline();
  PoseControlFreeLayout layout;
  layout.N = spline.N();
  layout.has_bg = layout.has_ba = layout.has_g = false;
  const Eigen::VectorXd d = makeDirection(layout);
  const V3D var_acc = V3D::Constant(0.02 * 0.02);
  const V3D var_gyr = V3D::Constant(0.005 * 0.005);

  const double dt1 = 0.002;
  const double dt2 = 0.001;
  const double I1 = directionalInformation(spline, layout, makeImu(spline, dt1),
                                            var_acc, var_gyr, d);
  const double I2 = directionalInformation(spline, layout, makeImu(spline, dt2),
                                            var_acc, var_gyr, d);
  const double count_ratio = static_cast<double>(makeImu(spline, dt2).size()) /
                             static_cast<double>(makeImu(spline, dt1).size());
  const double measured_ratio = I2 / I1;
  check(std::isfinite(measured_ratio) && measured_ratio > 0.0,
        "per-sample information ratio is finite", measured_ratio, 0.0);
  check(std::abs(measured_ratio / count_ratio - 1.0) < 0.10,
        "per-sample variance: information scales with sample count",
        measured_ratio / count_ratio, 0.10);
}

void testContinuousNoiseDensityDiscretization()
{
  const PoseControlSpline spline = makeSpline();
  PoseControlFreeLayout layout;
  layout.N = spline.N();
  layout.has_bg = layout.has_ba = layout.has_g = false;
  const Eigen::VectorXd d = makeDirection(layout);

  const double dt1 = 0.002;
  const double dt2 = 0.001;
  const V3D q_acc = V3D::Constant(0.02 * 0.02);
  const V3D q_gyr = V3D::Constant(0.005 * 0.005);

  // Interpret q_acc and q_gyr as continuous-time noise densities.
  // For the Riemann-sum approximation of the continuous-time objective,
  // the equivalent per-sample covariance is q / dt.
  const V3D var_acc_1 = q_acc / dt1;
  const V3D var_gyr_1 = q_gyr / dt1;
  const V3D var_acc_2 = q_acc / dt2;
  const V3D var_gyr_2 = q_gyr / dt2;

  const double I1 = directionalInformation(spline, layout, makeImu(spline, dt1),
                                            var_acc_1, var_gyr_1, d);
  const double I2 = directionalInformation(spline, layout, makeImu(spline, dt2),
                                            var_acc_2, var_gyr_2, d);
  const double ratio = I2 / I1;
  check(std::isfinite(ratio) && ratio > 0.0,
        "continuous-time density refinement ratio is finite", ratio, 0.0);
  check(std::abs(ratio - 1.0) < 0.10,
        "continuous-time density: information converges under timestep refinement",
        ratio, 0.10);
}
}

int main()
{
  std::printf("Pose-control IMU-prior discretization tests\n");
  testPerSampleVarianceScaling();
  testContinuousNoiseDensityDiscretization();
  return failures == 0 ? 0 : 1;
}
