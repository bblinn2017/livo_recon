#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using namespace livo_recon;

namespace
{
int failures = 0;

void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-78s %.6e tol %.6e\n",
              ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

PoseControlSpline stationarySpline(int N, double t0 = 0.0, double t1 = 0.1)
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

std::vector<ImuSample> zeroImu(const PoseControlSpline& s, double dt)
{
  std::vector<ImuSample> out;
  for (double t = s.t0(); t < s.t1() - 0.25 * dt; t += dt) {
    ImuSample m;
    m.t = t;
    m.acc = V3D::Zero();
    m.gyro = V3D::Zero();
    out.push_back(m);
  }
  return out;
}

void basis(double u, Eigen::Vector4d& b, Eigen::Vector4d& db, Eigen::Vector4d& ddb)
{
  const double u2 = u * u;
  const double om = 1.0 - u;
  b << om * om * om / 6.0,
       (3.0 * u * u * u - 6.0 * u2 + 4.0) / 6.0,
       (-3.0 * u * u * u + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
       u * u * u / 6.0;
  db << -0.5 * om * om,
        0.5 * (3.0 * u2 - 4.0 * u),
        0.5 * (-3.0 * u2 + 2.0 * u + 1.0),
        0.5 * u2;
  ddb << om, 3.0 * u - 2.0, -3.0 * u + 1.0, u;
}

Eigen::MatrixXd accelDesign(const PoseControlSpline& spline,
                            const std::vector<ImuSample>& imu)
{
  const int N = spline.N();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * static_cast<int>(imu.size()), 3 * N);
  const double inv_d2 = 1.0 / (spline.delta() * spline.delta());
  for (int i = 0; i < static_cast<int>(imu.size()); ++i) {
    int seg = 0;
    double u = 0.0;
    spline.locate(imu[i].t, seg, u);
    Eigen::Vector4d b, db, ddb;
    basis(u, b, db, ddb);
    for (int k = 0; k < 4; ++k) {
      const int j = seg + k;
      if (j < 0 || j >= N) continue;
      H.block<3, 3>(3 * i, 3 * j) = ddb[k] * inv_d2 * M3D::Identity();
    }
  }
  return H;
}

Eigen::MatrixXd omegaDesignAtZeroPhi(const PoseControlSpline& spline,
                                     const std::vector<ImuSample>& imu)
{
  const int N = spline.N();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * static_cast<int>(imu.size()), 3 * N);
  const double inv_d = 1.0 / spline.delta();
  for (int i = 0; i < static_cast<int>(imu.size()); ++i) {
    int seg = 0;
    double u = 0.0;
    spline.locate(imu[i].t, seg, u);
    Eigen::Vector4d b, db, ddb;
    basis(u, b, db, ddb);
    for (int k = 0; k < 4; ++k) {
      const int j = seg + k;
      if (j < 0 || j >= N) continue;
      H.block<3, 3>(3 * i, 3 * j) = db[k] * inv_d * M3D::Identity();
    }
  }
  return H;
}

Eigen::MatrixXd positionPhysicalJacobian(const PoseControlSpline& spline,
                                          const PoseControlHeadNullspace& hns,
                                          double t,
                                          int derivative_order)
{
  const int N = spline.N();
  const int n_eta = 3 * N - 6;
  const Eigen::MatrixXd Zp = hns.Z.topRows(3 * N).leftCols(n_eta);
  int seg = 0;
  double u = 0.0;
  spline.locate(t, seg, u);
  Eigen::Vector4d b, db, ddb;
  basis(u, b, db, ddb);
  Eigen::Vector4d coeff = derivative_order == 0 ? b : (derivative_order == 1 ? db : ddb);
  double scale = 1.0;
  if (derivative_order == 1) scale = 1.0 / spline.delta();
  if (derivative_order == 2) scale = 1.0 / (spline.delta() * spline.delta());
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 3 * N);
  for (int k = 0; k < 4; ++k) {
    const int j = seg + k;
    if (j < 0 || j >= N) continue;
    J.block<3, 3>(0, 3 * j) = coeff[k] * scale * M3D::Identity();
  }
  return J * Zp;
}

Eigen::MatrixXd svdPinv(const Eigen::MatrixXd& A, double rel = 1e-12)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& s = svd.singularValues();
  const double cutoff = rel * (s.size() ? s.maxCoeff() : 0.0);
  Eigen::VectorXd inv = s;
  for (int i = 0; i < inv.size(); ++i)
    inv(i) = inv(i) > cutoff ? 1.0 / inv(i) : 0.0;
  return svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
}

Eigen::MatrixXd selectedMatrix(const Eigen::MatrixXd& A, const std::vector<int>& idx)
{
  Eigen::MatrixXd out(idx.size(), idx.size());
  for (int r = 0; r < static_cast<int>(idx.size()); ++r)
    for (int c = 0; c < static_cast<int>(idx.size()); ++c)
      out(r, c) = A(idx[r], idx[c]);
  return out;
}

void removeTranslationModes(Eigen::VectorXd& d)
{
  const int N = static_cast<int>(d.size()) / 3;
  if (N <= 0) return;
  for (int axis = 0; axis < 3; ++axis) {
    double mean = 0.0;
    for (int k = 0; k < N; ++k) mean += d(3 * k + axis);
    mean /= static_cast<double>(N);
    for (int k = 0; k < N; ++k) d(3 * k + axis) -= mean;
  }
}


Eigen::MatrixXd buildProductionA(const PoseControlSpline& spline,
                                 const std::vector<ImuSample>& imu,
                                 const PoseControlFreeLayout& layout,
                                 const V3D& var_acc,
                                 const V3D& var_gyr)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
  Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
  buildPoseControlContinuousImuPrior(
      spline, layout, imu, V3D::Zero(), V3D::Zero(), V3D::Zero(),
      var_acc, var_gyr, A, b, nullptr, nullptr);
  return 0.5 * (A + A.transpose());
}

void testIndependentPhysicalCovarianceMonteCarlo()
{
  const int N = 7;
  const double dt = 0.001;
  const double var = 0.02 * 0.02;
  const PoseControlSpline spline = stationarySpline(N);
  const auto imu = zeroImu(spline, dt);

  PoseControlFreeLayout layout;
  layout.N = N;
  layout.has_bg = layout.has_ba = layout.has_g = false;
  const PoseControlHeadNullspace hns =
      buildPoseControlHeadNullspace(spline, spline.posAt(spline.t0()), spline.velAt(spline.t0()));
  const int n_eta = 3 * N - 6;

  const Eigen::MatrixXd Hraw = accelDesign(spline, imu);
  const Eigen::MatrixXd Zp = hns.Z.topRows(3 * N).leftCols(n_eta);
  const Eigen::MatrixXd H = Hraw * Zp;
  const double W = 1.0 / var;
  const Eigen::MatrixXd Aref = W * H.transpose() * H;
  const Eigen::MatrixXd Pref = svdPinv(Aref);
  const Eigen::MatrixXd Aprod = buildProductionA(
      spline, imu, layout, V3D::Constant(var), V3D::Constant(0.005 * 0.005));
  const Eigen::MatrixXd AprodRawPos = Aprod.topLeftCorner(3 * N, 3 * N);
  const Eigen::MatrixXd AprodPos = Zp.transpose() * AprodRawPos * Zp;

  const double hess_rel = (AprodPos - Aref).norm() /
                          std::max(1.0, std::max(AprodPos.norm(), Aref.norm()));
  check(hess_rel < 1e-10,
        "independent physical reference reproduces production position information", hess_rel, 1e-10);

  const int trials = 2500;
  std::mt19937 rng(20260924u);
  std::normal_distribution<double> normal(0.0, std::sqrt(var));
  Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(n_eta, trials);
  Eigen::LDLT<Eigen::MatrixXd> ldlt(Aref);
  bool solve_ok = ldlt.info() == Eigen::Success;
  for (int trial = 0; trial < trials && solve_ok; ++trial) {
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_eta);
    for (int i = 0; i < static_cast<int>(imu.size()); ++i) {
      const Eigen::Vector3d noise(normal(rng), normal(rng), normal(rng));
      rhs += -(1.0 / var) * H.block(3 * i, 0, 3, n_eta).transpose() * noise;
    }
    samples.col(trial) = ldlt.solve(rhs);
    solve_ok = samples.col(trial).allFinite();
  }
  check(solve_ok, "independent Monte Carlo covariance solve remains finite");
  if (!solve_ok) return;

  const Eigen::VectorXd mean = samples.rowwise().mean();
  const Eigen::MatrixXd centered = samples.colwise() - mean;
  const Eigen::MatrixXd Pmc = (centered * centered.transpose()) /
                              static_cast<double>(trials - 1);
  const double mc_rel = (Pmc - Pref).norm() /
                        std::max(1.0, Pref.norm());
  check(mc_rel < 0.20,
        "empirical spline-coordinate covariance agrees with independent statistical reference",
        mc_rel, 0.20);

  double worst_physical = 0.0;
  const Eigen::MatrixXd Pprod = svdPinv(AprodPos);
  for (double u : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double t = spline.t0() + u * (spline.t1() - spline.t0());
    const Eigen::MatrixXd Jp = positionPhysicalJacobian(spline, hns, t, 0);
    const Eigen::MatrixXd Jv = positionPhysicalJacobian(spline, hns, t, 1);
    const Eigen::MatrixXd Ja = positionPhysicalJacobian(spline, hns, t, 2);
    const Eigen::MatrixXd J = (Eigen::MatrixXd(9, n_eta) << Jp, Jv, Ja).finished();
    const Eigen::MatrixXd Cprod = J * Pprod * J.transpose();
    const Eigen::MatrixXd Cmc = J * Pmc * J.transpose();
    const double rel = (Cprod - Cmc).norm() /
                       std::max(1.0, std::max(Cprod.norm(), Cmc.norm()));
    worst_physical = std::max(worst_physical, rel);
  }
  check(worst_physical < 0.20,
        "mapped physical p/v/a covariance agrees with independent Monte Carlo",
        worst_physical, 0.20);
}

void testSharedPhysicalTrajectoryAcrossN()
{
  const double t0 = 0.0, t1 = 0.1, dt = 0.001;
  const double acc_true = 0.35;
  const double vel_true = 0.2;
  const double pos_true = -0.15;
  const double q = 4e-4;
  std::vector<double> times;
  for (double t = t0; t < t1 - 0.25 * dt; t += dt) times.push_back(t);

  std::vector<double> acceleration_traces;
  std::vector<double> information;
  for (int N : {4, 7, 13}) {
    PoseControlSpline spline = stationarySpline(N, t0, t1);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(times.size(), N);
    Eigen::VectorXd target(times.size());
    for (int i = 0; i < static_cast<int>(times.size()); ++i) {
      int seg = 0; double u = 0.0; spline.locate(times[i], seg, u);
      Eigen::Vector4d b, db, ddb; basis(u, b, db, ddb);
      for (int k = 0; k < 4; ++k) {
        const int j = seg + k;
        if (j >= 0 && j < N) B(i, j) = b[k];
      }
      target(i) = pos_true + vel_true * times[i] + 0.5 * acc_true * times[i] * times[i];
    }
    const Eigen::VectorXd cx = svdPinv(B) * target;
    for (int k = 0; k < N; ++k) spline.cp_p.col(k).x() = cx(k);

    double accel_rms = 0.0;
    for (double t : times) accel_rms += spline.accAt(t).x() * spline.accAt(t).x();
    accel_rms = std::sqrt(accel_rms / times.size());
    acceleration_traces.push_back(accel_rms);

    const auto imu = zeroImu(spline, dt);
    const Eigen::MatrixXd H = accelDesign(spline, imu);
    const double var = q / dt;
    const Eigen::VectorXd c = Eigen::Map<const Eigen::VectorXd>(spline.cp_p.data(), 3 * N);
    const double d2 = c.transpose() * H.transpose() * ((1.0 / var) * H) * c;
    information.push_back(d2);
  }

  const double accel_err = *std::max_element(acceleration_traces.begin(), acceleration_traces.end()) -
                           *std::min_element(acceleration_traces.begin(), acceleration_traces.end());
  check(accel_err < 1e-7,
        "same cubic physical trajectory is represented equivalently at N=4/7/13",
        accel_err, 1e-7);

  const double info_mean = (information[0] + information[1] + information[2]) / 3.0;
  const double info_spread = (*std::max_element(information.begin(), information.end()) -
                              *std::min_element(information.begin(), information.end())) /
                             std::max(info_mean, 1e-12);
  check(info_spread < 1e-7,
        "shared representable physical mode has N-invariant IMU information",
        info_spread, 1e-7);
}

void testHighFrequencyPhysicalSigmaScaling()
{
  const double dt = 0.001;
  std::vector<double> ratios;
  for (int N : {4, 7, 13}) {
    const PoseControlSpline spline = stationarySpline(N);
    const auto imu = zeroImu(spline, dt);
    const Eigen::MatrixXd H = accelDesign(spline, imu);
    const Eigen::MatrixXd A = H.transpose() * (1.0 / (0.02 * 0.02)) * H;
    const Eigen::MatrixXd P = svdPinv(A);

    Eigen::VectorXd hi = Eigen::VectorXd::Zero(3 * N);
    for (int k = 0; k < N; ++k)
      hi(3 * k) = (k & 1) ? -1.0 : 1.0;

    // Build a smooth quadratic mode independently from the basis, so its
    // acceleration is nonzero and it is not accidentally a translation or
    // constant-velocity nullspace of the acceleration objective.
    PoseControlSpline smooth = stationarySpline(N);
    const int samples = 80;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(samples, N);
    Eigen::VectorXd target(samples);
    for (int i = 0; i < samples; ++i) {
      const double t = spline.t0() + (spline.t1() - spline.t0()) * i / (samples - 1.0);
      int seg = 0; double u = 0.0; smooth.locate(t, seg, u);
      Eigen::Vector4d b, db, ddb; basis(u, b, db, ddb);
      for (int k = 0; k < 4; ++k) {
        const int j = seg + k;
        if (j >= 0 && j < N) B(i, j) = b[k];
      }
      const double tau = (t - spline.t0()) / (spline.t1() - spline.t0());
      target(i) = 0.5 * tau * tau;
    }
    const Eigen::VectorXd smooth_cp = svdPinv(B) * target;
    Eigen::VectorXd lo = Eigen::VectorXd::Zero(3 * N);
    for (int k = 0; k < N; ++k) lo(3 * k) = smooth_cp(k);

    removeTranslationModes(hi);
    removeTranslationModes(lo);
    hi.array() /= std::max(hi.norm(), 1e-12);
    lo.array() /= std::max(lo.norm(), 1e-12);
    const double hi_info = (hi.transpose() * A * hi)(0);
    const double lo_info = (lo.transpose() * A * lo)(0);
    const double ratio = hi_info / std::max(lo_info, 1e-18);
    ratios.push_back(ratio);

    const double sigma_mode = std::sqrt(std::max(0.0, (hi.transpose() * P * hi)(0)));
    const double z_for_1mm = 1e-3 / std::max(sigma_mode, 1e-18);
    check(std::isfinite(ratio) && ratio > 0.0 && std::isfinite(z_for_1mm),
          "high-frequency mode has finite physical information and sigma scale", ratio, 0.0);
  }

  check(ratios[2] > ratios[1] && ratios[1] >= ratios[0],
        "additional spline capacity increases high-frequency information relative to smooth mode",
        ratios[2], ratios[0]);
}

void testFullNoiseDensityRefinement()
{
  const PoseControlSpline spline = stationarySpline(13);
  const double q = 4e-4;
  const std::vector<double> dts = {0.004, 0.002, 0.001, 0.0005};
  Eigen::VectorXd mode = Eigen::VectorXd::Zero(3 * spline.N());
  for (int k = 3; k < spline.N(); ++k) mode(3 * k) = (k & 1) ? -1.0 : 1.0;
  removeTranslationModes(mode);
  mode.normalize();

  std::vector<double> info;
  for (double dt : dts) {
    const auto imu = zeroImu(spline, dt);
    const Eigen::MatrixXd H = accelDesign(spline, imu);
    const double var = q / dt;
    info.push_back((mode.transpose() * H.transpose() * ((1.0 / var) * H) * mode)(0));
  }
  const double ref = info.back();
  double max_rel = 0.0;
  for (double v : info) max_rel = std::max(max_rel, std::abs(v / ref - 1.0));
  check(max_rel < 0.08,
        "complete continuous-density IMU information converges over 8x timestep refinement",
        max_rel, 0.08);

  Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(6 * spline.N(), 6 * spline.N());
  Eigen::VectorXd b1 = Eigen::VectorXd::Zero(6 * spline.N());
  const V3D var1 = V3D::Constant(q / dts[2]);
  const V3D var2 = V3D::Constant(2.0 * q / dts[2]);
  auto imu = zeroImu(spline, dts[2]);
  PoseControlFreeLayout layout;
  layout.N = spline.N(); layout.has_bg = layout.has_ba = layout.has_g = false;
  buildPoseControlContinuousImuPrior(spline, layout, imu, V3D::Zero(), V3D::Zero(), V3D::Zero(), var1,
                                     V3D::Constant(0.005 * 0.005), A1, b1, nullptr, nullptr);
  const Eigen::MatrixXd A2 = buildProductionA(spline, imu, layout, var2, V3D::Constant(0.005 * 0.005));
  const PoseControlHeadNullspace hns =
      buildPoseControlHeadNullspace(spline, spline.posAt(spline.t0()), spline.velAt(spline.t0()));
  const Eigen::MatrixXd Zp = hns.Z.topRows(3 * spline.N()).leftCols(3 * spline.N() - 6);
  const Eigen::MatrixXd A1p = Zp.transpose() * A1.topLeftCorner(3 * spline.N(), 3 * spline.N()) * Zp;
  const Eigen::MatrixXd A2p = Zp.transpose() * A2.topLeftCorner(3 * spline.N(), 3 * spline.N()) * Zp;
  const Eigen::MatrixXd P1 = svdPinv(A1p);
  const Eigen::MatrixXd P2 = svdPinv(A2p);
  const double scale_ratio = P2.trace() / std::max(P1.trace(), 1e-18);
  check(std::isfinite(scale_ratio) && std::abs(scale_ratio - 2.0) < 1e-5,
        "doubling acceleration noise density doubles supported prior covariance",
        scale_ratio, 1e-5);
}

void testBiasUncertaintyForAcceleration()
{
  const int N = 7;
  const double dt = 0.001;
  const double var = 0.02 * 0.02;
  const PoseControlSpline spline = stationarySpline(N);
  const auto imu = zeroImu(spline, dt);
  PoseControlFreeLayout layout;
  layout.N = N; layout.has_bg = false; layout.has_ba = true; layout.has_g = false;
  const Eigen::MatrixXd Aprod = buildProductionA(spline, imu, layout,
                                                  V3D::Constant(var), V3D::Constant(0.005 * 0.005));
  const int ncp = 3 * N;
  const int iba = 6 * N;
  const std::vector<int> idx = [&] {
    std::vector<int> v; v.reserve(ncp + 3);
    for (int i = 0; i < ncp; ++i) v.push_back(i);
    for (int i = 0; i < 3; ++i) v.push_back(iba + i);
    return v;
  }();
  const Eigen::MatrixXd Apc = selectedMatrix(Aprod, idx);

  const Eigen::MatrixXd Hp = accelDesign(spline, imu);
  Eigen::MatrixXd Hjoint = Eigen::MatrixXd::Zero(Hp.rows(), ncp + 3);
  Hjoint.leftCols(ncp) = Hp;
  for (int i = 0; i < static_cast<int>(imu.size()); ++i)
    Hjoint.block<3, 3>(3 * i, ncp) = M3D::Identity();
  const Eigen::MatrixXd Aref = Hjoint.transpose() * (1.0 / var) * Hjoint;
  const double rel = (Apc - Aref).norm() / std::max(1.0, Aref.norm());
  check(rel < 1e-10, "acceleration bias joint information matches independent reference", rel, 1e-10);

  std::vector<double> bias_vars = {1e-6, 1e-4, 1e-2};
  std::vector<double> traces;
  std::vector<double> bias_norms;
  const Eigen::VectorXd measurement = Eigen::VectorXd::Constant(Hjoint.rows(), 0.08);
  for (double bias_var : bias_vars) {
    Eigen::MatrixXd A = Apc;
    A.bottomRightCorner(3, 3).diagonal().array() += 1.0 / bias_var;
    const Eigen::MatrixXd P = svdPinv(A);
    traces.push_back(P.topLeftCorner(ncp, ncp).trace());

    const Eigen::VectorXd info_rhs = Hjoint.transpose() * (1.0 / var) * measurement;
    const Eigen::VectorXd solution = svdPinv(A) * info_rhs;
    bias_norms.push_back(solution.tail(3).norm());
  }
  check(traces[0] <= traces[1] && traces[1] <= traces[2],
        "less certain acceleration bias cannot reduce trajectory uncertainty",
        traces[2], traces[0]);
  check(bias_norms[2] >= bias_norms[0],
        "less certain acceleration bias lets the joint MAP absorb more constant bias in the bias state",
        bias_norms[2], bias_norms[0]);
}

void testBiasUncertaintyForGyro()
{
  const int N = 7;
  const double dt = 0.001;
  const double var = 0.005 * 0.005;
  const PoseControlSpline spline = stationarySpline(N);
  const auto imu = zeroImu(spline, dt);
  PoseControlFreeLayout layout;
  layout.N = N; layout.has_bg = true; layout.has_ba = false; layout.has_g = false;
  const Eigen::MatrixXd Aprod = buildProductionA(spline, imu, layout,
                                                  V3D::Constant(0.02 * 0.02), V3D::Constant(var));
  const int ncp = 3 * N;
  const int ibg = 6 * N;
  const std::vector<int> idx = [&] {
    std::vector<int> v; v.reserve(ncp + 3);
    for (int i = 0; i < ncp; ++i) v.push_back(3 * N + i);
    for (int i = 0; i < 3; ++i) v.push_back(ibg + i);
    return v;
  }();
  const Eigen::MatrixXd Apc = selectedMatrix(Aprod, idx);

  const Eigen::MatrixXd Hphi = omegaDesignAtZeroPhi(spline, imu);
  Eigen::MatrixXd Hjoint = Eigen::MatrixXd::Zero(Hphi.rows(), ncp + 3);
  Hjoint.leftCols(ncp) = Hphi;
  for (int i = 0; i < static_cast<int>(imu.size()); ++i)
    Hjoint.block<3, 3>(3 * i, ncp) = M3D::Identity();
  const Eigen::MatrixXd Aref = Hjoint.transpose() * (1.0 / var) * Hjoint;
  const double rel = (Apc - Aref).norm() / std::max(1.0, Aref.norm());
  check(rel < 1e-10, "gyro bias joint information matches independent reference", rel, 1e-10);

  std::vector<double> bias_vars = {1e-8, 1e-6, 1e-4};
  std::vector<double> traces;
  std::vector<double> bias_norms;
  const Eigen::VectorXd measurement = Eigen::VectorXd::Constant(Hjoint.rows(), 0.02);
  for (double bias_var : bias_vars) {
    Eigen::MatrixXd A = Apc;
    A.bottomRightCorner(3, 3).diagonal().array() += 1.0 / bias_var;
    const Eigen::MatrixXd P = svdPinv(A);
    traces.push_back(P.topLeftCorner(ncp, ncp).trace());

    const Eigen::VectorXd info_rhs = Hjoint.transpose() * (1.0 / var) * measurement;
    const Eigen::VectorXd solution = svdPinv(A) * info_rhs;
    bias_norms.push_back(solution.tail(3).norm());
  }
  check(traces[0] <= traces[1] && traces[1] <= traces[2],
        "less certain gyro bias cannot reduce trajectory uncertainty",
        traces[2], traces[0]);
  check(bias_norms[2] >= bias_norms[0],
        "less certain gyro bias lets the joint MAP absorb more constant bias in the bias state",
        bias_norms[2], bias_norms[0]);
}

void testSyntheticImuLidarEndToEnd()
{
  const PoseControlSpline spline = stationarySpline(7);
  const auto imu = zeroImu(spline, 0.001);
  PoseControlFreeLayout layout;
  layout.N = spline.N(); layout.has_bg = layout.has_ba = layout.has_g = false;
  const Eigen::MatrixXd A = buildProductionA(spline, imu, layout,
                                              V3D::Constant(0.02 * 0.02), V3D::Constant(0.005 * 0.005));
  const int n_eta = 3 * spline.N() - 6;
  const PoseControlHeadNullspace hns =
      buildPoseControlHeadNullspace(spline, spline.posAt(spline.t0()), spline.velAt(spline.t0()));
  const Eigen::MatrixXd Zp = hns.Z.topRows(3 * spline.N()).leftCols(n_eta);
  const Eigen::MatrixXd Aeta = Zp.transpose() * A.topLeftCorner(3 * spline.N(), 3 * spline.N()) * Zp;
  Eigen::LDLT<Eigen::MatrixXd> prior_ldlt(0.5 * (Aeta + Aeta.transpose()));
  check(prior_ldlt.info() == Eigen::Success,
        "synthetic IMU prior reduced position covariance is SPD");
  if (prior_ldlt.info() != Eigen::Success) return;
  const Eigen::MatrixXd P =
      prior_ldlt.solve(Eigen::MatrixXd::Identity(n_eta, n_eta));
  const double tm = 0.05;
  const Eigen::MatrixXd Jp = positionPhysicalJacobian(spline, hns, tm, 0);
  Eigen::RowVectorXd H = Jp.row(0);

  const double R = 0.01 * 0.01;
  Eigen::MatrixXd Lambda = H.transpose() * (1.0 / R) * H;
  Eigen::MatrixXd Ppost;
  CovarianceUpdateDiagnostics diag;
  const bool ok = covarianceInformationUpdate(P, Lambda, Ppost, diag);
  check(ok, "synthetic IMU+LiDAR covariance update succeeds");
  if (!ok) return;

  Eigen::VectorXd truth = Eigen::VectorXd::Zero(n_eta);
  truth(0) = 0.01;
  const double y = (H * truth)(0);
  const Eigen::VectorXd delta_prod = Ppost * H.transpose() * (y / R);
  const Eigen::MatrixXd A_dense = svdPinv(P) + Lambda;
  const Eigen::VectorXd delta_ref = svdPinv(A_dense) * H.transpose() * (y / R);
  const double mean_err = (delta_prod - delta_ref).norm() / std::max(1.0, delta_ref.norm());
  check(mean_err < 1e-9,
        "synthetic observable LiDAR direction produces the dense-MAP mean update", mean_err, 1e-9);

  const double prior_obs = (H * P * H.transpose())(0, 0);
  const double post_obs = (H * Ppost * H.transpose())(0, 0);
  check(post_obs < prior_obs,
        "LiDAR information decreases uncertainty in its observable direction",
        post_obs / std::max(prior_obs, 1e-18), 1.0);

  // A direction v is truly untouched by a rank-1 covariance update when
  // H*P*v = 0, not merely when H*v = 0. Use the independent nullspace of
  // the covariance-weighted measurement row for this check.
  const Eigen::RowVectorXd HP = H * P;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(HP, Eigen::ComputeFullV);
  Eigen::VectorXd null_dir = svd.matrixV().col(1);
  const double null_prior = (null_dir.transpose() * P * null_dir)(0);
  const double null_post = (null_dir.transpose() * Ppost * null_dir)(0);
  check(std::abs(null_post - null_prior) /
            std::max(1.0, std::abs(null_prior)) < 1e-9,
        "LiDAR leaves an exactly unobserved synthetic direction unchanged",
        std::abs(null_post - null_prior), 1e-9);
}

}  // namespace

int main()
{
  std::printf("Pose-control final pre-real-data validation suite\n");
  testIndependentPhysicalCovarianceMonteCarlo();
  testSharedPhysicalTrajectoryAcrossN();
  testHighFrequencyPhysicalSigmaScaling();
  testFullNoiseDensityRefinement();
  testBiasUncertaintyForAcceleration();
  testBiasUncertaintyForGyro();
  testSyntheticImuLidarEndToEnd();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
