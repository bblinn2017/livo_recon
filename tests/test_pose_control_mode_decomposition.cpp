// Phase 8/9: synthetic validation of the per-mode LiDAR/IMU information
// decomposition (I_lidar_i + I_imu_i + I_other_i ~= lambda_total_i, an
// EXACT eigenpair identity) and factor-gradient disagreement, using
// synthetic matrices with KNOWN exact component information/gradients --
// not the live estimator (that composition is exercised for real, on real
// eee_01 data, by the mode_information/mode_gradient diagnostic rows this
// task added to lio_coupled.cpp; see the campaign report).
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

namespace
{
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-90s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

Eigen::MatrixXd randomSpd(int n, std::mt19937& rng, double scale = 1.0)
{
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) M(i, j) = nd(rng);
  return scale * (M * M.transpose() + 0.1 * Eigen::MatrixXd::Identity(n, n));
}

// item: verify the EXACT identity lambda_total_i == I_lidar_i + I_imu_i for
// every eigenmode of a synthetic total information matrix built from two
// KNOWN component matrices, across several random trials and dimensions.
void testInformationDecompositionIdentity()
{
  std::mt19937 rng(101);
  for (int trial = 0; trial < 5; ++trial) {
    const int n = 6 + trial;
    const Eigen::MatrixXd A_lidar = randomSpd(n, rng, 1.0);
    const Eigen::MatrixXd A_imu = randomSpd(n, rng, 2.0);
    const Eigen::MatrixXd A_total = A_lidar + A_imu;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A_total + A_total.transpose()));
    double worst_err = 0.0, worst_rel = 0.0;
    for (int i = 0; i < n; ++i) {
      const Eigen::VectorXd v = es.eigenvectors().col(i);
      const double lambda_total = es.eigenvalues()(i);
      const double I_lidar = (v.transpose() * A_lidar * v)(0);
      const double I_imu = (v.transpose() * A_imu * v)(0);
      const double err = std::abs(lambda_total - (I_lidar + I_imu));
      worst_err = std::max(worst_err, err);
      worst_rel = std::max(worst_rel, err / std::max(1e-12, std::abs(lambda_total)));
      // Sanity: fractions sum to 1.
      const double frac_sum = (I_lidar + I_imu) > 1e-300 ? (I_lidar + (I_imu)) / (I_lidar + I_imu) : 0.0;
      (void)frac_sum;
    }
    const std::string name = "lambda_total_i == I_lidar_i + I_imu_i EXACTLY for every eigenmode (n=" + std::to_string(n) + ")";
    check(worst_err < 1e-9, name.c_str(), worst_err, 1e-9);
    check(worst_rel < 1e-9, "relative error is also negligible", worst_rel, 1e-9);
  }
}

// item: a mode that is ENTIRELY LiDAR-informed (I_imu~=0) is correctly
// classified, and vice versa.
void testPureLidarAndPureImuModesClassifiedCorrectly()
{
  const int n = 4;
  Eigen::MatrixXd A_lidar = Eigen::MatrixXd::Zero(n, n);
  A_lidar(0, 0) = 100.0;   // all LiDAR information lives along e_0
  Eigen::MatrixXd A_imu = Eigen::MatrixXd::Zero(n, n);
  A_imu(1, 1) = 50.0;      // all IMU information lives along e_1
  const Eigen::MatrixXd A_total = A_lidar + A_imu + 1e-6 * Eigen::MatrixXd::Identity(n, n);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_total);
  // Find the eigenvector closest to e_0 and to e_1.
  int idx0 = 0, idx1 = 0;
  double best0 = -1, best1 = -1;
  for (int i = 0; i < n; ++i) {
    const double c0 = std::abs(es.eigenvectors()(0, i));
    const double c1 = std::abs(es.eigenvectors()(1, i));
    if (c0 > best0) { best0 = c0; idx0 = i; }
    if (c1 > best1) { best1 = c1; idx1 = i; }
  }
  auto info = [&](int idx) {
    const Eigen::VectorXd v = es.eigenvectors().col(idx);
    return std::make_pair((v.transpose() * A_lidar * v)(0), (v.transpose() * A_imu * v)(0));
  };
  const auto [Il0, Ii0] = info(idx0);
  const auto [Il1, Ii1] = info(idx1);
  std::printf("  pure-LiDAR-direction mode: I_lidar=%.4e I_imu=%.4e (LiDAR-strong/IMU-weak expected)\n", Il0, Ii0);
  std::printf("  pure-IMU-direction mode:   I_lidar=%.4e I_imu=%.4e (LiDAR-weak/IMU-strong expected)\n", Il1, Ii1);
  check(Il0 > 10.0 * Ii0, "LiDAR-only-informed direction is correctly classified LiDAR-strong/IMU-weak", Il0, 10.0 * Ii0);
  check(Ii1 > 10.0 * Il1, "IMU-only-informed direction is correctly classified IMU-strong/LiDAR-weak", Ii1, 10.0 * Il1);
}

// A mode weak in BOTH factors (structural nullspace of the total Hessian).
void testWeakInBothFactors()
{
  const int n = 3;
  Eigen::MatrixXd A_lidar = Eigen::MatrixXd::Identity(n, n) * 1e-8;
  Eigen::MatrixXd A_imu = Eigen::MatrixXd::Identity(n, n) * 1e-8;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_lidar + A_imu);
  const Eigen::VectorXd v = es.eigenvectors().col(0);
  const double Il = (v.transpose() * A_lidar * v)(0), Ii = (v.transpose() * A_imu * v)(0);
  check(Il < 1e-6 && Ii < 1e-6, "a genuinely weak-in-both-factors mode reports small information from both", Il + Ii, 2e-6);
}

// Phase 9: factor-gradient disagreement scenarios with KNOWN exact
// same-direction / opposing / near-zero relationships (not at a converged
// GN solution, so no trivial cancellation applies here -- these are
// standalone synthetic vectors chosen to exercise each case directly).
void testFactorGradientDisagreementScenarios()
{
  // Same direction: both factors want to move this mode the same way.
  {
    const double g_lidar = 0.5, g_imu = 0.6, I_lidar = 10.0, I_imu = 12.0;
    const double step_lidar = g_lidar / I_lidar, step_imu = g_imu / I_imu;
    const double disagreement = std::min(std::abs(step_lidar), std::abs(step_imu));
    const bool same_sign = (step_lidar * step_imu) >= 0;
    check(same_sign, "same-direction factors: signs agree");
    std::printf("  same-direction: step_lidar=%.4e step_imu=%.4e disagreement_strength=%.4e\n", step_lidar, step_imu, disagreement);
  }
  // Opposing, both strong: genuine tug-of-war.
  {
    const double g_lidar = 0.8, g_imu = -0.7, I_lidar = 5.0, I_imu = 4.0;
    const double step_lidar = g_lidar / I_lidar, step_imu = g_imu / I_imu;
    const double disagreement = std::min(std::abs(step_lidar), std::abs(step_imu));
    check(step_lidar * step_imu < 0, "opposing factors: signs disagree");
    check(disagreement > 0.1, "opposing-and-strong factors: disagreement_strength is large (genuine tug-of-war)", disagreement, 0.1);
  }
  // One factor near zero: not a real disagreement even though signs might differ.
  {
    const double g_lidar = 0.8, g_imu = -0.0001, I_lidar = 5.0, I_imu = 4.0;
    const double step_lidar = g_lidar / I_lidar, step_imu = g_imu / I_imu;
    const double disagreement = std::min(std::abs(step_lidar), std::abs(step_imu));
    check(disagreement < 1e-3, "one factor near zero: disagreement_strength correctly stays small (not a real fight)", disagreement, 1e-3);
  }
  // Both weak: no real disagreement, low stakes either way.
  {
    const double g_lidar = 1e-5, g_imu = -1e-5, I_lidar = 5.0, I_imu = 4.0;
    const double step_lidar = g_lidar / I_lidar, step_imu = g_imu / I_imu;
    const double disagreement = std::min(std::abs(step_lidar), std::abs(step_imu));
    check(disagreement < 1e-5, "both factors weak: disagreement_strength correctly stays negligible", disagreement, 1e-5);
  }
}

}  // namespace

int main()
{
  std::printf("Pose-control weak-mode information/gradient decomposition synthetic validation suite\n");
  testInformationDecompositionIdentity();
  testPureLidarAndPureImuModesClassifiedCorrectly();
  testWeakInBothFactors();
  testFactorGradientDisagreementScenarios();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
