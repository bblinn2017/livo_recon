// Registered synthetic/unit test for:
//   - reduceImuResidualSamples() / computePoseControlImuResidual()'s
//     physical-residual statistics (items 12/13/19/23/27 of the
//     implementation spec)
//   - applyPoseControlAdaptiveQBiasGravityCorrection() (items 14/15/22/23:
//     "bias uncertainty alone must not inflate Q", cross-covariance)
//   - analyzeDirectionalRedundancy() (items 26-30: surface-agnostic
//     directional information analysis)
//
// Build/run convention: see test_pose_control_prior_math.cpp (registered
// CMake executable target, not a standalone g++ script, because the
// transitive include chain pulls in ROS/OpenCV headers).
#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/pose_control_adaptive_q.h"
#include "livo_recon/lio/pose_control_directional_redundancy.h"
#include "livo_recon/lio/pose_control_lidar_factor.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <random>
#include <vector>

using namespace livo_recon;

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const std::string& name, double val = 0.0)
{
  std::printf("[%s] %-70s %+.6e\n", cond ? " ok " : "FAIL", name.c_str(), val);
  if (cond) ++g_pass; else ++g_fail;
}

static void checkNear(double a, double b, double tol, const std::string& name)
{
  const double err = std::abs(a - b);
  check(err <= tol, name + " (a=" + std::to_string(a) + " b=" + std::to_string(b) + ")", err);
}

// ============================================================================
// Part A: physical IMU-domain residual reduction + bias/gravity correction.
// ============================================================================

// A1: zero-extra-noise. Pure i.i.d. Gaussian residual samples with known
// sigma and zero temporal correlation -> cov_acc/cov_gyr near sigma^2,
// |acf1/2/5| small. This is the "known sensor noise, zero extra process/
// model noise" synthetic case (item 20 of the parent spec / addendum item
// 19's zero-extra-noise scenario), at the reduceImuResidualSamples() level.
static void testZeroExtraNoiseWhiteResidual()
{
  std::mt19937 rng(42);
  const double sigma_a = 0.02, sigma_g = 0.002;
  std::normal_distribution<double> na(0.0, sigma_a), ng(0.0, sigma_g);
  const int N = 4000;
  std::vector<V3D> ra(N), rw(N);
  for (int i = 0; i < N; ++i) {
    ra[i] = V3D(na(rng), na(rng), na(rng));
    rw[i] = V3D(ng(rng), ng(rng), ng(rng));
  }
  const SplineImuResidualStats st = reduceImuResidualSamples(ra, rw);
  check(st.valid(), "zero-extra-noise: stats valid", st.n);
  checkNear(st.cov_acc, sigma_a * sigma_a, 0.15 * sigma_a * sigma_a, "zero-extra-noise: cov_acc matches sigma_a^2 within 15%");
  checkNear(st.cov_gyr, sigma_g * sigma_g, 0.15 * sigma_g * sigma_g, "zero-extra-noise: cov_gyr matches sigma_g^2 within 15%");
  check(std::abs(st.acf1_acc) < 0.1, "zero-extra-noise: |acf1_acc| < 0.1 (white)", st.acf1_acc);
  check(std::abs(st.acf1_gyr) < 0.1, "zero-extra-noise: |acf1_gyr| < 0.1 (white)", st.acf1_gyr);
  check(std::abs(st.acf2_acc) < 0.1, "zero-extra-noise: |acf2_acc| < 0.1 (white)", st.acf2_acc);
  check(std::abs(st.acf5_acc) < 0.1, "zero-extra-noise: |acf5_acc| < 0.1 (white)", st.acf5_acc);
}

// A2: known-extra-noise. Residual variance is sigma_sensor^2 + extra^2 by
// construction (independent draws summed) -> cov_acc/cov_gyr should
// recover the TOTAL (sensor + extra), which is what
// applyPoseControlAdaptiveQBiasGravityCorrection() then further corrects
// for bias/gravity contribution (tested separately in A3/A4 below).
static void testKnownExtraNoiseRecovered()
{
  std::mt19937 rng(7);
  const double sigma_a = 0.02, extra_a = 0.03;
  std::normal_distribution<double> n_sensor(0.0, sigma_a), n_extra(0.0, extra_a);
  const int N = 4000;
  std::vector<V3D> ra(N), rw(N, V3D::Zero());
  for (int i = 0; i < N; ++i)
    ra[i] = V3D(n_sensor(rng) + n_extra(rng), n_sensor(rng) + n_extra(rng), n_sensor(rng) + n_extra(rng));
  const SplineImuResidualStats st = reduceImuResidualSamples(ra, rw);
  const double expected = sigma_a * sigma_a + extra_a * extra_a;
  checkNear(st.cov_acc, expected, 0.15 * expected, "known-extra-noise: cov_acc ~= sigma_sensor^2+extra^2 within 15%");
}

// A3: bias uncertainty alone must NOT inflate Q (item 22 -- MANDATORY).
// Construct C_emp_acc EXACTLY equal to the analytic bias/gravity
// contribution (P_ba alone, P_g=0, cross=0) -- i.e. the residual's entire
// empirical spread is, by construction, attributable to bias uncertainty
// and nothing else. The corrected cov_acc must come out ~0.
static void testBiasUncertaintyDoesNotInflate()
{
  Eigen::Matrix3d P_ba = Eigen::Matrix3d::Identity() * 0.0009;  // sigma_ba=0.03
  Eigen::Matrix3d P_bg = Eigen::Matrix3d::Identity() * 0.0001;
  Eigen::Matrix3d P_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
  SplineImuResidualStats st;
  st.n = 100;
  st.cov_acc = P_ba.trace() / 3.0;   // exactly the bias contribution, isotropic
  st.cov_gyr = P_bg.trace() / 3.0;
  applyPoseControlAdaptiveQBiasGravityCorrection(st, M3D::Identity(), P_ba, P_bg, P_g, cross);
  check(st.cov_acc < 1e-12, "bias-uncertainty-alone: corrected cov_acc ~ 0 (Q must not inflate on P_bias alone)", st.cov_acc);
  check(st.cov_gyr < 1e-12, "bias-uncertainty-alone: corrected cov_gyr ~ 0", st.cov_gyr);
}

// A4: cross-covariance (item 14/23 -- MANDATORY). applyPoseControlAdaptive
// QBiasGravityCorrection() works entirely in the SCALAR/isotropic regime
// (st.cov_acc is one number, matching AdaptiveQ's own Constant(...)-
// broadcast contract -- see this file's header doc comment's disclosed
// scope) -- it can only cancel EXACTLY when the true C_pred is itself
// isotropic (proportional to I); an anisotropic C_pred's off-diagonal/
// unequal-diagonal structure is necessarily lost by the trace/3 reduction,
// which is the disclosed limitation, not a bug. So this test constructs
// P_ba=a*I, P_g=g*I, and P_ba_g=c*R^T for the SAME R passed in -- both
// individually isotropic AND the cross term (-c*R^T*R - R^T*(c*R^T)^T =
// -2c*I, since R^T*R=I for any rotation) collapses to isotropic too, for
// this specific choice, WITHOUT zeroing out the cross term itself (c!=0
// still meaningfully participates: an implementation that omitted the
// cross term, or got its sign wrong, would NOT cancel exactly here).
// C_pred_ref = (a+g-2c)*I is then bit-for-bit exact against the
// production function's own reduction, not just its trace.
static void testCrossCovarianceExact()
{
  const M3D R = Exp(V3D(0.3, -0.2, 0.5));  // arbitrary non-identity rotation
  const double a = 0.02, g = 0.01, c = 0.005;
  const Eigen::Matrix3d P_ba = Eigen::Matrix3d::Identity() * a;
  const Eigen::Matrix3d P_g = Eigen::Matrix3d::Identity() * g;
  const Eigen::Matrix3d P_ba_g = c * R.transpose();
  const double expected_extra = 0.0;  // empirical spread set to EXACTLY the predicted isotropic value below
  SplineImuResidualStats st;
  st.n = 100;
  st.cov_acc = a + g - 2.0 * c;   // == (a*I + R^T*g*I*R - c*R^T*R - R^T*(c*R^T)^T).trace()/3, isotropic by construction
  st.cov_gyr = 0.0;
  applyPoseControlAdaptiveQBiasGravityCorrection(
      st, R, P_ba, Eigen::Matrix3d::Zero(), P_g, P_ba_g);
  checkNear(st.cov_acc, expected_extra, 1e-9,
             "cross-covariance: isotropic C_pred (incl. nonzero ba/gravity cross term) cancels exactly");
}

// A5: autocorrelated residual detection (item 19/27's own "if the residual
// is strongly correlated, do not blindly interpret it as white"). Build an
// AR(1) sequence with known rho and verify acf1 recovers rho approximately
// while acf5 (rho^5, much smaller) is markedly weaker -- confirming the
// lag-1/2/5 reduction actually measures decaying correlation, not just a
// single fixed number.
static void testAutocorrelatedResidualAcf()
{
  std::mt19937 rng(99);
  std::normal_distribution<double> nd(0.0, 1.0);
  const double rho = 0.8;
  const int N = 6000;
  std::vector<V3D> ra(N), rw(N, V3D::Zero());
  V3D prev = V3D::Zero();
  for (int i = 0; i < N; ++i) {
    const V3D innov(nd(rng), nd(rng), nd(rng));
    const V3D cur = rho * prev + std::sqrt(1.0 - rho * rho) * innov;
    ra[i] = cur;
    prev = cur;
  }
  const SplineImuResidualStats st = reduceImuResidualSamples(ra, rw);
  checkNear(st.acf1_acc, rho, 0.08, "autocorrelated: acf1_acc ~= rho=0.8");
  checkNear(st.acf2_acc, rho * rho, 0.1, "autocorrelated: acf2_acc ~= rho^2=0.64");
  check(st.acf5_acc < st.acf1_acc, "autocorrelated: acf5_acc < acf1_acc (decaying correlation)", st.acf5_acc);
}

// ============================================================================
// Part B: directional redundancy (surface-agnostic, no plane_id).
// ============================================================================

// B1: many repeated (near-identical) whitened rows in ONE direction ->
// effective_rank must be 1 regardless of row COUNT, and the dominant
// eigenvalue must equal M*|h|^2 exactly (verified against a plain,
// independently-written accumulation loop, not the production code path).
static void testRepeatedDirectionRankOne()
{
  const int d = 6;
  Eigen::VectorXd h = Eigen::VectorXd::Zero(d);
  h(2) = 0.7;  // one nonzero direction
  const int M = 500;
  Eigen::MatrixXd Lambda_ref = Eigen::MatrixXd::Zero(d, d);
  for (int i = 0; i < M; ++i) Lambda_ref += h * h.transpose();  // independent reference accumulation
  const DirectionalRedundancyStats st = analyzeDirectionalRedundancy(Lambda_ref, M);
  check(st.valid, "repeated-direction: stats valid");
  check(st.effective_rank == 1, "repeated-direction: effective_rank == 1 regardless of M=500", st.effective_rank);
  checkNear(st.eigenvalues(0), M * h.squaredNorm(), 1e-6 * M * h.squaredNorm(), "repeated-direction: dominant eigenvalue == M*|h|^2");
  for (int i = 1; i < d; ++i) check(st.eigenvalues(i) < 1e-9, "repeated-direction: eigenvalue[" + std::to_string(i) + "] ~ 0", st.eigenvalues(i));
}

// B2: several genuinely independent orthogonal directions with distinct
// magnitudes -> effective_rank == number of directions, and eigenvalues
// match an INDEPENDENT reference computed via JacobiSVD of the stacked
// (unaccumulated) M x d whitened-row matrix directly (singular_value^2 ==
// eigenvalue of Lambda = J^T J) -- a genuinely different algorithm/code
// path from analyzeDirectionalRedundancy()'s own SelfAdjointEigenSolver on
// the pre-accumulated d x d matrix.
static void testIndependentDirectionsMatchSvdReference()
{
  const int d = 5;
  std::vector<Eigen::VectorXd> rows;
  const std::vector<double> weights = {3.0, 1.5, 0.8};  // 3 independent directions, distinct magnitudes
  std::mt19937 rng(11);
  std::normal_distribution<double> jitter(0.0, 0.02);  // tiny jitter so rows aren't bit-identical within a direction
  for (int dir = 0; dir < 3; ++dir) {
    for (int rep = 0; rep < 40; ++rep) {
      Eigen::VectorXd h = Eigen::VectorXd::Zero(d);
      h(dir) = weights[dir] + jitter(rng);
      rows.push_back(h);
    }
  }
  Eigen::MatrixXd J(static_cast<int>(rows.size()), d);
  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(d, d);
  for (size_t i = 0; i < rows.size(); ++i) { J.row(static_cast<int>(i)) = rows[i].transpose(); Lambda += rows[i] * rows[i].transpose(); }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd sv2 = svd.singularValues().cwiseAbs2();  // descending, by JacobiSVD's own contract
  std::sort(sv2.data(), sv2.data() + sv2.size(), std::greater<double>());

  const DirectionalRedundancyStats st = analyzeDirectionalRedundancy(Lambda, static_cast<int>(rows.size()));
  check(st.effective_rank == 3, "independent-directions: effective_rank == 3", st.effective_rank);
  for (int i = 0; i < 3; ++i)
    checkNear(st.eigenvalues(i), sv2(i), 0.02 * std::max(sv2(i), 1e-9),
              "independent-directions: eigenvalue[" + std::to_string(i) + "] matches independent SVD reference");
}

// B3: known weak direction -- one direction with information far below the
// dominant one, on purpose. Verify it is (a) excluded from effective_rank
// at the default threshold, and (b) still numerically distinguishable from
// an EXACT structural zero (i.e. this is testing the same
// "numerically-weak-but-real" distinction test_pose_control_prior_math.cpp
// already validates for generalPseudoInverse(), here for the directional
// analysis's own rank cutoff).
static void testKnownWeakDirectionExcludedFromRank()
{
  const int d = 4;
  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(d, d);
  Lambda(0, 0) = 100.0;
  Lambda(1, 1) = 50.0;
  Lambda(2, 2) = 1e-8;   // weak-but-nonzero: below rel_rank_thresh*lambda_max=100*1e-6=1e-4
  Lambda(3, 3) = 0.0;    // exact structural zero
  const DirectionalRedundancyStats st = analyzeDirectionalRedundancy(Lambda, 10);
  check(st.effective_rank == 2, "weak-direction: effective_rank == 2 (weak + zero both excluded)", st.effective_rank);
  // eigenvalues are sorted descending: 100, 50, 1e-8, 0
  checkNear(st.eigenvalues(2), 1e-8, 1e-12, "weak-direction: weak eigenvalue reported as 1e-8, not silently zeroed");
  check(st.eigenvalues(3) < 1e-12, "weak-direction: structural-zero eigenvalue reported as exactly ~0", st.eigenvalues(3));
  check(st.eigenvalues(2) > st.eigenvalues(3), "weak-direction: weak (1e-8) numerically distinguishable from exact zero", st.eigenvalues(2) - st.eigenvalues(3));
}

// ============================================================================
// Part C: threading correctness (items 30-36). addPoseControlLidarFactor()
// was refactored from a strictly serial accumulation into parallel-loop +
// thread-local-accumulator + merge. Verify the 1-thread and N-thread runs
// agree to floating-point-reduction-order tolerance on synthetic data --
// the only way to check this WITHOUT a forbidden pose-control real-data run
// (item 42/48).
static void testThreadedLidarFactorMatchesSerial()
{
  PoseControlSpline spline;
  spline.init(13, 0.0, 0.1);
  PoseControlFreeLayout layout;
  layout.N = 13; layout.has_bg = layout.has_ba = layout.has_g = true;

  std::mt19937 rng(2024);
  std::uniform_real_distribution<double> ut(0.0, 0.1), uq(-1.0, 1.0), un(-1.0, 1.0), ud(-0.5, 0.5);
  std::vector<PoseControlLidarObs> obs;
  for (int i = 0; i < 5000; ++i) {
    PoseControlLidarObs o;
    o.t = ut(rng);
    o.q = V3D(uq(rng), uq(rng), uq(rng));
    V3D n(un(rng), un(rng), un(rng));
    if (n.norm() < 1e-6) n = V3D(0, 0, 1);
    o.normal = n.normalized();
    o.d = ud(rng);
    o.sigma2 = 0.01;
    obs.push_back(o);
  }

  const int dimZ = layout.dim();
  auto run = [&](int nthreads) {
    omp_set_num_threads(nthreads);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimZ, dimZ);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(dimZ);
    double E = 0.0;
    addPoseControlLidarFactor(spline, layout, obs, A, b, nullptr, &E, nullptr);
    return std::make_tuple(A, b, E);
  };

  auto [A1, b1, E1] = run(1);
  auto [A8, b8, E8] = run(8);

  const double A_err = (A1 - A8).norm();
  const double b_err = (b1 - b8).norm();
  check(A_err < 1e-8 * std::max(1.0, A1.norm()), "threading: 1-thread vs 8-thread A matches (linear merge is exact up to fp reordering)", A_err);
  check(b_err < 1e-8 * std::max(1.0, b1.norm()), "threading: 1-thread vs 8-thread b matches", b_err);
  checkNear(E1, E8, 1e-8 * std::max(1.0, std::abs(E1)), "threading: 1-thread vs 8-thread E_lidar matches");
}

int main()
{
  testZeroExtraNoiseWhiteResidual();
  testKnownExtraNoiseRecovered();
  testBiasUncertaintyDoesNotInflate();
  testCrossCovarianceExact();
  testAutocorrelatedResidualAcf();
  testRepeatedDirectionRankOne();
  testIndependentDirectionsMatchSvdReference();
  testKnownWeakDirectionExcludedFromRank();
  testThreadedLidarFactorMatchesSerial();

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  std::printf(g_fail == 0 ? "ALL PASS\n" : "SOME FAILED\n");
  return g_fail == 0 ? 0 : 1;
}
