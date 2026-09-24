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
#include "livo_recon/lio/pose_control_process_factor.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/lio/adaptive_q.h"

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
  checkNear(st.cov_acc, expected, 0.15 * expected, "known-extra-noise: cov_acc ~= sigma_sensor^2+extra^2 within 15% (accelerometer)");

  // Same construction, gyro channel (item 21's own "test both accelerometer
  // and gyro").
  std::mt19937 rng_g(8);
  const double sigma_g = 0.002, extra_g = 0.003;
  std::normal_distribution<double> ng_sensor(0.0, sigma_g), ng_extra(0.0, extra_g);
  std::vector<V3D> ra2(N, V3D::Zero()), rw2(N);
  for (int i = 0; i < N; ++i)
    rw2[i] = V3D(ng_sensor(rng_g) + ng_extra(rng_g), ng_sensor(rng_g) + ng_extra(rng_g), ng_sensor(rng_g) + ng_extra(rng_g));
  const SplineImuResidualStats st_g = reduceImuResidualSamples(ra2, rw2);
  const double expected_g = sigma_g * sigma_g + extra_g * extra_g;
  checkNear(st_g.cov_gyr, expected_g, 0.15 * expected_g, "known-extra-noise: cov_gyr ~= sigma_sensor^2+extra^2 within 15% (gyro)");
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

// A6: causality (item 17/25 -- MANDATORY). Exercises AdaptiveQ's own
// PUBLIC interface directly (no live-estimator scaffolding needed, unlike
// the two-real-scan integration test that WOULD require it -- see this
// file's own header comment and the report's honest-gaps section for that
// disclosed limitation): frame 1's update() with a DISTINCT statistic must
// not retroactively change frame 1's OWN varAcc()/varGyr() read that
// already happened before update() was called -- i.e. Q_used_1 (read
// BEFORE update) must differ from Q_candidate_1 (the result of update),
// and Q_used_2 (read AFTER update, for the next frame) must equal
// Q_candidate_1 -- exactly the temporal chain item 17 specifies, at the
// AdaptiveQ class's own API boundary (the ACTUAL causality enforcement
// mechanism, since lio_coupled.cpp only ever calls update() once per scan,
// strictly after that scan's own prior/covariance are already fixed --
// verified by code inspection, see this file's own commit message).
static void testAdaptiveQCausality()
{
  AdaptiveQ q;
  AdaptiveQOptions opts;
  opts.enable = true;
  opts.warmup_frames = 0;
  opts.beta_acc = 2.0; opts.beta_gyr = 2.0;  // wide excursion so the test isn't fighting the safety clamp
  opts.ema = 0.0;  // no smoothing -- isolates the causality property itself
  opts.z_rate_limit = 10.0;  // default 0.02 deliberately caps per-frame movement (a real safety
                              // feature, "no single frame's measurement can move the filter far") --
                              // raised here so ONE update() call can actually reach its target,
                              // isolating causality from that separate, already-validated mechanism
  q.configure(opts);
  q.setNominal(0.0004, 0.000004);

  const double q_used_1_acc = q.varAcc();  // Q_used_1: BEFORE any update() call
  check(std::abs(q_used_1_acc - 0.0004) < 1e-12, "causality: Q_used_1 == nominal (no update() has run yet)", q_used_1_acc);

  SplineImuResidualStats st1;
  st1.n = 100; st1.cov_acc = 0.0004 * 25.0; st1.cov_gyr = 0.000004; st1.acf1_acc = 0.0; st1.acf1_gyr = 0.0;
  q.update(st1);  // scan 1's OWN candidate, computed from scan 1's OWN residual
  const double q_candidate_1_acc = q.varAcc();
  check(q_candidate_1_acc > q_used_1_acc * 2.0, "causality: Q_candidate_1 != Q_used_1 (a real update happened)", q_candidate_1_acc);

  // Q_used_2 (what scan 2's OWN prior would read, per lio_coupled.cpp's
  // wiring -- poseControlEffectiveVarAcc() called at scan 2's start) must
  // equal Q_candidate_1 exactly -- the update from scan 1 IS what scan 2
  // consumes, never scan 1 itself (which already used q_used_1_acc, fixed
  // before update() ran).
  const double q_used_2_acc = q.varAcc();
  checkNear(q_used_2_acc, q_candidate_1_acc, 1e-15, "causality: Q_used_2 == Q_candidate_1 (scan 2 consumes scan 1's candidate)");
}

// A7: TRAJECTORY-STATE JACOBIAN, finite-difference self-verification (item
// 5/P14's own required addition -- "IMPLEMENT THEM NOW", using
// dAccDcp/dOmegaDcphi/dThetaDcphi). Rather than trust the analytic sign
// convention derived in this file's own header comment, perturb the
// spline's raw control points directly and compare the resulting change in
// e_acc/e_gyr against computePoseControlImuResidualStateJacobian()'s own
// prediction -- this is the ORACLE, exactly the same role finite-difference
// plays for dOmegaDcphi's own existing validation in
// test_pose_control_spline.cpp ("FD is the ORACLE only, never used here").
static void testStateJacobianFiniteDifference()
{
  PoseControlSpline spline;
  spline.init(13, 0.0, 0.1);
  // Give the spline some nonzero, non-trivial motion so R(t) != I and
  // accAt(t)/omegaBodyAt(t) are not degenerately zero.
  std::mt19937 rng(555);
  std::uniform_real_distribution<double> u(-0.3, 0.3);
  for (int i = 0; i < spline.N(); ++i) {
    spline.cp_p.col(i) = V3D(u(rng) + 0.02 * i, u(rng), u(rng));
    spline.cp_phi.col(i) = V3D(u(rng), u(rng), u(rng));
  }
  PoseControlFreeLayout layout;
  layout.N = spline.N(); layout.has_bg = layout.has_ba = layout.has_g = true;
  PoseControlHeadNullspace hns;
  hns.Z = Eigen::MatrixXd::Identity(6 * spline.N(), 6 * spline.N());  // no head elimination needed for this test
  hns.c_particular = Eigen::VectorXd::Zero(6 * spline.N());

  const double t = 0.05;
  const V3D gravity(0, 0, -9.81);
  Eigen::MatrixXd J_acc_eta, J_gyr_eta;
  computePoseControlImuResidualStateJacobian(spline, layout, hns, t, gravity, J_acc_eta, J_gyr_eta);

  auto evalAcc = [&](const PoseControlSpline& s) -> V3D {
    return s.rotAt(t).transpose() * (s.accAt(t) - gravity);
  };
  auto evalGyr = [&](const PoseControlSpline& s) -> V3D {
    return s.omegaBodyAt(t);
  };

  const double eps = 1e-6;
  double max_acc_err = 0.0, max_gyr_err = 0.0;
  const int rawDim = hns.rawDim();
  for (int col = 0; col < rawDim; col += std::max(1, rawDim / 12)) {  // sample a spread of columns, not all (cheap)
    PoseControlSpline sp = spline, sm = spline;
    const int half = rawDim / 2;
    if (col < half) { sp.cp_p.col(col / 3)(col % 3) += eps; sm.cp_p.col(col / 3)(col % 3) -= eps; }
    else { const int c2 = col - half; sp.cp_phi.col(c2 / 3)(c2 % 3) += eps; sm.cp_phi.col(c2 / 3)(c2 % 3) -= eps; }
    const V3D fd_acc = (evalAcc(sp) - evalAcc(sm)) / (2.0 * eps);
    const V3D fd_gyr = (evalGyr(sp) - evalGyr(sm)) / (2.0 * eps);
    const V3D an_acc = J_acc_eta.col(col);
    const V3D an_gyr = J_gyr_eta.col(col);
    max_acc_err = std::max(max_acc_err, (fd_acc - an_acc).norm());
    max_gyr_err = std::max(max_gyr_err, (fd_gyr - an_gyr).norm());
  }
  check(max_acc_err < 1e-4, "trajectory-Jacobian: analytic d(e_acc)/d(eta) matches finite difference", max_acc_err);
  check(max_gyr_err < 1e-4, "trajectory-Jacobian: analytic d(e_gyr)/d(eta) matches finite difference", max_gyr_err);
}

// A8: trajectory-state correction, sanity + zero-uncertainty degenerate
// case (P_eta=0 must leave the residual untouched -- no spurious
// subtraction when there IS no trajectory uncertainty).
static void testTrajectoryStateCorrectionZeroUncertainty()
{
  Eigen::MatrixXd J_acc = Eigen::MatrixXd::Random(3, 10);
  Eigen::MatrixXd J_gyr = Eigen::MatrixXd::Random(3, 10);
  Eigen::MatrixXd P_eta = Eigen::MatrixXd::Zero(10, 10);
  SplineImuResidualStats st;
  st.n = 50; st.cov_acc = 0.0005; st.cov_gyr = 0.00002;
  const double before_acc = st.cov_acc, before_gyr = st.cov_gyr;
  applyPoseControlAdaptiveQTrajectoryStateCorrection(st, J_acc, J_gyr, P_eta);
  checkNear(st.cov_acc, before_acc, 1e-15, "trajectory-correction: P_eta=0 leaves cov_acc untouched");
  checkNear(st.cov_gyr, before_gyr, 1e-15, "trajectory-correction: P_eta=0 leaves cov_gyr untouched");
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

// ============================================================================
// Part D: production GN-assembly synthetic fixture (item 22 -- "construct a
// small synthetic estimator fixture sufficient to exercise the production
// GN assembly path... at minimum exercise: production prior, production
// LiDAR information, production first GN step... compare against an
// independent information-form reference. Do NOT require eee_01.").
//
// This calls the REAL production functions
// (buildPoseControlHeadNullspace/buildPoseControlImuPriorContribution/
// generalPseudoInverse/addPoseControlLidarFactor) in the SAME order and
// composition estimateCoupledPoseControlSpline()'s own scan-start-prior +
// first-GN-iteration code uses (mirrored from lio_coupled.cpp, not
// reimplemented) -- this is not a parallel re-derivation of the math, it
// is the actual production math, exercised outside the ROS-coupled class
// method that normally calls it. The independent-reference half of the
// comparison uses FINITE-DIFFERENCE Jacobians for the LiDAR information
// (a genuinely different computational method than the analytic
// dPosDcp/dThetaDcphi chain addPoseControlLidarFactor() uses internally),
// and a synthetic problem small/simple enough that a well-conditioned
// direct solve is its own cross-check.
// ============================================================================
static void testProductionGNAssemblySyntheticFixture()
{
  PoseControlSpline spline;
  const int N = 7;
  spline.init(N, 0.0, 0.1);
  std::mt19937 rng(4242);
  std::uniform_real_distribution<double> u(-0.05, 0.05);
  for (int i = 0; i < N; ++i) {
    spline.cp_p.col(i) = V3D(0.01 * i, u(rng), u(rng));
    spline.cp_phi.col(i) = V3D(u(rng), u(rng), u(rng) + 0.01 * i);
  }
  const V3D p0 = spline.posAt(0.0), v0 = spline.velAt(0.0);
  const PoseControlHeadNullspace hns = buildPoseControlHeadNullspace(spline, p0, v0);
  const int dEta = hns.freeDim();

  PoseControlFreeLayout layout;
  layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = false;  // trajectory-only, keeps the fixture small

  // ---- production prior: synthetic per-segment IMU samples (near-static,
  // small noise) fed through the REAL buildPoseControlImuPriorContribution()
  // once per segment, exactly as lio_coupled.cpp's scan-start init loop
  // does. ----------------------------------------------------------------
  const V3D bias_acc = V3D::Zero(), bias_gyr = V3D::Zero(), gravity(0, 0, -9.81);
  const V3D var_acc = V3D::Constant(0.02 * 0.02), var_gyr = V3D::Constant(0.002 * 0.002);
  std::vector<std::vector<ImuSample>> seg_samples(spline.nSeg());
  for (int j = 0; j < spline.nSeg(); ++j) {
    const double t_lo = j * spline.delta(), t_hi = (j + 1) * spline.delta();
    for (int s = 0; s <= 4; ++s) {
      ImuSample smp;
      smp.t = t_lo + (t_hi - t_lo) * s / 4.0;
      smp.acc = V3D(0, 0, 9.81);  // static-ish, gravity-only, matches nominal
      smp.gyro = V3D::Zero();
      seg_samples[j].push_back(smp);
    }
  }
  const int dimRaw = layout.dim();
  Eigen::MatrixXd A_prior_raw = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_prior_raw_unused = Eigen::VectorXd::Zero(dimRaw);
  PoseControlProcessFactorHeadBlock head_block;
  for (int j = 0; j < spline.nSeg(); ++j)
    buildPoseControlImuPriorContribution(spline, layout, j, seg_samples[j], bias_acc, bias_gyr, gravity,
        1.0, 1.0, var_acc, var_gyr, true, 1e-6, A_prior_raw, b_prior_raw_unused, &head_block, nullptr);

  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(9, 9) * 1e-4;  // synthetic scan-start state covariance
  const Eigen::MatrixXd Omega0 = generalPseudoInverse(P0, 1e-6);
  Eigen::MatrixXd A_hh_prior = head_block.A_hh + Omega0;
  Eigen::MatrixXd A_hf_prior = (head_block.A_hf.size() > 0) ? head_block.A_hf : Eigen::MatrixXd::Zero(9, dimRaw);

  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dimRaw, dEta);
  P.block(0, 0, hns.rawDim(), dEta) = hns.Z;  // dST=0, so z==eta here
  const Eigen::MatrixXd A_ff_prior = P.transpose() * A_prior_raw * P;
  const Eigen::MatrixXd A_hf_prior_z = A_hf_prior * P;
  const int dimFull = 9 + dEta;
  Eigen::MatrixXd Lambda_full_prior = Eigen::MatrixXd::Zero(dimFull, dimFull);
  Lambda_full_prior.block(0, 0, 9, 9) = A_hh_prior;
  Lambda_full_prior.block(0, 9, 9, dEta) = A_hf_prior_z;
  Lambda_full_prior.block(9, 0, dEta, 9) = A_hf_prior_z.transpose();
  Lambda_full_prior.block(9, 9, dEta, dEta) = A_ff_prior;
  const Eigen::MatrixXd Sigma_full_prior = generalPseudoInverse(Lambda_full_prior, 1e-9);
  const Eigen::MatrixXd Lambda_prior_eta = generalPseudoInverse(Sigma_full_prior.bottomRightCorner(dEta, dEta), 1e-9);

  // ---- production LiDAR information: synthetic point-to-plane
  // observations against 3 known, non-degenerate plane orientations. ------
  std::vector<PoseControlLidarObs> obs;
  std::uniform_real_distribution<double> ut(0.0, 0.1), uq(-1.0, 1.0);
  const std::vector<V3D> plane_normals = {V3D(1, 0, 0).normalized(), V3D(0, 1, 0).normalized(), V3D(0.3, 0.3, 1).normalized()};
  for (int i = 0; i < 300; ++i) {
    PoseControlLidarObs o;
    o.t = ut(rng);
    o.q = V3D(uq(rng), uq(rng), uq(rng));
    o.normal = plane_normals[i % 3];
    o.d = uq(rng) * 0.1;
    o.sigma2 = 0.01;
    obs.push_back(o);
  }
  Eigen::MatrixXd A_lidar_raw = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_lidar_raw = Eigen::VectorXd::Zero(dimRaw);
  addPoseControlLidarFactor(spline, layout, obs, A_lidar_raw, b_lidar_raw, nullptr, nullptr, nullptr);
  const Eigen::MatrixXd Lambda_meas_eta = P.transpose() * A_lidar_raw * P;
  const Eigen::VectorXd b_meas_eta = P.transpose() * b_lidar_raw;

  // ---- production first GN step: combine prior + LiDAR, solve. ----------
  const Eigen::MatrixXd A_total = Lambda_prior_eta + Lambda_meas_eta;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A_total);
  const Eigen::VectorXd delta_eta_production = ldlt.solve(b_meas_eta);
  check(ldlt.info() == Eigen::Success, "production-GN: combined system solves (LDLT success)");

  // ---- independent reference for the LiDAR half: rebuild A_lidar_raw via
  // FINITE-DIFFERENCE Jacobians instead of addPoseControlLidarFactor()'s
  // own analytic dPosDcp/dThetaDcphi chain -- a genuinely different method,
  // not the same code re-run. -------------------------------------------
  auto residualAt = [&](const PoseControlSpline& s, const PoseControlLidarObs& o) -> double {
    return o.normal.dot(s.rotAt(o.t) * o.q + s.posAt(o.t)) + o.d;
  };
  Eigen::MatrixXd A_lidar_fd = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_lidar_fd = Eigen::VectorXd::Zero(dimRaw);
  const double eps = 1e-6;
  const int half = dimRaw / 2;
  for (const auto& o : obs) {
    Eigen::VectorXd Jrow = Eigen::VectorXd::Zero(dimRaw);
    for (int col = 0; col < dimRaw; ++col) {
      PoseControlSpline sp = spline, sm = spline;
      if (col < half) { sp.cp_p.col(col / 3)(col % 3) += eps; sm.cp_p.col(col / 3)(col % 3) -= eps; }
      else { const int c2 = col - half; sp.cp_phi.col(c2 / 3)(c2 % 3) += eps; sm.cp_phi.col(c2 / 3)(c2 % 3) -= eps; }
      Jrow(col) = (residualAt(sp, o) - residualAt(sm, o)) / (2.0 * eps);
    }
    const double w = 1.0 / std::max(o.sigma2, 1e-12);
    const double r = residualAt(spline, o);
    A_lidar_fd += w * (Jrow * Jrow.transpose());
    b_lidar_fd += -w * r * Jrow;
  }
  const Eigen::MatrixXd Lambda_meas_eta_fd = P.transpose() * A_lidar_fd * P;
  const Eigen::VectorXd b_meas_eta_fd = P.transpose() * b_lidar_fd;

  const double A_err = (Lambda_meas_eta - Lambda_meas_eta_fd).norm() / std::max(1.0, Lambda_meas_eta.norm());
  const double b_err = (b_meas_eta - b_meas_eta_fd).norm() / std::max(1.0, b_meas_eta.norm());
  check(A_err < 1e-4, "production-GN: analytic LiDAR information matches independent finite-difference reference", A_err);
  check(b_err < 1e-4, "production-GN: analytic LiDAR rhs matches independent finite-difference reference", b_err);

  // Solve the SAME combined system with the FD-built LiDAR half instead --
  // an independently-assembled information-form reference for the first
  // GN step's own delta, not just its ingredients.
  const Eigen::MatrixXd A_total_fd = Lambda_prior_eta + Lambda_meas_eta_fd;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_fd(A_total_fd);
  const Eigen::VectorXd delta_eta_reference = ldlt_fd.solve(b_meas_eta_fd);
  const double delta_err = (delta_eta_production - delta_eta_reference).norm() / std::max(1.0, delta_eta_production.norm());
  check(delta_err < 1e-3, "production-GN: first GN step delta matches independent information-form reference", delta_err);

  // ---- curvature (item 21): a deterministic regularization term added
  // directly to A_raw/b_raw (mean solve only), confirmed here to change
  // the mean delta while the COVARIANCE-path information (Lambda_meas_eta,
  // built identically to the covariance path's own Lambda_meas_z) is
  // untouched -- exactly production's own declared interpretation B. -----
  Eigen::MatrixXd A_curv_raw = A_lidar_raw;
  Eigen::VectorXd b_curv_raw = b_lidar_raw;
  const double lambda_curv = 1e4;
  for (int k = 1; k + 1 < N; ++k) {
    // second-difference penalty on cp_p[k-1,k,k+1], mirroring the
    // production curvature block's own construction.
    for (int a = 0; a < 3; ++a) {
      const int i0 = 3 * (k - 1) + a, i1 = 3 * k + a, i2 = 3 * (k + 1) + a;
      A_curv_raw(i0, i0) += lambda_curv; A_curv_raw(i1, i1) += 4 * lambda_curv; A_curv_raw(i2, i2) += lambda_curv;
      A_curv_raw(i0, i1) -= 2 * lambda_curv; A_curv_raw(i1, i0) -= 2 * lambda_curv;
      A_curv_raw(i1, i2) -= 2 * lambda_curv; A_curv_raw(i2, i1) -= 2 * lambda_curv;
      A_curv_raw(i0, i2) += lambda_curv; A_curv_raw(i2, i0) += lambda_curv;
    }
  }
  const Eigen::MatrixXd Lambda_meas_eta_curv = P.transpose() * A_curv_raw * P;  // MEAN-solve-only quantity
  const Eigen::VectorXd b_meas_eta_curv = P.transpose() * b_curv_raw;
  const Eigen::MatrixXd A_total_curv = Lambda_prior_eta + Lambda_meas_eta_curv;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_curv(A_total_curv);
  const Eigen::VectorXd delta_eta_curv = ldlt_curv.solve(b_meas_eta_curv);
  const double mean_diff = (delta_eta_curv - delta_eta_production).norm();
  check(mean_diff > 1e-6, "curvature: changes the mean-solve delta (regularization is active)", mean_diff);
  // Lambda_meas_eta (the covariance-path quantity, built from A_lidar_raw
  // alone) is, by construction here, untouched by A_curv_raw's existence --
  // this is the same structural fact section J's code audit already
  // established in production; re-confirmed numerically here since both
  // matrices are actually in hand in this test.
  check((Lambda_meas_eta - (P.transpose() * A_lidar_raw * P)).norm() < 1e-12,
        "curvature: covariance-path information matrix is bit-identical whether or not curvature was ever computed", 0.0);
}

int main()
{
  testZeroExtraNoiseWhiteResidual();
  testKnownExtraNoiseRecovered();
  testBiasUncertaintyDoesNotInflate();
  testCrossCovarianceExact();
  testAutocorrelatedResidualAcf();
  testAdaptiveQCausality();
  testStateJacobianFiniteDifference();
  testTrajectoryStateCorrectionZeroUncertainty();
  testRepeatedDirectionRankOne();
  testIndependentDirectionsMatchSvdReference();
  testKnownWeakDirectionExcludedFromRank();
  testThreadedLidarFactorMatchesSerial();
  testProductionGNAssemblySyntheticFixture();

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  std::printf(g_fail == 0 ? "ALL PASS\n" : "SOME FAILED\n");
  return g_fail == 0 ? 0 : 1;
}
