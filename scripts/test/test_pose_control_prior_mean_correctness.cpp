// Prior-mean-correctness phase, items 9-14: the required quadratic/
// gradient/Hessian-equivalence, IMU-only-optimum, and incremental-
// information tests for the FIX to lio_coupled.cpp's scan-start prior mean
// (previously z_imu_ was silently set to "whatever eta currently is",
// discarding the IMU factor's own gradient/information-vector -- see that
// file's own comment at the fix site). This test mirrors PRODUCTION's own
// z-prior formula EXACTLY:
//     A_ff = Ps^T * A_imu_raw * Ps            (head-fixed-conditional Hessian, z-only)
//     xi_z = Ps^T * b_imu_raw                 (head-fixed-conditional gradient, z-only)
//     z_prior = z0 + pinv(A_ff, thresh) * xi_z
// (production additionally marginalizes P0/head uncertainty into the
// COVARIANCE, a separate, unchanged concern -- these tests isolate the
// MEAN fix itself, i.e. the head-fixed IMU-only conditional problem, which
// is exactly what buildPoseControlContinuousImuPrior()'s own A/b already
// represent with x0 held at its linearization point).
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>

using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  std::printf("  [%s] %-64s %+.6e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) ++g_fail;
}

namespace {

struct Fixture {
  int N;
  PoseControlSpline spline;
  PoseControlFreeLayout layout;
  Eigen::MatrixXd Ps;       // dimRaw x dimZ, projects raw c/sT -> z=[eta;sT]
  Eigen::MatrixXd hnsZ;     // hns.Z alone (6N x dEta) -- for the RAW c-vector update in applyZ, which is
                            // only 6N-sized (poseControlFlatten), NOT dimRaw=6N+dST-sized like Ps's own rows.
  int dEta, dST, dimZ, dimRaw;
  V3D bias_acc, bias_gyr, gravity;
  V3D var_acc, var_gyr;
};

Fixture makeFixture(int N, unsigned seed) {
  Fixture f;
  f.N = N;
  const double t0 = 0.0, t1 = 0.1;
  f.spline.init(N, t0, t1);
  f.spline.R_anchor = M3D::Identity();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> up(-0.05, 0.05);
  for (int i = 0; i < N; ++i) {
    f.spline.cp_p.col(i) = V3D(0.02 * i, up(rng), up(rng));
    f.spline.cp_phi.col(i) = V3D(up(rng), up(rng), up(rng));
  }
  f.layout.N = N; f.layout.has_bg = f.layout.has_ba = f.layout.has_g = true; f.layout.fix_head = false;
  // A REAL head nullspace (not Identity) -- matters specifically for
  // testImuOnlyOptimum: without a genuine head anchor, raw IMU information
  // alone cannot observe absolute position or yaw (classic INS
  // unobservability), making "the IMU-only optimum" ill-posed/non-unique.
  // Production always applies the real head nullspace before the prior's
  // mean/covariance are derived, so this fixture matches that exactly.
  const V3D p0 = f.spline.posAt(t0), v0 = f.spline.velAt(t0);
  const PoseControlHeadNullspace hns = buildPoseControlHeadNullspace(f.spline, p0, v0);
  f.dEta = hns.freeDim();
  f.dST = f.layout.dimST();
  f.dimZ = f.dEta + f.dST;
  f.dimRaw = f.layout.dim();
  f.Ps = Eigen::MatrixXd::Zero(f.dimRaw, f.dimZ);
  f.Ps.block(0, 0, hns.rawDim(), f.dEta) = hns.Z;
  f.Ps.block(hns.rawDim(), f.dEta, f.dST, f.dST) = Eigen::MatrixXd::Identity(f.dST, f.dST);
  f.hnsZ = hns.Z;
  f.bias_acc = V3D(0.01, -0.02, 0.005);
  f.bias_gyr = V3D(0.001, 0.002, -0.0015);
  f.gravity = V3D(0, 0, -9.81);
  f.var_acc = V3D::Constant(0.02 * 0.02);
  f.var_gyr = V3D::Constant(0.002 * 0.002);
  return f;
}

// Unflatten z=[eta;bg;ba;g] into (spline, bias_acc, bias_gyr, gravity). z is
// a LOCAL PERTURBATION from f.spline/f.bias_*/f.gravity's own (baseline,
// z=0) values. f.Ps.leftCols(dEta) is hns.Z, so c_current + Z*eta_delta
// stays exactly on the head-constraint manifold for any eta_delta -- the
// SAME poseControlFlatten/poseControlUnflatten round trip production uses.
void applyZ(const Fixture& f, const Eigen::VectorXd& z, PoseControlSpline& spline_out,
            V3D& ba_out, V3D& bg_out, V3D& g_out) {
  const Eigen::VectorXd c_current = poseControlFlatten(f.spline);
  const Eigen::VectorXd c = c_current + f.hnsZ * z.head(f.dEta);
  spline_out = f.spline;
  poseControlUnflatten(c, spline_out);
  ba_out = f.bias_acc; bg_out = f.bias_gyr; g_out = f.gravity;
  if (f.layout.colBG() >= 0) bg_out += z.segment<3>(f.dEta + f.layout.colBG() - f.layout.dimCFree());
  if (f.layout.colBA() >= 0) ba_out += z.segment<3>(f.dEta + f.layout.colBA() - f.layout.dimCFree());
  if (f.layout.colG()  >= 0) g_out  += z.segment<3>(f.dEta + f.layout.colG()  - f.layout.dimCFree());
}

// The TRUE nonlinear IMU-only cost at z (item 9's C_raw, x0-prior term
// dropped since head is never perturbed here -- it is identically zero for
// every z in this test, a true constant, so it cannot affect any
// difference-is-constant check).
double rawCost(const Fixture& f, const Eigen::VectorXd& z, const std::vector<ImuSample>& imu) {
  PoseControlSpline s; V3D ba, bg, g;
  applyZ(f, z, s, ba, bg, g);
  double cost = 0.0;
  for (const auto& samp : imu) {
    if (samp.t < s.t0() || samp.t > s.t1()) continue;
    const M3D R = s.rotAt(samp.t);
    const V3D e_acc = R.transpose() * (s.accAt(samp.t) - g) + ba - samp.acc;
    const V3D e_gyr = s.omegaBodyAt(samp.t) + bg - samp.gyro;
    cost += 0.5 * (e_acc.x()*e_acc.x()/f.var_acc.x() + e_acc.y()*e_acc.y()/f.var_acc.y() + e_acc.z()*e_acc.z()/f.var_acc.z());
    cost += 0.5 * (e_gyr.x()*e_gyr.x()/f.var_gyr.x() + e_gyr.y()*e_gyr.y()/f.var_gyr.y() + e_gyr.z()*e_gyr.z()/f.var_gyr.z());
  }
  return cost;
}

// Builds A_ff (z-space Hessian) and xi_z (z-space gradient) via the REAL
// production function, at linearization point z0 (spline/bias/gravity
// already set to z0's values in `f`).
void buildAffXiZ(const Fixture& f, const std::vector<ImuSample>& imu,
                  Eigen::MatrixXd& A_ff, Eigen::VectorXd& xi_z) {
  Eigen::MatrixXd A_raw = Eigen::MatrixXd::Zero(f.dimRaw, f.dimRaw);
  Eigen::VectorXd b_raw = Eigen::VectorXd::Zero(f.dimRaw);
  buildPoseControlContinuousImuPrior(f.spline, f.layout, imu, f.bias_acc, f.bias_gyr, f.gravity,
      f.var_acc, f.var_gyr, A_raw, b_raw, nullptr, nullptr);
  A_ff = f.Ps.transpose() * A_raw * f.Ps;
  xi_z = f.Ps.transpose() * b_raw;
}

std::vector<ImuSample> synthesizeImu(const PoseControlSpline& s, const V3D& ba, const V3D& bg, const V3D& g,
                                       double rate_hz = 200.0, double noise_acc = 0.0, double noise_gyr = 0.0,
                                       unsigned seed = 1) {
  std::vector<ImuSample> imu;
  std::mt19937 rng(seed);
  std::normal_distribution<double> na(0.0, noise_acc), ng(0.0, noise_gyr);
  for (double t = s.t0(); t <= s.t1() + 1e-9; t += 1.0 / rate_hz) {
    ImuSample smp; smp.t = t;
    smp.acc = s.rotAt(t).transpose() * (s.accAt(t) - g) + ba + V3D(na(rng), na(rng), na(rng));
    smp.gyro = s.omegaBodyAt(t) + bg + V3D(ng(rng), ng(rng), ng(rng));
    imu.push_back(smp);
  }
  return imu;
}

}  // namespace

// ============================================================================
// items 9-12: quadratic/gradient/Hessian equivalence.
// ============================================================================
static void testQuadraticGradientHessianEquivalence() {
  Fixture f = makeFixture(7, 555);
  const auto imu = synthesizeImu(f.spline, f.bias_acc, f.bias_gyr, f.gravity, 200.0, 0.01, 0.001, 2024);

  Eigen::MatrixXd A_ff; Eigen::VectorXd xi_z;
  buildAffXiZ(f, imu, A_ff, xi_z);
  const Eigen::MatrixXd A_ff_pinv = generalPseudoInverse(A_ff, 1e-6);
  const Eigen::VectorXd delta_z_prior = A_ff_pinv * xi_z;
  const Eigen::VectorXd z0 = Eigen::VectorXd::Zero(f.dimZ);  // linearization point (current trial == 0 in this test's own z coordinate)
  const Eigen::VectorXd z_prior = z0 + delta_z_prior;

  // ---- item 9: quadratic equivalence -------------------------------------
  std::mt19937 rng(999);
  // Small perturbations -- this test validates the LOCAL quadratic
  // approximation (item 9's own cost formula is explicitly the LINEARIZED
  // residual's cost, r0 + J*(z-z0), not the true nonlinear cost at large
  // offsets): rotation enters e_acc/e_gyr nonlinearly (R(t)^T, Jr(phi)), so
  // a large-angle perturbation exercises curvature the linear model was
  // never meant to capture -- exactly analogous to how every other FD
  // Jacobian test in this codebase (e.g. testStateJacobianFiniteDifference)
  // uses eps ~ 1e-6, not an O(1) step.
  std::normal_distribution<double> pert(0.0, 1e-5);
  std::vector<double> diffs;
  const double c_raw_at_z0 = rawCost(f, z0, imu);
  const double c_gauss_at_z0 = 0.5 * (z0 - z_prior).dot(A_ff * (z0 - z_prior));
  const double C_offset = c_raw_at_z0 - c_gauss_at_z0;
  for (int i = 0; i < 200; ++i) {
    Eigen::VectorXd z = z0;
    for (int j = 0; j < f.dimZ; ++j) z(j) += pert(rng);
    const double c_raw = rawCost(f, z, imu);
    const double c_gauss = 0.5 * (z - z_prior).dot(A_ff * (z - z_prior)) + C_offset;
    diffs.push_back(c_raw - c_gauss);
  }
  double max_abs = 0.0, sum_sq = 0.0;
  for (double d : diffs) { max_abs = std::max(max_abs, std::abs(d)); sum_sq += d * d; }
  const double rms = std::sqrt(sum_sq / diffs.size());
  std::printf("  quadratic-equivalence: max_absolute_cost_difference_after_offset=%.6e  RMS=%.6e\n", max_abs, rms);
  check(max_abs < 1.0, "max_absolute_cost_difference_after_offset < 1.0 (item 9, MUST pass)", max_abs);
  check(rms < 0.5, "RMS_cost_difference_after_offset < 0.5", rms);

  // ---- item 11: gradient equivalence --------------------------------------
  double max_grad_err = 0.0;
  const double eps = 1e-5;
  for (int trial = 0; trial < 10; ++trial) {
    Eigen::VectorXd z = z0;
    for (int j = 0; j < f.dimZ; ++j) z(j) += pert(rng);
    Eigen::VectorXd grad_raw = Eigen::VectorXd::Zero(f.dimZ);
    for (int j = 0; j < f.dimZ; ++j) {
      Eigen::VectorXd zp = z, zm = z; zp(j) += eps; zm(j) -= eps;
      grad_raw(j) = (rawCost(f, zp, imu) - rawCost(f, zm, imu)) / (2 * eps);
    }
    const Eigen::VectorXd grad_gauss = A_ff * (z - z_prior);
    max_grad_err = std::max(max_grad_err, (grad_raw - grad_gauss).norm());
  }
  std::printf("  max_gradient_error=%.6e\n", max_grad_err);
  check(max_grad_err < 50.0, "gradient_raw ~= Lambda_prior*(z-z_prior) (item 11)", max_grad_err);

  // ---- item 12: Hessian equivalence ---------------------------------------
  Eigen::MatrixXd hess_num = Eigen::MatrixXd::Zero(f.dimZ, f.dimZ);
  for (int j = 0; j < f.dimZ; ++j) {
    for (int k = j; k < f.dimZ; ++k) {
      Eigen::VectorXd zpp = z0, zpm = z0, zmp = z0, zmm = z0;
      zpp(j) += eps; zpp(k) += eps;
      zpm(j) += eps; zpm(k) -= eps;
      zmp(j) -= eps; zmp(k) += eps;
      zmm(j) -= eps; zmm(k) -= eps;
      const double h = (rawCost(f, zpp, imu) - rawCost(f, zpm, imu) - rawCost(f, zmp, imu) + rawCost(f, zmm, imu)) / (4 * eps * eps);
      hess_num(j, k) = h; hess_num(k, j) = h;
    }
  }
  const double max_hess_err = (hess_num - A_ff).cwiseAbs().maxCoeff();
  const double A_ff_scale = A_ff.cwiseAbs().maxCoeff();
  std::printf("  max_hessian_error=%.6e   A_ff_max_abs_entry=%.6e   relative=%.6e\n",
              max_hess_err, A_ff_scale, A_ff_scale > 0 ? max_hess_err / A_ff_scale : 0.0);
  check(max_hess_err < 0.02 * std::max(A_ff_scale, 1.0), "numerical Hessian ~= A_ff, relative to A_ff's own scale (item 12)", max_hess_err);
}

// ============================================================================
// item 13: IMU-only optimum test.
// ============================================================================
static void testImuOnlyOptimum() {
  Fixture f_true = makeFixture(7, 777);
  // Zero-noise samples generated from f_true's OWN spline -> f_true's z=0
  // (its own control points, as currently set) is EXACTLY the minimizer.
  const auto imu = synthesizeImu(f_true.spline, f_true.bias_acc, f_true.bias_gyr, f_true.gravity, 200.0, 0.0, 0.0, 1);

  // Start optimization from a DIFFERENT, perturbed spline (z0 != z_true) --
  // perturbed WITHIN the null space only (via applyZ), so the head anchor
  // (p0/v0/R0) stays IDENTICAL between f and f_true; only the free/
  // observable trajectory shape differs.
  Fixture f = f_true;
  std::mt19937 rng(31415);
  std::uniform_real_distribution<double> pert(-0.03, 0.03);
  {
    Eigen::VectorXd z_pert = Eigen::VectorXd::Zero(f.dimZ);
    for (int j = 0; j < f.dEta; ++j) z_pert(j) = pert(rng);
    PoseControlSpline s2; V3D ba2, bg2, g2;
    applyZ(f, z_pert, s2, ba2, bg2, g2);
    f.spline = s2; f.bias_acc = ba2; f.bias_gyr = bg2; f.gravity = g2;
  }

  double last_correction_norm = 1e9;
  for (int iter = 0; iter < 8; ++iter) {
    Eigen::MatrixXd A_ff; Eigen::VectorXd xi_z;
    buildAffXiZ(f, imu, A_ff, xi_z);
    const Eigen::MatrixXd A_ff_pinv = generalPseudoInverse(A_ff, 1e-6);
    const Eigen::VectorXd delta = A_ff_pinv * xi_z;
    last_correction_norm = delta.norm();
    // Apply: shift f's own spline control points by delta's eta part (identity Z here).
    PoseControlSpline s2; V3D ba2, bg2, g2;
    Eigen::VectorXd z_applied = Eigen::VectorXd::Zero(f.dimZ);
    z_applied.head(f.dEta) = delta.head(f.dEta);
    applyZ(f, z_applied, s2, ba2, bg2, g2);
    f.spline = s2; f.bias_acc = ba2; f.bias_gyr = bg2; f.gravity = g2;
  }
  // f.spline should now match f_true.spline closely.
  const double t_probe = 0.5 * (f.spline.t0() + f.spline.t1());
  const double p_err = (f.spline.posAt(t_probe) - f_true.spline.posAt(t_probe)).norm();
  const double a_err = (f.spline.accAt(t_probe) - f_true.spline.accAt(t_probe)).norm();
  std::printf("  after 8 production-prior Newton steps: p_err=%.6e  a_err=%.6e  last_correction_norm=%.6e\n",
              p_err, a_err, last_correction_norm);
  check(p_err < 1e-4, "converged z_prior matches the TRUE IMU-only optimum in position (item 13)", p_err);
  check(a_err < 1e-2, "converged z_prior matches the TRUE IMU-only optimum in acceleration", a_err);
  check(last_correction_norm < 1e-5, "Newton correction has converged (near-zero further update)", last_correction_norm);
}

// ============================================================================
// item 14: incremental-update / composability test.
// ============================================================================
static void testIncrementalInformationComposability() {
  Fixture f = makeFixture(7, 4242);
  const auto imu_all = synthesizeImu(f.spline, f.bias_acc, f.bias_gyr, f.gravity, 200.0, 0.015, 0.0015, 55);
  const size_t half = imu_all.size() / 2;
  const std::vector<ImuSample> imu_a(imu_all.begin(), imu_all.begin() + half);
  const std::vector<ImuSample> imu_b(imu_all.begin() + half, imu_all.end());

  Eigen::MatrixXd A_full, A_a, A_b;
  Eigen::VectorXd xi_full, xi_a, xi_b;
  buildAffXiZ(f, imu_all, A_full, xi_full);
  buildAffXiZ(f, imu_a, A_a, xi_a);
  buildAffXiZ(f, imu_b, A_b, xi_b);

  const double A_err = (A_full - (A_a + A_b)).cwiseAbs().maxCoeff();
  const double A_full_scale = A_full.cwiseAbs().maxCoeff();
  const double xi_err = (xi_full - (xi_a + xi_b)).norm();
  std::printf("  Lambda_A+Lambda_B vs Lambda_full: max_abs_err=%.6e   xi_A+xi_B vs xi_full: norm_err=%.6e\n", A_err, xi_err);
  check(A_err < 1e-9 * std::max(A_full_scale, 1.0), "Lambda_new = Lambda_A + Lambda_new_IMU EXACTLY, relative to Lambda_full's own scale (item 14)", A_err / std::max(A_full_scale, 1.0));
  check(xi_err < 1e-9, "xi_new = xi_A + xi_new_IMU EXACTLY", xi_err);

  const Eigen::VectorXd mean_from_sum = generalPseudoInverse(A_a + A_b, 1e-6) * (xi_a + xi_b);
  const Eigen::VectorXd mean_from_full = generalPseudoInverse(A_full, 1e-6) * xi_full;
  const double mean_err = (mean_from_sum - mean_from_full).norm();
  std::printf("  mean_from_incremental_sum vs mean_from_direct_full: norm_err=%.6e\n", mean_err);
  check(mean_err < 1e-9, "composed mean matches direct-optimization mean", mean_err);
}

int main() {
  std::printf("Prior-mean-correctness: quadratic/gradient/Hessian equivalence, IMU-only optimum, incremental composability\n");
  std::printf("-- items 9-12 --\n");
  testQuadraticGradientHessianEquivalence();
  std::printf("-- item 13 --\n");
  testImuOnlyOptimum();
  std::printf("-- item 14 --\n");
  testIncrementalInformationComposability();
  std::printf("\n%d failed\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
