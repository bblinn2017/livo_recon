// Pure numerical validation of covarianceInformationUpdate() -- items
// 14/15/16/17 of the 2026-09-22 covariance correction. No LIO/estimator
// code involved; this tests the Woodbury-form update algebra in isolation.
#include "livo_recon/lio/pose_control_covariance.h"
#include <cstdio>
#include <random>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  printf("  [%s] %-56s %.3e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) g_fail++;
}

static Eigen::MatrixXd randomSpd(int n, std::mt19937& rng, double scale = 1.0) {
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) M(i, j) = nd(rng);
  return scale * (M * M.transpose() + 0.5 * Eigen::MatrixXd::Identity(n, n));
}

// ---- Item 14: agreement with a direct (P_prior^-1 + Lambda_meas)^-1 -------
static void test14() {
  printf("Item 14: covariance update algebra vs direct information-form inverse\n");
  std::mt19937 rng(7);
  const int n = 10, m = 6;
  for (int trial = 0; trial < 5; ++trial) {
    Eigen::MatrixXd P_prior = randomSpd(n, rng);
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(m, n);
    Eigen::MatrixXd Rinv = randomSpd(m, rng, 0.5);   // R^-1 (PD)
    Eigen::MatrixXd Lambda_meas = H.transpose() * Rinv * H;

    Eigen::MatrixXd P_post_a;
    CovarianceUpdateDiagnostics diag;
    bool ok = covarianceInformationUpdate(P_prior, Lambda_meas, P_post_a, diag);
    check(ok, "update succeeded");

    const Eigen::MatrixXd Lambda_prior = P_prior.inverse();
    const Eigen::MatrixXd P_post_b = (Lambda_prior + Lambda_meas).inverse();

    const double err_fro = (P_post_a - P_post_b).norm() / P_post_b.norm();
    const double err_abs = (P_post_a - P_post_b).cwiseAbs().maxCoeff();
    check(err_fro < 1e-8, "relative Frobenius error vs direct inverse", err_fro);
    check(err_abs < 1e-8, "max absolute error vs direct inverse", err_abs);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * ((P_prior - P_post_a) + (P_prior - P_post_a).transpose()));
    check(es.eigenvalues().minCoeff() > -1e-8, "P_prior - P_post is PSD", es.eigenvalues().minCoeff());
  }
}

// ---- Item 15: weak information is preserved, not zeroed -------------------
static void test15() {
  printf("Item 15: weak measurement information leaves posterior ~= prior\n");
  std::mt19937 rng(11);
  const int n = 8;
  Eigen::MatrixXd P_prior = randomSpd(n, rng);
  // A measurement with one PDF-tiny eigenvalue direction and otherwise zero.
  Eigen::VectorXd dir = Eigen::VectorXd::Random(n).normalized();
  const double tiny = 1e-8;
  Eigen::MatrixXd Lambda_meas = tiny * dir * dir.transpose();

  Eigen::MatrixXd P_post;
  CovarianceUpdateDiagnostics diag;
  bool ok = covarianceInformationUpdate(P_prior, Lambda_meas, P_post, diag);
  check(ok, "update succeeded");
  const double var_prior_dir = dir.transpose() * P_prior * dir;
  const double var_post_dir = dir.transpose() * P_post * dir;
  const double rel_change = std::abs(var_post_dir - var_prior_dir) / var_prior_dir;
  check(rel_change < 1e-3, "posterior variance ~= prior variance in the weak direction", rel_change);
  check(var_post_dir > 1e-6, "posterior variance NOT collapsed to zero (guards the old 1e-2 bug)", var_post_dir);
}

// ---- Item 16: zero measurement information leaves posterior EXACTLY prior -
static void test16() {
  printf("Item 16: zero measurement information -> P_post == P_prior exactly\n");
  std::mt19937 rng(13);
  const int n = 7;
  Eigen::MatrixXd P_prior = randomSpd(n, rng);
  Eigen::MatrixXd Lambda_meas = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd P_post;
  CovarianceUpdateDiagnostics diag;
  bool ok = covarianceInformationUpdate(P_prior, Lambda_meas, P_post, diag);
  check(ok, "update succeeded");
  double err = (P_post - P_prior).norm() / P_prior.norm();
  check(err < 1e-10, "P_post == P_prior to numerical precision", err);
}

// ---- Item 17: strong measurement contracts the observed direction only ---
static void test17() {
  printf("Item 17: strong measurement contracts ONLY the observed direction\n");
  std::mt19937 rng(17);
  const int n = 6;
  // DIAGONAL (uncorrelated) prior for this specific check -- "leaves
  // unobserved directions unchanged" only holds when there is no prior
  // correlation for the observed direction's information to propagate
  // through (a correlated prior legitimately updates correlated axes too,
  // which is correct Bayesian behavior, not a bug -- confirmed separately
  // by item 14's exact agreement with the direct inverse on random,
  // correlated priors).
  std::uniform_real_distribution<double> ud(0.5, 3.0);
  Eigen::MatrixXd P_prior = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) P_prior(i, i) = ud(rng);
  Eigen::VectorXd dir = Eigen::VectorXd::Unit(n, 0);   // observe axis 0 only
  Eigen::VectorXd orth = Eigen::VectorXd::Unit(n, 1);  // an unobserved axis

  double prev_var_dir = dir.transpose() * P_prior * dir;
  double var_orth_before = orth.transpose() * P_prior * orth;
  bool monotonic = true;
  double var_post_dir_final = 0;
  for (double w : {1.0, 10.0, 100.0, 1000.0, 1e6}) {
    Eigen::MatrixXd Lambda_meas = w * dir * dir.transpose();
    Eigen::MatrixXd P_post;
    CovarianceUpdateDiagnostics diag;
    covarianceInformationUpdate(P_prior, Lambda_meas, P_post, diag);
    double var_dir = dir.transpose() * P_post * dir;
    double var_orth = orth.transpose() * P_post * orth;
    if (var_dir > prev_var_dir + 1e-12) monotonic = false;
    prev_var_dir = var_dir;
    var_post_dir_final = var_dir;
    check(std::abs(var_orth - var_orth_before) < 1e-9, "unobserved direction unchanged", std::abs(var_orth - var_orth_before));
  }
  check(monotonic, "observed-direction variance decreases monotonically with weight");
  check(var_post_dir_final < 1e-4, "observed direction variance -> ~0 at very strong weight", var_post_dir_final);
}

int main() {
  test14();
  test15();
  test16();
  test17();
  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
