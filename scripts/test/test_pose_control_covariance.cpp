// Validates schurComplementFreeCovariance() against the textbook identity:
// for a jointly Gaussian [head;free] with covariance P_joint and
// information A_joint=P_joint^-1, the Schur complement of A_joint's head
// block IS exactly P_joint's own free-free covariance block -- ground
// truth taken from the covariance matrix directly (not re-derived via the
// function under test), so this is a genuine external check.
#include "livo_recon/lio/pose_control_covariance.h"
#include <cstdio>
#include <random>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  printf("  [%s] %-40s %.3e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) g_fail++;
}

static Eigen::MatrixXd randomSpd(int n, std::mt19937& rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) M(i, j) = nd(rng);
  return M * M.transpose() + 0.5 * Eigen::MatrixXd::Identity(n, n);
}

int main() {
  printf("Schur-complement free-covariance validation\n");
  std::mt19937 rng(31415);
  const int m = 9, n = 12;   // head dim, free dim

  for (int trial = 0; trial < 5; ++trial) {
    Eigen::MatrixXd P_joint = randomSpd(m + n, rng);
    Eigen::MatrixXd A_joint = P_joint.inverse();

    Eigen::MatrixXd A_hh = A_joint.block(0, 0, m, m);
    Eigen::MatrixXd A_hf = A_joint.block(0, m, m, n);
    Eigen::MatrixXd A_ff = A_joint.block(m, m, n, n);
    Eigen::MatrixXd P_ff_truth = P_joint.block(m, m, n, n);

    Eigen::MatrixXd P_free;
    bool ok = schurComplementFreeCovariance(A_hh, A_hf, A_ff, P_free);
    check(ok, "schur_solve_succeeded");
    double err = (P_free - P_ff_truth).norm() / P_ff_truth.norm();
    check(err < 1e-8, "P_free matches P_joint's own free-free block", err);
  }

  // Sanity: STRONGER head prior (smaller P0 -> larger A_hh) should still
  // give a well-defined, finite free covariance -- and, holding
  // everything else fixed, a WEAKER head prior (larger head uncertainty)
  // should make the free covariance LARGER (head uncertainty propagates
  // forward), which is exactly spec item 12's "P_tail_A != P_tail_B" test
  // at the level of this isolated linear-algebra mechanism.
  {
    Eigen::MatrixXd A_hf = Eigen::MatrixXd::Zero(m, n);
    A_hf.block(0, 0, m, std::min(m, n)) = 0.3 * Eigen::MatrixXd::Identity(m, std::min(m, n));
    Eigen::MatrixXd A_ff = Eigen::MatrixXd::Identity(n, n) * 2.0;

    Eigen::MatrixXd A_hh_strong = Eigen::MatrixXd::Identity(m, m) * 100.0;  // tight prior (small P0)
    Eigen::MatrixXd A_hh_weak   = Eigen::MatrixXd::Identity(m, m) * 1.0;    // loose prior (large P0)
    Eigen::MatrixXd P_strong, P_weak;
    check(schurComplementFreeCovariance(A_hh_strong, A_hf, A_ff, P_strong), "strong-prior solve ok");
    check(schurComplementFreeCovariance(A_hh_weak, A_hf, A_ff, P_weak), "weak-prior solve ok");
    double tr_strong = P_strong.trace(), tr_weak = P_weak.trace();
    check(tr_weak > tr_strong, "looser head prior -> larger free covariance (uncertainty propagates)",
          tr_weak - tr_strong);
    printf("    trace(P_free | tight P0) = %.6f, trace(P_free | loose P0) = %.6f\n", tr_strong, tr_weak);
  }

  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
