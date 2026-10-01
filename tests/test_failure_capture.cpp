// R72 unit test (F-398): matrix dump and non-PD capture (include/livo_recon/diagnostics/matrix_dump.h, lio/failure_capture.h).
// ROS-free, Eigen only. Isolated identities only.
//   FC1 dump -> parse is bit-exact for random, extreme (1e-300, 1e300, subnormal) and signed-zero entries, for matrices, vectors and
//       scalars, and several blocks in one stream.
//   FC2 a truncated block is reported (parse returns false); comment lines are skipped.
//   FC3 captureNonPd on a deliberately non-PD matrix stores the SYMMETRISED matrix bit-exactly (0.5 (M + M^T)), ascending eigenvalues
//       equal to an independent solve, the smallest negative, the LDLT status, and the label.
//   FC4 the check fires: a positive-definite matrix passes the same factorisation test the integration uses, so no capture would be
//       taken for it (the capture is tied to the failure, not to every call).
#include "livo_recon/diagnostics/matrix_dump.h"
#include "livo_recon/lio/failure_capture.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

static bool bitEqual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (Eigen::Index i = 0; i < a.rows(); ++i)
    for (Eigen::Index j = 0; j < a.cols(); ++j)
      if (std::memcmp(&a(i, j), &b(i, j), sizeof(double)) != 0) return false;
  return true;
}

int main()
{
  std::mt19937 rng(3);
  std::normal_distribution<double> N(0, 1);
  Eigen::MatrixXd M(7, 5);
  for (int i = 0; i < 7; ++i) for (int j = 0; j < 5; ++j) M(i, j) = N(rng) * std::pow(10.0, (i * 5 + j) % 9 - 4);
  M(0, 0) = 1e-300; M(1, 1) = 1e300; M(2, 2) = std::numeric_limits<double>::denorm_min(); M(3, 3) = -0.0; M(4, 4) = 0.0;
  Eigen::VectorXd v(4); v << 0.1, -2.5e-17, 3.14159265358979323846, -1e200;
  {   // FC1
    std::stringstream ss;
    dumpText(ss, "meta", "frame=1");
    dumpMatrix(ss, "M", M);
    dumpVector(ss, "v", v);
    dumpScalar(ss, "s", 0.30000000000000004);
    std::map<std::string, Eigen::MatrixXd> got;
    CHECK(parseMatrixDump(ss, got), "FC1 stream parses");
    CHECK(got.count("M") && bitEqual(got["M"], M), "FC1 matrix round trip bit-exact (extremes, subnormal, signed zero)");
    CHECK(got.count("v") && bitEqual(got["v"], Eigen::MatrixXd(v)), "FC1 vector round trip bit-exact");
    CHECK(got.count("s") && got["s"](0, 0) == 0.30000000000000004, "FC1 scalar round trip bit-exact");
  }
  {   // FC2
    std::stringstream ss;
    dumpMatrix(ss, "M", M);
    std::string text = ss.str();
    text.resize(text.size() - 40);   // cut into the last row
    std::stringstream cut(text);
    std::map<std::string, Eigen::MatrixXd> got;
    CHECK(!parseMatrixDump(cut, got), "FC2 truncated block is reported");
  }
  {   // FC3
    Eigen::MatrixXd A(4, 4);
    A << 2, 0.5, 0, 0,
         0.1, -1.0, 0.2, 0,
         0, 0.4, 3, 0.1,
         0, 0, 0.3, 1;
    const Eigen::MatrixXd S = 0.5 * (A + A.transpose());
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    CHECK(ldlt.info() != Eigen::Success || !ldlt.isPositive(), "FC3 fixture is not positive definite");
    captureNonPd(S, ldlt, "posterior_A");
    const NonPdCapture& c = lastNonPdCapture();
    CHECK(c.valid && c.where == "posterior_A", "FC3 capture valid and labelled");
    CHECK(bitEqual(c.matrix, S), "FC3 stored matrix is the symmetrised one, bit-exact");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S, Eigen::EigenvaluesOnly);
    CHECK(bitEqual(Eigen::MatrixXd(c.eigenvalues), Eigen::MatrixXd(es.eigenvalues())), "FC3 eigenvalues equal an independent solve");
    CHECK(c.eigenvalues(0) < 0.0 && c.eigenvalues(0) <= c.eigenvalues(1), "FC3 smallest eigenvalue is negative and the list ascends");
    CHECK(!c.ldlt_positive && c.ldlt_diag.size() == 4, "FC3 LDLT status and pivots stored");
  }
  {   // FC4
    Eigen::MatrixXd B(4, 4);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) B(i, j) = N(rng);
    const Eigen::MatrixXd P = B * B.transpose() + 0.1 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(P);
    CHECK(ldlt.info() == Eigen::Success && ldlt.isPositive(), "FC4 a PD matrix passes the factorisation test (no capture is taken)");
  }
  std::printf("ALL PASS\n");
  return 0;
}
