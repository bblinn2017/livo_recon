// R97A R4 unit test: calib/p0/gravity_cov_mode = fixed | calibration.
// Tests gravityCovarianceFromCalibration() (include/livo_recon/processing/calib_processing.h),
// the pure math factored out of CalibProc::applyCalibrationDerivedP0() so it can be exercised
// without a live NodeContext/ROS stack.
//   GC1 synthetic C_a with different transverse variances gives the expected C_tan (symmetric,
//       PSD, zero along u) -- the result is C_tan + s_r2*u*u^T, so subtracting s_r2*u*u^T
//       recovers C_tan exactly (the s2 tangent-gravity model was removed; gravDim is always 3).
//   GC2 "fixed" leaves the block equal to s_r2*I -- CalibProcOptions::p0_gravity_cov_mode
//       defaults to "fixed", and applyCalibrationDerivedP0() only calls
//       gravityCovarianceFromCalibration() when the mode is "calibration" (see calib_processing.cpp)
//       -- i.e. the gravity block is provably untouched at the default by construction, not
//       re-verified numerically here (that would need a full CalibProc/NodeContext fixture, which
//       this ROS-free test deliberately avoids -- see the report for this limitation).
#include <ros/ros.h>
#include "livo_recon/processing/calib_processing.h"
#include "livo_recon/utils/algo/math.h"

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  // GC1: gravDim=3 (always, since the s2 model was removed).
  {
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> Uvar(1e-6, 1e-3);
    const V3D u = V3D(0.1, -0.2, 0.97).normalized();
    const M3D R_init = M3D(Eigen::AngleAxisd(0.3, V3D(0.3, 0.6, 0.74).normalized()));
    const double s_r2 = 1.0e-5;

    for (int trial = 0; trial < 20; ++trial)
    {
      // A diagonal-in-body-frame C_a with distinct transverse variances (the exact direction C_a
      // is expressed in doesn't matter for this check -- symmetry/PSD/zero-along-u hold regardless).
      M3D C_a = M3D::Zero();
      C_a(0, 0) = Uvar(rng); C_a(1, 1) = Uvar(rng); C_a(2, 2) = Uvar(rng);

      const Eigen::MatrixXd P_g = gravityCovarianceFromCalibration(u, R_init, C_a, s_r2);
      CHECK(P_g.rows() == 3 && P_g.cols() == 3, "GC1 returns a 3x3 block");
      const M3D C_tan = M3D(P_g) - s_r2 * u * u.transpose();   // P_g = C_tan + s_r2*u*u^T by definition
      CHECK((C_tan - C_tan.transpose()).norm() < 1e-12 * std::max(1.0, C_tan.norm()), "GC1 C_tan is symmetric");
      Eigen::SelfAdjointEigenSolver<M3D> es(C_tan);
      CHECK(es.eigenvalues().minCoeff() > -1e-15, "GC1 C_tan is PSD");
      CHECK((C_tan * u).norm() < 1e-9 * std::max(1.0, C_tan.norm()), "GC1 C_tan is zero along u (T C T has u in its null space)");
    }
  }

  // GC2: see the file header -- "fixed" is a structural (not numerical) check in this ROS-free test.
  {
    CalibProcOptions opts;
    CHECK(opts.p0_gravity_cov_mode == "fixed", "GC2 calib/p0/gravity_cov_mode defaults to fixed");
  }

  std::printf("ALL PASS\n");
  return 0;
}
