#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

namespace {
int fail(const char* name, double value, double tol)
{
  std::printf("[FAIL] %s %.6e tol %.6e\n", name, value, tol);
  return 1;
}
}

int main()
{
  Eigen::Matrix2d Pk;
  Pk << 2.0, 0.3, 0.3, 1.5;
  Eigen::Matrix2d Phi;
  Phi << 1.0, 0.2, 0.0, 1.0;
  Eigen::Matrix2d Q;
  Q << 0.4, 0.05, 0.05, 0.2;
  Eigen::Matrix2d Pl = Phi * Pk * Phi.transpose() + Q;
  Eigen::Matrix2d Pkl = Pk * Phi.transpose();

  const Eigen::Matrix2d cond = Pl - Pkl.transpose() * Pk.inverse() * Pkl;
  if ((cond - Q).norm() > 1e-12)
    return fail("temporal conditional covariance equals propagated Q", (cond-Q).norm(), 1e-12);

  Eigen::Matrix<double,1,2> H;
  H << 1.0, -0.5;
  const double R = 0.3;
  const Eigen::Matrix<double,1,1> S_joint = H * Pl * H.transpose() + Eigen::Matrix<double,1,1>::Constant(R);
  const Eigen::Matrix<double,1,2> H_eff = H * Phi;
  const Eigen::Matrix<double,1,1> R_eff = Eigen::Matrix<double,1,1>::Constant(R + (H * Q * H.transpose())(0,0));
  const Eigen::Matrix<double,1,1> S_eff = H_eff * Pk * H_eff.transpose() + R_eff;
  if ((S_joint - S_eff).norm() > 1e-12)
    return fail("effective covariance gives same innovation covariance", (S_joint-S_eff).norm(), 1e-12);

  const Eigen::Vector2d xk0(0.6, -0.4);
  const Eigen::Vector2d xl0 = Phi * xk0;
  const Eigen::Matrix<double,1,1> z = H * xl0;
  const Eigen::Matrix2d Pk_post_joint =
      Pk - Pkl * H.transpose() * S_joint.inverse() * H * Pkl.transpose();
  const Eigen::Vector2d K_joint = Pkl * H.transpose() * S_joint.inverse();
  const Eigen::Vector2d xk_post_joint = xk0 + K_joint * (z(0) - (H * xl0)(0));
  const Eigen::Matrix2d Pk_post_eff =
      Pk - Pk * H_eff.transpose() * S_eff.inverse() * H_eff * Pk;
  const Eigen::Vector2d K_eff = Pk * H_eff.transpose() * S_eff.inverse();
  const Eigen::Vector2d xk_post_eff = xk0 + K_eff * (z(0) - (H_eff * xk0)(0));

  if ((Pk_post_joint - Pk_post_eff).norm() > 1e-12)
    return fail("joint and effective-covariance posterior covariance agree", (Pk_post_joint-Pk_post_eff).norm(), 1e-12);
  if ((xk_post_joint - xk_post_eff).norm() > 1e-12)
    return fail("joint and effective-covariance posterior mean agree", (xk_post_joint-xk_post_eff).norm(), 1e-12);

  // A common measurement can reduce uncertainty in a remote state through a nonzero cross-covariance.
  Eigen::Matrix4d P;
  P.setIdentity();
  P(0,2) = P(2,0) = 0.5;
  P(1,3) = P(3,1) = 0.4;
  Eigen::Matrix<double,1,4> Hg;
  Hg << 1.0, 0.0, 0.0, 0.0;
  const Eigen::Matrix<double,1,1> Rg = Eigen::Matrix<double,1,1>::Constant(0.2);
  const Eigen::Matrix4d reduction = P * Hg.transpose() *
      (Hg * P * Hg.transpose() + Rg).inverse() * Hg * P;
  if (reduction.block<2,2>(2,2).trace() <= 0.0)
    return fail("nonlocal covariance coupling produces remote uncertainty reduction", reduction.block<2,2>(2,2).trace(), 0.0);

  std::printf("reference temporal/latent-state covariance tests: PASS\n");
  return 0;
}
