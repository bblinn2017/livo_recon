#pragma once

#include <Eigen/Dense>

namespace livo_recon
{

struct CovarianceUpdateDiagnostics
{
  bool prior_symmetric = false, prior_psd = false;
  bool meas_symmetric = false, meas_psd = false;
  bool post_finite = false, post_symmetric = false, post_psd = false;
  double min_eig_prior = 0.0, min_eig_meas = 0.0, min_eig_post = 0.0;
  double max_eig_prior = 0.0, max_eig_meas = 0.0, max_eig_post = 0.0;
  double asymmetry_prior = 0.0, asymmetry_meas = 0.0, asymmetry_post = 0.0;
};

bool covarianceInformationUpdate(
    const Eigen::MatrixXd& P_prior, const Eigen::MatrixXd& Lambda_meas,
    Eigen::MatrixXd& P_post, CovarianceUpdateDiagnostics& diag,
    double tol = 1e-6);

Eigen::MatrixXd generalPseudoInverse(const Eigen::MatrixXd& matrix,
                                     double relative_threshold);

}  // namespace livo_recon
