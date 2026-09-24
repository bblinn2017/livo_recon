#include "livo_recon/lio/pose_control_directional_redundancy.h"

#include <Eigen/Eigenvalues>
#include <algorithm>

namespace livo_recon
{

DirectionalRedundancyStats analyzeDirectionalRedundancy(
    const Eigen::MatrixXd& Lambda, int raw_residual_count, double rel_rank_thresh)
{
  DirectionalRedundancyStats st;
  st.raw_residual_count = raw_residual_count;
  st.reduced_state_dimension = static_cast<int>(Lambda.rows());
  if (Lambda.rows() == 0 || Lambda.rows() != Lambda.cols() || !Lambda.allFinite()) return st;

  const Eigen::MatrixXd Lsym = 0.5 * (Lambda + Lambda.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Lsym);
  if (es.info() != Eigen::Success) return st;

  // SelfAdjointEigenSolver returns ASCENDING order -- reverse to descending
  // (largest information first, matching "dominant direction" first).
  const int d = static_cast<int>(Lsym.rows());
  Eigen::VectorXd ev = es.eigenvalues().cwiseMax(0.0).reverse();
  Eigen::MatrixXd evec = es.eigenvectors().rowwise().reverse();

  st.eigenvalues = ev;
  st.eigenvectors = evec;

  const double lambda_max = (d > 0) ? ev(0) : 0.0;
  const double thresh = rel_rank_thresh * std::max(lambda_max, 1e-300);
  int rank = 0;
  for (int i = 0; i < d; ++i) if (ev(i) > thresh) ++rank;
  st.effective_rank = rank;

  double lambda_min_nonzero = lambda_max;
  for (int i = 0; i < d; ++i) if (ev(i) > thresh) lambda_min_nonzero = std::min(lambda_min_nonzero, ev(i));
  st.condition_number = (lambda_min_nonzero > 0.0) ? lambda_max / lambda_min_nonzero : 0.0;

  const double total = ev.sum();
  st.cumulative_information_fraction = Eigen::VectorXd::Zero(d);
  if (total > 0.0) {
    double running = 0.0;
    for (int i = 0; i < d; ++i) { running += ev(i); st.cumulative_information_fraction(i) = running / total; }
  }

  st.valid = true;
  return st;
}

}  // namespace livo_recon
