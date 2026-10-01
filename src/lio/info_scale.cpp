#include "livo_recon/lio/info_scale.h"
#include "livo_recon/utils/algo/symmetrize.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace livo_recon
{

InfoScaleStats applyBlockInfoScale(const InfoScaleOptions& opts,
                                   Eigen::Matrix<double, 6, 6>& HtH,
                                   Eigen::Matrix<double, 6, 1>& Htz)
{
  InfoScaleStats st;
  st.trace_before = HtH.trace();
  st.gnorm_before = Htz.norm();
  st.trace_after = st.trace_before;
  st.gnorm_after = st.gnorm_before;
  if (!opts.on()) return st;

  Eigen::Matrix<double, 6, 1> s;
  s.head<3>().setConstant(1.0 / std::sqrt(opts.rotation));
  s.tail<3>().setConstant(1.0 / std::sqrt(opts.position));

  // pinv(Lambda) g through the symmetric eigendecomposition.
  const Eigen::Matrix<double, 6, 6> Lam = symmetrized(HtH);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Lam);
  const Eigen::Matrix<double, 6, 1> ev = es.eigenvalues();
  const double lmax = ev.maxCoeff();
  const double tol = 6.0 * std::numeric_limits<double>::epsilon() * std::max(lmax, 0.0);
  Eigen::Matrix<double, 6, 1> coef = es.eigenvectors().transpose() * Htz;
  for (int i = 0; i < 6; ++i) {
    if (ev(i) > tol) coef(i) /= ev(i);
    else { coef(i) = 0.0; ++st.discarded_modes; }
  }
  const Eigen::Matrix<double, 6, 1> x_ls = es.eigenvectors() * coef;   // measurement-only solution

  const Eigen::Matrix<double, 6, 6> Sm = s.asDiagonal();
  const Eigen::Matrix<double, 6, 6> Lam_new = Sm * HtH * Sm;
  Htz = Lam_new * x_ls;
  HtH = Lam_new;
  st.trace_after = HtH.trace();
  st.gnorm_after = Htz.norm();
  return st;
}

}  // namespace livo_recon
