#include "livo_recon/utils/eval/nees_logger.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <fstream>
#include <iomanip>

namespace livo_recon
{

NeesResult computeNeesPerDof(const M3D& R_est, const V3D& p_est,
                              const M3D& R_gt, const V3D& p_gt,
                              const Eigen::Matrix<double, 6, 6>& P)
{
  NeesResult out;
  Eigen::Matrix<double, 6, 1> e;
  e.segment<3>(0) = Log(R_gt.transpose() * R_est);
  e.segment<3>(3) = p_est - p_gt;

  // Force EXACT symmetry (see the earlier fix's own comment history) --
  // still worth doing so SelfAdjointEigenSolver operates on the matrix the
  // caller actually intended, not an accidental near-symmetric one.
  const Eigen::Matrix<double, 6, 6> Psym = 0.5 * (P + P.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Psym);
  const Eigen::Matrix<double, 6, 1> raw_eigvals = es.eigenvalues();
  out.max_eig = raw_eigvals.maxCoeff();
  out.min_eig = raw_eigvals.minCoeff();
  // CQ-60 item 2/5: coupled's own P is measurably non-PSD -- confirmed by
  // direct measurement on the real binary: min_eig is NEGATIVE on every
  // single scan across a 489-scan run (not an occasional outlier), ranging
  // from ~-12% of max_eig on early scans down to ~-0.1% to -0.9% by
  // mid/late-run, but never once crossing zero. A first attempt at this
  // function clipped the negative eigenvalue to a small positive floor and
  // computed NEES anyway -- WRONG: the resulting numbers (single-digit to
  // multi-million NEES between ADJACENT scans, no physical trend) are
  // floating-point noise amplified by dividing by a value indistinguishable
  // from zero, not a real measurement. Rule 58 applies here precisely:
  // this is an impossible condition (a Mahalanobis distance is undefined
  // against a non-PSD "covariance"), and the correct response is to abort
  // loudly and report the degeneracy itself -- min_eig/max_eig are ALWAYS
  // logged raw (see logNeesPerDof) specifically so THIS finding (P is
  // persistently non-PSD, not any particular NEES value) is what survives,
  // rather than a fabricated number that looks like real data.
  //
  // This is a REAL property of the CURRENT (pre-CQ-57) binary's covariance
  // construction, not a bug in this logger to paper over -- CQ-57 item 4
  // (adopting decoupled's applyCovarianceUpdate() scheme in coupled too)
  // is the actual fix, and it is EXPLICITLY BLOCKED until CQ-60 item 2
  // captures its baseline on THIS binary, so this logger must not "fix"
  // what it is supposed to be measuring.
  if (out.min_eig <= 0.0) return out;  // out.valid stays false
  const Eigen::Matrix<double, 6, 1> eigvals = raw_eigvals;
  const Eigen::Matrix<double, 6, 6> V = es.eigenvectors();
  const Eigen::Matrix<double, 6, 1> y = V.transpose() * e;
  out.per_dof_whitened_sq = y.cwiseProduct(y).cwiseQuotient(eigvals);
  // Derived from the SAME eigendecomposition as per_dof_whitened_sq, not a
  // separate matrix inverse -- guarantees out.nees == sum(per_dof_whitened_sq)
  // by construction, rather than by hoping two different numerical paths
  // agree on an ill-conditioned matrix (exactly the bug just described).
  out.nees = out.per_dof_whitened_sq.sum();
  out.whitening_axes = V;
  out.valid = true;
  return out;
}

namespace
{
// SVD-orthogonalized average of a small set of rotations -- valid for a
// small-angle spread (a genuinely stationary window), not a general SO(3)
// mean. Mirrors the same approach CQ-60's own post-hoc Tier 1 script used.
M3D meanRotation(const std::vector<M3D>& Rs)
{
  M3D M = M3D::Zero();
  for (const auto& R : Rs) M += R;
  M /= static_cast<double>(Rs.size());
  Eigen::JacobiSVD<M3D> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  M3D R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0) {
    M3D U = svd.matrixU();
    U.col(2) *= -1.0;
    R = U * svd.matrixV().transpose();
  }
  return R;
}
}  // namespace

void Tier1NeesBuffer::addScan(const char* channel, int scan_id, double t_abs,
                               const M3D& R, const V3D& p, const Eigen::Matrix<double, 6, 6>& P)
{
  if (window_size_ <= 0 || done_) return;
  buf_.push_back({scan_id, t_abs, R, p, P});
  if (static_cast<int>(buf_.size()) < window_size_) return;

  done_ = true;
  std::vector<M3D> Rs;
  Rs.reserve(buf_.size());
  V3D p_sum = V3D::Zero();
  for (const auto& e : buf_) { Rs.push_back(e.R); p_sum += e.p; }
  const M3D R_mean = meanRotation(Rs);
  const V3D p_mean = p_sum / static_cast<double>(buf_.size());

  for (const auto& e : buf_) {
    const NeesResult r = computeNeesPerDof(e.R, e.p, R_mean, p_mean, e.P);
    logNeesPerDof(channel, e.scan_id, e.t_abs, r);
  }
  buf_.clear();
  buf_.shrink_to_fit();
}

void logNeesPerDof(const char* channel, int scan_id, double t_abs, const NeesResult& r)
{
  static PersistentLogStream log("nees_per_dof.txt");
  bool just_opened = false;
  std::ofstream& ofs = log.stream(&just_opened);
  if (just_opened)
    ofs << "channel,scan_id,t_abs,valid,nees,min_eig,max_eig,"
           "whitened_sq_0,whitened_sq_1,whitened_sq_2,"
           "whitened_sq_3,whitened_sq_4,whitened_sq_5\n";
  ofs << channel << "," << scan_id << "," << std::setprecision(12) << t_abs << ","
      << (r.valid ? 1 : 0) << "," << r.nees << "," << r.min_eig << "," << r.max_eig;
  for (int i = 0; i < 6; ++i) ofs << "," << r.per_dof_whitened_sq(i);
  ofs << "\n";
  ofs.flush();
}

}  // namespace livo_recon
