// R73 (F-415): write-only per-iteration and per-scan update logs of the coupled (joint-knot) estimator.
#include "livo_recon/processing/lio_coupled.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/map/voxelmap.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace livo_recon
{
namespace
{
double blockNorm(const Eigen::VectorXd& v, int off, int len) { return v.segment(off, len).norm(); }

// sqrt of the sum over knots 1..N-1 of |segment|^2 for the 3-vector at knotOffset(k)+c
double acrossKnots(const Eigen::VectorXd& v, const JointKnotTrajectory& t, int c)
{
  double s = 0.0;
  for (int k = 1; k < t.knotCount(); ++k) s += v.segment<3>(t.knotOffset(k) + 3 * c).squaredNorm();
  return std::sqrt(s);
}
double acrossBias(const Eigen::VectorXd& v, const JointKnotTrajectory& t, int which)   // 0 = gyro bias, 1 = accel bias
{
  double s = 0.0;
  for (int k = 0; k < t.knotCount(); ++k) s += v.segment<3>(t.biasOffset(k) + 3 * which).squaredNorm();
  return std::sqrt(s);
}
double traceBlock(const Eigen::MatrixXd& P, int off) { return P.block<3, 3>(off, off).trace(); }
}  // namespace

void LioProcCoupled::writeUpdateLogIteration(const MeasureGroup& mg, int iteration, int n_residuals) const
{
  static PersistentLogStream log("coupled_update_iter.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  const JointKnotTrajectory& T = trajectory_;
  const JointKnotSolve& s = last_solve_;
  if (first)
  {
    o << "frame,t_abs,iter,n_residuals,imu_prior_info_scale,knot_count";
    for (const char* v : {"delta", "lidar_delta", "prior_delta"})
    {
      o << ',' << v << "_rot_allknots_rad," << v << "_pos_allknots_m," << v << "_vel_allknots_mps," << v << "_bg_allknots_radps,"
        << v << "_ba_allknots_mps2," << v << "_gravity_mps2," << v << "_rot_tail_rad," << v << "_pos_tail_m," << v << "_vel_tail_mps,"
        << v << "_norm_total";
    }
    o << ",A_eig_min,A_eig_max,A_cond,Gamma_L_eig_max,Gamma_L_rank_rel1e-9,vec_rot_allknots_rad,vec_pos_allknots_m\n";
  }
  o << voxel_map_->frame_idx_ << ',' << std::setprecision(17) << (mg.image.t + data_queues_->start_time) << std::setprecision(10) << ','
    << iteration << ',' << n_residuals << ',' << copts_.imu_prior_info_scale << ',' << T.knotCount();
  const int tail = T.knotCount() - 1;
  for (const Eigen::VectorXd* v : {&s.delta, &s.lidar_delta, &s.prior_delta})
  {
    o << ',' << acrossKnots(*v, T, 0) << ',' << acrossKnots(*v, T, 1) << ',' << acrossKnots(*v, T, 2) << ',' << acrossBias(*v, T, 0)
      << ',' << acrossBias(*v, T, 1) << ',' << blockNorm(*v, T.gravityOffset(), 3);
    if (tail >= 1)
      o << ',' << blockNorm(*v, T.knotOffset(tail), 3) << ',' << blockNorm(*v, T.knotOffset(tail) + 3, 3) << ','
        << blockNorm(*v, T.knotOffset(tail) + 6, 3);
    else
      o << ",nan,nan,nan";
    o << ',' << v->norm();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eA(s.A);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eG(s.Gamma_L);
  const double amin = eA.info() == Eigen::Success ? eA.eigenvalues()(0) : std::nan("");
  const double amax = eA.info() == Eigen::Success ? eA.eigenvalues()(eA.eigenvalues().size() - 1) : std::nan("");
  const double gmax = eG.info() == Eigen::Success ? eG.eigenvalues()(eG.eigenvalues().size() - 1) : std::nan("");
  int grank = 0;
  if (eG.info() == Eigen::Success) for (int i = 0; i < eG.eigenvalues().size(); ++i) if (eG.eigenvalues()(i) > 1e-9 * gmax) ++grank;
  o << ',' << amin << ',' << amax << ',' << (amin > 0.0 ? amax / amin : std::numeric_limits<double>::infinity()) << ',' << gmax << ','
    << grank << ',' << acrossKnots(s.vec, T, 0) << ',' << acrossKnots(s.vec, T, 1) << '\n';
  o.flush();
}

void LioProcCoupled::writeUpdateLogScan(const MeasureGroup& mg, const std::string& stop, int completed) const
{
  static PersistentLogStream log("coupled_update_scan.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  const JointKnotTrajectory& T = trajectory_;
  const Eigen::MatrixXd& Pp = joint_prior_.P;
  const Eigen::MatrixXd& Pq = last_solve_.posterior;
  if (first)
  {
    o << "frame,t_abs,stop,iterations,n_residuals,imu_prior_info_scale,knot_count,"
         "tail18_prior_trace_rot,tail18_prior_trace_pos,tail18_prior_trace_vel,tail18_post_trace_rot,tail18_post_trace_pos,"
         "tail18_post_trace_vel";
    for (int k = 1; k < T.knotCount(); ++k)
      for (const char* c : {"rot", "pos", "vel"}) o << ",knot" << k << "_prior_trace_" << c << ",knot" << k << "_post_trace_" << c;
    o << "\n";
  }
  const StateGroup& prop = state_propagat_;
  const StateGroup& post = *state_;
  o << voxel_map_->frame_idx_ << ',' << std::setprecision(17) << (mg.image.t + data_queues_->start_time) << std::setprecision(10) << ','
    << stop << ',' << completed << ',' << residuals_.size() << ',' << copts_.imu_prior_info_scale << ',' << T.knotCount();
  o << ',' << traceBlock(prop.cov(), StateGroup::idxR()) << ',' << traceBlock(prop.cov(), StateGroup::idxP()) << ','
    << traceBlock(prop.cov(), StateGroup::idxV()) << ',' << traceBlock(post.cov(), StateGroup::idxR()) << ','
    << traceBlock(post.cov(), StateGroup::idxP()) << ',' << traceBlock(post.cov(), StateGroup::idxV());
  for (int k = 1; k < T.knotCount(); ++k)
    for (int c = 0; c < 3; ++c)
      o << ',' << traceBlock(Pp, T.knotOffset(k) + 3 * c) << ',' << traceBlock(Pq, T.knotOffset(k) + 3 * c);
  o << '\n';
  o.flush();
}
}  // namespace livo_recon
