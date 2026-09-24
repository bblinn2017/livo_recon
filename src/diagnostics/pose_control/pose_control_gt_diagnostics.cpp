#include "livo_recon/diagnostics/pose_control/pose_control_gt_diagnostics.h"
#include "livo_recon/lio/pose_control_covariance.h"

namespace livo_recon
{

PoseControlGtError computeGtError(
    const V3D& p_est, const M3D& R_est, const V3D& v_est,
    const V3D& p_gt, const M3D& R_gt, const V3D& v_gt)
{
  PoseControlGtError e;
  e.e_p = p_est - p_gt;
  e.e_R = Log(M3D(R_gt.transpose() * R_est));
  e.e_v = v_est - v_gt;
  return e;
}

PoseControlNees computeNees(
    const PoseControlGtError& err, const Eigen::Matrix<double, 9, 9>& P_9, double rel_thresh)
{
  PoseControlNees n;
  const Eigen::MatrixXd P_pp = P_9.block<3, 3>(3, 3), P_RR = P_9.block<3, 3>(0, 0), P_vv = P_9.block<3, 3>(6, 6);
  n.nees_p = (err.e_p.transpose() * generalPseudoInverse(P_pp, rel_thresh) * err.e_p)(0) / 3.0;
  n.nees_R = (err.e_R.transpose() * generalPseudoInverse(P_RR, rel_thresh) * err.e_R)(0) / 3.0;
  n.nees_v = (err.e_v.transpose() * generalPseudoInverse(P_vv, rel_thresh) * err.e_v)(0) / 3.0;

  Eigen::Matrix<double, 9, 1> e9;
  e9.segment<3>(0) = err.e_R; e9.segment<3>(3) = err.e_p; e9.segment<3>(6) = err.e_v;
  n.nees_rpv = (e9.transpose() * generalPseudoInverse(P_9, rel_thresh) * e9)(0) / 9.0;
  return n;
}

}  // namespace livo_recon
