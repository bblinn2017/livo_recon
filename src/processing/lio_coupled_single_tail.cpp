#include "livo_recon/processing/pose_control_coupled_modes.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/utils/algo/math.h"

namespace livo_recon
{
bool solveSingleTailControl(const PoseControlSpline& spline,const PoseControlFreeLayout& layout,
 const PoseControlHeadNullspace& hns,const Eigen::MatrixXd& sigma_prior,
 const Eigen::MatrixXd& J_tail_full,const Eigen::Matrix<double,6,6>& P_tail,
 const Eigen::Matrix<double,6,1>& current_minus_prior,
 const PoseControlPhysicalLidarInformation& lidar,CoupledModeStepResult& out)
{
  const int d_eta=hns.freeDim(), dim_z=d_eta+layout.dimST();
  out.innovation_transform=Eigen::Matrix<double,6,6>::Identity()+lidar.Lambda*P_tail;
  Eigen::FullPivLU<Eigen::Matrix<double,6,6>> lu(out.innovation_transform);
  if(!lu.isInvertible()) return false;
  out.innovation_dual=lu.solve(lidar.b+lidar.Lambda*current_minus_prior);
  const Eigen::Matrix<double,6,1> post=P_tail*out.innovation_dual;
  out.physical_increment=post-current_minus_prior;
  out.physical_gain=P_tail*lu.solve(Eigen::Matrix<double,6,6>::Identity())*lidar.Lambda;
  out.physical_covariance=P_tail; out.physical_jacobian=J_tail_full;
  const auto jac=spline.jacobianAt(spline.t1());
  Eigen::VectorXd dc=Eigen::VectorXd::Zero(hns.rawDim());
  const int cp=layout.N-1, pc=layout.colPos(cp), rc=layout.colPhi(cp);
  if(pc<0 || rc<0 || std::abs(jac.b[3])<=1e-12) return false;
  dc.segment<3>(pc)=out.physical_increment.tail<3>()/jac.b[3];
  Eigen::FullPivLU<M3D> lr(spline.dThetaDcphi(jac,3,spline.t1()));
  if(!lr.isInvertible()) return false;
  dc.segment<3>(rc)=lr.solve(out.physical_increment.head<3>());
  out.delta_z=Eigen::VectorXd::Zero(dim_z);
  out.delta_z.head(d_eta)=hns.Z.transpose()*dc;
  const Eigen::MatrixXd Jz=J_tail_full.block(0,9,6,dim_z);
  out.lidar_information_z=Jz.transpose()*lidar.Lambda*Jz;
  out.lidar_rhs_z=Jz.transpose()*lidar.b;
  const Eigen::Matrix<double,6,6> Sinv=lu.solve(Eigen::Matrix<double,6,6>::Identity());
  const Eigen::MatrixXd reduction=sigma_prior*J_tail_full.transpose()*Sinv*lidar.Lambda*J_tail_full*sigma_prior;
  out.sigma_post=0.5*((sigma_prior-reduction)+(sigma_prior-reduction).transpose());
  return out.delta_z.allFinite() && out.sigma_post.allFinite();
}

bool realizeSingleTailPhysicalRpv(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, const Eigen::MatrixXd& Pz_cond,
    const Eigen::Matrix<double,9,1>& target, Eigen::VectorXd& delta_z,
    Eigen::Matrix<double,9,Eigen::Dynamic>& Jy,
    Eigen::Matrix<double,9,1>& realized, Eigen::Matrix<double,9,1>& error)
{
  const int d_eta = hns.freeDim(), dim_z = d_eta + layout.dimST();
  if (Pz_cond.rows() != dim_z || Pz_cond.cols() != dim_z || !Pz_cond.allFinite()) return false;
  const auto phys = evaluatePoseControlPhysicalSample(spline, layout, hns, spline.t1(), V3D::Zero());
  Jy = Eigen::Matrix<double,9,Eigen::Dynamic>::Zero(9, dim_z);
  Jy.block(0,0,3,d_eta)=phys.dtheta_deta; Jy.block(3,0,3,d_eta)=phys.dp_deta; Jy.block(6,0,3,d_eta)=phys.dv_deta;
  const int cp=layout.N-1, pos_col=layout.colPos(cp), phi_col=layout.colPhi(cp);
  if (pos_col < 0 || phi_col < 0) return false;
  Eigen::MatrixXd E=Eigen::MatrixXd::Zero(hns.rawDim(),6);
  E.block<3,3>(pos_col,0).setIdentity(); E.block<3,3>(phi_col,3).setIdentity();
  const Eigen::MatrixXd Pc=hns.Z*Pz_cond.topLeftCorner(d_eta,d_eta)*hns.Z.transpose();
  const auto jac=spline.jacobianAt(spline.t1());
  Eigen::Matrix<double,9,6> Jt=Eigen::Matrix<double,9,6>::Zero();
  Jt.block(0,3,3,3)=spline.dThetaDcphi(jac,3,spline.t1());
  Jt.block(3,0,3,3)=PoseControlSpline::dPosDcp(jac,3);
  Jt.block(6,0,3,3)=PoseControlSpline::dVelDcp(jac,3);
  const Eigen::MatrixXd Pt=E.transpose()*Pc*E;
  const Eigen::MatrixXd Py0=Jt*Pt*Jt.transpose();
  const Eigen::VectorXd dt=Pt*Jt.transpose()*generalPseudoInverse(0.5*(Py0+Py0.transpose()),1e-12)*target;
  delta_z.setZero(dim_z); delta_z.head(d_eta)=hns.Z.transpose()*E*dt;
  if (!delta_z.allFinite()) return false;
  realized=Jy*delta_z; error=realized-target;
  return realized.allFinite();
}
} // namespace livo_recon
