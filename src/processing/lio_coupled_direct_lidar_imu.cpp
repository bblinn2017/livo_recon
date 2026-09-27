#include "livo_recon/processing/pose_control_coupled_modes.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/algo/math.h"

namespace livo_recon
{
bool solveDirectLidarImuControl(const PoseControlSpline& spline,const PoseControlFreeLayout& layout,
 const PoseControlHeadNullspace& hns,const StateGroup& state,const Eigen::MatrixXd& J_tail_full,
 const PoseControlPhysicalLidarInformation& lidar,CoupledModeStepResult& out)
{
  const int d_eta=hns.freeDim(), dim_z=d_eta+layout.dimST();
  const Eigen::MatrixXd& P=state.cov();
  if(P.rows()!=state.dimState() || P.cols()!=state.dimState()) return false;
  Eigen::MatrixXd H=Eigen::MatrixXd::Zero(6,state.dimState());
  H.block<3,3>(0,StateGroup::idxR()).setIdentity(); H.block<3,3>(3,StateGroup::idxP()).setIdentity();
  out.physical_jacobian=H;
  out.physical_covariance=0.5*(H*P*H.transpose()+(H*P*H.transpose()).transpose());
  out.innovation_transform=Eigen::Matrix<double,6,6>::Identity()+lidar.Lambda*out.physical_covariance;
  Eigen::FullPivLU<Eigen::Matrix<double,6,6>> lu(out.innovation_transform);
  if(!lu.isInvertible()) return false;
  out.physical_increment=out.physical_covariance*lu.solve(lidar.b);
  const Eigen::Matrix<double,6,6> Sinv=lu.solve(Eigen::Matrix<double,6,6>::Identity());
  out.physical_gain=P*H.transpose()*Sinv*lidar.Lambda;
  out.full_state_delta=Eigen::VectorXd::Zero(state.dimState());
  out.full_state_delta.segment<3>(StateGroup::idxR())=out.physical_increment.head<3>();
  out.full_state_delta.segment<3>(StateGroup::idxP())=out.physical_increment.tail<3>();
  out.full_state_covariance=P-out.physical_gain*H*P;
  out.full_state_covariance=0.5*(out.full_state_covariance+out.full_state_covariance.transpose());
  const auto jac=spline.jacobianAt(spline.t1());
  Eigen::VectorXd dc=Eigen::VectorXd::Zero(hns.rawDim());
  const int cp=layout.N-1,pc=layout.colPos(cp),rc=layout.colPhi(cp);
  if(pc<0 || rc<0 || std::abs(jac.b[3])<=1e-12) return false;
  dc.segment<3>(pc)=out.physical_increment.tail<3>()/jac.b[3];
  Eigen::FullPivLU<M3D> lr(spline.dThetaDcphi(jac,3,spline.t1())); if(!lr.isInvertible())return false;
  dc.segment<3>(rc)=lr.solve(out.physical_increment.head<3>());
  out.delta_z=Eigen::VectorXd::Zero(dim_z); out.delta_z.head(d_eta)=hns.Z.transpose()*dc;
  const Eigen::MatrixXd Jz=J_tail_full.block(0,9,6,dim_z);
  out.lidar_information_z=Jz.transpose()*lidar.Lambda*Jz; out.lidar_rhs_z=Jz.transpose()*lidar.b;
  return out.delta_z.allFinite() && out.full_state_covariance.allFinite();
}

bool solveDirectLidarImuPhysicalRpv(const StateGroup& current, const StateGroup& propagated,
    const Eigen::MatrixXd& covariance, const PoseControlPhysicalLidarInformation& lidar,
    PhysicalRpvUpdate& out)
{
  const int ir=StateGroup::idxR(), ip=StateGroup::idxP(), iv=StateGroup::idxV();
  if (covariance.rows()<iv+3 || covariance.cols()<iv+3 || !covariance.allFinite()) return false;
  const int idx[3]={ir,ip,iv};
  for(int r=0;r<3;++r) for(int c=0;c<3;++c)
    out.P_prior.block<3,3>(3*r,3*c)=covariance.block<3,3>(idx[r],idx[c]);
  out.P_prior=0.5*(out.P_prior+out.P_prior.transpose());
  Eigen::LDLT<Eigen::Matrix<double,9,9>> lp(out.P_prior);
  if(lp.info()!=Eigen::Success) return false;
  out.P_prior_inverse=lp.solve(Eigen::Matrix<double,9,9>::Identity());
  out.H_full.setZero(); out.H_full.block<6,6>(0,0)=0.5*(lidar.Lambda+lidar.Lambda.transpose());
  out.A=0.5*(out.H_full+out.H_full.transpose())+out.P_prior_inverse;
  Eigen::LDLT<Eigen::Matrix<double,9,9>> la(out.A);
  if(la.info()!=Eigen::Success) return false;
  const auto inverse=la.solve(Eigen::Matrix<double,9,9>::Identity());
  out.K1=0.5*(inverse+inverse.transpose()); out.K1_pose=out.K1.leftCols<6>();
  out.G=out.K1_pose*out.H_full.block<6,6>(0,0);
  // This is explicitly propagated-minus-current.  It is not an absolute
  // gravity-aligned attitude, and diagnostics must log both quantities.
  out.vec.head<3>()=Log(current.rot().transpose()*propagated.rot());
  out.vec.segment<3>(3)=propagated.pos()-current.pos();
  out.vec.tail<3>()=propagated.vel()-current.vel();
  out.measurement_term=out.K1_pose*lidar.b;
  out.prior_term=out.vec-out.G*out.vec.head<6>();
  out.solution=out.measurement_term+out.prior_term;
  Eigen::LDLT<Eigen::Matrix<double,6,6>> ll(0.5*(lidar.Lambda+lidar.Lambda.transpose()));
  if(ll.info()==Eigen::Success) out.desired_pose=ll.solve(lidar.b);
  out.P_post=solveCovarianceFromA(out.A);
  if (!out.solution.allFinite() || !out.P_post.allFinite()) return false;

  // Lift the joint R/P/V posterior back into the complete EKF state.  This
  // is the defining difference from the spline-only families: bias, gravity,
  // and any other correlated state component receive their conditional mean
  // and covariance change instead of silently discarding cross covariance.
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(9, covariance.rows());
  E.block<3,3>(0,ir).setIdentity();
  E.block<3,3>(3,ip).setIdentity();
  E.block<3,3>(6,iv).setIdentity();
  const Eigen::MatrixXd cross = covariance * E.transpose();
  out.state_delta = cross * out.P_prior_inverse * out.solution;
  const Eigen::MatrixXd lift = cross * out.P_prior_inverse;
  out.P_full_post = covariance + lift * (out.P_post - out.P_prior) * lift.transpose();
  out.P_full_post = 0.5 * (out.P_full_post + out.P_full_post.transpose());
  return out.state_delta.allFinite() && out.P_full_post.allFinite();
}

bool realizeDirectLidarImuPhysicalRpv(const PoseControlSpline& s,const PoseControlFreeLayout& l,
 const PoseControlHeadNullspace& h,const Eigen::MatrixXd& P,const Eigen::Matrix<double,9,1>& y,
 Eigen::VectorXd& dz,Eigen::Matrix<double,9,Eigen::Dynamic>& J,Eigen::Matrix<double,9,1>& r,Eigen::Matrix<double,9,1>& e)
{
  // Direct-IMU first updates the complete EKF state above.  Reconcile its
  // physical R/P/V result with the spline using the IMU-derived coefficient
  // covariance; unlike local_spline this may use correlations with every
  // free knot, but state/bias mutation remains owned by this family.
  return realizeCovarianceAllKnotsPhysicalRpv(s,l,h,P,y,dz,J,r,e);
}
} // namespace livo_recon
