#include "livo_recon/processing/pose_control_coupled_modes.h"
#include "livo_recon/lio/pose_control_covariance.h"

namespace livo_recon
{
bool solveCovarianceAllKnotsControl(const PoseControlSpline& spline,const PoseControlFreeLayout& layout,
 const PoseControlHeadNullspace& hns,const V3D& gravity,const std::vector<Residual>& residuals,
 const Eigen::MatrixXd& sigma_prior,double q_pos,double q_rot,CoupledModeStepResult& out,
 CovarianceAllKnotsStatistics& stats,std::vector<CovarianceAllKnotsResidualTrace>* traces)
{
  const int d_eta=hns.freeDim(), dim_z=d_eta+layout.dimST(), dim_full=9+dim_z;
  out.sigma_post=sigma_prior;
  Eigen::VectorXd accumulated=Eigen::VectorXd::Zero(dim_full);
  out.lidar_information_z=Eigen::MatrixXd::Zero(dim_z,dim_z);
  out.lidar_rhs_z=Eigen::VectorXd::Zero(dim_z);
  for(std::size_t i=0;i<residuals.size();++i){
    const Residual& res=residuals[i];
    const auto physical=evaluatePoseControlPhysicalSample(spline,layout,hns,res.t,gravity);
    Eigen::MatrixXd J=Eigen::MatrixXd::Zero(6,dim_full);
    J.block(0,0,3,9)=physical.dtheta_dhead; J.block(0,9,3,d_eta)=physical.dtheta_deta;
    J.block(3,0,3,9)=physical.dp_dhead; J.block(3,9,3,d_eta)=physical.dp_deta;
    const Eigen::Matrix<double,6,1> h6=poseControlPhysicalLidarJacobianAtTime(spline,res);
    const Eigen::RowVectorXd H=h6.transpose()*J;
    const double R=std::max(res.sigma_squared,1e-12)+q_rot*h6.head<3>().squaredNorm()+q_pos*h6.tail<3>().squaredNorm();
    const double innovation=res.r+(H*accumulated)(0);
    const double S=(H*out.sigma_post*H.transpose())(0)+R;
    if(!std::isfinite(S) || S<=1e-15) continue;
    const double nis=innovation*innovation/S;
    const Eigen::VectorXd K=out.sigma_post*H.transpose()/S;
    if(traces) traces->push_back({i,J,H,h6,K,accumulated,R,innovation,S,nis});
    if(std::isfinite(nis)){stats.sum+=nis; stats.square_sum+=nis*nis; stats.maximum=std::max(stats.maximum,nis); ++stats.count; if(nis>3.841458820694124)++stats.above_3p84; if(nis>6.6348966010212145)++stats.above_6p63;}
    accumulated+=-K*innovation; accumulated.head(9).setZero();
    const Eigen::MatrixXd next=out.sigma_post-K*H*out.sigma_post;
    out.sigma_post=0.5*(next+next.transpose());
    if(!out.sigma_post.allFinite()) return false;
    const Eigen::RowVectorXd Hz=H.segment(9,dim_z);
    out.lidar_information_z+=(1.0/std::max(res.sigma_squared,1e-12))*Hz.transpose()*Hz;
    out.lidar_rhs_z+=-(1.0/std::max(res.sigma_squared,1e-12))*Hz.transpose()*res.r;
  }
  out.delta_z=accumulated.tail(dim_z);
  return out.delta_z.allFinite();
}

bool realizeCovarianceAllKnotsPhysicalRpv(const PoseControlSpline& s,const PoseControlFreeLayout& l,
 const PoseControlHeadNullspace& h,const Eigen::MatrixXd& P,const Eigen::Matrix<double,9,1>& y,
 Eigen::VectorXd& dz,Eigen::Matrix<double,9,Eigen::Dynamic>& J,Eigen::Matrix<double,9,1>& r,Eigen::Matrix<double,9,1>& e)
{
  const int d_eta=h.freeDim(), dim_z=d_eta+l.dimST();
  if(P.rows()!=dim_z || P.cols()!=dim_z || !P.allFinite()) return false;
  const auto physical=evaluatePoseControlPhysicalSample(s,l,h,s.t1(),V3D::Zero());
  J=Eigen::Matrix<double,9,Eigen::Dynamic>::Zero(9,dim_z);
  J.block(0,0,3,d_eta)=physical.dtheta_deta;
  J.block(3,0,3,d_eta)=physical.dp_deta;
  J.block(6,0,3,d_eta)=physical.dv_deta;
  const Eigen::MatrixXd Psym=0.5*(P+P.transpose());
  const Eigen::MatrixXd Py0=J*Psym*J.transpose();
  const Eigen::MatrixXd Py=0.5*(Py0+Py0.transpose());
  // Dense all-free-knot conditional-mean realization.  The dense support is
  // supplied by P_z,y, not by pretending the cubic B-spline basis is dense.
  dz=Psym*J.transpose()*generalPseudoInverse(Py,1e-12)*y;
  if(!dz.allFinite()) return false;
  r=J*dz; e=r-y;
  return r.allFinite();
}
} // namespace livo_recon
