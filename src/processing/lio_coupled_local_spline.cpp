#include "livo_recon/processing/pose_control_coupled_modes.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/utils/algo/math.h"

namespace livo_recon
{

bool parseCoupledMode(const std::string& name, CoupledModeSelection& out)
{
  out.physical_rpv = name.rfind("physical_rpv_", 0) == 0;
  const std::string base = out.physical_rpv ? name.substr(13) : name;
  if (base == "local_spline") out.family = CoupledModeFamily::LocalSpline;
  else if (base == "single_tail" || base == "tail") out.family = CoupledModeFamily::SingleTail;
  else if (base == "direct_lidar_imu") out.family = CoupledModeFamily::DirectLidarImu;
  else if (base == "covariance_all_knots" || base == "all_knots") out.family = CoupledModeFamily::CovarianceAllKnots;
  else return false;
  return true;
}

const char* coupledModeFamilyName(CoupledModeFamily family)
{
  switch (family) {
    case CoupledModeFamily::LocalSpline: return "local_spline";
    case CoupledModeFamily::SingleTail: return "single_tail";
    case CoupledModeFamily::DirectLidarImu: return "direct_lidar_imu";
    case CoupledModeFamily::CovarianceAllKnots: return "covariance_all_knots";
  }
  return "unknown";
}

void assembleLocalSplineControl(const PoseControlSpline& spline,const PoseControlFreeLayout& layout,
 const std::vector<PoseControlLidarObs>& observations,Eigen::MatrixXd& A,Eigen::VectorXd& b,
 double& energy,std::vector<PoseControlLidarRecord>* records)
{
  // The local family owns the ordinary time-local B-spline LiDAR factor.
  addPoseControlLidarFactor(spline,layout,observations,A,b,nullptr,&energy,records);
}

bool realizeLocalSplinePhysicalRpv(
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
  Jy.block(0,0,3,d_eta) = phys.dtheta_deta;
  Jy.block(3,0,3,d_eta) = phys.dp_deta;
  Jy.block(6,0,3,d_eta) = phys.dv_deta;
  // Preserve the defining local-spline property: use only the ordinary
  // B-spline tail Jacobian's local support.  This is deliberately not the
  // dense covariance-mediated all-knots realization below.  The minimum-
  // norm solve gives the locally supported coefficient step whose endpoint
  // [theta,p,v] change best realizes the common physical target.
  const Eigen::MatrixXd JJt0 = Jy * Jy.transpose();
  const Eigen::MatrixXd JJt = 0.5 * (JJt0 + JJt0.transpose());
  delta_z = Jy.transpose() * generalPseudoInverse(JJt, 1e-12) * target;
  if (!delta_z.allFinite()) return false;
  realized = Jy * delta_z;
  error = realized - target;
  return realized.allFinite();
}

} // namespace livo_recon
