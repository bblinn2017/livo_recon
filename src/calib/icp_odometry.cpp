#include "livo_recon/calib/icp_odometry.h"

#include "livo_recon/lio/lio_accumulator.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/algo/math.h"

namespace livo_recon
{
namespace calib
{

IcpReport icpRegister(const std::vector<PointXYZCov>& points, MapBackend& map, M3D& R, V3D& p, const IcpOptions& opt)
{
  IcpReport rep;
  for (int iter = 0; iter < opt.max_iterations; iter++) {
    std::vector<Residual> residuals;
    residuals.reserve(points.size());
    for (const auto& pt : points) {
      WorldPointCov pt_world;
      pt_world.point = R * pt.point + p;
      pt_world.sensor_cov = R * pt.sensor_cov * R.transpose();
      pt_world.pose_cov = M3D::Zero();
      // Deliberately false/zero: a Kalman-prior gate contribution would
      // reintroduce exactly the dependency this function exists to avoid
      // (see this file's header comment) -- there is no EKF prior here.
      pt_world.include_pose_cov_in_gate = false;
      pt_world.body_point = pt.point;
      pt_world.rot_transpose = R.transpose();
      Residual res{};
      if (!map.findPlaneResidual(pt_world, res)) continue;
      res.sigma_squared += res.plane_var_term;
      if (!std::isfinite(res.sigma_squared) || res.sigma_squared <= 0.0) continue;
      res.point_cross_normal = pt.point.cross(R.transpose() * res.normal);
      residuals.push_back(res);
    }
    rep.n_matched = static_cast<int>(residuals.size());
    if (rep.n_matched < opt.min_matches) { rep.degenerate = true; break; }

    EkfUpdate upd;
    accumulateLioResiduals(residuals, upd);

    // Zero-information-prior limit of EkfUpdate::applyMeanUpdate() (see
    // this file's header comment): no Kalman gain, no prior covariance,
    // just the Gauss-Newton normal-equation solve on the LiDAR residuals
    // alone. HtH/Htz use the same [rot(0:3); pos(3:6)] layout
    // accumulateLioResiduals() already builds (matches StateGroup::
    // idxR()=0/idxP()=3), so delta's two halves map directly onto a
    // right-multiplicative rotation update and an additive position one.
    Eigen::Matrix<double, 6, 6> H = upd.HtH;
    H.diagonal().array() += opt.damping;
    const Eigen::Matrix<double, 6, 1> delta = -H.ldlt().solve(upd.Htz);

    R = R * Exp(V3D(delta.segment<3>(0)));
    p = p + V3D(delta.segment<3>(3));
    rep.iterations = iter + 1;
    if (delta.segment<3>(0).norm() < opt.convergence_rot && delta.segment<3>(3).norm() < opt.convergence_pos) {
      rep.converged = true;
      break;
    }
  }
  return rep;
}

}  // namespace calib
}  // namespace livo_recon
