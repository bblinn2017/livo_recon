#include "livo_recon/lio/pose_control_lidar_factor.h"

namespace livo_recon
{

void addPoseControlLidarFactor(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const std::vector<PoseControlLidarObs>& obs,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlProcessFactorHeadBlock* head_block,
    double* out_E_lidar,
    std::vector<PoseControlLidarRecord>* out_records)
{
  if (out_records) out_records->reserve(out_records->size() + obs.size());
  const int dimZ = layout.dim();
  Eigen::VectorXd Jrow_z(dimZ);
  Eigen::Vector3d Jrow_head_theta, Jrow_head_p;   // 1x3 rows, transposed to 3x1 for storage

  for (const auto& o : obs)
  {
    const M3D Rt = spline.rotAt(o.t);
    const V3D pt = spline.posAt(o.t);
    const double r = o.normal.dot(Rt * o.q + pt) + o.d;
    const double w = 1.0 / std::max(o.sigma2, 1e-12);

    const auto jac = spline.jacobianAt(o.t);
    Jrow_z.setZero();
    bool touches_head_pos = false;

    // Rotation row (1x3): raw_body_point.cross(R(t)^T*normal) -- matched
    // convention, see header comment.
    const Eigen::RowVector3d dr_dtheta = o.q.cross(Rt.transpose() * o.normal).transpose();

    for (int k = 0; k < 4; ++k)
    {
      const int abs_k = jac.s + k;
      const Eigen::RowVector3d dr_dcp_pos = jac.b[k] * o.normal.transpose();
      const Eigen::RowVector3d dr_dcp_phi = dr_dtheta * spline.dThetaDcphi(jac, k, o.t);

      const int colp = layout.colPos(abs_k);
      if (colp >= 0) Jrow_z.segment<3>(colp) += dr_dcp_pos.transpose();
      const int colph = layout.colPhi(abs_k);
      if (colph >= 0) Jrow_z.segment<3>(colph) += dr_dcp_phi.transpose();

      if (abs_k < 3) touches_head_pos = true;   // LOCAL: only true when this
                                                 // point's own window touches
                                                 // cp[0..2]
    }

    if (head_block)
    {
      // Rotation's head sensitivity is GLOBAL (poseControlHeadRotJacobian
      // is nonzero for every t) -- always accumulate the theta0 row.
      Jrow_head_theta = (dr_dtheta * poseControlHeadRotJacobian(spline, o.t)).transpose();
      Jrow_head_p.setZero();
      Eigen::Vector3d Jrow_head_v0 = Eigen::Vector3d::Zero();
      if (touches_head_pos)
      {
        auto hs = poseControlHeadPosSensitivity(spline);
        M3D dp_dp0, dp_dv0, dv_dp0, dv_dv0;
        poseControlHeadPosJacobians(spline, hs, o.t, dp_dp0, dp_dv0, dv_dp0, dv_dv0);
        Jrow_head_p = (o.normal.transpose() * dp_dp0).transpose();
        Jrow_head_v0 = (o.normal.transpose() * dp_dv0).transpose();
      }

      Eigen::Matrix<double, 9, 1> Jhead;
      Jhead.segment<3>(0) = Jrow_head_theta;
      Jhead.segment<3>(3) = Jrow_head_p;
      Jhead.segment<3>(6) = Jrow_head_v0;

      head_block->A_hh += w * (Jhead * Jhead.transpose());
      if (head_block->A_hf.size() == 0) head_block->A_hf = Eigen::MatrixXd::Zero(9, dimZ);
      head_block->A_hf += w * (Jhead * Jrow_z.transpose());
    }

    A += w * (Jrow_z * Jrow_z.transpose());
    b += -w * r * Jrow_z;
    if (out_E_lidar) *out_E_lidar += 0.5 * w * r * r;
    if (out_records) {
      PoseControlLidarRecord rec;
      rec.Jrow_z = Jrow_z;
      rec.w = w;
      rec.sigma2 = o.sigma2;
      rec.plane_var_term = o.plane_var_term;
      rec.plane_id = o.plane_id;
      out_records->push_back(std::move(rec));
    }
  }
}

}  // namespace livo_recon
