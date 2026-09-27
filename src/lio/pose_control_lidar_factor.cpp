#include "livo_recon/lio/pose_control_lidar_factor.h"
#include "livo_recon/utils/algo/omp_utils.h"

namespace livo_recon
{

PoseControlPhysicalLidarInformation buildPoseControlPhysicalLidarInformation(
    const std::vector<Residual>& residuals)
{
  PoseControlPhysicalLidarInformation out;
  for (const Residual& res : residuals)
  {
    const double w = 1.0 / std::max(res.sigma_squared, 1e-12);
    Eigen::Matrix<double, 6, 1> J;
    J.head<3>() = res.point_cross_normal;
    J.tail<3>() = res.normal;
    out.Lambda.noalias() += w * (J * J.transpose());
    out.b.noalias() += -w * res.r * J;
    out.energy += 0.5 * w * res.r * res.r;
  }
  const Eigen::Matrix<double, 6, 6> S = 0.5 * (out.Lambda + out.Lambda.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(S);
  if (es.info() == Eigen::Success)
  {
    out.lambda_min = es.eigenvalues()(0);
    out.lambda_max = es.eigenvalues()(5);
    const double threshold = 1e-12 * std::max(std::abs(out.lambda_max), 1e-300);
    for (int i = 0; i < 6; ++i)
      if (es.eigenvalues()(i) > threshold) ++out.effective_rank;
  }
  return out;
}

Eigen::Matrix<double, 6, 1> poseControlPhysicalLidarJacobianAtTime(
    const PoseControlSpline& spline, const Residual& residual)
{
  const M3D R_t = spline.rotAt(residual.t);
  Eigen::Matrix<double, 6, 1> J;
  J.head<3>() = residual.raw_body_point.cross(R_t.transpose() * residual.normal);
  J.tail<3>() = residual.normal;
  return J;
}

// Per-thread accumulator for the parallel LiDAR-residual loop below -- item
// 30-35 of the implementation spec ("parallel loop -> thread-local
// accumulator -> merge... no per-residual/per-sample locks"). One instance
// per OMP thread, filled independently with zero cross-thread writes during
// the parallel region, then summed together (a linear reduction, exact --
// not an approximation of the serial result) after the implicit barrier.
struct PoseControlLidarThreadAccum
{
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  double E_lidar = 0.0;
  Eigen::Matrix<double, 9, 9> A_hh = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::MatrixXd A_hf;
  std::vector<PoseControlLidarRecord> records;
};

void addPoseControlLidarFactor(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const std::vector<PoseControlLidarObs>& obs,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlPriorHeadBlock* head_block,
    double* out_E_lidar,
    std::vector<PoseControlLidarRecord>* out_records)
{
  const int dimZ = layout.dim();
  const int n = static_cast<int>(obs.size());
  if (out_records) out_records->reserve(out_records->size() + obs.size());

  const int threads = std::max(1, std::min(cappedOmpThreads(), std::max(1, n)));
  std::vector<PoseControlLidarThreadAccum> acc(threads);
  for (auto& t : acc) {
    t.A = Eigen::MatrixXd::Zero(dimZ, dimZ);
    t.b = Eigen::VectorXd::Zero(dimZ);
    if (head_block) t.A_hf = Eigen::MatrixXd::Zero(9, dimZ);
    if (out_records) t.records.reserve((n + threads - 1) / threads);
  }

  #pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num();
    PoseControlLidarThreadAccum& acc_t = acc[tid];
    Eigen::VectorXd Jrow_z(dimZ);

    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i)
    {
      const PoseControlLidarObs& o = obs[i];
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
        const Eigen::Vector3d Jrow_head_theta = (dr_dtheta * poseControlHeadRotJacobian(spline, o.t)).transpose();
        Eigen::Vector3d Jrow_head_p = Eigen::Vector3d::Zero();
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

        acc_t.A_hh += w * (Jhead * Jhead.transpose());
        acc_t.A_hf += w * (Jhead * Jrow_z.transpose());
      }

      acc_t.A += w * (Jrow_z * Jrow_z.transpose());
      acc_t.b += -w * r * Jrow_z;
      if (out_E_lidar) acc_t.E_lidar += 0.5 * w * r * r;
      if (out_records) {
        PoseControlLidarRecord rec;
        rec.Jrow_z = Jrow_z;
        rec.w = w;
        rec.r = r;
        rec.sigma2 = o.sigma2;
        rec.plane_var_term = o.plane_var_term;
        rec.plane_id = o.plane_id;
        acc_t.records.push_back(std::move(rec));
      }
    }
  }

  // Merge (single-threaded, exact linear reduction -- matches the serial
  // sum bit-for-bit up to floating-point summation-order noise, the same
  // caveat any parallel reduction over floating point carries).
  for (auto& t : acc) {
    A += t.A;
    b += t.b;
    if (out_E_lidar) *out_E_lidar += t.E_lidar;
    if (head_block) {
      head_block->A_hh += t.A_hh;
      if (head_block->A_hf.size() == 0) head_block->A_hf = Eigen::MatrixXd::Zero(9, dimZ);
      head_block->A_hf += t.A_hf;
    }
    if (out_records) {
      for (auto& r : t.records) out_records->push_back(std::move(r));
    }
  }
}

}  // namespace livo_recon
