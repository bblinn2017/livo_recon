#pragma once

#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"   // PoseControlPriorHeadBlock

// ============================================================================
// LiDAR point-to-plane factor against PoseControlSpline, built directly in
// the REDUCED [c_free;sT] layout (spec item 6). Analytic throughout -- the
// rotation Jacobian reuses the SAME chain-through-Jr(phi) convention this
// codebase's existing point_time residual already uses (matched, not
// re-derived -- see pose_spline_system.h's own header comment, which
// documents this convention was independently verified against
// hcol_reldiff's raw_body_point.cross(R(t)^T*normal) row).
//
//     r_i = n_i^T ( R(t_i) q_i + p(t_i) ) + d_i
//     dr_i/dc_p[k]   = b_k(t_i) * n_i^T                          (1x3)
//     dr_i/dtheta_i  = q_i^T . [R(t_i)^T n_i]_x   (raw_body_point.cross(...))
//     dr_i/dc_phi[k] = dr_i/dtheta_i * Jr(phi(t_i)) * b_k(t_i)   (1x3)
//
// sT has NO LiDAR Jacobian column (bg/ba/g do not affect the point-to-plane
// residual at all) -- matches decoupled LIO's own zero-padding of its H
// matrix for bias/gravity (confirmed via research this session).
// ============================================================================

namespace livo_recon
{

struct PoseControlLidarObs
{
  double t = 0.0;
  V3D    q = V3D::Zero();        // raw body-frame point (before deskew)
  V3D    normal = V3D::Zero();
  double d = 0.0;                // plane offset
  double sigma2 = 1.0;
  // process-prior-REFORMULATION phase, items 24/25: shared-uncertainty
  // correlation source, copied straight from Residual::plane_id/
  // plane_var_term -- see pose_control_lidar_correlation.h.
  const void* plane_id = nullptr;
  double plane_var_term = 0.0;
};

// process-prior-REFORMULATION phase, item 21: one point's reduced Jacobian
// row (already projected into z-space), retained ONLY when
// addPoseControlLidarFactor's out_records is non-null (default nullptr ->
// zero extra cost, current behavior byte-for-byte unchanged).
struct PoseControlLidarRecord
{
  Eigen::VectorXd Jrow_z;
  double w = 0.0;            // 1/sigma2
  double r = 0.0;            // raw (unweighted) point-to-plane residual
  double sigma2 = 1.0;
  double plane_var_term = 0.0;
  const void* plane_id = nullptr;
};

// Adds every observation's contribution to A/b (layout.dim() square/long).
// Head-touching columns (k<3) are routed to head_block (nullptr to skip).
void addPoseControlLidarFactor(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const std::vector<PoseControlLidarObs>& obs,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlPriorHeadBlock* head_block,
    double* out_E_lidar = nullptr,
    std::vector<PoseControlLidarRecord>* out_records = nullptr);

}  // namespace livo_recon
