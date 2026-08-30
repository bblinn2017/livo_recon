#pragma once

#include <Eigen/Core>
#include <math.h>

namespace livo_recon
{

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

#define G_m_s2 (9.81)
#define LIDAR_HZ (1. / 240000.)

using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;
using V4D = Eigen::Vector4d;
using M4D = Eigen::Matrix4d;
using M2D = Eigen::Matrix2d;

const M3D Eye3d = M3D::Identity();

inline M3D Exp(const V3D &ang_vel, const double &dt)
{
  double ang_vel_norm = ang_vel.norm();
  if (ang_vel_norm > 0.0000001)
  {
    V3D r_axis = ang_vel / ang_vel_norm;
    M3D K;

    K << SKEW_SYM_MATRX(r_axis);

    double r_ang = ang_vel_norm * dt;

    /// Roderigous Tranformation
    return Eye3d + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  }
  else { return Eye3d; }
}

inline M3D Exp(const V3D &ang)
{
  double ang_norm = ang.norm();
  M3D Eye3 = M3D::Identity();
  if (ang_norm > 0.0000001)
  {
    V3D r_axis = ang / ang_norm;
    M3D K;
    K << SKEW_SYM_MATRX(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  }
  else { return Eye3; }
}

// SO3 right Jacobian: Jr(φ) such that Exp(φ + δ) ≈ Exp(φ) * Exp(Jr(φ) * δ)
// Jr(φ) = I - (1-cos|φ|)/|φ|² * [φ]× + (|φ|-sin|φ|)/|φ|³ * [φ]×²
inline M3D Jr(const V3D& phi)
{
  const double n = phi.norm();
  if (n < 1e-7) return M3D::Identity();
  M3D K;
  K << SKEW_SYM_MATRX(phi);
  return M3D::Identity()
       - (1.0 - std::cos(n)) / (n * n) * K
       + (n - std::sin(n))   / (n * n * n) * K * K;
}

// Logarithm of a rotation matrix (SO(3) log map): returns the axis-angle
// vector phi such that Exp(phi) == R (see Exp() above) -- the inverse of
// Exp(). Mirrors FAST-LIVO2's own utils/so3_math.h::Log() exactly. Added
// for StateGroup::boxminusFromPropagat()'s manifold-consistent (SO(3)-
// aware) rotation deviation, needed by the fixed-prior IEKF update (see
// ekf.h) -- this file previously had no log-map helper at all.
inline V3D Log(const M3D& R)
{
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  V3D K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

}
