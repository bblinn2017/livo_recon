#include "livo_recon/lio/pose_control_spline.h"

namespace livo_recon
{

namespace
{
inline M3D skew3(const V3D& v)
{
  M3D S;
  S <<     0, -v.z(),  v.y(),
        v.z(),      0, -v.x(),
       -v.y(),  v.x(),      0;
  return S;
}
}  // namespace

void PoseControlSpline::init(int N, double t0, double t1)
{
  N_ = N;
  n_seg_ = std::max(1, N_ - 3);
  t0_ = t0;
  t1_ = t1;
  delta_ = (t1_ - t0_) / static_cast<double>(n_seg_);
  inv_delta_ = (delta_ > 1e-12) ? 1.0 / delta_ : 0.0;
  cp_p.setZero(3, N_);
  cp_phi.setZero(3, N_);
  R_anchor = M3D::Identity();
}

void PoseControlSpline::locate(double t, int& s, double& u) const
{
  const double tc = std::min(std::max(t, t0_), t1_);
  double x = (tc - t0_) * inv_delta_;
  int si = static_cast<int>(std::floor(x));
  if (si >= n_seg_) { si = n_seg_ - 1; x = static_cast<double>(n_seg_); }
  if (si < 0)       { si = 0;          x = 0.0; }
  s = si;
  u = x - static_cast<double>(si);
}

PoseControlJac PoseControlSpline::jacobianAt(double t) const
{
  PoseControlJac j;
  double u;
  locate(t, j.s, u);
  poseControlBasisU(u, j.b, j.db, j.ddb);
  j.inv_delta = inv_delta_;
  j.inv_delta2 = inv_delta_ * inv_delta_;
  return j;
}

V3D PoseControlSpline::posAt(double t) const
{
  int s; double u; locate(t, s, u);
  Eigen::Vector4d b, db, ddb; poseControlBasisU(u, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += b[i] * cp_p.col(s + i);
  return r;
}

V3D PoseControlSpline::velAt(double t) const
{
  int s; double u; locate(t, s, u);
  Eigen::Vector4d b, db, ddb; poseControlBasisU(u, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += db[i] * cp_p.col(s + i);
  return r * inv_delta_;
}

V3D PoseControlSpline::accAt(double t) const
{
  int s; double u; locate(t, s, u);
  Eigen::Vector4d b, db, ddb; poseControlBasisU(u, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += ddb[i] * cp_p.col(s + i);
  return r * (inv_delta_ * inv_delta_);
}

V3D PoseControlSpline::phiAt(double t) const
{
  int s; double u; locate(t, s, u);
  Eigen::Vector4d b, db, ddb; poseControlBasisU(u, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += b[i] * cp_phi.col(s + i);
  return r;
}

V3D PoseControlSpline::phiDotAt(double t) const
{
  int s; double u; locate(t, s, u);
  Eigen::Vector4d b, db, ddb; poseControlBasisU(u, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += db[i] * cp_phi.col(s + i);
  return r * inv_delta_;
}

M3D PoseControlSpline::rotAt(double t) const
{
  return R_anchor * Exp(phiAt(t));
}

V3D PoseControlSpline::omegaBodyAt(double t) const
{
  return Jr(phiAt(t)) * phiDotAt(t);
}

void PoseControlSpline::poseAt(double t, M3D& R, V3D& p) const
{
  R = rotAt(t);
  p = posAt(t);
}

M3D PoseControlSpline::dOmegaDcphi(const PoseControlJac& j, int k, double t) const
{
  // omega(t) = Jr(phi(t)) * phidot(t). Leading-order (exact at phi=0,
  // accurate to the ~15 deg per-scan chords this codebase's own
  // CHART_MAX_PHI_RAD=1.0 rad regime assumes -- see this class's header
  // comment):
  //   d(Jr(phi)v)/dphi[dphi] ~= 0.5*[v]_x*dphi   (v = phidot(t) held fixed)
  //   d(phidot term)/dc_phi[k] = Jr(phi(t)) * bdot_k(t)
  const V3D wdot = phiDotAt(t);
  return j.db[k] * j.inv_delta * Jr(phiAt(t)) + 0.5 * j.b[k] * skew3(wdot);
}

}  // namespace livo_recon
