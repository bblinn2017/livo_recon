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

Eigen::Matrix3d headBoundaryM(const PoseControlSpline& spline)
{
  // Basis (b,db,ddb) at u=0, du/dt->inv_delta scaling already applied.
  const double invd = 1.0 / spline.delta();
  const double invd2 = invd * invd;
  Eigen::Matrix3d M;
  M << 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0,      // value row
      -0.5 * invd, 0.0, 0.5 * invd,          // rate row
      invd2, -2.0 * invd2, invd2;            // 2nd-derivative row
  return M;
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

void buildHeadConstraintRows(const PoseControlSpline& spline,
                              const V3D& p0, const V3D& v0, const V3D& omega0,
                              Eigen::MatrixXd& C, Eigen::VectorXd& d)
{
  const int N = spline.N();
  const int dim = 6 * N;
  const double inv_delta = 1.0 / spline.delta();
  C = Eigen::MatrixXd::Zero(12, dim);
  d = Eigen::VectorXd::Zero(12);

  auto idx_p = [&](int k, int axis) { return 3 * k + axis; };
  auto idx_phi = [&](int k, int axis) { return 3 * N + 3 * k + axis; };

  const V3D p_cur   = spline.posAt(spline.t0());
  const V3D v_cur    = spline.velAt(spline.t0());
  const V3D phi_cur  = spline.phiAt(spline.t0());
  const V3D wdot_cur = spline.phiDotAt(spline.t0());

  for (int a = 0; a < 3; ++a)
  {
    // position value: (1/6,4/6,1/6,0) . cp_p[0..3]
    int r0 = a;
    C(r0, idx_p(0, a)) = 1.0 / 6.0;
    C(r0, idx_p(1, a)) = 4.0 / 6.0;
    C(r0, idx_p(2, a)) = 1.0 / 6.0;
    d(r0) = p0(a) - p_cur(a);

    // position rate: inv_delta*(-1/2,0,1/2,0) . cp_p[0..3]
    int r1 = 3 + a;
    C(r1, idx_p(0, a)) = -0.5 * inv_delta;
    C(r1, idx_p(2, a)) = 0.5 * inv_delta;
    d(r1) = v0(a) - v_cur(a);

    // rotation value: phi(t0) target 0 (R_anchor convention -- see header)
    int r2 = 6 + a;
    C(r2, idx_phi(0, a)) = 1.0 / 6.0;
    C(r2, idx_phi(1, a)) = 4.0 / 6.0;
    C(r2, idx_phi(2, a)) = 1.0 / 6.0;
    d(r2) = 0.0 - phi_cur(a);

    // rotation rate: phidot(t0) target omega0 (Jr(0)=I)
    int r3 = 9 + a;
    C(r3, idx_phi(0, a)) = -0.5 * inv_delta;
    C(r3, idx_phi(2, a)) = 0.5 * inv_delta;
    d(r3) = omega0(a) - wdot_cur(a);
  }
}

PoseControlHeadPosSensitivity poseControlHeadPosSensitivity(const PoseControlSpline& spline)
{
  PoseControlHeadPosSensitivity hs;
  hs.Minv = headBoundaryM(spline).inverse();
  return hs;
}

void poseControlHeadPosJacobians(const PoseControlSpline& spline,
                                  const PoseControlHeadPosSensitivity& hs, double t,
                                  M3D& dp_dp0, M3D& dp_dv0, M3D& dv_dp0, M3D& dv_dv0)
{
  dp_dp0 = M3D::Zero(); dp_dv0 = M3D::Zero(); dv_dp0 = M3D::Zero(); dv_dv0 = M3D::Zero();
  int s; double u; spline.locate(t, s, u);
  Eigen::Vector4d b, db, ddb; poseControlBasisU(u, b, db, ddb);
  const double inv_delta = 1.0 / spline.delta();
  for (int k = 0; k < 4; ++k)
  {
    const int abs_k = s + k;
    if (abs_k >= 3) continue;   // only cp[0..2] depend on p0/v0 at all
    dp_dp0 += (b[k] * hs.Minv(abs_k, 0)) * M3D::Identity();
    dp_dv0 += (b[k] * hs.Minv(abs_k, 1)) * M3D::Identity();
    dv_dp0 += (db[k] * inv_delta * hs.Minv(abs_k, 0)) * M3D::Identity();
    dv_dv0 += (db[k] * inv_delta * hs.Minv(abs_k, 1)) * M3D::Identity();
  }
}

M3D poseControlHeadRotJacobian(const PoseControlSpline& spline, double t)
{
  // Exact (not leading-order): R(t)^T * R_anchor -- see header comment.
  return spline.rotAt(t).transpose() * spline.R_anchor;
}

bool solvePoseControlKkt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                          const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
                          Eigen::VectorXd& delta_c)
{
  const int n = static_cast<int>(A.rows());
  const int k = static_cast<int>(C.rows());
  Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + k, n + k);
  KKT.topLeftCorner(n, n) = A;
  KKT.topRightCorner(n, k) = C.transpose();
  KKT.bottomLeftCorner(k, n) = C;

  Eigen::VectorXd rhs(n + k);
  rhs.head(n) = b;
  rhs.tail(k) = d;

  Eigen::LDLT<Eigen::MatrixXd> ldlt(KKT);
  if (ldlt.info() != Eigen::Success) return false;
  const Eigen::VectorXd sol = ldlt.solve(rhs);
  if (!sol.allFinite()) return false;
  delta_c = sol.head(n);
  return true;
}

void solveHeadControlPoints(const PoseControlSpline& spline,
                             const V3D& p0, const V3D& v0, const V3D& a0,
                             const V3D& omega0, const V3D& alpha0,
                             V3D cp_p_head[3], V3D cp_phi_head[3])
{
  const Eigen::Matrix3d Minv = headBoundaryM(spline).inverse();

  for (int a = 0; a < 3; ++a)
  {
    Eigen::Vector3d rhs_p(p0(a), v0(a), a0(a));
    Eigen::Vector3d sol_p = Minv * rhs_p;
    cp_p_head[0](a) = sol_p(0); cp_p_head[1](a) = sol_p(1); cp_p_head[2](a) = sol_p(2);

    // phi0 target is 0 -- caller sets spline.R_anchor = R0 so this is exact.
    Eigen::Vector3d rhs_phi(0.0, omega0(a), alpha0(a));
    Eigen::Vector3d sol_phi = Minv * rhs_phi;
    cp_phi_head[0](a) = sol_phi(0); cp_phi_head[1](a) = sol_phi(1); cp_phi_head[2](a) = sol_phi(2);
  }
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
