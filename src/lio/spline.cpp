#include "livo_recon/lio/spline.h"

#include <algorithm>
#include <cmath>

namespace livo_recon
{

namespace
{

// Uniform cubic B-spline basis on u in [0,1) and its first two derivatives
// WITH RESPECT TO u (the caller scales by inv_delta powers for d/dt).
// Standard de Boor uniform basis; b sums to 1, db and ddb each sum to 0 --
// which is what makes the derivatives independent of a common offset of the
// control points, i.e. translation-invariant, as they must be.
inline void basisU(double u, Eigen::Vector4d& b, Eigen::Vector4d& db, Eigen::Vector4d& ddb)
{
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double om = 1.0 - u;

  b[0] = om * om * om / 6.0;
  b[1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
  b[2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
  b[3] = u3 / 6.0;

  db[0] = -0.5 * om * om;
  db[1] = 0.5 * (3.0 * u2 - 4.0 * u);
  db[2] = 0.5 * (-3.0 * u2 + 2.0 * u + 1.0);
  db[3] = 0.5 * u2;

  ddb[0] = om;
  ddb[1] = 3.0 * u - 2.0;
  ddb[2] = -3.0 * u + 1.0;
  ddb[3] = u;
}

}  // namespace

void ScanSpline::basisAt(double t, int& first_cp, Eigen::Vector4d& b,
                         Eigen::Vector4d& db, Eigen::Vector4d& ddb) const
{
  const double tc = std::min(std::max(t, t0_), t1_);
  double x = (tc - t0_) * inv_delta_;
  int s = static_cast<int>(std::floor(x));
  // The right endpoint lands exactly on the last knot; put it in the last
  // segment at u = 1 rather than one segment past the end.
  if (s >= n_seg_) { s = n_seg_ - 1; x = static_cast<double>(n_seg_); }
  if (s < 0)       { s = 0;          x = 0.0; }
  const double u = x - static_cast<double>(s);
  basisU(u, b, db, ddb);
  first_cp = s;
}

bool ScanSpline::fit(const std::vector<Pose6D>& poses, double t0, double t1,
                     const SplineOptions& opts)
{
  valid_ = false;
  if (poses.size() < 2) return false;
  if (!(t1 > t0)) return false;

  // ---- choose n_cp -------------------------------------------------------
  int n_cp = opts.n_control_points;
  if (opts.control_point_hz > 0.0)
    n_cp = static_cast<int>(std::lround(opts.control_point_hz * (t1 - t0))) + 3;
  n_cp = std::max(4, std::min(n_cp, opts.n_control_points_max));

  // A cubic fit needs strictly more independent samples than unknowns; with
  // n_samples <= n_cp the normal matrix is singular and only the
  // regularizer would be holding the answer up. Shrink rather than fail, so
  // a sparse scan degrades to a coarser spline instead of falling back.
  const int n_samples = static_cast<int>(poses.size());
  if (n_samples < 5) return false;
  n_cp = std::min(n_cp, n_samples - 1);
  if (n_cp < 4) return false;

  n_cp_  = n_cp;
  n_seg_ = n_cp_ - 3;
  t0_ = t0; t1_ = t1;
  delta_ = (t1_ - t0_) / static_cast<double>(n_seg_);
  if (!(delta_ > 1e-9)) return false;
  inv_delta_ = 1.0 / delta_;

  // ---- rotation anchor ---------------------------------------------------
  // Anchor at the MIDDLE pose, not an endpoint: it halves the maximum
  // |phi| the small-angle tangent parameterisation has to carry, which is
  // the assumption the whole rotation fit rests on.
  R_anchor_ = poses[poses.size() / 2].rot;
  const M3D R_anchor_T = R_anchor_.transpose();

  // ---- assemble the normal equations -------------------------------------
  // The design matrix is shared between position and rotation (same basis,
  // same sample times), so build A^T A once and solve with two right-hand
  // sides. A is n_samples x n_cp with only 4 nonzeros per row.
  Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(n_cp_, n_cp_);
  Eigen::MatrixXd Atb_p = Eigen::MatrixXd::Zero(n_cp_, 3);
  Eigen::MatrixXd Atb_r = Eigen::MatrixXd::Zero(n_cp_, 3);

  Eigen::Vector4d b, db, ddb;
  int first_cp = 0;

  std::vector<Eigen::Vector4d> rows;   rows.reserve(n_samples);
  std::vector<int>             row_cp; row_cp.reserve(n_samples);
  std::vector<V3D>             tgt_p;  tgt_p.reserve(n_samples);
  std::vector<V3D>             tgt_r;  tgt_r.reserve(n_samples);

  for (const auto& ps : poses)
  {
    if (ps.t < t0_ - 1e-9 || ps.t > t1_ + 1e-9) continue;
    basisAt(ps.t, first_cp, b, db, ddb);
    const V3D phi = Log(R_anchor_T * ps.rot);

    rows.push_back(b);
    row_cp.push_back(first_cp);
    tgt_p.push_back(ps.pos);
    tgt_r.push_back(phi);

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
        AtA(first_cp + i, first_cp + j) += b[i] * b[j];
      Atb_p.row(first_cp + i) += b[i] * ps.pos.transpose();
      Atb_r.row(first_cp + i) += b[i] * phi.transpose();
    }
  }
  if (static_cast<int>(rows.size()) < n_cp_ + 1) return false;

  // ---- second-difference regularizer -------------------------------------
  // Penalises curvature IN THE CONTROL POLYGON, not in the trajectory, so it
  // damps ringing without pulling the fitted acceleration toward zero -- the
  // fitted acceleration is exactly what the Q estimate reads, and a
  // regularizer that flattened it would manufacture a low noise estimate.
  const double data_trace = AtA.trace();
  double reg_trace = 0.0;
  if (opts.fit_regularization > 0.0 && n_cp_ >= 3)
  {
    const double w = opts.fit_regularization * data_trace / static_cast<double>(n_cp_);
    for (int i = 0; i + 2 < n_cp_; ++i)
    {
      const int idx[3] = { i, i + 1, i + 2 };
      const double c[3] = { 1.0, -2.0, 1.0 };
      for (int a = 0; a < 3; ++a)
        for (int bb = 0; bb < 3; ++bb)
          AtA(idx[a], idx[bb]) += w * c[a] * c[bb];
      reg_trace += w * (1.0 + 4.0 + 1.0);
    }
  }
  fit_reg_frac_ = (data_trace > 0.0) ? (reg_trace / (data_trace + reg_trace)) : 1.0;
  if (fit_reg_frac_ > opts.fit_reg_max_frac) return false;

  Eigen::LDLT<Eigen::MatrixXd> ldlt(AtA);
  if (ldlt.info() != Eigen::Success) return false;

  const Eigen::MatrixXd Xp = ldlt.solve(Atb_p);
  const Eigen::MatrixXd Xr = ldlt.solve(Atb_r);
  if (!Xp.allFinite() || !Xr.allFinite()) return false;

  cp_p_.resize(3, n_cp_);
  cp_phi_.resize(3, n_cp_);
  for (int i = 0; i < n_cp_; ++i)
  {
    cp_p_.col(i)   = Xp.row(i).transpose();
    cp_phi_.col(i) = Xr.row(i).transpose();
  }

  // ---- fit residuals (reported, never silently swallowed) ----------------
  double sp = 0.0, sr = 0.0;
  for (size_t k = 0; k < rows.size(); ++k)
  {
    V3D pp = V3D::Zero(), pr = V3D::Zero();
    for (int i = 0; i < 4; ++i)
    {
      pp += rows[k][i] * cp_p_.col(row_cp[k] + i);
      pr += rows[k][i] * cp_phi_.col(row_cp[k] + i);
    }
    sp += (pp - tgt_p[k]).squaredNorm();
    sr += (pr - tgt_r[k]).squaredNorm();
  }
  fit_res_pos_ = std::sqrt(sp / static_cast<double>(rows.size()));
  fit_res_rot_ = std::sqrt(sr / static_cast<double>(rows.size()));

  valid_ = true;
  return true;
}

V3D ScanSpline::phiAt(double t) const
{
  Eigen::Vector4d b, db, ddb; int s = 0;
  basisAt(t, s, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += b[i] * cp_phi_.col(s + i);
  return r;
}

V3D ScanSpline::phiDotAt(double t) const
{
  Eigen::Vector4d b, db, ddb; int s = 0;
  basisAt(t, s, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += db[i] * cp_phi_.col(s + i);
  return r * inv_delta_;
}

M3D ScanSpline::rotAt(double t) const
{
  if (!valid_) return M3D::Identity();
  return R_anchor_ * Exp(phiAt(t));
}

V3D ScanSpline::posAt(double t) const
{
  if (!valid_) return V3D::Zero();
  Eigen::Vector4d b, db, ddb; int s = 0;
  basisAt(t, s, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += b[i] * cp_p_.col(s + i);
  return r;
}

V3D ScanSpline::velAt(double t) const
{
  if (!valid_) return V3D::Zero();
  Eigen::Vector4d b, db, ddb; int s = 0;
  basisAt(t, s, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += db[i] * cp_p_.col(s + i);
  return r * inv_delta_;
}

V3D ScanSpline::accAt(double t) const
{
  if (!valid_) return V3D::Zero();
  Eigen::Vector4d b, db, ddb; int s = 0;
  basisAt(t, s, b, db, ddb);
  V3D r = V3D::Zero();
  for (int i = 0; i < 4; ++i) r += ddb[i] * cp_p_.col(s + i);
  return r * (inv_delta_ * inv_delta_);
}

V3D ScanSpline::omegaBodyAt(double t) const
{
  if (!valid_) return V3D::Zero();
  // R = R_a Exp(phi)  =>  Rdot = R [ Jr(phi) phidot ]_x  =>  omega_body =
  // Jr(phi) phidot.  Exact for this parameterisation; no small-angle
  // approximation enters HERE (only the parameterisation's own validity,
  // which rotationChordDeg() monitors).
  return Jr(phiAt(t)) * phiDotAt(t);
}

void ScanSpline::anchorTo(double t_ref, const M3D& R_ref, const V3D& p_ref)
{
  if (!valid_) return;
  const M3D R_now = rotAt(t_ref);
  const V3D p_now = posAt(t_ref);

  const M3D dR = R_ref * R_now.transpose();

  // Position control points rotate about the reference point and translate.
  for (int i = 0; i < n_cp_; ++i)
    cp_p_.col(i) = dR * (cp_p_.col(i) - p_now) + p_ref;

  // Rotation: left-multiplying the whole trajectory by dR is exactly a
  // change of anchor, so the tangent control points are untouched.  This is
  // why the rigid re-anchor is O(n_cp) and cannot distort the shape.
  R_anchor_ = dR * R_anchor_;
}

double ScanSpline::rotationChordDeg() const
{
  if (!valid_) return 0.0;
  const M3D dR = rotAt(t0_).transpose() * rotAt(t1_);
  return Log(dR).norm() * 180.0 / M_PI;
}

SplineImuResidualStats computeSplineImuResidual(
    const ScanSpline& spline,
    const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity)
{
  SplineImuResidualStats st;
  if (!spline.valid() || imu.size() < 8) return st;

  std::vector<V3D> ra; ra.reserve(imu.size());
  std::vector<V3D> rw; rw.reserve(imu.size());

  for (const auto& s : imu)
  {
    // Skip rather than clamp: evaluating outside the fitted window
    // extrapolates the cubic, and that extrapolation error would land in
    // the statistic we are about to call "IMU noise".
    if (s.t < spline.t0() || s.t > spline.t1()) continue;

    const M3D R = spline.rotAt(s.t);
    const V3D a_pred = R.transpose() * (spline.accAt(s.t) - gravity) + bias_acc;
    const V3D w_pred = spline.omegaBodyAt(s.t) + bias_gyr;

    ra.push_back(a_pred - s.acc);
    rw.push_back(w_pred - s.gyro);
  }

  st.n = static_cast<int>(ra.size());
  if (st.n < 8) { st.n = 0; return st; }

  V3D ma = V3D::Zero(), mw = V3D::Zero();
  for (int i = 0; i < st.n; ++i) { ma += ra[i]; mw += rw[i]; }
  ma /= st.n; mw /= st.n;

  // Isotropic (trace/3) variance about the residual's own mean -- the same
  // reduction calib_processing.cpp's computeBiasAndNoise() applies, so this
  // number is directly comparable to the calibration floor.  Subtracting the
  // mean also makes the estimate insensitive to a slowly-varying bias error,
  // which is a bias problem and not a noise problem.
  double sa = 0.0, sw = 0.0;
  for (int i = 0; i < st.n; ++i)
  {
    sa += (ra[i] - ma).squaredNorm();
    sw += (rw[i] - mw).squaredNorm();
    st.max_abs_acc = std::max(st.max_abs_acc, (ra[i] - ma).cwiseAbs().maxCoeff());
    st.max_abs_gyr = std::max(st.max_abs_gyr, (rw[i] - mw).cwiseAbs().maxCoeff());
  }
  st.cov_acc = sa / (3.0 * (st.n - 1));
  st.cov_gyr = sw / (3.0 * (st.n - 1));

  // Lag-1 autocorrelation, pooled over axes.  This is the whiteness test:
  // if the residual is still correlated the spline has NOT absorbed the real
  // motion and the "noise" estimate above is contaminated by signal,
  // regardless of how small it is.  AdaptiveQ gates on it.
  double na = 0.0, nw = 0.0;
  for (int i = 1; i < st.n; ++i)
  {
    na += (ra[i] - ma).dot(ra[i - 1] - ma);
    nw += (rw[i] - mw).dot(rw[i - 1] - mw);
  }
  if (sa > 0.0) st.acf1_acc = na / sa;
  if (sw > 0.0) st.acf1_gyr = nw / sw;

  return st;
}

}  // namespace livo_recon
