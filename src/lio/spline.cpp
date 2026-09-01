#include "livo_recon/lio/spline.h"

#include <algorithm>
#include <cmath>

namespace livo_recon
{

namespace
{

// Uniform cubic B-spline basis on u in [0,1) and its first two derivatives
// WITH RESPECT TO u.  b sums to 1, db and ddb each sum to 0 -- which is what
// makes the derivatives independent of a common offset of the control points.
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

// CUMULATIVE basis, Btilde_j(u) = sum_{k>=j} b_k(u) for j = 1,2,3 (j = 0 is
// identically 1 and never needed -- it multiplies the base rotation R_s).
// Derived directly from basisU above; the closed forms are
//   Btilde_1 = ( u^3 - 3u^2 + 3u + 5 ) / 6
//   Btilde_2 = ( -2u^3 + 3u^2 + 3u + 1 ) / 6
//   Btilde_3 = u^3 / 6
// and their u-derivatives
//   Btilde_1' = (1-u)^2 / 2
//   Btilde_2' = ( -2u^2 + 2u + 1 ) / 2
//   Btilde_3' = u^2 / 2
inline void cumBasisU(double u, Eigen::Vector3d& Bt, Eigen::Vector3d& dBt)
{
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double om = 1.0 - u;

  Bt[0] = (u3 - 3.0 * u2 + 3.0 * u + 5.0) / 6.0;
  Bt[1] = (-2.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
  Bt[2] = u3 / 6.0;

  dBt[0] = 0.5 * om * om;
  dBt[1] = 0.5 * (-2.0 * u2 + 2.0 * u + 1.0);
  dBt[2] = 0.5 * u2;
}

}  // namespace

void ScanSpline::basisAt(double t, int& first_cp, Eigen::Vector4d& b,
                         Eigen::Vector4d& db, Eigen::Vector4d& ddb) const
{
  const double tc = std::min(std::max(t, t0_), t1_);
  double x = (tc - t0_) * inv_delta_;
  int s = static_cast<int>(std::floor(x));
  if (s >= n_seg_) { s = n_seg_ - 1; x = static_cast<double>(n_seg_); }
  if (s < 0)       { s = 0;          x = 0.0; }
  const double u = x - static_cast<double>(s);
  basisU(u, b, db, ddb);
  first_cp = s;
}

void ScanSpline::cumBasisAt(double t, int& first_cp, Eigen::Vector3d& Bt,
                            Eigen::Vector3d& dBt) const
{
  const double tc = std::min(std::max(t, t0_), t1_);
  double x = (tc - t0_) * inv_delta_;
  int s = static_cast<int>(std::floor(x));
  if (s >= n_seg_) { s = n_seg_ - 1; x = static_cast<double>(n_seg_); }
  if (s < 0)       { s = 0;          x = 0.0; }
  const double u = x - static_cast<double>(s);
  cumBasisU(u, Bt, dBt);
  first_cp = s;
}

bool ScanSpline::fit(const std::vector<Pose6D>& poses, double t0, double t1,
                     const SplineOptions& opts)
{
  valid_ = false;
  refine_rejects_ = 0;
  refine_applied_ = 0;
  last_refine_step_ = 0.0;
  if (poses.size() < 2) return false;
  if (!(t1 > t0)) return false;

  cumulative_ = (opts.rot_mode != "tangent");

  int n_cp = opts.n_control_points;
  if (opts.control_point_hz > 0.0)
    n_cp = static_cast<int>(std::lround(opts.control_point_hz * (t1 - t0))) + 3;
  n_cp = std::max(4, std::min(n_cp, opts.n_control_points_max));
  n_cp_req_ = n_cp;

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

  // Anchor at the MIDDLE pose: halves the maximum |phi| the tangent
  // parameterisation has to carry, and gives the cumulative mode a
  // well-conditioned initialisation.
  R_anchor_ = poses[poses.size() / 2].rot;
  const M3D R_anchor_T = R_anchor_.transpose();

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

  // Second-difference regulariser on the control POLYGON, not on the
  // trajectory -- it damps ringing without flattening the fitted
  // acceleration, which is exactly what the Q estimate reads.
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
      reg_trace += w * 6.0;
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

  if (cumulative_ && !fitRotationCumulative(poses, opts.rot_fit_iters))
  {
    // Fall back to the tangent parameterisation rather than to a
    // half-initialised cumulative one -- a wrong rotation spline is worse
    // than an approximate one, and the mode is reported in the log.
    cumulative_ = false;
  }
  return valid_;
}

// Initialise the control rotations from the tangent fit evaluated at the
// Greville abscissae, then refine by Gauss-Newton against the pose sequence.
// The Greville abscissa of control point i for a uniform cubic is
// t0 + (i-1)*delta; the first and last lie outside [t0,t1] and are clamped by
// the evaluator, which is correct -- they are extrapolation control points and
// the GN step below moves them to wherever the data actually wants them.
bool ScanSpline::fitRotationCumulative(const std::vector<Pose6D>& poses, int gn_iters)
{
  cp_R_.assign(n_cp_, M3D::Identity());
  for (int i = 0; i < n_cp_; ++i)
  {
    const double tg = t0_ + (static_cast<double>(i) - 1.0) * delta_;
    cp_R_[i] = R_anchor_ * Exp(phiAt(tg));
  }

  std::vector<double> ts;  ts.reserve(poses.size());
  std::vector<M3D>    Rs;  Rs.reserve(poses.size());
  for (const auto& ps : poses)
  {
    if (ps.t < t0_ - 1e-9 || ps.t > t1_ + 1e-9) continue;
    ts.push_back(ps.t);
    Rs.push_back(ps.rot);
  }
  if (static_cast<int>(ts.size()) < n_cp_ + 1) return false;

  Eigen::Vector4d b, db, ddb;
  for (int it = 0; it < std::max(0, gn_iters); ++it)
  {
    // Normal equations in the ORDINARY basis: for small incremental
    // rotations the derivative of Log(R_pose^T R_spline) with respect to a
    // right perturbation of control rotation s+j is b_j(u) * I to first
    // order.  The initialisation is already close (the tangent fit is
    // accurate to ~1e-10 rad on benign motion), so first order converges in
    // the two iterations the default asks for -- and the moving-axis test
    // measures whether it actually did.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_cp_, n_cp_);
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(n_cp_, 3);
    int first_cp = 0;

    for (size_t k = 0; k < ts.size(); ++k)
    {
      basisAt(ts[k], first_cp, b, db, ddb);
      const V3D e = Log(rotAt(ts[k]).transpose() * Rs[k]);
      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j) H(first_cp + i, first_cp + j) += b[i] * b[j];
        g.row(first_cp + i) += b[i] * e.transpose();
      }
    }
    // Light ridge: the outermost control points are supported by few samples.
    const double ridge = 1e-9 * std::max(1.0, H.trace() / static_cast<double>(n_cp_));
    for (int i = 0; i < n_cp_; ++i) H(i, i) += ridge;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success) return false;
    const Eigen::MatrixXd D = ldlt.solve(g);
    if (!D.allFinite()) return false;

    for (int i = 0; i < n_cp_; ++i)
      cp_R_[i] = cp_R_[i] * Exp(V3D(D.row(i).transpose()));
  }

  double sr = 0.0;
  for (size_t k = 0; k < ts.size(); ++k)
    sr += Log(rotAt(ts[k]).transpose() * Rs[k]).squaredNorm();
  fit_res_rot_ = std::sqrt(sr / static_cast<double>(ts.size()));
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
  if (!cumulative_ || cp_R_.empty()) return R_anchor_ * Exp(phiAt(t));

  Eigen::Vector3d Bt, dBt; int s = 0;
  cumBasisAt(t, s, Bt, dBt);
  M3D R = cp_R_[s];
  for (int j = 1; j <= 3; ++j)
  {
    const V3D d = Log(cp_R_[s + j - 1].transpose() * cp_R_[s + j]);
    R = R * Exp(V3D(Bt[j - 1] * d));
  }
  return R;
}

V3D ScanSpline::omegaBodyAt(double t) const
{
  if (!valid_) return V3D::Zero();
  if (!cumulative_ || cp_R_.empty())
    // R = R_a Exp(phi)  =>  Rdot = R [ Jr(phi) phidot ]_x.  Exact for this
    // parameterisation; the approximation is in the parameterisation itself.
    return Jr(phiAt(t)) * phiDotAt(t);

  Eigen::Vector3d Bt, dBt; int s = 0;
  cumBasisAt(t, s, Bt, dBt);

  // Each A_j has a FIXED axis d_j, so Adot_j = A_j [ dBtilde_j/dt * d_j ]_x
  // with no right-Jacobian term.  Pushing all three through R^T Rdot gives
  //   omega = (A_2 A_3)^T w_1 + A_3^T w_2 + w_3.
  M3D A[3];
  V3D w[3];
  for (int j = 1; j <= 3; ++j)
  {
    const V3D d = Log(cp_R_[s + j - 1].transpose() * cp_R_[s + j]);
    A[j - 1] = Exp(V3D(Bt[j - 1] * d));
    w[j - 1] = (dBt[j - 1] * inv_delta_) * d;
  }
  return (A[1] * A[2]).transpose() * w[0] + A[2].transpose() * w[1] + w[2];
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

void ScanSpline::anchorTo(double t_ref, const M3D& R_ref, const V3D& p_ref)
{
  if (!valid_) return;
  const M3D R_now = rotAt(t_ref);
  const V3D p_now = posAt(t_ref);

  const M3D dR = R_ref * R_now.transpose();

  for (int i = 0; i < n_cp_; ++i)
    cp_p_.col(i) = dR * (cp_p_.col(i) - p_now) + p_ref;

  // Left-multiplying the whole trajectory by dR is exactly a change of
  // anchor, so the SHAPE is untouched.  In tangent mode that means the
  // anchor absorbs it; in cumulative mode every control rotation is
  // left-multiplied, which leaves every incremental d_j -- and therefore
  // omega_body -- bit-identical.
  R_anchor_ = dR * R_anchor_;
  for (auto& R : cp_R_) R = dR * R;
}

bool ScanSpline::refineWithLidar(const std::vector<SplineLidarObs>& obs,
                                 const SplineOptions& opts)
{
  if (!valid_ || !opts.lidar_refine_cp) return false;
  if (static_cast<int>(obs.size()) < n_cp_) return false;

  const int dim = 3 * n_cp_;
  const Eigen::Matrix<double, 3, Eigen::Dynamic> cp_prior = cp_p_;

  bool any = false;
  for (int it = 0; it < std::max(1, opts.lidar_refine_iters); ++it)
  {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(dim);

    Eigen::Vector4d b, db, ddb;
    int s = 0;
    int used = 0;

    for (const auto& o : obs)
    {
      if (!(o.sigma2 > 0.0) || !std::isfinite(o.r)) continue;
      if (o.t < t0_ - 1e-9 || o.t > t1_ + 1e-9) continue;
      basisAt(o.t, s, b, db, ddb);
      const double w = 1.0 / o.sigma2;
      const V3D& n = o.normal;

      // r_i is linear in the control points: dr/dcp[s+j] = b_j * n^T.
      // Gauss-Newton on 0.5 * sum w r^2 gives H = sum w (b_j n)(b_k n)^T,
      // g = sum w r b_j n, step = -H^{-1} g.
      for (int j = 0; j < 4; ++j)
      {
        const int rj = 3 * (s + j);
        for (int k = 0; k < 4; ++k)
          H.block<3, 3>(rj, 3 * (s + k)).noalias() += (w * b[j] * b[k]) * (n * n.transpose());
        g.segment<3>(rj).noalias() += (w * b[j] * o.r) * n;
      }
      ++used;
    }
    if (used < n_cp_) return any;

    // Prior toward the pre-refinement (IMU-only) fit.  Without it the system
    // is rank-deficient in every direction the plane normals do not span --
    // and on a corridor or a facade that is a large subspace.
    const double scale = std::max(1e-12, H.trace() / static_cast<double>(dim));
    const double wp = opts.lidar_refine_prior_w * scale;
    if (wp > 0.0)
    {
      for (int i = 0; i < n_cp_; ++i)
      {
        H.block<3, 3>(3 * i, 3 * i).diagonal().array() += wp;
        g.segment<3>(3 * i).noalias() += wp * (cp_p_.col(i) - cp_prior.col(i));
      }
    }
    // Levenberg damping on top, relative to the same scale.
    const double lm = std::max(0.0, opts.lidar_refine_damping) * scale;
    for (int i = 0; i < dim; ++i) H(i, i) += lm;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success) { ++refine_rejects_; return any; }
    const Eigen::VectorXd step = -ldlt.solve(g);
    if (!step.allFinite()) { ++refine_rejects_; return any; }

    double max_step = 0.0;
    for (int i = 0; i < n_cp_; ++i)
      max_step = std::max(max_step, step.segment<3>(3 * i).norm());
    last_refine_step_ = max_step;

    // Fail safe to the unrefined spline rather than applying a correction
    // this system's documented sensitivity cannot absorb.  Counted, not
    // hidden -- refineRejects() is logged per scan.
    if (!(max_step <= opts.lidar_refine_max_step)) { ++refine_rejects_; return any; }

    for (int i = 0; i < n_cp_; ++i) cp_p_.col(i) += step.segment<3>(3 * i);
    ++refine_applied_;
    any = true;
  }
  return any;
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
