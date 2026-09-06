#include "livo_recon/lio/spline.h"

#include <algorithm>
#include <cmath>

namespace livo_recon
{

namespace
{

// Fixed refinement regularisation, replacing the removed prior_w/damping
// config keys, and the step size above which a refinement is COUNTED as
// extreme (it is no longer vetoed).
constexpr double REFINE_TIKHONOV  = 1e-2;   // relative to trace(H)/dim
constexpr double REFINE_STEP_WARN = 0.10;   // metres

// Boundary-freeze helpers (SplineOptions::boundary_anchor_mode).  Two
// variants because the fit()/refineWithLidar()/fitRotationCumulative()
// solves are two different KINDS of linear system:
//
//   "absolute" systems (fit()'s position/tangent-rotation/gyro/accel
//   solves) solve DIRECTLY for the control point VALUES. Freezing here
//   must (a) subtract the frozen columns' known contribution from the
//   free rows' RHS -- otherwise the free solve silently assumes the
//   frozen columns are still 0 -- then (b) decouple the frozen rows/cols
//   and pin their RHS to the target, so solving gives exactly
//   X[i]=target for i<n_frozen and the mathematically correct
//   constrained value for i>=n_frozen.
//
//   "incremental" systems (refineWithLidar()'s Gauss-Newton step,
//   fitRotationCumulative()'s Gauss-Newton step) solve for a STEP/
//   increment applied on top of the CURRENT (already-frozen) values. The
//   desired step for a frozen index is exactly zero, and forcing that
//   makes the coupling term to the free rows vanish on its own -- no RHS
//   adjustment needed, just decouple and zero.
//
// Both assume `target`/frozen rows are indices [0, n_frozen) of a 3-wide
// (position/tangent-phi) or 3x3-block (LiDAR H, cumulative-rotation H)
// system; n_frozen is 0, 1 or 3 (SplineOptions::nFrozenCp()).

// A is n_cp x n_cp (one SHARED scalar matrix reused for all 3 spatial
// dims -- fit()'s own AtA/Atb_p / Atb_r pattern), b is n_cp x 3.
void freezeAbsoluteScalarSystem(Eigen::MatrixXd& A, Eigen::MatrixXd& b,
                                int n_frozen, const V3D& target)
{
  if (n_frozen <= 0) return;
  const int n = static_cast<int>(A.rows());
  for (int i = n_frozen; i < n; ++i)
    for (int j = 0; j < n_frozen; ++j)
      b.row(i) -= A(i, j) * target.transpose();
  for (int i = 0; i < n_frozen; ++i)
  {
    for (int j = 0; j < n; ++j) { A(i, j) = 0.0; A(j, i) = 0.0; }
    A(i, i) = 1.0;
    b.row(i) = target.transpose();
  }
}

// Tail variants of the two helpers above: same construction, indices
// [n_cp - n_frozen, n_cp) instead of [0, n_frozen).  The uniform cubic basis
// at u=1 is [0, 1/6, 4/6, 1/6] and sums to 1, so freezing the LAST three
// control points to a repeated value forces spline(t1) to that value exactly,
// mirroring the clamped-B-spline identity at t0.
void freezeAbsoluteScalarSystemTail(Eigen::MatrixXd& A, Eigen::MatrixXd& b,
                                    int n_frozen, const V3D& target)
{
  if (n_frozen <= 0) return;
  const int n = static_cast<int>(A.rows());
  const int first = n - n_frozen;
  if (first <= 0) return;
  for (int i = 0; i < first; ++i)
    for (int j = first; j < n; ++j)
      b.row(i) -= A(i, j) * target.transpose();
  for (int i = first; i < n; ++i)
  {
    for (int j = 0; j < n; ++j) { A(i, j) = 0.0; A(j, i) = 0.0; }
    A(i, i) = 1.0;
    b.row(i) = target.transpose();
  }
}

void freezeIncrementalBlockSystemTail(Eigen::MatrixXd& H, Eigen::VectorXd& g,
                                      int n_cp, int n_frozen)
{
  if (n_frozen <= 0) return;
  const int dim = static_cast<int>(H.rows());
  const int first = 3 * (n_cp - n_frozen);
  if (first < 0) return;
  for (int i = first; i < dim; ++i)
  {
    for (int j = 0; j < dim; ++j) { H(i, j) = 0.0; H(j, i) = 0.0; }
    H(i, i) = 1.0;
    g(i) = 0.0;
  }
}

// H is 3*n_cp x 3*n_cp (full 3x3 coupling blocks -- refineWithLidar()'s/
// fitRotationCumulative()'s own pattern), g is 3*n_cp. Increment/step
// variant: forces the solved step to 0 for the frozen blocks.
void freezeIncrementalBlockSystem(Eigen::MatrixXd& H, Eigen::VectorXd& g, int n_frozen)
{
  if (n_frozen <= 0) return;
  const int dim = static_cast<int>(H.rows());
  const int frozen_dim = 3 * n_frozen;
  for (int i = 0; i < frozen_dim; ++i)
  {
    for (int j = 0; j < dim; ++j) { H(i, j) = 0.0; H(j, i) = 0.0; }
    H(i, i) = 1.0;
    g(i) = 0.0;
  }
}

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

// Inverse SO(3) right Jacobian.  Jr(phi)^-1 = I + [phi]x/2 + c(theta) [phi]x^2
// with c = 1/theta^2 - (1+cos theta)/(2 theta sin theta), which tends to 1/12.
// The left inverse is its transpose: Jl(phi) = Jr(phi)^T, so Jl^-1 = (Jr^-1)^T.
inline M3D JrInv(const V3D& phi)
{
  const double n = phi.norm();
  M3D K; K << SKEW_SYM_MATRX(phi);
  double c;
  if (n < 1e-4)
    c = 1.0 / 12.0 + n * n / 720.0;          // series; the closed form is 0/0 here
  else
    c = 1.0 / (n * n) - (1.0 + std::cos(n)) / (2.0 * n * std::sin(n));
  return M3D::Identity() + 0.5 * K + c * K * K;
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
  // NOTE: the refinement counters are NOT reset here.  With
  // spline.reintegrate_each_iteration on, fit() runs once per IEKF iteration,
  // so resetting here would make spline_q.csv report the last iteration
  // instead of the frame.  LioProc calls resetRefineStats() once per frame.
  if (poses.size() < 2) return false;
  if (!(t1 > t0)) return false;

  cumulative_ = true;

  // n_cp comes from the control-point RATE and nothing else: a fixed n_cp is
  // a different control rate on every sequence with a different scan
  // duration.  Floor of 7 because both ends are clamped -- six control points
  // are spent on the two clamps, so below 7 there is no free interior at all.
  int n_cp = static_cast<int>(std::lround(opts.control_point_hz * (t1 - t0))) + 3;
  n_cp = std::max(7, n_cp);
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

  // Boundary freeze, BOTH ENDS.  Position and tangent-rotation need
  // different frozen targets, so each gets its own copy of AtA.  Three
  // control points at each end is the clamped-B-spline identity in both
  // directions: the uniform cubic basis is [1/6,4/6,1/6,0] at u=0 and
  // [0,1/6,4/6,1/6] at u=1, each summing to 1.
  fit_reg_frac_ = 0.0;
  const V3D frozen_phi  = (n_frozen_cp_ > 0) ? V3D(Log(R_anchor_T * frozen_rot_))  : V3D::Zero();
  const V3D frozen_phi1 = (n_frozen_cp_ > 0) ? V3D(Log(R_anchor_T * frozen_rot1_)) : V3D::Zero();
  Eigen::MatrixXd AtA_p = AtA, AtA_r = AtA;
  freezeAbsoluteScalarSystem(AtA_p, Atb_p, n_frozen_cp_, frozen_pos_);
  freezeAbsoluteScalarSystem(AtA_r, Atb_r, n_frozen_cp_, frozen_phi);
  freezeAbsoluteScalarSystemTail(AtA_p, Atb_p, n_frozen_cp_, frozen_pos1_);
  freezeAbsoluteScalarSystemTail(AtA_r, Atb_r, n_frozen_cp_, frozen_phi1);

  Eigen::LDLT<Eigen::MatrixXd> ldlt_p(AtA_p), ldlt_r(AtA_r);
  if (ldlt_p.info() != Eigen::Success || ldlt_r.info() != Eigen::Success) return false;

  const Eigen::MatrixXd Xp = ldlt_p.solve(Atb_p);
  const Eigen::MatrixXd Xr = ldlt_r.solve(Atb_r);
  if (!Xp.allFinite() || !Xr.allFinite()) return false;

  cp_p_.resize(3, n_cp_);
  cp_phi_.resize(3, n_cp_);
  for (int i = 0; i < n_cp_; ++i)
  {
    cp_p_.col(i)   = Xp.row(i).transpose();
    cp_phi_.col(i) = Xr.row(i).transpose();
  }

  // ── optional raw-IMU term ────────────────────────────────────────────────
  // Both weights default to 0, in which case not a single line below runs and
  // the fit is bit-identical to the pose-only one.  valid_ is set here rather
  // than at the end because the passes below evaluate the spline they are
  // solving for (phiAt/rotAt refuse to answer while invalid), and the object
  // is in fact a complete, usable fit at this point.
  const bool use_acc = (imu != nullptr) && imu->usable() && opts.imuFitAcc();
  const bool use_gyr = (imu != nullptr) && imu->usable() && opts.imuFitGyr();

  bias_acc_delta_ = V3D::Zero();
  bias_gyr_delta_ = V3D::Zero();
  gravity_delta_  = V3D::Zero();

  if (!fitRotationCumulative(poses, SplineOptions::ROT_FIT_ITERS))
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
  // Boundary freeze: the Greville seeding above is already approximately
  // close (cp_phi_ was frozen by the caller before this ran), but not
  // exactly -- overwrite directly so cp_R_[i] starts EXACT for i<n_frozen,
  // and the GN loop below (via freezeIncrementalBlockSystem) never moves
  // it away from this value.
  for (int i = 0; i < n_frozen_cp_ && i < n_cp_; ++i) cp_R_[i] = frozen_rot_;
  for (int i = 0; i < n_frozen_cp_ && i < n_cp_; ++i)
    cp_R_[n_cp_ - 1 - i] = frozen_rot1_;

  std::vector<double> ts;  ts.reserve(poses.size());
  std::vector<M3D>    Rs;  Rs.reserve(poses.size());
  for (const auto& ps : poses)
  {
    if (ps.t < t0_ - 1e-9 || ps.t > t1_ + 1e-9) continue;
    ts.push_back(ps.t);
    Rs.push_back(ps.rot);
  }
  if (static_cast<int>(ts.size()) < n_cp_ + 1) return false;

  // ── Gauss-Newton with the EXACT Jacobian of the cumulative product ───────
  //
  // R(u) = R_s A_1 A_2 A_3,  A_j = Exp(Btilde_j(u) d_j),  d_j = Log(R_{s+j-1}^T R_{s+j})
  //
  // Perturb control rotation s+k on the right, R_{s+k} <- R_{s+k} Exp(delta_k),
  // and collect the induced right perturbation of R:  R <- R Exp(sum_k J_k delta_k).
  // Each d_j depends on TWO control rotations, which is what the old
  // first-order stand-in dropped:
  //
  //     dd_j/ddelta_{j}    =  Jr(d_j)^-1
  //     dd_j/ddelta_{j-1}  = -Jl(d_j)^-1
  //
  // A right perturbation w_j of A_j moves through the trailing product as
  // C_j^T w_j with C_j = A_{j+1}...A_3 (for SO(3), R^-1 Exp(w) R = Exp(R^T w)),
  // and dA_j = A_j Exp(Btilde_j Jr(Btilde_j d_j) dd_j).  Writing
  // T_j = C_j^T Btilde_j Jr(Btilde_j d_j):
  //
  //     J_0 = C_0^T        - T_1 Jl(d_1)^-1        C_0 = A_1 A_2 A_3
  //     J_1 = T_1 Jr(d_1)^-1 - T_2 Jl(d_2)^-1
  //     J_2 = T_2 Jr(d_2)^-1 - T_3 Jl(d_3)^-1
  //     J_3 = T_3 Jr(d_3)^-1
  //
  // ANALYTIC CHECK, and it is the reason this can be trusted without a fixture:
  // as every rotation increment goes to zero (A_j -> I, C_j -> I, all Jacobians
  // -> I) the four blocks collapse to
  //     (1 - Bt_1) I,  (Bt_1 - Bt_2) I,  (Bt_2 - Bt_3) I,  Bt_3 I
  // and those are EXACTLY the ordinary cubic basis b_0..b_3 -- (1-u)^3/6,
  // (3u^3-6u^2+4)/6, (-3u^3+3u^2+3u+1)/6, u^3/6 -- which is what the previous
  // implementation used at every rotation magnitude.  So the old code was the
  // correct zeroth-order limit of this one, and this one reduces to it exactly.
  //
  // The residual is r = Log((R Exp(eps))^T R_pose) ~= e - Jl(e)^-1 eps with
  // e = Log(R^T R_pose), so each block is premultiplied by Jl(e)^-1.
  //
  // The system is 3*n_cp square (24x24 at n_cp=8) and banded -- the same shape
  // and cost as refineWithLidar()'s.  The previous version solved an n_cp
  // SCALAR system, which is what a Jacobian of b_j(u)*I permits and the exact
  // one does not.
  const int dim = 3 * n_cp_;
  for (int it = 0; it < std::max(0, gn_iters); ++it)
  {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(dim);
    Eigen::Vector3d Bt, dBt;
    int s = 0;
    int used = 0;

    for (size_t k = 0; k < ts.size(); ++k)
    {
      cumBasisAt(ts[k], s, Bt, dBt);
      if (s < 0 || s + 3 >= n_cp_) continue;

      V3D d[3]; M3D Aj[3];
      for (int j = 0; j < 3; ++j)
      {
        d[j]  = Log(cp_R_[s + j].transpose() * cp_R_[s + j + 1]);
        Aj[j] = Exp(V3D(Bt[j] * d[j]));
      }

      const M3D R = cp_R_[s] * Aj[0] * Aj[1] * Aj[2];
      const V3D e = Log(R.transpose() * Rs[k]);
      if (!e.allFinite()) continue;

      M3D C[3];                       // C[j] = product of the A's AFTER j
      C[2] = M3D::Identity();
      C[1] = Aj[2];
      C[0] = Aj[1] * Aj[2];
      const M3D C0 = Aj[0] * C[0];    // the whole product, for R_s

      M3D T[3], JrI[3], JlI[3];
      for (int j = 0; j < 3; ++j)
      {
        T[j]   = C[j].transpose() * (Bt[j] * Jr(V3D(Bt[j] * d[j])));
        JrI[j] = JrInv(d[j]);
        JlI[j] = JrI[j].transpose();
      }

      M3D J[4];
      J[0] = C0.transpose() - T[0] * JlI[0];
      J[1] = T[0] * JrI[0]  - T[1] * JlI[1];
      J[2] = T[1] * JrI[1]  - T[2] * JlI[2];
      J[3] = T[2] * JrI[2];

      const M3D Le = JrInv(e).transpose();   // Jl(e)^-1
      M3D M[4];
      for (int a = 0; a < 4; ++a) M[a] = Le * J[a];

      for (int a = 0; a < 4; ++a)
      {
        g.segment<3>(3 * (s + a)) += M[a].transpose() * e;
        for (int bb = 0; bb < 4; ++bb)
          H.block<3, 3>(3 * (s + a), 3 * (s + bb)) += M[a].transpose() * M[bb];
      }
      ++used;
    }
    if (used < n_cp_) return false;

    // Light ridge: the outermost control points are supported by few samples.
    const double ridge = 1e-9 * std::max(1.0, H.trace() / static_cast<double>(dim));
    for (int i = 0; i < dim; ++i) H(i, i) += ridge;

    // Boundary freeze: force the solved STEP to 0 for i<n_frozen_cp_, so
    // cp_R_[i] (already exactly frozen_rot_, set above) never moves.
    freezeIncrementalBlockSystem(H, g, n_frozen_cp_);
    freezeIncrementalBlockSystemTail(H, g, n_cp_, n_frozen_cp_);

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success) return false;
    const Eigen::VectorXd D = ldlt.solve(g);
    if (!D.allFinite()) return false;

    for (int i = 0; i < n_cp_; ++i)
      cp_R_[i] = cp_R_[i] * Exp(V3D(D.segment<3>(3 * i)));
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
// Move the TAIL clamp to a new scan-end pose without re-fitting, so any
// refinement already applied to the interior survives.
//
// The correction is distributed PROPORTIONALLY TO ELAPSED TIME: zero at the
// head clamp, full at the tail.  Control point i sits at the Greville
// abscissa (i - 1) * delta after t0 for a uniform cubic, so its normalised
// time is (i - 1) / (n_cp - 3); clamping that to [0,1] puts w = 0 on cp_0..2
// and w = 1 on the last three, which is what keeps BOTH identities exact.
//
// *** THE DISTRIBUTION IS A MODELLING CHOICE AND IT IS UNVALIDATED. ***
// Pinning both ends over-determines the trajectory relative to preserving
// the IMU's own relative increments -- the difference IS the drift the
// filter corrected -- so something must absorb it, and no measurement in
// this project says what.  Time-proportional is the same random-walk
// assumption Q itself encodes: process noise accumulates with time, so a
// control point halfway through the scan has accumulated half the drift.
// It is the `w` line below and nothing else depends on the choice.
void ScanSpline::moveTailClamp(const V3D& pos1, const M3D& rot1)
{
  if (!valid_ || n_cp_ <= 0) return;

  const V3D dp = pos1 - frozen_pos1_;
  const V3D dphi = Log(M3D(frozen_rot1_.transpose() * rot1));
  if (!dp.allFinite() || !dphi.allFinite()) return;

  const double span = std::max(1, n_cp_ - 3);
  for (int i = 0; i < n_cp_; ++i)
  {
    const double w = std::clamp((static_cast<double>(i) - 1.0) / span, 0.0, 1.0);
    cp_p_.col(i) += w * dp;
    if (cumulative_ && i < static_cast<int>(cp_R_.size()))
      cp_R_[i] = cp_R_[i] * Exp(V3D(w * dphi));
    else
      cp_phi_.col(i) += w * dphi;
  }

  frozen_pos1_ = pos1;
  frozen_rot1_ = rot1;
}

bool ScanSpline::refineWithLidar(const std::vector<SplineLidarObs>& obs,
                                 const SplineOptions& opts)
{
  if (!valid_ || !opts.refineOn()) return false;
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
    if (used < n_cp_) { accumulateRefineDisplacement(cp_prior); return any; }

    // Prior toward the pre-refinement (IMU-only) fit.  Without it the system
    // is rank-deficient in every direction the plane normals do not span --
    // and on a corridor or a facade that is a large subspace.
    // FIXED internal Tikhonov, no longer a config key (Bryce, 2026-09-06
    // removed prior_w and damping as knobs).  It cannot go to zero: without
    // ANY regulariser this system is rank-deficient in every direction the
    // plane normals do not span -- a corridor or a facade being exactly that
    // -- and the LDLT would fail into the unrefined spline in precisely the
    // scenes refinement exists for.  Scaled to the data term's own mean
    // diagonal so it is dimensionless.
    const double scale = std::max(1e-12, H.trace() / static_cast<double>(dim));
    for (int i = 0; i < dim; ++i) H(i, i) += REFINE_TIKHONOV * scale;

    // Boundary freeze: force the solved step to 0 for i<n_frozen_cp_ (cp_p_
    // was already exactly frozen_pos_ from fit() -- this keeps it there).
    // Both clamps: the refinement owns the interior SHAPE only.  The ESIKF
    // owns both endpoints and the solved step is forced to zero on them, so
    // the two estimators cannot fight over the same quantity -- structurally,
    // rather than by being overwritten afterwards.
    freezeIncrementalBlockSystem(H, g, n_frozen_cp_);
    freezeIncrementalBlockSystemTail(H, g, n_cp_, n_frozen_cp_);

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success)
    { ++refine_rejects_; accumulateRefineDisplacement(cp_prior); return any; }
    const Eigen::VectorXd step = -ldlt.solve(g);
    if (!step.allFinite())
    { ++refine_rejects_; accumulateRefineDisplacement(cp_prior); return any; }

    double max_step = 0.0;
    for (int i = 0; i < n_cp_; ++i)
      max_step = std::max(max_step, step.segment<3>(3 * i).norm());
    last_refine_step_ = max_step;

    // The max_step VETO is gone (Bryce, 2026-09-06): with both endpoints
    // clamped an interior excursion has to return to two fixed points, so
    // the damage is bounded geometrically rather than by a threshold.  The
    // COUNTER stays -- with prior_w gone the fixed Tikhonov above is the
    // only regulariser left, and refine_rejects_ is the sole signal that it
    // is being asked for something extreme.
    if (!(max_step <= REFINE_STEP_WARN)) ++refine_rejects_;

    for (int i = 0; i < n_cp_; ++i) cp_p_.col(i) += step.segment<3>(3 * i);
    ++refine_applied_;
    any = true;
  }
  accumulateRefineDisplacement(cp_prior);
  return any;
}

// NET displacement from the pre-refinement fit -- see refineDcpMax()'s doc
// comment for why an acceptance count is not a measurement.  Called on every
// exit path that may have modified cp_p_, and accumulated with std::max /
// quadrature so a frame whose re-deskew runs several times reports the
// largest and the aggregate, not the last.
void ScanSpline::accumulateRefineDisplacement(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& cp_prior)
{
  if (cp_prior.cols() != cp_p_.cols() || n_cp_ <= 0) return;
  double sum_sq = 0.0, mx = 0.0;
  for (int i = 0; i < n_cp_; ++i)
  {
    const double d = (cp_p_.col(i) - cp_prior.col(i)).norm();
    if (!std::isfinite(d)) continue;
    sum_sq += d * d;
    mx = std::max(mx, d);
  }
  refine_dcp_max_ = std::max(refine_dcp_max_, mx);
  const double rms = std::sqrt(sum_sq / static_cast<double>(n_cp_));
  refine_dcp_rms_ = std::sqrt(refine_dcp_rms_ * refine_dcp_rms_ + rms * rms);
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
