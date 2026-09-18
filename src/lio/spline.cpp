#include "livo_recon/lio/spline.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace livo_recon
{

namespace
{

// Fixed refinement regularisation, replacing the removed prior_w/damping
// config keys, and the step size above which a refinement is COUNTED as
// extreme (it is no longer vetoed).
constexpr double REFINE_TIKHONOV  = 1e-2;   // relative to trace(H)/dim
constexpr double REFINE_STEP_WARN = 0.10;   // metres

// CQ-41 follow-up (2026-09-18): REFINE_TIKHONOV regularises each control
// point's DISPLACEMENT magnitude toward its pre-refinement prior, but
// nothing regularises the SHAPE of the resulting control polygon -- a few
// cm of independent per-point noise pulling adjacent control points in
// opposite directions is enough to make ddb.cp (and therefore accAt(),
// which scales it by inv_delta_^2 ~ 1e4 at 100Hz control points) blow up to
// thousands of m/s^2, even though each individual displacement is well
// under REFINE_STEP_WARN. This second-difference (discrete curvature)
// penalty targets exactly that: it penalises ||D * cp_new||^2, where D is
// the standard [1,-2,1] second-difference operator over control points, so
// a smooth/gently-curving control polygon costs nothing while an
// alternating high-frequency one is directly suppressed -- unlike
// REFINE_TIKHONOV, which cannot distinguish the two (both cost the same
// under a pure magnitude-toward-prior penalty). Weight is a config knob
// (SplineOptions::refine_curvature_weight, default 0.0/off) rather than a
// fixed constant like REFINE_TIKHONOV -- see that field's doc comment for
// the A/B numbers motivating it.

// CQ-41, Bryce 2026-09-18: the repeated-control-point freeze this comment
// block used to describe forced acceleration and angular velocity to
// exactly zero at both ends as an unintended side effect of the repeated-
// triple identity -- "we shouldn't be able to fix those." Replaced
// throughout this file by an explicit KKT-constrained least-squares
// system: the boundary condition (position, optionally velocity, always
// attitude -- never acceleration, angular velocity, or angular
// acceleration, none of which has a code path anywhere below) is a LINEAR
// EQUALITY solved jointly with the data term, not a value forced into the
// unknown vector by decoupling rows.
//
//   [ AtA   C^T ] [ X ]   [ Atb ]
//   [  C     0  ] [ L ] = [  D  ]
//
// C is k x n_cp (k constraint rows, shared across the 3 spatial dims, same
// "one scalar system reused via a 3-wide RHS" pattern AtA/Atb_p/Atb_r
// already use), D is k x 3 (the constraint targets). The matrix is
// symmetric INDEFINITE by construction (the zero block guarantees negative
// pivots) -- see fit()'s own comment for why the pivot-guard diagnostic
// must NOT be read off this matrix.

// Builds the k x n_cp constraint-row matrix C and k x 3 target D for one
// channel (position or rotation) of the KKT system above. include_rate
// adds a velocity/rate row at each end (position channel only, when
// SplineOptions::end_constraint_velocity is on) -- head_rate/tail_rate are
// ignored when include_rate is false. Basis values match basisU(0,...)/
// basisU(1,...) exactly (b=[1/6,4/6,1/6,0], db=[-1/2,0,1/2,0] at u=0; the
// mirror image at u=1) -- see item (2)'s own worked-out rows.
void buildEndConstraints(int n_cp, bool include_rate,
                         const V3D& head_target, const V3D& tail_target,
                         const V3D& head_rate, const V3D& tail_rate, double delta,
                         Eigen::MatrixXd& C, Eigen::MatrixXd& D)
{
  const int k_end = include_rate ? 2 : 1;
  const int k = 2 * k_end;
  C = Eigen::MatrixXd::Zero(k, n_cp);
  D = Eigen::MatrixXd::Zero(k, 3);

  C(0, 0) = 1.0 / 6.0; C(0, 1) = 4.0 / 6.0; C(0, 2) = 1.0 / 6.0;
  D.row(0) = head_target.transpose();
  if (include_rate) {
    C(1, 0) = -0.5; C(1, 2) = 0.5;
    D.row(1) = (head_rate * delta).transpose();   // d/dt = (db.cp)/delta
  }

  const int t0 = n_cp - 3;
  C(k_end, t0) = 1.0 / 6.0; C(k_end, t0 + 1) = 4.0 / 6.0; C(k_end, t0 + 2) = 1.0 / 6.0;
  D.row(k_end) = tail_target.transpose();
  if (include_rate) {
    C(k_end + 1, t0) = -0.5; C(k_end + 1, t0 + 2) = 0.5;
    D.row(k_end + 1) = (tail_rate * delta).transpose();
  }
}

// Builds and factors the (n_cp+k) x (n_cp+k) KKT matrix from AtA/Atb (the
// data term, shared across channels before this call) and C/D (one
// channel's constraint rows), solves for X (n_cp x 3, the control points --
// the Lagrange multipliers are discarded, this system is never queried for
// them), and leaves the factorization in ldlt_out for reuse (CQ-41 item 3c:
// "the KKT factorisation depends only on AtA and C, never on the RHS, so it
// can be REUSED" -- moveTailClamp()'s own constraint-increment solve is
// exactly that reuse, with a different, zero-data-term RHS).
bool solveKkt(const Eigen::MatrixXd& AtA, const Eigen::MatrixXd& Atb,
             const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
             Eigen::LDLT<Eigen::MatrixXd>& ldlt_out, Eigen::MatrixXd& X)
{
  const int n = static_cast<int>(AtA.rows());
  const int k = static_cast<int>(C.rows());
  Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + k, n + k);
  KKT.topLeftCorner(n, n) = AtA;
  KKT.topRightCorner(n, k) = C.transpose();
  KKT.bottomLeftCorner(k, n) = C;

  Eigen::MatrixXd rhs(n + k, 3);
  rhs.topRows(n) = Atb;
  rhs.bottomRows(k) = D;

  ldlt_out.compute(KKT);
  if (ldlt_out.info() != Eigen::Success) return false;
  const Eigen::MatrixXd sol = ldlt_out.solve(rhs);
  if (!sol.allFinite()) return false;
  X = sol.topRows(n);
  return true;
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


bool ScanSpline::fit(const std::vector<Pose6D>& poses, double t0, double t1,
                     const SplineOptions& opts)
{
  valid_ = false;
  // CQ-22 item (4): both reset every call, same reasoning as valid_ above --
  // fail_cause_ names WHY this call returned false (kNone on success);
  // chart_guard_warned_ is this call's own chart-guard observation, never
  // carried over from a previous frame.
  fail_cause_ = FitFailCause::kNone;
  chart_guard_warned_ = false;
  chart_guard_hard_ = false;
  pivot_guard_ = false;
  last_max_abs_cp_phi_ = 0.0;
  // CQ-24 item (1): -1 is the "never reached the LDLT solve this call"
  // sentinel -- a real pivot is always >= 0 (AtA is PSD by construction),
  // so -1 cannot be confused with a genuine (possibly exactly-zero) pivot.
  dmin_p_ = dmax_p_ = dmin_r_ = dmax_r_ = -1.0;
  // NOTE: the refinement counters are NOT reset here.  With
  // spline.reintegrate_each_iteration on, fit() runs once per IEKF iteration,
  // so resetting here would make spline_q.csv report the last iteration
  // instead of the frame.  LioProc calls resetRefineStats() once per frame.
  if (poses.size() < 2) { fail_cause_ = FitFailCause::kTooFewPoses; return false; }
  if (!(t1 > t0)) { fail_cause_ = FitFailCause::kBadWindow; return false; }


  // n_cp comes from the control-point RATE and nothing else: a fixed n_cp is
  // a different control rate on every sequence with a different scan
  // duration.  Floor of 7 because both ends are clamped -- six control points
  // are spent on the two clamps, so below 7 there is no free interior at all.
  int n_cp = static_cast<int>(std::lround(opts.control_point_hz * (t1 - t0))) + 3;
  n_cp = std::max(7, n_cp);
  n_cp_req_ = n_cp;

  const int n_samples = static_cast<int>(poses.size());
  if (n_samples < 5) { fail_cause_ = FitFailCause::kTooFewSamples; return false; }
  n_cp = std::min(n_cp, n_samples - 1);
  if (n_cp < 4) { fail_cause_ = FitFailCause::kNCpTooSmall; return false; }

  n_cp_  = n_cp;
  n_seg_ = n_cp_ - 3;
  t0_ = t0; t1_ = t1;
  delta_ = (t1_ - t0_) / static_cast<double>(n_seg_);
  if (!(delta_ > 1e-9)) { fail_cause_ = FitFailCause::kDegenerateDelta; return false; }
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
  if (static_cast<int>(rows.size()) < n_cp_ + 1) { fail_cause_ = FitFailCause::kUnderdetermined; return false; }

  // CQ-41: boundary condition, BOTH ENDS, as KKT linear-equality constraint
  // rows -- see the anonymous-namespace comment above for the full
  // derivation. Position and rotation(-attitude) each get their own
  // constraint set (buildEndConstraints()) but SHARE the same AtA/Atb data
  // term, unlike the old freeze which needed a separately-modified AtA per
  // channel.
  fit_reg_frac_ = 0.0;
  const V3D frozen_phi  = (n_frozen_cp_ > 0) ? V3D(Log(R_anchor_T * frozen_rot_))  : V3D::Zero();
  const V3D frozen_phi1 = (n_frozen_cp_ > 0) ? V3D(Log(R_anchor_T * frozen_rot1_)) : V3D::Zero();

  // CQ-41 item (3a): the pivot-guard diagnostic is read off AtA ALONE, via
  // its own LDLT -- NOT off the KKT matrix below, which is symmetric
  // INDEFINITE by construction (its zero block guarantees negative pivots),
  // so dmin_*/dmax_* from it would be meaningless and PIVOT_MIN_FLOOR would
  // misfire the moment it's ever above its shipped no-op value. AtA no
  // longer differs between the position and rotation channels (nothing
  // modifies it before this point any more, unlike the old per-channel-
  // frozen AtA_p/AtA_r) -- dmin_p_==dmin_r_ and dmax_p_==dmax_r_ are now the
  // same number by construction. Both names are kept (spline_q.csv schema
  // continuity), logging the identical value twice rather than collapsing
  // to one column.
  Eigen::LDLT<Eigen::MatrixXd> ldlt_diag(AtA);
  // CQ-24 item (1): logged BEFORE anything is refused, on every solve
  // regardless of info().
  dmin_p_ = dmin_r_ = ldlt_diag.vectorD().minCoeff();
  dmax_p_ = dmax_r_ = ldlt_diag.vectorD().maxCoeff();
  if (ldlt_diag.info() != Eigen::Success)
  { fail_cause_ = FitFailCause::kSolveFailed; return false; }

  // CQ-24 item (3): info() above cannot catch a rank-deficient-but-
  // "successful" solve (CQ-23's own finding). Absolute floor, not a
  // ratio. Checked BEFORE the KKT solve below: a near-singular data term is
  // not worth constraining and solving at all, let alone using.
  if (std::min(dmin_p_, dmin_r_) < SplineOptions::PIVOT_MIN_FLOOR)
  {
    pivot_guard_ = true;
    fail_cause_ = FitFailCause::kPivotGuard;
    return false;
  }

  // k=0 (no rows) whenever n_frozen_cp_<=0 -- the first spline scan of a
  // run has no previous boundary to constrain to, so it fits fully
  // unconstrained OLS (CQ-41 item 4's own instruction: do not invent a head
  // velocity, or any boundary, for it).
  //
  // CQ-41 item (3b), post-landing investigation, 2026-09-18: accAt(t0)/
  // accAt(t1) are now large in practice (median ~2000 m/s^2, eee_01,
  // control_point_hz=100) -- item 3b's own "report a worse number, do not
  // treat it as a failure" prediction, confirmed and explained rather than
  // just reproduced: (a) accAt scales curvature by 1/delta^2 (~1e4 at
  // delta~=0.01s), (b) the boundary is genuinely data-sparse (mg.poses is
  // IMU-SAMPLE-rate, not LiDAR-point-rate -- ~39 samples/scan here, a
  // handful of which fall near either boundary), and (c) DOUBLING
  // control_point_hz to 200 made it ~4.4x WORSE, not better (matching the
  // 1/delta^2 scaling almost exactly, with the same fixed IMU-sample budget
  // now spread across nearly twice as many control points) -- ruling out
  // "too coarse a spline" as the explanation. velocity:false (this
  // constraint fully off) does not shrink it either -- vel_err at the
  // boundary is then itself large (1-28 m/s) instead. Read together: this
  // is high-order-derivative sensitivity to IMU-propagated position noise
  // at a locally sparse boundary, not a defect in the constraint math
  // (verified separately: pos_err/vel_err at the CONSTRAINED boundary are
  // ~1e-14, i.e. the KKT solve meets exactly what it's asked to meet).
  const bool have_boundary = n_frozen_cp_ > 0;
  Eigen::MatrixXd C_p, D_p, C_r, D_r;
  if (have_boundary) {
    buildEndConstraints(n_cp_, opts.end_constraint_velocity,
                        frozen_pos_, frozen_pos1_, frozen_vel_, frozen_vel1_,
                        delta_, C_p, D_p);
    buildEndConstraints(n_cp_, /*include_rate=*/false,
                        frozen_phi, frozen_phi1, V3D::Zero(), V3D::Zero(),
                        delta_, C_r, D_r);
  } else {
    C_p = Eigen::MatrixXd(0, n_cp_); D_p = Eigen::MatrixXd(0, 3);
    C_r = Eigen::MatrixXd(0, n_cp_); D_r = Eigen::MatrixXd(0, 3);
  }

  Eigen::MatrixXd Xp, Xr;
  // kkt_ldlt_p_/kkt_ldlt_r_/kkt_k_p_/kkt_k_r_ are class members -- cached
  // here for moveTailClamp()'s later constraint-increment reuse (item 3c).
  kkt_k_p_ = static_cast<int>(C_p.rows());
  kkt_k_r_ = static_cast<int>(C_r.rows());
  const bool ok_p = solveKkt(AtA, Atb_p, C_p, D_p, kkt_ldlt_p_, Xp);
  const bool ok_r = solveKkt(AtA, Atb_r, C_r, D_r, kkt_ldlt_r_, Xr);
  if (!ok_p || !ok_r)
  { fail_cause_ = FitFailCause::kNonFinite; return false; }
  kkt_C_p_ = C_p;   // cached raw (unfactored) for refineWithLidar()'s
                     // block-expanded null-space constraint (item 6)

  cp_p_.resize(3, n_cp_);
  cp_phi_.resize(3, n_cp_);
  for (int i = 0; i < n_cp_; ++i)
  {
    cp_p_.col(i)   = Xp.row(i).transpose();
    cp_phi_.col(i) = Xr.row(i).transpose();
  }

  // valid_ is set here rather than at the end because the passes below
  // evaluate the spline they are solving for (phiAt/rotAt refuse to answer
  // while invalid), and the object is in fact a complete, usable fit at
  // this point.
  //
  // BUG FOUND+FIXED (CQ-21 verification, 2026-09-14): this assignment was
  // missing from the pushed restructure entirely -- valid_ was set false
  // at the top of fit() and never set true anywhere, so fit() always
  // returned false regardless of whether the solve succeeded. Confirmed
  // via spline_q.csv: spline_ok=0 on 100% of frames on a smoke cell,
  // engagement.txt reporting redeskew/refine both INERT despite
  // spline/mode=spline+refine.
  valid_ = true;
  bias_acc_delta_ = V3D::Zero();
  bias_gyr_delta_ = V3D::Zero();
  gravity_delta_  = V3D::Zero();

  // CHART GUARD, TWO QUESTIONS (CQ-22 stood it down to a warning; CQ-23
  // split it in two -- see SplineOptions::CHART_MAX_PHI_RAD/
  // CHART_HARD_PHI_RAD for the full history/reasoning). The uniform cubic
  // basis is non-negative and sums to 1, so phi(t) is a CONVEX COMBINATION
  // of the control points and max_t |phi(t)| <= max_i |cp_phi_[i]| -- a
  // cheap conservative bound, no sampling required.
  //
  // The value is stored FIRST, unconditionally, before either threshold is
  // consulted (CQ-23 item 2) -- a HARD-refused fit's own value must stay
  // observable through maxAbsCpPhi(), not disappear behind valid_=false.
  // The loop always runs to completion so the true max is found regardless
  // of which control point trips a threshold first.
  double max_phi = 0.0;
  for (int i = 0; i < n_cp_; ++i)
    max_phi = std::max(max_phi, cp_phi_.col(i).norm());
  last_max_abs_cp_phi_ = max_phi;

  if (max_phi > SplineOptions::CHART_HARD_PHI_RAD)
  {
    // The chart is not merely degrading -- CQ-23's own evidence (scan
    // 3133: Eigen::LDLT reported Success on both solves while
    // vectorD().minCoeff() was exactly 0) is that this is what a diverged,
    // numerically singular solve looks like. Treat it as the ninth hard
    // failure cause: refuse, fall back to deskewPoints(), count it
    // separately from the seven pre-existing causes and from the warning.
    chart_guard_hard_ = true;
    fail_cause_ = FitFailCause::kChartGuard;
    valid_ = false;
    return false;
  }
  if (max_phi > SplineOptions::CHART_MAX_PHI_RAD)
    chart_guard_warned_ = true;

  return valid_;
}

// CQ-22 item (5): fit_res_pos_/fit_res_rot_ used to be computed inside
// fit() itself -- i.e. against the control points as the propagation-time
// boundary freeze left them, BEFORE moveTailClamp() ever runs. Neither
// getter could therefore observe what the tail-clamp ramp actually does:
// confirmed by measurement, fit_res_rot_'s old in-fit computation changed
// by only 0.3% between a normal run and one with the ramp disabled
// entirely. Callers must invoke this AFTER the frame's last
// moveTailClamp() (LioProc does so from finalizeSplineAndQ()) so both
// residuals reflect the converged tail rather than the frozen-at-
// propagation one.
void ScanSpline::updateFitResiduals(const std::vector<Pose6D>& poses)
{
  if (!valid_) { fit_res_pos_ = 0.0; fit_res_rot_ = 0.0; return; }
  double sp = 0.0, sr = 0.0; int n = 0;
  for (const auto& ps : poses)
  {
    if (ps.t < t0_ - 1e-9 || ps.t > t1_ + 1e-9) continue;
    sp += (posAt(ps.t) - ps.pos).squaredNorm();
    sr += Log(M3D(rotAt(ps.t).transpose() * ps.rot)).squaredNorm();
    ++n;
  }
  fit_res_pos_ = (n > 0) ? std::sqrt(sp / static_cast<double>(n)) : 0.0;
  fit_res_rot_ = (n > 0) ? std::sqrt(sr / static_cast<double>(n)) : 0.0;
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

// ── the deskewPoints() hoist, for the spline ────────────────────────────────
// locate() is the whole per-point index cost; buildSegView() is the per-
// SEGMENT cost that used to be paid per point; poseAtSeg() is what is left
// over once both are hoisted.
void ScanSpline::locate(double t, int& s, double& u) const
{
  const double tc = std::min(std::max(t, t0_), t1_);
  double x = (tc - t0_) * inv_delta_;
  int si = static_cast<int>(std::floor(x));
  if (si >= n_seg_) { si = n_seg_ - 1; x = static_cast<double>(n_seg_); }
  if (si < 0)       { si = 0;          x = 0.0; }
  s = si;
  u = x - static_cast<double>(si);
}

void ScanSpline::buildSegView(int s, SegView& v) const
{
  if (v.s == s) return;                    // already built for this segment
  v.s = s;
  for (int i = 0; i < 4; ++i)
  {
    v.cp[i]  = cp_p_.col(s + i);
    v.phi[i] = cp_phi_.col(s + i);
  }
}

void ScanSpline::poseAtSeg(const SegView& v, double u, M3D& R, V3D& p) const
{
  Eigen::Vector4d b, db, ddb;
  basisU(u, b, db, ddb);
  p = b[0] * v.cp[0] + b[1] * v.cp[1] + b[2] * v.cp[2] + b[3] * v.cp[3];
  const V3D phi = b[0] * v.phi[0] + b[1] * v.phi[1]
                + b[2] * v.phi[2] + b[3] * v.phi[3];
  R = R_anchor_ * Exp(phi);
}

// Both halves of the pose from ONE segment lookup and ONE basis evaluation.
// Position and rotation share the ordinary cubic basis in the tangent
// parameterisation, which is the whole reason this is one call and not two.
void ScanSpline::poseAt(double t, M3D& R, V3D& p) const
{
  if (!valid_) { R = M3D::Identity(); p = V3D::Zero(); return; }
  Eigen::Vector4d b, db, ddb; int s = 0;
  basisAt(t, s, b, db, ddb);
  p = V3D::Zero(); V3D phi = V3D::Zero();
  for (int i = 0; i < 4; ++i)
  {
    p   += b[i] * cp_p_.col(s + i);
    phi += b[i] * cp_phi_.col(s + i);
  }
  R = R_anchor_ * Exp(phi);
}

M3D ScanSpline::rotAt(double t) const
{
  if (!valid_) return M3D::Identity();
  return R_anchor_ * Exp(phiAt(t));
}

// ANGULAR VELOCITY, AND THE RIGHT JACOBIAN IS NOT OPTIONAL.
//
//   R(t) = R_a Exp(phi(t))   =>   Rdot = R [ Jr(phi) phidot ]_x
//   so  omega_body = Jr(phi) phidot,  NOT phidot.
//
// Jr(phi) = I + O(theta), so dropping it is exact only as theta -> 0 or when
// the rotation AXIS is fixed.  At the ~13 deg this window carries off the
// mid-anchor that is a ~10% systematic error in angular velocity -- and
// omega_body is what computeSplineImuResidual() differences against the
// bias-corrected raw gyro to estimate sigma_gyr.  A bias here propagates
// into AdaptiveQ, hence into Q, hence into P.
//
// This is EXACT for this parameterisation.  The approximation lives in the
// parameterisation itself, not here.
V3D ScanSpline::omegaBodyAt(double t) const
{
  if (!valid_) return V3D::Zero();
  return Jr(phiAt(t)) * phiDotAt(t);
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
// CQ-41 item (3c): move the TAIL constraint to a new scan-end pose without
// re-fitting, so any refinement already applied to the interior survives.
// REPLACES the old time-proportional ramp (kept only at HEAD~1 -- see
// CQ-41's own filing) with a constraint-INCREMENT solve reusing fit()'s own
// KKT factorization: "The KKT factorisation depends only on AtA and C,
// never on the RHS, so it can be REUSED."
//
//   [ AtA   C^T ] [ dX ]   [  0  ]
//   [  C     0  ] [ dL ] = [ dD  ]
//
// dD is nonzero ONLY at the tail constraint rows (the head target hasn't
// moved), so this is the minimum-AtA-norm change to the control points that
// achieves the requested tail move: zero at the head by construction,
// distributed through the interior according to the fit's own metric
// (rather than a hand-chosen time ramp), and it disturbs applied
// refinement as little as that metric allows.
void ScanSpline::moveTailClamp(const V3D& pos1, const M3D& rot1, const V3D& vel1)
{
  if (!valid_ || n_cp_ <= 0) return;
  // No boundary this scan (kkt_k_p_==0, k=0 -- see fit()'s own comment):
  // nothing was constrained, so there is no constraint to move. Still
  // record the new targets so state stays consistent if a later scan finds
  // a boundary again.
  if (kkt_k_p_ <= 0)
  { frozen_pos1_ = pos1; frozen_rot1_ = rot1; frozen_vel1_ = vel1; return; }

  const V3D dp = pos1 - frozen_pos1_;
  const V3D dv = vel1 - frozen_vel1_;
  // In the tangent chart the clamp target is phi1 = Log(R_a^T Y), so the
  // move is a plain VECTOR DIFFERENCE in that chart -- not Log(Y^T Y').
  const M3D R_aT = R_anchor_.transpose();
  const V3D dphi = V3D(Log(M3D(R_aT * rot1))) - V3D(Log(M3D(R_aT * frozen_rot1_)));
  if (!dp.allFinite() || !dphi.allFinite() || !dv.allFinite()) return;

  // CQ-43 item 0: record BEFORE the early-return-on-solve-failure paths
  // below, same discipline as fit()'s own dmin_p_/dmax_p_ (CQ-24 item 1) --
  // this is what the caller's own moveTailClamp() invocation actually asked
  // for, independent of whether the KKT increment solve below succeeds.
  last_tail_move_dp_norm_   = dp.norm();
  last_tail_move_dphi_norm_ = dphi.norm() * (180.0 / M_PI);

  // k_end = rows PER END = kkt_k_p_/2 (2 if end_constraint_velocity was on
  // for the fit() this scan's cache came from, else 1) -- rotation's own
  // k_end is always 1 (attitude only, no rate row, either setting).
  const int k_end_p = kkt_k_p_ / 2;
  Eigen::MatrixXd D_p = Eigen::MatrixXd::Zero(kkt_k_p_, 3);
  D_p.row(k_end_p) = dp.transpose();
  if (k_end_p == 2) D_p.row(k_end_p + 1) = (dv * delta_).transpose();
  Eigen::MatrixXd D_r = Eigen::MatrixXd::Zero(kkt_k_r_, 3);
  D_r.row(1) = dphi.transpose();

  Eigen::MatrixXd rhs_p(n_cp_ + kkt_k_p_, 3); rhs_p.setZero();
  rhs_p.bottomRows(kkt_k_p_) = D_p;
  Eigen::MatrixXd rhs_r(n_cp_ + kkt_k_r_, 3); rhs_r.setZero();
  rhs_r.bottomRows(kkt_k_r_) = D_r;

  const Eigen::MatrixXd dXp = kkt_ldlt_p_.solve(rhs_p);
  const Eigen::MatrixXd dXr = kkt_ldlt_r_.solve(rhs_r);
  if (!dXp.allFinite() || !dXr.allFinite()) return;

  for (int i = 0; i < n_cp_; ++i)
  {
    cp_p_.col(i)   += dXp.row(i).transpose();
    cp_phi_.col(i) += dXr.row(i).transpose();
  }

  frozen_pos1_ = pos1;
  frozen_rot1_ = rot1;
  frozen_vel1_ = vel1;
}

bool ScanSpline::refineWithLidar(const std::vector<SplineLidarObs>& obs,
                                 const SplineOptions& opts,
                                 const std::vector<ImuSample>& imu_raw,
                                 const V3D& bias_acc, const V3D& gravity,
                                 double var_acc_floor, const V3D& cov_bias_acc,
                                 bool log_debug_en)
{
  if (!valid_ || !opts.refineOn()) return false;
  if (static_cast<int>(obs.size()) < n_cp_) return false;

  const int dim = 3 * n_cp_;
  const bool solve_bias = opts.refine_imu_acc_weight > 0.0 && opts.refine_imu_acc_solve_bias;
  const int  bias_off = dim;               // delta_ba occupies [dim, dim+3) when solve_bias
  const int  ext_dim  = dim + (solve_bias ? 3 : 0);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> cp_prior = cp_p_;

  // CQ-39: `obs` is a const input that never changes across passes of this
  // loop, and neither does anything H/g are built from (o.r/o.normal/
  // o.sigma2/o.t) -- only cp_p_ (the ACCUMULATED solution) changes, via the
  // `cp_p_.col(i) += step` at the loop's end, and nothing re-reads cp_p_ to
  // recompute o.r before the next pass. So every pass solves the IDENTICAL
  // linear system and produces the IDENTICAL step -- lidar_refine_iters=k
  // applies k times the SAME step, a step-magnitude multiplier, not k
  // independent re-linearizations. The genuine re-linearization against a
  // moved state happens one level up: LioProc::refineSplineFromResiduals()
  // is called fresh from inside the IEKF loop, against that iteration's own
  // freshly-rebuilt residual set. Verified via the per-pass log below --
  // see refineSplineFromResiduals()'s own doc comment for how this is
  // exercised.
  bool any = false;
  for (int it = 0; it < std::max(1, opts.lidar_refine_iters); ++it)
  {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(ext_dim, ext_dim);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(ext_dim);

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

    // Curvature (second-difference) prior -- see REFINE_CURVATURE_TIKHONOV's
    // doc comment above. Penalises ||D * (cp_prior + step)||^2, D being the
    // [1,-2,1] operator over control points (n_cp-2 rows); contributes
    // H += lambda * D^T D, g += lambda * D^T D * cp_prior (Gauss-Newton
    // normal-equation form for a linear residual D*cp_new, matching the data
    // term's own r_i-is-linear-in-cp derivation above). Block-expanded with
    // 3x3 identity blocks, same interleaved [x,y,z]-per-control-point layout
    // H/g already use.
    if (n_cp_ >= 3 && opts.refine_curvature_weight > 0.0) {
      const double lambda = opts.refine_curvature_weight * scale;
      for (int i = 0; i + 2 < n_cp_; ++i) {
        // Row i of D: +1 at cp i, -2 at cp i+1, +1 at cp i+2.
        static constexpr int    idx[3] = {0, 1, 2};
        static constexpr double val[3] = {1.0, -2.0, 1.0};
        for (int a = 0; a < 3; ++a) {
          const int ca = i + idx[a];
          for (int b = 0; b < 3; ++b) {
            const int cb = i + idx[b];
            const double w = lambda * val[a] * val[b];
            H.block<3, 3>(3 * ca, 3 * cb).diagonal().array() += w;
          }
          g.segment<3>(3 * ca).noalias() +=
              (lambda * val[a]) * (val[0] * cp_prior.col(i) + val[1] * cp_prior.col(i + 1) +
                                    val[2] * cp_prior.col(i + 2));
        }
      }
    }

    // IMU-acceleration anchor -- see SplineOptions::refine_imu_acc_weight's
    // doc comment. Penalises how far the refined spline's predicted
    // (body-frame) acceleration is from what the raw accelerometer actually
    // measured, weighted by 1/var_acc_floor. Linear in cp_p_ because
    // rotation (R(t), via rotAt()) is NOT touched by this position-only
    // refinement -- R^T R cancels to identity in H, so despite the rotation
    // this reduces to the same "linear-in-cp, isotropic 3x3 blocks" form the
    // data/curvature terms above already have. accAt(t) = ddb.cp * c with
    // c = inv_delta_^2 (same as the member accAt(), but evaluated against
    // cp_prior -- the fixed linearisation point every term in this pass
    // uses -- rather than live cp_p_, matching CQ-39's "every pass solves
    // the identical system" invariant when obs/imu_raw don't change).
    if (opts.refine_imu_acc_weight > 0.0 && var_acc_floor > 0.0) {
      const double w_imu = opts.refine_imu_acc_weight / var_acc_floor;
      const double c = inv_delta_ * inv_delta_;
      Eigen::Vector4d bi, dbi, ddbi;
      int si = 0;
      for (const auto& samp : imu_raw) {
        if (samp.t < t0_ - 1e-9 || samp.t > t1_ + 1e-9) continue;
        basisAt(samp.t, si, bi, dbi, ddbi);
        const M3D R = rotAt(samp.t);
        V3D acc_pred_world = V3D::Zero();
        for (int i = 0; i < 4; ++i) acc_pred_world += ddbi[i] * cp_prior.col(si + i);
        acc_pred_world *= c;
        const V3D r = R.transpose() * (acc_pred_world - gravity) + bias_acc - samp.acc;
        for (int j = 0; j < 4; ++j) {
          const int rj = 3 * (si + j);
          for (int k = 0; k < 4; ++k)
            H.block<3, 3>(rj, 3 * (si + k)).diagonal().array() +=
                w_imu * ddbi[j] * ddbi[k] * c * c;
          g.segment<3>(rj).noalias() += (w_imu * ddbi[j] * c) * (R * r);

          // Bryce, 2026-09-18: optional extra unknown delta_ba (accel bias
          // correction, constant over the scan) -- see
          // SplineOptions::refine_imu_acc_solve_bias's doc comment. Since
          // r(cp,delta_ba) = r_base + J_cp*step_cp + I3*delta_ba (linear),
          // Gauss-Newton adds a cross block J_cp_j^T W I3 = w_imu*ddb[j]*c*R
          // and its transpose, plus J_cp_j and I3 both contribute to
          // H(rj,rj)/H(bias,bias) as usual.
          if (solve_bias) {
            H.block<3, 3>(rj, bias_off).noalias()      += (w_imu * ddbi[j] * c) * R;
            H.block<3, 3>(bias_off, rj).noalias()      += (w_imu * ddbi[j] * c) * R.transpose();
          }
        }
        if (solve_bias) {
          H.block<3, 3>(bias_off, bias_off).diagonal().array() += w_imu;
          g.segment<3>(bias_off).noalias() += w_imu * r;
        }
      }
      // Prior toward zero correction, variance = cov_bias_acc * scan
      // duration -- the SAME per-scan process-noise budget the EKF's own
      // bias random walk already allows (ImuProc::propagate()'s
      // q_alpha_bias term, scaled by dt there exactly as here). Keeps this
      // diagnostic correction consistent with what the EKF's own model
      // would consider a plausible amount of bias drift within one scan,
      // rather than letting it fit noise unconstrained.
      if (solve_bias) {
        const double dt_scan = std::max(1e-6, t1_ - t0_);
        const double var_ba = std::max(1e-12, cov_bias_acc.mean() * dt_scan);
        H.block<3, 3>(bias_off, bias_off).diagonal().array() += 1.0 / var_ba;
      }
    }

    // CQ-41 item (6): the refinement step must respect the SAME position
    // constraints fit()/moveTailClamp() do, not a separate freeze -- a
    // NULL-SPACE solve (C.delta = 0) rather than forcing the step to zero
    // on the boundary control points. The refinement owns the interior
    // SHAPE subject to the endpoint constraints; it no longer "cannot
    // reach" the boundary by having those rows decoupled to identity, it is
    // constrained not to violate them -- structurally the same division of
    // labour the old freeze gave, expressed as an equality instead of a
    // fixed value.
    //
    //   [ H + tikhonov   Cb^T ] [ d ]   [ -g ]
    //   [ Cb             0    ] [ l ] = [  0 ]
    //
    // Cb = kkt_C_p_ (fit()'s own cached, unfactored position-channel
    // constraint rows) expanded to block form: each scalar entry becomes a
    // 3x3 diagonal block (one row -> 3 rows, one per spatial dim) so it
    // acts on H/g's interleaved [x,y,z] per-control-point block layout.
    // k=0 (kkt_k_p_==0, no boundary this scan) degenerates this to plain
    // unconstrained Gauss-Newton, matching fit()'s own k=0 case.
    // Cb stays sized on `dim` columns (delta_ba, when present, is entirely
    // unconstrained by the boundary condition -- it's a bias correction,
    // not part of the position spline) and gets zero-padded up to ext_dim
    // below.
    const int k = 3 * kkt_k_p_;
    Eigen::MatrixXd Cb = Eigen::MatrixXd::Zero(k, dim);
    for (int ci = 0; ci < kkt_k_p_; ++ci)
      for (int cj = 0; cj < n_cp_; ++cj)
        if (kkt_C_p_(ci, cj) != 0.0)
          Cb.block<3, 3>(3 * ci, 3 * cj) = kkt_C_p_(ci, cj) * M3D::Identity();

    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(ext_dim + k, ext_dim + k);
    KKT.topLeftCorner(ext_dim, ext_dim) = H;
    KKT.block(0, ext_dim, dim, k) = Cb.transpose();
    KKT.block(ext_dim, 0, k, dim) = Cb;
    Eigen::VectorXd rhs(ext_dim + k);
    rhs.setZero();
    rhs.head(ext_dim) = -g;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(KKT);
    if (ldlt.info() != Eigen::Success)
    { ++refine_rejects_; accumulateRefineDisplacement(cp_prior); return any; }
    const Eigen::VectorXd sol = ldlt.solve(rhs);
    if (!sol.allFinite())
    { ++refine_rejects_; accumulateRefineDisplacement(cp_prior); return any; }
    const Eigen::VectorXd step = sol.head(dim);
    if (solve_bias) last_delta_bias_acc_ = sol.segment<3>(bias_off);

    double max_step = 0.0;
    for (int i = 0; i < n_cp_; ++i)
      max_step = std::max(max_step, step.segment<3>(3 * i).norm());
    last_refine_step_ = max_step;

    // CQ-39 item (1): the proof. If the inner-loop comment above is right,
    // it/max_step/g.norm()/H.trace()/step.norm() are IDENTICAL across every
    // pass here whenever lidar_refine_iters > 1, since none of H/g/step's
    // inputs (obs) change between passes.
    if (log_debug_en) {
      static PersistentLogStream log("spline_refine_iters_debug.txt");
      std::ofstream& ofs = log.stream();
      ofs << "it=" << it << " max_step=" << max_step
          << " g_norm=" << g.norm() << " H_trace=" << H.trace()
          << " step_norm=" << step.norm() << "\n";
      ofs.flush();
    }

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

// See this method's own doc comment in spline.h. Deliberately duplicates
// (rather than shares) refineWithLidar()'s data-term-building loop -- the
// two need different constraint sets (both ends vs. head-only) applied to
// the SAME H/g, and factoring that out cleanly is more machinery than this
// one-off diagnostic is worth.
bool ScanSpline::diagnosticFreeTailFit(const std::vector<SplineLidarObs>& obs,
                                       const SplineOptions& opts,
                                       const std::vector<ImuSample>& imu_raw,
                                       const V3D& bias_acc, const V3D& gravity,
                                       double var_acc_floor,
                                       const std::vector<Pose6D>& poses,
                                       V3D& pos1_free, V3D& vel1_free,
                                       V3D& acc1_free, M3D& cov_pos1_free,
                                       Eigen::MatrixXd& cp_free_out,
                                       double& fit_res_pos_free_out) const
{
  if (!valid_ || !opts.refineOn()) return false;
  if (static_cast<int>(obs.size()) < n_cp_) return false;
  if (n_frozen_cp_ <= 0 || kkt_k_p_ <= 0) return false;

  const int dim = 3 * n_cp_;
  const Eigen::Matrix<double, 3, Eigen::Dynamic>& cp_prior = cp_p_;
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
    for (int j = 0; j < 4; ++j)
    {
      const int rj = 3 * (s + j);
      for (int k = 0; k < 4; ++k)
        H.block<3, 3>(rj, 3 * (s + k)).noalias() += (w * b[j] * b[k]) * (n * n.transpose());
      g.segment<3>(rj).noalias() += (w * b[j] * o.r) * n;
    }
    ++used;
  }
  if (used < n_cp_) return false;

  const double scale = std::max(1e-12, H.trace() / static_cast<double>(dim));
  for (int i = 0; i < dim; ++i) H(i, i) += REFINE_TIKHONOV * scale;

  // Same curvature prior refineWithLidar() applies -- see its own comment.
  if (n_cp_ >= 3 && opts.refine_curvature_weight > 0.0) {
    const double lambda = opts.refine_curvature_weight * scale;
    for (int i = 0; i + 2 < n_cp_; ++i) {
      static constexpr int    idx[3] = {0, 1, 2};
      static constexpr double val[3] = {1.0, -2.0, 1.0};
      for (int a = 0; a < 3; ++a) {
        const int ca = i + idx[a];
        for (int bb = 0; bb < 3; ++bb) {
          const int cb = i + idx[bb];
          H.block<3, 3>(3 * ca, 3 * cb).diagonal().array() += lambda * val[a] * val[bb];
        }
        g.segment<3>(3 * ca).noalias() +=
            (lambda * val[a]) * (val[0] * cp_prior.col(i) + val[1] * cp_prior.col(i + 1) +
                                  val[2] * cp_prior.col(i + 2));
      }
    }
  }

  // Same IMU-acceleration anchor refineWithLidar() applies (bias-only, no
  // delta_ba solve here -- that's a separate experiment) -- see its own
  // comment.
  if (opts.refine_imu_acc_weight > 0.0 && var_acc_floor > 0.0) {
    const double w_imu = opts.refine_imu_acc_weight / var_acc_floor;
    const double c = inv_delta_ * inv_delta_;
    Eigen::Vector4d bi, dbi, ddbi;
    int si = 0;
    for (const auto& samp : imu_raw) {
      if (samp.t < t0_ - 1e-9 || samp.t > t1_ + 1e-9) continue;
      basisAt(samp.t, si, bi, dbi, ddbi);
      const M3D R = rotAt(samp.t);
      V3D acc_pred_world = V3D::Zero();
      for (int i = 0; i < 4; ++i) acc_pred_world += ddbi[i] * cp_prior.col(si + i);
      acc_pred_world *= c;
      const V3D r = R.transpose() * (acc_pred_world - gravity) + bias_acc - samp.acc;
      for (int j = 0; j < 4; ++j) {
        const int rj = 3 * (si + j);
        for (int k = 0; k < 4; ++k)
          H.block<3, 3>(rj, 3 * (si + k)).diagonal().array() +=
              w_imu * ddbi[j] * ddbi[k] * c * c;
        g.segment<3>(rj).noalias() += (w_imu * ddbi[j] * c) * (R * r);
      }
    }
  }

  // HEAD-ONLY constraint: kkt_C_p_'s first half of rows -- buildEndConstraints()
  // always emits head rows first, tail rows second, in equal k_end-sized
  // halves (see its own comment).
  const int k_end = kkt_k_p_ / 2;
  const int k = 3 * k_end;
  Eigen::MatrixXd Cb = Eigen::MatrixXd::Zero(k, dim);
  for (int ci = 0; ci < k_end; ++ci)
    for (int cj = 0; cj < n_cp_; ++cj)
      if (kkt_C_p_(ci, cj) != 0.0)
        Cb.block<3, 3>(3 * ci, 3 * cj) = kkt_C_p_(ci, cj) * M3D::Identity();

  Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(dim + k, dim + k);
  KKT.topLeftCorner(dim, dim) = H;
  KKT.topRightCorner(dim, k) = Cb.transpose();
  KKT.bottomLeftCorner(k, dim) = Cb;
  Eigen::VectorXd rhs(dim + k);
  rhs.setZero();
  rhs.head(dim) = -g;

  Eigen::LDLT<Eigen::MatrixXd> ldlt(KKT);
  if (ldlt.info() != Eigen::Success) return false;
  const Eigen::VectorXd sol = ldlt.solve(rhs);
  if (!sol.allFinite()) return false;
  const Eigen::VectorXd step = sol.head(dim);

  // Evaluate the hypothetical free-tail control points at t1, on a LOCAL
  // copy -- cp_p_ itself is never touched.
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_free = cp_p_;
  for (int i = 0; i < n_cp_; ++i) cp_free.col(i) += step.segment<3>(3 * i);

  Eigen::Vector4d bb, dbb, ddbb;
  int ss = 0;
  basisAt(t1_, ss, bb, dbb, ddbb);
  pos1_free = V3D::Zero(); vel1_free = V3D::Zero(); acc1_free = V3D::Zero();
  for (int i = 0; i < 4; ++i) {
    pos1_free += bb[i] * cp_free.col(ss + i);
    vel1_free += dbb[i] * cp_free.col(ss + i);
    acc1_free += ddbb[i] * cp_free.col(ss + i);
  }
  vel1_free *= inv_delta_;
  acc1_free *= (inv_delta_ * inv_delta_);

  // TQ-36: expose the full control-point matrix (every d_i = cp_free_out.
  // col(i) - cp_p_.col(i), not just the t1 evaluation), and the free-tail
  // fit's own RMS residual against `poses` -- same formula as
  // updateFitResiduals()'s position term, evaluated against cp_free
  // instead of the live cp_p_ member (posAt() can't be reused here for
  // exactly that reason).
  cp_free_out = cp_free;
  {
    double sp = 0.0; int np = 0;
    Eigen::Vector4d bp, dbp, ddbp; int sp_idx = 0;
    for (const auto& ps : poses)
    {
      if (ps.t < t0_ - 1e-9 || ps.t > t1_ + 1e-9) continue;
      basisAt(ps.t, sp_idx, bp, dbp, ddbp);
      V3D pos_free_t = V3D::Zero();
      for (int i = 0; i < 4; ++i) pos_free_t += bp[i] * cp_free.col(sp_idx + i);
      sp += (pos_free_t - ps.pos).squaredNorm();
      ++np;
    }
    fit_res_pos_free_out = (np > 0) ? std::sqrt(sp / static_cast<double>(np)) : 0.0;
  }

  // TQ-35 item (5): the marginal covariance of pos1_free, for the
  // free_tail_d statistic d = (pos1_free-state_pos)^T (P_pp+Sigma_spline)^-1
  // (...). H was assembled from properly-weighted (1/variance) terms
  // throughout (data term: 1/sigma2; curvature/imu-accel: 1/var_acc_floor-
  // scaled), so the KKT inverse's leading dim x dim block IS the standard
  // Gauss-Newton/LS covariance of the control points -- solve for just the
  // 12 columns (4 relevant control points x 3 dims) touching t1's basis
  // support, rather than the full (expensive, mostly unused) inverse.
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(dim + k, 12);
  E.block<12, 12>(3 * ss, 0) = Eigen::MatrixXd::Identity(12, 12);
  const Eigen::MatrixXd cov_cols = ldlt.solve(E);
  if (!cov_cols.allFinite()) return false;
  const Eigen::MatrixXd cov_cp = cov_cols.block<12, 12>(3 * ss, 0);
  cov_pos1_free = M3D::Zero();
  for (int j = 0; j < 4; ++j)
    for (int kk = 0; kk < 4; ++kk)
      cov_pos1_free += (bb[j] * bb[kk]) * cov_cp.block<3, 3>(3 * j, 3 * kk);

  return true;
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
  st.mean_abs_acc = ma.norm();
  st.mean_abs_gyr = mw.norm();

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
