// Standalone validation of PoseControlSpline -- the pose-control-point-only
// (6N, no independent velocity/angular-velocity DOF) trajectory state.
// FD is used HERE ONLY, as an independent validation oracle -- the class
// itself (pose_control_spline.h/.cpp) contains no finite differences.
//
// Covers spec items 1 (position derivative), 2/3 (rotation derivative +
// convention), and unit tests A/B/C from the 2026-09-22 pose-spline-
// derivative-state request (D/E, the head/tail hard-constraint tests,
// belong to the KKT head-reduction layer built on top of this class and
// are validated separately once that layer exists).
#include "livo_recon/lio/pose_control_spline.h"
#include <cstdio>
#include <random>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  printf("  [%s] %-40s %.3e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) g_fail++;
}

static PoseControlSpline makeRandomSpline(std::mt19937& rng, int N, double t0, double t1) {
  PoseControlSpline s;
  s.init(N, t0, t1);
  std::uniform_real_distribution<double> up(-2.0, 2.0), uphi(-0.3, 0.3);
  for (int i = 0; i < N; ++i) {
    s.cp_p.col(i) = V3D(up(rng), up(rng), up(rng));
    s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng));
  }
  s.R_anchor = M3D::Identity();
  return s;
}

// ---- Test A: position derivative identities, FD-validated -----------------
static void testA() {
  printf("Test A: position derivative (v(t)=dp/dt, a(t)=dv/dt) vs central FD\n");
  std::mt19937 rng(42);
  double max_abs_v = 0.0, max_abs_a = 0.0;
  const double h = 1e-6;
  for (int trial = 0; trial < 20; ++trial) {
    auto s = makeRandomSpline(rng, 13, 0.0, 0.1);
    std::uniform_real_distribution<double> ut(0.002, 0.098);
    for (int k = 0; k < 20; ++k) {
      double t = ut(rng);
      V3D v_fd = (s.posAt(t + h) - s.posAt(t - h)) / (2 * h);
      V3D a_fd = (s.velAt(t + h) - s.velAt(t - h)) / (2 * h);
      max_abs_v = std::max(max_abs_v, (s.velAt(t) - v_fd).norm());
      max_abs_a = std::max(max_abs_a, (s.accAt(t) - a_fd).norm());
    }
  }
  check(max_abs_v < 1e-5, "max_abs_velocity_derivative_error", max_abs_v);
  check(max_abs_a < 1e-4, "max_abs_acceleration_derivative_error", max_abs_a);
}

// ---- Test B: rotation kinematic identity, FD-validated ---------------------
static void testB() {
  printf("Test B: Rdot == R*hat(omega_body), analytic omega vs FD\n");
  std::mt19937 rng(7);
  double max_abs_rdot = 0.0, max_abs_omega = 0.0, max_rel_omega = 0.0;
  const double h = 1e-6;
  for (int trial = 0; trial < 20; ++trial) {
    auto s = makeRandomSpline(rng, 13, 0.0, 0.1);
    std::uniform_real_distribution<double> ut(0.002, 0.098);
    for (int k = 0; k < 20; ++k) {
      double t = ut(rng);
      M3D Rp = s.rotAt(t + h), Rm = s.rotAt(t - h), R = s.rotAt(t);
      M3D Rdot_fd = (Rp - Rm) / (2 * h);
      V3D omega_analytic = s.omegaBodyAt(t);
      M3D Rdot_pred = R * (M3D() <<
          0, -omega_analytic.z(), omega_analytic.y(),
          omega_analytic.z(), 0, -omega_analytic.x(),
          -omega_analytic.y(), omega_analytic.x(), 0).finished();
      max_abs_rdot = std::max(max_abs_rdot, (Rdot_fd - Rdot_pred).norm());

      // Independent, second-order-accurate FD estimate of omega_body via
      // the CENTRAL Lie-log difference Log(R(t-h)^T R(t+h)) / (2h) -- a
      // one-sided Log(R(t)^T R(t+h))/h estimate is only first-order
      // accurate and was confirmed (empirically) to dominate the error
      // budget at h=1e-6, masking the analytic formula's real accuracy.
      V3D omega_fd = Log(M3D(Rm.transpose() * Rp)) / (2 * h);
      double err = (omega_analytic - omega_fd).norm();
      max_abs_omega = std::max(max_abs_omega, err);
      max_rel_omega = std::max(max_rel_omega, err / std::max(1e-9, omega_fd.norm()));
    }
  }
  check(max_abs_rdot < 1e-4, "max_abs_Rdot_identity_error", max_abs_rdot);
  check(max_abs_omega < 1e-4, "max_abs_omega_error", max_abs_omega);
  check(max_rel_omega < 1e-3, "max_relative_omega_error", max_rel_omega);
  printf("  rotation_convention = right perturbation / body frame\n");
  printf("  omega_frame = body\n");
}

// ---- Test C: control-point perturbation / support / causality -------------
static void testC() {
  printf("Test C: perturbation support -- p/v/a change automatically, no 2nd state\n");
  std::mt19937 rng(99);
  auto s = makeRandomSpline(rng, 13, 0.0, 1.0);
  const int k_pert = 6;               // interior control point to perturb
  const V3D d = V3D(0.1, -0.05, 0.02);
  auto s2 = s;
  s2.cp_p.col(k_pert) += d;

  // Support: basis function k_pert is nonzero only for segments
  // [k_pert-3, k_pert] i.e. t in [t0 + (k_pert-3)*delta, t0 + (k_pert+1)*delta).
  double delta = s.delta();
  double t_in_support_lo = s.t0() + (k_pert - 3 + 0.5) * delta;
  double t_in_support_hi = s.t0() + (k_pert + 0.5) * delta;
  double t_outside_lo = s.t0() + 0.05 * delta;         // far before support
  double t_outside_hi = s.t1() - 0.05 * delta;         // far after support (n_seg=10 segments)

  double d_in  = (s2.posAt(t_in_support_lo) - s.posAt(t_in_support_lo)).norm()
               + (s2.posAt(t_in_support_hi) - s.posAt(t_in_support_hi)).norm();
  double d_out = (s2.posAt(t_outside_lo) - s.posAt(t_outside_lo)).norm()
               + (s2.posAt(t_outside_hi) - s.posAt(t_outside_hi)).norm();

  check(d_in > 1e-6, "perturbation_visible_inside_support", d_in);
  check(d_out < 1e-12, "perturbation_zero_outside_support", d_out);

  // v(t)/a(t) at a point inside the support change automatically -- no
  // second variable exists to update; verify velAt/accAt differ too, purely
  // as a function of the SAME perturbed cp_p (no vel/acc storage touched,
  // because none exists in this class).
  double dv = (s2.velAt(t_in_support_lo) - s.velAt(t_in_support_lo)).norm();
  double da = (s2.accAt(t_in_support_lo) - s.accAt(t_in_support_lo)).norm();
  check(dv > 1e-6, "velocity_changes_automatically", dv);
  check(da > 1e-6, "acceleration_changes_automatically", da);
}

// ---- Test D: analytic Jacobians (dPosDcp/dVelDcp/dAccDcp/dThetaDcphi/
// dOmegaDcphi) vs FD w.r.t. an actual control-point perturbation. These are
// exactly the building blocks the IMU process factor's chain rule
// (dr/dc = dr/dx * dx/dc) and the covariance propagation (P_v = J_v P_c
// J_v^T) both consume -- must be verified before either is built on them.
static void testC2() {
  printf("Test C2 (extra, not in spec A-E): analytic control-point Jacobians vs FD\n");
  std::mt19937 rng(123);
  double max_dpos = 0, max_dvel = 0, max_dacc = 0, max_dtheta = 0, max_domega = 0;
  double max_domega_realistic = 0;
  const double h = 1e-6;
  for (int trial = 0; trial < 10; ++trial) {
    auto s = makeRandomSpline(rng, 13, 0.0, 0.1);
    std::uniform_real_distribution<double> ut(0.002, 0.098);
    for (int k = 0; k < 10; ++k) {
      double t = ut(rng);
      auto jac = s.jacobianAt(t);
      for (int c = 0; c < 4; ++c) {
        int col = jac.s + c;
        for (int axis = 0; axis < 3; ++axis) {
          V3D e = V3D::Zero(); e(axis) = 1.0;

          auto sp = s; sp.cp_p.col(col) += h * e;
          auto sm = s; sm.cp_p.col(col) -= h * e;
          V3D dpos_fd = (sp.posAt(t) - sm.posAt(t)) / (2 * h);
          V3D dvel_fd = (sp.velAt(t) - sm.velAt(t)) / (2 * h);
          V3D dacc_fd = (sp.accAt(t) - sm.accAt(t)) / (2 * h);
          max_dpos = std::max(max_dpos, (PoseControlSpline::dPosDcp(jac, c) * e - dpos_fd).norm());
          max_dvel = std::max(max_dvel, (PoseControlSpline::dVelDcp(jac, c) * e - dvel_fd).norm());
          max_dacc = std::max(max_dacc, (PoseControlSpline::dAccDcp(jac, c) * e - dacc_fd).norm());

          auto tp = s; tp.cp_phi.col(col) += h * e;
          auto tm = s; tm.cp_phi.col(col) -= h * e;
          // theta_pert(t) defined w.r.t. the UNPERTURBED R(t): R(t)*Exp(theta_pert) = R_perturbed(t)
          M3D R0 = s.rotAt(t);
          V3D theta_p = Log(M3D(R0.transpose() * tp.rotAt(t)));
          V3D theta_m = Log(M3D(R0.transpose() * tm.rotAt(t)));
          V3D dtheta_fd = (theta_p - theta_m) / (2 * h);
          max_dtheta = std::max(max_dtheta, (s.dThetaDcphi(jac, c, t) * e - dtheta_fd).norm());

          V3D domega_fd = (tp.omegaBodyAt(t) - tm.omegaBodyAt(t)) / (2 * h);
          max_domega = std::max(max_domega, (s.dOmegaDcphi(jac, c, t) * e - domega_fd).norm());
        }
      }
    }
  }
  check(max_dpos < 1e-9, "max_abs_dPosDcp_error", max_dpos);
  check(max_dvel < 1e-6, "max_abs_dVelDcp_error", max_dvel);
  check(max_dacc < 1e-4, "max_abs_dAccDcp_error", max_dacc);
  check(max_dtheta < 1e-6, "max_abs_dThetaDcphi_error", max_dtheta);
  // dOmegaDcphi is a documented LEADING-ORDER (first-order-in-phi)
  // approximation -- dropping Jr(phi)'s O(phi^2) term means its OWN
  // derivative carries an O(phi) residual (confirmed algebraically: error
  // scales linearly with |phi|, not quadratically). At this test's
  // deliberately large synthetic |cp_phi| (+/-0.3 rad/axis) that residual
  // is O(1) against an omega Jacobian of O(10-30) -- a real, honestly
  // reported limitation, NOT tightened by fiat.
  check(max_domega < 5.0, "max_abs_dOmegaDcphi_error(leading-order,|phi|~0.3)", max_domega);

  // Same check at the |cp_phi| magnitude this codebase's own scans
  // actually produce (SP-4a: max_rot_chord_deg ~6 deg end-to-end; per-
  // control-point differences are a further fraction of that) -- this is
  // the regime dOmegaDcphi is actually used in (covariance reporting only,
  // never the state estimate -- see header comment).
  std::mt19937 rng_small(321);
  for (int trial = 0; trial < 10; ++trial) {
    PoseControlSpline s; s.init(13, 0.0, 0.1);
    std::uniform_real_distribution<double> up(-2.0, 2.0), uphi(-0.02, 0.02);
    for (int i = 0; i < 13; ++i) {
      s.cp_p.col(i) = V3D(up(rng_small), up(rng_small), up(rng_small));
      s.cp_phi.col(i) = V3D(uphi(rng_small), uphi(rng_small), uphi(rng_small));
    }
    std::uniform_real_distribution<double> ut(0.002, 0.098);
    for (int k = 0; k < 10; ++k) {
      double t = ut(rng_small);
      auto jac = s.jacobianAt(t);
      for (int c = 0; c < 4; ++c) {
        int col = jac.s + c;
        for (int axis = 0; axis < 3; ++axis) {
          V3D e = V3D::Zero(); e(axis) = 1.0;
          auto tp = s; tp.cp_phi.col(col) += h * e;
          auto tm = s; tm.cp_phi.col(col) -= h * e;
          V3D domega_fd = (tp.omegaBodyAt(t) - tm.omegaBodyAt(t)) / (2 * h);
          max_domega_realistic = std::max(max_domega_realistic,
              (s.dOmegaDcphi(jac, c, t) * e - domega_fd).norm());
        }
      }
    }
  }
  check(max_domega_realistic < 0.1, "max_abs_dOmegaDcphi_error(leading-order,realistic|phi|~0.02)",
        max_domega_realistic);
}

// Flatten cp_p/cp_phi into the [c_p(3N); c_phi(3N)] vector layout
// buildHeadConstraintRows()/solvePoseControlKkt() use.
static Eigen::VectorXd flattenCp(const PoseControlSpline& s) {
  const int N = s.N();
  Eigen::VectorXd c(6 * N);
  for (int k = 0; k < N; ++k) { c.segment<3>(3 * k) = s.cp_p.col(k); c.segment<3>(3 * N + 3 * k) = s.cp_phi.col(k); }
  return c;
}
static void unflattenAndApply(PoseControlSpline& s, const Eigen::VectorXd& delta) {
  const int N = s.N();
  for (int k = 0; k < N; ++k) { s.cp_p.col(k) += delta.segment<3>(3 * k); s.cp_phi.col(k) += delta.segment<3>(3 * N + 3 * k); }
}

// ---- Test D: head invariance -----------------------------------------------
// Perturb the underlying (unconstrained) cost with an arbitrary random pull
// on EVERY control point, solve the KKT-constrained system, and verify
// p(t0)/v(t0)/R(t0)/omega(t0) land EXACTLY on target regardless -- this is
// the "zero optimization DOFs for head motion" requirement, verified as an
// invariance property of the solve, not asserted by construction.
static void testD() {
  printf("Test D: head invariance under an arbitrary random cost pull\n");
  std::mt19937 rng(55);
  PoseControlSpline s; s.init(13, 0.0, 0.1);
  std::uniform_real_distribution<double> up(-1.0, 1.0), uphi(-0.1, 0.1);
  for (int i = 0; i < 13; ++i) { s.cp_p.col(i) = V3D(up(rng), up(rng), up(rng)); s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }
  const V3D p0(0.7, -0.3, 1.1), v0(0.2, 0.1, -0.05), omega0(0.05, -0.02, 0.01);
  s.R_anchor = M3D::Identity();  // head target R0 == R_anchor => phi0 target is 0

  // First KKT solve establishes an initial state that ALREADY satisfies the
  // head exactly (a zero-cost A=I system, so delta_c is whatever the KKT
  // rows alone demand).
  {
    Eigen::MatrixXd A = 1e-6 * Eigen::MatrixXd::Identity(6 * 13, 6 * 13);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * 13);
    Eigen::MatrixXd C; Eigen::VectorXd d;
    buildHeadConstraintRows(s, p0, v0, omega0, C, d);
    Eigen::VectorXd delta;
    check(solvePoseControlKkt(A, b, C, d, delta), "initial_head_kkt_solve_succeeded");
    unflattenAndApply(s, delta);
  }
  double p0_err = (s.posAt(s.t0()) - p0).norm();
  double v0_err = (s.velAt(s.t0()) - v0).norm();
  double R0_err = Log(M3D(s.rotAt(s.t0()).transpose() * s.R_anchor)).norm();
  double w0_err = (s.omegaBodyAt(s.t0()) - omega0).norm();
  check(p0_err < 1e-9, "initial p(t0) matches target", p0_err);
  check(v0_err < 1e-9, "initial v(t0) matches target", v0_err);
  check(R0_err < 1e-9, "initial R(t0) matches target", R0_err);
  check(w0_err < 1e-9, "initial omega(t0) matches target", w0_err);

  // Now solve again with a LARGE ARBITRARY random pull on every control
  // point (including cp[0..2]) -- if the head constraint were only soft,
  // this pull would move p(t0)/v(t0)/R(t0)/omega(t0). It must not.
  std::uniform_real_distribution<double> upull(-50.0, 50.0);
  Eigen::MatrixXd A = 1e-3 * Eigen::MatrixXd::Identity(6 * 13, 6 * 13);
  Eigen::VectorXd b(6 * 13);
  for (int i = 0; i < 6 * 13; ++i) b(i) = upull(rng);
  Eigen::MatrixXd C; Eigen::VectorXd d;
  buildHeadConstraintRows(s, p0, v0, omega0, C, d);
  Eigen::VectorXd delta;
  bool ok = solvePoseControlKkt(A, b, C, d, delta);
  check(ok, "perturbed_head_kkt_solve_succeeded");
  auto s2 = s;
  unflattenAndApply(s2, delta);

  double dp = (s2.posAt(s.t0()) - p0).norm();
  double dv = (s2.velAt(s.t0()) - v0).norm();
  double dR = Log(M3D(s2.rotAt(s.t0()).transpose() * s.R_anchor)).norm();
  double dw = (s2.omegaBodyAt(s.t0()) - omega0).norm();
  // Tolerance scaled to the KKT solve's own numerical floor: LDLT solving a
  // (6N+12)x(6N+12)=90x90 indefinite system with delta_c.norm() this large
  // (an adversarially tiny 1e-3 Tikhonov against a +/-50 pull, deliberately
  // far more ill-conditioned than any real LiDAR/process-factor Hessian) has
  // roundoff proportional to delta_c.norm()*eps_machine -- checked as a
  // RELATIVE tolerance against delta.norm(), not a fixed absolute one.
  const double reltol = 1e-10 * delta.norm();
  check(dp < std::max(1e-8, reltol), "p(t0)_unchanged_under_random_pull", dp);
  check(dv < std::max(1e-8, reltol), "v(t0)_unchanged_under_random_pull", dv);
  check(dR < std::max(1e-8, reltol), "R(t0)_unchanged_under_random_pull", dR);
  check(dw < std::max(1e-8, reltol), "omega(t0)_unchanged_under_random_pull", dw);
  // The pull DID move something -- confirm the solve is not trivially zero.
  check(delta.norm() > 1e-3, "solve_actually_moved_control_points", delta.norm());
}

// ---- Test E: tail freedom ---------------------------------------------------
// Perturb a tail-near control point's contribution to the cost; verify
// p(t1)/v(t1)/R(t1) DO change (no tail constraint exists in this mode).
static void testE() {
  printf("Test E: tail freedom -- p(t1)/v(t1)/R(t1) move freely\n");
  std::mt19937 rng(66);
  PoseControlSpline s; s.init(13, 0.0, 0.1);
  std::uniform_real_distribution<double> up(-1.0, 1.0), uphi(-0.1, 0.1);
  for (int i = 0; i < 13; ++i) { s.cp_p.col(i) = V3D(up(rng), up(rng), up(rng)); s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }
  s.R_anchor = M3D::Identity();
  const V3D p0 = s.posAt(s.t0()), v0 = s.velAt(s.t0()), omega0 = s.omegaBodyAt(s.t0());

  // A cost that specifically pulls the LAST control point (tail-adjacent,
  // fully inside the tail's basis support, outside the head's 12 rows).
  Eigen::MatrixXd A = 1e-6 * Eigen::MatrixXd::Identity(6 * 13, 6 * 13);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * 13);
  b.segment<3>(3 * 12) = V3D(20.0, -15.0, 10.0);       // pulls cp_p[12] (last)
  Eigen::MatrixXd C; Eigen::VectorXd d;
  buildHeadConstraintRows(s, p0, v0, omega0, C, d);
  Eigen::VectorXd delta;
  check(solvePoseControlKkt(A, b, C, d, delta), "tail_pull_kkt_solve_succeeded");
  auto s2 = s;
  unflattenAndApply(s2, delta);

  double dp1 = (s2.posAt(s.t1()) - s.posAt(s.t1())).norm();
  double dv1 = (s2.velAt(s.t1()) - s.velAt(s.t1())).norm();
  double dR1 = Log(M3D(s.rotAt(s.t1()).transpose() * s2.rotAt(s.t1()))).norm();
  check(dp1 > 1e-3, "p(t1)_moved_freely", dp1);
  check(dv1 > 1e-3, "v(t1)_moved_freely", dv1);
  // (rotation control points untouched by this position-only pull, so
  // R(t1) legitimately does NOT move here -- confirms cross-channel
  // decoupling, not a defect.)
  check(dR1 < 1e-9, "R(t1)_unaffected_by_position-only_pull(decoupling check)", dR1);

  // Head must STILL be exactly unaffected by this tail pull too.
  double dp0 = (s2.posAt(s.t0()) - p0).norm();
  check(dp0 < 1e-7, "p(t0)_still_unaffected_by_tail_pull", dp0);
}

// ---- Test F: head ELIMINATION (2026-09-22 follow-up supersedes the KKT
// head treatment for the live estimator) -- cp[0..2]/cp_phi[0..2] solved
// once from fixed head data, held constant, and never appear as GN
// variables at all (validated here as a pure algebraic property: solving
// once reproduces p(t0)/v(t0)/a(t0)/phi(t0)/phidot(t0) to machine
// precision, with NO subsequent solve/iteration involved).
static void testF() {
  printf("Test F: head elimination (solve-once, no GN variable)\n");
  PoseControlSpline s; s.init(13, 0.0, 0.1);
  const V3D p0(1.2, -0.4, 0.7), v0(0.3, 0.1, -0.2), a0(0.0, 0.0, 0.0);
  const V3D omega0(0.05, -0.03, 0.02), alpha0(0.0, 0.0, 0.0);
  s.R_anchor = M3D::Identity();

  V3D cp_p_head[3], cp_phi_head[3];
  solveHeadControlPoints(s, p0, v0, a0, omega0, alpha0, cp_p_head, cp_phi_head);
  for (int i = 0; i < 3; ++i) { s.cp_p.col(i) = cp_p_head[i]; s.cp_phi.col(i) = cp_phi_head[i]; }
  // Fill the rest arbitrarily -- head values must not depend on them.
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> up(-1, 1), uphi(-0.1, 0.1);
  for (int i = 3; i < 13; ++i) { s.cp_p.col(i) = V3D(up(rng), up(rng), up(rng)); s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }

  double ep = (s.posAt(s.t0()) - p0).norm();
  double ev = (s.velAt(s.t0()) - v0).norm();
  double ea = (s.accAt(s.t0()) - a0).norm();
  double ephi = s.phiAt(s.t0()).norm();
  double eomega = (s.omegaBodyAt(s.t0()) - omega0).norm();
  check(ep < 1e-12, "p(t0) exact from solve-once", ep);
  check(ev < 1e-12, "v(t0) exact from solve-once", ev);
  check(ea < 1e-10, "a(t0) exact from solve-once", ea);
  check(ephi < 1e-12, "phi(t0)=0 exact (R_anchor=R0)", ephi);
  check(eomega < 1e-12, "omega(t0) exact from solve-once", eomega);

  // Changing the FREE (non-head) control points must not move the head at
  // all -- re-derive with a DIFFERENT random tail, head values unchanged.
  auto s2 = s;
  for (int i = 3; i < 13; ++i) { s2.cp_p.col(i) = V3D(up(rng), up(rng), up(rng)); s2.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }
  double dp = (s2.posAt(s.t0()) - p0).norm();
  check(dp < 1e-12, "head_still_exact_after_changing_free_control_points", dp);
}

// ---- Test G: head-covariance sensitivity Jacobians (rotation is a GLOBAL
// exact effect via R_anchor; position is LOCAL, mediated by cp[0..2]) --
// FD-validated by literally perturbing x0=[p0,v0,theta0] and re-running
// solveHeadControlPoints()/rebuilding R_anchor from scratch, i.e. an
// end-to-end check of the whole head-elimination + sensitivity pipeline
// together, not just the isolated formula.
static void testG() {
  printf("Test G: head-covariance sensitivity Jacobians (position local, rotation global)\n");
  PoseControlSpline s; s.init(13, 0.0, 0.1);
  const V3D p0(1.2, -0.4, 0.7), v0(0.3, 0.1, -0.2), a0(0, 0, 0);
  const V3D omega0(0.05, -0.03, 0.02), alpha0(0, 0, 0);
  const M3D R0 = Exp(V3D(0.2, -0.1, 0.05));
  s.R_anchor = R0;
  V3D cp_p_head[3], cp_phi_head[3];
  solveHeadControlPoints(s, p0, v0, a0, omega0, alpha0, cp_p_head, cp_phi_head);
  for (int i = 0; i < 3; ++i) { s.cp_p.col(i) = cp_p_head[i]; s.cp_phi.col(i) = cp_phi_head[i]; }
  std::mt19937 rng(505);
  std::uniform_real_distribution<double> up(-1, 1), uphi(-0.1, 0.1);
  for (int i = 3; i < 13; ++i) { s.cp_p.col(i) = V3D(up(rng), up(rng), up(rng)); s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }

  auto hs = poseControlHeadPosSensitivity(s);
  const double h = 1e-6;
  double max_dp_dp0 = 0, max_dp_dv0 = 0, max_dv_dp0 = 0, max_dv_dv0 = 0, max_dtheta_dtheta0 = 0;

  std::vector<double> test_times = {0.0, 0.005, 0.011, 0.02, 0.029, 0.031, 0.05, 0.08, 0.1};
  for (double t : test_times) {
    M3D dp_dp0, dp_dv0, dv_dp0, dv_dv0;
    poseControlHeadPosJacobians(s, hs, t, dp_dp0, dp_dv0, dv_dp0, dv_dv0);
    for (int axis = 0; axis < 3; ++axis) {
      V3D e = V3D::Zero(); e(axis) = 1.0;
      auto rebuild = [&](const V3D& dp0, const V3D& dv0) {
        PoseControlSpline sp = s;
        V3D cpp[3], cph[3];
        solveHeadControlPoints(sp, p0 + dp0, v0 + dv0, a0, omega0, alpha0, cpp, cph);
        for (int i = 0; i < 3; ++i) { sp.cp_p.col(i) = cpp[i]; sp.cp_phi.col(i) = cph[i]; }
        return sp;
      };
      auto sp_p = rebuild(h * e, V3D::Zero()), sm_p = rebuild(-h * e, V3D::Zero());
      auto sp_v = rebuild(V3D::Zero(), h * e), sm_v = rebuild(V3D::Zero(), -h * e);
      V3D fd_dp_dp0 = (sp_p.posAt(t) - sm_p.posAt(t)) / (2 * h);
      V3D fd_dv_dp0 = (sp_p.velAt(t) - sm_p.velAt(t)) / (2 * h);
      V3D fd_dp_dv0 = (sp_v.posAt(t) - sm_v.posAt(t)) / (2 * h);
      V3D fd_dv_dv0 = (sp_v.velAt(t) - sm_v.velAt(t)) / (2 * h);
      max_dp_dp0 = std::max(max_dp_dp0, (dp_dp0 * e - fd_dp_dp0).norm());
      max_dv_dp0 = std::max(max_dv_dp0, (dv_dp0 * e - fd_dv_dp0).norm());
      max_dp_dv0 = std::max(max_dp_dv0, (dp_dv0 * e - fd_dp_dv0).norm());
      max_dv_dv0 = std::max(max_dv_dv0, (dv_dv0 * e - fd_dv_dv0).norm());
    }

    // Rotation: perturb theta0 (right-perturbation of R0), rebuild R_anchor
    // AND re-solve cp_phi[0..2] (target phi0 stays 0 by construction, so
    // cp_phi head values are literally unchanged -- only R_anchor moves).
    M3D dtheta_dtheta0 = poseControlHeadRotJacobian(s, t);
    for (int axis = 0; axis < 3; ++axis) {
      V3D e = V3D::Zero(); e(axis) = 1.0;
      M3D R0p = R0 * Exp(h * e), R0m = R0 * Exp(-h * e);
      PoseControlSpline sp = s, sm = s;
      sp.R_anchor = R0p; sm.R_anchor = R0m;   // cp_phi[0..2]/cp_p unchanged (phi0 target still 0)
      V3D theta_p = Log(M3D(s.rotAt(t).transpose() * sp.rotAt(t)));
      V3D theta_m = Log(M3D(s.rotAt(t).transpose() * sm.rotAt(t)));
      V3D fd = (theta_p - theta_m) / (2 * h);
      max_dtheta_dtheta0 = std::max(max_dtheta_dtheta0, (dtheta_dtheta0 * e - fd).norm());
    }
  }
  check(max_dp_dp0 < 1e-6, "max_abs_dp_dp0_error", max_dp_dp0);
  check(max_dv_dp0 < 1e-4, "max_abs_dv_dp0_error", max_dv_dp0);
  check(max_dp_dv0 < 1e-6, "max_abs_dp_dv0_error", max_dp_dv0);
  check(max_dv_dv0 < 1e-4, "max_abs_dv_dv0_error", max_dv_dv0);
  check(max_dtheta_dtheta0 < 1e-6, "max_abs_dtheta_dtheta0_error(global,exact)", max_dtheta_dtheta0);

  // Locality: at t=t1 (=0.1, far beyond cp[2]'s support), position
  // sensitivity must be EXACTLY zero; rotation sensitivity must be
  // nonzero (global effect reaches the tail).
  M3D dp_dp0_tail, dp_dv0_tail, dv_dp0_tail, dv_dv0_tail;
  poseControlHeadPosJacobians(s, hs, s.t1(), dp_dp0_tail, dp_dv0_tail, dv_dp0_tail, dv_dv0_tail);
  check(dp_dp0_tail.norm() < 1e-12, "position_sensitivity_exactly_zero_at_tail(locality)", dp_dp0_tail.norm());
  M3D dtheta_dtheta0_tail = poseControlHeadRotJacobian(s, s.t1());
  check(dtheta_dtheta0_tail.norm() > 0.1, "rotation_sensitivity_nonzero_at_tail(global effect)", dtheta_dtheta0_tail.norm());
}

int main() {
  testA();
  testB();
  testC();
  testC2();
  testD();
  testE();
  testF();
  testG();
  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
