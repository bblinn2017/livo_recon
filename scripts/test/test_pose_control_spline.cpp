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
static void testD() {
  printf("Test D: analytic control-point Jacobians vs FD\n");
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

int main() {
  testA();
  testB();
  testC();
  testD();
  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
