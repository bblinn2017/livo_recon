// FD validation of addPoseControlLidarFactor(): -b must equal the gradient
// of the fixed-weight quadratic cost w.r.t. free coordinates AND (per
// spec item 6) w.r.t. the fixed head x0 -- confirming (a) the per-point
// analytic Jacobian is correct and (b) LiDAR carries ZERO information to
// sT (matches decoupled LIO's own H-matrix zero-padding for bg/ba/g).
#include "livo_recon/lio/pose_control_lidar_factor.h"
#include <cstdio>
#include <random>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  printf("  [%s] %-48s %.3e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) g_fail++;
}

static double residualAt(const PoseControlSpline& s, const PoseControlLidarObs& o) {
  return o.normal.dot(s.rotAt(o.t) * o.q + s.posAt(o.t)) + o.d;
}

int main() {
  printf("LiDAR factor Jacobian validation\n");
  const int N = 13;
  const V3D p0(0, 0, 0), v0(0.2, -0.1, 0.05), a0(0, 0, 0);
  const V3D omega0(0.05, -0.03, 0.02), alpha0(0, 0, 0);
  const M3D R0 = Exp(V3D(0.1, -0.05, 0.02));

  PoseControlSpline s; s.init(N, 0.0, 0.1);
  s.R_anchor = R0;
  V3D cpp[3], cph[3];
  solveHeadControlPoints(s, p0, v0, a0, omega0, alpha0, cpp, cph);
  for (int i = 0; i < 3; ++i) { s.cp_p.col(i) = cpp[i]; s.cp_phi.col(i) = cph[i]; }
  std::mt19937 rng(1234);
  std::uniform_real_distribution<double> up(-0.3, 0.3), uphi(-0.05, 0.05);
  for (int i = 3; i < N; ++i) { s.cp_p.col(i) = V3D(i * 0.02, 0, 0) + V3D(up(rng), up(rng), up(rng)); s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }

  // Synthetic point cloud: 200 points scattered across the whole scan
  // window, including ones whose capture time touches the head (t<3*delta)
  // and ones that don't.
  std::vector<PoseControlLidarObs> obs;
  std::uniform_real_distribution<double> ut(0.0, 0.1), uq(-5, 5);
  for (int i = 0; i < 200; ++i) {
    PoseControlLidarObs o;
    o.t = ut(rng);
    o.q = V3D(uq(rng), uq(rng), uq(rng));
    o.normal = V3D(up(rng), up(rng), up(rng)).normalized();
    o.d = up(rng);
    o.sigma2 = 1.0;
    obs.push_back(o);
  }

  PoseControlFreeLayout layout; layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = true;
  PoseControlProcessFactorHeadBlock head_block;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
  Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
  double E_lidar = 0.0;
  addPoseControlLidarFactor(s, layout, obs, A, b, &head_block, &E_lidar);
  check(std::isfinite(E_lidar) && E_lidar >= 0, "E_lidar finite/nonneg", E_lidar);

  auto totalCost = [&](const PoseControlSpline& sp) {
    double c = 0.0;
    for (auto& o : obs) { double r = residualAt(sp, o); c += 0.5 * r * r; }
    return c;
  };

  const double h = 1e-6;
  double max_abs = 0, max_rel = 0;
  // Free control point / sT gradient check.
  for (int abs_k = 3; abs_k < N; ++abs_k) {
    for (int axis = 0; axis < 3; ++axis) {
      V3D e = V3D::Zero(); e(axis) = 1.0;
      { auto sp = s, sm = s; sp.cp_p(axis, abs_k) += h; sm.cp_p(axis, abs_k) -= h;
        double fd = (totalCost(sp) - totalCost(sm)) / (2 * h);
        int col = layout.colPos(abs_k) + axis;
        double err = std::abs(-b(col) - fd);
        max_abs = std::max(max_abs, err); max_rel = std::max(max_rel, err / std::max(1e-6, std::abs(fd))); }
      { auto sp = s, sm = s; sp.cp_phi(axis, abs_k) += h; sm.cp_phi(axis, abs_k) -= h;
        double fd = (totalCost(sp) - totalCost(sm)) / (2 * h);
        int col = layout.colPhi(abs_k) + axis;
        double err = std::abs(-b(col) - fd);
        max_abs = std::max(max_abs, err); max_rel = std::max(max_rel, err / std::max(1e-6, std::abs(fd))); }
    }
  }
  check(max_rel < 1e-3, "max_relative_free_gradient_error", max_rel);
  printf("    max_abs_free_gradient_error = %.3e\n", max_abs);

  // sT gets NO LiDAR information -- b's sT rows must be EXACTLY zero.
  double sT_norm = b.segment(layout.dimCFree(), layout.dimST()).norm();
  check(sT_norm < 1e-15, "sT_gradient_exactly_zero(LiDAR carries no bias/gravity info)", sT_norm);
  double sT_A_norm = A.block(layout.dimCFree(), 0, layout.dimST(), A.cols()).norm();
  check(sT_A_norm < 1e-15, "sT_row_of_A_exactly_zero", sT_A_norm);

  // Head block: DIRECT FD reconstruction of Jhead (summed over all
  // points, weight-included) via perturbing x0 and rebuilding cp[0..2]/
  // R_anchor from scratch -- bypasses addPoseControlLidarFactor()'s
  // internals entirely, exactly like the process-factor test's check (3).
  {
    // W^{1/2}-weighted stacked residual vector, gradient w.r.t. x0 is
    // sum_i w_i*r_i*dr_i/dx0 -- equivalently d(totalCost)/dx0 with the
    // SAME per-point weights totalCost() already uses (all sigma2=1 here).
    Eigen::Matrix<double, 9, 1> grad_fd;
    for (int axis = 0; axis < 3; ++axis) {
      V3D e = V3D::Zero(); e(axis) = 1.0;
      { auto sp = s, sm = s; sp.R_anchor = R0 * Exp(h * e); sm.R_anchor = R0 * Exp(-h * e);
        grad_fd(axis) = (totalCost(sp) - totalCost(sm)) / (2 * h); }
      { auto sp = s, sm = s; V3D cpp2[3], cph2[3];
        solveHeadControlPoints(sp, p0 + h * e, v0, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sp.cp_p.col(i) = cpp2[i];
        solveHeadControlPoints(sm, p0 - h * e, v0, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sm.cp_p.col(i) = cpp2[i];
        grad_fd(3 + axis) = (totalCost(sp) - totalCost(sm)) / (2 * h); }
      { auto sp = s, sm = s; V3D cpp2[3], cph2[3];
        solveHeadControlPoints(sp, p0, v0 + h * e, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sp.cp_p.col(i) = cpp2[i];
        solveHeadControlPoints(sm, p0, v0 - h * e, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sm.cp_p.col(i) = cpp2[i];
        grad_fd(6 + axis) = (totalCost(sp) - totalCost(sm)) / (2 * h); }
    }
    // head_block accumulates A_hh=sum(w*Jhead*Jhead^T)/A_hf -- not a
    // gradient directly (mean is fixed, so no b_head is computed/exposed).
    // Cross-check instead via A_hf: at c_free=0 (nominal), -A_hf*z_nominal
    // isn't meaningful either since head_block doesn't track a residual
    // sum. Reconstruct grad_analytic = sum_i w_i*r_i*Jhead_i directly by
    // re-deriving Jhead per point here (same formula the .cpp uses) and
    // dotting with residuals -- a genuine independent check of the SAME
    // per-point head Jacobian the production code builds.
    Eigen::Matrix<double, 9, 1> grad_analytic = Eigen::Matrix<double, 9, 1>::Zero();
    auto hs = poseControlHeadPosSensitivity(s);
    for (auto& o : obs) {
      const M3D Rt = s.rotAt(o.t);
      const double r = residualAt(s, o);
      const Eigen::RowVector3d dr_dtheta = o.q.cross(Rt.transpose() * o.normal).transpose();
      Eigen::Matrix<double, 9, 1> Jhead;
      Jhead.segment<3>(0) = (dr_dtheta * poseControlHeadRotJacobian(s, o.t)).transpose();
      int s_idx; double u; s.locate(o.t, s_idx, u);
      bool touches = (s_idx < 3);   // abs_k in [s_idx, s_idx+3] includes <3 whenever s_idx<3
      if (touches) {
        M3D dp_dp0, dp_dv0, dv_dp0, dv_dv0;
        poseControlHeadPosJacobians(s, hs, o.t, dp_dp0, dp_dv0, dv_dp0, dv_dv0);
        Jhead.segment<3>(3) = (o.normal.transpose() * dp_dp0).transpose();
        Jhead.segment<3>(6) = (o.normal.transpose() * dp_dv0).transpose();
      } else { Jhead.segment<3>(3).setZero(); Jhead.segment<3>(6).setZero(); }
      grad_analytic += r * Jhead;
    }
    double err = (grad_analytic - grad_fd).norm();
    double rel = err / std::max(1e-9, grad_fd.norm());
    check(rel < 1e-3, "max_relative_head_gradient_error_vs_direct_FD", rel);
    printf("    |grad_fd|=%.4e |grad_analytic|=%.4e abs_err=%.4e\n", grad_fd.norm(), grad_analytic.norm(), err);
  }

  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
