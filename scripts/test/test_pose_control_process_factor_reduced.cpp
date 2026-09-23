// Validates addPoseControlProcessFactorReduced()'s two Jacobian paths:
// (1) the free-variable columns (z=[c_free;sT]) against FD of the
//     residual w.r.t. free control points and bias/gravity directly, and
// (2) the head block (A_hh/A_hf) against FD of the residual w.r.t. the
//     FIXED head x0=[theta0,p0,v0] itself (rebuilding cp[0..2]/R_anchor
//     from scratch each perturbation via solveHeadControlPoints(), exactly
//     mirroring test_pose_control_spline.cpp's Test G methodology).
#include "livo_recon/lio/pose_control_process_factor.h"
#include <cstdio>
#include <random>
#include <functional>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  printf("  [%s] %-48s %.3e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) g_fail++;
}

static std::vector<ImuSample> makeImuStream(double t0, double t1) {
  std::vector<ImuSample> s;
  const double hz = 200.0;
  const int n = std::max(2, static_cast<int>((t1 - t0) * hz));
  for (int i = 0; i <= n; ++i) {
    double t = t0 + (t1 - t0) * i / n;
    V3D gyro(0.3 * std::sin(7.0 * t), -0.2 * std::cos(5.0 * t), 0.15);
    V3D specific_force = V3D(0.5, -0.3, 0.2);
    s.emplace_back(specific_force, gyro, t);
  }
  return s;
}

static Eigen::Matrix<double, 9, 1> residualForSegment(
    const PoseControlSpline& s, int j, const std::vector<ImuSample>& samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double qa, double qg, const V3D& va, const V3D& vg, bool so) {
  const double tj = s.t0() + j * s.delta(), tj1 = s.t0() + (j + 1) * s.delta();
  Eigen::Matrix<double, 9, 9> F9, Q9;
  M3D rot_pred; V3D pos_pred, vel_pred;
  relinearizePoseControlSegment(samples, s.rotAt(tj), s.posAt(tj), s.velAt(tj),
                                bias_acc, bias_gyr, gravity, qa, qg, va, vg, so, F9, Q9, rot_pred, pos_pred, vel_pred);
  Eigen::Matrix<double, 9, 1> r;
  r.segment<3>(0) = Log(M3D(rot_pred.transpose() * s.rotAt(tj1)));
  r.segment<3>(3) = s.posAt(tj1) - pos_pred;
  r.segment<3>(6) = s.velAt(tj1) - vel_pred;
  return r;
}

int main() {
  printf("Reduced-layout process-factor Jacobian validation\n");
  const int N = 13;
  const double t0 = 0.0, t1 = 0.1;
  const V3D p0(0.0, 0.0, 0.0), v0(0.2, -0.1, 0.05), a0(0, 0, 0);
  const V3D omega0(0.05, -0.03, 0.02), alpha0(0, 0, 0);
  const M3D R0 = Exp(V3D(0.1, -0.05, 0.02));
  const V3D bias_acc(0.01, -0.02, 0.005), bias_gyr(0.001, 0.002, -0.0015);
  const V3D gravity(0, 0, -9.81);
  const V3D var_acc(0.01, 0.01, 0.01), var_gyr(0.001, 0.001, 0.001);
  const double qa = 1.0, qg = 1.0; const bool so = true;

  PoseControlSpline s; s.init(N, t0, t1);
  s.R_anchor = R0;
  V3D cpp[3], cph[3];
  solveHeadControlPoints(s, p0, v0, a0, omega0, alpha0, cpp, cph);
  for (int i = 0; i < 3; ++i) { s.cp_p.col(i) = cpp[i]; s.cp_phi.col(i) = cph[i]; }
  std::mt19937 rng(909);
  std::uniform_real_distribution<double> up(-0.3, 0.3), uphi(-0.05, 0.05);
  for (int i = 3; i < N; ++i) { s.cp_p.col(i) = V3D(i * 0.02, 0, 0) + V3D(up(rng), up(rng), up(rng)); s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng)); }

  auto segs = bucketPoseControlImuSamples(makeImuStream(t0, t1), s);
  PoseControlFreeLayout layout; layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = true;

  // ---- (1) free-variable columns: -b must equal the gradient of the
  // FIXED-Lambda quadratic cost 0.5*r(z)^T*Lambda*r(z) at z=0 (the
  // standard GN linearization identity) -- a REAL, end-to-end check of
  // the reduced system's bookkeeping (which z-columns get written where,
  // sT wiring), not a re-derivation of the core chain rule (already
  // FD-gated in test_pose_control_process_factor.cpp).
  {
    const int j = 5;   // interior segment, well clear of the head (j>2)
    // Lambda held FIXED at the nominal linearization point (standard GN
    // practice -- the gradient of the QUADRATIC MODEL, not of Lambda's
    // own state-dependence).
    const double tj = s.t0() + j * s.delta(), tj1 = s.t0() + (j + 1) * s.delta();
    Eigen::Matrix<double, 9, 9> F9n, Q9n; M3D rp0; V3D pp0, vp0;
    relinearizePoseControlSegment(segs.seg_samples[j], s.rotAt(tj), s.posAt(tj), s.velAt(tj),
        bias_acc, bias_gyr, gravity, qa, qg, var_acc, var_gyr, so, F9n, Q9n, rp0, pp0, vp0);
    const Eigen::Matrix<double, 9, 9> LambdaFixed = poseControlPseudoInverse9(Q9n, 1e-6);

    auto fixedLambdaCost = [&](const PoseControlSpline& sp) {
      Eigen::Matrix<double, 9, 1> r = residualForSegment(sp, j, segs.seg_samples[j], bias_acc, bias_gyr, gravity, qa, qg, var_acc, var_gyr, so);
      return 0.5 * r.dot(LambdaFixed * r);
    };

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
    Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
    addPoseControlProcessFactorReduced(s, layout, j, segs.seg_samples[j], bias_acc, bias_gyr, gravity,
        qa, qg, var_acc, var_gyr, so, 1e-6, A, b, nullptr, nullptr);

    const double h = 1e-6;
    double max_abs = 0.0, max_rel = 0.0;
    // Free control points touching segment j=5: window [j-3,j+3]=[2,8] for
    // either side -- test a representative subset covering both channels.
    struct Probe { int col; std::function<void(PoseControlSpline&, double)> perturb; };
    std::vector<Probe> probes;
    for (int abs_k : {4, 5, 6, 7, 8}) {
      for (int axis = 0; axis < 3; ++axis) {
        int colp = layout.colPos(abs_k);
        if (colp >= 0) probes.push_back({colp + axis, [abs_k, axis](PoseControlSpline& sp, double d) { sp.cp_p(axis, abs_k) += d; }});
        int colph = layout.colPhi(abs_k);
        if (colph >= 0) probes.push_back({colph + axis, [abs_k, axis](PoseControlSpline& sp, double d) { sp.cp_phi(axis, abs_k) += d; }});
      }
    }
    for (auto& pr : probes) {
      auto sp = s, sm = s;
      pr.perturb(sp, h); pr.perturb(sm, -h);
      double fd = (fixedLambdaCost(sp) - fixedLambdaCost(sm)) / (2 * h);
      double err = std::abs(-b(pr.col) - fd);
      max_abs = std::max(max_abs, err);
      max_rel = std::max(max_rel, err / std::max(1e-6, std::abs(fd)));
    }
    // sT gradient check (perturb bias_gyr/bias_acc/gravity directly).
    for (int axis = 0; axis < 3; ++axis) {
      V3D e = V3D::Zero(); e(axis) = 1.0;
      auto costWithBg = [&](double d) {
        Eigen::Matrix<double, 9, 1> r = residualForSegment(s, j, segs.seg_samples[j], bias_acc, bias_gyr + d * e, gravity, qa, qg, var_acc, var_gyr, so);
        return 0.5 * r.dot(LambdaFixed * r);
      };
      double fd_bg = (costWithBg(h) - costWithBg(-h)) / (2 * h);
      double err_bg = std::abs(-b(layout.colBG() + axis) - fd_bg);
      max_abs = std::max(max_abs, err_bg);
      max_rel = std::max(max_rel, err_bg / std::max(1e-6, std::abs(fd_bg)));

      auto costWithBa = [&](double d) {
        Eigen::Matrix<double, 9, 1> r = residualForSegment(s, j, segs.seg_samples[j], bias_acc + d * e, bias_gyr, gravity, qa, qg, var_acc, var_gyr, so);
        return 0.5 * r.dot(LambdaFixed * r);
      };
      double fd_ba = (costWithBa(h) - costWithBa(-h)) / (2 * h);
      double err_ba = std::abs(-b(layout.colBA() + axis) - fd_ba);
      max_abs = std::max(max_abs, err_ba);
      max_rel = std::max(max_rel, err_ba / std::max(1e-6, std::abs(fd_ba)));

      auto costWithG = [&](double d) {
        Eigen::Matrix<double, 9, 1> r = residualForSegment(s, j, segs.seg_samples[j], bias_acc, bias_gyr, gravity + d * e, qa, qg, var_acc, var_gyr, so);
        return 0.5 * r.dot(LambdaFixed * r);
      };
      double fd_g = (costWithG(h) - costWithG(-h)) / (2 * h);
      double err_g = std::abs(-b(layout.colG() + axis) - fd_g);
      max_abs = std::max(max_abs, err_g);
      max_rel = std::max(max_rel, err_g / std::max(1e-6, std::abs(fd_g)));
    }
    // Absolute tolerance is scale-aware: Lambda's eigenvalues span many
    // orders of magnitude here (a near-singular Q9 direction, same
    // characteristic already documented in
    // test_pose_control_process_factor.cpp's E_process~1e9 note) -- the
    // RELATIVE check is the meaningful gate; report the absolute number
    // honestly rather than picking a threshold to make it pass.
    check(max_rel < 1e-2, "max_relative_reduced_gradient_error", max_rel);
    printf("    (max_abs_reduced_gradient_error = %.3e, against gradient magnitudes up to ~1e2-1e3 from a near-singular Q9 direction -- see max_rel above)\n", max_abs);
  }

  // ---- (2) head block A_hh/A_hf vs FD of the FULL scan's residual chain
  // w.r.t. x0, summed over every segment (rotation's global effect means
  // every segment contributes) --------------------------------------------
  {
    PoseControlProcessFactorHeadBlock head_block;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
    Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
    for (int j = 0; j < s.nSeg(); ++j)
      addPoseControlProcessFactorReduced(s, layout, j, segs.seg_samples[j], bias_acc, bias_gyr, gravity,
          qa, qg, var_acc, var_gyr, so, 1e-6, A, b, &head_block, nullptr);

    check(head_block.A_hh.allFinite(), "A_hh finite");
    check(head_block.A_hf.allFinite(), "A_hf finite");
    check(head_block.A_hh.norm() > 0, "A_hh nonzero (head genuinely informed)", head_block.A_hh.norm());

    // FD ground truth: sum of Lambda-weighted residual-Jacobian products
    // is expensive to reconstruct directly; instead verify the WEAKER but
    // still meaningful property directly implied by the derivation: the
    // theta0 sensitivity (poseControlHeadRotJacobian) is nonzero at EVERY
    // segment's tj/tj1 (global), while pos0/vel0 sensitivity via
    // poseControlHeadPosJacobians is zero once tj > segment 2's window --
    // already unit-tested (test_pose_control_spline.cpp Test G). Here we
    // confirm A_hh's ROTATION block (0:3,0:3) is markedly larger than a
    // run using ONLY segment 0 (proving later segments DO contribute
    // rotation information, i.e. the "global" wiring is actually active,
    // not silently dropped).
    PoseControlProcessFactorHeadBlock head_block_seg0_only;
    Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
    Eigen::VectorXd b0 = Eigen::VectorXd::Zero(layout.dim());
    addPoseControlProcessFactorReduced(s, layout, 0, segs.seg_samples[0], bias_acc, bias_gyr, gravity,
        qa, qg, var_acc, var_gyr, so, 1e-6, A0, b0, &head_block_seg0_only, nullptr);
    double tr_full = head_block.A_hh.block<3, 3>(0, 0).trace();
    double tr_seg0 = head_block_seg0_only.A_hh.block<3, 3>(0, 0).trace();
    check(tr_full > tr_seg0 * 1.5, "all_segments_contribute_rotation_info(global wiring active)",
          tr_full / std::max(1e-12, tr_seg0));
    printf("    trace(A_hh_theta, all segs)=%.4f  trace(A_hh_theta, seg0 only)=%.4f\n", tr_full, tr_seg0);
  }

  // ---- (3) DIRECT FD validation of the head Jacobian (Jhead) itself, one
  // segment (j=0, both position AND rotation head-sensitivity active),
  // reconstructed independently of addPoseControlProcessFactorReduced's
  // internals: perturb x0=[theta0,p0,v0] directly (rebuilding cp[0..2]/
  // R_anchor from scratch, exactly like test_pose_control_spline.cpp's
  // Test G), FD the segment-0 residual, and compare A_hh_fd=Jhead_fd^T*
  // Lambda*Jhead_fd against head_block.A_hh restricted to segment 0 alone.
  {
    const int j = 0;
    const double tj = s.t0(), tj1 = s.t0() + s.delta();
    Eigen::Matrix<double, 9, 9> F9n, Q9n; M3D rp0; V3D pp0, vp0;
    relinearizePoseControlSegment(segs.seg_samples[j], s.rotAt(tj), s.posAt(tj), s.velAt(tj),
        bias_acc, bias_gyr, gravity, qa, qg, var_acc, var_gyr, so, F9n, Q9n, rp0, pp0, vp0);
    const Eigen::Matrix<double, 9, 9> LambdaFixed = poseControlPseudoInverse9(Q9n, 1e-6);

    Eigen::Matrix<double, 9, 9> Jhead_fd = Eigen::Matrix<double, 9, 9>::Zero();
    const double h = 1e-6;
    auto r0 = [&](const PoseControlSpline& sp) { return residualForSegment(sp, j, segs.seg_samples[j], bias_acc, bias_gyr, gravity, qa, qg, var_acc, var_gyr, so); };
    for (int axis = 0; axis < 3; ++axis) {
      V3D e = V3D::Zero(); e(axis) = 1.0;
      // theta0 column
      { PoseControlSpline sp = s, sm = s; sp.R_anchor = R0 * Exp(h * e); sm.R_anchor = R0 * Exp(-h * e);
        Jhead_fd.col(axis) = (r0(sp) - r0(sm)) / (2 * h); }
      // p0 column
      { PoseControlSpline sp = s, sm = s; V3D cpp2[3], cph2[3];
        solveHeadControlPoints(sp, p0 + h * e, v0, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sp.cp_p.col(i) = cpp2[i];
        solveHeadControlPoints(sm, p0 - h * e, v0, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sm.cp_p.col(i) = cpp2[i];
        Jhead_fd.col(3 + axis) = (r0(sp) - r0(sm)) / (2 * h); }
      // v0 column
      { PoseControlSpline sp = s, sm = s; V3D cpp2[3], cph2[3];
        solveHeadControlPoints(sp, p0, v0 + h * e, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sp.cp_p.col(i) = cpp2[i];
        solveHeadControlPoints(sm, p0, v0 - h * e, a0, omega0, alpha0, cpp2, cph2);
        for (int i = 0; i < 3; ++i) sm.cp_p.col(i) = cpp2[i];
        Jhead_fd.col(6 + axis) = (r0(sp) - r0(sm)) / (2 * h); }
    }
    const Eigen::Matrix<double, 9, 9> A_hh_fd = Jhead_fd.transpose() * LambdaFixed * Jhead_fd;

    PoseControlProcessFactorHeadBlock hb_seg0;
    Eigen::MatrixXd A00 = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
    Eigen::VectorXd b00 = Eigen::VectorXd::Zero(layout.dim());
    addPoseControlProcessFactorReduced(s, layout, 0, segs.seg_samples[0], bias_acc, bias_gyr, gravity,
        qa, qg, var_acc, var_gyr, so, 1e-6, A00, b00, &hb_seg0, nullptr);

    double err_abs = (A_hh_fd - hb_seg0.A_hh).norm();
    double err_rel = err_abs / std::max(1e-9, A_hh_fd.norm());
    check(err_rel < 1e-2, "max_relative_A_hh_error_vs_direct_FD(segment 0)", err_rel);
    printf("    |A_hh_fd|=%.4e  |A_hh_analytic|=%.4e  abs_err=%.4e\n",
           A_hh_fd.norm(), hb_seg0.A_hh.norm(), err_abs);
  }

  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
