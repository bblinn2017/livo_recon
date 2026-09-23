// Spec item 13, 2026-09-22: "For one scan and one segment, independently
// verify the process residual Jacobian against a finite-difference
// derivative of the WHOLE nonlinear residual... Do not proceed to a bag if
// this fails." FD is the validation ORACLE here only -- the process
// factor itself (pose_control_process_factor.cpp) contains no FD.
#include "livo_recon/lio/pose_control_process_factor.h"
#include <cstdio>
#include <random>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  printf("  [%s] %-48s %.3e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) g_fail++;
}

// Synthetic 200 Hz IMU stream over one segment window, gravity + a
// realistic small rotation rate/acceleration -- not zero, so F9/Q9 are
// genuinely nontrivial (a zero-motion stream would make F9 trivially
// near-identity and mask a real Jacobian bug).
static std::vector<ImuSample> makeImuStream(double t0, double t1, const V3D& gravity) {
  std::vector<ImuSample> s;
  const double hz = 200.0;
  const int n = std::max(2, static_cast<int>((t1 - t0) * hz));
  for (int i = 0; i <= n; ++i) {
    double t = t0 + (t1 - t0) * i / n;
    V3D gyro(0.3 * std::sin(7.0 * t), -0.2 * std::cos(5.0 * t), 0.15);
    V3D specific_force = V3D(0.5, -0.3, 0.2) - gravity;  // body accel + gravity comp (bias/grav both 0 here)
    s.emplace_back(specific_force, gyro, t);
  }
  return s;
}

// Nonlinear residual r_j(c) for segment j, recomputed from scratch for an
// ARBITRARY (possibly perturbed) spline -- used only by this FD oracle.
static Eigen::Matrix<double, 9, 1> residualAt(
    const PoseControlSpline& spline, int j, const std::vector<ImuSample>& samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr, const V3D& var_acc, const V3D& var_gyr, bool second_order) {
  const double tj = spline.t0() + j * spline.delta();
  const double tj1 = spline.t0() + (j + 1) * spline.delta();
  const M3D Rj = spline.rotAt(tj);
  const V3D pj = spline.posAt(tj), vj = spline.velAt(tj);
  const M3D Rj1 = spline.rotAt(tj1);
  const V3D pj1 = spline.posAt(tj1), vj1 = spline.velAt(tj1);
  Eigen::Matrix<double, 9, 9> F9, Q9;
  M3D rot_pred; V3D pos_pred, vel_pred;
  relinearizePoseControlSegment(samples, Rj, pj, vj, bias_acc, bias_gyr, gravity,
                                q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order,
                                F9, Q9, rot_pred, pos_pred, vel_pred);
  Eigen::Matrix<double, 9, 1> r;
  r.segment<3>(0) = Log(M3D(rot_pred.transpose() * Rj1));
  r.segment<3>(3) = pj1 - pos_pred;
  r.segment<3>(6) = vj1 - vel_pred;
  return r;
}

int main() {
  printf("Process-factor Jacobian FD validation (spec item 13)\n");
  const int N = 13;
  const double t0 = 0.0, t1 = 0.1;
  PoseControlSpline s; s.init(N, t0, t1);
  std::mt19937 rng(2024);
  std::uniform_real_distribution<double> up(-0.3, 0.3), uphi(-0.05, 0.05);
  for (int i = 0; i < N; ++i) {
    s.cp_p.col(i) = V3D(i * 0.05, 0.0, 0.0) + V3D(up(rng), up(rng), up(rng));
    s.cp_phi.col(i) = V3D(uphi(rng), uphi(rng), uphi(rng));
  }
  s.R_anchor = M3D::Identity();

  const V3D bias_acc = V3D(0.01, -0.02, 0.005);
  const V3D bias_gyr = V3D(0.001, 0.002, -0.0015);
  const V3D gravity(0, 0, -9.81);
  const V3D var_acc(0.01, 0.01, 0.01), var_gyr(0.001, 0.001, 0.001);
  const double q_alpha_acc = 1.0, q_alpha_gyr = 1.0;
  const bool second_order = true;

  auto segs = bucketPoseControlImuSamples(makeImuStream(t0, t1, gravity), s);
  check(static_cast<int>(segs.seg_samples.size()) == s.nSeg(), "n_seg buckets built",
        static_cast<double>(segs.seg_samples.size()));

  // Test on an interior segment (has full 4-control-point support on both
  // sides, not truncated by the array boundary).
  const int j = 4;
  const auto& samples = segs.seg_samples[j];
  check(samples.size() >= 2, "segment has >=2 samples", static_cast<double>(samples.size()));

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6 * N, 6 * N);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * N);
  double E_process = 0.0;
  addPoseControlProcessFactor(s, j, samples, bias_acc, bias_gyr, gravity,
                              q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order,
                              1e-6, A, b, &E_process);
  check(std::isfinite(E_process) && E_process >= 0.0, "E_process finite/nonneg", E_process);

  // Reconstruct Jcur (9 x 6N) exactly as addPoseControlProcessFactor()
  // built it internally, by re-deriving via the SAME public pieces (F9,
  // Lambda, Jxj/Jxj1) so this test doesn't just re-check its own math --
  // it FD-validates the analytic Jcur against the actual residual.
  const double tj = s.t0() + j * s.delta(), tj1 = s.t0() + (j + 1) * s.delta();
  Eigen::Matrix<double, 9, 9> F9, Q9;
  M3D rot_pred; V3D pos_pred, vel_pred;
  relinearizePoseControlSegment(samples, s.rotAt(tj), s.posAt(tj), s.velAt(tj),
                                bias_acc, bias_gyr, gravity, q_alpha_acc, q_alpha_gyr,
                                var_acc, var_gyr, second_order, F9, Q9, rot_pred, pos_pred, vel_pred);
  const auto jac_j = s.jacobianAt(tj), jac_j1 = s.jacobianAt(tj1);
  Eigen::MatrixXd Jxj = Eigen::MatrixXd::Zero(9, 6 * N), Jxj1 = Eigen::MatrixXd::Zero(9, 6 * N);
  for (int k = 0; k < 4; ++k) {
    const int cj = jac_j.s + k, cj1 = jac_j1.s + k;
    Jxj.block<3, 3>(0, 3 * N + 3 * cj) = s.dThetaDcphi(jac_j, k, tj);
    Jxj.block<3, 3>(3, 3 * cj) = PoseControlSpline::dPosDcp(jac_j, k);
    Jxj.block<3, 3>(6, 3 * cj) = PoseControlSpline::dVelDcp(jac_j, k);
    Jxj1.block<3, 3>(0, 3 * N + 3 * cj1) = s.dThetaDcphi(jac_j1, k, tj1);
    Jxj1.block<3, 3>(3, 3 * cj1) = PoseControlSpline::dPosDcp(jac_j1, k);
    Jxj1.block<3, 3>(6, 3 * cj1) = PoseControlSpline::dVelDcp(jac_j1, k);
  }
  Eigen::MatrixXd Jcur = Jxj1 - F9 * Jxj;

  const Eigen::Matrix<double, 9, 1> r0 = residualAt(s, j, samples, bias_acc, bias_gyr, gravity,
                                                     q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order);
  // Theta-row boxminus correction (see pose_control_process_factor.cpp's
  // in-file comment on this exact derivation) -- reimplemented here (not
  // exported) purely for this test's independent reconstruction of Jcur.
  {
    const V3D phi = r0.segment<3>(0);
    const double n = phi.norm();
    M3D K; K << 0, -phi.z(), phi.y(), phi.z(), 0, -phi.x(), -phi.y(), phi.x(), 0;
    double c = (n < 1e-4) ? (1.0 / 12.0 + n * n / 720.0)
                          : (1.0 / (n * n) - (1.0 + std::cos(n)) / (2.0 * n * std::sin(n)));
    M3D JrInv = M3D::Identity() + 0.5 * K + c * K * K;
    const M3D g = rot_pred.transpose() * s.rotAt(tj1);
    const Eigen::MatrixXd theta_row_j  = F9.block<3, 9>(0, 0) * Jxj;
    const Eigen::MatrixXd theta_row_j1 = Jxj1.block(0, 0, 3, Jxj1.cols());
    Jcur.block(0, 0, 3, Jcur.cols()) = JrInv * (theta_row_j1 - g.transpose() * theta_row_j);
  }

  double max_abs = 0.0, max_rel = 0.0;
  const double h = 1e-6;
  // Only columns with a nonzero Jcur column are expected to move the
  // residual (support-limited) -- but sweep ALL 6N columns anyway so an
  // unexpectedly-nonzero FD column outside Jcur's support would be caught.
  for (int col = 0; col < 6 * N; ++col) {
    PoseControlSpline sp = s, sm = s;
    if (col < 3 * N) { sp.cp_p(col % 3, col / 3) += h; sm.cp_p(col % 3, col / 3) -= h; }
    else { int cc = col - 3 * N; sp.cp_phi(cc % 3, cc / 3) += h; sm.cp_phi(cc % 3, cc / 3) -= h; }
    Eigen::Matrix<double, 9, 1> rp = residualAt(sp, j, samples, bias_acc, bias_gyr, gravity,
                                                 q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order);
    Eigen::Matrix<double, 9, 1> rm = residualAt(sm, j, samples, bias_acc, bias_gyr, gravity,
                                                 q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order);
    Eigen::Matrix<double, 9, 1> fd_col = (rp - rm) / (2 * h);
    Eigen::Matrix<double, 9, 1> an_col = Jcur.col(col);
    double err = (an_col - fd_col).norm();
    max_abs = std::max(max_abs, err);
    max_rel = std::max(max_rel, err / std::max(1e-6, fd_col.norm()));
  }
  check(max_abs < 1e-3, "max_abs_process_jacobian_error", max_abs);
  check(max_rel < 1e-2, "max_relative_process_jacobian_error", max_rel);

  // ---- G9 bias/gravity Jacobian FD validation --------------------------
  // The only channel by which measurement information can reach the free
  // tail state sT=[bg,ba,g] in this architecture (the head correction is
  // fixed at exactly zero, so decoupled LIO's usual pose-correction-via-
  // prior-cross-covariance pathway to bg/ba/g does not apply here).
  printf("\nG9 (bias/gravity process-factor Jacobian) FD validation\n");
  {
    Eigen::Matrix<double, 9, 9> F9b, Q9b, G9;
    M3D rp; V3D pp, vp;
    relinearizePoseControlSegmentWithBiasJac(samples, s.rotAt(tj), s.posAt(tj), s.velAt(tj),
        bias_acc, bias_gyr, gravity, q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order,
        F9b, Q9b, G9, rp, pp, vp);

    double max_abs_g9 = 0.0, max_rel_g9 = 0.0;
    const double hb = 1e-7;
    for (int col = 0; col < 9; ++col) {
      V3D e = V3D::Zero(); e(col % 3) = 1.0;
      V3D bap = bias_acc, bam = bias_acc, bgp = bias_gyr, bgm = bias_gyr, gp = gravity, gm = gravity;
      if (col < 3)      { bgp += hb * e; bgm -= hb * e; }
      else if (col < 6) { bap += hb * e; bam -= hb * e; }
      else              { gp  += hb * e; gm  -= hb * e; }

      Eigen::Matrix<double, 9, 9> Fd, Qd; M3D rot_p, rot_m; V3D pos_p, pos_m, vel_p, vel_m;
      relinearizePoseControlSegment(samples, s.rotAt(tj), s.posAt(tj), s.velAt(tj),
          bap, bgp, gp, q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order, Fd, Qd, rot_p, pos_p, vel_p);
      relinearizePoseControlSegment(samples, s.rotAt(tj), s.posAt(tj), s.velAt(tj),
          bam, bgm, gm, q_alpha_acc, q_alpha_gyr, var_acc, var_gyr, second_order, Fd, Qd, rot_m, pos_m, vel_m);

      Eigen::Matrix<double, 9, 1> fd_col;
      fd_col.segment<3>(0) = Log(M3D(rot_m.transpose() * rot_p)) / (2 * hb);
      fd_col.segment<3>(3) = (pos_p - pos_m) / (2 * hb);
      fd_col.segment<3>(6) = (vel_p - vel_m) / (2 * hb);

      Eigen::Matrix<double, 9, 1> an_col = G9.col(col);
      double err = (an_col - fd_col).norm();
      max_abs_g9 = std::max(max_abs_g9, err);
      max_rel_g9 = std::max(max_rel_g9, err / std::max(1e-6, fd_col.norm()));
    }
    check(max_abs_g9 < 1e-4, "max_abs_G9_bias_jacobian_error", max_abs_g9);
    check(max_rel_g9 < 1e-3, "max_relative_G9_bias_jacobian_error", max_rel_g9);
  }

  printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
