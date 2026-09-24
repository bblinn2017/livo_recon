// Targeted validation phase, items 9-13: does the pose-control spline
// represent the SAME physical trajectory the same way regardless of N?
//
// Methodology: fit a fixed physically-defined trajectory p(t)/phi(t) (a
// smooth, nontrivial 6-DOF motion, NOT zero-motion) to the actual
// production PoseControlSpline basis at N=4,7,13 via linear least squares
// (the basis weights b_k(t)/db_k(t)/ddb_k(t) are read directly off
// spline.jacobianAt(t), the SAME primitives production code uses -- no
// separate basis implementation). hns.Z = Identity (no head elimination)
// throughout -- these tests are about the raw spline's own parameterization
// behavior, not the head-constraint machinery (already covered elsewhere).
//
// Outputs are printed to stdout (parsed by the campaign's aggregation
// script into parameterization_invariance/covariance_invariance/
// information_invariance/q_invariance CSV rows) and self-checked with
// generous, physically-motivated bounds (this is a MEASUREMENT tool first,
// a pass/fail gate second -- see the report for interpretation).
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_physical_diagnostics.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  std::printf("  [%s] %-58s %+.6e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) ++g_fail;
}

// The one physically-defined trajectory every N fits independently.
static V3D posFunc(double t) {
  return V3D(0.5 * t + 0.05 * std::sin(2 * M_PI * 5 * t),
             0.02 * std::cos(2 * M_PI * 5 * t),
             0.01 * std::sin(2 * M_PI * 3 * t));
}
static V3D phiFunc(double t) {
  return V3D(0.10 * std::sin(2 * M_PI * 4 * t),
             0.05 * std::cos(2 * M_PI * 4 * t),
             0.02 * std::sin(2 * M_PI * 2 * t));
}

// Least-squares fit of cp_p/cp_phi (independently, per-dimension decoupled
// basis) against posFunc/phiFunc sampled densely over [t0,t1], using
// spline.jacobianAt(t)'s own basis weights b[k] as the design matrix -- the
// SAME basis production code evaluates with, not a re-derivation.
static PoseControlSpline fitTrajectory(int N, double t0, double t1) {
  PoseControlSpline s;
  s.init(N, t0, t1);
  s.R_anchor = M3D::Identity();

  const int M = 400;  // dense sample count for the LS fit
  Eigen::MatrixXd Bmat = Eigen::MatrixXd::Zero(M, N);
  Eigen::MatrixXd Ppos(M, 3), Pphi(M, 3);
  for (int i = 0; i < M; ++i) {
    const double t = t0 + (t1 - t0) * (i + 0.5) / M;
    const auto jac = s.jacobianAt(t);
    for (int k = 0; k < 4; ++k) Bmat(i, jac.s + k) = jac.b[k];
    Ppos.row(i) = posFunc(t).transpose();
    Pphi.row(i) = phiFunc(t).transpose();
  }
  const Eigen::MatrixXd BtB = Bmat.transpose() * Bmat;
  const Eigen::MatrixXd BtB_pinv = generalPseudoInverse(BtB, 1e-9);
  const Eigen::MatrixXd cp_p_sol = BtB_pinv * Bmat.transpose() * Ppos;    // N x 3
  const Eigen::MatrixXd cp_phi_sol = BtB_pinv * Bmat.transpose() * Pphi; // N x 3
  for (int k = 0; k < N; ++k) {
    s.cp_p.col(k) = cp_p_sol.row(k).transpose();
    s.cp_phi.col(k) = cp_phi_sol.row(k).transpose();
  }
  return s;
}

static PoseControlHeadNullspace identityHns(int N) {
  PoseControlHeadNullspace hns;
  hns.Z = Eigen::MatrixXd::Identity(6 * N, 6 * N);
  hns.c_particular = Eigen::VectorXd::Zero(6 * N);
  return hns;
}

int main() {
  std::printf("Parameterization-invariance measurement (items 9-13)\n");
  const double t0 = 0.0, t1 = 0.1;  // matches real per-scan window (~0.1s)
  const std::vector<int> Ns = {4, 7, 13};
  const V3D gravity(0, 0, -9.81);

  struct Fit { int N; PoseControlSpline spline; PoseControlFreeLayout layout; PoseControlHeadNullspace hns; double delta; };
  std::vector<Fit> fits;
  for (int N : Ns) {
    Fit f;
    f.N = N;
    f.spline = fitTrajectory(N, t0, t1);
    f.layout.N = N; f.layout.has_bg = f.layout.has_ba = f.layout.has_g = true; f.layout.fix_head = false;
    f.hns = identityHns(N);
    f.delta = f.spline.delta();
    fits.push_back(f);
  }

  // ---- item 11: knot-spacing scaling -------------------------------------
  std::printf("\n-- item 11: knot spacing / Jacobian scaling --\n");
  for (auto& f : fits) {
    const auto jac = f.spline.jacobianAt(0.5 * (t0 + t1));
    const double expected_inv_delta2 = 1.0 / (f.delta * f.delta);
    std::printf("  N=%2d  Delta_t=%.6f  inv_delta=%.4f  inv_delta2=%.4f  expected_inv_delta2=%.4f\n",
                f.N, f.delta, jac.inv_delta, jac.inv_delta2, expected_inv_delta2);
    check(std::abs(jac.inv_delta2 - expected_inv_delta2) < 1e-9 * expected_inv_delta2,
          "inv_delta2 == 1/Delta_t^2 exactly (production scaling)", jac.inv_delta2 - expected_inv_delta2);
  }

  // ---- item 9/10: parameterization invariance + FD-validated Jacobians --
  std::printf("\n-- item 9/10: physical sample invariance + Jacobian FD --\n");
  const int n_probe = 15;
  std::vector<std::vector<PoseControlPhysicalSample>> samples_by_fit(fits.size());
  for (size_t fi = 0; fi < fits.size(); ++fi) {
    auto& f = fits[fi];
    for (int i = 0; i < n_probe; ++i) {
      const double t = t0 + (t1 - t0) * (i + 0.5) / n_probe;
      samples_by_fit[fi].push_back(evaluatePoseControlPhysicalSample(f.spline, f.layout, f.hns, t, gravity));
    }
  }
  // Cross-N comparison at shared physical times (N=4 vs N=13, the extremes).
  double max_dp = 0, max_dv = 0, max_da = 0, max_domega = 0;
  for (int i = 0; i < n_probe; ++i) {
    const auto& s4 = samples_by_fit[0][i];
    const auto& s13 = samples_by_fit[2][i];
    max_dp = std::max(max_dp, (s4.p - s13.p).norm());
    max_dv = std::max(max_dv, (s4.v - s13.v).norm());
    max_da = std::max(max_da, (s4.a - s13.a).norm());
    max_domega = std::max(max_domega, (s4.omega - s13.omega).norm());
  }
  std::printf("  max|p_4-p_13|=%.6e  max|v_4-v_13|=%.6e  max|a_4-a_13|=%.6e  max|omega_4-omega_13|=%.6e\n",
              max_dp, max_dv, max_da, max_domega);
  // Physically-motivated bounds: the underlying trajectory has position
  // amplitude ~0.05m, velocity ~1.6 m/s peak (dominant 0.5 m/s ramp term),
  // acceleration ~50 m/s^2 peak (2*pi*5)^2*0.05 -- N=4 (1 single cubic
  // segment across the whole 0.1s window) is a genuinely coarse
  // representation of a 5Hz oscillation and is EXPECTED to disagree with
  // N=13 by an appreciable fraction of the signal's own amplitude; the
  // failure mode this test exists to catch is acceleration disagreement
  // many ORDERS OF MAGNITUDE beyond the signal's own scale (a parameterization
  // bug), not an honest representation-error gap between a 1-segment and a
  // 10-segment fit of a 5Hz signal.
  check(max_dp < 0.2, "position disagreement stays within a fraction of trajectory amplitude (no gross bug)", max_dp);
  check(max_da < 500.0, "acceleration disagreement stays within ~10x signal scale, not orders of magnitude", max_da);

  // Jacobian FD validation, one representative fit (N=13) -- central
  // difference on a raw control-point perturbation vs evaluatePoseControlPhysicalSample's
  // analytic dp/dv/da/domega_deta (item 10).
  {
    auto& f = fits[2];
    const double t_probe = 0.5 * (t0 + t1);
    const auto sample0 = evaluatePoseControlPhysicalSample(f.spline, f.layout, f.hns, t_probe, gravity);
    const double eps = 1e-6;
    double max_dp_err = 0, max_dv_err = 0, max_da_err = 0, max_domega_err = 0;
    const int rawDim = f.hns.rawDim();
    for (int col = 0; col < rawDim; col += std::max(1, rawDim / 20)) {
      PoseControlSpline sp = f.spline, sm = f.spline;
      const int half = rawDim / 2;
      if (col < half) { sp.cp_p.col(col / 3)(col % 3) += eps; sm.cp_p.col(col / 3)(col % 3) -= eps; }
      else { const int c2 = col - half; sp.cp_phi.col(c2 / 3)(c2 % 3) += eps; sm.cp_phi.col(c2 / 3)(c2 % 3) -= eps; }
      const V3D fd_p = (sp.posAt(t_probe) - sm.posAt(t_probe)) / (2 * eps);
      const V3D fd_v = (sp.velAt(t_probe) - sm.velAt(t_probe)) / (2 * eps);
      const V3D fd_a = (sp.accAt(t_probe) - sm.accAt(t_probe)) / (2 * eps);
      const V3D fd_omega = (sp.omegaBodyAt(t_probe) - sm.omegaBodyAt(t_probe)) / (2 * eps);
      max_dp_err = std::max(max_dp_err, (fd_p - sample0.dp_deta.col(col)).norm());
      max_dv_err = std::max(max_dv_err, (fd_v - sample0.dv_deta.col(col)).norm());
      max_da_err = std::max(max_da_err, (fd_a - sample0.da_deta.col(col)).norm());
      max_domega_err = std::max(max_domega_err, (fd_omega - sample0.domega_deta.col(col)).norm());
    }
    check(max_dp_err < 1e-4, "max_dp_dc_fd_error", max_dp_err);
    check(max_dv_err < 1e-3, "max_dv_dc_fd_error", max_dv_err);
    check(max_da_err < 1e-1, "max_da_dc_fd_error", max_da_err);
    check(max_domega_err < 1e-3, "max_domega_dc_fd_error", max_domega_err);
  }

  // ---- item 12/13: physical covariance/information invariance -----------
  // Build the PRODUCTION scan-start prior for each N from a SYNTHETIC IMU
  // stream generated FROM THE SAME fitted trajectory (so all three N's
  // priors are built from physically-identical acceleration/gyro data,
  // differing only in how each N's spline represents/absorbs it), then map
  // the resulting raw-space covariance into physical p/v/a/omega covariance
  // via dp_deta etc, and compare across N at a shared physical time.
  std::printf("\n-- item 12/13: physical covariance/information invariance --\n");
  const V3D bias_acc = V3D::Zero(), bias_gyr = V3D::Zero();
  std::vector<double> trace_Pa_by_N, trace_Pomega_by_N;
  for (auto& f : fits) {
    // Synthetic 200Hz IMU stream implied by the fitted trajectory itself
    // (a_meas = R(t)^T*(a(t)-g), omega_meas = omega(t) -- zero-noise
    // "perfect" measurements, since this test is about the COORDINATE
    // TRANSFORMATION, not sensor noise).
    std::vector<ImuSample> imu;
    for (double t = t0; t <= t1 + 1e-9; t += 1.0 / 200.0) {
      ImuSample s;
      s.t = t;
      s.acc = f.spline.rotAt(t).transpose() * (f.spline.accAt(t) - gravity);
      s.gyro = f.spline.omegaBodyAt(t);
      imu.push_back(s);
    }
    const int dimRaw = f.layout.dim();
    Eigen::MatrixXd A_raw = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
    Eigen::VectorXd b_raw = Eigen::VectorXd::Zero(dimRaw);
    const V3D var_acc(1e-4, 1e-4, 1e-4), var_gyr(1e-6, 1e-6, 1e-6);
    buildPoseControlContinuousImuPrior(f.spline, f.layout, imu, bias_acc, bias_gyr, gravity,
        var_acc, var_gyr, A_raw, b_raw, nullptr, nullptr);
    const Eigen::MatrixXd P_raw = generalPseudoInverse(A_raw, 1e-9);
    const double t_probe = 0.5 * (t0 + t1);
    const auto sample = evaluatePoseControlPhysicalSample(f.spline, f.layout, f.hns, t_probe, gravity);
    const Eigen::MatrixXd P_c = P_raw.topLeftCorner(6 * f.N, 6 * f.N);  // control-point block only (raw, no sT here)
    const Eigen::Matrix3d P_p = sample.dp_deta * P_c * sample.dp_deta.transpose();
    const Eigen::Matrix3d P_v = sample.dv_deta * P_c * sample.dv_deta.transpose();
    const Eigen::Matrix3d P_a = sample.da_deta * P_c * sample.da_deta.transpose();
    const Eigen::Matrix3d P_omega = sample.domega_deta * P_c * sample.domega_deta.transpose();
    trace_Pa_by_N.push_back(P_a.trace());
    trace_Pomega_by_N.push_back(P_omega.trace());
    std::printf("  N=%2d  trace(P_p)=%.6e  trace(P_v)=%.6e  trace(P_a)=%.6e  trace(P_omega)=%.6e  trace(P_control_raw)=%.6e\n",
                f.N, P_p.trace(), P_v.trace(), P_a.trace(), P_omega.trace(), P_c.trace());

    // item 15: the prior MEAN, mapped to physical p(t_probe), should be a
    // near-zero correction at every N -- these IMU samples were generated
    // FROM f.spline itself (zero noise), so f.spline is ALREADY its own
    // IMU-only optimum; z_prior's correction should confirm that (not
    // silently return some other, spurious nonzero shift) regardless of N.
    const Eigen::MatrixXd A_c = A_raw.topLeftCorner(6 * f.N, 6 * f.N);
    const Eigen::VectorXd b_c = b_raw.head(6 * f.N);
    const Eigen::VectorXd delta_c_prior = generalPseudoInverse(A_c, 1e-6) * b_c;
    const V3D delta_p_prior = sample.dp_deta * delta_c_prior;
    const V3D delta_a_prior = sample.da_deta * delta_c_prior;
    std::printf("         prior-mean physical correction at t_probe: |delta_p|=%.6e  |delta_a|=%.6e\n",
                delta_p_prior.norm(), delta_a_prior.norm());
    check(delta_p_prior.norm() < 1e-6, "item 15: prior-mean position correction ~0 (spline already IMU-optimal) at this N", delta_p_prior.norm());
  }
  const double pa_ratio_13_over_4 = trace_Pa_by_N[0] > 1e-300 ? trace_Pa_by_N[2] / trace_Pa_by_N[0] : 0.0;
  const double pomega_ratio_13_over_4 = trace_Pomega_by_N[0] > 1e-300 ? trace_Pomega_by_N[2] / trace_Pomega_by_N[0] : 0.0;
  std::printf("  physical_information_invariance: trace(P_a)_N13/trace(P_a)_N4 = %.4e, trace(P_omega)_N13/trace(P_omega)_N4 = %.4e\n",
              pa_ratio_13_over_4, pomega_ratio_13_over_4);
  // Report only for the ratio itself -- no a priori "correct" value.
  // item 40 (N=13 pathology regression, continuous-prior phase): the
  // production prior's own physical acceleration covariance must remain
  // FINITE and NOT unboundedly large at N=13 -- this is a genuine gate,
  // unlike the ratio above.
  for (size_t i = 0; i < fits.size(); ++i) {
    check(std::isfinite(trace_Pa_by_N[i]) && trace_Pa_by_N[i] < 1e6,
          "item 40 regression: trace(P_a) is finite and physically bounded (N=4/7/13)", trace_Pa_by_N[i]);
  }

  std::printf("\n%d failed (measurement checks above; see printed ratios for the invariance answer)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
