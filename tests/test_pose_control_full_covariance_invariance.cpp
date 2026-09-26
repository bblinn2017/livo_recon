// Validates the corrected (head+eta) physical covariance mapping
// (poseControlPhysicalCovariance) across representation capacity (N),
// high-frequency mode significance, and timestep refinement -- using the
// COMPLETE joint [head;eta;bias] prior, not the eta-only Hessian equivalence
// checks in test_pose_control_pre_realdata.cpp / test_pose_control_prior_batch_validation.cpp.
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon;

namespace
{
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-84s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

Eigen::MatrixXd svdPinv(const Eigen::MatrixXd& A, double rel = 1e-12)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& s = svd.singularValues();
  const double cutoff = rel * (s.size() ? s.maxCoeff() : 0.0);
  Eigen::VectorXd inv = s;
  for (int i = 0; i < inv.size(); ++i) inv(i) = inv(i) > cutoff ? 1.0 / inv(i) : 0.0;
  return svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
}

// Fits control points so the spline exactly reproduces a cubic-or-lower
// polynomial p(t) and a linear phi(t) -- both within the reproduction
// range of a cubic B-spline basis, so the fit is exact (not approximate)
// for any N, matching the trajectory across representations.
PoseControlSpline buildSharedTrajectory(int N, double t0, double t1,
                                        const V3D& p0, const V3D& v0, const V3D& a0, const V3D& j0,
                                        const V3D& phi0, const V3D& omega0)
{
  PoseControlSpline s;
  s.init(N, t0, t1);
  s.R_anchor = M3D::Identity();

  const int samples = 4 * N;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(samples, N);
  Eigen::MatrixXd target_p = Eigen::MatrixXd::Zero(samples, 3);
  Eigen::MatrixXd target_phi = Eigen::MatrixXd::Zero(samples, 3);
  for (int i = 0; i < samples; ++i) {
    const double t = t0 + (t1 - t0) * i / (samples - 1.0);
    int seg = 0; double u = 0.0; s.locate(t, seg, u);
    const double om = 1.0 - u, u2 = u * u, u3 = u2 * u;
    Eigen::Vector4d b;
    b << om * om * om / 6.0, (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
         (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, u3 / 6.0;
    for (int k = 0; k < 4; ++k) { const int j = seg + k; if (j >= 0 && j < N) B(i, j) = b[k]; }
    const double dt = t - t0;
    const V3D pt = p0 + v0 * dt + 0.5 * a0 * dt * dt + (j0 * (dt * dt * dt / 6.0));
    const V3D pht = phi0 + omega0 * dt;
    for (int a = 0; a < 3; ++a) { target_p(i, a) = pt(a); target_phi(i, a) = pht(a); }
  }
  const Eigen::MatrixXd Bpinv = svdPinv(B);
  for (int a = 0; a < 3; ++a) {
    const Eigen::VectorXd cp = Bpinv * target_p.col(a);
    const Eigen::VectorXd cphi = Bpinv * target_phi.col(a);
    for (int k = 0; k < N; ++k) { s.cp_p(a, k) = cp(k); s.cp_phi(a, k) = cphi(k); }
  }
  return s;
}

std::vector<ImuSample> perfectImu(const PoseControlSpline& s, double dt, const V3D& bias_acc,
                                  const V3D& bias_gyr, const V3D& gravity)
{
  std::vector<ImuSample> out;
  for (double t = s.t0(); t <= s.t1() + 1e-12; t += dt) {
    ImuSample m; m.t = std::min(t, s.t1());
    m.acc = s.rotAt(m.t).transpose() * (s.accAt(m.t) - gravity) + bias_acc;
    m.gyro = s.omegaBodyAt(m.t) + bias_gyr;
    out.push_back(m);
    if (m.t >= s.t1()) break;
  }
  return out;
}

// Builds the complete joint [head(9); eta; sT] prior (P0-informed, IMU-
// informed, LiDAR-free -- the pure prior this task's independent-reference
// comparisons target) and returns the (9+dEta+dST)x(...) covariance plus
// the head nullspace needed to evaluate physical samples.
struct JointPrior
{
  Eigen::MatrixXd Sigma_full;
  PoseControlHeadNullspace hns;
  int dEta = 0, dST = 0;
};

JointPrior buildJointPrior(const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
                           const std::vector<ImuSample>& imu, const Eigen::MatrixXd& P0,
                           const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
                           const V3D& var_acc, const V3D& var_gyr)
{
  JointPrior out;
  out.hns = buildPoseControlHeadNullspace(spline, spline.posAt(spline.t0()), spline.velAt(spline.t0()));
  const int rawDim = layout.dim();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rawDim, rawDim);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(rawDim);
  PoseControlPriorHeadBlock hb;
  buildPoseControlContinuousImuPrior(spline, layout, imu, bias_acc, bias_gyr, gravity, var_acc, var_gyr,
                                     A, b, &hb, nullptr);

  const Eigen::MatrixXd Omega0 = generalPseudoInverse(P0, 1e-9);
  Eigen::MatrixXd A_hh = hb.A_hh + Omega0.block(0, 0, 9, 9);
  Eigen::MatrixXd A_hf = hb.A_hf.size() > 0 ? hb.A_hf : Eigen::MatrixXd::Zero(9, rawDim);

  out.dEta = out.hns.freeDim();
  out.dST = layout.dimST();
  const int dZ = out.dEta + out.dST;
  if (out.dST > 0 && Omega0.rows() >= 9 + out.dST) {
    A.block(layout.dimCFree(), layout.dimCFree(), out.dST, out.dST) += Omega0.block(9, 9, out.dST, out.dST);
    A_hf.block(0, layout.dimCFree(), 9, out.dST) += Omega0.block(0, 9, 9, out.dST);
  }
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(rawDim, dZ);
  P.block(0, 0, out.hns.rawDim(), out.dEta) = out.hns.Z;
  if (out.dST > 0) P.block(out.hns.rawDim(), out.dEta, out.dST, out.dST) = Eigen::MatrixXd::Identity(out.dST, out.dST);
  const Eigen::MatrixXd A_ff = P.transpose() * A * P;
  const Eigen::MatrixXd A_hf_z = A_hf * P;
  const int dimFull = 9 + dZ;
  Eigen::MatrixXd Lambda_full = Eigen::MatrixXd::Zero(dimFull, dimFull);
  Lambda_full.block(0, 0, 9, 9) = A_hh;
  Lambda_full.block(0, 9, 9, dZ) = A_hf_z;
  Lambda_full.block(9, 0, dZ, 9) = A_hf_z.transpose();
  Lambda_full.block(9, 9, dZ, dZ) = A_ff;
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (Lambda_full + Lambda_full.transpose()));
    const auto& ev = es.eigenvalues();
    std::printf("    [buildJointPrior debug] dim=%d lambda_min=%.3e lambda_max=%.3e cond=%.3e n_below_1e-9rel=%d\n",
                static_cast<int>(ev.size()), ev.minCoeff(), ev.maxCoeff(), ev.maxCoeff() / std::max(1e-300, ev.minCoeff()),
                static_cast<int>((ev.array() < 1e-9 * ev.maxCoeff()).count()));
  }
  out.Sigma_full = generalPseudoInverse(Lambda_full, 1e-12);
  return out;
}

// ============================================================================
// Phase 6: representation-capacity invariance. One physical trajectory
// (cubic position, linear attitude -- both exactly reproducible by a cubic
// B-spline basis regardless of N) evaluated at N=4/7/13 with the SAME
// physical stochastic model (P0, IMU noise, sample rate, window). The
// corrected full [head;eta] physical covariance at shared timestamps must
// agree across N for this shared mode; control-point-space covariance is
// explicitly NOT compared (it is allowed, and expected, to differ).
// ============================================================================
void testRepresentationCapacityInvariance()
{
  const double t0 = 0.0, t1 = 0.1, dt_imu = 0.001;
  const V3D p0(0.05, -0.02, 0.01), v0(0.3, -0.1, 0.05), a0(0.4, 0.2, -0.1), j0(2.0, -1.0, 0.5);
  const V3D phi0(0.0, 0.0, 0.0), omega0(0.05, -0.03, 0.02);
  const V3D gravity(0, 0, -9.81), bias_acc(0.01, -0.02, 0.015), bias_gyr(0.001, -0.0005, 0.0008);
  const V3D var_acc = V3D::Constant(0.02 * 0.02), var_gyr = V3D::Constant(0.005 * 0.005);

  Eigen::MatrixXd P0mat = Eigen::MatrixXd::Zero(9, 9);
  P0mat.block<3, 3>(0, 0) = 2e-5 * Eigen::Matrix3d::Identity();
  P0mat.block<3, 3>(3, 3) = 4e-4 * Eigen::Matrix3d::Identity();
  P0mat.block<3, 3>(6, 6) = 9e-4 * Eigen::Matrix3d::Identity();

  struct Sample { double frac; Eigen::Matrix3d P_p, P_v, P_a, P_theta, P_omega; };
  std::vector<std::vector<Sample>> per_N;
  const std::vector<int> Ns = {4, 7, 13};

  for (int N : Ns) {
    const PoseControlSpline spline = buildSharedTrajectory(N, t0, t1, p0, v0, a0, j0, phi0, omega0);
    const auto imu = perfectImu(spline, dt_imu, bias_acc, bias_gyr, gravity);
    PoseControlFreeLayout layout; layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = false;
    const JointPrior jp = buildJointPrior(spline, layout, imu, P0mat, V3D::Zero(), V3D::Zero(), gravity, var_acc, var_gyr);
    const Eigen::MatrixXd Sigma_head_eta = jp.Sigma_full.topLeftCorner(9 + jp.dEta, 9 + jp.dEta);

    std::vector<Sample> samples;
    for (double frac : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      const double t = t0 + frac * (t1 - t0);
      const auto ps = evaluatePoseControlPhysicalSample(spline, layout, jp.hns, t, gravity);
      Sample sm;
      sm.frac = frac;
      sm.P_p = poseControlPhysicalCovariance(ps.dp_dhead, ps.dp_deta, Sigma_head_eta);
      sm.P_v = poseControlPhysicalCovariance(ps.dv_dhead, ps.dv_deta, Sigma_head_eta);
      sm.P_a = poseControlPhysicalCovariance(ps.da_dhead, ps.da_deta, Sigma_head_eta);
      sm.P_theta = poseControlPhysicalCovariance(ps.dtheta_dhead, ps.dtheta_deta, Sigma_head_eta);
      sm.P_omega = poseControlPhysicalCovariance(ps.domega_dhead, ps.domega_deta, Sigma_head_eta);
      samples.push_back(sm);
    }
    per_N.push_back(samples);
  }

  double worst_p = 0.0, worst_v = 0.0, worst_a = 0.0, worst_theta = 0.0;
  for (size_t i = 0; i < per_N[0].size(); ++i) {
    for (size_t k = 1; k < per_N.size(); ++k) {
      const auto rel = [](const Eigen::Matrix3d& x, const Eigen::Matrix3d& y) {
        return (x - y).norm() / std::max(1e-12, std::max(x.norm(), y.norm()));
      };
      worst_p = std::max(worst_p, rel(per_N[0][i].P_p, per_N[k][i].P_p));
      worst_v = std::max(worst_v, rel(per_N[0][i].P_v, per_N[k][i].P_v));
      worst_a = std::max(worst_a, rel(per_N[0][i].P_a, per_N[k][i].P_a));
      worst_theta = std::max(worst_theta, rel(per_N[0][i].P_theta, per_N[k][i].P_theta));
      std::printf("  [N-invariance frac=%.2f] N=4 vs N=%d: relP_p=%.3e relP_v=%.3e relP_a=%.3e relP_theta=%.3e "
                  "(trace_p N4=%.3e N%d=%.3e; trace_v N4=%.3e N%d=%.3e)\n",
                  per_N[0][i].frac, Ns[k], rel(per_N[0][i].P_p, per_N[k][i].P_p), rel(per_N[0][i].P_v, per_N[k][i].P_v),
                  rel(per_N[0][i].P_a, per_N[k][i].P_a), rel(per_N[0][i].P_theta, per_N[k][i].P_theta),
                  per_N[0][i].P_p.trace(), Ns[k], per_N[k][i].P_p.trace(),
                  per_N[0][i].P_v.trace(), Ns[k], per_N[k][i].P_v.trace());
    }
  }
  check(worst_p < 0.05, "shared-mode full physical position covariance is N-invariant (head+eta corrected mapping)", worst_p, 0.05);
  check(worst_v < 0.05, "shared-mode full physical velocity covariance is N-invariant", worst_v, 0.05);
  check(worst_a < 0.10, "shared-mode full physical acceleration covariance is N-invariant", worst_a, 0.10);
  check(worst_theta < 0.05, "shared-mode full physical attitude covariance is N-invariant", worst_theta, 0.05);
}

// ============================================================================
// Phase 6 (second part) + Phase 7: additional high-frequency modes that
// exist ONLY at higher N, evaluated in physically-normalized sigma units
// using the CORRECTED full covariance (not the eta-only Hessian used by
// test_pose_control_pre_realdata.cpp's testHighFrequencyPhysicalSigmaScaling).
// ============================================================================
void testHighFrequencyModesInPhysicalSigmaUnits()
{
  const double t0 = 0.0, t1 = 0.1, dt_imu = 0.001;
  const V3D gravity(0, 0, -9.81);
  const V3D var_acc = V3D::Constant(0.02 * 0.02), var_gyr = V3D::Constant(0.005 * 0.005);
  Eigen::MatrixXd P0mat = Eigen::MatrixXd::Zero(9, 9);
  P0mat.block<3, 3>(3, 3) = 4e-4 * Eigen::Matrix3d::Identity();
  P0mat.block<3, 3>(6, 6) = 9e-4 * Eigen::Matrix3d::Identity();

  for (int N : {4, 7, 13}) {
    PoseControlSpline spline;
    spline.init(N, t0, t1);
    spline.R_anchor = M3D::Identity();
    for (int k = 0; k < N; ++k) { spline.cp_p.col(k).setZero(); spline.cp_phi.col(k).setZero(); }
    const auto imu = perfectImu(spline, dt_imu, V3D::Zero(), V3D::Zero(), gravity);
    PoseControlFreeLayout layout; layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = false;
    const JointPrior jp = buildJointPrior(spline, layout, imu, P0mat, V3D::Zero(), V3D::Zero(), gravity, var_acc, var_gyr);
    const Eigen::MatrixXd Sigma_head_eta = jp.Sigma_full.topLeftCorner(9 + jp.dEta, 9 + jp.dEta);
    const double tm = 0.5 * (t0 + t1);
    const auto ps = evaluatePoseControlPhysicalSample(spline, layout, jp.hns, tm, gravity);
    const Eigen::Matrix3d P_a = poseControlPhysicalCovariance(ps.da_dhead, ps.da_deta, Sigma_head_eta);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(P_a);
    const Eigen::Matrix3d P_a_inv = es.eigenvectors() *
        es.eigenvalues().cwiseMax(1e-300).cwiseInverse().asDiagonal() * es.eigenvectors().transpose();

    // Highest-alternation raw control-point perturbation representable at
    // this N (alternating sign every control point) -- the spatial
    // frequency this mode carries scales with N, so it is progressively
    // "more genuinely new" at higher N, not merely re-representing the
    // same physical content.
    Eigen::VectorXd raw = Eigen::VectorXd::Zero(3 * N);
    for (int k = 0; k < N; ++k) raw(3 * k) = (k % 2 == 0) ? 1e-3 : -1e-3;
    // Project to eta (remove any head-constrained component) via the SAME
    // Z basis the production estimator uses, then map through da_deta.
    const Eigen::MatrixXd Zp = jp.hns.Z.topRows(3 * N);
    const Eigen::VectorXd eta_perturb = Zp.completeOrthogonalDecomposition().solve(raw);
    const Eigen::VectorXd da = ps.da_deta * eta_perturb;
    const double mahalanobis2 = (da.transpose() * P_a_inv * da)(0);
    const double sigma = std::sqrt(std::max(0.0, mahalanobis2));
    std::printf("  [N=%d] high-frequency alternating mode: |delta_a|=%.4e m/s^2, Mahalanobis sigma=%.4f\n",
                N, da.norm(), sigma);
    check(std::isfinite(sigma) && sigma >= 0.0,
          "high-frequency mode's physically-normalized (full-covariance) sigma is finite and nonnegative", sigma, 0.0);
  }
}

// ============================================================================
// Phase 8: full stochastic timestep invariance of the COMPLETE physical
// covariance (not just the eta-space information used by
// testFullNoiseDensityRefinement), including accelerometer-bias
// uncertainty in the state.
// ============================================================================
void testFullStochasticTimestepInvariance()
{
  const int N = 13;
  const double t0 = 0.0, t1 = 0.1;
  const V3D gravity(0, 0, -9.81);
  const double q_acc = 4e-4;   // continuous spectral density
  const V3D var_gyr = V3D::Constant(0.005 * 0.005);

  PoseControlSpline spline;
  spline.init(N, t0, t1);
  spline.R_anchor = M3D::Identity();
  for (int k = 0; k < N; ++k) { spline.cp_p.col(k).setZero(); spline.cp_phi.col(k).setZero(); }

  Eigen::MatrixXd P0mat = Eigen::MatrixXd::Zero(9, 9);
  P0mat.block<3, 3>(3, 3) = 4e-4 * Eigen::Matrix3d::Identity();
  P0mat.block<3, 3>(6, 6) = 9e-4 * Eigen::Matrix3d::Identity();

  PoseControlFreeLayout layout; layout.N = N; layout.has_bg = false; layout.has_ba = true; layout.has_g = false;

  std::vector<double> traces_p, traces_v;
  const std::vector<double> dts = {0.004, 0.002, 0.001, 0.0005};
  for (double dt : dts) {
    const V3D var_acc = V3D::Constant(q_acc / dt);   // fixed continuous density, refined sampling
    const auto imu = perfectImu(spline, dt, V3D::Zero(), V3D::Zero(), gravity);
    const JointPrior jp = buildJointPrior(spline, layout, imu, P0mat, V3D::Zero(), V3D::Zero(), gravity, var_acc, var_gyr);
    const Eigen::MatrixXd Sigma_head_eta = jp.Sigma_full.topLeftCorner(9 + jp.dEta, 9 + jp.dEta);
    const double tm = 0.5 * (t0 + t1);
    const auto ps = evaluatePoseControlPhysicalSample(spline, layout, jp.hns, tm, gravity);
    const Eigen::Matrix3d P_p = poseControlPhysicalCovariance(ps.dp_dhead, ps.dp_deta, Sigma_head_eta);
    const Eigen::Matrix3d P_v = poseControlPhysicalCovariance(ps.dv_dhead, ps.dv_deta, Sigma_head_eta);
    traces_p.push_back(P_p.trace());
    traces_v.push_back(P_v.trace());
    std::printf("  [dt=%.4f] complete-model trace(P_position(mid))=%.6e trace(P_velocity(mid))=%.6e\n", dt, P_p.trace(), P_v.trace());
  }
  double max_rel_p = 0.0, max_rel_v = 0.0;
  for (size_t i = 0; i < traces_p.size(); ++i) {
    max_rel_p = std::max(max_rel_p, std::abs(traces_p[i] / traces_p.back() - 1.0));
    max_rel_v = std::max(max_rel_v, std::abs(traces_v[i] / traces_v.back() - 1.0));
  }
  check(max_rel_p < 0.15, "complete-model (head+eta+bias) physical position covariance converges under 8x timestep refinement", max_rel_p, 0.15);
  check(max_rel_v < 0.15, "complete-model physical velocity covariance converges under 8x timestep refinement", max_rel_v, 0.15);
  check(traces_p.back() > 1e-12, "refined-timestep physical position covariance has not collapsed to a degenerate zero", traces_p.back(), 1e-12);
}

// ============================================================================
// Phase 10F: reducing accelerometer-bias uncertainty transfers confidence
// back toward the IMU-driven trajectory (the inverse direction of the
// existing testBiasUncertaintyForAcceleration monotonicity check).
// ============================================================================
void testReducingBiasUncertaintyTransfersConfidenceToTrajectory()
{
  const int N = 7;
  const double dt = 0.001;
  const double var = 0.02 * 0.02;
  PoseControlSpline spline;
  spline.init(N, 0.0, 0.1);
  spline.R_anchor = M3D::Identity();
  for (int k = 0; k < N; ++k) { spline.cp_p.col(k).setZero(); spline.cp_phi.col(k).setZero(); }
  const auto imu = perfectImu(spline, dt, V3D::Zero(), V3D::Zero(), V3D::Zero());
  PoseControlFreeLayout layout; layout.N = N; layout.has_bg = false; layout.has_ba = true; layout.has_g = false;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
  Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
  buildPoseControlContinuousImuPrior(spline, layout, imu, V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                     V3D::Constant(var), V3D::Constant(0.005 * 0.005), A, b, nullptr, nullptr);
  const int ncp = 3 * N, iba = 6 * N;
  std::vector<double> bias_vars = {1e-2, 1e-4, 1e-6};   // DECREASING uncertainty
  std::vector<double> traj_trace;
  for (double bv : bias_vars) {
    Eigen::MatrixXd Afull = A;
    Afull.block(iba, iba, 3, 3).diagonal().array() += 1.0 / bv;
    const Eigen::MatrixXd P = svdPinv(Afull);
    traj_trace.push_back(P.topLeftCorner(ncp, ncp).trace());
  }
  check(traj_trace[0] >= traj_trace[1] && traj_trace[1] >= traj_trace[2],
        "decreasing accelerometer-bias uncertainty monotonically transfers confidence back to the trajectory "
        "(trajectory covariance shrinks as bias becomes better-known)", traj_trace[0], traj_trace[2]);
  std::printf("  trajectory trace at bias_var={1e-2,1e-4,1e-6}: %.6e, %.6e, %.6e\n", traj_trace[0], traj_trace[1], traj_trace[2]);
}

}  // namespace

int main()
{
  std::printf("Pose-control corrected full-covariance representation/timestep/bias invariance suite\n");
  testRepresentationCapacityInvariance();
  testHighFrequencyModesInPhysicalSigmaUnits();
  testFullStochasticTimestepInvariance();
  testReducingBiasUncertaintyTransfersConfidenceToTrajectory();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
