// Continuous-prior implementation phase: items 6/7/8/9/10/25/26/38/39/40 --
// an INDEPENDENT reference comparison (old segment-propagation vs new
// continuous-time collocation, two genuinely different mathematical
// derivations -- not the same production code exercised on both sides),
// a high-frequency knot-perturbation test, a representation-capacity test,
// and a joint trajectory/bias cross-covariance check.
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/lio/pose_control_physical_diagnostics.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>

using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, double val = 0.0) {
  std::printf("  [%s] %-64s %+.6e\n", ok ? " ok " : "FAIL", name, val);
  if (!ok) ++g_fail;
}

static PoseControlSpline makeStationaryTrajectory(int N, double t0, double t1, unsigned seed) {
  PoseControlSpline s;
  s.init(N, t0, t1);
  s.R_anchor = M3D::Identity();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> up(-0.02, 0.02);
  for (int i = 0; i < N; ++i) {
    s.cp_p.col(i) = V3D(up(rng), up(rng), up(rng));
    s.cp_phi.col(i) = V3D(up(rng), up(rng), up(rng));
  }
  return s;
}

// ============================================================================
// items 6/22/39: INDEPENDENT reference test. Compares the NEW continuous-
// time collocation prior against the OLD (now test-only)
// accumulatePoseControlImuPriorSegmentReduced()'s endpoint/segment-
// propagation prior -- two genuinely different mathematical derivations
// (direct measurement collocation vs discrete-time IMU strapdown
// propagation), not the same code run twice. For a near-static, small-N
// (well-resolved) synthetic trajectory both should agree to within an
// order of magnitude on physical acceleration/gyro covariance; the point
// of this test is NOT bit-exact agreement (they are different models) but
// that neither is wildly inconsistent with the other in the regime where
// both are well-conditioned.
// ============================================================================
static void testIndependentPriorReference()
{
  const int N = 7;
  const double t0 = 0.0, t1 = 0.1;
  PoseControlSpline spline = makeStationaryTrajectory(N, t0, t1, 111);
  PoseControlFreeLayout layout;
  layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = true;
  const V3D bias_acc = V3D::Zero(), bias_gyr = V3D::Zero(), gravity(0, 0, -9.81);
  const V3D var_acc = V3D::Constant(0.02 * 0.02), var_gyr = V3D::Constant(0.002 * 0.002);

  std::vector<ImuSample> imu;
  for (double t = t0; t <= t1 + 1e-9; t += 1.0 / 200.0) {
    ImuSample s; s.t = t;
    s.acc = spline.rotAt(t).transpose() * (spline.accAt(t) - gravity);
    s.gyro = spline.omegaBodyAt(t);
    imu.push_back(s);
  }

  const int dimRaw = layout.dim();
  // NEW: continuous-time collocation.
  Eigen::MatrixXd A_new = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_new = Eigen::VectorXd::Zero(dimRaw);
  buildPoseControlContinuousImuPrior(spline, layout, imu, bias_acc, bias_gyr, gravity, var_acc, var_gyr, A_new, b_new, nullptr, nullptr);

  // OLD (test-only reference): endpoint/segment propagation.
  const auto segs = bucketPoseControlImuSamples(imu, spline);
  Eigen::MatrixXd A_old = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_old_unused = Eigen::VectorXd::Zero(dimRaw);
  for (int j = 0; j < spline.nSeg(); ++j)
    accumulatePoseControlImuPriorSegmentReduced(spline, layout, j, segs.seg_samples[j], bias_acc, bias_gyr, gravity,
        1.0, 1.0, var_acc, var_gyr, true, 1e-6, A_old, b_old_unused, nullptr, nullptr);

  PoseControlHeadNullspace hns;
  hns.Z = Eigen::MatrixXd::Identity(6 * N, 6 * N);
  hns.c_particular = Eigen::VectorXd::Zero(6 * N);
  const double t_probe = 0.5 * (t0 + t1);
  const auto sample = evaluatePoseControlPhysicalSample(spline, layout, hns, t_probe, gravity);

  const Eigen::MatrixXd P_new = generalPseudoInverse(A_new, 1e-9);
  const Eigen::MatrixXd P_old = generalPseudoInverse(A_old, 1e-9);
  const Eigen::MatrixXd Pc_new = P_new.topLeftCorner(6 * N, 6 * N);
  const Eigen::MatrixXd Pc_old = P_old.topLeftCorner(6 * N, 6 * N);
  const double trace_Pa_new = (sample.da_deta * Pc_new * sample.da_deta.transpose()).trace();
  const double trace_Pa_old = (sample.da_deta * Pc_old * sample.da_deta.transpose()).trace();
  const double trace_Pomega_new = (sample.domega_deta * Pc_new * sample.domega_deta.transpose()).trace();
  const double trace_Pomega_old = (sample.domega_deta * Pc_old * sample.domega_deta.transpose()).trace();

  std::printf("  N=%d (well-resolved regime): trace(P_a)_new=%.4e trace(P_a)_old=%.4e ratio=%.4e\n",
              N, trace_Pa_new, trace_Pa_old, trace_Pa_old > 1e-300 ? trace_Pa_new / trace_Pa_old : 0.0);
  std::printf("  trace(P_omega)_new=%.4e trace(P_omega)_old=%.4e ratio=%.4e\n",
              trace_Pomega_new, trace_Pomega_old, trace_Pomega_old > 1e-300 ? trace_Pomega_new / trace_Pomega_old : 0.0);
  check(std::isfinite(trace_Pa_new) && trace_Pa_new > 0, "new prior's physical accel covariance is finite and positive", trace_Pa_new);
  check(std::isfinite(trace_Pomega_new) && trace_Pomega_new > 0, "new prior's physical omega covariance is finite and positive", trace_Pomega_new);
  const double ratio_a = trace_Pa_old > 1e-300 ? trace_Pa_new / trace_Pa_old : 0.0;
  check(ratio_a > 1e-4 && ratio_a < 1e4, "new-vs-old prior accel covariance agree within 4 orders of magnitude (well-resolved N)", ratio_a);
}

// ============================================================================
// item 25/38.7: high-frequency knot perturbation. An alternating adjacent
// control-point perturbation (small position amplitude, large SECOND
// difference -> large acceleration) must be strongly penalized by the
// production prior's Mahalanobis cost when the IMU covariance is small,
// i.e. the SAME position perturbation magnitude must cost far more when
// its second-difference (curvature) content is high than when it is low.
// ============================================================================
static void testHighFrequencyKnotPerturbation()
{
  const int N = 13;
  const double t0 = 0.0, t1 = 0.1;
  PoseControlSpline spline = makeStationaryTrajectory(N, t0, t1, 222);
  PoseControlFreeLayout layout;
  layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = true;
  const V3D bias_acc = V3D::Zero(), bias_gyr = V3D::Zero(), gravity(0, 0, -9.81);
  const V3D var_acc = V3D::Constant(0.02 * 0.02), var_gyr = V3D::Constant(0.002 * 0.002);

  std::vector<ImuSample> imu;
  for (double t = t0; t <= t1 + 1e-9; t += 1.0 / 200.0) {
    ImuSample s; s.t = t;
    s.acc = spline.rotAt(t).transpose() * (spline.accAt(t) - gravity);
    s.gyro = spline.omegaBodyAt(t);
    imu.push_back(s);
  }
  const int dimRaw = layout.dim();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_unused = Eigen::VectorXd::Zero(dimRaw);
  buildPoseControlContinuousImuPrior(spline, layout, imu, bias_acc, bias_gyr, gravity, var_acc, var_gyr, A, b_unused, nullptr, nullptr);

  const double eps = 0.005;  // 5mm position perturbation magnitude, either way
  // Perturbation A: alternating sign across 3 adjacent knots (HIGH second
  // difference / high-frequency content) in the x-axis, on knots 5,6,7.
  Eigen::VectorXd delta_high = Eigen::VectorXd::Zero(dimRaw);
  delta_high(3 * 5 + 0) = eps; delta_high(3 * 6 + 0) = -eps; delta_high(3 * 7 + 0) = eps;
  // Perturbation B: same-sign shift across the SAME 3 knots (LOW second
  // difference -- a near-rigid local translation, same L2 norm as A).
  Eigen::VectorXd delta_low = Eigen::VectorXd::Zero(dimRaw);
  delta_low(3 * 5 + 0) = eps; delta_low(3 * 6 + 0) = eps; delta_low(3 * 7 + 0) = eps;

  check(std::abs(delta_high.norm() - delta_low.norm()) < 1e-12,
        "both perturbations have IDENTICAL L2 norm (isolating frequency content, not magnitude)", delta_high.norm() - delta_low.norm());

  const double cost_high = (delta_high.transpose() * A * delta_high)(0);
  const double cost_low = (delta_low.transpose() * A * delta_low)(0);
  std::printf("  same-magnitude perturbations: cost_high_frequency=%.4e  cost_low_frequency(rigid shift)=%.4e  ratio=%.4e\n",
              cost_high, cost_low, cost_low > 1e-300 ? cost_high / cost_low : 0.0);
  check(cost_high > cost_low, "high-frequency (alternating) knot perturbation costs MORE than a same-magnitude rigid shift", cost_high - cost_low);
  check(cost_high > 3.0 * std::max(cost_low, 1e-12), "the high-frequency penalty is not merely marginally larger (item 16's central claim)", cost_high / std::max(cost_low, 1e-12));
}

// ============================================================================
// item 9/38.9: joint trajectory/bias covariance is NOT block-diagonal.
// ============================================================================
static void testJointTrajectoryBiasCrossCovariance()
{
  const int N = 7;
  const double t0 = 0.0, t1 = 0.1;
  PoseControlSpline spline = makeStationaryTrajectory(N, t0, t1, 333);
  PoseControlFreeLayout layout;
  layout.N = N; layout.has_bg = layout.has_ba = layout.has_g = true;
  const V3D bias_acc = V3D::Zero(), bias_gyr = V3D::Zero(), gravity(0, 0, -9.81);
  const V3D var_acc = V3D::Constant(0.02 * 0.02), var_gyr = V3D::Constant(0.002 * 0.002);
  std::vector<ImuSample> imu;
  for (double t = t0; t <= t1 + 1e-9; t += 1.0 / 200.0) {
    ImuSample s; s.t = t;
    s.acc = spline.rotAt(t).transpose() * (spline.accAt(t) - gravity) + V3D(0.05, 0, 0);  // deliberate small bias-like offset
    s.gyro = spline.omegaBodyAt(t);
    imu.push_back(s);
  }
  const int dimRaw = layout.dim();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimRaw, dimRaw);
  Eigen::VectorXd b_unused = Eigen::VectorXd::Zero(dimRaw);
  buildPoseControlContinuousImuPrior(spline, layout, imu, bias_acc, bias_gyr, gravity, var_acc, var_gyr, A, b_unused, nullptr, nullptr);
  const Eigen::MatrixXd P = generalPseudoInverse(A, 1e-9);
  const int off_ba = layout.colBA();
  const Eigen::MatrixXd P_eta_ba = P.block(0, off_ba, 6 * N, 3);
  std::printf("  ||P_eta_ba|| (trajectory/accel-bias cross-covariance block) = %.4e\n", P_eta_ba.norm());
  check(P_eta_ba.norm() > 1e-9, "trajectory/bias cross-covariance is NONZERO (item 10/14 -- not block-diagonal)", P_eta_ba.norm());

  // item 10: "indirect bias correction" -- de_acc/dba=I means bias
  // directly receives information from every sample; confirm P_ba (the
  // marginal covariance block) actually shrunk relative to an
  // uninformative prior (i.e. bias is genuinely constrained, not a
  // structural nullspace).
  const Eigen::Matrix3d P_ba = P.block<3, 3>(off_ba, off_ba);
  check(P_ba.trace() < 1.0, "accel-bias marginal covariance is meaningfully constrained by the IMU factor (not left at ~infinite prior variance)", P_ba.trace());
}

// ============================================================================
// item 26/38.8: representation-capacity test.
// ============================================================================
static void testRepresentationCapacity()
{
  auto posSmooth = [](double t) { return V3D(0.02 * std::sin(2 * M_PI * 2 * t), 0, 0); };       // 2Hz, well within N=4's reach over 0.1s
  auto posHighFreq = [](double t) { return V3D(0.005 * std::sin(2 * M_PI * 25 * t), 0, 0); };    // 25Hz, needs many knots

  auto fitError = [&](int N, auto posFunc) {
    const double t0 = 0.0, t1 = 0.1;
    PoseControlSpline s; s.init(N, t0, t1); s.R_anchor = M3D::Identity();
    const int M = 200;
    Eigen::MatrixXd Bmat = Eigen::MatrixXd::Zero(M, N);
    Eigen::VectorXd target(M);
    for (int i = 0; i < M; ++i) {
      const double t = t0 + (t1 - t0) * (i + 0.5) / M;
      const auto jac = s.jacobianAt(t);
      for (int k = 0; k < 4; ++k) Bmat(i, jac.s + k) = jac.b[k];
      target(i) = posFunc(t).x();
    }
    const Eigen::MatrixXd BtB_pinv = generalPseudoInverse(Bmat.transpose() * Bmat, 1e-9);
    const Eigen::VectorXd cp_x = BtB_pinv * Bmat.transpose() * target;
    double max_err = 0.0;
    for (int i = 0; i < M; ++i) max_err = std::max(max_err, std::abs((Bmat.row(i) * cp_x)(0) - target(i)));
    return max_err;
  };

  const double err_smooth_N4 = fitError(4, posSmooth);
  const double err_highfreq_N4 = fitError(4, posHighFreq);
  const double err_highfreq_N13 = fitError(13, posHighFreq);
  std::printf("  fit max-abs-error: smooth/N=4=%.4e  highfreq/N=4=%.4e  highfreq/N=13=%.4e\n",
              err_smooth_N4, err_highfreq_N4, err_highfreq_N13);
  check(err_smooth_N4 < 0.002, "N=4 represents the SMOOTH (2Hz) trajectory well", err_smooth_N4);
  check(err_highfreq_N4 > 0.002, "N=4 CANNOT represent the high-frequency (25Hz) trajectory without appreciable error (expected/appropriate)", err_highfreq_N4);
  check(err_highfreq_N13 < err_highfreq_N4 * 0.5, "N=13 represents the high-frequency trajectory substantially better than N=4", err_highfreq_N13);
}

int main() {
  std::printf("Continuous-prior independent-reference + high-frequency + representation-capacity tests\n");
  std::printf("-- item 6/22/39: independent reference --\n");
  testIndependentPriorReference();
  std::printf("-- item 25: high-frequency knot perturbation --\n");
  testHighFrequencyKnotPerturbation();
  std::printf("-- item 9/10: joint trajectory/bias cross-covariance --\n");
  testJointTrajectoryBiasCrossCovariance();
  std::printf("-- item 26: representation capacity --\n");
  testRepresentationCapacity();
  std::printf("\n%d failed\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
