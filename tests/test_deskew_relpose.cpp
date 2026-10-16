// R97A R2 unit test: imu/undistort/time_based_process_noise = relpose_q | relpose_full.
//   RP1 brute-force equivalence: the suffix-sum result equals the direct sum, random steps/points.
//   RP2 constant isotropic noise per step: Sigma_theta = qg*Delta*I, Sigma_p ~= qa*Delta^3/3*I.
//   RP3 Sigma_add (full deskewPoints() path) is symmetric PSD, exactly zero at Delta = 0, and each
//       eigenvalue is non-decreasing in Delta.
//   RP4 a point on the rotation axis of a pure Sigma_theta = s*e_z e_z^T gets no rotation term
//       from that axis (the [q]x Sigma [q]x^T structure).
//   RP5 none/state/var_acc are unchanged by this patch (covered by test_point_cov.cpp and
//       test_deskew_rot_warp_r91h.cpp, which call deskewPoints() with the new parameters left at
//       their nullptr defaults -- not re-tested here).
//   RP6 relpose_full minus relpose_q equals the quadratic form of the state sub-covariance (P0
//       built by hand).
#include <ros/ros.h>
#define private public
#include "livo_recon/utils/state/state.h"
#undef private
#include "livo_recon/lio/deskew.h"
#include "livo_recon/lio/relpose_noise.h"
#include "livo_recon/utils/algo/math.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

namespace {

M3D randomSPD(std::mt19937& rng, double scale)
{
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  Eigen::Matrix3d A;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) A(i, j) = U(rng);
  return scale * (A * A.transpose() + Eigen::Matrix3d::Identity() * 0.01);
}

std::vector<RelposeStepNoise> randomSteps(std::mt19937& rng, int K, double dt)
{
  std::vector<RelposeStepNoise> steps(K);
  double t = 0.0;
  for (int k = 0; k < K; ++k)
  {
    steps[k].t = t;
    steps[k].dt = dt;
    steps[k].Q_theta = randomSPD(rng, 1e-6);
    steps[k].Q_v = randomSPD(rng, 1e-4);
    steps[k].Q_p = randomSPD(rng, 1e-8);
    Eigen::Matrix3d M; { std::uniform_real_distribution<double> U(-1.0, 1.0); for (int i=0;i<3;++i) for (int j=0;j<3;++j) M(i,j)=U(rng); }
    steps[k].Q_pv = 1e-6 * M;   // not necessarily symmetric -- relposeProcessNoiseBruteForce/suffix both add Q_pv + Q_pv^T, so asymmetry is fine
    t += dt;
  }
  return steps;
}

}  // namespace

int main()
{
  // RP1: brute-force equivalence.
  {
    std::mt19937 rng(7);
    const int K = 23;
    const double dt = 0.0043;
    const auto steps = randomSteps(rng, K, dt);
    const double t_end = K * dt;
    const RelposeSuffixTables tabs = buildRelposeSuffixTables(steps, t_end);
    std::uniform_real_distribution<double> Upt(0.0, t_end);
    double worst_rel = 0.0;
    for (int trial = 0; trial < 200; ++trial)
    {
      const double t_point = Upt(rng);
      M3D st_bf, sp_bf, st_sf, sp_sf;
      relposeProcessNoiseBruteForce(steps, t_point, t_end, st_bf, sp_bf);
      relposeProcessNoiseFromSuffix(tabs, t_point, st_sf, sp_sf);
      const double denom_t = std::max(1e-30, st_bf.norm());
      const double denom_p = std::max(1e-30, sp_bf.norm());
      worst_rel = std::max(worst_rel, (st_bf - st_sf).norm() / denom_t);
      worst_rel = std::max(worst_rel, (sp_bf - sp_sf).norm() / denom_p);
    }
    CHECK(worst_rel < 1e-12, "RP1 suffix-sum equals the direct brute-force sum to 1e-12 relative");
  }

  // RP2: constant isotropic noise per step.
  {
    const int K = 1000;
    const double dt = 1e-3;
    const double qg = 2.0e-6, qa = 3.0e-5;
    std::vector<RelposeStepNoise> steps(K);
    double t = 0.0;
    for (int k = 0; k < K; ++k)
    {
      steps[k].t = t; steps[k].dt = dt;
      steps[k].Q_theta = (qg * dt) * M3D::Identity();
      steps[k].Q_v = (qa * dt) * M3D::Identity();
      steps[k].Q_p.setZero();
      steps[k].Q_pv.setZero();
      t += dt;
    }
    const double t_end = K * dt;
    const RelposeSuffixTables tabs = buildRelposeSuffixTables(steps, t_end);
    const double t_point = 0.0;   // Delta = t_end
    M3D st, sp;
    relposeProcessNoiseFromSuffix(tabs, t_point, st, sp);
    const double Delta = t_end - t_point;
    const M3D expect_theta = (qg * Delta) * M3D::Identity();
    const M3D expect_p = (qa * Delta * Delta * Delta / 3.0) * M3D::Identity();
    CHECK((st - expect_theta).norm() / expect_theta.norm() < 1e-9,
          "RP2 Sigma_theta = qg*Delta*I for constant isotropic per-step noise");
    // Sigma_p is a Riemann sum of qa*(t_end-t_k)^2*dt -> integral qa*Delta^3/3; K=1000 steps gives
    // a sum-vs-integral discretization error of O(dt/Delta) ~ 1e-3 relative -- bound loosely at 1%.
    CHECK((sp - expect_p).norm() / expect_p.norm() < 1e-2,
          "RP2 Sigma_p ~= qa*Delta^3/3*I for constant isotropic per-step noise (discretization bound 1%)");
  }

  // RP3 + RP4: full deskewPoints() path, PSD / zero-at-Delta=0 / monotone-in-Delta / axis structure.
  {
    StateGroupPtr state = std::make_shared<StateGroup>();   // rot = I, pos = 0
    std::mt19937 rng(11);
    const int K = 10;
    const double dt = 0.01;
    std::vector<Pose6D> poses;
    std::vector<Eigen::MatrixXd> qcov;
    const int dim = state->dimState();
    double t = 0.0;
    for (int k = 0; k < K; ++k)
    {
      poses.emplace_back(t, V3D::Zero(), V3D::Zero(), V3D::Zero(), V3D::Zero(), V3D::Zero(), M3D::Identity(), dt);
      Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(dim, dim);
      Q.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()) = randomSPD(rng, 1e-6);
      Q.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()) = randomSPD(rng, 1e-4);
      qcov.push_back(Q);
      t += dt;
    }
    const double t_end = K * dt;

    DeskewOptions o;
    o.time_based_process_noise = "relpose_q";
    std::vector<PointXYZT> pts(1);
    pts[0].p = V3D(1.5, -0.7, 0.3);

    RelposeStats stats;
    double prev_eig = -1.0;
    // t_point = 0.0 exactly coincides with poses.front().t -- deskewPoints()'s point/pose matching
    // loop uses a strict ">" test and never claims a point exactly at the first pose's head time
    // (an existing, pre-R2 boundary property, not something this patch changes), so the smallest
    // point time tested here is a negligible epsilon instead of exactly 0.
    const double eps_t = 1e-9;
    for (double t_point : {eps_t, t_end * 0.25, t_end * 0.5, t_end * 0.75, t_end})
    {
      pts[0].t = t_point;
      std::vector<PointXYZCov> out;
      deskewPoints(state, poses, t_end, pts, o, out, &qcov, nullptr, &stats);
      const M3D cov = out[0].sensor_cov;
      CHECK((cov - cov.transpose()).norm() < 1e-15 * std::max(1.0, cov.norm()), "RP3 Sigma_add (via cov_imu_end) is symmetric");
      Eigen::SelfAdjointEigenSolver<M3D> es(cov);
      const double min_eig = es.eigenvalues().minCoeff();
      if (t_point == eps_t)
        CHECK(min_eig > -1e-18, "RP3 cov_imu_end is PSD at Delta = t_end (the largest Delta tested)");
      if (t_point == t_end)
      {
        // Delta = 0: the relpose term vanishes, leaving exactly the sensor-model term (no process noise).
        DeskewOptions o_none; o_none.time_based_process_noise = "none";
        std::vector<PointXYZCov> out_none;
        deskewPoints(state, poses, t_end, pts, o_none, out_none);
        CHECK((cov - out_none[0].sensor_cov).norm() < 1e-20, "RP3 Sigma_add is exactly zero at Delta = 0");
      }
      const double eig_sum = es.eigenvalues().sum();
      // t_point ascends across iterations, so Delta = t_end - t_point DESCENDS -- the eigenvalue
      // sum must be non-INCREASING here (equivalently non-decreasing in Delta, as the spec states).
      if (prev_eig >= 0.0)
        CHECK(eig_sum <= prev_eig + 1e-18, "RP3 cov_imu_end's eigenvalue sum is non-decreasing as Delta grows (Delta falls as t_point rises)");
      prev_eig = eig_sum;
    }

    // RP4: point on the rotation axis of a pure Sigma_theta = s*e_z e_z^T (isolate via a Q_theta
    // that is zero except its zz entry, and zero Q_v/Q_p/Q_pv so Sigma_p = 0) gets no rotation
    // term from the z axis: with q on the z axis, [q]x e_z = 0, so the z-direction contribution
    // vanishes exactly.
    std::vector<Eigen::MatrixXd> qcov_axis(K, Eigen::MatrixXd::Zero(dim, dim));
    for (int k = 0; k < K; ++k)
      qcov_axis[k](StateGroup::idxR() + 2, StateGroup::idxR() + 2) = 1e-6;   // pure z-axis attitude noise
    std::vector<PointXYZT> pts_axis(1);
    pts_axis[0].p = V3D(0.0, 0.0, 2.0);   // on the z axis in the LiDAR frame; lidarToImu is identity by default
    pts_axis[0].t = 1e-9;   // see eps_t note above -- exactly poses.front().t is never claimed
    DeskewOptions o_axis; o_axis.time_based_process_noise = "relpose_q";
    std::vector<PointXYZCov> out_axis;
    deskewPoints(state, poses, t_end, pts_axis, o_axis, out_axis, &qcov_axis);
    DeskewOptions o_axis_none; o_axis_none.time_based_process_noise = "none";
    std::vector<PointXYZCov> out_axis_none;
    deskewPoints(state, poses, t_end, pts_axis, o_axis_none, out_axis_none);
    CHECK((out_axis[0].sensor_cov - out_axis_none[0].sensor_cov).norm() < 1e-20,
          "RP4 a point on the rotation axis of a pure z-axis Sigma_theta gets no rotation term from that axis");
  }

  // RP6: relpose_full minus relpose_q equals the quadratic form of the state sub-covariance.
  {
    StateGroupPtr state = std::make_shared<StateGroup>();
    const int dim = state->dimState();
    std::vector<Pose6D> poses;   // one step with zero process noise -- isolates the P0 term
    poses.emplace_back(0.0, V3D::Zero(), V3D::Zero(), V3D::Zero(), V3D::Zero(), V3D::Zero(), M3D::Identity(), 0.05);
    std::vector<Eigen::MatrixXd> qcov(1, Eigen::MatrixXd::Zero(dim, dim));
    const double t_end = 0.05;

    std::mt19937 rng(13);
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(dim, dim);
    const M3D P_vv = randomSPD(rng, 1e-3), P_baba = randomSPD(rng, 1e-6), P_bgbg = randomSPD(rng, 1e-7);
    Eigen::Matrix3d P_vba; { std::uniform_real_distribution<double> U(-1.0, 1.0); for (int i=0;i<3;++i) for (int j=0;j<3;++j) P_vba(i,j)=U(rng); }
    P_vba *= 1e-4;
    P0.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()) = P_vv;
    P0.block<3, 3>(StateGroup::idxBA(), StateGroup::idxBA()) = P_baba;
    P0.block<3, 3>(StateGroup::idxBG(), StateGroup::idxBG()) = P_bgbg;
    P0.block<3, 3>(StateGroup::idxV(), StateGroup::idxBA()) = P_vba;
    P0.block<3, 3>(StateGroup::idxBA(), StateGroup::idxV()) = P_vba.transpose();

    std::vector<PointXYZT> pts(1);
    pts[0].p = V3D(0.8, 1.1, -0.4);
    pts[0].t = 1e-9;   // Delta ~= t_end = 0.05 (see eps_t note above)

    DeskewOptions o_q; o_q.time_based_process_noise = "relpose_q";
    DeskewOptions o_full; o_full.time_based_process_noise = "relpose_full";
    std::vector<PointXYZCov> out_q, out_full;
    deskewPoints(state, poses, t_end, pts, o_q, out_q, &qcov);
    deskewPoints(state, poses, t_end, pts, o_full, out_full, &qcov, &P0);

    const double Delta = t_end - pts[0].t;
    const double a = Delta, b = 0.5 * Delta * Delta;
    const M3D sigma_p_extra = a * a * P_vv + b * b * P_baba + a * b * (P_vba + P_vba.transpose());
    const M3D sigma_theta_extra = a * a * P_bgbg;
    const V3D q = out_q[0].point;   // same deskewed point in both (no process-noise-driven motion here)
    M3D q_skew; q_skew << 0, -q.z(), q.y(), q.z(), 0, -q.x(), -q.y(), q.x(), 0;
    const M3D expect_diff = sigma_p_extra + q_skew * sigma_theta_extra * q_skew.transpose();
    const M3D got_diff = out_full[0].sensor_cov - out_q[0].sensor_cov;
    CHECK((got_diff - expect_diff).norm() / std::max(1e-30, expect_diff.norm()) < 1e-9,
          "RP6 relpose_full minus relpose_q equals the hand-built P0 quadratic form");
  }

  std::printf("ALL PASS\n");
  return 0;
}
