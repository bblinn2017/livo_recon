// CQ-50/CQ-53: standalone finite-difference verification of
// propagateCoupled()'s phi_x_head (specifically the phi_R rows) against a
// numerical re-integration, evaluated at an INTERMEDIATE time t_k (not just
// t1). Same "no ROS/PCL/OpenCV, Eigen-only" convention as test_spline.cpp/
// test_indirect.cpp in this directory -- see README.md.
//
// Built during CQ-50's investigation into the coupled estimator's H(t1)/
// Phi(t_k) time-mismatch: refutes a specific sign/frame-convention
// hypothesis raised by coupled_estimator.h's own (since-corrected) doc
// comment, which described phi_R as a "LEFT-tangent... Log(R_true^T
// R_nominal)" convention -- this test's own dtheta_numeric/phi_R_analytic
// ratio comes out ~1.0000 (matching the STANDARD right/body-frame dtheta
// convention, R_true = R_nominal * Exp(phi_R)), not ~-1.0000 (which the
// stale comment's convention would have predicted). It also confirms
// worldRotAt() and interpolatePhiX() are mutually consistent.
//
//   g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
//       scripts/test/test_coupled_phi_fd.cpp src/lio/coupled_estimator.cpp \
//       -o /tmp/test_coupled_phi_fd && /tmp/test_coupled_phi_fd
//
// coupled_estimator.h/.cpp only depend on utils/algo/math.h and
// utils/data/data_wrappers.h (the latter needs OpenCV for ImageData --
// see this directory's README.md note on that).
#include "livo_recon/lio/coupled_estimator.h"
#include <cstdio>
#include <vector>

using namespace livo_recon;

// Build N synthetic IMU segments with a modest, non-trivial rotation rate,
// mirroring what deskewPoints()/ImuProc::propagate() would hand
// propagateCoupled() as raw_poses.
static std::vector<Pose6D> buildRawPoses(int N, double dt, const M3D& rot0,
                                          const V3D& pos0, const V3D& vel0,
                                          const V3D& gravity0)
{
  std::vector<Pose6D> poses;
  poses.reserve(N);
  M3D R = rot0;
  V3D p = pos0, v = vel0;
  // A fixed body-frame gyro rate (rad/s) and a mild body-frame accel, so the
  // raw chain has genuine curvature (not degenerate).
  const V3D gyr(0.05, -0.03, 0.4);   // ~23 deg/s about z -- "real sustained rotation"
  const V3D acc_body(0.2, -0.1, 9.85);
  for (int k = 0; k < N; ++k)
  {
    const double t = k * dt;
    // Pose6D's own contract: acc_head/acc_tail = R*a_body + g (world-frame,
    // WITH gravity folded in) -- propagateCoupled() strips gravity0 back out
    // via R_head_raw^T*(seg.acc_head - gravity0), so this must match that.
    const V3D acc_head = R * acc_body + gravity0;
    const M3D R_tail = R * Exp(gyr, dt);
    const V3D acc_tail = R_tail * acc_body + gravity0;
    poses.push_back(Pose6D{t, acc_head, acc_tail, gyr, v, p, R, dt});
    // advance nominal raw chain (unbiased, simple midpoint)
    const V3D acc_avr_world = 0.5 * (acc_head + acc_tail);
    p = p + v * dt + 0.5 * acc_avr_world * dt * dt;
    v = v + acc_avr_world * dt;
    R = R_tail;
  }
  return poses;
}

int main()
{
  const int N = 20;
  const double dt = 0.005;  // 5ms per IMU segment, 0.1s total scan
  const M3D rot0 = M3D::Identity();
  const V3D pos0 = V3D::Zero(), vel0 = V3D::Zero();
  const V3D gravity0(0, 0, -9.81);
  const V3D gravity = gravity0;  // no gravity correction in this test
  const V3D delta_bg = V3D::Zero(), delta_ba = V3D::Zero();
  const int n_c = 4;
  std::vector<V3D> c_acc(n_c, V3D::Zero()), c_gyr(n_c, V3D::Zero());

  std::vector<Pose6D> raw_poses = buildRawPoses(N, dt, rot0, pos0, vel0, gravity0);
  const M3D raw_rot1 = raw_poses.back().rot * Exp(raw_poses.back().gyr, dt);
  const double scan_end_time = N * dt;

  CoupledPropagation prop0;
  propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0, pos0, vel0,
                    gravity, gravity0, delta_bg, delta_ba, c_acc, c_gyr, n_c, prop0);

  // Perturb delta_bg (a single-scalar param this scan's own coefficients
  // don't touch -- isolates the delta_bg columns of phi_x_head cleanly)
  // along each axis by epsilon, re-propagate, compare against phi_x_head's
  // own prediction at an INTERMEDIATE time t_k (not t0, not t1).
  const double eps = 1e-6;
  bool all_pass = true;
  for (int axis = 0; axis < 3; ++axis)
  {
    V3D delta_bg_pert = V3D::Zero();
    delta_bg_pert(axis) = eps;

    CoupledPropagation prop1;
    propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0, pos0, vel0,
                      gravity, gravity0, delta_bg_pert, delta_ba, c_acc, c_gyr, n_c, prop1);

    const double t_k = 0.6 * scan_end_time;  // intermediate, not an endpoint

    const M3D R0_tk = worldRotAt(prop0, t_k);
    const M3D R1_tk = worldRotAt(prop1, t_k);
    // Numerical dtheta (STANDARD body-frame convention: R_true = R_nom *
    // Exp(dtheta), so dtheta = Log(R_nom^T * R_true)).
    const V3D dtheta_numeric = Log(R0_tk.transpose() * R1_tk) / eps;

    const Eigen::Matrix<double, 9, 18> Phix_tk = interpolatePhiX(prop0, t_k);
    // delta_bg occupies columns [9,12) of the 18-col layout.
    const V3D phi_R_analytic = Phix_tk.block<3, 1>(0, 9 + axis);

    const double ratio_dom = phi_R_analytic(axis) / (dtheta_numeric(axis) + 1e-30);
    const bool pass = std::abs(ratio_dom - 1.0) < 0.05;  // within 5% on the dominant term
    all_pass = all_pass && pass;
    printf("axis=%d  dtheta_numeric=(%.6f %.6f %.6f)  phi_R_analytic=(%.6f %.6f %.6f)  "
           "dominant_ratio=%.4f  %s\n",
           axis, dtheta_numeric.x(), dtheta_numeric.y(), dtheta_numeric.z(),
           phi_R_analytic.x(), phi_R_analytic.y(), phi_R_analytic.z(),
           ratio_dom, pass ? "PASS" : "FAIL");
  }

  // CQ-53 item 0d: the FIRST check above only tests phi_x (state
  // sensitivity, via delta_bg). It says NOTHING about phi_c (coefficient
  // sensitivity, interpolatePhi()/phi_head) -- the half that actually
  // carries c_acc/c_gyr into the measurement row at lio_coupled.cpp's
  // Jrow.segment(ncol_s, ncol_c) = H * Phic_pt.topRows(6). A sign/scale/
  // column-ordering bug in phi_c's gyro block would be INVISIBLE to the
  // phi_x check above and would look exactly like what CQ-50/52 observed:
  // the state half behaves, the coefficient half poisons the joint solve.
  // Perturb c_gyr_j (rotation rows) and c_acc_j (position rows) at a low,
  // middle, and high j, compare against interpolatePhi()'s own columns.
  for (int j : {0, n_c / 2, n_c - 1})
  {
    // c_gyr_j -> rotation rows (0:3), column 3*n_c + 3*j in phi_head's
    // [c_acc(3*n_c), c_gyr(3*n_c)] layout.
    {
      std::vector<V3D> c_gyr_pert = c_gyr;
      c_gyr_pert[j] = V3D(0, 0, eps);  // z-axis perturbation, arbitrary but nonzero
      CoupledPropagation prop1;
      propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0, pos0, vel0,
                        gravity, gravity0, delta_bg, delta_ba, c_acc, c_gyr_pert, n_c, prop1);
      const double t_k = 0.6 * scan_end_time;
      const M3D R0_tk = worldRotAt(prop0, t_k);
      const M3D R1_tk = worldRotAt(prop1, t_k);
      const V3D dtheta_numeric = Log(R0_tk.transpose() * R1_tk) / eps;
      const Eigen::Matrix<double, 9, Eigen::Dynamic> Phic_tk = interpolatePhi(prop0, t_k);
      const V3D phi_R_analytic = Phic_tk.block<3, 1>(0, 3 * n_c + 3 * j + 2);
      const double ratio = phi_R_analytic.z() / (dtheta_numeric.z() + 1e-30);
      const bool pass = std::abs(ratio - 1.0) < 0.05;
      all_pass = all_pass && pass;
      printf("c_gyr[j=%d,z]  dtheta_numeric=(%.6f %.6f %.6f)  phi_R_analytic=(%.6f %.6f %.6f)  "
             "dominant_ratio=%.4f  %s\n",
             j, dtheta_numeric.x(), dtheta_numeric.y(), dtheta_numeric.z(),
             phi_R_analytic.x(), phi_R_analytic.y(), phi_R_analytic.z(), ratio, pass ? "PASS" : "FAIL");
    }
    // c_acc_j -> position rows (3:6), column 3*j.
    {
      std::vector<V3D> c_acc_pert = c_acc;
      c_acc_pert[j] = V3D(0, 0, eps);
      CoupledPropagation prop1;
      propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0, pos0, vel0,
                        gravity, gravity0, delta_bg, delta_ba, c_acc_pert, c_gyr, n_c, prop1);
      // Numerical dp/dc_acc at the ENDPOINT (exact, no interpolation needed --
      // simpler than probing an intermediate t_k, and still a real test of
      // phi_c's position-row correctness since t1 is just another bracket
      // point interpolatePhi() must get right).
      const V3D dp_numeric = (prop1.pos1 - prop0.pos1) / eps;
      const Eigen::Matrix<double, 9, Eigen::Dynamic> Phic_t1 = interpolatePhi(prop0, scan_end_time);
      const V3D phi_P_analytic = Phic_t1.block<3, 1>(3, 3 * j + 2);
      const double ratio = phi_P_analytic.z() / (dp_numeric.z() + 1e-30);
      const bool pass = std::abs(ratio - 1.0) < 0.05;
      all_pass = all_pass && pass;
      printf("c_acc[j=%d,z]  dp_numeric=(%.6f %.6f %.6f)  phi_P_analytic=(%.6f %.6f %.6f)  "
             "dominant_ratio=%.4f (at t1)  %s\n",
             j, dp_numeric.x(), dp_numeric.y(), dp_numeric.z(),
             phi_P_analytic.x(), phi_P_analytic.y(), phi_P_analytic.z(), ratio, pass ? "PASS" : "FAIL");
    }
  }

  // Also check t1 endpoint itself, as a sanity cross-check against the
  // known-working phi_at_scan_end path.
  {
    const V3D delta_bg_pert(0, 0, eps);
    CoupledPropagation prop1;
    propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0, pos0, vel0,
                      gravity, gravity0, delta_bg_pert, delta_ba, c_acc, c_gyr, n_c, prop1);
    const V3D dtheta_numeric = Log(prop0.rot1.transpose() * prop1.rot1) / eps;
    const V3D phi_R_analytic = prop0.phi_x_head.back().block<3, 1>(0, 9 + 2);
    const double ratio = phi_R_analytic.z() / (dtheta_numeric.z() + 1e-30);
    const bool pass = std::abs(ratio - 1.0) < 0.05;
    all_pass = all_pass && pass;
    printf("t1 check (z axis): dtheta_numeric=(%.6f %.6f %.6f)  phi_R_analytic=(%.6f %.6f %.6f)  %s\n",
           dtheta_numeric.x(), dtheta_numeric.y(), dtheta_numeric.z(),
           phi_R_analytic.x(), phi_R_analytic.y(), phi_R_analytic.z(),
           pass ? "PASS" : "FAIL");
  }

  printf(all_pass ? "ALL PASS\n" : "SOME FAILED\n");
  return all_pass ? 0 : 1;
}
