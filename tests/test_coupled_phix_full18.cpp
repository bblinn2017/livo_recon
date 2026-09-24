// CQ-71 item 4 (#5, #6): full 18-direction finite-difference verification of
// propagateCoupled()'s phi_x_head against a numerical re-propagation, at
// scan endpoint t1, for BOTH a low-rotation and a high-rotation synthetic
// scan (total_dtheta_deg > 0.3 deg/segment -- CQ-55's own Bug-A onset
// threshold -- for the "high" case). Complements test_coupled_phi_fd.cpp,
// which only perturbs delta_bg (3 of the 18 columns); this covers all six
// 3-column blocks [delta_phi0, delta_p0, delta_v0, delta_bg, delta_ba,
// delta_g] against all three 3-row output blocks [dtheta, dp, dv], per
// coupled_estimator.h's own doc comment on phi_x_head's column layout.
//
// #6 (rotation-convention sign check) is answered by the delta_phi0 block's
// own result here: perturbing rot0 by a KNOWN delta_phi directly (not via
// delta_bg, unlike the existing test) and checking
// Log(R_perturbed^T R_nominal) against Phix_R * delta_phi is exactly the
// delta_phi0 input-block / dtheta output-block cell of the grid below --
// reported as a SIGNED ratio, not a norm, per the card's own instruction
// (a left/right convention mismatch shows as a sign flip, not a magnitude
// error).
//
// RESULT (both scenarios, forward AND central difference -- identical to 4
// significant figures either way, ruling out FD truncation error as the
// cause): 15 of 18 (input_block, output_block) cells match to <1%, most to
// <1e-8. THREE cells sit at a small, eps-independent, non-truncation
// mismatch: delta_ba->dp (~5.0%, both scenarios -- this one IS already
// named in coupled_estimator.cpp's own Gx comment: "idxP<-idxBA omitted"),
// and delta_bg->dp (~7.5-7.7%) / delta_bg->dv (~5.0-5.4%), which are NOT
// named in that comment but are the same class of omission: Gx only
// injects delta_bg's DIRECT rotation-sensitivity (R<-delta_bg); the
// resulting P/V sensitivity is expected to accumulate INDIRECTLY through
// the Fx*Phix recursion's own P<-R/V<-R blocks each step, and a small,
// consistent residual (this magnitude, this eps-independent) is left
// unaccounted for. Recommend coupled_estimator.cpp's Gx comment be
// extended to name idxP<-idxBG/idxV<-idxBG as the same class of documented
// simplification idxP<-idxBA already is -- a documentation gap, not
// something this diagnostic run itself fixes.
//
// No ROS/PCL/OpenCV, Eigen-only -- same convention as this directory's
// other tests. Never linked into livo_recon_node; pure standalone
// diagnostic tooling, zero risk to any shipped behavior.
//
//   g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
//       tests/test_coupled_phix_full18.cpp src/lio/coupled_estimator.cpp \
//       -o /tmp/test_coupled_phix_full18 && /tmp/test_coupled_phix_full18
#include "livo_recon/lio/coupled_estimator.h"
#include <cstdio>
#include <vector>

using namespace livo_recon;

static std::vector<Pose6D> buildRawPoses(int N, double dt, const M3D& rot0,
                                          const V3D& pos0, const V3D& vel0,
                                          const V3D& gravity0, const V3D& gyr,
                                          const V3D& acc_body)
{
  std::vector<Pose6D> poses;
  poses.reserve(N);
  M3D R = rot0;
  V3D p = pos0, v = vel0;
  for (int k = 0; k < N; ++k)
  {
    const double t = k * dt;
    const V3D acc_head = R * acc_body + gravity0;
    const M3D R_tail = R * Exp(gyr, dt);
    const V3D acc_tail = R_tail * acc_body + gravity0;
    poses.push_back(Pose6D{t, acc_head, acc_tail, gyr, v, p, R, dt});
    const V3D acc_avr_world = 0.5 * (acc_head + acc_tail);
    p = p + v * dt + 0.5 * acc_avr_world * dt * dt;
    v = v + acc_avr_world * dt;
    R = R_tail;
  }
  return poses;
}

// Runs the full 18-direction FD check for one gyro-rate scenario. Returns
// true iff every (input_block, output_block) cell's max relative error is
// under the 5% tolerance the existing test file already uses.
static bool runScenario(const char* label, const V3D& gyr, const V3D& acc_body)
{
  const int N = 20;
  const double dt = 0.005;
  const double per_segment_deg = gyr.norm() * dt * (180.0 / M_PI);
  printf("=== %s  (gyr_norm=%.4f rad/s, %.4f deg/segment) ===\n",
         label, gyr.norm(), per_segment_deg);

  const M3D rot0 = M3D::Identity();
  const V3D pos0 = V3D::Zero(), vel0 = V3D(0.3, -0.2, 0.1);
  const V3D gravity0(0, 0, -9.81);
  const V3D gravity = gravity0;
  const V3D delta_bg = V3D::Zero(), delta_ba = V3D::Zero();
  const int n_c = 4;
  std::vector<V3D> c_acc(n_c, V3D::Zero()), c_gyr(n_c, V3D::Zero());

  std::vector<Pose6D> raw_poses = buildRawPoses(N, dt, rot0, pos0, vel0, gravity0, gyr, acc_body);
  const M3D raw_rot1 = raw_poses.back().rot * Exp(raw_poses.back().gyr, dt);
  const double scan_end_time = N * dt;

  CoupledPropagation prop0;
  propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0, pos0, vel0,
                    gravity, gravity0, delta_bg, delta_ba, c_acc, c_gyr, n_c, prop0);

  const Eigen::Matrix<double, 9, 18>& Phix_t1 = prop0.phi_x_head.back();
  // 1e-6 matches test_coupled_phi_fd.cpp's own proven choice -- 1e-7
  // underflows into floating-point cancellation noise over this 20-segment
  // recursion (confirmed directly: at 1e-7 the delta_phi0 block's numeric
  // dtheta came back bit-exact zero, not merely small, while 1e-3 recovered
  // the expected ~eps-scale signal cleanly -- a finite-difference epsilon
  // pitfall in this test, not a defect in propagateCoupled() itself).
  const double eps = 1e-6;
  static const char* BLOCK_NAMES[6] = {
    "delta_phi0", "delta_p0", "delta_v0", "delta_bg", "delta_ba", "delta_g"};
  static const char* OUT_NAMES[3] = {"dtheta", "dp", "dv"};

  bool all_pass = true;
  double max_signed_ratio_delta_phi0_dtheta = 0.0;  // #6's own signed check

  for (int block = 0; block < 6; ++block)
  {
    double block_max_rel_err[3] = {0.0, 0.0, 0.0};  // per output block
    for (int axis = 0; axis < 3; ++axis)
    {
      V3D e = V3D::Zero(); e(axis) = eps;

      // Central difference: cancels O(eps) truncation error, leaving O(eps^2)
      // -- needed because a forward difference alone left three
      // second-order-coupled cells (delta_bg/delta_ba -> dp/dv) sitting at
      // 5-7.7% relative error, ambiguous against the 5% tolerance.
      auto perturbed = [&](double sign, M3D& rot0_p, V3D& pos0_p, V3D& vel0_p,
                            V3D& gravity_p, V3D& delta_bg_p, V3D& delta_ba_p) {
        rot0_p = rot0; pos0_p = pos0; vel0_p = vel0; gravity_p = gravity;
        delta_bg_p = delta_bg; delta_ba_p = delta_ba;
        const V3D se = sign * e;
        switch (block) {
          case 0: rot0_p = rot0 * Exp(se); break;               // delta_phi0
          case 1: pos0_p = pos0 + se; break;                    // delta_p0
          case 2: vel0_p = vel0 + se; break;                    // delta_v0
          case 3: delta_bg_p = delta_bg + se; break;             // delta_bg
          case 4: delta_ba_p = delta_ba + se; break;             // delta_ba
          case 5: gravity_p = gravity + se; break;               // delta_g (gravity0 fixed)
        }
      };

      M3D rot0_pp, rot0_pm; V3D pos0_pp, pos0_pm, vel0_pp, vel0_pm;
      V3D gravity_pp, gravity_pm, delta_bg_pp, delta_bg_pm, delta_ba_pp, delta_ba_pm;
      perturbed(+1.0, rot0_pp, pos0_pp, vel0_pp, gravity_pp, delta_bg_pp, delta_ba_pp);
      perturbed(-1.0, rot0_pm, pos0_pm, vel0_pm, gravity_pm, delta_bg_pm, delta_ba_pm);

      CoupledPropagation prop_p, prop_m;
      propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0_pp, pos0_pp, vel0_pp,
                        gravity_pp, gravity0, delta_bg_pp, delta_ba_pp, c_acc, c_gyr, n_c, prop_p);
      propagateCoupled(raw_poses, raw_rot1, scan_end_time, rot0_pm, pos0_pm, vel0_pm,
                        gravity_pm, gravity0, delta_bg_pm, delta_ba_pm, c_acc, c_gyr, n_c, prop_m);

      const V3D dtheta_num = Log(prop_m.rot1.transpose() * prop_p.rot1) / (2.0 * eps);
      const V3D dp_num = (prop_p.pos1 - prop_m.pos1) / (2.0 * eps);
      const V3D dv_num = (prop_p.vel1 - prop_m.vel1) / (2.0 * eps);

      const Eigen::Matrix<double, 9, 1> analytic_col = Phix_t1.col(3 * block + axis);
      const V3D dtheta_an = analytic_col.segment<3>(0);
      const V3D dp_an = analytic_col.segment<3>(3);
      const V3D dv_an = analytic_col.segment<3>(6);

      // #6's own signed check: delta_phi0 block (0), dtheta output (rows
      // 0-2), the SAME axis (a rotation perturbed about axis i should show
      // up dominantly in dtheta's own axis-i component).
      if (block == 0) {
        const double num = dtheta_num(axis), an = dtheta_an(axis);
        const double signed_ratio = an / (num + (num == 0.0 ? 1e-30 : 0.0));
        if (std::abs(signed_ratio) > std::abs(max_signed_ratio_delta_phi0_dtheta))
          max_signed_ratio_delta_phi0_dtheta = signed_ratio;
      }

      auto relErr = [](const V3D& a, const V3D& n) {
        const double denom = std::max({a.norm(), n.norm(), 1e-9});
        return (a - n).norm() / denom;
      };
      block_max_rel_err[0] = std::max(block_max_rel_err[0], relErr(dtheta_an, dtheta_num));
      block_max_rel_err[1] = std::max(block_max_rel_err[1], relErr(dp_an, dp_num));
      block_max_rel_err[2] = std::max(block_max_rel_err[2], relErr(dv_an, dv_num));

      // CQ-71 follow-up: report a SIGNED ratio (analytic/numeric, dominant
      // component) for the three known-mismatching cells, same construction
      // as #6's own signed check above -- lets a reader confirm the
      // mismatch is a small consistent UNDER/OVER-count (ratio near but not
      // at 1.0), not a sign flip (which would show as a negative ratio).
      auto signedRatioDominant = [](const V3D& an, const V3D& num) {
        int i_dom = 0; double best = -1.0;
        for (int i = 0; i < 3; ++i) {
          if (std::abs(an(i)) > best) { best = std::abs(an(i)); i_dom = i; }
        }
        const double n = num(i_dom);
        return an(i_dom) / (n + (n == 0.0 ? 1e-30 : 0.0));
      };
      if (block == 3) {  // delta_bg -> dp/dv, every perturbation axis
        printf("    [signed axis=%d] delta_bg  -> dp  ratio=%.6f   delta_bg -> dv  ratio=%.6f\n",
               axis, signedRatioDominant(dp_an, dp_num), signedRatioDominant(dv_an, dv_num));
      }
      if (block == 4) {  // delta_ba -> dp, every perturbation axis
        printf("    [signed axis=%d] delta_ba  -> dp  ratio=%.6f\n",
               axis, signedRatioDominant(dp_an, dp_num));
      }
    }
    for (int out = 0; out < 3; ++out) {
      const bool pass = block_max_rel_err[out] < 0.05;
      all_pass = all_pass && pass;
      printf("  %-11s -> %-6s  max_rel_err=%.6e  %s\n",
             BLOCK_NAMES[block], OUT_NAMES[out], block_max_rel_err[out], pass ? "PASS" : "FAIL");
    }
  }
  printf("  #6 signed check (delta_phi0 -> dtheta, dominant axis): ratio=%.6f "
         "(near +1.0 = standard right/body convention confirmed; near -1.0 "
         "would mean a left-tangent/sign-flipped convention)\n",
         max_signed_ratio_delta_phi0_dtheta);
  printf("  %s: %s\n\n", label, all_pass ? "ALL PASS" : "SOME FAILED");
  return all_pass;
}

int main()
{
  bool all_pass = true;
  // LOW rotation: well under CQ-55's 0.3 deg/segment Bug-A onset threshold.
  all_pass &= runScenario("LOW rotation", V3D(0.01, -0.005, 0.02), V3D(0.1, -0.05, 9.81));
  // HIGH rotation: clearly over the 0.3 deg/segment threshold (gyr_z=2.0
  // rad/s * 0.005s = 0.573 deg/segment).
  all_pass &= runScenario("HIGH rotation", V3D(0.3, -0.2, 2.0), V3D(0.5, -0.3, 9.85));

  printf(all_pass ? "ALL SCENARIOS PASS\n" : "SOME SCENARIOS FAILED\n");
  return all_pass ? 0 : 1;
}
