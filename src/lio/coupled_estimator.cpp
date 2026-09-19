#include "livo_recon/lio/coupled_estimator.h"

namespace livo_recon
{
namespace
{
// Same cubic weights as spline.cpp's basisU(), duplicated (that one is
// file-local to spline.cpp) -- see coupled_estimator.h's header comment for
// why this module does not share ScanSpline's basis.
inline void cubicB(double u, double b[4])
{
  const double u2 = u * u, u3 = u2 * u, om = 1.0 - u;
  b[0] = om * om * om / 6.0;
  b[1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
  b[2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
  b[3] = u3 / 6.0;
}

// First control-point index + local u for `t`, n_c control points, n_seg =
// n_c - 3 segments over [t0,t1]. Clamps outside the window, matching
// ScanSpline::basisAt()'s own clamp convention.
inline void bracket(double t, int n_c, double t0, double t1, int& first_cp, double& u)
{
  const int n_seg = n_c - 3;
  const double delta = (t1 - t0) / static_cast<double>(n_seg);
  const double tc = std::min(std::max(t, t0), t1);
  double x = (tc - t0) / delta;
  int s = static_cast<int>(std::floor(x));
  if (s >= n_seg) { s = n_seg - 1; x = static_cast<double>(n_seg); }
  if (s < 0)      { s = 0;          x = 0.0; }
  u = x - static_cast<double>(s);
  first_cp = s;
}
}  // namespace

double basisWeight(int j, int n_c, double t0, double t1, double t)
{
  int first_cp; double u;
  bracket(t, n_c, t0, t1, first_cp, u);
  const int local = j - first_cp;
  if (local < 0 || local > 3) return 0.0;
  double b[4];
  cubicB(u, b);
  return b[local];
}

void propagateCoupled(const std::vector<Pose6D>& raw_poses,
                      const M3D& raw_rot1,
                      double scan_end_time,
                      const M3D& rot0, const V3D& pos0, const V3D& vel0,
                      const V3D& gravity,
                      const std::vector<V3D>& c_acc, const std::vector<V3D>& c_gyr,
                      int n_c, CoupledPropagation& out)
{
  out.poses.clear();
  out.phi_head.clear();
  const int N = static_cast<int>(raw_poses.size());
  out.poses.reserve(N);
  out.phi_head.reserve(N + 1);

  const double t0 = N > 0 ? raw_poses.front().t : scan_end_time;
  const double t1 = scan_end_time;
  const int ncol = 6 * n_c;

  M3D R = rot0;
  V3D p = pos0, v = vel0;
  Eigen::Matrix<double, 9, Eigen::Dynamic> Phi(9, ncol);
  Phi.setZero();
  out.phi_head.push_back(Phi);

  for (int k = 0; k < N; ++k)
  {
    const Pose6D& seg = raw_poses[k];
    const double dt = seg.dt, dt2 = dt * dt;
    const double t_head = seg.t, t_tail = seg.t + dt;

    // Recover the RAW body-frame accel measurement (bias-corrected, not yet
    // rotated) at head/tail from the ORIGINAL raw-chain rotations -- see
    // header comment. seg.rot is the raw chain's own head rotation for this
    // segment; its tail rotation is the NEXT segment's head, or (for the
    // last segment) the raw chain's own endpoint, raw_rot1.
    const M3D& R_head_raw = seg.rot;
    const M3D& R_tail_raw = (k + 1 < N) ? raw_poses[k + 1].rot : raw_rot1;
    const V3D a_body_head = R_head_raw.transpose() * (seg.acc_head - gravity);
    const V3D a_body_tail = R_tail_raw.transpose() * (seg.acc_tail - gravity);

    // Basis weights at this segment's own head/tail times, per coefficient.
    // wa/wg[j] pairs (head,tail); the "_avr" convention (0.5*(head+tail))
    // mirrors angvel_avr's own averaging exactly (imu_processing.cpp).
    std::vector<double> wa_head(n_c), wa_tail(n_c), wg_head(n_c), wg_tail(n_c);
    for (int j = 0; j < n_c; ++j)
    {
      wa_head[j] = basisWeight(j, n_c, t0, t1, t_head);
      wa_tail[j] = basisWeight(j, n_c, t0, t1, t_tail);
      wg_head[j] = wa_head[j];
      wg_tail[j] = wa_tail[j];
    }

    V3D delta_a_head = V3D::Zero(), delta_a_tail = V3D::Zero();
    V3D delta_w_head = V3D::Zero(), delta_w_tail = V3D::Zero();
    for (int j = 0; j < n_c; ++j)
    {
      delta_a_head += wa_head[j] * c_acc[j];
      delta_a_tail += wa_tail[j] * c_acc[j];
      delta_w_head += wg_head[j] * c_gyr[j];
      delta_w_tail += wg_tail[j] * c_gyr[j];
    }
    const V3D angvel_avr = seg.gyr + 0.5 * (delta_w_head + delta_w_tail);

    // ---- nominal (corrected) propagation, mirroring imu_processing.cpp ----
    const M3D Exp_f = Exp(angvel_avr, dt);
    const V3D acc_world_head = R * (a_body_head + delta_a_head) + gravity;
    const M3D R_tail = R * Exp_f;
    const V3D acc_world_tail = R_tail * (a_body_tail + delta_a_tail) + gravity;
    const V3D acc_avr_world = 0.5 * (acc_world_head + acc_world_tail);

    const M3D R_head = R;  // this segment's head (before the rotation update)
    const V3D p_head = p, v_head = v;

    p = p + v * dt + 0.5 * acc_avr_world * dt2;
    v = v + acc_avr_world * dt;
    R = R_tail;

    // ---- store the Pose6D for this segment (deskewPoints()'s own shape:
    // head-time snapshot + the acc_head/acc_tail this segment's interior
    // interpolates between) ----
    out.poses.push_back(Pose6D{
        t_head, acc_world_head, acc_world_tail, angvel_avr,
        v_head, p_head, R_head, dt});

    // ---- Jacobian recursion: Phi_{k+1} = F_x * Phi_k + G_k ----
    // F_x: SAME 9x9 (R,P,V) block ImuProc::propagate() uses for its own P
    // recursion (second_order terms included, gravity/bias columns dropped
    // -- c has no direct sensitivity there). acc_avr_skew uses the
    // CORRECTED acc_avr in body frame at head time, matching F_x's own
    // acc_avr (there: measured minus bias; here: measured plus correction,
    // same role in the skew term).
    const V3D acc_avr_body_head = a_body_head + delta_a_head;
    M3D acc_avr_skew; acc_avr_skew << SKEW_SYM_MATRX(acc_avr_body_head);

    Eigen::Matrix<double, 9, 9> Fx = Eigen::Matrix<double, 9, 9>::Zero();
    Fx.block<3, 3>(0, 0) = Exp_f.transpose();                          // R<-R
    Fx.block<3, 3>(3, 6) = M3D::Identity() * dt;                       // P<-V
    Fx.block<3, 3>(3, 0) = -0.5 * R_head * acc_avr_skew * dt2;         // P<-R (2nd order)
    Fx.block<3, 3>(6, 0) = -R_head * acc_avr_skew * dt;                // V<-R

    Eigen::Matrix<double, 9, Eigen::Dynamic> G(9, ncol);
    G.setZero();
    for (int j = 0; j < n_c; ++j)
    {
      // c_acc_j columns [3j, 3j+3): direct V and P sensitivity (mirrors the
      // bias-accel role -Rot*dt / -0.5*Rot*dt^2 would have, sign-flipped
      // since a correction ADDS where bias SUBTRACTS), basis-weighted per
      // head/tail rather than the uniform per-dof identity a bias uses.
      const M3D Wa = 0.5 * (wa_head[j] * R_head + wa_tail[j] * R_tail);
      G.block<3, 3>(6, 3 * j) = Wa * dt;
      G.block<3, 3>(3, 3 * j) = 0.5 * Wa * dt2;

      // c_gyr_j columns [3n_c+3j, 3n_c+3j+3): direct rotation sensitivity.
      const double wg = 0.5 * (wg_head[j] + wg_tail[j]);
      G.block<3, 3>(0, 3 * n_c + 3 * j) = M3D::Identity() * wg * dt;
    }

    Phi = Fx * Phi + G;
    out.phi_head.push_back(Phi);
  }

  out.rot1 = R; out.pos1 = p; out.vel1 = v;
}

Eigen::Matrix<double, 9, Eigen::Dynamic> interpolatePhi(
    const CoupledPropagation& prop, double t)
{
  const int N = static_cast<int>(prop.poses.size());
  if (N == 0) return prop.phi_head.empty()
      ? Eigen::Matrix<double, 9, Eigen::Dynamic>()
      : prop.phi_head[0];
  if (t <= prop.poses.front().t) return prop.phi_head.front();
  const double t1 = prop.poses.back().t + prop.poses.back().dt;
  if (t >= t1) return prop.phi_head.back();

  for (int k = 0; k < N; ++k)
  {
    const double th = prop.poses[k].t, tt = th + prop.poses[k].dt;
    if (t >= th && t <= tt)
    {
      const double alpha = (prop.poses[k].dt > 1e-12) ? (t - th) / prop.poses[k].dt : 0.0;
      return (1.0 - alpha) * prop.phi_head[k] + alpha * prop.phi_head[k + 1];
    }
  }
  return prop.phi_head.back();
}

}  // namespace livo_recon
