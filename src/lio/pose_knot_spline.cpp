#include "livo_recon/lio/pose_knot_spline.h"

#include <algorithm>
#include <cmath>

namespace livo_recon
{

namespace
{

// Cubic Hermite basis (and derivatives, w.r.t. the NORMALIZED parameter u)
// -- item 3's "interpolating Hermite-like representation" for position,
// exact at both endpoints in value AND (via h10/h11) first derivative.
struct HermiteBasis { double h00, h10, h01, h11; };
HermiteBasis hermite(double u)
{
  const double u2 = u * u, u3 = u2 * u;
  return { 2*u3 - 3*u2 + 1, u3 - 2*u2 + u, -2*u3 + 3*u2, u3 - u2 };
}
HermiteBasis hermiteD1(double u)  // d/du
{
  const double u2 = u * u;
  return { 6*u2 - 6*u, 3*u2 - 4*u + 1, -6*u2 + 6*u, 3*u2 - 2*u };
}
HermiteBasis hermiteD2(double u)  // d2/du2
{
  return { 12*u - 6, 6*u - 4, -12*u + 6, 6*u - 2 };
}

}  // namespace

// See pose_knot_spline.h's own doc comment: a marginal [theta,p,v] 9x9
// block of the EXACT SAME F_x/cov_w formula imu_processing.cpp's
// ImuProc::propagate() builds (that file's own acc_avr_skew/Exp_f/
// second_order construction, lines ~223-277 as of this writing) -- copied
// rather than a live shared call site, bias/gravity rows/columns simply
// absent (held fixed, item 22).
void buildImuStep9x9(const M3D& rot_imu, const V3D& acc_avr, const V3D& angvel_avr,
                     double dt, const V3D& var_acc, const V3D& var_gyr,
                     double q_alpha_acc, double q_alpha_gyr, bool second_order,
                     Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9)
{
  constexpr int iR = 0, iP = 3, iV = 6;
  const double dt2 = dt * dt;
  M3D acc_avr_skew; acc_avr_skew << SKEW_SYM_MATRX(acc_avr);
  const M3D acc_noise_world = rot_imu * var_acc.asDiagonal() * rot_imu.transpose();
  const M3D Exp_f = Exp(angvel_avr, dt);

  F9.setIdentity();
  Q9.setZero();

  // Rotation
  F9.block<3, 3>(iR, iR) = Exp_f.transpose();
  // Position
  F9.block<3, 3>(iP, iV) = M3D::Identity() * dt;
  if (second_order)
    F9.block<3, 3>(iP, iR) += -0.5 * rot_imu * acc_avr_skew * dt2;
  // Velocity
  F9.block<3, 3>(iV, iR) = -rot_imu * acc_avr_skew * dt;

  // Rotation noise
  Q9.block<3, 3>(iR, iR).diagonal() = q_alpha_gyr * var_gyr * dt2;
  // Velocity noise
  Q9.block<3, 3>(iV, iV) = q_alpha_acc * acc_noise_world * dt2;
  if (second_order) {
    Q9.block<3, 3>(iP, iP) = q_alpha_acc * 0.25 * dt2 * dt2 * acc_noise_world;
    Q9.block<3, 3>(iP, iV) = q_alpha_acc * 0.5 * dt * dt2 * acc_noise_world;
    Q9.block<3, 3>(iV, iP) = Q9.block<3, 3>(iP, iV);
  }
}

bool PoseKnotSpline::init(const std::vector<Pose6D>& imu_poses, double t0, double t1,
                          int n_knots, const V3D& p0, const M3D& R0, const V3D& v0,
                          const Eigen::Matrix<double, 9, 9>& P0,
                          const std::vector<ImuSample>& /*imu_raw*/,
                          const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
                          double q_alpha_acc, double q_alpha_gyr,
                          const V3D& var_acc, const V3D& var_gyr, bool second_order)
{
  valid_ = false;
  knots_.clear();
  seg_F9_.clear();
  seg_Q9_.clear();
  if (n_knots < 2 || imu_poses.empty() || !(t1 > t0)) return false;

  // Item 3: knot times, evenly spaced. Item 3's own preferred alternative
  // ("reuse the exact control-point timing convention you've already
  // established") is the SAME formula ScanSpline's control_point_hz uses
  // when spread evenly -- kept as the plain, unambiguous n_knots form here
  // since this class owns its own knot count, not a derived rate.
  knots_.resize(n_knots);
  const double dt_knot = (t1 - t0) / static_cast<double>(n_knots - 1);
  for (int j = 0; j < n_knots; ++j) knots_[j].t = t0 + j * dt_knot;
  knots_.back().t = t1;  // avoid roundoff drift on the last knot

  // Item 3: initialize p_j/R_j/v_j from the existing IMU-propagated chain
  // (imu_poses == mg.poses), linear-interpolating pos/vel and SLERPing
  // rot between the two bracketing Pose6D samples. imu_poses stores each
  // sample's HEAD state; the true tail-end state is (p0,R0,v0) themselves
  // (the caller's own current ESIKF propagated state, mg.image.t == t1).
  auto poseAtHead = [&](double t, V3D& p, M3D& R, V3D& v) {
    if (t <= imu_poses.front().t) { p = imu_poses.front().pos; R = imu_poses.front().rot; v = imu_poses.front().vel; return; }
    for (size_t i = 0; i + 1 < imu_poses.size(); ++i) {
      if (t >= imu_poses[i].t && t <= imu_poses[i + 1].t) {
        const double a = (t - imu_poses[i].t) / std::max(imu_poses[i + 1].t - imu_poses[i].t, 1e-12);
        p = (1 - a) * imu_poses[i].pos + a * imu_poses[i + 1].pos;
        v = (1 - a) * imu_poses[i].vel + a * imu_poses[i + 1].vel;
        const M3D dR = imu_poses[i].rot.transpose() * imu_poses[i + 1].rot;
        R = imu_poses[i].rot * Exp(V3D(a * Log(dR)));
        return;
      }
    }
    p = imu_poses.back().pos; R = imu_poses.back().rot; v = imu_poses.back().vel;
  };
  // Knot 0 is (p0,R0,v0) -- the caller's OWN scan-start ESIKF state,
  // per item 18/3 ("do not allow the initial spline and the ESIKF state to
  // start from different poses"). Interior knots are linear/SLERP-
  // interpolated from imu_poses (item 3). The LAST knot has no bracketing
  // Pose6D pair past it (imu_poses only stores HEAD states, up to but not
  // including t1) -- forward-integrate ONE final half-step from the last
  // Pose6D sample as its INITIAL GUESS ONLY (item 19: the tail is free to
  // move during the solve, so approximation quality here doesn't matter).
  knots_[0].pos = p0; knots_[0].rot = R0; knots_[0].vel = v0;
  for (int j = 1; j + 1 < n_knots; ++j) {
    V3D p, v; M3D R;
    poseAtHead(knots_[j].t, p, R, v);
    knots_[j].pos = p; knots_[j].rot = R; knots_[j].vel = v;
  }
  if (n_knots >= 2) {
    const auto& last = imu_poses.back();
    knots_[n_knots - 1].pos = last.pos + last.vel * last.dt;
    knots_[n_knots - 1].rot = last.rot * Exp(V3D(last.gyr * last.dt));
    knots_[n_knots - 1].vel = last.vel;
  }

  // Item 18: knot 0's prior covariance is the caller's own P0 (state_->
  // cov()'s [theta,p,v] block at scan start) -- a SOFT prior consumed by
  // the solver's own prior factor, not written here as a hard value.
  knots_[0].P = P0;

  // Items 5/6: causal forward propagation, knot j -> knot j+1, walking the
  // Pose6D samples already computed by ImuProc::propagate() (imu_poses)
  // that fall in [t_j, t_{j+1}) -- each Pose6D already carries dt/rot/
  // acc_head(world)/gyr(body, bias-corrected) at its own head time, so
  // buildImuStep9x9() is fed EXACTLY the same per-step inputs
  // ImuProc::propagate() itself used, just re-derived here (see that
  // function's own doc comment for why this is a copy, not a shared call).
  seg_F9_.assign(n_knots - 1, Eigen::Matrix<double, 9, 9>::Identity());
  seg_Q9_.assign(n_knots - 1, Eigen::Matrix<double, 9, 9>::Zero());
  size_t sample_idx = 0;
  for (int j = 0; j + 1 < n_knots; ++j) {
    Eigen::Matrix<double, 9, 9> F_seg = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 9> Q_seg = Eigen::Matrix<double, 9, 9>::Zero();
    while (sample_idx < imu_poses.size() && imu_poses[sample_idx].t < knots_[j + 1].t - 1e-9) {
      const auto& ps = imu_poses[sample_idx];
      if (ps.t >= knots_[j].t - 1e-9) {
        const V3D acc_avr_body = ps.rot.transpose() * (ps.acc_head - gravity);
        Eigen::Matrix<double, 9, 9> F9, Q9;
        buildImuStep9x9(ps.rot, acc_avr_body, ps.gyr, ps.dt, var_acc, var_gyr,
                        q_alpha_acc, q_alpha_gyr, second_order, F9, Q9);
        F_seg = F9 * F_seg;
        Q_seg = F9 * Q_seg * F9.transpose() + Q9;
      }
      ++sample_idx;
    }
    seg_F9_[j] = F_seg;
    seg_Q9_[j] = Q_seg;
    knots_[j + 1].P = F_seg * knots_[j].P * F_seg.transpose() + Q_seg;
  }
  (void)bias_acc; (void)bias_gyr;  // held fixed (item 22); accepted for interface symmetry/future use

  valid_ = true;
  return true;
}

void PoseKnotSpline::bracket(double t, int& j, double& u) const
{
  const int n = nKnots();
  j = 0;
  for (int k = 0; k + 1 < n; ++k) { if (t <= knots_[k + 1].t) { j = k; break; } j = k; }
  const double dt = knots_[j + 1].t - knots_[j].t;
  u = (dt > 1e-12) ? (t - knots_[j].t) / dt : 0.0;
  u = std::min(std::max(u, 0.0), 1.0);
}

V3D PoseKnotSpline::positionAt(double t) const
{
  int j; double u;
  bracket(t, j, u);
  const auto& a = knots_[j]; const auto& b = knots_[j + 1];
  const double dt = b.t - a.t;
  const HermiteBasis h = hermite(u);
  return h.h00 * a.pos + h.h10 * dt * a.vel + h.h01 * b.pos + h.h11 * dt * b.vel;
}

V3D PoseKnotSpline::velocityAt(double t) const
{
  int j; double u;
  bracket(t, j, u);
  const auto& a = knots_[j]; const auto& b = knots_[j + 1];
  const double dt = b.t - a.t;
  const HermiteBasis dh = hermiteD1(u);
  const V3D dpdu = dh.h00 * a.pos + dh.h10 * dt * a.vel + dh.h01 * b.pos + dh.h11 * dt * b.vel;
  return dpdu / std::max(dt, 1e-12);
}

V3D PoseKnotSpline::accelerationAt(double t) const
{
  int j; double u;
  bracket(t, j, u);
  const auto& a = knots_[j]; const auto& b = knots_[j + 1];
  const double dt = b.t - a.t;
  const HermiteBasis d2h = hermiteD2(u);
  const V3D d2pdu2 = d2h.h00 * a.pos + d2h.h10 * dt * a.vel + d2h.h01 * b.pos + d2h.h11 * dt * b.vel;
  return d2pdu2 / std::max(dt * dt, 1e-12);
}

M3D PoseKnotSpline::rotationAt(double t) const
{
  int j; double u;
  bracket(t, j, u);
  const auto& a = knots_[j]; const auto& b = knots_[j + 1];
  const V3D phi_rel = Log(M3D(a.rot.transpose() * b.rot));
  return a.rot * Exp(V3D(u * phi_rel));
}

V3D PoseKnotSpline::angularVelocityAt(double t) const
{
  int j; double u;
  bracket(t, j, u);
  const auto& a = knots_[j]; const auto& b = knots_[j + 1];
  const double dt = std::max(b.t - a.t, 1e-12);
  const V3D phi_rel = Log(M3D(a.rot.transpose() * b.rot));
  // d/dt [Exp(u*phi_rel)] = Jr(u*phi_rel) * phi_rel/dt, in the LOCAL
  // (body-at-t) frame -- the standard SO(3) geodesic angular-velocity
  // identity, constant magnitude direction, varying only through Jr.
  return Jr(V3D(u * phi_rel)) * (phi_rel / dt);
}

}  // namespace livo_recon
