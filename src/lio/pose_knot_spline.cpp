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

void integrateAndAccumulateStep(
    const ImuSample& head, const ImuSample& tail,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    const V3D& var_acc, const V3D& var_gyr,
    double q_alpha_acc, double q_alpha_gyr, bool second_order,
    M3D& rot_imu, V3D& pos_imu, V3D& vel_imu,
    Eigen::Matrix<double, 9, 9>& F_seg, Eigen::Matrix<double, 9, 9>& Q_seg)
{
  const double dt = tail.t - head.t;
  if (!(dt > 0.0)) return;
  // EXACT match to ImuProc::propagate()'s own per-sample quantities.
  const V3D acc_avr = 0.5 * (head.acc + tail.acc) - bias_acc;
  const V3D angvel_avr = 0.5 * (head.gyro + tail.gyro) - bias_gyr;

  Eigen::Matrix<double, 9, 9> F9, Q9;
  buildImuStep9x9(rot_imu, acc_avr, angvel_avr, dt, var_acc, var_gyr,
                  q_alpha_acc, q_alpha_gyr, second_order, F9, Q9);
  F_seg = F9 * F_seg;
  Q_seg = F9 * Q_seg * F9.transpose() + Q9;

  // State update -- EXACT match to ImuProc::propagate()'s own
  // acc_world_head/Exp_f/acc_world_tail/pos_imu/vel_imu construction (uses
  // rot_imu BEFORE the update for the head term, after for the tail term).
  const V3D acc_world_head = rot_imu * (head.acc - bias_acc) + gravity;
  const M3D Exp_f = Exp(angvel_avr, dt);
  rot_imu = rot_imu * Exp_f;
  const V3D acc_world_tail = rot_imu * (tail.acc - bias_acc) + gravity;
  const V3D acc_avr_world = 0.5 * (acc_world_head + acc_world_tail);
  pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_avr_world * dt * dt;
  vel_imu = vel_imu + acc_avr_world * dt;
}

bool PoseKnotSpline::init(double t0, double t1, int n_knots,
                          const V3D& p0, const M3D& R0, const V3D& v0,
                          const Eigen::Matrix<double, 9, 9>& P0,
                          const std::vector<ImuSample>& imu_raw,
                          const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
                          double q_alpha_acc, double q_alpha_gyr,
                          const V3D& var_acc, const V3D& var_gyr, bool second_order)
{
  valid_ = false;
  knots_.clear();
  seg_F9_.clear();
  seg_Q9_.clear();
  if (n_knots < 2 || imu_raw.size() < 2 || !(t1 > t0)) return false;

  // Item 3: knot times, evenly spaced.
  knots_.resize(n_knots);
  const double dt_knot = (t1 - t0) / static_cast<double>(n_knots - 1);
  for (int j = 0; j < n_knots; ++j) knots_[j].t = t0 + j * dt_knot;
  knots_.back().t = t1;  // avoid roundoff drift on the last knot

  // POST-REVIEW FIX: knot 0 is the caller's own ESIKF state (item 18) --
  // P_0 = P0 (soft prior, consumed by the solver's own prior factor, not
  // a hard clamp). Every LATER knot's nominal (pos,rot,vel) is now the
  // ACTUAL INTEGRATED RESULT of walking imu_raw from knot 0 forward,
  // interpolating a boundary sample at each knot time crossed -- so
  // knot[j+1].{pos,rot,vel} == f(knot[j].{pos,rot,vel}, raw samples in
  // [t_j,t_{j+1})) EXACTLY, by construction, rather than an independently
  // interpolated mg.poses value that could (and did) disagree with what
  // the process factor's own F9/Q9 implied. This is what makes
  // estimateCoupledPoseKnotSpline()'s process-factor residual
  // (x_{j+1}-F_j*x_j, in CORRECTION variables) genuinely centered at the
  // true nominal-trajectory inconsistency (zero, here) rather than
  // silently assuming it away.
  knots_[0].pos = p0; knots_[0].rot = R0; knots_[0].vel = v0;
  knots_[0].P_prior = P0;

  seg_F9_.assign(n_knots - 1, Eigen::Matrix<double, 9, 9>::Identity());
  seg_Q9_.assign(n_knots - 1, Eigen::Matrix<double, 9, 9>::Zero());

  M3D rot_imu = R0;
  V3D pos_imu = p0, vel_imu = v0;
  Eigen::Matrix<double, 9, 9> P_running = P0;
  Eigen::Matrix<double, 9, 9> F_seg = Eigen::Matrix<double, 9, 9>::Identity();
  Eigen::Matrix<double, 9, 9> Q_seg = Eigen::Matrix<double, 9, 9>::Zero();

  int knot_idx = 1;
  ImuSample head = imu_raw.front();
  for (size_t i = 1; i < imu_raw.size() && knot_idx < n_knots; ++i) {
    ImuSample tail = imu_raw[i];
    // Interpolate a boundary sample at EVERY knot time this [head,tail]
    // micro-interval crosses (mirroring ImuProc::propagate()'s own
    // single-boundary-at-t_curr interpolation, generalized to N-1
    // internal boundaries) -- fixes the pre-fix version's "segment
    // boundaries do not align with knot times" bug.
    while (knot_idx < n_knots && knots_[knot_idx].t <= tail.t + 1e-9) {
      const double kt = knots_[knot_idx].t;
      ImuSample boundary;
      if (kt >= tail.t - 1e-12) {
        boundary = tail;
      } else {
        const double a = (kt - head.t) / std::max(tail.t - head.t, 1e-12);
        boundary.t = kt;
        boundary.acc = (1.0 - a) * head.acc + a * tail.acc;
        boundary.gyro = (1.0 - a) * head.gyro + a * tail.gyro;
      }
      integrateAndAccumulateStep(head, boundary, bias_acc, bias_gyr, gravity,
                                 var_acc, var_gyr, q_alpha_acc, q_alpha_gyr, second_order,
                                 rot_imu, pos_imu, vel_imu, F_seg, Q_seg);
      knots_[knot_idx].pos = pos_imu;
      knots_[knot_idx].rot = rot_imu;
      knots_[knot_idx].vel = vel_imu;
      P_running = F_seg * P_running * F_seg.transpose() + Q_seg;
      knots_[knot_idx].P_prior = P_running;
      seg_F9_[knot_idx - 1] = F_seg;
      seg_Q9_[knot_idx - 1] = Q_seg;
      F_seg.setIdentity();
      Q_seg.setZero();
      head = boundary;
      ++knot_idx;
      if (boundary.t >= tail.t - 1e-12) break;  // boundary WAS tail -- nothing left in this micro-step
    }
    if (knot_idx >= n_knots) break;
    if (tail.t > head.t + 1e-12) {
      integrateAndAccumulateStep(head, tail, bias_acc, bias_gyr, gravity,
                                 var_acc, var_gyr, q_alpha_acc, q_alpha_gyr, second_order,
                                 rot_imu, pos_imu, vel_imu, F_seg, Q_seg);
      head = tail;
    }
  }
  // valid_ requires every knot to have actually been reached by the raw
  // IMU stream -- imu_raw is documented (imu_processing.cpp) to span the
  // whole scan window [t0,t1], and knots_.back().t==t1 exactly, so this
  // should always succeed when imu_raw itself is well-formed; refuse
  // loudly (report false) rather than leave trailing knots at their
  // default-constructed Zero()/Identity() state if it doesn't.
  valid_ = (knot_idx >= n_knots);
  return valid_;
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
