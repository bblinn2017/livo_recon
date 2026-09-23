#include "livo_recon/lio/pose_control_process_factor.h"

namespace livo_recon
{

namespace
{
inline M3D skew3v(const V3D& v)
{
  M3D S;
  S <<     0, -v.z(),  v.y(),
        v.z(),      0, -v.x(),
       -v.y(),  v.x(),      0;
  return S;
}

// SO(3) inverse right Jacobian -- needed because the theta residual is
// r_theta = Log(rot_pred^T R_{j+1}), and d(Log(M*Exp(eps*d)))/deps =
// JrInv(Log(M)) * d, NOT the identity, unless Log(M) is exactly zero.
// "dr/dx_{j+1}=I / dr/dx_j=-F9" (spec item 7) is the standard boxminus
// convention EXPRESSED IN THE LOCAL TANGENT AT THE RESIDUAL ITSELF -- this
// JrInv(r_theta) factor is what carries that local tangent frame back to
// the actual Log() residual reported, and was the source of a measured
// ~3.5% cross-axis Jacobian error before being added (confirmed via
// test_pose_control_process_factor.cpp's FD check -- see that file).
M3D jrInvLocal(const V3D& phi)
{
  const double n = phi.norm();
  M3D K = skew3v(phi);
  double c;
  if (n < 1e-4) c = 1.0 / 12.0 + n * n / 720.0;
  else c = 1.0 / (n * n) - (1.0 + std::cos(n)) / (2.0 * n * std::sin(n));
  return M3D::Identity() + 0.5 * K + c * K * K;
}

ImuSample lerpSample(const ImuSample& head, const ImuSample& tail, double t)
{
  const double a = (t - head.t) / std::max(tail.t - head.t, 1e-12);
  ImuSample s;
  s.t = t;
  s.acc = (1.0 - a) * head.acc + a * tail.acc;
  s.gyro = (1.0 - a) * head.gyro + a * tail.gyro;
  return s;
}
}  // namespace

PoseControlImuSegments bucketPoseControlImuSamples(
    const std::vector<ImuSample>& imu_raw, const PoseControlSpline& spline)
{
  PoseControlImuSegments out;
  const int n_seg = spline.nSeg();
  out.seg_samples.assign(n_seg, {});
  if (imu_raw.size() < 2 || n_seg < 1) return out;

  std::vector<double> bt(n_seg + 1);
  for (int j = 0; j <= n_seg; ++j) bt[j] = spline.t0() + j * spline.delta();
  bt[n_seg] = spline.t1();

  int seg_idx = 0;
  ImuSample head = imu_raw.front();
  out.seg_samples[0].push_back(head);
  for (size_t i = 1; i < imu_raw.size() && seg_idx < n_seg; ++i)
  {
    ImuSample tail = imu_raw[i];
    while (seg_idx < n_seg && bt[seg_idx + 1] <= tail.t + 1e-9)
    {
      const double bt_j = bt[seg_idx + 1];
      ImuSample boundary = (bt_j >= tail.t - 1e-12) ? tail : lerpSample(head, tail, bt_j);
      out.seg_samples[seg_idx].push_back(boundary);
      ++seg_idx;
      if (seg_idx < n_seg) out.seg_samples[seg_idx].push_back(boundary);
      head = boundary;
    }
    if (seg_idx < n_seg) { out.seg_samples[seg_idx].push_back(tail); head = tail; }
  }
  return out;
}

void relinearizePoseControlSegment(
    const std::vector<ImuSample>& samples,
    const M3D& rot_j, const V3D& pos_j, const V3D& vel_j,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9,
    M3D& rot_pred, V3D& pos_pred, V3D& vel_pred)
{
  F9 = Eigen::Matrix<double, 9, 9>::Identity();
  Q9 = Eigen::Matrix<double, 9, 9>::Zero();
  M3D rot_imu = rot_j;
  V3D pos_imu = pos_j, vel_imu = vel_j;
  for (size_t k = 0; k + 1 < samples.size(); ++k)
    integrateAndAccumulateStep(samples[k], samples[k + 1], bias_acc, bias_gyr, gravity,
                               var_acc, var_gyr, q_alpha_acc, q_alpha_gyr, second_order,
                               rot_imu, pos_imu, vel_imu, F9, Q9);
  rot_pred = rot_imu; pos_pred = pos_imu; vel_pred = vel_imu;
}

namespace
{
// One micro-step, EXACT duplicate of integrateAndAccumulateStep()'s own
// arithmetic (pose_knot_spline.cpp) so F9/Q9/state-update stay identical --
// but additionally returns the LOCAL bias/gravity partials needed to chain
// G9 = F9_step*G9 + L_step. See this file's header comment for the
// derivation; this function IS that derivation, one line per term.
void imuStepWithBiasJac(
    const ImuSample& head, const ImuSample& tail,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    const V3D& var_acc, const V3D& var_gyr,
    double q_alpha_acc, double q_alpha_gyr, bool second_order,
    M3D& rot_imu, V3D& pos_imu, V3D& vel_imu,
    Eigen::Matrix<double, 9, 9>& F_seg, Eigen::Matrix<double, 9, 9>& Q_seg,
    Eigen::Matrix<double, 9, 9>& G_seg)
{
  const double dt = tail.t - head.t;
  if (!(dt > 0.0)) return;
  const V3D acc_avr = 0.5 * (head.acc + tail.acc) - bias_acc;
  const V3D angvel_avr = 0.5 * (head.gyro + tail.gyro) - bias_gyr;

  Eigen::Matrix<double, 9, 9> F9, Q9;
  buildImuStep9x9(rot_imu, acc_avr, angvel_avr, dt, var_acc, var_gyr,
                  q_alpha_acc, q_alpha_gyr, second_order, F9, Q9);

  const M3D rot_k = rot_imu;   // pre-step rotation, R_k
  const V3D acc_world_head = rot_k * (head.acc - bias_acc) + gravity;
  const M3D Exp_f = Exp(angvel_avr, dt);
  const M3D rot_new = rot_k * Exp_f;
  const V3D acc_world_tail = rot_new * (tail.acc - bias_acc) + gravity;
  const V3D acc_avr_world = 0.5 * (acc_world_head + acc_world_tail);

  // Local partials (entering state R_k/p_k/v_k held fixed).
  const M3D dR_local = -dt * Jr(angvel_avr * dt);   // d(theta_new)/d(bg), local

  Eigen::Matrix<double, 3, 9> dAccAvrWorld = Eigen::Matrix<double, 3, 9>::Zero();
  // d/d(bg): only through rot_new's OWN local dependence on bg, via the
  // tail term (head term uses R_k, no local bg dependence).
  dAccAvrWorld.block<3, 3>(0, 0) =
      -0.5 * rot_new * skew3v(tail.acc - bias_acc) * dR_local;
  // d/d(ba): direct, both head (via R_k) and tail (via rot_new) terms.
  dAccAvrWorld.block<3, 3>(0, 3) = -0.5 * (rot_k + rot_new);
  // d/d(g): direct, both terms contribute I.
  dAccAvrWorld.block<3, 3>(0, 6) = M3D::Identity();

  Eigen::Matrix<double, 9, 9> L_step = Eigen::Matrix<double, 9, 9>::Zero();
  L_step.block<3, 3>(0, 0) = dR_local;                       // theta row, bg col
  L_step.block<3, 9>(3, 0) = 0.5 * dt * dt * dAccAvrWorld;   // pos row
  L_step.block<3, 9>(6, 0) = dt * dAccAvrWorld;              // vel row

  G_seg = F9 * G_seg + L_step;
  F_seg = F9 * F_seg;
  Q_seg = F9 * Q_seg * F9.transpose() + Q9;

  rot_imu = rot_new;
  pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_avr_world * dt * dt;
  vel_imu = vel_imu + acc_avr_world * dt;
}
}  // namespace

void relinearizePoseControlSegmentWithBiasJac(
    const std::vector<ImuSample>& samples,
    const M3D& rot_j, const V3D& pos_j, const V3D& vel_j,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9,
    Eigen::Matrix<double, 9, 9>& G9,
    M3D& rot_pred, V3D& pos_pred, V3D& vel_pred)
{
  F9 = Eigen::Matrix<double, 9, 9>::Identity();
  Q9 = Eigen::Matrix<double, 9, 9>::Zero();
  G9 = Eigen::Matrix<double, 9, 9>::Zero();
  M3D rot_imu = rot_j;
  V3D pos_imu = pos_j, vel_imu = vel_j;
  for (size_t k = 0; k + 1 < samples.size(); ++k)
    imuStepWithBiasJac(samples[k], samples[k + 1], bias_acc, bias_gyr, gravity,
                       var_acc, var_gyr, q_alpha_acc, q_alpha_gyr, second_order,
                       rot_imu, pos_imu, vel_imu, F9, Q9, G9);
  rot_pred = rot_imu; pos_pred = pos_imu; vel_pred = vel_imu;
}

Eigen::Matrix<double, 9, 9> poseControlPseudoInverse9(
    const Eigen::Matrix<double, 9, 9>& M, double rel_thresh)
{
  const Eigen::Matrix<double, 9, 9> Ms = 0.5 * (M + M.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(Ms);
  const auto& evals = es.eigenvalues();
  const auto& evecs = es.eigenvectors();
  const double thresh = rel_thresh * std::max(evals.maxCoeff(), 0.0);
  Eigen::Matrix<double, 9, 9> out = Eigen::Matrix<double, 9, 9>::Zero();
  for (int k = 0; k < 9; ++k)
    if (evals(k) > thresh) out += (1.0 / evals(k)) * (evecs.col(k) * evecs.col(k).transpose());
  return out;
}

void addPoseControlProcessFactor(
    const PoseControlSpline& spline, int j,
    const std::vector<ImuSample>& samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double q_alpha_acc, double q_alpha_gyr,
    const V3D& var_acc, const V3D& var_gyr, bool second_order,
    double q_pinv_rel_thresh,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    double* out_E_process)
{
  const int N = spline.N();
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

  // Nonlinear residual, current trial vs. current trial's own re-integrated
  // prediction -- never discarded (spec item 7).
  Eigen::Matrix<double, 9, 1> r;
  r.segment<3>(0) = Log(M3D(rot_pred.transpose() * Rj1));
  r.segment<3>(3) = pj1 - pos_pred;
  r.segment<3>(6) = vj1 - vel_pred;

  const Eigen::Matrix<double, 9, 9> Lambda =
      poseControlPseudoInverse9(Q9, q_pinv_rel_thresh);

  // J_{x_j}, J_{x_{j+1}}: 9 x 6N, nonzero only in the 4 active control
  // points' columns at tj/tj1 respectively (position rows use c_p
  // columns, rotation rows use c_phi columns -- exactly PoseControlSpline's
  // own decoupling).
  auto jacAt = [&](double t) { return spline.jacobianAt(t); };
  const auto jac_j = jacAt(tj);
  const auto jac_j1 = jacAt(tj1);

  // dr/dx_j = -F9, dr/dx_{j+1} = I (spec item 7). Build the two 9x12
  // "active-columns-only" Jacobians (4 control points x 3 axes each for
  // BOTH p and phi -- 4*3 + 4*3 = 24 columns, but position/rotation rows
  // only touch their own 12) then chain through -F9 / I.
  // We assemble directly into a dense 9 x 24 block per breakpoint using
  // the fact position rows [0:3) depend only on c_p columns and rotation
  // rows... wait: r's rotation block depends on Rj1 (c_phi at j+1 side)
  // AND (through rot_pred, which is a function of Rj i.e. c_phi at j) --
  // rot_pred's own dependence on c_phi[j-side] is folded into F9's
  // rotation-rotation block by construction (F9 IS d(rot_pred)/d(theta_j)
  // in the propagation's own error-state convention), so J_{x_j}'s theta
  // row uses dThetaDcphi exactly as the propagation model expects.
  Eigen::MatrixXd Jxj = Eigen::MatrixXd::Zero(9, 6 * N);
  Eigen::MatrixXd Jxj1 = Eigen::MatrixXd::Zero(9, 6 * N);
  for (int k = 0; k < 4; ++k)
  {
    const int col_p_j  = jac_j.s + k, col_p_j1  = jac_j1.s + k;
    const int col_ph_j = jac_j.s + k, col_ph_j1 = jac_j1.s + k;
    // theta row block [0:3), pos row block [3:6), vel row block [6:9)
    Jxj.block<3, 3>(0, 3 * N + 3 * col_ph_j) = spline.dThetaDcphi(jac_j, k, tj);
    Jxj.block<3, 3>(3, 3 * col_p_j)          = PoseControlSpline::dPosDcp(jac_j, k);
    Jxj.block<3, 3>(6, 3 * col_p_j)          = PoseControlSpline::dVelDcp(jac_j, k);

    Jxj1.block<3, 3>(0, 3 * N + 3 * col_ph_j1) = spline.dThetaDcphi(jac_j1, k, tj1);
    Jxj1.block<3, 3>(3, 3 * col_p_j1)          = PoseControlSpline::dPosDcp(jac_j1, k);
    Jxj1.block<3, 3>(6, 3 * col_p_j1)          = PoseControlSpline::dVelDcp(jac_j1, k);
  }

  Eigen::MatrixXd Jcur = Jxj1 - F9 * Jxj;    // 9 x 6N (correct as-is for the
                                              // position/velocity rows 3-8:
                                              // plain Euclidean residuals).
  // Theta row (0-2) is r_theta = Log(g), g = rot_pred^T R_{j+1} -- a
  // manifold residual, so "dr/dx=I / -F9" needs the boxminus chain rule,
  // NOT a raw linear combination:
  //   perturbing R_{j+1} by a RIGHT delta u1: g -> g*Exp(u1)
  //     => dr/du1 = JrInv(r_theta) * u1
  //   perturbing R_j by a RIGHT delta u_j (rot_pred's own right-
  //     perturbation model, exactly what F9's theta-row encodes):
  //     rot_pred -> rot_pred*Exp(F9_theta_row * u_j)
  //     => g -> Exp(-F9_theta_row*u_j) * g = g*Exp(-g^T F9_theta_row u_j)
  //        (using Exp(w)*R = R*Exp(R^T w))
  //     => dr/du_j = -JrInv(r_theta) * g^T * F9_theta_row * u_j
  // Confirmed empirically: the naive uniform "JrInv(r)*(Jxj1-F9*Jxj)"
  // (missing the g^T factor on the j-side term) measured WORSE against
  // FD (2.8e-2) than omitting the correction entirely (1.76e-2) -- this
  // g^T-corrected form is what test_pose_control_process_factor.cpp's FD
  // check validates.
  {
    const M3D g = rot_pred.transpose() * Rj1;
    const Eigen::MatrixXd theta_row_j  = F9.block<3, 9>(0, 0) * Jxj;   // 3 x 6N
    const Eigen::MatrixXd theta_row_j1 = Jxj1.block(0, 0, 3, Jxj1.cols());
    Jcur.block(0, 0, 3, Jcur.cols()) =
        jrInvLocal(r.segment<3>(0)) * (theta_row_j1 - g.transpose() * theta_row_j);
  }
  const Eigen::MatrixXd JtL = Jcur.transpose() * Lambda;

  A += JtL * Jcur;
  b += -JtL * r;

  if (out_E_process) *out_E_process += 0.5 * r.dot(Lambda * r);
}

}  // namespace livo_recon
