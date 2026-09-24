#include "livo_recon/lio/imu_process_step9.h"

namespace livo_recon
{

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
  const V3D acc_avr = 0.5 * (head.acc + tail.acc) - bias_acc;
  const V3D angvel_avr = 0.5 * (head.gyro + tail.gyro) - bias_gyr;

  Eigen::Matrix<double, 9, 9> F9, Q9;
  buildImuStep9x9(rot_imu, acc_avr, angvel_avr, dt, var_acc, var_gyr,
                  q_alpha_acc, q_alpha_gyr, second_order, F9, Q9);
  F_seg = F9 * F_seg;
  Q_seg = F9 * Q_seg * F9.transpose() + Q9;

  const V3D acc_world_head = rot_imu * (head.acc - bias_acc) + gravity;
  const M3D Exp_f = Exp(angvel_avr, dt);
  rot_imu = rot_imu * Exp_f;
  const V3D acc_world_tail = rot_imu * (tail.acc - bias_acc) + gravity;
  const V3D acc_avr_world = 0.5 * (acc_world_head + acc_world_tail);
  pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_avr_world * dt * dt;
  vel_imu = vel_imu + acc_avr_world * dt;
}

}  // namespace livo_recon
