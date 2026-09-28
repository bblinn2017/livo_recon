#include "livo_recon/diagnostics/init_consistency.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/algo/chi_square.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace livo_recon
{
namespace
{

// P(chi2 <= bound) == p, for the three required confidence levels. 0.6827 is
// the exact 1-DOF identity (P(|z|<=1)=0.6827 for z~N(0,1), i.e. chi2<=1 at
// DOF=1) generalized to every DOF via the same chi-square quantile function,
// rather than a "68%"-labelled table value that would only be exact at
// DOF=1.
constexpr double kP68 = 0.6827, kP95 = 0.95, kP99 = 0.99;

std::array<bool, 3> chi2Flags(double eps, bool valid, int dof)
{
  if (!valid || !std::isfinite(eps))
    return {false, false, false};
  return {eps <= chiSquareInverseCdf(kP68, dof),
          eps <= chiSquareInverseCdf(kP95, dof),
          eps <= chiSquareInverseCdf(kP99, dof)};
}

template <int N>
Eigen::Matrix<double, N, 1> eigenvaluesOf(const Eigen::Matrix<double, N, N>& m)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es(m);
  return es.eigenvalues();
}

}  // namespace

InitConsistencyResult computeInitConsistency(
    const M3D& R_ref, const V3D& p_ref, const V3D& v_ref,
    const M3D& R, const V3D& p, const V3D& v,
    const Eigen::MatrixXd& P_RPV)
{
  InitConsistencyResult r;

  // Errors, in the estimator's right/body-frame perturbation convention:
  // Log(R_ref^T R) is the tangent-space vector that right-multiplies R_ref
  // (via Exp) to reach R -- the same convention applyDelta()/
  // gravityAlignedAttitudeCovariance() use.
  r.eR = Log(M3D(R_ref.transpose() * R));
  r.ep = p - p_ref;
  r.ev = v - v_ref;
  r.eR_norm = r.eR.norm();
  r.ep_norm = r.ep.norm();
  r.ev_norm = r.ev.norm();

  // Gravity/yaw axis in the reference's own tangent frame.
  const V3D gravity_world(0.0, 0.0, -1.0);
  const V3D a = (R_ref.transpose() * gravity_world).normalized();
  r.yaw_error = a.dot(r.eR);
  r.tilt = r.eR - r.yaw_error * a;
  r.tilt_norm = r.tilt.norm();

  // Exact (nonlinear) gravity-tilt angle -- independent of the linearized
  // tangent-space decomposition above, per INSTRUCTIONS.md's explicit
  // "retain the exact angle separately" requirement.
  const V3D cur_gravity_body = R.transpose() * gravity_world;
  r.exact_gravity_tilt = std::acos(std::clamp(a.dot(cur_gravity_body), -1.0, 1.0));

  const M3D P_RR = P_RPV.block<3, 3>(0, 0);
  const M3D P_PP = P_RPV.block<3, 3>(3, 3);
  const M3D P_VV = P_RPV.block<3, 3>(6, 6);
  const M3D P_RP = P_RPV.block<3, 3>(0, 3);
  const M3D P_RV = P_RPV.block<3, 3>(0, 6);
  const M3D P_PV = P_RPV.block<3, 3>(3, 6);

  r.Prr_trace = P_RR.trace();
  r.Ppp_trace = P_PP.trace();
  r.Pvv_trace = P_VV.trace();
  {
    const V3D e = eigenvaluesOf<3>(P_RR);
    r.Prr_eig = {e(0), e(1), e(2)};
  }
  {
    const V3D e = eigenvaluesOf<3>(P_PP);
    r.Ppp_eig = {e(0), e(1), e(2)};
  }
  {
    const V3D e = eigenvaluesOf<3>(P_VV);
    r.Pvv_eig = {e(0), e(1), e(2)};
  }
  r.Prp_fro = P_RP.norm();
  r.Prv_fro = P_RV.norm();
  r.Ppv_fro = P_PV.norm();

  // Orthonormal basis for the tangent plane perpendicular to the gravity
  // axis. Any orthonormal basis works -- computeInitConsistency's eps_tilt/
  // tilt2d_eig are provably invariant to the choice (a similarity
  // transform under a change of orthonormal basis; see
  // test_joint_knot_estimator.cpp's basis-invariance test).
  const V3D u1 = a.unitOrthogonal();
  const V3D u2 = a.cross(u1);
  Eigen::Matrix<double, 3, 2> U;
  U.col(0) = u1;
  U.col(1) = u2;
  const Eigen::Matrix2d P_tilt2D = U.transpose() * P_RR * U;
  const V3D e_tilt_2d_3 = U * (U.transpose() * r.tilt);  // unused, kept for clarity
  (void)e_tilt_2d_3;
  const Eigen::Vector2d e_tilt_2d = U.transpose() * r.tilt;
  {
    const Eigen::Vector2d e = eigenvaluesOf<2>(P_tilt2D);
    r.tilt2d_eig = {e(0), e(1)};
  }
  r.yaw_variance = a.dot(P_RR * a);

  {
    const Eigen::VectorXd e = P_RPV.selfadjointView<Eigen::Upper>().eigenvalues();
    r.Prpv_min_eig = e.minCoeff();
    r.Prpv_max_eig = e.maxCoeff();
    r.Prpv_condition = (r.Prpv_min_eig > 0.0)
        ? r.Prpv_max_eig / r.Prpv_min_eig
        : std::numeric_limits<double>::infinity();
  }

  std::vector<double> pivots;

  {
    Eigen::LDLT<M3D> ldlt(P_RR);
    r.rr_solve_valid = ldlt.info() == Eigen::Success && ldlt.isPositive();
    if (r.rr_solve_valid) {
      r.eps_R = r.eR.dot(ldlt.solve(r.eR));
      pivots.push_back(ldlt.vectorD().minCoeff());
      // Yaw's denominator a^T P_RR a is guaranteed strictly positive
      // whenever P_RR itself is PD (a is a unit vector) -- no separate
      // yaw_solve_valid column is needed; it is exactly rr_solve_valid.
      if (r.yaw_variance > 0.0) r.eps_yaw = (r.yaw_error * r.yaw_error) / r.yaw_variance;
    }
  }
  {
    Eigen::LDLT<M3D> ldlt(P_PP);
    r.pp_solve_valid = ldlt.info() == Eigen::Success && ldlt.isPositive();
    if (r.pp_solve_valid) {
      r.eps_p = r.ep.dot(ldlt.solve(r.ep));
      pivots.push_back(ldlt.vectorD().minCoeff());
    }
  }
  {
    Eigen::LDLT<M3D> ldlt(P_VV);
    r.vv_solve_valid = ldlt.info() == Eigen::Success && ldlt.isPositive();
    if (r.vv_solve_valid) {
      r.eps_v = r.ev.dot(ldlt.solve(r.ev));
      pivots.push_back(ldlt.vectorD().minCoeff());
    }
  }
  {
    Eigen::LDLT<Eigen::Matrix2d> ldlt(P_tilt2D);
    r.tilt_solve_valid = ldlt.info() == Eigen::Success && ldlt.isPositive();
    if (r.tilt_solve_valid) {
      r.eps_tilt = e_tilt_2d.dot(ldlt.solve(e_tilt_2d));
      pivots.push_back(ldlt.vectorD().minCoeff());
    }
  }
  {
    Eigen::VectorXd e_rpv(9);
    e_rpv << r.eR, r.ep, r.ev;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(P_RPV);
    r.rpv_solve_valid = ldlt.info() == Eigen::Success && ldlt.isPositive();
    if (r.rpv_solve_valid) {
      r.eps_RPV = e_rpv.dot(ldlt.solve(e_rpv));
      pivots.push_back(ldlt.vectorD().minCoeff());
    }
  }
  r.min_ldlt_pivot = pivots.empty()
      ? std::numeric_limits<double>::quiet_NaN()
      : *std::min_element(pivots.begin(), pivots.end());

  r.chi2_R = chi2Flags(r.eps_R, r.rr_solve_valid, 3);
  r.chi2_p = chi2Flags(r.eps_p, r.pp_solve_valid, 3);
  r.chi2_v = chi2Flags(r.eps_v, r.vv_solve_valid, 3);
  r.chi2_tilt = chi2Flags(r.eps_tilt, r.tilt_solve_valid, 2);
  r.chi2_yaw = chi2Flags(r.eps_yaw, r.rr_solve_valid, 1);
  r.chi2_RPV = chi2Flags(r.eps_RPV, r.rpv_solve_valid, 9);

  return r;
}

namespace
{

void writeRow(std::ofstream& out, const std::string& run_id, int scan_id, double t_abs,
              const char* phase, double init_pos, double init_vel, double init_rot_tilt,
              double init_rot_yaw, double init_gravity, double init_bg, double init_ba,
              int residual_count, int completed_iterations,
              const InitConsistencyResult& r, const InitConsistencyResult* prev)
{
  const auto b = [](bool v) { return v ? 1 : 0; };
  out << std::setprecision(17)
      << run_id << ',' << scan_id << ',' << t_abs << ',' << phase << ','
      << init_pos << ',' << init_vel << ',' << init_rot_tilt << ',' << init_rot_yaw << ','
      << init_gravity << ',' << init_bg << ',' << init_ba << ','
      << residual_count << ',' << completed_iterations << ','
      << r.eR.x() << ',' << r.eR.y() << ',' << r.eR.z() << ',' << r.eR_norm << ','
      << r.ep.x() << ',' << r.ep.y() << ',' << r.ep.z() << ',' << r.ep_norm << ','
      << r.ev.x() << ',' << r.ev.y() << ',' << r.ev.z() << ',' << r.ev_norm << ','
      << r.tilt.x() << ',' << r.tilt.y() << ',' << r.tilt.z() << ',' << r.tilt_norm << ','
      << r.yaw_error << ',' << r.exact_gravity_tilt << ','
      << r.Prr_trace << ',' << r.Prr_eig[0] << ',' << r.Prr_eig[1] << ',' << r.Prr_eig[2] << ','
      << r.Ppp_trace << ',' << r.Ppp_eig[0] << ',' << r.Ppp_eig[1] << ',' << r.Ppp_eig[2] << ','
      << r.Pvv_trace << ',' << r.Pvv_eig[0] << ',' << r.Pvv_eig[1] << ',' << r.Pvv_eig[2] << ','
      << r.Prp_fro << ',' << r.Prv_fro << ',' << r.Ppv_fro << ','
      << r.tilt2d_eig[0] << ',' << r.tilt2d_eig[1] << ',' << r.yaw_variance << ','
      << r.Prpv_min_eig << ',' << r.Prpv_max_eig << ',' << r.Prpv_condition << ','
      << r.eps_R << ',' << r.eps_p << ',' << r.eps_v << ',' << r.eps_tilt << ','
      << r.eps_yaw << ',' << r.eps_RPV << ','
      << b(r.rr_solve_valid) << ',' << b(r.pp_solve_valid) << ',' << b(r.vv_solve_valid) << ','
      << b(r.tilt_solve_valid) << ',' << b(r.rpv_solve_valid) << ',' << r.min_ldlt_pivot << ','
      << b(r.chi2_R[0]) << ',' << b(r.chi2_R[1]) << ',' << b(r.chi2_R[2]) << ','
      << b(r.chi2_p[0]) << ',' << b(r.chi2_p[1]) << ',' << b(r.chi2_p[2]) << ','
      << b(r.chi2_v[0]) << ',' << b(r.chi2_v[1]) << ',' << b(r.chi2_v[2]) << ','
      << b(r.chi2_tilt[0]) << ',' << b(r.chi2_tilt[1]) << ',' << b(r.chi2_tilt[2]) << ','
      << b(r.chi2_yaw[0]) << ',' << b(r.chi2_yaw[1]) << ',' << b(r.chi2_yaw[2]) << ','
      << b(r.chi2_RPV[0]) << ',' << b(r.chi2_RPV[1]) << ',' << b(r.chi2_RPV[2]);
  if (prev == nullptr) {
    // post_imu row: no "change from post_imu" to report.
    out << ",,,,,,,,\n";
  } else {
    const double d_er = r.eR_norm - prev->eR_norm;
    const double d_ep = r.ep_norm - prev->ep_norm;
    const double d_ev = r.ev_norm - prev->ev_norm;
    const double d_Prr = r.Prr_trace - prev->Prr_trace;
    const double d_Ppp = r.Ppp_trace - prev->Ppp_trace;
    const double d_Pvv = r.Pvv_trace - prev->Pvv_trace;
    const double error_size_before = std::sqrt(prev->eR_norm * prev->eR_norm +
        prev->ep_norm * prev->ep_norm + prev->ev_norm * prev->ev_norm);
    const double error_size_after = std::sqrt(r.eR_norm * r.eR_norm +
        r.ep_norm * r.ep_norm + r.ev_norm * r.ev_norm);
    const double cov_size_before = prev->Prr_trace + prev->Ppp_trace + prev->Pvv_trace;
    const double cov_size_after = r.Prr_trace + r.Ppp_trace + r.Pvv_trace;
    const bool error_contracted = error_size_after < error_size_before;
    const bool cov_contracted = cov_size_after < cov_size_before;
    out << ',' << d_er << ',' << d_ep << ',' << d_ev << ','
        << d_Prr << ',' << d_Ppp << ',' << d_Pvv << ','
        << b(error_contracted && cov_contracted) << ','
        << b(!error_contracted && cov_contracted) << '\n';
  }
}

}  // namespace

void writeInitializationConsistencyDiagnostics(
    const std::string& run_id, int scan_id, double t_abs,
    double init_pos, double init_vel, double init_rot_tilt, double init_rot_yaw,
    double init_gravity, double init_bg, double init_ba,
    int residual_count, int completed_iterations,
    const M3D& R_ref, const V3D& p_ref, const V3D& v_ref,
    const M3D& R_post_imu, const V3D& p_post_imu, const V3D& v_post_imu,
    const Eigen::MatrixXd& P_RPV_post_imu,
    const M3D& R_post_lio, const V3D& p_post_lio, const V3D& v_post_lio,
    const Eigen::MatrixXd& P_RPV_post_lio)
{
  static PersistentLogStream log("initialization_consistency_all_scans.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "run_id,scan_id,t_abs,phase,"
           "init_pos,init_vel,init_rot_tilt,init_rot_yaw,init_gravity,init_bg,init_ba,"
           "residual_count,completed_iterations,"
           "er_x,er_y,er_z,er_norm,ep_x,ep_y,ep_z,ep_norm,ev_x,ev_y,ev_z,ev_norm,"
           "tilt_x,tilt_y,tilt_z,tilt_norm,yaw_error,exact_gravity_tilt,"
           "Prr_trace,Prr_eig1,Prr_eig2,Prr_eig3,"
           "Ppp_trace,Ppp_eig1,Ppp_eig2,Ppp_eig3,"
           "Pvv_trace,Pvv_eig1,Pvv_eig2,Pvv_eig3,"
           "Prp_fro,Prv_fro,Ppv_fro,"
           "tilt2d_eig1,tilt2d_eig2,yaw_variance,"
           "Prpv_min_eig,Prpv_max_eig,Prpv_condition,"
           "eps_R,eps_p,eps_v,eps_tilt,eps_yaw,eps_RPV,"
           "rr_solve_valid,pp_solve_valid,vv_solve_valid,tilt_solve_valid,rpv_solve_valid,"
           "min_ldlt_pivot,"
           "chi2_R_68,chi2_R_95,chi2_R_99,"
           "chi2_p_68,chi2_p_95,chi2_p_99,"
           "chi2_v_68,chi2_v_95,chi2_v_99,"
           "chi2_tilt_68,chi2_tilt_95,chi2_tilt_99,"
           "chi2_yaw_68,chi2_yaw_95,chi2_yaw_99,"
           "chi2_RPV_68,chi2_RPV_95,chi2_RPV_99,"
           "delta_er_norm,delta_ep_norm,delta_ev_norm,"
           "delta_Prr_trace,delta_Ppp_trace,delta_Pvv_trace,"
           "error_and_covariance_both_contracted,error_increased_while_covariance_contracted\n";

  const InitConsistencyResult post_imu = computeInitConsistency(
      R_ref, p_ref, v_ref, R_post_imu, p_post_imu, v_post_imu, P_RPV_post_imu);
  const InitConsistencyResult post_lio = computeInitConsistency(
      R_ref, p_ref, v_ref, R_post_lio, p_post_lio, v_post_lio, P_RPV_post_lio);

  writeRow(out, run_id, scan_id, t_abs, "post_imu", init_pos, init_vel, init_rot_tilt,
           init_rot_yaw, init_gravity, init_bg, init_ba, residual_count,
           completed_iterations, post_imu, nullptr);
  writeRow(out, run_id, scan_id, t_abs, "post_lio", init_pos, init_vel, init_rot_tilt,
           init_rot_yaw, init_gravity, init_bg, init_ba, residual_count,
           completed_iterations, post_lio, &post_imu);
  out.flush();
}

}  // namespace livo_recon
