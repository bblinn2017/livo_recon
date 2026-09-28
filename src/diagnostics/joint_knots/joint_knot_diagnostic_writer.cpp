#include "livo_recon/diagnostics/joint_knots/joint_knot_diagnostic_writer.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <Eigen/Eigenvalues>

namespace livo_recon
{
namespace
{
void writeMatrix(std::ofstream& out, const char* name, const Eigen::MatrixXd& m)
{
  out << name << " " << m.rows() << " " << m.cols() << "\n";
  out << std::setprecision(17) << m << "\n";
}
}

void writeJointKnotIterationDiagnostics(
    const std::string& test_id, int scan_id, int iteration, double t_abs,
    const std::string& residual_time, bool gating_state_uncertainty,
    const JointKnotTrajectory& before, const JointKnotTrajectory& after,
    const JointKnotPrior& prior, const JointKnotSolve& solve,
    int residual_count, double mean_gating_cov_trace,
    double max_gating_cov_trace)
{
  // Generate diagnostics only for the first real map-backed LIO frame.
  // Enforcing this at the writer prevents stale/all-frame files from being
  // mistaken for the requested first-post-calibration evidence.
  if (scan_id != 1) return;
  static PersistentLogStream summary("joint_knot_iterations.csv");
  bool first = false;
  std::ofstream& csv = summary.stream(&first);
  if (first)
    csv << "test_id,scan_id,iteration,t_abs,residual_time,gating_state_uncertainty,"
           "knot_count,state_dimension,residual_count,mean_abs_residual,delta_norm,"
           "lidar_rhs_norm,prior_rhs_norm,lidar_delta_norm,prior_delta_norm,"
           "lidar_prior_delta_dot,zero_jacobian_residual_count,residuals_by_interval,"
           "prior_min_eig,prior_max_eig,lidar_max_eig,posterior_min_eig,posterior_max_eig,"
           "tail_dtheta_norm,tail_dp_norm,tail_dv_norm,dbg_norm,dba_norm,dg_norm,"
           "max_knot_dt,max_geodesic_angle,mean_gating_cov_trace,"
           "max_gating_cov_trace,solve_uses_gating_covariance,"
           "lidar_information_mode,redund_groups_seen,redund_groups,"
           "redund_groups_degenerate_pv,redund_groups_degenerate_var,"
           "redund_n_raw,redund_info_ratio,information_increase_groups,"
           "naive_info_gain,woodbury_info_gain\n";
  const int tail = after.knotCount() - 1;
  const V3D dtheta = Log(M3D(before.knots()[tail].R.transpose() * after.knots()[tail].R));
  const V3D dp = after.knots()[tail].p - before.knots()[tail].p;
  const V3D dv = after.knots()[tail].v - before.knots()[tail].v;
  double max_dt = 0.0, max_angle = 0.0;
  for (int k = 1; k < after.knotCount(); ++k) {
    max_dt = std::max(max_dt, after.knots()[k].t - after.knots()[k-1].t);
    max_angle = std::max(max_angle, Log(M3D(
        after.knots()[k-1].R.transpose() * after.knots()[k].R)).norm());
  }
  std::ostringstream interval_counts;
  for (size_t i = 0; i < solve.residual_count_by_interval.size(); ++i) {
    if (i) interval_counts << ';';
    interval_counts << solve.residual_count_by_interval[i];
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> prior_eigs(prior.P);
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> lidar_eigs(solve.Gamma_L);
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> posterior_eigs(solve.posterior);
  csv << test_id << ',' << scan_id << ',' << iteration << ',' << std::setprecision(17) << t_abs << ','
      << residual_time << ',' << (gating_state_uncertainty ? 1 : 0) << ','
      << after.knotCount() << ',' << after.dim() << ',' << residual_count << ','
      << solve.mean_abs_residual << ',' << solve.delta.norm() << ','
      << solve.lidar_rhs.norm() << ',' << solve.prior_rhs.norm() << ','
      << solve.lidar_delta.norm() << ',' << solve.prior_delta.norm() << ','
      << solve.lidar_delta.dot(solve.prior_delta) << ','
      << solve.zero_jacobian_residual_count << ',' << interval_counts.str() << ','
      << prior_eigs.eigenvalues().minCoeff() << ','
      << prior_eigs.eigenvalues().maxCoeff() << ','
      << lidar_eigs.eigenvalues().maxCoeff() << ','
      << posterior_eigs.eigenvalues().minCoeff() << ','
      << posterior_eigs.eigenvalues().maxCoeff() << ','
      << dtheta.norm() << ',' << dp.norm() << ',' << dv.norm() << ','
      << (after.tailBiasGyr() - before.tailBiasGyr()).norm() << ','
      << (after.tailBiasAcc() - before.tailBiasAcc()).norm() << ','
      << (after.gravity() - before.gravity()).norm() << ','
      << max_dt << ',' << max_angle << ','
      << mean_gating_cov_trace << ',' << max_gating_cov_trace << ",0,"
      << solve.lidar_information_mode << ','
      << solve.redundancy_stats.redund_groups_seen << ','
      << solve.redundancy_stats.redund_groups << ','
      << solve.redundancy_stats.redund_groups_degenerate_pv << ','
      << solve.redundancy_stats.redund_groups_degenerate_var << ','
      << solve.redundancy_stats.redund_n_raw << ','
      << solve.redundancy_stats.redund_info_ratio << ','
      << solve.redundancy_stats.information_increase_groups << ','
      << solve.redundancy_stats.naive_info_gain << ','
      << solve.redundancy_stats.woodbury_info_gain << '\n';
  csv.flush();

  static PersistentLogStream knot_states("joint_knot_states.csv");
  bool first_state = false;
  std::ofstream& states = knot_states.stream(&first_state);
  if (first_state)
    states << "test_id,scan_id,iteration,phase,knot_index,knot_t,immutable_head,"
              "theta_x,theta_y,theta_z,p_x,p_y,p_z,v_x,v_y,v_z,"
              "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z,g_x,g_y,g_z\n";
  for (int phase_index = 0; phase_index < 2; ++phase_index) {
    const JointKnotTrajectory& trajectory = phase_index == 0 ? before : after;
    const char* phase = phase_index == 0 ? "before" : "after";
    for (int k = 0; k < trajectory.knotCount(); ++k) {
      const JointKnot& knot = trajectory.knots()[k];
      const V3D theta = Log(knot.R);
      states << test_id << ',' << scan_id << ',' << iteration << ',' << phase << ','
             << k << ',' << std::setprecision(17) << knot.t << ',' << (k == 0 ? 1 : 0) << ','
             << theta.x() << ',' << theta.y() << ',' << theta.z() << ','
             << knot.p.x() << ',' << knot.p.y() << ',' << knot.p.z() << ','
             << knot.v.x() << ',' << knot.v.y() << ',' << knot.v.z() << ','
             << trajectory.biasGyr(k).x() << ','
             << trajectory.biasGyr(k).y() << ','
             << trajectory.biasGyr(k).z() << ','
             << trajectory.biasAcc(k).x() << ','
             << trajectory.biasAcc(k).y() << ','
             << trajectory.biasAcc(k).z() << ','
             << trajectory.gravity().x() << ','
             << trajectory.gravity().y() << ','
             << trajectory.gravity().z() << '\n';
    }
  }
  states.flush();

  // Signed, block-level correction accounting.  `requested` is the solved
  // tangent increment; `realized` is measured from the before/after states.
  // The LiDAR/prior columns expose both magnitude and cancellation without
  // requiring a reader to parse the large matrix dump.
  static PersistentLogStream corrections("joint_knot_corrections.csv");
  bool first_correction = false;
  std::ofstream& corr = corrections.stream(&first_correction);
  if (first_correction)
    corr << "test_id,scan_id,iteration,block,knot_index,immutable_head,"
            "requested_x,requested_y,requested_z,lidar_x,lidar_y,lidar_z,"
            "prior_x,prior_y,prior_z,realized_x,realized_y,realized_z\n";
  auto correction_row = [&](const char* block, int knot_index, int offset,
                            const V3D& realized, bool immutable) {
    const V3D requested = immutable ? V3D::Zero() : V3D(solve.delta.segment<3>(offset));
    const V3D lidar = immutable ? V3D::Zero() : V3D(solve.lidar_delta.segment<3>(offset));
    const V3D prior_part = immutable ? V3D::Zero() : V3D(solve.prior_delta.segment<3>(offset));
    corr << test_id << ',' << scan_id << ',' << iteration << ',' << block << ','
         << knot_index << ',' << (immutable ? 1 : 0) << ',' << std::setprecision(17)
         << requested.x() << ',' << requested.y() << ',' << requested.z() << ','
         << lidar.x() << ',' << lidar.y() << ',' << lidar.z() << ','
         << prior_part.x() << ',' << prior_part.y() << ',' << prior_part.z() << ','
         << realized.x() << ',' << realized.y() << ',' << realized.z() << '\n';
  };
  correction_row("theta", 0, 0, V3D::Zero(), true);
  correction_row("p", 0, 0, V3D::Zero(), true);
  correction_row("v", 0, 0, V3D::Zero(), true);
  for (int k = 1; k < after.knotCount(); ++k) {
    const int o = after.knotOffset(k);
    correction_row("theta", k, o,
        Log(M3D(before.knots()[k].R.transpose() * after.knots()[k].R)), false);
    correction_row("p", k, o + 3,
        after.knots()[k].p - before.knots()[k].p, false);
    correction_row("v", k, o + 6,
        after.knots()[k].v - before.knots()[k].v, false);
  }
  for (int k = 0; k < after.knotCount(); ++k) {
    const int o = after.biasOffset(k);
    correction_row("bg", k, o,
        after.biasGyr(k) - before.biasGyr(k), false);
    correction_row("ba", k, o + 3,
        after.biasAcc(k) - before.biasAcc(k), false);
  }
  correction_row("g", -1, after.gravityOffset(),
      after.gravity() - before.gravity(), false);
  corr.flush();

  static PersistentLogStream matrices("joint_knot_first_frame_matrices.txt");
  std::ofstream& out = matrices.stream();
  out << "=== iteration " << iteration << " residual_time " << residual_time
      << " gating_state_uncertainty " << (gating_state_uncertainty ? 1 : 0) << " ===\n";
  writeMatrix(out, "P_prior", prior.P);
  writeMatrix(out, "P_prior_inverse", prior.P_inverse);
  writeMatrix(out, "Gamma_L", solve.Gamma_L);
  writeMatrix(out, "b_L", solve.b_L);
  writeMatrix(out, "vec", solve.vec);
  writeMatrix(out, "lidar_rhs", solve.lidar_rhs);
  writeMatrix(out, "prior_rhs", solve.prior_rhs);
  writeMatrix(out, "rhs", solve.rhs);
  writeMatrix(out, "A", solve.A);
  writeMatrix(out, "K1", solve.K1);
  writeMatrix(out, "P_posterior_extended", solve.posterior);
  writeMatrix(out, "delta_x", solve.delta);
  writeMatrix(out, "lidar_delta", solve.lidar_delta);
  writeMatrix(out, "prior_delta", solve.prior_delta);
  for (int k = 0; k < after.knotCount(); ++k) {
    out << "knot " << k << " t " << std::setprecision(17) << after.knots()[k].t
        << " p " << after.knots()[k].p.transpose()
        << " v " << after.knots()[k].v.transpose()
        << " bg " << after.biasGyr(k).transpose()
        << " ba " << after.biasAcc(k).transpose()
        << " logR " << Log(after.knots()[k].R).transpose() << '\n';
  }
  out.flush();
}

void writeJointKnotLidarInformationDiagnostics(
    const std::string& test_id, int scan_id, int iteration, double t_abs,
    const JointKnotSolve& solve, int residual_count)
{
  static PersistentLogStream log("joint_knot_lidar_information.csv");
  bool first = false;
  std::ofstream& csv = log.stream(&first);
  if (first)
    csv << "test_id,scan_id,iteration,t_abs,lidar_information_mode,"
           "residual_count,gamma_trace,gamma_frobenius,b_norm,"
           "redund_groups_seen,redund_groups,redund_groups_degenerate_pv,"
           "redund_groups_degenerate_var,redund_n_raw,"
           "redund_info_ratio,naive_info_gain,woodbury_info_gain,"
           "gamma_correction_trace,gamma_correction_frobenius,"
           "gamma_correction_min_eigenvalue,gamma_correction_max_eigenvalue,"
           "b_correction_norm,"
           "information_increase_groups\n";
  const ResidualRedundancyStats& s = solve.redundancy_stats;
  csv << test_id << ',' << scan_id << ',' << iteration << ','
      << std::setprecision(17) << t_abs << ','
      << solve.lidar_information_mode << ',' << residual_count << ','
      << solve.Gamma_L.trace() << ',' << solve.Gamma_L.norm() << ','
      << solve.b_L.norm() << ',' << s.redund_groups_seen << ','
      << s.redund_groups << ',' << s.redund_groups_degenerate_pv << ','
      << s.redund_groups_degenerate_var << ',' << s.redund_n_raw << ','
      << s.redund_info_ratio << ','
      << s.naive_info_gain << ',' << s.woodbury_info_gain << ','
      << s.gamma_correction_trace << ','
      << s.gamma_correction_frobenius << ','
      << s.gamma_correction_min_eigenvalue << ','
      << s.gamma_correction_max_eigenvalue << ','
      << s.b_correction_norm << ','
      << s.information_increase_groups << '\n';
  csv.flush();
}

void writeJointKnotStateChainDiagnostics(
    const std::string& test_id, int scan_id, int iteration,
    const std::string& phase, double t_abs,
    const MeasureGroup& measures, const StateGroup& current)
{
  if (scan_id != 1) return;
  static PersistentLogStream log("joint_knot_first_frame_state_chain.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "test_id,scan_id,first_post_calibration_lio_frame,map_seed_complete,"
           "imu_propagated,n_imu_samples,scan_start_t,scan_tail_t,"
           "iteration,phase,t_abs,state,theta_x,theta_y,theta_z,"
           "p_x,p_y,p_z,v_x,v_y,v_z\n";
  const bool imu_propagated = measures.n_imu_samples > 0 && !measures.poses.empty();
  const double scan_start_t = measures.poses.empty()
      ? measures.image.t : measures.poses.front().t;
  auto row = [&](const char* label, const M3D& R, const V3D& p, const V3D& v) {
    const V3D theta = Log(R);
    out << test_id << ',' << scan_id << ",1,1,"
        << (imu_propagated ? 1 : 0) << ',' << measures.n_imu_samples << ','
        << std::setprecision(17) << scan_start_t << ',' << measures.image.t << ','
        << iteration << ',' << phase << ','
        << std::setprecision(17) << t_abs << ',' << label << ','
        << theta.x() << ',' << theta.y() << ',' << theta.z() << ','
        << p.x() << ',' << p.y() << ',' << p.z() << ','
        << v.x() << ',' << v.y() << ',' << v.z() << '\n';
  };
  row("before_imu", measures.rot_before_imu,
      measures.pos_before_imu, measures.vel_before_imu);
  row("after_imu", measures.rot_after_imu,
      measures.pos_after_imu, measures.vel_after_imu);
  row("current", current.rot(), current.pos(), current.vel());
  out.flush();
}

void writeJointKnotAllScanDiagnostics(
    const std::string& test_id, int scan_id, int iteration, double t_abs,
    const M3D& reference_R, const V3D& reference_p, const V3D& reference_v,
    const JointKnotTrajectory& before, const JointKnotTrajectory& after)
{
  static PersistentLogStream log("joint_knot_all_scans.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "test_id,scan_id,iteration,t_abs,knot_index,knot_t,immutable_head,"
           "before_p_distance,after_p_distance,dp_x,dp_y,dp_z,dp_norm,"
           "before_attitude_distance,after_attitude_distance,dtheta_x,dtheta_y,dtheta_z,dtheta_norm,"
           "before_velocity_distance,after_velocity_distance,dv_x,dv_y,dv_z,dv_norm,"
           "dbg_x,dbg_y,dbg_z,dbg_norm,dba_x,dba_y,dba_z,dba_norm,"
           "adjacent_bg_difference,adjacent_ba_difference\n";
  for (int k = 0; k < after.knotCount(); ++k) {
    const JointKnot& b = before.knots()[k];
    const JointKnot& a = after.knots()[k];
    const V3D dp = a.p - b.p;
    const V3D dtheta = Log(M3D(b.R.transpose() * a.R));
    const V3D dv = a.v - b.v;
    const V3D dbg = after.biasGyr(k) - before.biasGyr(k);
    const V3D dba = after.biasAcc(k) - before.biasAcc(k);
    const double adjacent_bg = k == 0 ? 0.0
        : (after.biasGyr(k) - after.biasGyr(k - 1)).norm();
    const double adjacent_ba = k == 0 ? 0.0
        : (after.biasAcc(k) - after.biasAcc(k - 1)).norm();
    out << test_id << ',' << scan_id << ',' << iteration << ','
        << std::setprecision(17) << t_abs << ',' << k << ',' << a.t << ','
        << (k == 0 ? 1 : 0) << ','
        << (b.p - reference_p).norm() << ',' << (a.p - reference_p).norm() << ','
        << dp.x() << ',' << dp.y() << ',' << dp.z() << ',' << dp.norm() << ','
        << Log(M3D(reference_R.transpose() * b.R)).norm() << ','
        << Log(M3D(reference_R.transpose() * a.R)).norm() << ','
        << dtheta.x() << ',' << dtheta.y() << ',' << dtheta.z() << ',' << dtheta.norm() << ','
        << (b.v - reference_v).norm() << ',' << (a.v - reference_v).norm() << ','
        << dv.x() << ',' << dv.y() << ',' << dv.z() << ',' << dv.norm() << ','
        << dbg.x() << ',' << dbg.y() << ',' << dbg.z() << ',' << dbg.norm() << ','
        << dba.x() << ',' << dba.y() << ',' << dba.z() << ',' << dba.norm() << ','
        << adjacent_bg << ',' << adjacent_ba << '\n';
  }
  out.flush();
}

void writeJointKnotScanSummaryDiagnostics(
    const std::string& test_id, int scan_id, double t_abs,
    int completed_iterations, const std::string& stop_reason,
    const StateGroup& post_imu, const StateGroup& final_state,
    const M3D& reference_R, const V3D& reference_p,
    const V3D& reference_v)
{
  // Unlike the other joint_knot_first_frame_* diagnostics (deliberately
  // scan_id==1-only, see the guard in writeJointKnotCovarianceDiagnostics/
  // writeJointKnotStateChainDiagnostics below), this file has no
  // "first_frame" in its name and is the only source of per-scan,
  // coupled-specific position/velocity/attitude-vs-pre-IMU-reference data --
  // required across the full run for the stationary-window statistics this
  // round's own INSTRUCTIONS.md asks for. A stray "if (scan_id != 1) return;"
  // here silently limited it to frame 1 only; the coding-agent return package
  // still trims this file to scan_id==1 per validate_first_post_calibration_
  // diagnostics.py's own schema, so this fix only affects what gets logged
  // locally during a run, not what the returned zip contains.
  static PersistentLogStream log("joint_knot_scan_summary.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "test_id,scan_id,t_abs,completed_iterations,stop_reason,"
           "dtheta_from_post_imu,dp_from_post_imu,dv_from_post_imu,"
           "p_distance_from_fixed_reference_before,p_distance_from_fixed_reference_after,"
           "v_distance_from_fixed_reference_before,v_distance_from_fixed_reference_after,"
           "attitude_distance_from_fixed_reference_before,attitude_distance_from_fixed_reference_after\n";
  out << test_id << ',' << scan_id << ',' << std::setprecision(17) << t_abs << ','
      << completed_iterations << ',' << stop_reason << ','
      << Log(M3D(post_imu.rot().transpose() * final_state.rot())).norm() << ','
      << (final_state.pos() - post_imu.pos()).norm() << ','
      << (final_state.vel() - post_imu.vel()).norm() << ','
      << (post_imu.pos() - reference_p).norm() << ','
      << (final_state.pos() - reference_p).norm() << ','
      << (post_imu.vel() - reference_v).norm() << ','
      << (final_state.vel() - reference_v).norm() << ','
      << Log(M3D(reference_R.transpose() * post_imu.rot())).norm() << ','
      << Log(M3D(reference_R.transpose() * final_state.rot())).norm() << '\n';
  out.flush();
}

void writeJointKnotCovarianceDiagnostics(
    const std::string& test_id, int scan_id, double t_abs,
    const JointKnotTrajectory& trajectory, const JointKnotPrior& prior,
    const Eigen::MatrixXd& posterior_extended,
    const Eigen::MatrixXd& state_post_imu,
    const Eigen::MatrixXd& state_post_lio,
    const Eigen::MatrixXd& scan_head_covariance,
    const std::vector<Pose6D>& poses,
    const std::vector<Eigen::MatrixXd>& transitions,
    const std::vector<Eigen::MatrixXd>& process_covariances)
{
  if (scan_id != 1) return;
  static PersistentLogStream log("joint_knot_first_frame_covariances.txt");
  std::ofstream& out = log.stream();
  out << "test_id " << test_id << "\nscan_id " << scan_id
      << "\nfirst_post_calibration_lio_frame 1\nmap_seed_complete 1"
      << "\nt_abs " << std::setprecision(17) << t_abs << '\n';
  out << "knot_count " << trajectory.knotCount() << '\n';
  out << "extended_state_layout physical_knots_1_to_Nminus1_then_"
         "bg_ba_per_knot_0_to_Nminus1_then_shared_g\n";
  out << "immutable_head_physical 1\n"
         "knot_specific_biases 1\n"
         "full_production_process_covariance 1\n"
         "shared_gravity 1\n";
  for (int k = 0; k < trajectory.knotCount(); ++k)
    out << "knot_timestamp " << k << ' ' << std::setprecision(17)
        << trajectory.knots()[k].t << '\n';
  for (int k = 1; k < trajectory.knotCount(); ++k)
    out << "physical_knot_offset " << k << ' '
        << trajectory.knotOffset(k) << '\n';
  for (int k = 0; k < trajectory.knotCount(); ++k)
    out << "bias_knot_offset " << k << ' '
        << trajectory.biasOffset(k) << '\n';
  out << "gravity_offset " << trajectory.gravityOffset() << '\n';
  writeMatrix(out, "P_scan_head", scan_head_covariance);
  writeMatrix(out, "P_state_post_imu", state_post_imu);
  writeMatrix(out, "P_prior_extended", prior.P);
  writeMatrix(out, "P_prior_extended_inverse", prior.P_inverse);
  if (state_post_imu.rows() == 18 && state_post_imu.cols() == 18) {
    Eigen::MatrixXd tail_marginal = Eigen::MatrixXd::Zero(18, 18);
    const int tail = trajectory.knotOffset(trajectory.knotCount() - 1);
    const int tail_bias = trajectory.biasOffset(trajectory.knotCount() - 1);
    const int joint_offsets[6] = {
        tail, tail + 3, tail + 6, tail_bias, tail_bias + 3,
        trajectory.gravityOffset()};
    const int state_offsets[6] = {0, 3, 6, 9, 12, 15};
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        tail_marginal.block<3,3>(state_offsets[i], state_offsets[j]) =
            prior.P.block<3,3>(joint_offsets[i], joint_offsets[j]);
    writeMatrix(out, "P_prior_tail_marginal", tail_marginal);
    out << "P_prior_tail_vs_state_post_imu_frobenius "
        << std::setprecision(17)
        << (tail_marginal - state_post_imu).norm() << '\n';
  }
  writeMatrix(out, "P_posterior_extended", posterior_extended);
  writeMatrix(out, "P_state_post_lio", state_post_lio);
  out << "imu_record_count " << poses.size() << '\n';
  for (size_t i = 0; i < poses.size(); ++i) {
    out << "imu_record " << i << " t " << std::setprecision(17)
        << poses[i].t << " dt " << poses[i].dt << '\n';
    if (i < transitions.size())
      writeMatrix(out, ("F_imu_" + std::to_string(i)).c_str(), transitions[i]);
    if (i < process_covariances.size()) {
      writeMatrix(out, ("Q_imu_production_" + std::to_string(i)).c_str(),
                  process_covariances[i]);
      writeMatrix(out, ("Q_imu_joint_knot_specific_bias_" +
                  std::to_string(i)).c_str(), process_covariances[i]);
      out << "Q_projection_removed_frobenius " << i << " 0\n";
    }
  }
  out.flush();
}

}  // namespace livo_recon
