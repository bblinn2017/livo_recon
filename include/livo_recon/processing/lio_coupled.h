#pragma once

#include "livo_recon/processing/lio_base.h"
#include "livo_recon/lio/coupled_estimator.h"
#include "livo_recon/lio/pose_spline_system.h"
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_lidar_factor.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/lio/adaptive_q.h"
#include "livo_recon/lio/residual_redundancy.h"
#include "livo_recon/lio/pose_control_lidar_correlation.h"
#include "livo_recon/diagnostics/eval/nees_logger.h"

#include <limits>
#include <unordered_set>
#include <deque>

namespace livo_recon
{

struct LioProcCoupledOptions
{
  // control_point_hz -- a separate resolution knob for a different spline).
  int n_c = 4;

  std::string spline_mode = "raw_imu";
  static constexpr const char* SPLINE_MODES[] = { "raw_imu", "pose", "pose_control" };
  bool poseBasis() const { return spline_mode == "pose"; }
  bool poseControlSplineBasis() const { return spline_mode == "pose_control"; }
  // Config key: estimator/coupled/pose_control/n_control_points.
  int pose_control_n = 13;
  // Relative eigenvalue floor for every generalPseudoInverse() call in the
  // pose-control estimator (joint head/eta/bias information, its
  // marginals, and every physical-covariance Mahalanobis normalization):
  // an eigenvalue at or below rel_thresh*lambda_max is treated as an exact
  // structural nullspace (contributes exactly zero to the resulting
  // covariance); above it, it is retained as a large-but-finite
  // contribution. 1e-12 sits two orders of magnitude above IEEE double
  // precision's own resolvable relative range for a symmetric
  // eigendecomposition of an O(10-100)-dimensional matrix
  // (~dim*machine_eps, machine_eps~2.2e-16), while staying far enough
  // below any real (non-structural) information eigenvalue this estimator
  // produces -- including as the joint information matrix's own condition
  // number grows with representation size N -- to keep such directions
  // classified as retained rather than discarded. See
  // test_pose_control_full_covariance_invariance.cpp's representation-
  // capacity-invariance test for the concrete numerical margin this
  // threshold must satisfy. Config key: estimator/coupled/pose_control/q_pinv_rel_thresh.
  double pose_control_q_pinv_rel_thresh = 1e-12;
  double pose_control_mean_pinv_rel_thresh = 1e-12;
  bool pose_control_lidar_enable = true;
  double pose_control_p0_scale = 1.0;
  double pose_control_curvature_weight_pos = 0.0;
  double pose_control_curvature_weight_rot = 0.0;
  std::string pose_control_test_id = "unlabeled";
  bool pose_control_freeze_geometry = false;
  bool pose_control_deskew_log_en = false;
  bool pose_control_covariance_cross_time_log_en = false;
  // Selects how LiDAR information enters pose-control mode. The default
  // local_spline path preserves the existing estimator. single_tail solves
  // one shared 6-DOF physical tail correction and projects it into the
  // spline. covariance_all_knots uses the scan-start joint covariance to
  // propagate that same physical correction into the full z=[eta;sT] state.
  std::string pose_control_lidar_update_mode = "local_spline";
  // Optional process uncertainty for the covariance-mediated latent
  // physical pose. Zero recovers the deterministic covariance coupling; a
  // positive value models additional pose uncertainty between the spline
  // trajectory and the LiDAR-time physical state.
  double pose_control_lidar_latent_pose_q_pos_m2 = 0.0;
  double pose_control_lidar_latent_pose_q_rot_rad2 = 0.0;
  AdaptiveQOptions pose_control_adaptive_q;
  // LiDAR information -- reuses ResidualRedundancyOptions (generic struct,
  // not decoupled-specific). Off by default (mode=="off").
  ResidualRedundancyOptions pose_control_lidar_correlation;

  double pose_imu_weight_acc = 1.0;
  double pose_imu_weight_gyr = 1.0;
  double pose_curvature_weight_pos = 0.0;
  double pose_curvature_weight_rot = 0.0;
  int pose_head_freeze_cp = 0;

  double pose_tikhonov_eps = 1e-6;

  double pose_imu_var_acc = 1e-4;
  double pose_imu_var_gyr = 1e-4;

  double pose_gn_max_step_pos_m = 0.5;
  double pose_gn_max_step_rot_rad = 0.2;

  double max_scan_displacement_m = 0.0;
  // (a constant delta_a and a constant -delta_ba are indistinguishable over
  // one scan, same for delta_omega/delta_bg) and solvable only via the
  // Lambda-vs-bias-prior ratio unless this is set. When true, constrains
  // sum_samples beta(t)^T c = 0 per axis (6 linear equality rows) via a
  // KKT-bordered solve, giving c the within-scan SHAPE only and the bias the
  // offset. Default false (the un-constrained, prior-determined arm).
  bool zero_mean = false;
  // Diagnostic/ablation toggle: drops c_gyr (the within-scan rotation
  // CORRECTION basis) out of the joint solve entirely, leaving delta_bg
  // (a single rigid per-scan gyro-bias correction) as the only rotation
  // correction mechanism. Default false.
  bool disable_cgyr = false;
  std::string jacobian_time_mode = "legacy_mismatched";
  static constexpr double JOINT_PIVOT_MIN_FLOOR = -1.0;
  bool log_jrow_leverage_en = false;
  bool freeze_bg = false;
  bool adaptive_sigma = false;
  bool bias_freeze_on_vibration = false;
  // Threshold for bias_freeze_on_vibration, disabled-by-default in effect
  // since the guard itself defaults off -- no validated threshold exists
  // yet (rule 26: a real numerics default is Bryce's call), mirrors
  // JOINT_PIVOT_MIN_FLOOR's own shipped-inert-until-validated pattern.
  static constexpr double BIAS_FREEZE_VIBRATION_FACTOR_DEFAULT = 3.0;
  double bias_freeze_vibration_factor = BIAS_FREEZE_VIBRATION_FACTOR_DEFAULT;
  bool bias_anchor = false;
  // Provisional precision for bias_anchor, disabled-by-default in effect
  // since the flag itself defaults off -- no validated in-run bias-drift
  // bound exists yet (rule 26). Set generously above the quoted MEMS
  // in-run stability (~10 deg/hr = 0.0028 deg/s) at 0.05 deg/s (=
  // against, not a validated physical spec.
  static constexpr double BIAS_ANCHOR_SIGMA_RAD_S_DEFAULT = 8.72665e-4;
  double smoothness_weight_acc = 0.0;
  double smoothness_weight_gyr = 0.0;
  double traj_deviation_weight = 0.0;
  bool bias_observable_only = false;
  double imu_deviation_weight = 1.0;
  double mean_weight = 0.0;
  bool log_traj_dev_en = false;
  bool prior_per_axis_sigma = false;
  std::string robust_loss = "none";
  bool psd_audit_en = false;
  bool log_bg_projection_en = false;
  bool log_cov_repropagation_en = false;
  // Same imu/* keys imu_processing.cpp itself reads (imu/q_alpha_gyr,
  // imu/q_alpha_acc, imu/second_order) -- read again here (not shared)
  // since this class has no reference to ImuProc's own options struct.
  // Only used when log_cov_repropagation_en is true.
  double repro_q_alpha_gyr = 1.0, repro_q_alpha_acc = 1.0;
  bool repro_second_order = true;
  bool q_bias_rw_en = false;
  double q_alpha_bias = 1.0;  // imu/q_alpha_bias, same key imu_processing.cpp reads.
  bool q_out_of_band_en = false;
  double q_out_of_band_scale = 1.0;
  double q_out_of_band_fraction_acc = 0.0, q_out_of_band_fraction_gyr = 0.0;
  bool log_cp_constraint_en = false;
  bool final_redeskew = false;
  bool final_relinearize_cov = false;
  bool prior_at_scan_start = false;
  bool add_q_scan_to_posterior = false;
  bool final_redeskew_map_uses_pre = false;
  bool log_point_plane_en = false;
  int  log_point_plane_hist_start_scan = 0;
  int  log_point_plane_hist_n_scans = 20;
};

class LioProcCoupled : public LioProcBase
{
public:
  explicit LioProcCoupled(NodeContext& ctx);
  ~LioProcCoupled() override;

  std::string loadParameters(ros::NodeHandle& pnh) override;
  std::string engagementReport() const override;
  void deskewAndDownsample(MeasureGroup& mg) override;
  std::string processLIO(MeasureGroup& mg) override;

  double estimateCoupledCorrection(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);

private:
  double estimateCoupledCorrectionPoseBasis(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);
  // One GN iteration of the pose-control-point-only estimator
  // (spline_mode=pose_control). Rebuilds LiDAR + IMU prior normal
  // equations fresh each call against z=[c_free;sT] (coupled_pose_control_layout_),
  // solves, applies the mean update in place to coupled_pose_control_spline_
  // (free control points) and coupled_pose_control_sT_{bg,ba,g}_, reconciles
  // the tail (R/p/v read directly off the optimized spline at t1, never an
  // independent variable), and writes state_->setPropagatedState(...) every
  // call.
  // Covariance is NOT written here (spec: mean every iteration, covariance
  // once after convergence) -- see processLIO()'s own
  // copts_.poseControlSplineBasis() early-return block for that.
  double estimateCoupledPoseControlSpline(MeasureGroup& mg, V3D& dtheta_out, V3D& dt_out);
  struct CoupledSystemBuild
  {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double sum_abs_r = 0.0, sum_sq_r = 0.0, sum_wr2 = 0.0, sum_sigma_squared = 0.0;
    double sum_floor_S = 0.0, sum_sdiag_S = 0.0, sum_pvar_S = 0.0, sum_prior_pose_S = 0.0;
    double sum_weight_this_iter = 0.0;
    std::vector<double> hcol_reldiff;
    Eigen::Matrix<double, 6, 6> HtH_pose_lidar = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> Htz_pose_lidar = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> H6_raw_accum = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::MatrixXd phic_spread_sum;
    double phic_spread_sumsq = 0.0;
    int phic_spread_n = 0;
    double sum_nis = 0.0;
    int n_nis = 0;
    double sum_nis_est = 0.0;
    int n_nis_est = 0;
  };

  CoupledSystemBuild buildImuCorrectionSystem(
      MeasureGroup& mg, double t0, double t1, int n_c, int ncol, int ncol_s, int ncol_c,
      double sigma_a, double sigma_g, double sigma_a_floor, double sigma_g_floor,
      const Eigen::MatrixXd& Pi_ss, const Eigen::VectorXd& s_vec);

  LioProcCoupledOptions copts_;

  // Persisted across this scan's own GN iterations (reset at the top of
  // processLIO() each frame); NOT carried scan-to-scan -- c_prior = 0 every
  std::vector<V3D> coupled_c_acc_, coupled_c_gyr_;
  ScanSpline coupled_pose_spline_;
  bool coupled_pose_spline_valid_ = false;
  std::vector<V3D> coupled_c_pos_, coupled_c_rot_;
  Eigen::Matrix<double, 6, 6> coupled_pose_head_cov_ = Eigen::Matrix<double, 6, 6>::Zero();

  PoseControlSpline coupled_pose_control_spline_;
  bool coupled_pose_control_valid_ = false;
  PoseControlFreeLayout coupled_pose_control_layout_;
  PoseControlHeadNullspace coupled_pose_control_hns_;
  Eigen::VectorXd coupled_pose_control_eta_;
  Eigen::VectorXd coupled_pose_control_eta_scan_start_;
  PoseControlSpline coupled_pose_control_spline_scan_start_;
  // coherent current non-trajectory tail state, updated by INCREMENT
  // (delta_bg/ba/g solved by the GN step) every iteration -- never treated
  // as itself the optimization variable (the variable is the increment).
  // tail_prior is the FIXED scan-entry value the increment is measured
  // against (mean prior residual = tail_trial - tail_prior, consistent
  V3D coupled_pose_control_bg_trial_ = V3D::Zero(), coupled_pose_control_ba_trial_ = V3D::Zero(),
      coupled_pose_control_g_trial_ = V3D::Zero();
  V3D coupled_pose_control_bg_prior_ = V3D::Zero(), coupled_pose_control_ba_prior_ = V3D::Zero(),
      coupled_pose_control_g_prior_ = V3D::Zero();
  // Per-sample e_acc/e_gyr residuals against the SCAN-START (pre-LiDAR)
  // spline, produced as a byproduct of buildPoseControlContinuousImuPrior()
  // -- the ONE evaluation of the continuous-time IMU prior's own residual,
  // redundant computePoseControlImuSplineResidualSamples() call. Distinct
  // from (and NOT a substitute for) adaptive-Q's own residual, which is
  // deliberately evaluated against the CONVERGED post-LiDAR spline (see
  // the adaptive-Q block's own comment) -- these are two genuinely
  // different linearization points, not duplicated work.
  std::vector<ImuSplineResidualSample> coupled_pose_control_imu_residual_samples_;
  // The reduced posterior z=[eta;delta_sT] covariance from the LAST
  // (converged) GN iteration's own information matrix -- written once,
  // post-loop, in processLIO()'s own poseControlSplineBasis() block (never
  // inside the per-iteration solve -- spec: mean every iteration,
  // covariance once after convergence).
  Eigen::MatrixXd coupled_pose_control_P_z_post_;
  // by estimateCoupledPoseControlSpline() (mean solve) for the post-loop
  // covariance block to log alongside Lambda_lidar/Lambda_prior traces.
  double coupled_pose_control_last_lambda_curvature_trace_ = 0.0;
  Eigen::MatrixXd coupled_pose_control_last_A_lidar_reduced_;
  Eigen::VectorXd coupled_pose_control_eta_imu_;
  Eigen::Matrix<double, 6, 1> coupled_pose_control_last_physical_lidar_delta_ = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::MatrixXd coupled_pose_control_sigma_full_prior_;
  // Lambda_prior_z = pinv(Sigma_full_prior_'s z=[eta;sT] marginal block) --
  // the information the mean solve adds every iteration. z_imu_ is the
  // z-space value ([eta_imu;0,0,0], sT's prior deviation is zero at scan
  // start by construction) the residual is measured against.
  Eigen::MatrixXd coupled_pose_control_lambda_prior_z_;
  Eigen::VectorXd coupled_pose_control_z_imu_;
  // mid-iteration (using THAT iteration's own LiDAR linearization + the
  // fixed prior), held here until the actual delta_z for the SAME
  // iteration is available a few lines later in the same function.
  Eigen::VectorXd coupled_pose_control_ekf_ref_pending_;
  bool coupled_pose_control_ekf_ref_pending_valid_ = false;
  // (reuses the existing AdaptiveQ class -- its math is fully generic
  // despite SplineImuResidualStats' name; populated here from process-
  // segment residuals, not a ScanSpline). primed_ tracks whether
  // setNominal/setFloor has run yet (once, first valid scan).
  AdaptiveQ coupled_pose_control_adaptive_q_;
  bool coupled_pose_control_adaptive_q_primed_ = false;
  double coupled_pose_control_effective_var_acc_used_ = 0.0;
  double coupled_pose_control_effective_var_gyr_used_ = 0.0;
  // construction/diagnostics go through -- returns the adaptive value
  // (already one-scan-causal by construction) when enabled and primed,
  // else state_->varAcc()/varGyr() exactly as before (identity when off).
  V3D poseControlEffectiveVarAcc() const;
  V3D poseControlEffectiveVarGyr() const;
  // mean, used only to compute a multi-scan lag-1 autocorrelation
  // (item 17) -- a single scan's handful of segments is too few samples on
  // its own. Bounded ring buffer (see .cpp for the cap).
  std::deque<V3D> coupled_pose_control_acc_resid_hist_;
  std::deque<V3D> coupled_pose_control_gyr_resid_hist_;
  std::unordered_set<std::size_t> coupled_prev_iter_planes_;
  std::vector<PoseControlLidarObs> coupled_pose_control_frozen_lidar_obs_;
  V3D coupled_delta_v_ = V3D::Zero(), coupled_delta_bg_ = V3D::Zero(),
      coupled_delta_ba_ = V3D::Zero(), coupled_delta_g_ = V3D::Zero();
  // delta_p(t0) are solved for too, not held at zero. This is a one-scan
  // fixed-lag smoother: the previous scan's published pose is genuinely
  // revised by the converged value of these two (the map already built from
  // the old pose is NOT retroactively updated -- a named consequence).
  V3D coupled_delta_phi0_ = V3D::Zero(), coupled_delta_pos0_ = V3D::Zero();
  V3D coupled_v0_pre_, coupled_bg0_pre_, coupled_ba0_pre_, coupled_g0_pre_;
  CoupledPropagation coupled_prop_;
  double coupled_solve_ms_ = -1.0;
  int    coupled_iters_ = 0;

  // by the final GN iteration.
  double coupled_c_acc_dc_over_sigma_ = -1.0, coupled_c_gyr_dc_over_sigma_ = -1.0;
  double coupled_dba_over_sigma_ = -1.0, coupled_dbg_over_sigma_ = -1.0;
  double coupled_c_acc_over_sigma_ = -1.0, coupled_c_gyr_over_sigma_ = -1.0;
  double coupled_c_acc_total_norm_ = -1.0, coupled_c_gyr_total_norm_ = -1.0;
  double coupled_sum_S_ = -1.0;
  double coupled_bg_var_degenerate_ = -1.0, coupled_bg_var_observed_ = -1.0;
  Tier1NeesBuffer coupled_tier1_nees_{0};
  double coupled_delta_v_norm_ = -1.0, coupled_delta_g_norm_ = -1.0;
  double coupled_trP_vel_ = -1.0, coupled_trP_grav_ = -1.0;
  // The LAST GN iteration's own joint A matrix, kept so the item-5
  // covariance term can be applied ONCE after the loop converges.
  Eigen::MatrixXd coupled_last_A_;
  Eigen::MatrixXd coupled_last_Lambda_;

  // ask/got/refusal discriminator, restricted to the [delta_phi0, delta_p0]
  // 6-dim sub-block of the FINAL GN iteration's pure-LiDAR-info
  // accumulation -- a NAMED APPROXIMATION (not marginalised over
  // v/bg/ba/g/c, so it overstates the information actually available to
  // phi0/p0 alone by whatever those directions correlate away).
  double coupled_ask_ = -1.0, coupled_got_ = -1.0, coupled_refusal_ = std::numeric_limits<double>::quiet_NaN();
  int    coupled_n_residuals_ = -1;
  int    coupled_prev_n_residuals_ = -1;
  double coupled_sum_weight_ = -1.0;
  double coupled_h_pp_min_eig_ = -1.0, coupled_h_rr_min_eig_ = -1.0;
  double coupled_h_pp_max_eig_ = -1.0;
  double coupled_h_rr_trace_ = -1.0, coupled_htth_pos_trace_ = -1.0;
  double coupled_htz_rot_norm_ = -1.0, coupled_htz_pos_norm_ = -1.0;
  double coupled_kappa_eff_ = -1.0;
  double coupled_kappa_gev_[6] = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
  bool   coupled_kappa_gev_ok_ = false;
  double coupled_floor_share_ = -1.0, coupled_sdiag_share_ = -1.0;
  double coupled_pvar_share_ = -1.0, coupled_prior_pose_share_ = -1.0;
  double coupled_nis_ = -1.0, coupled_nis_est_ = -1.0;
  double coupled_dx_rot_deg_ = 0.0, coupled_dx_pos_mm_ = 0.0;

  double coupled_joint_dmin_ = -1.0, coupled_joint_dmax_ = -1.0;
  double coupled_state_dmin_ = -1.0, coupled_state_dmax_ = -1.0;
  double coupled_coeff_dmin_ = -1.0, coupled_coeff_dmax_ = -1.0;
  // True the iteration coupled_joint_dmin_ < JOINT_PIVOT_MIN_FLOOR -- at the
  // shipped -1.0 floor this can never fire; exists so the refusal SHAPE is
  // present and testable ahead of an actual validated threshold.
  bool   coupled_pivot_guard_ = false;

  double coupled_hcol_reldiff_p10_ = -1.0, coupled_hcol_reldiff_p50_ = -1.0;
  double coupled_hcol_reldiff_p90_ = -1.0, coupled_hcol_reldiff_max_ = -1.0;

  double coupled_last_delta_s_norm_ = -1.0, coupled_last_delta_c_norm_ = -1.0;
  double coupled_last_delta_phi0_norm_ = -1.0, coupled_last_delta_p0_norm_ = -1.0,
         coupled_last_delta_v_norm_step_ = -1.0, coupled_last_delta_bg_norm_step_ = -1.0,
         coupled_last_delta_ba_norm_step_ = -1.0, coupled_last_delta_g_norm_step_ = -1.0;
  double coupled_last_delta_c_acc_norm_ = -1.0, coupled_last_delta_c_gyr_norm_ = -1.0;
  Eigen::VectorXd coupled_last_delta_c_;

  V3D coupled_prev_traj_dev_end_vec_ = V3D::Zero();
  bool coupled_prev_traj_dev_valid_ = false;

  double coupled_res_rms_ = -1.0;
  double coupled_mean_sigma_squared_ = -1.0;

  double coupled_acc_world_mag_ = -1.0, coupled_gravity_dir_err_deg_ = -1.0;

  double boundary_dpos_ = 0.0;
  double boundary_drot_deg_ = 0.0;

  double coupled_sigma_a_used_ = -1.0, coupled_sigma_g_used_ = -1.0;
  double coupled_sigma_a_ratio_ = -1.0, coupled_sigma_g_ratio_ = -1.0;
  bool coupled_bias_freeze_active_ = false;
  long coupled_bias_freeze_active_count_ = 0, coupled_bias_freeze_scan_count_ = 0;
  double coupled_w_from_c_deg_s_ = -1.0, coupled_w_from_bg_deg_s_ = -1.0, coupled_w_net_deg_s_ = -1.0;

  V3D coupled_bg_calib_ = V3D::Zero();
  bool coupled_bg_calib_set_ = false;

  double coupled_reduced_chi2_ = -1.0;
};

}  // namespace livo_recon
