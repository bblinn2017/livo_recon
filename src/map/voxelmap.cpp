#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/voxelplane.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace livo_recon
{

namespace
{
// History (17-24): see docs/livo_recon_changelog.md#src-map-voxelmap.cpp-17
void debugLogFrameStats(double t_abs, int frame_idx, int denom_rejected_count,
                        double max_plane_var_trace, const LioFrameDiag& lio, int n_planes,
                        int n_voxels, int n_voxels_is_plane, int n_voxels_converged,
                        double max_plane_covariance_trace)
{
  // CQ-36: PersistentLogStream re-resolves debugLogPath() on every call
  // (reopening only if it actually changed) and the caller flushes after
  // writing -- see its own doc comment for the CQ-35 regression this
  // fixes (a bare static std::ofstream froze onto the first-ever resolved
  // path and never flushed, so this file landed at /tmp, empty, for an
  // entire run).
  static PersistentLogStream log("frame_stats.txt");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call)
    ofs << "t,frame_idx,denom_rejected_count,max_plane_var_trace"
           ",n_residuals,n_planes,h_pp_min_eig,h_rr_min_eig,sum_weight"
           ",h_rr_trace,htz_rot_norm,htz_pos_norm,ask,got,refusal"
           ",iters,dx_rot_deg,dx_pos_mm,trP_pos_pre"
           ",boundary_dpos,boundary_drot_deg"
           // P-F (BASE, 2026-09-04): L1 (DECIMATION) -- see LioFrameDiag's
           // doc comment. n_points_raw is NOT here yet (needs a queue-level
           // accumulator, not captured this pass).
           ",n_points_after_pfn,n_points_after_ds"
           // P-C (BASE, 2026-09-03): a direct per-frame CENSUS of
           // VoxelStats' live atomic counters -- n_planes above is a
           // TRANSITION counter (increments/decrements on state change,
           // known negative on the debiased path, see the bug ledger)
           // while these are the counters' own current values, always
           // non-negative by construction. n_voxels = stats_->total()
           // (open+converged+disabled); n_voxels_is_plane = stats_->planes
           // (non-disabled, non-parent nodes currently fitting a plane);
           // n_voxels_converged = stats_->converged.
           ",n_voxels,n_voxels_is_plane,n_voxels_converged"
           // CQ-18 item (2): the S decomposition (sum_S + four shares) and
           // the two per-frame NIS calibration tests, see LioFrameDiag's
           // own doc comment.
           ",sum_S,floor_share,sdiag_share,pvar_share,prior_pose_share,nis,nis_est"
           // CQ-19(a): P (prior_cov_/state_->cov()) decomposed beyond the
           // bare trace already in trP_pos_pre above.
           ",p_pos_eig_min_pre,p_pos_eig_mid_pre,p_pos_eig_max_pre"
           ",p_pos_eig_min_post,p_pos_eig_mid_post,p_pos_eig_max_post"
           ",p_rot_trace_pre,p_rot_eig_min_pre"
           ",p_pos_vel_fro_pre,p_pos_bias_fro_pre"
           // CQ-19(b): the dot product that signs rho_ref.
           ",cos_pmin_hmin"
           // CQ-19(c): AdaptiveQ's own gate state.
           ",q_z_acc,q_z_gyr,q_acf1_acc,q_acf1_gyr,q_active,q_clamped"
           // CQ-26: update()'s per-frame decision (status + the four gate
           // reads it already computes and previously discarded) plus
           // q_active_frame, the per-frame counterpart to the latched
           // q_active above -- see LioFrameDiag's doc comment.
           ",q_status,q_white_acc,q_white_gyr,q_above_floor_acc,q_above_floor_gyr,q_active_frame"
           // CQ-28: residual-redundancy-correction engagement/magnitude --
           // see LioFrameDiag's own doc comment.
           ",redund_groups,redund_n_raw,redund_n_eff,redund_info_ratio"
           // CQ-34 item 4: a zero redund_groups conflates three distinct
           // facts -- see LioFrameDiag's own doc comment.
           ",redund_groups_seen,redund_groups_degenerate_pv,redund_groups_degenerate_var"
           // TQ-20 item 1/6: kappa = P^-1/HtH -- one derived scalar
           // (kappa_eff, from the already-logged ask/got) plus the 6
           // generalized eigenvalues of the (HtH,P) pencil itself.
           ",kappa_eff,kappa_gev0,kappa_gev1,kappa_gev2,kappa_gev3,kappa_gev4,kappa_gev5,kappa_gev_ok"
           // CQ-31 item 7/8: naive vs Woodbury-corrected info gain (always
           // populated, including mode=="off") and reduced_chi2 (never
           // retained before this -- TQ-23 flagged its absence).
           ",naive_info_gain,woodbury_info_gain,reduced_chi2"
           // CQ-36 item 1: the HtH-level control column -- see LioFrameDiag's
           // own comment. CQ-37: axis A/B/D engagement, always populated
           // (0/1.0 at every axis's off default, matching the redund_*
           // columns' own "populated, mode-gated only on the ekf mutation"
           // precedent).
           ",htth_pos_trace"
           ",collapse_groups_collapsed,collapse_residuals_removed"
           ",per_residual_touched,per_residual_renorm_factor,per_residual_mean_scale"
           ",sigma_scale_applied,sigma_scale_chi2_ema"
           // CQ-56 item 3: this frame's largest VoxelPlane::covariance_
           // trace -- see updateMaxPlaneCovarianceTrace()'s own doc
           // comment. Appended, not interleaved, per this file's own
           // convention.
           ",max_plane_covariance_trace\n";
  first_call = false;
  // t_abs is an epoch-scale double (~1.6e9) -- default ostream formatting
  // (6 significant figures) collapses every frame in a run to the same
  // printed value without an explicit precision. Matches the setprecision(9)
  // convention already used by pose_pair.csv/spline_q.csv's own t columns.
  ofs << std::fixed << std::setprecision(9) << t_abs << std::defaultfloat
      << "," << frame_idx << "," << denom_rejected_count << "," << max_plane_var_trace
      << "," << lio.n_residuals << "," << n_planes
      << "," << lio.h_pp_min_eig << "," << lio.h_rr_min_eig << "," << lio.sum_weight
      << "," << lio.h_rr_trace << "," << lio.htz_rot_norm << "," << lio.htz_pos_norm
      << "," << lio.ask << "," << lio.got << "," << lio.refusal
      << "," << lio.iters << "," << lio.dx_rot_deg << "," << lio.dx_pos_mm
      << "," << lio.trP_pos_pre
      << "," << lio.boundary_dpos << "," << lio.boundary_drot_deg
      << "," << lio.n_points_after_pfn << "," << lio.n_points_after_ds
      << "," << n_voxels << "," << n_voxels_is_plane << "," << n_voxels_converged
      << "," << lio.sum_S << "," << lio.floor_share << "," << lio.sdiag_share
      << "," << lio.pvar_share << "," << lio.prior_pose_share
      << "," << lio.nis << "," << lio.nis_est
      << "," << lio.p_pos_eig_min_pre << "," << lio.p_pos_eig_mid_pre << "," << lio.p_pos_eig_max_pre
      << "," << lio.p_pos_eig_min_post << "," << lio.p_pos_eig_mid_post << "," << lio.p_pos_eig_max_post
      << "," << lio.p_rot_trace_pre << "," << lio.p_rot_eig_min_pre
      << "," << lio.p_pos_vel_fro_pre << "," << lio.p_pos_bias_fro_pre
      << "," << lio.cos_pmin_hmin
      << "," << lio.q_z_acc << "," << lio.q_z_gyr << "," << lio.q_acf1_acc << "," << lio.q_acf1_gyr
      << "," << (lio.q_active ? 1 : 0) << "," << (lio.q_clamped ? 1 : 0)
      << "," << lio.q_status
      << "," << (lio.q_white_acc ? 1 : 0) << "," << (lio.q_white_gyr ? 1 : 0)
      << "," << (lio.q_above_floor_acc ? 1 : 0) << "," << (lio.q_above_floor_gyr ? 1 : 0)
      << "," << (lio.q_active_frame ? 1 : 0)
      << "," << lio.redund_groups << "," << lio.redund_n_raw << "," << lio.redund_n_eff
      << "," << lio.redund_info_ratio
      << "," << lio.redund_groups_seen << "," << lio.redund_groups_degenerate_pv
      << "," << lio.redund_groups_degenerate_var
      << "," << lio.kappa_eff
      << "," << lio.kappa_gev0 << "," << lio.kappa_gev1 << "," << lio.kappa_gev2
      << "," << lio.kappa_gev3 << "," << lio.kappa_gev4 << "," << lio.kappa_gev5
      << "," << (lio.kappa_gev_ok ? 1 : 0)
      << "," << lio.naive_info_gain << "," << lio.woodbury_info_gain << "," << lio.reduced_chi2
      << "," << lio.htth_pos_trace
      << "," << lio.collapse_groups_collapsed << "," << lio.collapse_residuals_removed
      << "," << lio.per_residual_touched << "," << lio.per_residual_renorm_factor
      << "," << lio.per_residual_mean_scale
      << "," << lio.sigma_scale_applied << "," << lio.sigma_scale_chi2_ema
      << "," << max_plane_covariance_trace
      << "\n";
  ofs.flush();
}
}  // namespace

VoxelMap::VoxelMap(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues)
  : opts_(std::make_shared<VoxelOpts>())
  , stats_(std::make_shared<VoxelStats>())
  , state_(state)
  , profiler_(profiler)
  , data_queues_(data_queues)
{}

std::string VoxelMap::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<double>(pnh, "voxel_map/map/voxel_size",   opts_->voxel_size,  0.5);
  paramWarn<int>(pnh, "voxel_map/map/max_layer",       opts_->max_layer,    2);
  paramWarn<bool>(pnh, "voxel_map/plane/sensor_noise_floor_eig0", opts_->sensor_noise_floor_eig0, false);
  paramWarn<bool>(pnh, "voxel_map/plane/centred_accumulation", opts_->centred_accumulation, false);
  paramWarn<std::string>(pnh, "voxel_map/plane/convergence_mode", opts_->convergence_mode, "normal");
  paramWarn<int>(pnh, "voxel_map/plane/min_frames_to_converge", opts_->min_frames_to_converge, 5);
  paramWarn<int>(pnh, "voxel_map/plane/min_frames_to_init", opts_->min_frames_to_init, 1);

  paramWarn<double>(pnh, "voxel_map/plane/plane_threshold",           opts_->plane_threshold,           0.01);

  paramWarn<int>(pnh, "voxel_map/points/min_init",   opts_->min_init_points,  3);
  paramWarn<int>(pnh, "voxel_map/points/max",        opts_->max_points,       50);
  paramWarn<int>(pnh, "voxel_map/points/min_update", opts_->min_update_points,5);

  { double sigma_num; paramWarn<double>(pnh, "voxel_map/residual/sigma_num", sigma_num, 3.); opts_->sigma_num_squared = sigma_num * sigma_num; }
  paramWarn<double>(pnh, "voxel_map/residual/max_radius",  opts_->max_radius,          3.);
  paramWarn<bool>(pnh, "voxel_map/residual/pose_cov_in_sigma", opts_->pose_cov_in_sigma, false);
  paramWarn<bool>(pnh, "voxel_map/plane/log_debug_en", opts_->log_debug_en, false);
  paramWarn<bool>(pnh, "voxel_map/plane/log_variance_shares_en", opts_->log_variance_shares_en, false);
  paramWarn<int>(pnh, "voxel_map/search/neighborhood_size", opts_->neighborhood_size,   1);
  {
    // Modes go through ConfigResolver so an unrecognised value REFUSES rather
    // than falling through to whichever arm the if-chain happens to reach
    // last, and so the sub-parameters of an unselected mode cannot be set
    // silently.  See config_resolve.h.
    ConfigResolver cfg(pnh);
    cfg.mode("voxel_map/plane/plane_gate_mode", opts_->plane_gate_mode, "disc",
             { "disc", "ellipse", "ellipse_area_matched" });
    // Resolved once here, not per-candidate-point -- see voxelplane.cpp's
    // gate(), which used to re-compare opts_->plane_gate_mode (a
    // std::string) against these same literals on every single candidate,
    // every IEKF iteration. Mirrors gate()'s own branch structure exactly:
    // "ellipse_area_matched" first (most specific), then "ellipse", else
    // "disc" (the default, and the fallback for any unrecognized value --
    // though cfg.mode() above already refuses those before this runs).
    opts_->plane_gate_mode_enum =
        (opts_->plane_gate_mode == "ellipse_area_matched") ? PlaneGateMode::EllipseAreaMatched :
        (opts_->plane_gate_mode == "ellipse")               ? PlaneGateMode::Ellipse :
                                                                PlaneGateMode::Disc;

    // ── residual-weight floor ────────────────────────────────────────────
    // sigma_r2 is MIRRORED from imu/sensor/range_err rather than given its
    // own key: the point of "sensor_range" is that the floor IS the sensor's
    // range variance, and a second copy of that number would drift.
    double range_err = 0.05;
    paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
    opts_->weight_sigma_r2 = range_err * range_err;
    cfg.derived("voxel_map/plane/weight_floor/sigma_r2 (= imu/sensor/range_err^2)",
                std::to_string(opts_->weight_sigma_r2));
    cfg.mode("voxel_map/plane/weight_floor/mode", opts_->weight_floor_mode,
             "sensor_range",
             { "sensor_range", "incidence", "constant", "none", "legacy",
               "roughness" });
    // Resolved once here, not per-candidate-point -- see voxelplane.cpp's
    // weightFloor(), which used to re-compare opts_->weight_floor_mode (a
    // std::string) against up to 5 literals sequentially, called TWICE per
    // candidate (once from gate(), once from computeResidual()), on every
    // point, every IEKF iteration. Mirrors weightFloor()'s own if-chain
    // order exactly (legacy/none/constant/roughness explicit, "incidence"
    // next, else sensor_range -- the default and the fallback for any
    // unrecognized value, though cfg.mode() above already refuses those).
    opts_->weight_floor_mode_enum =
        (opts_->weight_floor_mode == "legacy")    ? WeightFloorMode::Legacy :
        (opts_->weight_floor_mode == "none")      ? WeightFloorMode::None :
        (opts_->weight_floor_mode == "constant")  ? WeightFloorMode::Constant :
        (opts_->weight_floor_mode == "roughness") ? WeightFloorMode::Roughness :
        (opts_->weight_floor_mode == "incidence") ? WeightFloorMode::Incidence :
                                                     WeightFloorMode::SensorRange;
    // The roughness/information combination check lives below, after
    // plane_var_mode is actually loaded (see there) -- opts_->plane_var_mode
    // still holds its struct default ("eigengap") at this point in
    // loadParameters(), so checking the pairing here would refuse the one
    // combination it is meant to allow. Found 2026-09-02 dispatching LD-1's
    // R2 rung: the effective-config summary printed plane_var_mode=
    // information correctly (set moments later, same function) while this
    // check had already fired against the stale default.
    cfg.nested<double>(opts_->weight_floor_mode == "constant" ||
                           opts_->weight_floor_mode == "legacy",
                       "voxel_map/plane/weight_floor/mode=constant",
                       "voxel_map/plane/weight_floor/constant",
                       opts_->weight_floor_constant, 1e-3);
    cfg.nested<double>(opts_->weight_floor_mode == "incidence",
                       "voxel_map/plane/weight_floor/mode=incidence",
                       "voxel_map/plane/weight_floor/incidence_k",
                       opts_->weight_incidence_k, 1.0);

    cfg.mode("voxel_map/plane/plane_fit_mode", opts_->plane_fit_mode, "pca",
             { "pca", "debiased" });
    // ── the plane covariance model ───────────────────────────────────────
    // See VoxelOpts::plane_var_mode.  "information" replaces the eigenvector
    // -perturbation form with the directional Fisher information in the
    // chart J_nq already uses, which is what makes roughness, sampling
    // anisotropy and directional coverage three readings of one matrix
    // instead of three bolted-on corrections.  Transitional: both modes
    // exist so the incumbent stays a representable control.
    cfg.mode("voxel_map/plane/plane_var_mode", opts_->plane_var_mode,
             "eigengap", { "eigengap", "information", "information_directional" });
    // "roughness" returns VoxelPlane::roughness_, which only the information
    // model computes.  Pairing it with the eigengap model would floor every
    // residual at zero and read as a silent "none" -- refuse instead.  This
    // is the same class of defect as use_bins-on-debiased. Checked here,
    // not where weight_floor/mode is loaded above, because plane_var_mode
    // is not yet read at that point in this function.
    if (opts_->weight_floor_mode == "roughness" &&
        opts_->plane_var_mode != "information" &&
        opts_->plane_var_mode != "information_directional")
    {
      cfg.requireCombination(
          "voxel_map/plane/weight_floor/mode = roughness requires "
          "voxel_map/plane/plane_var_mode = information -- only that model "
          "computes a per-plane roughness, and pairing them the other way "
          "would floor every residual at 0.0 and read as a silent 'none'");
    }
    // The reverse direction (2026-09-03, register-flagged gap): S always
    // adds weightFloor()'s floor_term unconditionally (see the call site in
    // voxelplane.cpp), regardless of plane_var_mode -- weightFloor() itself
    // has no idea which variance model is active. Under the information
    // model, sigma_bar2_/plane_var_ already fully accounts for the plane's
    // variance (roughness included), so sensor_range/incidence/constant
    // would silently ADD a second, redundant floor on top of it --
    // double-counting exactly the way roughness-under-eigengap would read
    // as a silent zero. weight_floor/mode must be roughness or none under
    // information*; only "legacy" is exempt (an explicit reproducibility
    // control, not a live model choice).
    if ((opts_->plane_var_mode == "information" ||
         opts_->plane_var_mode == "information_directional") &&
        opts_->weight_floor_mode != "roughness" &&
        opts_->weight_floor_mode != "none" &&
        opts_->weight_floor_mode != "legacy")
    {
      cfg.requireCombination(
          "voxel_map/plane/plane_var_mode = information* requires "
          "voxel_map/plane/weight_floor/mode = roughness or none -- "
          "sensor_range/incidence/constant would add a second, redundant "
          "floor on top of a variance the information model already "
          "accounts for in full, double-counting exactly the way "
          "roughness-under-eigengap would silently zero every residual");
    }
    // P7 (2026-09-03).  debiased+information_directional cannot engage:
    // refitDebiased() runs from persistent unweighted accumulators, after
    // the points are gone, so the per-point weighted-H path (built in
    // update(), which only the pca fit path calls with a point vector in
    // hand) is structurally unreachable on the debiased arm. DX-1 found
    // this the hard way -- L5d equalled L4d exactly, info_path was 100%
    // fallback -- and rule 4b forbids a flag set true that increments
    // nothing. Refuse rather than run it as a silent alias for
    // debiased+information. The alternative (teaching refitDebiased() to
    // retain enough state for the exact path) was considered and rejected:
    // the per-point weights depend on the fitted normal, which isn't known
    // until the fit runs, so they cannot be accumulated before it.
    if (opts_->plane_fit_mode == "debiased" &&
        opts_->plane_var_mode == "information_directional")
    {
      cfg.requireCombination(
          "voxel_map/plane/plane_fit_mode = debiased with "
          "voxel_map/plane/plane_var_mode = information_directional is "
          "UNREACHABLE: the directional weighting needs update()'s point "
          "vector and refitDebiased() never has it. Use plane_var_mode = "
          "information (identical behaviour, honestly named) or "
          "plane_fit_mode = pca with information_directional.");
    }
    // The T8-b plane-confidence inflations and the eigengap denominator
    // floor are all corrections TO the eigengap model.  Under the
    // information model they are not merely unnecessary, they DOUBLE COUNT:
    // redundancy is already in the design-effect N_eff, coverage is already
    // the leverage a^T I^-1 a, and there is no denominator to floor.
    // Leaving them settable would let a cell inflate a variance twice and
    // report it as the new model's behaviour, so they are nested off.
    const bool eigengap = (opts_->plane_var_mode == "eigengap");
    const char* kEigScope = "voxel_map/plane/plane_var_mode=eigengap";
    cfg.nested<bool>(eigengap, kEigScope, "voxel_map/plane/plane_var_denom_floor_en",
                     opts_->plane_var_denom_floor_en, false);
    cfg.mode("voxel_map/plane/plane_fit_pose_cov_mode",
             opts_->plane_fit_pose_cov_mode, "combined",
             { "combined", "sensor_only" });
    // ── binning (mode 2 of the three plane-confidence corrections) ───────
    // Binning exists ONLY on the pca path.  VoxelNode::insertPoints()'s
    // debiased branch calls VoxelPlane::addPoints() with the raw
    // points_world and never touches bins_/buildBinReps(), so use_bins is
    // read, validated, printed and then does NOTHING in debiased mode.
    //
    // That silence was not harmless.  N_acc_ counts every redundant return
    // and plane_var_ scales as 1/N, so unbinned debiased is over-confident
    // by the redundancy factor precisely where binning exists to prevent
    // it -- and because the flag was inert rather than refused, EVERY
    // pca-vs-debiased comparison this project has run is fit_mode (x)
    // binning, with no way to tell after the fact which cells meant to bin.
    // Refusing at startup makes the confound visible on the config that
    // causes it instead of in a re-analysis two rounds later.
    //
    // Not "fixed" by teaching the debiased path to bin: binning is itself a
    // decimation heuristic (a leaf-size proxy for redundancy) carrying three
    // constants and two mode enums, and the replacement is a directional
    // information model that measures redundancy from the geometry that
    // caused it.  See claude/unified-plane-confidence.md.  Until that lands,
    // an inert flag is a lie and a refusal is the truth.
    const bool binning_available = (opts_->plane_fit_mode == "pca");
    cfg.nested<bool>(binning_available,
                     "voxel_map/plane/plane_fit_mode=pca  (binning is "
                     "unimplemented on the debiased path)",
                     "voxel_map/plane/use_bins", opts_->use_bins, false);
    cfg.nested<double>(opts_->use_bins, "voxel_map/plane/use_bins=true",
                       "voxel_map/plane/bin_size_fraction",
                       opts_->bin_size_fraction, 0.2);
    cfg.nestedMode(opts_->use_bins, "voxel_map/plane/use_bins=true",
                   "voxel_map/plane/bin_weight_mode_fit",
                   opts_->bin_weight_mode_fit, "count", { "count", "uniform" });
    cfg.nestedMode(opts_->use_bins, "voxel_map/plane/use_bins=true",
                   "voxel_map/plane/bin_weight_mode_var",
                   opts_->bin_weight_mode_var, "count", { "count", "uniform" });

    // ── consistency logging tier ─────────────────────────────────────────
    // See VoxelOpts::log_consistency_mode. The two booleans this replaces
    // made the cheap per-scan aggregate reachable only by also paying for
    // the ~1 GB/job per-correspondence rows.
    cfg.mode("voxel_map/plane/log_consistency_mode", opts_->log_consistency_mode,
             "off", { "off", "scan", "corr", "corr+covariates" });
    cfg.nested<int>(opts_->logCorrRows(),
                    "voxel_map/plane/log_consistency_mode=corr|corr+covariates",
                    "voxel_map/plane/log_consistency_corr_stride",
                    opts_->log_consistency_corr_stride, 1);
    cfg.refuseIfSet("voxel_map/plane/log_consistency_corr_en",
                    "Use voxel_map/plane/log_consistency_mode instead: "
                    "false -> 'off' (or 'scan' if you want the cheap per-scan "
                    "aggregates, which this flag used to gate too), "
                    "true -> 'corr'.");
    cfg.refuseIfSet("voxel_map/plane/log_consistency_covariates_en",
                    "Use voxel_map/plane/log_consistency_mode instead: "
                    "true -> 'corr+covariates'.");

    if (!cfg.ok())
    {
      ROS_FATAL_STREAM("\n" << cfg.report());
      throw std::runtime_error(
          "[config] refused: " + std::to_string(cfg.errors().size()) +
          " voxel_map option(s) outside their allowed set or set into a dead "
          "scope -- see the [config/REFUSED] block above");
    }
    ROS_INFO_STREAM("\n" << cfg.report());
  }
  paramWarn<int>(pnh, "voxel_map/map/shuffle_insertion_seed", opts_->shuffle_insertion_seed, 0);
  paramWarn<bool>(pnh, "voxel_map/map/log_frame_stats_en", opts_->log_frame_stats_en, false);

  std::ostringstream oss;
  oss << "[params/voxel_map]"
      << "\n  map/voxel_size:                  " << opts_->voxel_size
      << "\n  map/max_layer:                   " << opts_->max_layer
      << "\n  plane/sensor_noise_floor_eig0:   " << (opts_->sensor_noise_floor_eig0 ? "true" : "false")
      << "\n  plane/centred_accumulation:      " << (opts_->centred_accumulation ? "true" : "false")
      << "\n  plane/convergence_mode:          " << opts_->convergence_mode
      << "\n  plane/min_frames_to_converge:    " << opts_->min_frames_to_converge
      << "\n  plane/min_frames_to_init:        " << opts_->min_frames_to_init
      << "\n  plane/plane_threshold:           " << opts_->plane_threshold
      << "\n  plane/plane_fit_mode:            " << opts_->plane_fit_mode
      << "\n  plane/plane_var_mode:            " << opts_->plane_var_mode
      << "\n  plane/plane_fit_pose_cov_mode:   " << opts_->plane_fit_pose_cov_mode
      << "\n  plane/bin_weight_mode_fit:       " << opts_->bin_weight_mode_fit
      << "\n  plane/bin_weight_mode_var:       " << opts_->bin_weight_mode_var
      << "\n  plane/bin_size_fraction:         " << opts_->bin_size_fraction
      << "\n  plane/use_bins:                  " << (opts_->use_bins ? "true" : "false")
      << "\n  plane/log_variance_shares_en:    " << (opts_->log_variance_shares_en ? "true" : "false")
      << "\n  plane/plane_gate_mode:           " << opts_->plane_gate_mode
      << "\n  plane/weight_floor/mode:         " << opts_->weight_floor_mode
      << "\n  plane/weight_floor/sigma_r2:     " << opts_->weight_sigma_r2
      << "\n  plane/log_consistency_mode:      " << opts_->log_consistency_mode
      << "\n  plane/log_consistency_corr_stride: " << opts_->log_consistency_corr_stride
      << "\n  plane/plane_var_denom_floor_en:  " << (opts_->plane_var_denom_floor_en ? "true" : "false")
      << "\n  map/shuffle_insertion_seed:      " << opts_->shuffle_insertion_seed
      << "\n  map/log_frame_stats_en:          " << (opts_->log_frame_stats_en ? "true" : "false")
      << "\n  points/min_init:  " << opts_->min_init_points
      << "\n  points/max:       " << opts_->max_points
      << "\n  points/min_update:" << opts_->min_update_points
      << "\n  residual/sigma_num_squared:  " << opts_->sigma_num_squared
      << "\n  residual/max_radius:         " << opts_->max_radius
      << "\n  residual/pose_cov_in_sigma:  " << (opts_->pose_cov_in_sigma ? "true" : "false")
      << "\n  search/neighborhood_size:    " << opts_->neighborhood_size;
  return oss.str();
}

bool VoxelMap::isEmpty() const { return voxel_map_.empty(); }

VoxelKey VoxelMap::worldToKey(const V3D& p_world) const
{
  return worldToKeyFn(p_world, opts_->voxel_size);
}

void VoxelMap::updateMap(MeasureGroup& mg) {
  TimedScope ts_total(profiler_, "voxelmap");
  setCurrentFrame(frame_idx_++);
  if (opts_->log_frame_stats_en) voxelPlaneFrameStatsReset();
  // The single point set (point_filter_num + ds_leaf_size, see measures.h's
  // docs on MeasureGroup::points) used for both voxel-map insertion and
  // LIO's own residual matching -- no separate full-resolution stream.
  const std::vector<PointXYZCov>& map_pts = mg.points;
  {
    TimedScope ts(profiler_, "voxelmap/bucket_build");
    const int np = static_cast<int>(map_pts.size());
    pts_world_.resize(np);

    {
      TimedScope ts(profiler_, "voxelmap/bucket_build/clear");
      for (const auto& key : active_keys_) {
        auto it = bucket_map_.find(key);
        it->second.world.clear();
        it->second.body.clear();
      }
      active_keys_.clear();
    }

    {
      TimedScope ts(profiler_, "voxelmap/bucket_build/transform");
      const int threads = cappedOmpThreads();
      thread_keys_.resize(threads);
      for (auto& v : thread_keys_) v.clear();

      // History (165-172): see docs/livo_recon_changelog.md#src-map-voxelmap.cpp-165
      std::vector<int> proc_order(np);
      std::iota(proc_order.begin(), proc_order.end(), 0);
      if (opts_->shuffle_insertion_seed != 0) {
        std::mt19937 rng(static_cast<unsigned int>(opts_->shuffle_insertion_seed) ^
                          static_cast<unsigned int>(frame_idx_));
        std::shuffle(proc_order.begin(), proc_order.end(), rng);
      }

      #pragma omp parallel for schedule(static) num_threads(threads)
      for (int oi = 0; oi < np; ++oi) {
        const int i = proc_order[oi];
        pts_world_[i] = state_->toWorld(map_pts[i]);
        // sensor_cov (see PointXYZCov's docs) is always sensor-only, never
        // mutated downstream -- a consumer needing the isotropic proxy
        // derives it inline (sensor_cov.trace()/3.0), no separate stored
        // scalar. pos_cov is captured directly here too (NOT derived by
        // subtracting from any combined value) -- a consumer that needs
        // the combined sensor+pose covariance (e.g. the PCA plane fit's
        // own residual-weighting propagation) adds `sensor_cov + pos_cov`
        // explicitly at the point of use. Matches gate()'s own
        // WorldPointCov, which already kept these separate.
        pts_world_[i].pos_cov = state_->poseCovAt(map_pts[i].point);
        thread_keys_[omp_get_thread_num()].emplace_back(
          worldToKey(pts_world_[i].point), i);
      }
    }

    {
      TimedScope ts(profiler_, "voxelmap/bucket_build/merge");
      for (const auto& local : thread_keys_)
        for (auto [key, i] : local) {
          auto& entry = bucket_map_[key];
          if (entry.world.empty()) active_keys_.push_back(key);
          entry.world.push_back(pts_world_[i]);
          entry.body.push_back(map_pts[i].point);
        }
    }
  }

  {
    // No insertions happen here — pointers into bucket_map_ are stable
    TimedScope ts(profiler_, "voxelmap/ensure_nodes");
    bucket_flat_.clear();
    bucket_flat_.reserve(active_keys_.size());
    for (const auto& key : active_keys_) {
      auto it = bucket_map_.find(key);
      ensureNode(key);
      bucket_flat_.emplace_back(&it->first, &it->second);
    }
  }

  last_n_map_pts_ = static_cast<int>(map_pts.size());
  last_n_active_voxels_ = static_cast<int>(active_keys_.size());

  const int n = static_cast<int>(bucket_flat_.size());

  {
    TimedScope ts(profiler_, "voxelmap/insert");
    const int threads = cappedOmpThreads();
    thread_plane_updates_.resize(threads);
    for (auto& v : thread_plane_updates_) v.clear();

    // CQ-35: `omp parallel` wrapping `omp for`, not a combined `omp parallel
    // for`, so flushPlaneFitStatsLog() can run once per thread after the
    // `omp for`'s implicit barrier -- see buildResiduals()' identical
    // pattern (lio_processing.cpp) and voxelplane.h's doc comment.
    #pragma omp parallel num_threads(threads)
    {
      #pragma omp for schedule(guided)
      for (int i = 0; i < n; ++i) {
        const auto& [key, entry] = bucket_flat_[i];
        auto it = voxel_map_.find(*key);
        assert(it != voxel_map_.end() && it->second != nullptr);
        it->second->insertPoints(entry->world, thread_plane_updates_[omp_get_thread_num()]);
      }
      flushPlaneFitStatsLog();
    }

    for (const auto& thread_updates : thread_plane_updates_) {
      for (const auto& u : thread_updates) {
        viz_dirty_[u.node_id] = u.deleted;
        if (u.deleted)
          viz_cache_.erase(u.node_id);
        else
          viz_cache_[u.node_id] = u.info;
      }
    }
  }

  if (opts_->log_frame_stats_en) {
    int denom_rejected_count = 0;
    double max_plane_var_trace = -1.0;
    voxelPlaneFrameStatsRead(denom_rejected_count, max_plane_var_trace);
    debugLogFrameStats(mg.image.t + data_queues_->start_time, frame_idx_ - 1,
                        denom_rejected_count, max_plane_var_trace,
                        lio_frame_diag_, stats_->planes.load(std::memory_order_relaxed),
                        stats_->total(),
                        stats_->planes.load(std::memory_order_relaxed),
                        stats_->converged.load(std::memory_order_relaxed),
                        voxelPlaneMaxCovarianceTrace());
  }
}

// ─────────────────────────────────────────────
//  VoxelPlane
// ─────────────────────────────────────────────

std::string VoxelMap::statsString() const
{
  std::ostringstream oss;
  oss << "[VoxelMap] total=" << stats_->total()
      << "  open="      << stats_->open
      << "  converged=" << stats_->converged
      << "  disabled="  << stats_->disabled
      << "  planes="    << stats_->planes;
  return oss.str();
}

void VoxelMap::ensureNode(const VoxelKey& key)
{
  auto [it, inserted] = voxel_map_.try_emplace(key, nullptr);
  if (inserted) {
    const double vs = opts_->voxel_size;
    const V3D center((key.x + 0.5) * vs, (key.y + 0.5) * vs, (key.z + 0.5) * vs);
    it->second = new VoxelNode(opts_, stats_, 0, center);
  }
}

bool VoxelMap::findPlaneResidualDirectional(const WorldPointCov& pt, const VoxelKey& base, Residual &res,
                                            VoxelKey* tried_key, int scan_id,
                                            bool* had_converged_neighbor) const
{
  // FAST-LIVO2-style single-neighbor step: the primary voxel missed, so
  // step to the one neighbor the point actually leans toward -- per axis,
  // compare the point's world position against that axis's voxel center
  // +/- a quarter voxel width, and step +1/-1 only on the axis(es) where
  // the point sits past that boundary. Cheap (at most one extra hash
  // lookup), but only ever tries that single candidate -- see
  // findPlaneResidual() for the tiered fallback this feeds into.
  const double vs = opts_->voxel_size;
  const double quarter = vs / 4.0;
  const V3D center((base.x + 0.5) * vs, (base.y + 0.5) * vs, (base.z + 0.5) * vs);

  VoxelKey near_key = base;
  bool has_neighbor = false;
  for (int axis = 0; axis < 3; ++axis) {
    const double d = pt.point[axis] - center[axis];
    int* comp = (axis == 0) ? &near_key.x : (axis == 1) ? &near_key.y : &near_key.z;
    if (d > quarter)       { *comp += 1; has_neighbor = true; }
    else if (d < -quarter) { *comp -= 1; has_neighbor = true; }
  }
  if (tried_key) *tried_key = near_key;
  if (!has_neighbor) return false;

  auto it_near = voxel_map_.find(near_key);
  if (it_near == voxel_map_.end() || !it_near->second) return false;
  if (had_converged_neighbor && it_near->second->hasConvergedPlane()) *had_converged_neighbor = true;
  if (!it_near->second->findPlaneResidual(pt, res, scan_id)) return false;
  res.match_tier = 1;
  return true;
}

bool VoxelMap::findPlaneResidualNeighborhood(const WorldPointCov& pt, const VoxelKey& base, Residual &res,
                                             const VoxelKey* exclude, int scan_id,
                                             bool* had_converged_neighbor) const
{
  // Exhaustive box search (original livo_recon approach): score every
  // candidate voxel within neighborhood_size in each direction and keep
  // the best-scoring match, rather than committing to a single candidate.
  // `exclude` (if non-null) skips one key already tried by an earlier tier
  // (the primary voxel and/or the directional neighbor), to avoid a
  // redundant findPlaneResidual() call against it.
  const int n = opts_->neighborhood_size;
  double best_score = -1.0;
  bool found = false;

  for (int dx = -n; dx <= n; ++dx)
  for (int dy = -n; dy <= n; ++dy)
  for (int dz = -n; dz <= n; ++dz)
  {
    VoxelKey key = base;
    key.x += dx; key.y += dy; key.z += dz;

    if (key == base) continue;
    if (exclude && key == *exclude) continue;

    auto it = voxel_map_.find(key);
    if (it == voxel_map_.end() || !it->second) continue;

    if (had_converged_neighbor && it->second->hasConvergedPlane()) *had_converged_neighbor = true;

    Residual cand;
    if (!it->second->findPlaneResidual(pt, cand, scan_id)) continue;

    const double score = (1.0 / std::sqrt(cand.sigma_squared)) *
                         std::exp(-0.5 * cand.r * cand.r / cand.sigma_squared);
    if (score > best_score) {
      best_score = score;
      res = cand;
      res.match_tier = 2;
      found = true;
    }
  }

  return found;
}

bool VoxelMap::hasConvergedNeighbor(const V3D& p_world) const
{
  const VoxelKey base = worldToKey(p_world);
  const int n = opts_->neighborhood_size;
  for (int dx = -n; dx <= n; ++dx)
  for (int dy = -n; dy <= n; ++dy)
  for (int dz = -n; dz <= n; ++dz) {
    VoxelKey key = base;
    key.x += dx; key.y += dy; key.z += dz;
    auto it = voxel_map_.find(key);
    if (it != voxel_map_.end() && it->second && it->second->hasConvergedPlane())
      return true;
  }
  return false;
}

bool VoxelMap::findPlaneResidual(const WorldPointCov& pt, Residual &res, bool* tier0_had_plane,
                                 bool* had_converged_neighbor) const
{
  const VoxelKey base = worldToKey(pt.point);

  auto it = voxel_map_.find(base);
  if (it != voxel_map_.end() && it->second && it->second->findPlaneResidual(pt, res, allow_consistency_log_ ? frame_idx_ : -1))
    return true;

  const bool base_converged = (it != voxel_map_.end() && it->second && it->second->hasConvergedPlane());
  if (tier0_had_plane)
    *tier0_had_plane = base_converged;
  // had_converged_neighbor accumulates the SAME footprint hasConvergedNeighbor()
  // independently scans (base + every cell within neighborhood_size) -- see
  // that method's own doc comment. base's contribution is known here already
  // (base_converged, just computed above); the directional/neighborhood
  // calls below add their own visited cells' contributions as they go,
  // giving the identical answer for free instead of via a second full box
  // scan on a miss.
  if (had_converged_neighbor && base_converged) *had_converged_neighbor = true;

  // Tiered fallback: (1) primary voxel [above] -- (2) the single directional
  // neighbor the point actually leans toward (cheap, catches the common
  // case) -- (3) only if that also misses, the full neighborhood_size box
  // search (expensive, but a real safety net -- see the residual-count
  // collapse this was reinstated to fix: the directional-only fallback let
  // matching thin out badly during dynamic motion, since a wrong single
  // guess had no second chance). frame_idx_ (this frame's scan_id, for
  // T0-D's corr.csv) is threaded through every tier the same way.
  VoxelKey directional_key = base;
  if (findPlaneResidualDirectional(pt, base, res, &directional_key, allow_consistency_log_ ? frame_idx_ : -1,
                                    had_converged_neighbor))
    return true;

  return findPlaneResidualNeighborhood(pt, base, res, &directional_key, allow_consistency_log_ ? frame_idx_ : -1,
                                       had_converged_neighbor);
}

// ── Visualization list helpers ────────────────────────────────────────────────

void VoxelMap::eraseDisc(int32_t id)
{
  auto it = disc_seg_idx_.find(id);
  if (it == disc_seg_idx_.end()) return;

  const uint32_t seg    = it->second;
  const uint32_t stride = DISC_SEGS * 3;
  const uint32_t p0     = seg * stride;
  auto& pts  = disc_list_.points;
  auto& cols = disc_list_.colors;
  const uint32_t last_seg = static_cast<uint32_t>(disc_seg_owner_.size()) - 1;

  if (seg != last_seg) {
    const uint32_t tail = last_seg * stride;
    std::copy(pts.begin()  + tail, pts.begin()  + tail + stride, pts.begin()  + p0);
    std::copy(cols.begin() + tail, cols.begin() + tail + stride, cols.begin() + p0);
    disc_seg_owner_[seg] = disc_seg_owner_[last_seg];
    disc_seg_idx_[disc_seg_owner_[seg]] = seg;
  }

  pts.resize(pts.size() - stride);
  cols.resize(cols.size() - stride);
  disc_seg_owner_.pop_back();
  disc_seg_idx_.erase(it);
}

void VoxelMap::eraseDot(int32_t id)
{
  auto it = dot_pt_idx_.find(id);
  if (it == dot_pt_idx_.end()) return;

  const uint32_t idx  = it->second;
  const uint32_t last = static_cast<uint32_t>(dot_pt_owner_.size()) - 1;
  auto& pts  = dot_list_.points;
  auto& cols = dot_list_.colors;

  if (idx != last) {
    pts[idx]  = pts[last];
    cols[idx] = cols[last];
    dot_pt_owner_[idx] = dot_pt_owner_[last];
    dot_pt_idx_[dot_pt_owner_[idx]] = idx;
  }

  pts.pop_back();
  cols.pop_back();
  dot_pt_owner_.pop_back();
  dot_pt_idx_.erase(it);
}

void VoxelMap::appendDisc(int32_t id, const PlaneVizInfo& p)
{
  // colour by world-space azimuth (HSV: S=0.85, V=0.9)
  const double hue = std::fmod(std::atan2(p.center.y(), p.center.x()) / (2.0 * M_PI) + 1.0, 1.0);
  const double h6  = hue * 6.0;
  const int    hi  = static_cast<int>(h6) % 6;
  const double f   = h6 - std::floor(h6);
  const double sv = 0.9, s = 0.85;
  const double pv = sv*(1-s), qv = sv*(1-s*f), tv = sv*(1-s*(1-f));
  float r, g, b;
  switch (hi) {
    case 0: r=sv; g=tv; b=pv; break;
    case 1: r=qv; g=sv; b=pv; break;
    case 2: r=pv; g=sv; b=tv; break;
    case 3: r=pv; g=qv; b=sv; break;
    case 4: r=tv; g=pv; b=sv; break;
    default:r=sv; g=pv; b=qv; break;
  }

  std_msgs::ColorRGBA col;
  col.r = r; col.g = g; col.b = b; col.a = 0.7f;

  // Build orthonormal basis in the plane
  const V3D& n = p.normal;
  const V3D  t = (std::abs(n.z()) < 0.9) ? V3D(0, 0, 1) : V3D(1, 0, 0);
  const V3D  xa = n.cross(t).normalized();
  const V3D  ya = n.cross(xa);

  // Square with side = 2*radius, built from 2 triangles
  const V3D v0 = p.center + p.radius * ( xa + ya);
  const V3D v1 = p.center + p.radius * (-xa + ya);
  const V3D v2 = p.center + p.radius * (-xa - ya);
  const V3D v3 = p.center + p.radius * ( xa - ya);

  const auto toPoint = [](const V3D& v) {
    geometry_msgs::Point pt;
    pt.x = v.x(); pt.y = v.y(); pt.z = v.z();
    return pt;
  };

  disc_list_.points.push_back(toPoint(v0));
  disc_list_.points.push_back(toPoint(v1));
  disc_list_.points.push_back(toPoint(v2));
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);
  disc_list_.points.push_back(toPoint(v0));
  disc_list_.points.push_back(toPoint(v2));
  disc_list_.points.push_back(toPoint(v3));
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);
  disc_list_.colors.push_back(col);

  disc_seg_idx_[id] = static_cast<uint32_t>(disc_seg_owner_.size());
  disc_seg_owner_.push_back(id);
}

void VoxelMap::appendDot(int32_t id, const PlaneVizInfo& p)
{
  geometry_msgs::Point pt;
  pt.x = p.center.x(); pt.y = p.center.y(); pt.z = p.center.z();
  dot_list_.points.push_back(pt);

  std_msgs::ColorRGBA col;
  col.r = 1.f; col.g = 1.f; col.b = 1.f; col.a = 1.f;
  dot_list_.colors.push_back(col);

  dot_pt_idx_[id] = static_cast<uint32_t>(dot_pt_owner_.size());
  dot_pt_owner_.push_back(id);
}

visualization_msgs::MarkerArray VoxelMap::buildVizMarkers(const std::string& frame_id,
                                                          const ros::Time& stamp)
{
  TimedScope ts(profiler_, "voxelmap/viz_markers");

  // Lazy-initialise the two list markers
  if (disc_list_.type == 0) {
    disc_list_.header.frame_id = frame_id;
    disc_list_.ns              = "voxel_planes";
    disc_list_.id              = 0;
    disc_list_.type            = visualization_msgs::Marker::TRIANGLE_LIST;
    disc_list_.action          = visualization_msgs::Marker::ADD;
    disc_list_.scale.x         = 1.0;
    disc_list_.scale.y         = 1.0;
    disc_list_.scale.z         = 1.0;
    disc_list_.pose.orientation.w = 1.0;
    disc_list_.color.a         = 1.0f;  // RViz gates TRIANGLE_LIST rendering on color.a > 0

    dot_list_.header.frame_id  = frame_id;
    dot_list_.ns               = "voxel_planes_converged";
    dot_list_.id               = 0;
    dot_list_.type             = visualization_msgs::Marker::SPHERE_LIST;
    dot_list_.action           = visualization_msgs::Marker::ADD;
    dot_list_.scale.x          = 0.04;
    dot_list_.scale.y          = 0.04;
    dot_list_.scale.z          = 0.04;
    dot_list_.pose.orientation.w = 1.0;
  }

  disc_list_.header.stamp = stamp;
  dot_list_.header.stamp  = stamp;

  for (const auto& [id, deleted] : viz_dirty_) {
    eraseDisc(id);
    eraseDot(id);
    if (!deleted) {
      const PlaneVizInfo& p = viz_cache_.at(id);
      appendDisc(id, p);
      if (p.is_converged) appendDot(id, p);
    }
  }

  viz_dirty_.clear();

  visualization_msgs::MarkerArray arr;
  arr.markers = {disc_list_, dot_list_};
  return arr;
}

}
