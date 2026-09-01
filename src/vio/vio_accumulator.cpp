#include "livo_recon/vio/vio_accumulator.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/log/consistency_log.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <omp.h>

namespace livo_recon
{

namespace
{

// Shared opt-in debug-log append helper -- mirrors vio_processing.cpp's own
// logIteration() (kept duplicated rather than shared across translation
// units for this one small helper, to avoid adding a header just for it).
void logIteration(const std::string& path, const std::string& msg)
{
  static std::set<std::string> truncated_paths;
  const bool first_call_for_path = truncated_paths.insert(path).second;
  std::ofstream ofs(path, first_call_for_path ? std::ios::trunc : std::ios::app);
  ofs << msg << "\n";
}

// Point-to-LINE distance (r=N/D, l=t_cc x y, D=||l_xy||) with the FULL
// Jacobian: dr/dx = dN/dx/D - r*(dD/dx)/D. See VioAccumulator::
// computeDirectionalEpipolar's doc comment for the caller-side context.
bool lineResidualAndJacobian(const V3D& t_cc, const V3D& y,
                             const M3D& dtcc_dtheta, const M3D& dtcc_dt, const M3D& dy_dtheta,
                             const V3D& x_compare,
                             double& residual_out, Eigen::Matrix<double, 1, 6>& jacobian_out,
                             double& norm_l_out)
{
  const V3D l = t_cc.cross(y);
  const double norm_l = std::sqrt(l.x() * l.x() + l.y() * l.y());
  norm_l_out = norm_l;
  if (norm_l < 1e-9) return false;  // degenerate epipolar geometry (near-zero baseline, e.g. pure rotation)

  M3D skew_y, skew_tcc;
  skew_y   << SKEW_SYM_MATRX(y);
  skew_tcc << SKEW_SYM_MATRX(t_cc);
  const M3D dl_dtheta = -skew_y * dtcc_dtheta + skew_tcc * dy_dtheta;
  const M3D dl_dt     = -skew_y * dtcc_dt;

  residual_out = x_compare.dot(l) / norm_l;
  jacobian_out.block<1, 3>(0, 0) = (x_compare.transpose() * dl_dtheta) / norm_l;
  jacobian_out.block<1, 3>(0, 3) = (x_compare.transpose() * dl_dt) / norm_l;

  const Eigen::RowVector2d l_xy(l.x(), l.y());
  const Eigen::RowVector3d dD_dtheta = (l_xy * dl_dtheta.topRows(2)) / norm_l;
  const Eigen::RowVector3d dD_dt     = (l_xy * dl_dt.topRows(2)) / norm_l;
  const double r_normalized = residual_out;
  jacobian_out.block<1, 3>(0, 0) -= r_normalized * dD_dtheta / norm_l;
  jacobian_out.block<1, 3>(0, 3) -= r_normalized * dD_dt / norm_l;
  return true;
}

// Ray-gap residual for "mixed" residual_mode: perpendicular distance between
// the anchor ray P1(s)=t_cc+s*y and the current ray P2(u)=u*x1 at their
// closest-approach depths s,u (already solved by solveClosestRayPoints(),
// same clamp-at-anchor_split_depth logic the epipolar residual's depth-
// clamped-weight decision uses). Ported verbatim from FAST-LIVO2's
// vio_proc.cpp rayGapResidualAndJacobian() -- s,u held fixed w.r.t. state in
// the Jacobian (envelope theorem, valid whether free or clamped). Unlike the
// epipolar residual, this is UNSQUARED and METRIC (meters) -- never scaled
// by pixel_scale -- so its weight constants (VioAccumulateOptions::
// pos_/neg_ray_distance_residual_var) are on a completely different numeric
// scale (~1000/~10) than the epipolar ones (~0.05/~0.03); the two residuals'
// raw magnitudes are not directly comparable, only their weighted HtH/Htz
// contributions are.
void rayGapResidualAndJacobian(const V3D& y, const V3D& x1, const V3D& t_cc,
                               const V3D& w, const M3D& Ric, const M3D& Rcw,
                               double s, double u,
                               double& residual_out, Eigen::Matrix<double, 1, 6>& jacobian_out)
{
  const V3D v = (t_cc + s * y) - u * x1;
  const double gap = v.norm();
  if (gap < 1e-12) { residual_out = 0.0; jacobian_out.setZero(); return; }
  residual_out = gap;

  M3D skew_y, skew_w;
  skew_y << SKEW_SYM_MATRX(y);
  skew_w << SKEW_SYM_MATRX(w);

  const M3D dtcc_dtheta = skew_w * Ric;
  const M3D dy_dtheta   = skew_y * Ric;
  const M3D dv_dtheta   = dtcc_dtheta + s * dy_dtheta;
  const M3D dv_dt       = -Rcw;

  jacobian_out.block<1, 3>(0, 0) = (v.transpose() * dv_dtheta) / gap;
  jacobian_out.block<1, 3>(0, 3) = (v.transpose() * dv_dt) / gap;
}

}  // namespace

VioAccumulator::VioAccumulator(StateGroupPtr state)
  : state_(state)
{}

ReprojectionPrecomputed VioAccumulator::computeReprojectionPrecomputed(const M3D& rot_curr, const V3D& pos_curr,
                                                                        const M3D& Rcw_curr,
                                                                        const M3D& rot_anchor, const V3D& pos_anchor) const
{
  ReprojectionPrecomputed pre;
  pre.R_common = rot_curr.transpose() * rot_anchor;
  pre.t_common = rot_curr.transpose() * (pos_anchor - pos_curr);
  pre.Ric = state_->Ric();
  pre.tic = state_->tic();
  pre.Rcw = Rcw_curr;
  return pre;
}

RayClosestApproach VioAccumulator::solveClosestRayPoints(const V3D& y, const V3D& x_compare, const V3D& t_cc,
                                                          double min_depth) const
{
  const double a = y.dot(y);
  const double b = y.dot(x_compare);
  const double c = x_compare.dot(x_compare);
  const double d = y.dot(t_cc);
  const double e = x_compare.dot(t_cc);
  const double denom = a * c - b * b;

  constexpr double kEps = 1e-9;
  RayClosestApproach out;

  if (std::fabs(denom) > kEps)
  {
    const double s0 = (b * e - c * d) / denom;
    const double u0 = (a * e - b * d) / denom;
    if (s0 >= min_depth && u0 >= min_depth)
    {
      out.s = s0; out.u = u0;
    }
    else if (s0 < min_depth)
    {
      out.s = min_depth; out.s_clamped = true;
      const double u_resolved = (e + out.s * b) / c;
      if (u_resolved >= min_depth) out.u = u_resolved;
      else { out.u = min_depth; out.u_clamped = true; }
    }
    else
    {
      out.u = min_depth; out.u_clamped = true;
      const double s_resolved = (out.u * b - d) / a;
      if (s_resolved >= min_depth) out.s = s_resolved;
      else { out.s = min_depth; out.s_clamped = true; }
    }
  }
  else
  {
    out.parallel = true;
    out.s = min_depth; out.s_clamped = true;
    const double u_resolved = (e + out.s * b) / c;
    if (u_resolved >= min_depth) out.u = u_resolved;
    else { out.u = min_depth; out.u_clamped = true; }
  }

  return out;
}

bool VioAccumulator::computeDirectionalEpipolar(const V3D& t_cc, const V3D& y,
                                                const M3D& dtcc_dtheta, const M3D& dtcc_dt, const M3D& dy_dtheta,
                                                const V3D& x_compare, const VioAccumulateOptions& opts,
                                                double& residual_out, Eigen::Matrix<double, 1, 6>& jacobian_out,
                                                double& norm_l_out, bool* depth_clamped_out) const
{
  const RayClosestApproach free_diag =
      solveClosestRayPoints(y, x_compare, t_cc, -std::numeric_limits<double>::infinity());
  if (depth_clamped_out) *depth_clamped_out = free_diag.s < opts.anchor_split_depth;

  return lineResidualAndJacobian(t_cc, y, dtcc_dtheta, dtcc_dt, dy_dtheta, x_compare,
                                 residual_out, jacobian_out, norm_l_out);
}

double VioAccumulator::epipolarLineWeightVar(const V3D& t_cc, const V3D& y, const V3D& x1, const M3D& R_cc,
                                             const Eigen::Matrix2d& anchor_uv_jacobian,
                                             const Eigen::Matrix2d& curr_uv_jacobian,
                                             double fx, double fy, const VioAccumulateOptions& opts) const
{
  const V3D l = t_cc.cross(y);
  const double norm_l = std::sqrt(l.x() * l.x() + l.y() * l.y());
  if (norm_l < 1e-9) return std::numeric_limits<double>::infinity();

  const double r_epi = x1.dot(l) / norm_l;
  const Eigen::RowVector2d g1(l.x() / norm_l, l.y() / norm_l);

  M3D skew_tcc;
  skew_tcc << SKEW_SYM_MATRX(t_cc);
  const Eigen::Matrix<double, 3, 2> dl_du0v0 = (-skew_tcc * R_cc).leftCols(2);
  const Eigen::RowVector2d dN_du0v0 = (x1.transpose() * dl_du0v0) / norm_l;
  const Eigen::RowVector2d l_xy(l.x(), l.y());
  const Eigen::RowVector2d dD_du0v0 = (l_xy * dl_du0v0.topRows(2)) / norm_l;
  const Eigen::RowVector2d g0 = dN_du0v0 - r_epi * dD_du0v0 / norm_l;

  const double sigma_px2 = opts.weight_pixel_noise_px * opts.weight_pixel_noise_px;
  const Eigen::Matrix2d sigma_uv0 = anchor_uv_jacobian * (sigma_px2 * Eigen::Matrix2d::Identity()) * anchor_uv_jacobian.transpose();
  const Eigen::Matrix2d sigma_uv1 = curr_uv_jacobian * (sigma_px2 * Eigen::Matrix2d::Identity()) * curr_uv_jacobian.transpose();

  const double var_normalized = (g0 * sigma_uv0 * g0.transpose())(0, 0)
                               + (g1 * sigma_uv1 * g1.transpose())(0, 0);
  const double pixel_scale = 0.5 * (fx + fy);
  return std::max(var_normalized * pixel_scale * pixel_scale, std::numeric_limits<double>::min());
}

bool VioAccumulator::computeEpipolarResidualAndJacobian(const ReprojectionPrecomputed& pre,
                                                        double fx, double fy, double cx, double cy,
                                                        const cv::Point2f& uv_seed,
                                                        const cv::Point2f& uv_tracked,
                                                        const Eigen::Matrix2d& anchor_uv_jacobian,
                                                        const Eigen::Matrix2d& curr_uv_jacobian,
                                                        const VioAccumulateOptions& opts,
                                                        double& residual_out,
                                                        Eigen::Matrix<double, 1, 6>& jacobian_out,
                                                        double& norm_l_out,
                                                        bool& depth_clamped_out,
                                                        double* weight_var_out) const
{
  depth_clamped_out = false;
  const V3D x0((uv_seed.x - cx) / fx, (uv_seed.y - cy) / fy, 1.0);
  const V3D x1((uv_tracked.x - cx) / fx, (uv_tracked.y - cy) / fy, 1.0);

  const M3D R_cc = pre.Ric * pre.R_common * pre.Ric.transpose();
  const V3D t_cc = (M3D::Identity() - R_cc) * pre.tic + pre.Ric * pre.t_common;
  const V3D y = R_cc * x0;

  const V3D w = pre.Ric * pre.t_common - R_cc * pre.tic;

  M3D skew_y, skew_w;
  skew_y << SKEW_SYM_MATRX(y);
  skew_w << SKEW_SYM_MATRX(w);
  const M3D dtcc_dtheta = skew_w * pre.Ric;
  const M3D dtcc_dt     = -pre.Rcw;
  const M3D dy_dtheta   = skew_y * pre.Ric;

  const bool forward_ok = computeDirectionalEpipolar(t_cc, y, dtcc_dtheta, dtcc_dt, dy_dtheta, x1, opts,
                                                      residual_out, jacobian_out, norm_l_out, &depth_clamped_out);
  if (!forward_ok) return false;

  if (weight_var_out)
    *weight_var_out = epipolarLineWeightVar(t_cc, y, x1, R_cc, anchor_uv_jacobian, curr_uv_jacobian, fx, fy, opts);

  if (opts.bidirectional)
  {
    const M3D Rcc_T = R_cc.transpose();
    const V3D y_rev = Rcc_T * x1;
    const V3D t_cc_rev = -Rcc_T * t_cc;
    const V3D tcc_minus_w = t_cc - w;
    M3D skew_x1, skew_tcc_minus_w;
    skew_x1          << SKEW_SYM_MATRX(x1);
    skew_tcc_minus_w << SKEW_SYM_MATRX(tcc_minus_w);
    const M3D dy_rev_dtheta   = -Rcc_T * skew_x1 * pre.Ric;
    const M3D dtcc_rev_dtheta = Rcc_T * skew_tcc_minus_w * pre.Ric;
    const M3D dtcc_rev_dt     = Rcc_T * pre.Rcw;

    double r_back = 0.0;
    Eigen::Matrix<double, 1, 6> jac_back = Eigen::Matrix<double, 1, 6>::Zero();
    double norm_l_rev_dummy = 0.0;
    const bool reverse_ok = computeDirectionalEpipolar(t_cc_rev, y_rev, dtcc_rev_dtheta, dtcc_rev_dt, dy_rev_dtheta,
                                                        x0, opts, r_back, jac_back, norm_l_rev_dummy, nullptr);
    if (reverse_ok)
    {
      residual_out += r_back;
      jacobian_out += jac_back;
    }
  }

  const double pixel_scale = 0.5 * (fx + fy);
  residual_out *= pixel_scale;
  jacobian_out *= pixel_scale;

  return true;
}

bool VioAccumulator::computeRayDistanceResidualAndJacobian(const ReprojectionPrecomputed& pre,
                                                            double fx, double fy, double cx, double cy,
                                                            const cv::Point2f& uv_seed,
                                                            const cv::Point2f& uv_tracked,
                                                            const Eigen::Matrix2d& anchor_uv_jacobian,
                                                            const Eigen::Matrix2d& curr_uv_jacobian,
                                                            const VioAccumulateOptions& opts,
                                                            double& residual_out,
                                                            Eigen::Matrix<double, 1, 6>& jacobian_out,
                                                            bool& depth_clamped_out,
                                                            double* weight_var_out) const
{
  const V3D x0((uv_seed.x - cx) / fx, (uv_seed.y - cy) / fy, 1.0);
  const V3D x1((uv_tracked.x - cx) / fx, (uv_tracked.y - cy) / fy, 1.0);

  const M3D R_cc = pre.Ric * pre.R_common * pre.Ric.transpose();
  const V3D t_cc = (M3D::Identity() - R_cc) * pre.tic + pre.Ric * pre.t_common;
  const V3D y = R_cc * x0;
  const V3D w = pre.Ric * pre.t_common - R_cc * pre.tic;

  const RayClosestApproach diag = solveClosestRayPoints(y, x1, t_cc, opts.anchor_split_depth);
  depth_clamped_out = diag.s_clamped || diag.u_clamped;

  rayGapResidualAndJacobian(y, x1, t_cc, w, pre.Ric, pre.Rcw, diag.s, diag.u, residual_out, jacobian_out);

  // See computeRayDistanceResidualAndJacobian's own doc comment: FAST-LIVO2
  // reuses epipolarLineWeightVar() verbatim for raydist's distortion weight
  // too (a fresh from-scratch Var(||v||) derivation was tried and found
  // worse there) -- matched here rather than deriving a new one.
  if (weight_var_out)
    *weight_var_out = epipolarLineWeightVar(t_cc, y, x1, R_cc, anchor_uv_jacobian, curr_uv_jacobian, fx, fy, opts);

  return true;
}

bool VioAccumulator::accumulate(const TrackedFrame& frame, const std::vector<AnchorPoint>& anchors,
                                const VioAccumulateOptions& opts, EkfUpdate& out,
                                float& avg_error, int& n_valid, double t_abs, int iter) const
{
  const M3D rot_curr = state_->rot();
  const V3D pos_curr = state_->pos();
  const M3D Rcw = state_->Rcw();
  const double fx = state_->cameraFx(), fy = state_->cameraFy();
  const double cx = state_->cameraCx(), cy = state_->cameraCy();

  const ReprojectionPrecomputed pre = computeReprojectionPrecomputed(rot_curr, pos_curr, Rcw,
                                                                     frame.rot_anchor, frame.pos_anchor);

  const M3D R_cc_frame = state_->Ric() * pre.R_common * state_->Ric().transpose();
  const double baseline_cc = ((M3D::Identity() - R_cc_frame) * pre.tic + state_->Ric() * pre.t_common).norm();

  out.reset();

  const std::vector<int>& indices = frame.indices;
  const std::vector<cv::Point2f>& uv_curr = frame.uv_curr;
  const int n = static_cast<int>(indices.size());
  const int n_threads = cappedOmpThreads();
  // Per-POINT (not per-thread) accumulators for HtH/Htz/err -- see the doc
  // comment on the final reduction loop below for why: this makes the
  // floating-point summation order depend only on the point count (fixed
  // by this frame's own tracked-point data), never on how many OMP threads
  // happened to run this call. Fixes a confirmed run-to-run non-
  // determinism (2026-08-16): varying thread counts produced different
  // summation groupings and therefore different LSB-level rounding in
  // double-precision HtH/Htz, which the iterative IEKF solve then
  // amplified into materially, occasionally catastrophically different
  // results for byte-identical input on ill-conditioned frames -- see
  // pinOmpThreadsForDeterminism()'s doc comment (utils/algo/omp_utils.h)
  // for the original investigation. That fix pins the thread count so a
  // given machine is internally reproducible; THIS fix makes the actual
  // numeric result independent of thread count entirely, so two machines
  // (or the same machine before/after a hardware/OMP_NUM_THREADS change)
  // computing the identical frame get the identical HtH/Htz too.
  std::vector<Eigen::Matrix<double, 6, 6>> point_HtH(n, Eigen::Matrix<double, 6, 6>::Zero());
  std::vector<Eigen::Matrix<double, 6, 1>> point_Htz(n, Eigen::Matrix<double, 6, 1>::Zero());
  std::vector<double> point_err(n, 0.0);
  std::vector<int> thread_n(n_threads, 0);
  std::vector<int> thread_n_reject(n_threads, 0);
  std::vector<int> thread_n_low_norml(n_threads, 0);
  const bool do_point_log = opts.log_point_residuals;
  std::vector<std::string> thread_point_log(do_point_log ? n_threads : 0);

  #pragma omp parallel for schedule(static) num_threads(n_threads)
  for (int i = 0; i < n; ++i)
  {
    const AnchorPoint& anchor = anchors[indices[i]];
    const int t = omp_get_thread_num();

    if (opts.residual_mode == "mixed")
    {
      // See VioAccumulateOptions::residual_mode's doc comment -- two
      // independently-weighted, DOF-disjoint residuals per point instead of
      // one conflated one. Ported from FAST-LIVO2's ResidualMode::kMixed.
      double residual_epi = 0.0, residual_ray = 0.0, norm_l = 0.0;
      Eigen::Matrix<double, 1, 6> J_epi, J_ray;
      bool depth_clamped_epi = false, depth_clamped_ray = false;
      double weight_var_epi = -1.0, weight_var_ray = -1.0;

      const bool ok_epi = computeEpipolarResidualAndJacobian(
          pre, fx, fy, cx, cy, anchor.uv_seed, uv_curr[i],
          anchor.uv_jacobian_seed, frame.uv_curr_uv_jacobian[i], opts,
          residual_epi, J_epi, norm_l, depth_clamped_epi,
          opts.distortion_weight_on ? &weight_var_epi : nullptr);
      const bool ok_ray = computeRayDistanceResidualAndJacobian(
          pre, fx, fy, cx, cy, anchor.uv_seed, uv_curr[i],
          anchor.uv_jacobian_seed, frame.uv_curr_uv_jacobian[i], opts,
          residual_ray, J_ray, depth_clamped_ray,
          opts.distortion_weight_on ? &weight_var_ray : nullptr);

      // Point accepted only if BOTH residuals compute successfully --
      // matches FAST-LIVO2's own "reject the point entirely unless BOTH
      // succeed" behavior (a point with no valid rotation vote AND no valid
      // position vote contributes nothing useful either way).
      if (!ok_epi || !ok_ray)
      {
        if (opts.log_residual_counts) ++thread_n_reject[t];
        if (do_point_log)
        {
          std::ostringstream ps;
          ps << "[vio/point_residual]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
             << "  iter=" << iter << "  pt=" << i << "  ok=0\n";
          thread_point_log[t] += ps.str();
        }
        continue;
      }

      double weight_epi = depth_clamped_epi ? opts.neg_epipolar_residual_var
                                             : opts.pos_epipolar_residual_var;
      double weight_ray = depth_clamped_ray ? opts.neg_ray_distance_residual_var
                                             : opts.pos_ray_distance_residual_var;
      if (opts.distortion_weight_on && weight_var_epi > 0.0 && std::isfinite(weight_var_epi))
        weight_epi *= 1.0 / weight_var_epi;
      if (opts.distortion_weight_on && weight_var_ray > 0.0 && std::isfinite(weight_var_ray))
        weight_ray *= 1.0 / weight_var_ray;

      if (opts.log_residual_counts && norm_l < opts.log_norm_l_low_frac * baseline_cc)
        ++thread_n_low_norml[t];

      // Zero the columns each residual shouldn't influence -- epipolar
      // keeps rotation only, raydist keeps position only.
      Eigen::Matrix<double, 1, 6> J_epi_rot_only = J_epi; J_epi_rot_only.tail<3>().setZero();
      Eigen::Matrix<double, 1, 6> J_ray_pos_only = J_ray; J_ray_pos_only.head<3>().setZero();

      const double mdist2 = residual_epi * residual_epi * weight_epi
                           + residual_ray * residual_ray * weight_ray;

      point_HtH[i].noalias() += J_epi_rot_only.transpose() * weight_epi * J_epi_rot_only;
      point_HtH[i].noalias() += J_ray_pos_only.transpose() * weight_ray * J_ray_pos_only;
      point_Htz[i].noalias() += J_epi_rot_only.transpose() * weight_epi * residual_epi;
      point_Htz[i].noalias() += J_ray_pos_only.transpose() * weight_ray * residual_ray;
      point_err[i] = mdist2;
      ++thread_n[t];

      if (do_point_log)
      {
        std::ostringstream ps;
        ps << "[vio/point_residual]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
           << "  iter=" << iter << "  pt=" << i << "  ok=1"
           << std::scientific << std::setprecision(4)
           << "  residual_epi=" << residual_epi << "  residual_ray=" << residual_ray
           << "  norm_l=" << norm_l
           << "  depth_clamped_epi=" << (depth_clamped_epi ? 1 : 0)
           << "  depth_clamped_ray=" << (depth_clamped_ray ? 1 : 0)
           << "  weight_epi=" << weight_epi << "  weight_ray=" << weight_ray
           << "  htH_rr=" << (J_epi_rot_only.head<3>().squaredNorm() * weight_epi)
           << "  htH_pp=" << (J_ray_pos_only.tail<3>().squaredNorm() * weight_ray) << "\n";
        thread_point_log[t] += ps.str();
      }
      continue;
    }

    double residual, norm_l;
    Eigen::Matrix<double, 1, 6> J;
    bool depth_clamped = false;
    double weight_var = -1.0;
    const bool ok = computeEpipolarResidualAndJacobian(
        pre, fx, fy, cx, cy, anchor.uv_seed, uv_curr[i],
        anchor.uv_jacobian_seed, frame.uv_curr_uv_jacobian[i], opts,
        residual, J, norm_l, depth_clamped,
        opts.distortion_weight_on ? &weight_var : nullptr);

    if (!ok)
    {
      if (opts.log_residual_counts) ++thread_n_reject[t];
      if (do_point_log)
      {
        std::ostringstream ps;
        ps << "[vio/point_residual]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
           << "  iter=" << iter << "  pt=" << i << "  ok=0\n";
        thread_point_log[t] += ps.str();
      }
      continue;
    }

    double weight = depth_clamped ? opts.neg_epipolar_residual_var
                                   : opts.pos_epipolar_residual_var;

    if (opts.distortion_weight_on && weight_var > 0.0 && std::isfinite(weight_var))
      weight *= 1.0 / weight_var;

    if (opts.log_residual_counts && norm_l < opts.log_norm_l_low_frac * baseline_cc)
      ++thread_n_low_norml[t];

    // avg_error/mdist2 (used for both solveSystem()'s reject-if-worse check
    // and the whole-frame max_avg_error threshold) stays on the base,
    // un-split weight -- only HtH/Htz's actual position-column contribution
    // is scaled down below, so existing max_avg_error tuning keeps meaning
    // what it always meant.
    const double mdist2 = residual * residual * weight;

    if (opts.log_consistency_corr_en) {
      // Logged BEFORE any further rejection, for the same reason corr.csv
      // logs candidates rather than acceptances: a NIS computed over the
      // survivors of its own gate is biased low, which T0-D measured at
      // 2.59 -> 1.85 on a fixture. weight is an inverse variance, so S = 1/weight.
      logConsistencyVioCorr(t_abs, iter, residual, (weight > 0.0 ? 1.0 / weight : -1.0),
                            opts.residual_mode.c_str());
    }

    // See VioAccumulateOptions::pos_weight_scale's doc comment -- scale
    // J's position columns (last 3) directly before forming the outer
    // product, leaving rotation columns (first 3) untouched. This makes
    // HtH's rotation-rotation block exactly as before, its position-
    // position block scaled by pos_weight_scale^2, its rotation/position
    // cross terms scaled by pos_weight_scale^1, and Htz's position rows
    // scaled by pos_weight_scale^1 -- the standard effect of reducing a
    // scalar residual's effective sensitivity to a subset of state
    // dimensions (equivalent to trusting this residual's implied position
    // correction less, without discarding its rotation information).
    Eigen::Matrix<double, 1, 6> J_scaled = J;
    J_scaled.tail<3>() *= opts.pos_weight_scale;

    point_HtH[i].noalias() += J_scaled.transpose() * weight * J_scaled;
    point_Htz[i].noalias() += J_scaled.transpose() * weight * residual;
    point_err[i] = mdist2;
    ++thread_n[t];

    if (do_point_log)
    {
      const Eigen::Matrix<double, 6, 1> Htz_i = J_scaled.transpose() * weight * residual;
      std::ostringstream ps;
      ps << "[vio/point_residual]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
         << "  iter=" << iter << "  pt=" << i << "  ok=1"
         << std::scientific << std::setprecision(4)
         << "  residual=" << residual << "  norm_l=" << norm_l
         << "  depth_clamped=" << (depth_clamped ? 1 : 0)
         << "  weight=" << weight << "  weight_var=" << weight_var
         << "  htH_rr=" << (J.head<3>().squaredNorm() * weight)
         << "  htH_pp=" << (J.tail<3>().squaredNorm() * weight * opts.pos_weight_scale)
         << "  J=[" << J << "]"
         << "  Htz_i=[" << Htz_i.transpose() << "]\n";
      thread_point_log[t] += ps.str();
    }
  }

  // Deterministic, thread-count-independent reduction: always sums in
  // fixed point-index order 0..n-1, regardless of how many OMP threads
  // computed the individual point_HtH[i]/point_Htz[i]/point_err[i] entries
  // above -- see their declaration's doc comment.
  double total_err = 0.0;
  for (int i = 0; i < n; ++i)
  {
    out.HtH.noalias() += point_HtH[i];
    out.Htz.noalias() += point_Htz[i];
    total_err += point_err[i];
  }

  n_valid = 0;
  int n_reject = 0, n_low_norml = 0;
  for (int t = 0; t < n_threads; ++t)
  {
    n_valid += thread_n[t];
    n_reject += thread_n_reject[t];
    n_low_norml += thread_n_low_norml[t];
    if (do_point_log) logIteration(opts.log_path, thread_point_log[t]);
  }
  out.n_meas = n_valid;

  if (opts.log_residual_counts)
  {
    std::ostringstream rc;
    rc << "[vio/residual_counts]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
       << "  iter=" << iter << "  n_valid=" << n_valid
       << "  n_reject_degenerate=" << n_reject
       << "  frac_low_norm_l=" << (n_valid > 0 ? static_cast<double>(n_low_norml) / n_valid : 0.0)
       << "  baseline_cc=" << std::scientific << std::setprecision(3) << baseline_cc;
    logIteration(opts.log_path, rc.str());
  }

  if (n_valid == 0) return false;

  avg_error = static_cast<float>(total_err / n_valid);
  return true;
}

}  // namespace livo_recon
