#include "livo_recon/lio/voxelplane.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include <fstream>
#include <sstream>

namespace livo_recon
{

namespace
{
void debugLogNoiseFloor(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("noise_floor.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}
}  // namespace

VoxelPlane::VoxelPlane(VoxelOptsPtr opts)
  : opts_(opts)
{
  plane_var_ = M3D::Zero();
  covariance_ = M3D::Zero();
  for (int a = 0; a < 3; ++a) {
    W_[a] = M3D::Zero();
    for (int b = 0; b < 3; ++b) V_[a][b] = M3D::Zero();
  }
}

bool VoxelPlane::gate(const V3D& p, const M3D& sensor_cov, const M3D& pose_cov,
                      double& r, double& sigma_diag_squared, double& plane_var_term,
                      Eigen::Matrix<double, 1, 3>& J_nq) const
{
  const V3D& n = plane_.normal;
  r = n.dot(p) + plane_.d;
  if (!std::isfinite(r)) return false;

  const double dis_to_center = (p - plane_.center).squaredNorm();
  const double range_dis = std::sqrt(std::max(0.0, dis_to_center - r * r));
  if (range_dis > opts_->max_radius * radius_) return false;

  const V3D d_center = p - plane_.center;
  J_nq(0, 0) = d_center.dot(y_normal_);
  J_nq(0, 1) = d_center.dot(x_normal_);
  J_nq(0, 2) = 1.0;

  // sigma_diag_squared: sensor + (optionally) pose VARIANCE. See
  // VoxelOpts::pose_cov_in_sigma's docs for the double-counting tradeoff
  // this toggles.
  sigma_diag_squared = n.dot(sensor_cov * n);
  if (opts_->pose_cov_in_sigma)
    sigma_diag_squared += n.dot(pose_cov * n);
  if (!std::isfinite(sigma_diag_squared) || sigma_diag_squared <= 0.0) return false;
  if (sigma_diag_squared < 1e-6) sigma_diag_squared = 1e-6;

  plane_var_term = (J_nq * plane_var_ * J_nq.transpose()).value();
  const double sigma_gate_squared = sigma_diag_squared + plane_var_term;
  if (!std::isfinite(sigma_gate_squared) || sigma_gate_squared <= 0.0) return false;

  return r * r <= opts_->sigma_num_squared * sigma_gate_squared;
}

bool VoxelPlane::computeResidual(const WorldPointCov& pt, Residual& res) const
{
  if (!is_plane_) return false;

  double r, sigma_diag_squared, plane_var_term;
  Eigen::Matrix<double, 1, 3> J_nq;
  if (!gate(pt.point, pt.sensor_cov, pt.pose_cov, r, sigma_diag_squared, plane_var_term, J_nq)) return false;

  res.r              = r;
  res.normal         = plane_.normal;
  res.sigma_squared  = 1e-3 + sigma_diag_squared;
  res.plane_id       = this;
  res.plane_var_term = plane_var_term;
  return true;
}

bool VoxelPlane::getVizInfo(PlaneVizInfo& info) const
{
  if (!is_plane_) return false;
  info.center       = plane_.center;
  info.normal       = plane_.normal;
  info.radius       = radius_;
  info.is_converged = isFull();
  return true;
}

void VoxelPlane::update(const std::vector<PointXYZCov>& points, int total_count,
                        const std::vector<double>* weights_in,
                        const RunningMoments* running)
{
  plane_var_.setZero();
  covariance_.setZero();
  plane_ = {};
  is_plane_ = false;

  const int N = (int)points.size();
  points_size_ = (total_count >= 0) ? total_count : N;
  if (N < 3) return;

  // `weights`: supplied by the caller (VoxelNode always pre-bins whenever
  // convergence_mode=="always_update" -- see useBins()) or left null for
  // the plain unweighted path below.
  const std::vector<double>* weights = weights_in;
  double weight_sum = N;
  const bool use_weights = (weights_in != nullptr);

  if (running) {
    // O(1) mean/covariance from pre-accumulated running sums (see
    // RunningMoments' docs) instead of the two O(N) loops below -- exact,
    // not an approximation (only ever passed for VoxelNode's equal-weight
    // points_ path, so this and use_weights are mutually exclusive).
    const V3D center_rel = running->sum_p / running->sum_w;
    covariance_   = running->sum_pp / running->sum_w - center_rel * center_rel.transpose();
    plane_.center = running->ref + center_rel;
  } else if (use_weights) {
    weight_sum = 0.0;
    for (double w : *weights) weight_sum += w;

    for (int i = 0; i < N; ++i)
      plane_.center += (*weights)[i] * points[i].point;
    plane_.center /= weight_sum;

    for (int i = 0; i < N; ++i) {
      V3D d = points[i].point - plane_.center;
      covariance_ += (*weights)[i] * (d * d.transpose());
    }
    covariance_ /= weight_sum;
  } else {
    for (const auto& pt : points)
      plane_.center += pt.point;
    plane_.center /= N;

    for (const auto& pt : points) {
      V3D d = pt.point - plane_.center;
      covariance_ += d * d.transpose();
    }
    covariance_ /= N;
  }

  Eigen::SelfAdjointEigenSolver<M3D> solver(covariance_);
  if (solver.info() != Eigen::Success) return;

  eigen_values_ = solver.eigenvalues();
  const M3D evecs = solver.eigenvectors();

  if (eigen_values_(1) < 1e-8 || eigen_values_(2) < 1e-8)
    return;

  if (opts_->sensor_noise_floor_eig0) {
    // Uses points[i].sensor_cov's isotropic (trace/3) proxy rather than
    // points[i].sensor_cov+points[i].pos_cov (combined): pose
    // uncertainty is common to every point in a frame
    // and reflects "how sure am I of my own pose", not per-point
    // measurement precision, and its range^2 lever-arm scaling can dwarf
    // true sensor noise for far points, which made an earlier cov-based
    // version of this floor reject almost everything (confirmed
    // empirically: median floor ~119 vs plane_threshold=0.0025, a ~5
    // order-of-magnitude mismatch).
    double sensor_floor = 0.0;
    for (int i = 0; i < N; ++i) {
      const double w = use_weights ? ((*weights)[i] / weight_sum) : (1.0 / N);
      sensor_floor += w * (points[i].sensor_cov.trace() / 3.0);
    }
    if (opts_->log_debug_en)
    {
      std::ostringstream dbg;
      dbg << "N=" << N << " raw_eig0=" << eigen_values_(0) << " sensor_floor=" << sensor_floor
          << " plane_threshold=" << opts_->plane_threshold;
      debugLogNoiseFloor(dbg.str());
    }
    eigen_values_(0) = std::max(eigen_values_(0), sensor_floor);
    // The floor alone already explains the observed "flatness" -- this
    // candidate can't be distinguished from non-planar, so reject rather
    // than risk denom1=eigen_values_(0)-eigen_values_(1) collapsing to
    // ~zero in the plane_var_ Jacobian below.
    if (eigen_values_(0) >= eigen_values_(1)) return;
  }

  // Acceptance is min-eigenvalue-only, matching FAST-LIVO2's init_plane()
  // exactly (see voxel_map.cpp there) -- an extra eigen0/eigen1 ratio check
  // this used to also require was confirmed (via a 4-run A/B on NTU VIRAL
  // eee_01) to reject ~90% of livo_recon's plane candidates that this
  // eigen0-only test accepts (44% vs 94% acceptance rate), starving the
  // map of usable planes and driving a 2.16m full-run ATE divergence
  // FAST-LIVO2 doesn't share; removing it (alone) cut that to 0.093m.
  is_plane_ = eigen_values_(0) < opts_->plane_threshold;
  if (!is_plane_) return;

  // NOTE: no denom1/denom2 near-degenerate-eigengap guard here, unlike
  // refitDebiased() below -- tried applying the same guard to this PCA
  // path too (this failure mode isn't theoretically unique to debiasing),
  // but it measurably changed PCA-mode's own output (eee_01 ATE 0.0260m
  // -> 0.0249m), i.e. it DOES trigger occasionally even in plain PCA.
  // PCA has long-validated production behavior with no evidence this
  // mechanism is a real problem for it (the debiased-mode blowup is
  // specifically because debiasing shrinks eig0 toward eig1/eig2
  // together, making a small gap far more common there) -- reverted here
  // to preserve PCA's exact original byte-identical behavior. Revisit
  // only if PCA mode is later found to have its own denom-blowup problem.
  const double denom1 = eigen_values_(0) - eigen_values_(1);
  const double denom2 = eigen_values_(0) - eigen_values_(2);

  plane_.normal = evecs.col(0).normalized();
  y_normal_     = evecs.col(1);
  x_normal_     = evecs.col(2);
  plane_.d      = -plane_.normal.dot(plane_.center);
  radius_       = (float)std::sqrt(eigen_values_(2));

  const double inv_N  = 1.0 / N;

  for (int i = 0; i < N; ++i) {
    const auto& pt = points[i];
    const V3D z  = pt.point - plane_.center;
    const double a0 = plane_.normal.dot(z);
    const double a1 = y_normal_.dot(z);
    const double a2 = x_normal_.dot(z);

    // Each point's true contribution to d(mean)/d(point_i) and
    // d(covariance)/d(point_i) is w_i/weight_sum under the weighted
    // mean/covariance computed above -- inv_N (equal 1/N weighting) is
    // only correct in the unweighted path. Using inv_N unconditionally
    // here when weighting is on would over-count dense/high-weight
    // entries and under-count sparse/low-weight ones in this
    // fit-uncertainty propagation.
    const double w = use_weights ? ((*weights)[i] / weight_sum) : inv_N;

    // Minimal 3x3 Jmin, the direct equivalent of T * J_old (see
    // plane_var_'s docs in voxelplane.h) -- row0/row1 are
    // y_normal_^T*J_n / x_normal_^T*J_n (J_n's output is confined to
    // span{y_normal_,x_normal_}, so these pick out its two components
    // exactly), row2 is -normal_^T*(w*I) (the center-perturbation block's
    // projection onto d).
    Eigen::Matrix<double, 1, 3> row_theta1 =
        (w / denom1) * (a1 * plane_.normal.transpose() + a0 * y_normal_.transpose());
    Eigen::Matrix<double, 1, 3> row_theta2 =
        (w / denom2) * (a2 * plane_.normal.transpose() + a0 * x_normal_.transpose());
    Eigen::Matrix<double, 1, 3> row_d = -w * plane_.normal.transpose();

    M3D Jmin;
    Jmin.row(0) = row_theta1;
    Jmin.row(1) = row_theta2;
    Jmin.row(2) = row_d;

    // "combined" (default, unchanged behavior): full sensor+pose noise
    // budget, same spirit as always. "sensor_only" (new ablation, see
    // VoxelOpts::plane_fit_pose_cov_mode's docs): excludes pos_cov from
    // this fit-uncertainty propagation entirely, testing the same
    // exclusion idea already validated for gate()'s sigma_diag_squared
    // (pose_cov_in_sigma=false) but applied to plane_var_ instead -- a
    // DIFFERENT quantity, unconditionally combined here regardless of
    // pose_cov_in_sigma. Unlike refitDebiased()'s fit-BIAS correction
    // (which must not naively combine sensor+pose, see that function),
    // this term is a residual-weighting VARIANCE, where combining is
    // always at least conservative -- "sensor_only" is a genuine
    // ablation, not a correctness fix.
    const M3D residual_cov = (opts_->plane_fit_pose_cov_mode == "sensor_only")
        ? pt.sensor_cov : M3D(pt.sensor_cov + pt.pos_cov);
    plane_var_.noalias() += Jmin * residual_cov * Jmin.transpose();
  }
}

void VoxelPlane::addPoints(const std::vector<PointXYZCov>& points, int total_count,
                           int distinct_frames, bool trust_sensor_noise)
{
  for (const auto& pt : points) {
    const V3D& p = pt.point;
    // Combined sensor+pose covariance -- for plane_var_'s accumulators
    // only (Scov_/V_/W_), which legitimately want the full residual-noise
    // budget, same as the PCA path's Jmin loop above. sensor_cov alone
    // (independent per point) is what M_debiased's own fit-bias
    // correction uses, via Scov_sensor_ below -- see refitDebiased() for
    // why the shared pose component can't be folded in there the same
    // naive way.
    // 2026-08-24 pass4 diagnostic finding: plane_var_ blowups (traces up
    // to ~965,882 seen live) persist even when the denom1/denom2 guard
    // below passes comfortably (e.g. denom1=-0.003, eps=0.00025) -- the
    // real driver is pos_cov's own magnitude, not the eigengap. pos_cov
    // (poseCovAt()'s range^2 lever-arm term) is documented elsewhere in
    // this codebase to "dwarf true sensor noise for far points"
    // (VoxelOpts::pose_cov_in_sigma's docs, which is why THAT flag
    // defaults false) -- every blown-up sample here was F=1, N=4-6: a
    // fresh, barely-initialized voxel whose Scov_/V_/W_ accumulators are
    // raw sums of only a handful of points, not yet averaged down, so one
    // large pos_cov contribution dominates. Respect the same
    // plane_fit_pose_cov_mode ablation used for the PCA path's Jmin loop
    // here too (deviating from the original spec, which scoped it
    // PCA-only, based on this evidence -- see
    // docs/debiased_voxel_plane_fit_2026aug24.md).
    const M3D combined_cov = (opts_->plane_fit_pose_cov_mode == "sensor_only")
        ? pt.sensor_cov : M3D(pt.sensor_cov + pt.pos_cov);
    N_acc_ += 1.0;
    Sp_    += p;
    Spp_.noalias() += p * p.transpose();
    Scov_  += combined_cov;
    // Only fold this point's sensor_cov into the SUBTRACTED correction
    // (Scov_sensor_/sum_sensor_var_) when it comes from a trusted,
    // independent-viewpoint source -- see addPoints()'s trust_sensor_noise
    // docs. Bootstrap points still contribute to Sp_/Spp_/Scov_ above (the
    // fit itself, and plane_var_'s combined-cov accumulators, which stay
    // conservative either way since they only ever ADD uncertainty) --
    // just not to the bias-correction term that gets subtracted.
    if (trust_sensor_noise) {
      Scov_sensor_ += pt.sensor_cov;
      sum_sensor_var_ += pt.sensor_cov.trace() / 3.0;
    }
    for (int a = 0; a < 3; ++a) {
      W_[a] += p(a) * combined_cov;
      for (int b = a; b < 3; ++b) {
        const M3D term = (p(a) * p(b)) * combined_cov;
        V_[a][b] += term;
        if (b != a) V_[b][a] += term;
      }
    }
  }
  points_size_ = (total_count >= 0) ? total_count : (int)N_acc_;
  if (distinct_frames > 0) distinct_frames_ = distinct_frames;
  refitDebiased();
}

void VoxelPlane::refitDebiased()
{
  plane_var_.setZero();
  covariance_.setZero();
  plane_ = {};
  is_plane_ = false;

  const double N = N_acc_;
  if (N < 3) return;

  const V3D mean = Sp_ / N;
  // M_debiased: noise-corrected structure tensor -- see
  // docs/debiased_voxel_plane_fit_2026aug24.md. covariance_ is reused here
  // (same field PCA mode fills) purely for getVizInfo()/diagnostics parity.
  //
  // Sensor noise (independent per point) is subtracted in full, every
  // time -- Scov_sensor_/N. Pose noise (Scov_ - Scov_sensor_, SHARED
  // across every point within one frame, not independent) is subtracted
  // with a (F-1)/F shrinkage factor, F = distinct_frames_: a shared
  // per-frame shift cancels out of a SINGLE frame's own scatter entirely
  // (subtracting it in full there is a pure over-correction -- confirmed
  // numerically, drives eig0 negative), and only becomes a genuine
  // between-frame variance contribution as more independent frames
  // accumulate (F=1 -> 0 correction, F->inf -> full correction). See the
  // doc file's "single-frame over-subtraction" section for the derivation
  // and numeric verification.
  const M3D Scov_pose = Scov_ - Scov_sensor_;
  const double pose_shrink = (distinct_frames_ > 1)
      ? (distinct_frames_ - 1.0) / distinct_frames_ : 0.0;
  covariance_ = Spp_ / N - mean * mean.transpose()
              - Scov_sensor_ / N - pose_shrink * (Scov_pose / N);

  Eigen::SelfAdjointEigenSolver<M3D> solver(covariance_);
  if (solver.info() != Eigen::Success) return;

  eigen_values_ = solver.eigenvalues();
  const M3D evecs = solver.eigenvectors();

  if (eigen_values_(1) < 1e-8 || eigen_values_(2) < 1e-8) return;

  // A population covariance's eigenvalues can never be negative -- a
  // negative SAMPLE eigenvalue here is definitionally sampling noise (the
  // "smallest of three correlated noisy quantities" selection is a biased
  // estimator of the true smallest eigenvalue, especially at low N/F), not
  // evidence the true scatter is negative. Clamping (rather than the old
  // reject-on-any-negative behavior) lets a mildly-negative result --
  // exactly what a genuinely flat surface's corrected estimate looks like
  // some fraction of the time -- correctly read as "flat" instead of being
  // thrown out. Confirmed empirically 2026-08-24: rejected candidates at
  // N=8-24, F=2 had raw (uncorrected) eig0 of 1e-8 to 1e-4 -- genuinely
  // near-flat -- pushed to ~-0.0002 by a sensor-noise correction of
  // comparable magnitude to its own sampling noise, not by a real
  // over-subtraction. See docs/debiased_voxel_plane_fit_2026aug24.md.
  // Guarding against the SEPARATE, larger-magnitude over-subtraction
  // failure mode (correlated noise from non-independent samples) is now
  // handled at the source, via trust_sensor_noise excluding bootstrap
  // points from the correction entirely, rather than by this clamp.
  eigen_values_(0) = std::max(eigen_values_(0), 0.0);

  if (opts_->sensor_noise_floor_eig0) {
    const double sensor_floor = sum_sensor_var_ / N;
    if (opts_->log_debug_en) {
      std::ostringstream dbg;
      dbg << "[debiased] N=" << N << " raw_eig0=" << eigen_values_(0)
          << " sensor_floor=" << sensor_floor << " plane_threshold=" << opts_->plane_threshold;
      debugLogNoiseFloor(dbg.str());
    }
    eigen_values_(0) = std::max(eigen_values_(0), sensor_floor);
    if (eigen_values_(0) >= eigen_values_(1)) return;
  }

  // Non-finite is still a hard reject (NaN/Inf from a degenerate solve) --
  // the negative case is handled by the clamp above now.
  if (!std::isfinite(eigen_values_(0))) return;

  is_plane_ = eigen_values_(0) < opts_->plane_threshold;
  if (!is_plane_) return;

  const double denom1 = eigen_values_(0) - eigen_values_(1);
  const double denom2 = eigen_values_(0) - eigen_values_(2);
  // Near-degenerate-eigengap guard, debiased-mode-only (see update()'s
  // comment on why this was NOT kept for PCA mode -- it measurably changed
  // PCA's own output there, unlike here where it's the confirmed fix for
  // the plane_var_ blowup (traces of 13.998/18.541 seen for a 9-point
  // debiased fit at eig0~2.3e-7): debiasing shrinks eig0 (and often
  // eig1/eig2 together, since the same noise floor gets subtracted from
  // all three), making a small denom1/denom2 far more common than in
  // plain PCA.
  const double eps_denom = std::max(1e-8, 0.1 * opts_->plane_threshold);
  const bool denom_rejected = std::fabs(denom1) < eps_denom || std::fabs(denom2) < eps_denom;
  if (opts_->log_debug_en) {
    std::ostringstream dbg;
    dbg << "[debiased_denom] N=" << N << " F=" << distinct_frames_
        << " eig0=" << eigen_values_(0) << " eig1=" << eigen_values_(1)
        << " eig2=" << eigen_values_(2) << " denom1=" << denom1 << " denom2=" << denom2
        << " eps=" << eps_denom << " rejected=" << (denom_rejected ? 1 : 0);
    debugLogNoiseFloor(dbg.str());
  }
  if (denom_rejected) {
    is_plane_ = false;
    return;
  }

  plane_.normal = evecs.col(0).normalized();
  y_normal_     = evecs.col(1);
  x_normal_     = evecs.col(2);
  plane_.center = mean;
  plane_.d      = -plane_.normal.dot(plane_.center);
  radius_       = (float)std::sqrt(eigen_values_(2));

  const double inv_N2 = 1.0 / (N * N);
  M3D Vw[3][3], Ww[3];
  for (int a = 0; a < 3; ++a) {
    Ww[a] = W_[a] * inv_N2;
    for (int b = 0; b < 3; ++b) Vw[a][b] = V_[a][b] * inv_N2;
  }
  const M3D Scov_w = Scov_ * inv_N2;

  auto U = [&](int a, int b) -> M3D {
    return Vw[a][b] - mean(a) * Ww[b] - mean(b) * Ww[a] + mean(a) * mean(b) * Scov_w;
  };
  auto Wc = [&](int a) -> V3D {
    return (Ww[a] - mean(a) * Scov_w) * plane_.normal;
  };

  const M3D M1 = plane_.normal * y_normal_.transpose() + y_normal_ * plane_.normal.transpose();
  const M3D M2 = plane_.normal * x_normal_.transpose() + x_normal_ * plane_.normal.transpose();

  double s_t1t1 = 0.0, s_t2t2 = 0.0, s_t1t2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const M3D Uab = U(a, b);
      s_t1t1 += M1.col(a).dot(Uab * M1.col(b));
      s_t2t2 += M2.col(a).dot(Uab * M2.col(b));
      s_t1t2 += M1.col(a).dot(Uab * M2.col(b));
    }
  }

  double s_t1d = 0.0, s_t2d = 0.0;
  for (int a = 0; a < 3; ++a) {
    s_t1d += M1.col(a).dot(Wc(a));
    s_t2d += M2.col(a).dot(Wc(a));
  }

  plane_var_(0, 0) = s_t1t1 / (denom1 * denom1);
  plane_var_(1, 1) = s_t2t2 / (denom2 * denom2);
  plane_var_(0, 1) = plane_var_(1, 0) = s_t1t2 / (denom1 * denom2);
  plane_var_(2, 2) = plane_.normal.dot(Scov_w * plane_.normal);
  plane_var_(0, 2) = plane_var_(2, 0) = -s_t1d / denom1;
  plane_var_(1, 2) = plane_var_(2, 1) = -s_t2d / denom2;

  // Sanity ceiling on plane_var_ itself, not just the denom1/denom2 gap
  // that feeds it -- confirmed live (2026-08-24) that denom1/denom2 can sit
  // comfortably outside eps_denom (e.g. denom1=-0.003, eps=0.001) while
  // plane_var_'s trace still reaches ~965,882: a barely-initialized voxel
  // (F=1, N=4-6) has Scov_/V_/W_ built from only a handful of points, so
  // one large pos_cov contribution (documented elsewhere in this codebase
  // to "dwarf true sensor noise for far points", see pose_cov_in_sigma's
  // docs) dominates a still-small denominator's amplification. Rather than
  // trying to characterize every combination of small-N/large-pos_cov/
  // small-denom that can produce this, cap the actual output directly --
  // this is the quantity that widens gate()'s acceptance
  // (sigma_gate_squared = sigma_diag_squared + plane_var_term) and lets
  // garbage-magnitude residuals through, which is the direct divergence
  // mechanism. 1.0 is generous (orientation variance in [theta1,theta2] is
  // radians^2-scale and d's variance is meters^2-scale; both should be
  // small fractions for anything worth trusting as a residual source) --
  // reject (not clamp) so an under-converged voxel just waits for more
  // points/frames rather than silently supplying an overconfident-looking
  // but wrong uncertainty. addPoints() keeps accumulating regardless, so
  // this voxel can still become usable once F/N grow past this point.
  constexpr double kPlaneVarCeiling = 1.0;
  if (!plane_var_.allFinite() || plane_var_.trace() > kPlaneVarCeiling) {
    is_plane_ = false;
    if (opts_->log_debug_en) {
      std::ostringstream dbg;
      dbg << "[debiased_ceiling_reject] N=" << N << " F=" << distinct_frames_
          << " plane_var_trace=" << plane_var_.trace()
          << " denom1=" << denom1 << " denom2=" << denom2;
      debugLogNoiseFloor(dbg.str());
    }
    return;
  }

  if (opts_->log_debug_en) {
    std::ostringstream dbg;
    dbg << "[debiased_fit] N=" << N << " F=" << distinct_frames_
        << " eig0=" << eigen_values_(0) << " eig1=" << eigen_values_(1)
        << " eig2=" << eigen_values_(2) << " plane_var_trace=" << plane_var_.trace()
        << " normal=[" << plane_.normal.transpose() << "]";
    debugLogNoiseFloor(dbg.str());
  }
}

}  // namespace livo_recon
