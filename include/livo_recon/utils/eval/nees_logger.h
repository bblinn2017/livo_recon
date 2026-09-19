#pragma once

#include "livo_recon/utils/algo/math.h"
#include <Eigen/Core>
#include <vector>

namespace livo_recon
{

// CQ-60 item 5: the ONE shared per-dof whitened-error (NEES) computation
// and logger, used by all three CQ-60 tiers -- built once, here, rather
// than reimplemented per call site. Pure math (no I/O); the rotation error
// uses this codebase's own Log() (SO(3) log map, math.h), the position
// error is linear. P is the 6x6 pose covariance in [rotation(3);
// position(3)] order -- StateGroup::idxR()=0/idxP()=3's own layout, so a
// caller can build P directly from state_->cov()'s R,P sub-blocks (see
// lio_coupled.cpp's Prr/Ppp/Prp fields for the coupled path's own version
// of exactly this).
struct NeesResult
{
  double nees = -1.0;  // 6-dof Mahalanobis distance -- averages 6 when P is calibrated.
  // Per-DOF whitened error, squared -- averages 1 in each of the 6
  // components when P is calibrated. Order is the eigenbasis of P, NOT
  // [roll,pitch,yaw,x,y,z] directly (P's off-diagonal R-P coupling means
  // the two rarely coincide) -- see whitening_axes for that basis itself,
  // if a caller wants to relate a component back to a physical direction.
  Eigen::Matrix<double, 6, 1> per_dof_whitened_sq = Eigen::Matrix<double, 6, 1>::Constant(-1.0);
  Eigen::Matrix<double, 6, 6> whitening_axes = Eigen::Matrix<double, 6, 6>::Zero();
  bool valid = false;  // false if P was not invertible (rule 58: report, don't substitute)
  double min_eig = -1.0, max_eig = -1.0;  // always populated, even when !valid
};

// e = [Log(R_gt^T * R_est); p_est - p_gt], whitened by P (see NeesResult's
// own doc comment for P's required layout/order).
NeesResult computeNeesPerDof(const M3D& R_est, const V3D& p_est,
                              const M3D& R_gt, const V3D& p_gt,
                              const Eigen::Matrix<double, 6, 6>& P);

// CQ-60 Tier 1: the stationary-window's "truth" (the window's own mean
// pose) is only knowable once the window has been fully observed -- this
// buffers (scan_id, t_abs, R, p, P) for the first `window_size` calls,
// then on the call that fills it, computes the mean rotation (SVD-
// orthogonalized average, appropriate for a small-angle stationary window)
// and mean position, and replays every buffered entry through
// computeNeesPerDof()/logNeesPerDof() under `channel`. One-shot: inert on
// every call after the window fills (Tier 1 measures the bag's own
// stationary PREFIX once, not a repeating window). Shared by both the
// coupled and decoupled paths -- each owns its own instance (this class
// holds per-instance state, not global/static).
class Tier1NeesBuffer
{
public:
  explicit Tier1NeesBuffer(int window_size) : window_size_(window_size) {}

  // Call once per scan with that scan's own (scan_id, t_abs, R, p, P).
  // window_size <= 0 disables this (every call is a no-op) -- the caller's
  // own eval/nees_per_dof_en gate should already prevent calls entirely in
  // that case, this is a second, cheap line of defense.
  void addScan(const char* channel, int scan_id, double t_abs,
               const M3D& R, const V3D& p, const Eigen::Matrix<double, 6, 6>& P);

private:
  int window_size_;
  bool done_ = false;
  struct Entry { int scan_id; double t_abs; M3D R; V3D p; Eigen::Matrix<double, 6, 6> P; };
  std::vector<Entry> buf_;
};

// Appends one row to nees_per_dof.txt (report-only, PersistentLogStream --
// redirected through outputs/debug_log_dir like every other diagnostic in
// this codebase). `channel` distinguishes which tier/path/config produced
// the row (e.g. "tier1_decoupled", "tier2_coupled_n13") since multiple
// sources can share one log file within a run. Callers gate the call to
// this function on eval/nees_per_dof_en themselves (this function does not
// re-check it) -- matches every other opts_-gated diagnostic in this
// codebase's own convention.
void logNeesPerDof(const char* channel, int scan_id, double t_abs, const NeesResult& r);

}  // namespace livo_recon
