#pragma once

#include <vector>

#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/cuda/lio_cuda.h"

// LIO's point-to-plane residual -> HtH/Htz accumulation, lifted out of
// LioProc (2026-08-14) so both LioProc's own sequential solve AND
// CombinedProc's joint LIO+VIO solve can call the exact same math without
// CombinedProc depending on LioProc's private internals. Deliberately only
// depends on Residual/EkfUpdate/LioCudaBuffers -- not on LioProc, NodeContext,
// or anything ROS -- so touching this file only rebuilds its few actual
// callers, not the whole package (see node_context.h's doc comment for the
// broader rebuild-cascade context this follows).
namespace livo_recon
{

// CPU/OMP accumulation: fills out.HtH/Htz/n_meas from the weighted normal
// equations (w = 1/sigma_squared) over every residual. Does NOT call
// out.applyMeanUpdate()/applyCovarianceUpdate() -- callers own the EKF
// solve step (LioProc's own sequential path calls applyMeanUpdate right
// after; CombinedProc sums this with VIO's own accumulation first).
// Resets `out` (HtH/Htz/n_meas only -- dtheta/dt are untouched, they're
// only ever written by applyMeanUpdate) before accumulating. Fixed 6-dim
// (R,P) system.
//
// 2026-08-24: REMOVED the `wide` (12-dim R,P,V,W) accumulation mode --
// existed solely to support the removed iterative-deskew mechanism's
// wide_jacobian_vw option (confirmed regressed, never usable). See
// docs/removed_livo_recon_spline_deskew_2026aug24.md.
void accumulateLioResiduals(const std::vector<Residual>& residuals, EkfUpdate& out);

// CUDA accumulation: same contract, using the H^T*W*H reduction kernel via
// runLioAccumulation(). `buf` is caller-owned (LioProc keeps one persistent
// LioCudaBuffers across frames to amortize allocation).
void accumulateLioResidualsCuda(const std::vector<Residual>& residuals, EkfUpdate& out,
                                 LioCudaBuffers& buf);

}  // namespace livo_recon
