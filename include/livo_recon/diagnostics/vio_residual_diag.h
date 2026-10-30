#pragma once
// TEMPORARY diagnostic (2026-10-27, mid360 lio/sequential/combined VIO-
// error comparison investigation). Logs ONE uniform "post-refinement avg
// VIO pixel-reprojection error" row per frame, for ALL THREE modes
// (lio_only/sequential/combined) alike -- re-evaluates
// VioProc::accumulateForCombined() read-only (no state mutation) against
// each frame's FINAL committed pose, after estimateState() has already
// decided that pose for this frame via whichever path (combined joint
// solve, sequential LIO+VIO, or LIO-only). This makes lio_only mode
// produce a comparable number even though it never runs any VIO
// correction itself -- see diag/force_vio_residual_log's doc comment in
// livo_recon_node.h for how the tracker is forced to start so
// mg.tracked_frame is populated in that mode too.
//
// Delete this file + its call site (livo_recon_node.cpp's estimateState())
// + the diag/force_vio_residual_log option once the investigation is
// done -- this is not meant to be a permanent feature.
#include <string>

namespace livo_recon
{

void writeVioResidualDiagRow(int scan_id, double t_abs, bool ok, float avg_error,
                             float avg_abs_pixel_residual, int n_valid);

}  // namespace livo_recon
