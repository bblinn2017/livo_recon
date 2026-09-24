#pragma once
// History (2-17): see docs/livo_recon_changelog.md#include-livo_recon-utils-log-consistency_log.h-2
#include <string>

namespace livo_recon {

// channel is "lio" or "vio"; it becomes a column so a combined run's rows
// stay separable after the fact.
void logConsistencyNll(const char* channel, double t_abs, double nll, int n_residuals,
                       double pivot_ratio, double kalman_gain_norm);
void logConsistencyScan(const char* channel, int scan_id, double t, double dt,
                        double trP_pos, double trP_vel, double trP_att,
                        double omega_norm, double acc_norm,
                        double ba_x, double ba_y, double ba_z,
                        double bg_x, double bg_y, double bg_z);
// One row per visual residual, mirroring corr.csv's nu/S so the two channels
// are scored by the same statistic rather than by two different ones.
// VioAccumulator has no per-frame scan_id counter the way VoxelMap does (see
// LioProc's own frame_idx_) -- t_abs/iter identify the row instead; t_abs is
// shared with nll.txt's own t column so the two can be joined on it.
void logConsistencyVioCorr(double t_abs, int iter, double nu, double S, const char* residual_mode);

}  // namespace livo_recon
