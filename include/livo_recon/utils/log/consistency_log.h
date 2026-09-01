#pragma once
// G6 blocker (2026-09-01). nll.txt and scan.csv are written by two static
// helpers in the anonymous namespace of lio_processing.cpp, so they are
// reachable ONLY from LioProc::processLIO(). CombinedProc::processCombined()
// never calls processLIO() except on the VIO-rejection rollback path, and
// never references log_nll_en or log_consistency_scan_en at all.
//
// Consequence, stated plainly: with combined/enable true, this project has NO
// consistency instrument. Every NIS number in the register is LIO-only. The
// question "are the LiDAR and visual channels calibrated consistently
// relative to each other" -- which is the measured argument for or against
// collapsing the two filters -- has not been askable, and unifying the
// filters cannot be evaluated until it is.
//
// This header hoists the two writers out of that anonymous namespace so both
// processors can call them. No behaviour change on the LIO path: same
// filenames, same columns, same first-call truncate semantics.
#include <string>

namespace livo_recon {

// channel is "lio" or "vio"; it becomes a column so a combined run's rows
// stay separable after the fact.
void logConsistencyNll(const char* channel, double t_abs, double nll, int n_residuals,
                       double pivot_ratio, double kalman_gain_norm);
void logConsistencyScan(const char* channel, int scan_id, double t, double dt,
                        double trP_pos, double trP_vel, double trP_att,
                        double omega_norm, double acc_norm);
// One row per visual residual, mirroring corr.csv's nu/S so the two channels
// are scored by the same statistic rather than by two different ones.
// VioAccumulator has no per-frame scan_id counter the way VoxelMap does (see
// LioProc's own frame_idx_) -- t_abs/iter identify the row instead; t_abs is
// shared with nll.txt's own t column so the two can be joined on it.
void logConsistencyVioCorr(double t_abs, int iter, double nu, double S, const char* residual_mode);

}  // namespace livo_recon
