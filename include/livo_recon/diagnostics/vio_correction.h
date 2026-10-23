#pragma once
// Per-frame diagnostic: how much does VIO move the pose, on top of
// whatever LIO already produced this frame?
//
// Sequential mode (vio.enable=true, combined.enable=false): the state right
// after LIO and right after VIO are two genuinely distinct checkpoints
// (mg.pos_after_lio/rot_after_lio vs mg.pos_after_vio/rot_after_vio in
// livo_recon_node.cpp's estimateState()) -- writeVioCorrectionRow() logs
// their delta directly (translation vector + norm, rotation angle in
// degrees) to vio_correction.csv.
//
// Combined mode: LIO and VIO residuals are fused into ONE joint update per
// iteration, so there is no separate "post-LIO, pre-VIO" checkpoint to
// diff against (mg.pos_after_lio == mg.pos_after_vio always in this mode,
// see CombinedProc::processCombined()'s own comment). The closest
// available per-frame signal is how much visual residual data actually
// participated in that joint solve -- writeCombinedVioContributionRow()
// logs avg_err_vio/n_meas_vio/vio_ever_contributed to
// combined_vio_contribution.csv instead. This is NOT the same quantity as
// vio_correction.csv's pose delta -- don't compare the two files directly.
//
// Both are write-only (numerics-inert).
#include <Eigen/Dense>

namespace livo_recon
{

void writeVioCorrectionRow(int scan_id, double t_abs,
                           const Eigen::Vector3d& pos_after_lio, const Eigen::Matrix3d& rot_after_lio,
                           const Eigen::Vector3d& pos_after_vio, const Eigen::Matrix3d& rot_after_vio);

void writeCombinedVioContributionRow(int scan_id, double t_abs, float avg_err_vio, int n_meas_vio,
                                     bool vio_ever_contributed);

}  // namespace livo_recon
