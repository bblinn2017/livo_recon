#pragma once
// TEMPORARY diagnostic (2026-10-27, "VIO ask vs. got" investigation).
// Logs, for the FIRST EKF iteration of a frame (iter==0, before any state
// mutation that frame), what each measurement source's evidence ALONE
// implies the correction should be (the "ask": -HtH^-1 * Htz, the
// unweighted/prior-free normal-equations solution -- NOT the same as the
// Kalman-gain-weighted term applyMeanUpdate() actually uses, which is
// damped by the prior's confidence), and, for combined mode, a one-
// iteration counterfactual isolating VIO's marginal contribution (joint
// LIO+VIO vs. LIO-alone, both solved via a scratch state copy so neither
// touches the real committed state).
//
// Sequential mode's "got" is NOT logged here -- it's already directly
// observable as post_vio - post_lio in the existing state_trace.csv, no
// extra computation needed.
//
// Delete this file + its call sites (vio_processing.cpp, combined_
// processing.cpp) once the investigation is done.
#include <Eigen/Dense>

namespace livo_recon
{

// mode: "sequential" or "combined".
void writeCorrectionBreakdownRow(
    int scan_id, double t_abs, const std::string& mode,
    const Eigen::Vector3d& ask_vio_dtheta, const Eigen::Vector3d& ask_vio_dt, int n_meas_vio,
    bool have_lio_ask, const Eigen::Vector3d& ask_lio_dtheta, const Eigen::Vector3d& ask_lio_dt, int n_meas_lio,
    bool have_counterfactual,
    const Eigen::Vector3d& counterfactual_lio_only_dtheta, const Eigen::Vector3d& counterfactual_lio_only_dt,
    const Eigen::Vector3d& counterfactual_joint_dtheta, const Eigen::Vector3d& counterfactual_joint_dt,
    // Rotation/position block trace of each source's own HtH (6x6, block(0,0)=rotation,
    // block(3,3)=position -- see StateGroup::idxR()/idxP()) at the SAME ask-computation
    // point as the vectors above: a measure of how much "information"/precision each
    // source's unweighted normal equations carry in that block, independent of direction
    // -- lets LIO's vs. VIO's rotation-constraint strength be compared directly. 0.0 when
    // that source's ask wasn't computed this row (n_meas==0 or have_*_ask/have_counterfactual
    // false for that side).
    double tr_HtH_rot_lio = 0.0, double tr_HtH_pos_lio = 0.0,
    double tr_HtH_rot_vio = 0.0, double tr_HtH_pos_vio = 0.0);

}  // namespace livo_recon
