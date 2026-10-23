#pragma once

#include "livo_recon/utils/algo/math.h"

namespace livo_recon {

// log/state_correction_trace (default false): per-frame position/velocity/
// attitude at each correction stage -- before IMU propagation, after IMU
// propagation (pure prediction), after LIO, and after VIO/combined (the
// final committed state). Unlike log/state_trace/enable (default true),
// which only ever logs pre_imu/post_imu/post_lio and nothing for
// combined mode, this captures the SAME four checkpoints (mg.pos_before_imu/
// pos_after_imu/pos_after_lio/pos_after_vio and rot_/vel_ counterparts) in
// both sequential and combined mode, since those fields are populated
// identically by both branches of LivoReconNode::estimateState().
// No vel_after_vio -- VIO has no direct velocity observability, so
// MeasureGroup never tracks a post-VIO velocity checkpoint.
void writeStateCorrectionTraceRow(
    int scan_id, double t_abs,
    const V3D& pos_before_imu, const M3D& rot_before_imu, const V3D& vel_before_imu,
    const V3D& pos_after_imu, const M3D& rot_after_imu, const V3D& vel_after_imu,
    const V3D& pos_after_lio, const M3D& rot_after_lio, const V3D& vel_after_lio,
    const V3D& pos_after_vio, const M3D& rot_after_vio);

}  // namespace livo_recon
