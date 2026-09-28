# R44 validation plan: stabilized P0, selected candidate, and motion Q

All behavioral runs use real NTU VIRAL `eee_01` data for exactly 30 seconds.
Do not use movement-time ATE: ground truth is absent for most of this stationary
window. Keep `max_iterations` as the only enabled iterative stop.

## Phase 1 runs

1. `configured_fixed`: `p0_configured_control.yaml`.
2. `candidate_fixed`: `p0_candidate_fixed_q.yaml`.
3. `zero_ambiguity_fixed`: candidate, except ambiguity std is zero. This is a
   covariance-validity regression only; it must reach all scans and report a
   post-floor minimum eigenvalue at or above the numerical floor.

For runs 1 and 2, retain the full first-post-calibration-frame solve evidence.
Run 3 needs `calibration_p0.txt`, completion status, and compact scan-phase data;
do not return its large first-frame matrices unless it fails.

## Phase 2 runs

Use the selected candidate P0 and gating disabled:

1. fixed-Q control;
2. isotropic, beta=0, acc/gyro scales 1/1;
3. axis-aware, beta=0, acc/gyro scales 1/1;
4. isotropic and axis-aware with beta=0.9, scales 1/1.

This is an engagement/behavior validation, not final scale tuning. The scale
parameters are not identifiable from a stationary calibration window and no
claim should label 1.0 a calibrated VN-100 value.

## Exact compact return schema

Return one ZIP under 30 MB. Construct the following compact CSVs directly;
do not include the wider source CSV when columns below are sufficient.

### `scan_phase_compact.csv`

All scans, both `post_imu` and `post_lio` rows:

`run_id,scan_id,t_abs,phase,p0_mode,init_pos,init_vel,init_rot_tilt,init_rot_yaw,init_bg,init_ba,residual_count,completed_iterations,er_x,er_y,er_z,er_norm,ep_x,ep_y,ep_z,ep_norm,ev_x,ev_y,ev_z,ev_norm,tilt_x,tilt_y,tilt_z,tilt_norm,yaw_error,Prr_trace,Prr_eig1,Prr_eig2,Prr_eig3,Ppp_trace,Ppp_eig1,Ppp_eig2,Ppp_eig3,Pvv_trace,Pvv_eig1,Pvv_eig2,Pvv_eig3,Prp_fro,Prv_fro,Ppv_fro,tilt2d_eig1,tilt2d_eig2,yaw_variance,Prpv_min_eig,Prpv_max_eig,eps_R,eps_p,eps_v,eps_tilt,eps_yaw,eps_RPV,chi2_R_95,chi2_p_95,chi2_v_95,chi2_tilt_95,chi2_yaw_95,chi2_RPV_95,delta_er_norm,delta_ep_norm,delta_ev_norm,error_and_covariance_both_contracted,error_increased_while_covariance_contracted`

Explicitly omit configured-input columns duplicated by resolved configs, the
68/99% indicator columns, condition number, validity columns, covariance-trace
deltas, and LDLT pivot from the returned scan table.

### `gating_compact.csv`

All scans and iterations:

`run_id,scan_id,iteration,t_abs,gating_state_uncertainty,input_points,statistical_candidates,statistical_gate_rejections,coverage_misses,mismatch_misses,accepted_count,accepted_index_hash,solve_input_hash,mean_gating_cov_trace,max_gating_cov_trace,solve_uses_gating_covariance`

Omit accepted-index sum and XOR; the stronger hashes are retained.

### `motion_q_scan.csv`

Combine all per-run files and prepend `run_id`; otherwise retain every source
column unchanged. It is already one compact row per scan.

### `motion_q_first_frame.csv`

Combine all per-run files and prepend `run_id`; otherwise retain every source
column unchanged for every phase-2 run. It contains only the first
post-calibration propagation at IMU-sample resolution.

### `first_frame_corrections.csv`

For `configured_fixed`, `candidate_fixed`, fixed-Q phase-2 control,
isotropic beta=0, and axis-aware beta=0, retain:

`run_id,scan_id,iteration,block,knot_index,immutable_head,requested_x,requested_y,requested_z,lidar_x,lidar_y,lidar_z,prior_x,prior_y,prior_z,realized_x,realized_y,realized_z`

### Other retained evidence

- full `calibration_p0.txt` for every run;
- resolved config for every run;
- odometry columns `run_id,t_bag,px,py,pz,qx,qy,qz,qw` for every run;
- full first-frame matrices, covariances, and state-chain only for
  `configured_fixed`, `candidate_fixed`, isotropic beta=0, and axis-aware
  beta=0, plus any failed/non-PD/extreme run;
- build and test logs, final source, cumulative patch, raw manifest, and the
  implementation-only coding-agent report required by the standing rules.

Do not return the original wide all-scan CSVs after producing and hashing the
compact extracts. The manifest must state each source selection predicate and
the exact omitted columns.
