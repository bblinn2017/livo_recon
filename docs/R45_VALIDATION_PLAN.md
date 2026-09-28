# R45 real-data validation plan

The coding agent compiles and runs these experiments. It returns raw data and
an implementation/build/run-status report only; interpretation is done by the
analysis agent. Use eee_01, 30 s runs, and max-iteration-only stopping.

## Phase 1/2 matrix

Keep the selected calibration-derived P0 and configured/fixed-Q control.
For isotropic noise-debiased Q, run the Cartesian grid:

- `acc_scale`: 0.0, 0.3, 0.6, 1.0
- `gyro_scale`: 20, 30, 45, 60
- `beta`: 0.0, 0.5, 0.9

Run axis-aware controls at the best isotropic settings and at the fixed-Q
control. Do not infer the scales from stationary calibration: stationary data
identifies the floor, not dynamic scale error.

## Phase 3 base matrix

For both coupled residual-time modes (`measurement`, `tail`), compare:

- independent residual covariance;
- matched-plane Woodbury with rho 0.25, 0.5, and 1.0;
- gating-state uncertainty off and on for each cell.

Also run splineless and decoupled with independent residual covariance as
reference controls using the same duration, initialization, and stop policy.
Their state layouts differ, so do not apply the joint-knot Gamma_L port to
them merely to make the labels match.

First run the dense-covariance algebra test. Behavioral claims use real data.
The beta=0 Q cells remain diagnostic controls; select a production candidate
only from filtered cells because per-sample floor subtraction plus clipping
has a positive rectification bias at rest.

This campaign validates only exact per-`VoxelPlane` correlated information
against independent residuals. Do not combine it with cross-voxel surface
clustering or RBF covariance. The broader model is a later stage, conditional
on this experiment showing improved consistency without unacceptable
stationary-trajectory degradation.

## Compact return schema

Return only the following analysis-sufficient tables (compressed CSV or
Parquet), plus the short status report. Do not return matrix dumps, pointwise
residual files, full knot-state files, or unrelated logs.

`trajectory.csv`, one row for post-IMU and post-LIO per scan:

`run_id,scan_id,t_abs,phase,residual_count,completed_iterations,ep_x,ep_y,ep_z,ep_norm,ev_x,ev_y,ev_z,ev_norm,tilt_x,tilt_y,tilt_z,tilt_norm,yaw_error,er_norm`

`covariance.csv`, post-LIO only:

`run_id,scan_id,t_abs,Prr_trace,Ppp_trace,Pvv_trace,Prp_fro,Prv_fro,Ppv_fro,tilt_eig1,tilt_eig2,yaw_variance,Prpv_min_eig,Prpv_max_eig,eps_R,eps_p,eps_v,eps_tilt,eps_yaw,eps_RPV`

`motion_q.csv`, one row per scan:

`run_id,scan_id,t_abs,model,beta,acc_scale,gyro_scale,mean_debiased_acc_energy_x,mean_debiased_acc_energy_y,mean_debiased_acc_energy_z,mean_debiased_gyr_energy_x,mean_debiased_gyr_energy_y,mean_debiased_gyr_energy_z,mean_applied_acc_variance_x,mean_applied_acc_variance_y,mean_applied_acc_variance_z,mean_applied_gyr_variance_x,mean_applied_gyr_variance_y,mean_applied_gyr_variance_z,acc_cap_fraction,gyr_cap_fraction`

`lidar_information.csv`, one row per iteration:

`run_id,scan_id,iteration,t_abs,residual_time,gating_state_uncertainty,lidar_information_mode,residual_count,gamma_trace,gamma_frobenius,b_norm,redund_groups_seen,redund_groups,redund_groups_degenerate_pv,redund_groups_degenerate_var,redund_n_raw,redund_info_ratio,information_increase_groups,naive_info_gain,woodbury_info_gain`

The returned status report states build/test/run failures, missing cells,
configuration actually engaged, row counts, and any source changes required
to complete the run. It does not analyze estimator performance.
