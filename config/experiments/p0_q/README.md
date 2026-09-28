# R44 P0 and motion-Q controls

Load one P0 file after `ntu_viral.yaml` and `coupled_joint_knots.yaml`:

- `p0_configured_control.yaml`: configured P0 + fixed Q baseline.
- `p0_candidate_fixed_q.yaml`: selected R43 candidate + fixed Q.

For phase 2, load either motion-Q file after `p0_candidate_fixed_q.yaml`.
The initial scales are experiment seeds, not calibrated VN-100 constants.
The real-data sweep must vary both scales and beta; conclusions about P0
remain conditional on Q and the still-uncorrected correlated LiDAR model.
