# R44 P0 and motion-Q controls

Load one P0 file after `ntu_viral.yaml` and `coupled_joint_knots.yaml`:

- `p0_configured_control.yaml`: configured P0 + fixed Q baseline.
- `p0_candidate_fixed_q.yaml`: selected R43 candidate + fixed Q.

For phase 2, load either motion-Q file after `p0_candidate_fixed_q.yaml`.
The initial scales are experiment seeds, not calibrated VN-100 constants.
Dynamic energy is noise-debiased before scale and cap are applied, so the
calibrated stationary floor is not counted once as baseline noise and again
as motion. With `beta=0`, rectification after per-sample floor subtraction can
still have positive statistical bias; it is a requested control, not an
assumption that filtering is unnecessary. The real-data sweep must vary both
scales and beta; conclusions about P0 remain conditional on Q and Gamma_L.
