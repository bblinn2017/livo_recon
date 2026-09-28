# Combined stationary-initialization campaign

Load one fragment after `ntu_viral.yaml`. Use real NTU VIRAL `eee_01`, exactly
30 seconds, VIO disabled, `max_iter` as the only numerical stop, and the
coupled measurement-time estimator with state-aware gating enabled.

The matched baseline is the unmodified `ntu_viral.yaml`. Run the seven cells
in numerical order so effects can be attributed incrementally:

1. moderate position + velocity;
2. tight position + velocity;
3. add gravity-aligned tight tilt while leaving yaw loose;
4. add tight gravity;
5. add tight calibrated biases;
6. reduce yaw uncertainty moderately;
7. make yaw as tight as tilt, reproducing the successful isotropic `all_tight`
   endpoint while showing whether yaw tightening was actually necessary.

For every cell return raw `stationary_iteration.csv`, `gating_all_scans.csv`,
`joint_knot_all_scans.csv`, complete first-frame matrices/covariances/state
chain, resolved configuration, and engagement output. Do not replace these
with an agent-generated scientific report. The report should cover only
implementation/build/test/run status, errors, repairs, and missing evidence.

The analysis must distinguish full SO(3) error, gravity-tilt error, and the
remaining yaw-like component. Record initialized, post-IMU, and post-LIO
attitude covariance eigenvectors/eigenvalues so the gravity-axis construction
can be verified in the non-identity aligned attitude.

After the seven primary cells, run the baseline and cells 03, 05, 06, and 07
for splineless and tail-time controls. These are controls, not candidates:
they determine whether the initialization improvement is estimator-specific.
