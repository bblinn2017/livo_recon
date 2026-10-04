# Per-point pose covariance (F-169): what R65 implements

Planning agent, 2026-09-29. Design note; nothing here has been built or run by me.

## Today (R64 code, ESTABLISHED by reading)

`StateGroup::poseCovAt(p_body)` returns J P J^T with J = [-R [p]x, I] on (rotation, position) and P the state's own (R,P) block. It is evaluated

- in the residual gate (`lio_base.cpp` buildResiduals): state_->cov() during the iterations is the frame's PRIOR (propagated) covariance; used for the statistical gate always, and inside the residual sigma only when `voxel_map/residual/pose_cov_in_sigma` is true;
- at map insertion (`voxelmap.cpp` bucket build): state_->cov() is the POSTERIOR by then; stored as `pos_cov` on every inserted point; read by the plane fit only when `plane_fit_pose_cov_mode` is `combined`.

So every point of a scan gets the same end-of-scan covariance (only the lever arm p_body differs), nothing depends on the point's capture time t_p, and the gate and the map use different covariances (prior vs posterior).

## Proposed model

World position of a point captured at t_p is w = R_p p_b + p_p, the pose at t_p applied to the raw point. The scan's estimate is the pose at the scan end t_e; the pose at t_p is reached from it through the IMU-propagated relative motion. To first order

    Sigma_pt(t_p) = J(p) ( P_e + N(t_p -> t_e) ) J(p)^T

- P_e: the (R,P) block of the end-of-scan covariance (prior at the gate, posterior at insertion in R64; `insertion_source=prior` uses the prior at insertion too, so gate and map see the same basis);
- N(t_p -> t_e): covariance of the noise accrued between the point's time and the scan end, N_k = sum_{j>=k} Phi_{j+1} Q_j Phi_{j+1}^T, computed exactly from the stored per-step IMU transitions F_j and process covariances Q_j by one backward sweep (`relCovBlocksBackward`, header `utils/state/rel_cov.h`); bias-times-dt terms are inside it because the bias states are part of F, Q. Restricted to the (R,P) block, with the same Jacobian J(p).

Keys (defaults reproduce R64 bit for bit): `lio/point_cov/mode` = `end_pose` | `propagated`; `lio/point_cov/insertion_source` = `posterior` | `prior`. Write-only diagnostic: `voxel_map/point_cov_stats_en` (point_cov_stats.csv: per insertion, mean/min/max of trace(pos_cov)).

## Approximations and open points (HYPOTHESIS)

- The step used for a point is the IMU step whose head is at or before t_p, so N is at most one IMU period (about 5 ms) too long: conservative.
- P_e and N(t_p -> t_e) are added as if uncorrelated. P_e already contains the propagated start-of-frame uncertainty (Bryce's item (a)); the cross-covariance between the end pose and the relative motion is neglected.
- Where the model matters: the statistical gate always; the residual sigma with `pose_cov_in_sigma`; the plane fit with `combined`. With the baseline settings (`sensor_only` fit, sigma without pose covariance) only the gate changes.
