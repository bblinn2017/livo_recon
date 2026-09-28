# Frame-1 initial-covariance ablation

Load exactly one fragment after `config/ntu_viral.yaml`. The unchanged NTU
VIRAL configuration is the baseline (`rot_tilt=rot_yaw=pos=vel=1e-2`,
`bg=ba=gravity=1e-5`). These keys initialize the state covariance; they do
not change accelerometer/gyroscope measurement noise or bias random-walk
process noise.

Use real `eee_01` data and a 30-second duration for every cell. Run all four
architectures (splineless, decoupled spline, coupled measurement time, and
coupled tail time), using `max_iter` as the only enabled iteration stop.

Run order:

1. Baseline plus `pos_1e-4`, `pos_1e-6`, and `pos_1e-8`.
2. Baseline plus each one-factor fragment for rotation, velocity, biases, and
   gravity. Rotation and velocity each have moderate and strong settings.
3. `all_tight` only as an interaction check; do not use it to attribute an
   effect to any individual state block.

For frame 1 report the exact pre-IMU, post-IMU, and post-LIO covariance
blocks; `lidar_rhs`, `prior_rhs`, their corresponding deltas, total requested
delta, realized delta, residual count/mean, and position/attitude/velocity
distance to the fixed stationary reference after every iteration. Also
report the 30-second stationary statistics so a smaller first-frame update
is not mistaken for improved stability.

Attitude must be reported two ways: the complete SO(3) reference error and
the gravity-tilt error (angle between the reference and current body-frame
gravity directions). Gravity alignment constrains tilt but does not observe
yaw, so do not describe all three attitude degrees of freedom as known from
gravity alone. Because the platform is stationary, also report absolute
speed and the velocity-vector error relative to the zero-velocity reference.

Do not use zero covariance: the information-form prior must remain positive
definite. Do not use synthetic trajectory data for acceptance; synthetic
checks remain limited to isolated covariance/information algebra.
