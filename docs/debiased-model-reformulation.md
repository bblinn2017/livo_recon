# The debiased plane fit, reformulated (planning agent, 2026-09-29)

Status of everything below: **derivation and reading of code, not measurement.** The R64 arms `P_*` are the tests. Nothing here has been run.

## 1. Are we adding pose covariance, and are we subtracting it?

Two config values decide it (`voxel_map/plane/plane_fit_pose_cov_mode`):

- `sensor_only`: the per-point covariance used in the fit is the sensor covariance $\Sigma_i^{s}$ alone. The pose part of the correction, $\bar\Sigma^{p}=\bar\Sigma^{all}-\bar\Sigma^{s}$, is then exactly zero, so the term $\frac{F-1}{F}\bar\Sigma^{p}$ in the code is inert.
- `combined` (the compiled default): $\Sigma_i = \Sigma_i^{s}+\Sigma_i^{p}$ with $\Sigma_i^{p}$ the pose covariance mapped to the point (a rotation covariance times range squared plus a translation covariance, identical for every point of one frame). It is added to the accumulators and a shrunk copy is subtracted.

**Every R63 arm, including all debiased ones, was configured `sensor_only`**: the R63 manifest sets `voxel_map.plane.plane_fit_pose_cov_mode: sensor_only` on every job, and the effective-config listings I hold (six arms, none of them debiased) show it applied. The debiased arms' own effective-config listings were not returned, so for them this rests on the manifest, not on a listing; R64 returns one per family. So in R63 pose covariance was neither added to the debiased fit nor subtracted from it. Removing the pose subtraction cannot change the R63 debiased result. It can only matter under `combined`, which R64 tests directly (`P_debiased_full_eigengap_pose_combined_60s` against `P_debiased_full_eigengap_60s`).

## 2. Does it make sense to subtract pose covariance at all?

Write point $i$ of frame $f$ as $p_i = q_i + s_i + e_f(q_i)$, where $q_i$ is the true surface point, $s_i$ is independent sensor noise (covariance $\Sigma_i^{s}$), and $e_f(q)=\delta_f+\theta_f\times q$ is the pose error of frame $f$ seen at point $q$ ($\delta_f$ translation error, $\theta_f$ rotation error). The scatter matrix the fit sees, with $\bar p$ the voxel mean, is $S=\frac1N\sum_i (p_i-\bar p)(p_i-\bar p)^T$.

1. **Within one frame** the translation error $\delta_f$ is a common shift and cancels from $S$ entirely. The rotation error does not cancel: $\theta_f\times q$ varies linearly with position, so it shears the frame's cloud and adds a term $\propto [q]_\times\Sigma_\theta[q]_\times^T$ (position-dependent, not a constant matrix). The code's single-frame rule (correction $=0$ at $F=1$) is therefore right for translation and wrong for rotation.
2. **Across frames** what enters $S$ is the *difference* of the pose errors of the frames that contributed to the voxel, $e_f-e_g$, not the absolute error. The filter's error is strongly correlated from frame to frame (a stationary sensor keeps the same offset; the state covariance $P$ describes uncertainty about that shared offset). The relevant variance is $\operatorname{Var}(e_f-e_g)=P_{ff}+P_{gg}-2P_{fg}$, which can be orders of magnitude smaller than $P_{ff}$. The code has only the current frame's $P_{ff}$ (through `pos_cov`) and treats each frame's error as independent, i.e. it subtracts the *absolute* covariance as if it were between-frame scatter.
3. **A map built in one frame of reference is consistent with itself** even if that frame is offset from truth. What corrupts a plane is inconsistency between the frames that built it, not the common offset.

Conclusion (derivation, to be tested): subtracting absolute pose covariance from the scatter is conceptually the wrong quantity, over-corrects by the ratio of absolute to relative pose variance, and can drive $\lambda_0$ (and the smaller in-plane eigenvalues) negative. Sensor noise is the one term that is independent per point and known per point, so it is the one term the debiasing is entitled to subtract. Recommendation: keep `sensor_only` as the meaning of "debiased", and treat `combined` as an ablation only.

## 3. The model, stated once

The fit estimates the covariance of the noise-free surface points from the observed scatter,

$$\hat\Sigma_{\text{true}} = S - \bar\Sigma^{s},\qquad \bar\Sigma^{s}=\frac1N\sum_i\Sigma^{s}_i ,$$

then takes the normal $n$ as the eigenvector of the smallest eigenvalue, and accepts the plane when the smallest eigenvalue $\lambda_0<\tau$ (`plane_threshold`, 0.01 m$^2$).

Two things about $\bar\Sigma^{s}$ make the eigendecomposition step fragile even though the expectation is right:

- **It is anisotropic and range-dependent.** For a beam at range $r$ the covariance is about $\sigma_r^2 uu^T + r^2\sigma_a^2(I-uu^T)$ ($u$ the beam direction, $\sigma_r$ range noise, $\sigma_a$ angular noise). Subtracting it from all three eigenvalues reduces the *in-plane* eigenvalues $\lambda_1,\lambda_2$ by the tangential noise $\approx r^2\sigma_a^2$, which grows with range. A voxel with a narrow in-plane extent (a sliver: forward-looking scan rings on a wall give a small $\lambda_1$) can end with $\lambda_1$ or $\lambda_2$ at or below zero. The code then rejects it (`eig1 < 1e-8`), or the eigengap $\lambda_0-\lambda_m$ falls under the guard ($0.1\tau=10^{-3}$) and it is rejected there, or the eigenvector ordering flips. The raw-PCA path never subtracts anything from $\lambda_1,\lambda_2$ and does not reject these voxels.
- **Correlated noise within a viewpoint.** The repository's own note on `sensor_noise_floor_eig0` records that single-viewpoint points have correlated noise, so $\bar\Sigma^{s}$ (an independent-noise sum) over-subtracts on early, few-frame voxels.

First-order alternative, tested as `debias_mode = lambda0_only`: take $n$ and $\lambda_1,\lambda_2$ from the raw scatter $S$ and correct only the out-of-plane eigenvalue,

$$\lambda_0^{\text{deb}}=\max\!\big(0,\ \lambda_0(S)-n^T\bar\Sigma^{s}n\big).$$

Only the along-normal component of the noise reaches a point-to-plane residual, so this subtracts exactly what the acceptance test is about and nothing that can flip in-plane eigenvalues. `debias_mode = none` is the control: raw scatter, no correction, everything else in the debiased code path unchanged.

## 4. The lifecycle differs from PCA, independent of the arithmetic

Reading `VoxelNode::insertPoints()`:

| | pca path | debiased path |
|---|---|---|
| refit | every `min_update_points` (5) new points, until the voxel locks | on **every** insertion, forever |
| a plane that later fails the test | cannot happen once locked (`CONVERGED` voxels ignore later points) | plane revoked, voxel marked `PARENT`, `plane_retired_ = true` (never used for residuals again), children receive **only the current batch** |
| at the deepest layer | locks `DISABLED` | locks `DISABLED` |
| bootstrap frame 0 | points inserted once | same, but sensor covariance of frame 0 excluded from the correction |

So a debiased voxel that fails a single refit is lost together with its history, whereas a PCA voxel that passed once is kept. This alone can produce fewer residuals independent of any arithmetic (R63: 9,975 vs 12,490 residuals per frame, constant from the start).

R64 measures which mechanism it is, per frame from initialization, for both paths: `plane_fit_reasons.csv` (counts of accept / each reject reason / retire / subdivide / disabled per frame per path) and `plane_fit_trace.csv` (one row per fit attempt for frames 0..12: $N$, $F$, raw eigenvalues, $n^T\bar\Sigma^{s}n$, the eigenvalues the test saw, the threshold used, reason). Arms: `P_debiased_full` (R63 arithmetic), `P_debiased_none` (lifecycle only), `P_debiased_lambda0_only` (arithmetic change only), `P_debiased_*_hysteresis` (revoke less readily), `P_*_min_frames_to_init`, `P_*_information`, `P_*_pose_combined`, `P_*_floor_eig0`, full-bag versions, and identity controls.
