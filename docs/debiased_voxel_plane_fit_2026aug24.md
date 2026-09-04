# Debiased (covariance-aware) voxel plane fit — 2026-08-24

## Motivation

Today's PCA plane fit (`VoxelOpts::plane_fit_mode == "pca"`, unchanged
default) uses unweighted PCA on raw point positions to get the normal, and
only afterward propagates each point's own covariance (`PointXYZCov::cov`)
into `plane_var_` via a per-point Jacobian sandwich (`Jmin * cov *
Jmin^T`). Point covariance never informs the fit itself, and that Jacobian
loop is O(N) over every historically accumulated raw point on every refit.

`plane_fit_mode == "debiased"` fits the normal directly as a function of
accumulated point positions AND covariances, and computes `plane_var_`
from a second set of fixed-size accumulators — both O(1) per incoming
point, forever, with no raw-point storage or replay.

## The math

Under additive per-point measurement noise (`p_i = p_i_true + ε_i`,
`ε_i` independent, covariance `Cov_i`), the raw empirical scatter matrix
is a biased estimate of the true (noise-free) structure tensor:

```
E[(p_i - mean)(p_i - mean)^T] = TrueScatter + Cov_i
```

Subtracting the accumulated noise term gives a consistent (unbiased)
closed-form estimate — no iteration needed (contrast with the
IRLS-weighted total-least-squares estimator discussed earlier in this
investigation, which is more statistically efficient but requires
iterating since its weights depend on the unknown normal):

```
M_debiased = Spp/N - mean·mean^T - Scov/N
normal = eigenvector of M_debiased's smallest eigenvalue
```

Accumulators (raw, unweighted, updated O(1) per point in
`VoxelPlane::addPoints()`, never reset/replayed):
- `N`      — point count
- `Sp`     = Σ p_i
- `Spp`    = Σ p_i p_i^T
- `Scov`   = Σ Cov_i
- `V[a][b]` (a≤b ∈ {0,1,2}) = Σ p_i(a) p_i(b) Cov_i   (6 unique 3×3 blocks)
- `W[a]`   = Σ p_i(a) Cov_i                            (3 3×3 blocks)
- `sum_sensor_var` = Σ sensor_var_i (for `sensor_noise_floor_eig0`)

At query time (`VoxelPlane::refitDebiased()`), reconstruct the
mean-centered cross tensor and combine with the current eigenbasis
(`M1 = n·y^T + y·n^T`, `M2 = n·x^T + x·n^T`, `denom1/2` as in the
existing PCA Jacobian):

```
Vw[a][b] = V[a][b]/N²,  Ww[a] = W[a]/N²,  Scov_w = Scov/N²
U(a,b)  = Vw[a][b] − mean(a)·Ww[b] − mean(b)·Ww[a] + mean(a)·mean(b)·Scov_w
Wc(a)   = (Ww[a] − mean(a)·Scov_w) · normal

plane_var_(θ1,θ1) = [Σ_ab M1.col(a)·U(a,b)·M1.col(b)] / denom1²
plane_var_(θ2,θ2) = [Σ_ab M2.col(a)·U(a,b)·M2.col(b)] / denom2²
plane_var_(θ1,θ2) = [Σ_ab M1.col(a)·U(a,b)·M2.col(b)] / (denom1·denom2)
plane_var_(d,d)   = normal · Scov_w · normal
plane_var_(θ1,d)  = −[Σ_a M1.col(a)·Wc(a)] / denom1
plane_var_(θ2,d)  = −[Σ_a M2.col(a)·Wc(a)] / denom2
```

These formulas were derived by re-summing the existing per-point `Jmin`
construction (`voxelplane.cpp`'s `update()`, lines ~198-232) algebraically
and confirmed to match it exactly (to float precision) on a synthetic
7-point/random-SPD-covariance numerical cross-check before implementation.

## Config

`voxel_map/plane/plane_fit_mode`: `"pca"` (default, unchanged) or
`"debiased"`.

## Lifecycle differences from PCA mode

`VoxelNode` never locks a `plane_fit_mode == "debiased"` voxel into
`CONVERGED` — there's no O(N) cost to keep refitting, so no reason to
stop incorporating new points (this also fixes, for debiased voxels, the
otherwise-standing issue that a `CONVERGED` PCA voxel silently discards
every point that arrives after lock-in). Consequently `min_update_points`,
`max_points`, `convergence_mode`, and `min_frames_to_converge` are
PCA-mode-only under `"debiased"` — they're left as-is in code (gated
behind the non-debiased branch), just inert for this mode.
`min_init_points`/`min_frames_to_init` still apply identically to both
modes for the first fit.

The rare non-planar/subdivide path (give up, push points down to child
voxels) has no raw-point history to hand off in debiased mode (the whole
point of the accumulator design is not retaining raw points) — it falls
back to handing children just the current incoming batch, the same
accepted-approximation spirit as the existing `bins_`-collapsed-to-
representative-points path for PCA's `always_update` mode.

## 2026-08-24 (pass 2): sensor/pose split + shrinkage, and a second bug found

The first pass's smoke test diverged catastrophically (ATE > 25,000m). Root
cause investigated and (partially) fixed:

**Confirmed bug #1 — naive per-point pose_cov subtraction over-corrects.**
`PointXYZCov::cov` is `sensor_cov + pose_cov` combined, but `pose_cov`
(from `state_->poseCovAt()`) is *shared* across every point in one frame,
not independent per point. A purely-shared additive shift cancels
identically out of that frame's own scatter about its own mean — it
contributes **zero** true bias for a single frame, so subtracting the full
`Cov_i` (including `pose_cov`) once per point over-corrects, and can drive
`M_debiased`'s eigenvalue negative. Confirmed numerically with a synthetic
multi-frame simulation (shared per-frame bias + independent per-point
sensor noise): naive full-`Cov_i` debiasing gave eig0 = **−0.0025** at
F=1 frame (should be ≈0, matching raw/true), only becoming accurate as
F → ∞ (converges correctly at F=20/100, matching a proper ANOVA
within-frame/between-frame decomposition of scatter).

**Fix implemented**: split `PointXYZCov` into `cov` (unchanged, combined,
still used by `plane_var_`) and a new `sensor_cov` field (full anisotropic
sensor-only matrix, captured in `voxelmap.cpp::updateMap()` before the
`poseCovAt()` fold-in). `VoxelPlane` now accumulates `Scov_sensor_`
separately and computes:

```
pose_shrink = (F > 1) ? (F-1)/F : 0        (F = distinct_frames_)
M_debiased = Spp/N - mean·mean^T - Scov_sensor_/N - pose_shrink·(Scov_-Scov_sensor_)/N
```

`(F-1)/F` shrinkage: zero correction at F=1 (matches the proof that a
single frame's shared bias contributes no real scatter), approaching full
correction as F grows. Verified numerically against the same synthetic
simulation: F=1 gives eig0=0.000007 (vs. true 0.000001, no longer
negative), F=20/100 give 0.00015-0.00018 (much closer to true than the
uncorrected raw scatter's 0.0026-0.0028, though not perfectly converged --
a known limitation of this simple shrinkage vs. a full per-frame ANOVA).
`plane_var_`'s own accumulators (`Scov_`, `V_`, `W_`) are UNCHANGED --
still use the full combined `cov`, since that term represents residual
noise for weighting/gating purposes, not a fit-determining bias
correction, and over-representing it there is conservative (safe), not
dangerous, unlike in the fit itself.

Also fixed: `VoxelNode::insertPoints()`'s debiased branch forced
`should_lock = false` unconditionally, which also suppressed the
legitimate non-planar `-> DISABLED` lock-out at `max_layer` (not just the
intended "never freeze a converged good plane" suppression). Fixed to
`should_lock = !now_plane && (layer_ >= opts_->max_layer)` for debiased
mode. Not the crash's root cause (residual generation gates on
`is_plane_`, not `VoxelStatus`) but a real gap, fixed regardless.

**Result: pca-mode confirmed still byte-identical (ATE=0.0260m). Debiased
mode STILL diverges after this fix** (ATE=50,340m on the retest) --
the pose_cov shrinkage fix was necessary but not sufficient.

**Bug #2, found but NOT fixed this pass**: a short debug run
(`log_debug_en=true`, killed after ~745MB/a few seconds of `plane_init.txt`
-- do not rerun this without a tight time/size limit, it reproduces the
original runaway-log symptom almost immediately once any voxel starts
producing bad fits) showed `plane_var_trace` values of **13.998 and
18.541** for early 9-point fits, where `eig0 ≈ 2.3e-7` and
`plane_threshold = 0.01` -- i.e. `plane_var_` is 3+ orders of magnitude
larger than any sane residual-noise scale (sensor noise is ~1e-4 range,
`sigma_squared` floors at 1e-3). Likely mechanism: `plane_var_`'s existing
formula (both the original O(N) Jmin loop and pass-1's accumulator
reconstruction, same 1/denom1²·1/denom2² structure) amplifies sharply when
`denom1 = eig0-eig1` or `denom2 = eig0-eig2` is small -- in this sample,
`eig0=2.33e-7`, `eig1=4.76e-4`, so `denom1 ≈ -4.76e-4`, and the `1/denom1²`
factor blows up. This amplification risk exists in PCA mode's identical
formula too, but is plausibly far more common in debiased mode because the
debiasing subtraction can push `eig0` anomalously close to `eig1` (by
construction, it's specifically shrinking eig0 toward/through zero).
A huge `plane_var_` inflates `gate()`'s `sigma_gate_squared`
(`sigma_diag_squared + plane_var_term`), which WIDENS point-to-plane
acceptance (more, worse points get admitted as residuals) rather than
narrowing it -- a plausible direct mechanism for feeding bad residuals
into LIO and diverging. **Not yet fixed or confirmed as the actual root
cause** -- next step would be adding a `denom1`/`denom2` floor (reject the
fit, or clamp `plane_var_` when `|denom1|`/`|denom2|` is below some
epsilon, mirroring the existing `eigen_values_(1) < 1e-8` guard) and
re-testing. Out of scope for this pass per the "stop after ~2 failed
hypotheses" instruction -- reporting back instead.

## Known caveat — correlated single-viewpoint noise

The debiasing correction assumes independent per-point noise. This
codebase has documented empirical evidence (`VoxelOpts::
sensor_noise_floor_eig0`'s doc comment, `voxelmap_utils.h`) that points
from a single lidar frame/viewpoint have *correlated* noise, making
empirical eig0 look ~200x smaller than reality. For the debiased fit this
risks over-subtraction (`Scov` larger than the true single-viewpoint
scatter), driving the debiased eigenvalue negative on early/single-frame
fits. `min_frames_to_init` and `sensor_noise_floor_eig0`'s floor both
still apply as guards.

## 2026-08-24/25: root cause found and fixed (main session, foreground)

Passes 3/4/4b/4c (all forks, not individually detailed above) tried a
`denom1`/`denom2` floor and a `plane_var_` trace ceiling -- both real,
correct fixes for genuine failure modes, but neither was the actual root
cause: debiased mode still diverged catastrophically on `eee_01`
(47,091m) even with both in place. Direct investigation (voxel-init
logging added to `VoxelNode::insertPoints()`, comparing PCA vs debiased
side-by-side on the same bag) found the true mechanism:

1. **`is_plane_`'s reject-on-any-negative-eig0 rule was wrong.** A
   population covariance's eigenvalues can never be negative -- a negative
   *sample* eigenvalue is definitionally sampling noise (picking the
   smallest of three correlated noisy quantities is a biased estimator of
   the true smallest eigenvalue, especially at low N/F), not evidence the
   surface isn't flat. Rejecting on any negative `eig0` threw out
   genuinely-flat, healthy candidates whose corrected estimate landed
   mildly negative by chance -- confirmed empirically: rejected candidates
   at N=6-24, F=2 had *raw* (uncorrected) eig0 of 1e-8 to 1e-4 (genuinely
   near-flat), pushed to ~-0.0002 by a sensor-noise correction of
   comparable magnitude to its own sampling noise. **Fix: clamp
   `eigen_values_(0)` at 0 instead of rejecting** (`voxelplane.cpp`,
   `refitDebiased()`) -- a mildly-negative result now correctly reads as
   "flat," while `!isfinite` stays a hard reject.
2. **The calibration/bootstrap window (frame 0) was polluting the sensor-
   noise correction.** That window aggregates many scans from a
   *stationary* sensor looking at the same geometry from the same
   vantage point repeatedly -- per-point sensor noise is correlated
   across those points (shared range/incidence-angle error), not
   independent the way it is across genuinely different frames/
   viewpoints once the sensor is moving. Subtracting the full per-point
   `sensor_cov` sum as if independent over-corrected there, the same
   mechanism as the already-fixed shared-`pose_cov` bug, just for
   `sensor_cov` during bootstrap specifically. **Fix: `VoxelPlane::
   addPoints()` gained a `trust_sensor_noise` parameter** (false for
   `g_current_frame_idx == 0`, true otherwise) -- bootstrap points still
   contribute to the fit itself (`Sp_`/`Spp_`), just not to the
   subtracted correction (`Scov_sensor_`/`sum_sensor_var_`).
3. **A real, independent bug in `VoxelNode::insertPoints()`'s debiased
   branch**: it never gated on `min_init_points` before acting on
   `is_plane_`/subdividing -- it was evaluating and rejecting fits from
   as few as N=1, and subdividing the voxel immediately on rejection.
   Confirmed via `plane_init.txt` logging: 117,955/118,073 first-fit
   attempts on an 8s `eee_01` window were rejected, 114,599 of those at
   `n_points=1`. **Fix**: `if (!plane_ptr_->isInit()) return;` added
   right after `addPoints()`, mirroring the PCA branch's existing
   `min_init_points` gate.

**Result with all three fixes**: `eee_01` LIO-only ATE went from
catastrophic divergence to **0.0260m — an exact match to PCA**, with
`plane_fit_pose_cov_mode: "sensor_only"` (excludes `pos_cov` from
`plane_var_`'s accumulators too, extended there from its original
PCA-only scope after evidence it mattered for debiased mode's small-N
voxels). `n_active_voxels` also converged to match PCA almost exactly.

**Full 25-sequence sweep result (2026-08-24, `ablations/
debiased_fullsuite`, since deleted -- see the note below)**: NTU_VIRAL
9/9 healthy, closely matching PCA (0.0275-0.0378m vs PCA's
0.0260-0.0373m). **HILTI mostly still broken** -- only `exp21` clearly
healthy (0.15m), `exp01`/`site1_handheld_1` marginal but degraded
(4.45m/6.49m vs PCA's 0.069m/~0.6m), everything else catastrophic
(hundreds to hundreds-of-thousands of meters), plus one crash
(`site2_robot_1`, not reproduced in isolation). Every diagnostic run that
led to the three fixes above was on `eee_01`/NTU_VIRAL only -- HILTI
clearly hits something this investigation never characterized (different
LiDAR/point density, possibly the correlated-noise caveat above biting
harder there). **This remains open** -- picking the debiased-mode
investigation back up should start with the same voxel-init logging
technique, applied to a representative HILTI failure, before assuming
the mechanism is the same as `eee_01`'s.

**2026-08-25: raw sweep/debug data deleted.** The `plane_init.txt`/
`noise_floor.txt` debug dumps and job-queue scratch from passes 3/4/4b/4c
and the diagnostic sessions above (`ablations/_job_queue/{pass3_check,
pass4_check,pass4b_check,pass4c_check,dbg2_check,debiased_check,
debiased_check2,fg_check}` and `ablations/{debiased_check,debiased_check2,
debiased_v2,debiased_dbg,debiased_fullsuite,eee01_pfn_diag,
eee01_pfn_diag2,eee01_pfn1_fix,eee01_dryrun,dryrun_sanity}`) were removed
as resolved-investigation scratch -- the mechanism, fixes, and final
numbers are all captured above. If HILTI's remaining failure is
revisited, start fresh rather than expecting to find old diagnostic
output.
