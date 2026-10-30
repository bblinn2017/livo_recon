# Q-scale calibration restructure: move off li_init's EKF, build on livo_recon

## Why (2026-10-29 investigation summary)

A multi-day investigation into a persistent ~9.9-10Hz acceleration-ripple
artifact in `LiDAR_IMU_Init`'s (li_init's) LiDAR-only trajectory -- the
reference li_init's Q-scale calibration pipeline (`Qscale_calib.sh`,
`ct_refine`, `qscale_fit`) uses to derive livo_recon's `imu.process_noise.motion.{acc,gyro}_scale`
-- found:

1. The ripple is a genuine, reproducible oscillation in li_init's own EKF
   state (not a finite-differencing artifact of the trajectory output):
   the EKF's internal `state.vel_end` AND its position-velocity
   cross-covariance block both show a continuous ~9.9Hz oscillation,
   confirmed via a dedicated diagnostic (`debug/log_vel_cov` in
   `laserMapping.cpp`) that logs these directly, not via finite-differenced
   position.
2. It is NOT a map-corruption / self-referential-feedback artifact: it
   reproduces (even more strongly) when the EKF is reset to localize
   against a frozen, independently-pre-built static map
   (`mapping/prior_map_path`, `map_frozen=true`) built from a clean,
   known-stationary 19.2s window, rather than the normal
   incrementally-updated map.
3. It is NOT specific to offline (direct bag-read) testing: live mode
   (genuine `rosbag play -r 1.0` with real ROS scheduling/queuing jitter)
   reproduces it to 2-3 significant figures.
4. It survives `ct_refine`'s B-spline batch smoothing (median ripple
   magnitude roughly halved, but NOT eliminated -- FFT peak amplitude and
   worst-case excursions remain comparable to li_init's raw trajectory).
5. Likely root cause: li_init's measurement model uses a single hardcoded
   per-point weight (`R_inv = 1000` i.e. every point-to-plane residual
   assumed variance 0.001 m^2, in `laserMapping.cpp`), ignoring that
   nearby points on the same surface are correlated, not independent (an
   issue already established earlier in this same investigation via
   livo_recon's own `residual_binning.h`, with 65-sigma-level evidence of
   the overconfidence this produces). An EKF whose aggregate per-scan
   information gain is systematically overconfident relative to its
   process-noise assumption is a classic recipe for oscillatory
   (under-damped/ringing) covariance convergence, synchronized to the scan
   update rate -- which is exactly ~9.9-10Hz here (one native full-FOV
   scan period).
6. **The deeper problem, and why patching li_init's constants is NOT the
   fix**: li_init's EKF process-noise constants (`acc_cov=2`, `gyr_cov=2`,
   bias covariances, etc. in `config/mid360.yaml`) are generic
   FAST-LIO/FAST-LIVO2-lineage defaults, never independently validated for
   this sensor. Using an EKF with unvalidated process-noise tuning to
   produce the reference trajectory that is THEN used to derive IMU
   calibration constants (`acc_scale`/`gyro_scale`) is circular: we would
   be calibrating the IMU against a trajectory whose own fidelity depends
   on a different set of equally-unvalidated assumptions. Porting
   livo_recon's own (more principled) per-point measurement-noise model
   into li_init's EKF would still leave the *process*-noise side of the
   circularity unresolved, since livo_recon's LIO is also an EKF with its
   own process-noise prior.

## The fix: a batch (not sequential/EKF) geometry-only trajectory estimator

A batch nonlinear-least-squares spline fit over LiDAR point-to-plane
residuals needs NO IMU process-noise prior at all -- there is no
"predict" step trusting an assumed motion model, only a geometric
fit. This breaks the circularity for the *translation/rotation* reference
used by Q-scale fitting. `ct_refine` (`src/ct_refine.cpp`,
`src/ct_refine_node.cpp`, both currently living in
`src/LiDAR_IMU_Init/`) already does exactly this kind of estimation (a
`ceres`-based batch optimization over a cumulative B-spline,
`splineEval()`), and is already used as a post-hoc refiner today -- but
it currently uses the SAME kind of flat, non-adaptive residual weight
(`1.0 / opt.sigma`, one global scalar) as li_init's EKF does, so simply
using its existing output does not actually solve the identified problem
(point 4 above: the ripple survives it).

**Decision (2026-10-29, explicit user instruction): abandon
`LiDAR_IMU_Init` as the home for this work entirely. Rebuild the whole
Q-scale reference-trajectory pipeline inside `livo_recon`, reusing its
existing, already-principled code wherever possible, rather than
continuing to patch or port pieces into the legacy li_init codebase.**

## Three pieces to build/merge, all living in livo_recon from now on

1. **Bag reading / point preprocessing**: reuse livo_recon's existing
   callback/preprocessing path (`src/processing/` `cbk_proc_`-equivalent
   code, see `livo_recon_node.cpp`'s `LivoReconNode` member list:
   `cbk_proc_`, `imu_proc_`) instead of li_init's own
   `src/LiDAR_IMU_Init/src/preprocess.cpp`. This is already the production
   code path, so the reference trajectory is built from the same raw-point
   handling (timing, reflectivity, per-point fields) the deployed system
   actually uses -- not a separately-maintained parallel implementation.

2. **Per-point measurement weight**: reuse (do not reimplement)
   `VoxelPlane::gate()`'s `sigma_diag_squared` / `plane_var_term`
   computation (`include/livo_recon/lio/voxelplane.h`,
   `src/processing/lio_iekf.cpp` around line 1147's
   `S_i = floor_term + sigma_diag_squared + plane_var_term + s_prior_pose`)
   as the per-residual weight in the new batch spline-fit cost function,
   in place of ct_refine's current single global `1.0/opt.sigma`. This is
   safe to reuse directly without reintroducing the circularity: it is a
   property of the *measurement* (sensor noise projected onto the matched
   plane's normal, plus the plane fit's own uncertainty), not an assumed
   process/motion-model prior.

3. **Trajectory optimization**: a new batch B-spline optimizer, living in
   livo_recon (e.g. `src/tools/trajectory_refine.cpp` /
   `include/livo_recon/calib/trajectory_refine.h`, names TBD), structurally
   based on `ct_refine.cpp`'s existing `splineEval()`/`PlaneCost`/ceres
   solve machinery (the batch-estimation APPROACH is right and should be
   kept; only the point-source, the residual weighting, and the home
   codebase change) but fed by (1) and weighted by (2).

**Unchanged**: `qscale_fit.cpp`'s acc_scale/gyro_scale estimation itself
(currently `src/LiDAR_IMU_Init/src/qscale_fit.cpp`) treats the input
trajectory as a given reference and fits IMU scale factors against it --
this logic doesn't need to change, only what feeds it. It should be
ported into livo_recon too (so the whole pipeline lives in one codebase
per the user's instruction), but as a near-verbatim move, not a redesign.

## Concrete migration plan

- [ ] New livo_recon module layout (names TBD, follow existing
      `include/livo_recon/<area>/` conventions):
      - `include/livo_recon/calib/trajectory_refine.h` + `src/calib/trajectory_refine.cpp`
        -- the B-spline batch optimizer, ported from `ct_refine.cpp`
      - `include/livo_recon/calib/qscale_fit.h` + `src/calib/qscale_fit.cpp`
        -- near-verbatim port of `qscale_fit.cpp`'s IMU-scale-factor estimation
      - `src/tools/qscale_calib_node.cpp` or similar -- the new
        standalone entry point replacing `Qscale_calib.sh`'s
        `ct_refine` + `qscale_fit` two-step call
- [ ] Wire the new trajectory_refine's per-point residual weight to
      `VoxelPlane::gate()`'s sigma computation -- likely needs
      `VoxelPlane`/`voxelplane.h` to expose a standalone/static entry
      point usable outside the live LIO iekf loop (currently gate() is a
      `VoxelPlane` member, called from inside `lio_iekf.cpp`'s per-scan
      processing -- check whether it can be called directly against a
      point+existing voxel map, or needs restructuring to be batch-callable)
- [ ] Feed trajectory_refine from livo_recon's own bag reading /
      preprocessing instead of li_init's `preprocess.cpp`
- [ ] Validate on `calib_10_28.bag` (the same bag this whole
      investigation used): re-run the established 8-12Hz
      envelope/FFT-peak diagnostic (methodology in this session's
      scratchpad scripts, e.g. `compare_az_incidence_correction.py`'s
      `ripple_stats()`) on the NEW pipeline's output trajectory and
      confirm the ripple is actually gone (not just reduced) -- this is
      the acceptance test for the whole restructuring effort
- [ ] Re-derive `acc_scale`/`gyro_scale` for the mid360 device from the
      new pipeline and compare against the currently-live production
      values (`gyro_scale=0.0134`, `acc_scale=0.6417`, linear scale form,
      committed 2026-10-2x) -- expect these to change meaningfully if the
      old values were corrupted by the ripple
- [ ] Update `BEST_CONFIGS_LIVO_RECON.md` / `MANIFEST.md` once the new
      values are validated; retire `Qscale_calib.sh` and the
      `LiDAR_IMU_Init` calibration path for this device once the new
      pipeline is confirmed working (do not delete `LiDAR_IMU_Init`
      itself without separate confirmation -- other things may still
      depend on it; this plan only covers the Q-scale calibration use case)

## Concrete integration design (resolved 2026-10-29, during scaffolding)

- **`VoxelPlane::gate()` does NOT need restructuring.** Its caller,
  `VoxelMap::findPlaneResidual(const WorldPointCov& pt, Residual &res, ...)`
  (`include/livo_recon/map/voxelmap.h`), is already public and callable
  against ANY already-built `VoxelMap` -- live-incremental or a
  separately pre-built/frozen one (exactly like li_init's
  `mapping/prior_map_path`/`map_frozen` pattern this investigation already
  validated works cleanly). A batch estimator should build the map ONCE
  (via `VoxelMap::bootstrap()` or `updateMap()` over the whole bag, the
  same way li_init's Pass-1-build-a-map test did this session) and then
  only ever READ from it via `findPlaneResidual()` -- never call
  `updateMap()` again during the spline solve.
- **No ceres-autodiff blocker.** `VoxelMap::findPlaneResidual()` is plain
  `double`-valued (not Jet-templated), so ceres cannot autodiff through it
  directly -- but this is fine, because `ct_refine.cpp`'s EXISTING
  architecture already separates association from optimization: it
  associates each point to a plane (an external, non-differentiated step)
  BEFORE each ceres solve, then runs the solve with that iteration's
  {normal, d} TREATED AS FIXED CONSTANTS inside `PlaneCost` (only the
  spline control points are differentiated). The new tool keeps this
  exact pattern: call `findPlaneResidual()` once per point per
  outer-iteration (like li_init's own iterated-EKF re-association, or
  `ct_refine`'s existing iterate-then-solve loop) to get
  `{r, normal, sigma_squared}`, construct `WorldPointCov` for the query
  using the CURRENT spline-evaluated pose estimate (so `sensor_cov`
  rotates correctly as the trajectory estimate updates across outer
  iterations), then build/solve one `ceres::Problem` with
  `weight = 1.0 / std::sqrt(res.sigma_squared)` per residual, in place of
  `PlaneCost`'s current single global `1.0/opt.sigma`. Re-associate and
  re-solve for a few outer iterations until convergence, same as
  `ct_refine` already does today.
- **Full data-flow chain confirmed by reading `src/processing/lio_base.cpp`
  (~line 668)**: each raw point already carries a BODY-FRAME `sensor_cov`
  as a `PointXYZCov` (built once during preprocessing -- pose-independent,
  directly reusable as-is); `state_->toWorld(pts[i])` is the only
  pose-DEPENDENT step, rotating `{point, sensor_cov}` into world frame
  using whatever the CURRENT pose estimate is. In the batch spline
  optimizer, this becomes: at each outer (association) iteration,
  evaluate the spline at each point's own capture time to get that
  iteration's pose estimate, call the equivalent of `toWorld()` with it
  (not `state_`, which is the live EKF's state -- the new tool needs its
  own small helper doing the same rotation, trivial given the spline
  already produces a pose), then `VoxelMap::findPlaneResidual()`. This
  closes the design: body-frame `sensor_cov` -> pose-dependent world
  rotation (from the CURRENT spline iterate, not an EKF state) ->
  `findPlaneResidual()` -> `{r, sigma_squared, normal}` -> fixed-weight
  `PlaneCost` for that outer iteration's ceres solve. No further
  unresolved design questions; next phase is implementation (new ceres
  cost functor + CMake target + bag-reading glue + the inevitable
  build/test/debug cycles each new tool in this project has needed).
- `ceres`/`glog` are currently only installed in the `livo_test_h2`
  container (now also added to `Dockerfile` for future builds, but
  existing h3-h12 containers don't have it without a manual
  `apt-get install libgoogle-glog-dev` or image rebuild) -- confirm
  livo_recon's own build already links these where needed, or whether
  this new tool introduces a new dependency edge.
- Confirm whether `combined.enable=true` also needs `vio.enable=true` to
  be internally consistent for whatever livo_recon LIO-only code path
  trajectory_refine reuses (separate, currently-unresolved finding from
  this same session: a `vio.enable=true`/`combined.enable=true` run
  produced an odometry.txt byte-identical to the LIO-only run -- a
  silent VIO-admission-gate-lockout-style bug, traced to somewhere in/
  after `CombinedProc`'s real `ekf_.applyMeanUpdate()` call but not yet
  root-caused; irrelevant to this LIO-only trajectory_refine work but
  worth keeping separate so it doesn't get conflated).

## Phase 1 status (2026-10-29, same session): builds and runs, poses are NaN

`qscale_refine_livo_recon` (new executable, `src/tools/qscale_refine.cpp`,
wired into `CMakeLists.txt` as its own target per explicit instruction --
NOT a mode of `livo_recon_node`) **compiles, links, and runs end-to-end**
on `calib_10_28.bag` via a new `launch/qscale_refine_mid360.launch`,
writing one TUM-format pose row per scan at the right cadence (189 rows
for a 19s window, ~10Hz, matching the native scan rate). Two real bugs
were found and fixed along the way, both confirmed by direct
before/after testing:

1. livo_recon's strict unclaimed-param scanner (`[config/REFUSED]`,
   aborts via `std::runtime_error` if any loaded rosparam key has no
   reader) requires calling `loadParameters()` on every processor the
   real node calls it on, even ones this tool never uses for actual
   processing (`VioProc`, `CombinedProc`, `ctx.printer`,
   `ctx.voxel_map`, plus a few node-level `paramWarn` reads for
   `run/*`/`log/profiler_path`) -- fixed by mirroring
   `LivoReconNode::loadParameters()`'s exact call order/set, even though
   this tool's main loop only ever calls LIO functions.
2. With `vio/enable=true` (mid360.yaml's default), `CbkProc::syncMeasures()`
   apparently waits for synchronized image data before ever completing a
   measure group -- without subscribing to/feeding the image topic too,
   `popMeasureGroup()` silently never succeeds (0 poses written,
   regardless of bag duration budget). Fixed by adding the image topic to
   the tool's `rosbag::View` query and dispatching it to
   `cbk_proc.imageCallback()`, exactly like `LivoReconNode::runOffline()`'s
   own `feedNextMessage()` does.

**Update (2026-10-29, same session): NaN bug root-caused and fixed, three bugs total.**
Added temporary `ROS_INFO_STREAM` state prints (now removed) right after
calibration and after each of the first 3 scans' `processIMU`/`deskewAndDownsample`/
`processLIO` calls, per the plan above -- this localized it immediately.

3. **Root cause**: `CalibProc::estimateFromBuffer()`'s log line read
   `calib_imu_samples.size()` as "0 samples collected" even on the
   successful run -- `CalibProcOptions::num_frames` (required > 0,
   `calib/stationary/num_frames: 5` in `mid360.yaml`) stayed at its
   **default-constructed `0`** because this tool never called
   `calib_proc.loadParameters(pnh)` at all (every other processor's
   `loadParameters()` was called, this one was missed). With
   `num_frames=0`, `CalibProc::collectSamples()`'s loop condition
   (`groups_qualifying < num_frames`) is false before the loop ever
   runs, so it returns `0 >= 0` = true *vacuously*, without consuming
   any data -- calibration "succeeds" instantly on an empty buffer,
   writing NaN bias/rotation straight into the shared state. Fixed by
   adding the missing `calib_proc.loadParameters(pnh)` call (mirrors
   `LivoReconNode::loadParameters()`'s exact set now, with none missing).
4. Separately (would have caused a different failure, a `deskewPoints`
   FATAL on garbage point timestamps, once bug 3 was fixed and real data
   started flowing): `DataQueues::ready()`'s internal quiet-stream check
   branches on `offline_mode` -- wall-clock-based by default, bag-time-based
   when `setOfflineMode(true)` is called. `LivoReconNode::runOffline()`
   calls this before its own feed loop; this tool never did, so `ready()`
   used the wall-clock path, which (feeding messages with no real-time
   pacing) either never settles or settles on a race with an empty image
   queue. Fixed by adding `ctx.data_queues->setOfflineMode(true)` right
   before the bag-reading loop, exactly where `runOffline()` places it.
5. Also found while fixing bug 3: the tool's CALIB loop originally wrote
   `ROS_INFO_STREAM_THROTTLE(2.0, calib_proc.estimateFromBuffer())` --
   throttle macros only evaluate their argument inside the throttle
   window, so `estimateFromBuffer()` (which is what actually drains
   `DataQueues` via `collectSamples()`, not just a log line) was being
   skipped on almost every loop iteration. With no real-time pacing,
   `feedNextMessage()` raced ahead and exhausted the entire bag before
   calibration ever ran a second time. Fixed by calling
   `estimateFromBuffer()` unconditionally every iteration and only
   throttling the *logging* of its return value.

**Current status: Phase 1 works end-to-end.** Verified on a 19s window:
calibration converges to a sane rotation/gravity/bias (not NaN), the
voxel map engages (not `INERT`, `binned_fits > 0`), and position grows
smoothly scan-to-scan (184-189 poses for 19s, matching the ~10Hz native
rate). Also ran clean to completion on the full 188s `calib_10_28.bag`
(1883 poses, zero NaN/inf anywhere in the output).

**The 8-12Hz ripple diagnostic does not port directly, and the first
attempt at a workaround was inconclusive -- documenting both honestly
rather than overselling either.**

1. livo_recon has no equivalent of li_init's `cut_frame_num` scan-split
   (grepped `cbk_processing.{h,cpp}` -- nothing). li_init corrects its
   EKF 4x per raw lidar packet (40Hz output) specifically because it
   splits each mid360 scan into 4 sub-scans; livo_recon corrects once
   per native scan (~10Hz). Nyquist at 10Hz is 5Hz, so the one-pose-per-
   scan output (`qscale_refine`'s primary TUM file) cannot represent an
   8-12Hz signal AT ALL -- not "weak signal", structurally un-plottable.
   This isn't a bug to fix; it's a real design difference between the
   two pipelines' correction cadence.
2. Tried a workaround: added an optional `tools/qscale_refine/
   highrate_out_path` dump of `MeasureGroup::poses` (populated by
   `ImuProc::processIMU()` at the IMU's native rate, ~225Hz measured).
   Turned out to be a genuinely continuous dead-reckoned trajectory
   (only one reset at the very start of the whole run, not per-scan as
   first assumed) -- but it's **pre-LIO-correction, open-loop IMU
   integration**. Its acceleration spectrum is dominated by raw IMU
   vibration content (94.6Hz peak, unrelated to anything in this
   investigation), not by anything downstream of the EKF correction
   step. This data answers "does the raw IMU vibrate" (already known,
   yes, ~28Hz line + harmonics from earlier in this investigation), not
   "does livo_recon's correction step reintroduce li_init's ringing" --
   the actual question. **Not a useful proxy for the acceptance test as
   implemented; abandoning this specific approach rather than
   over-interpreting its 9.66Hz 8-12Hz-band FFT peak, which has no clear
   causal story and shouldn't be read as either confirming or refuting
   anything.**

**What WOULD actually answer the question, for a future continuation**:
either (a) add an actual scan-split option to `CbkProc`/`LioProcIekf`
matching li_init's `cut_frame_num=4` so corrections happen at a
comparable ~40Hz cadence and the same bandpass/envelope/FFT methodology
ports directly, or (b) accept the two pipelines correct at genuinely
different rates and instead evaluate the 10Hz output's full observable
band (0-5Hz) for anomalous narrowband power/roughness relative to a
smoothed/expected trajectory, rather than chasing the specific 8-12Hz
band that's an artifact of li_init's own 40Hz design. (a) is more
rigorous (apples-to-apples) but is a real code change to a shared,
widely-used processor, not a quick diagnostic tweak -- (b) is cheaper
and was not attempted this session due to time/scope, not because it's
wrong.

Independent of this unresolved diagnostic: Phase 1's own qualitative
behavior (no NaN, no divergence, voxel map engaging normally, smooth
position growth over the full 188s) is itself evidence the adaptive
per-point weighting + the new validated `acc_scale=0.6417`/
`gyro_scale=0.0134` (linear form, from the `nosplit` calibration fork)
is at least not obviously broken -- just not yet proof it specifically
fixes the ringing this whole investigation started from.

## Phase 2 implemented and run on the full bag (2026-10-29, same session)

`src/calib/trajectory_refine.cpp`/`.h` (new) implements the batch B-spline
optimizer designed above: `SplineTrajectory` (ported verbatim from
`ct_refine.cpp`'s `CtTrajectory` -- same cumulative-SO(3)/plain-R^3 uniform
cubic B-spline math), `PlaneCost`/`KnotPrior` ceres functors (same, but
`PlaneCost::w` is now per-residual, set from `MapBackend::
findPlaneResidual()`'s real `sigma_squared + plane_var_term` instead of a
flat config constant), and `refineTrajectory()` (same outer-iteration
re-association + windowed ceres solve structure, but association is one
`findPlaneResidual()` call per point against the frozen `VoxelMap` built by
Phase 1, not ct_refine's own from-scratch kd-tree + PCA plane fit). Wired
into `qscale_refine.cpp` as an in-process second pass after Phase 1's main
loop (reuses the same bag-read/CbkProc/map, no re-reading the bag or
rebuilding the map) -- new launch args `refine_out_path`/`refine_window_s`
on `qscale_refine_mid360.launch`, and new `tools/qscale_refine/refine/*`
params (`knot`, `prior_rot`, `prior_pos`, `outer`, `window_s`,
`sample_rate`, `voxel`, `per_scan_cap`, `still_t0`/`still_t1`).

**A real crash was found and fixed**: the first end-to-end run (30s
window) died with glibc's `double free or corruption (out)` inside the
ceres solve. Localized via `gdb` (installed on h2 for this) wrapped
around the node via `roslaunch`'s `launch-prefix` (direct `gdb -p <pid>`
attach is blocked by this container's ptrace_scope, but gdb launching its
OWN child process is unaffected by that restriction) -- the backtrace
showed the corruption surfacing inside `Eigen::SimplicialCholeskyBase::
ordering()` / `permute_symm_to_fullsymm()`, called from ceres'
`SPARSE_NORMAL_CHOLESKY`/`EIGEN_SPARSE` linear solver. Bisection (shrinking
the problem to ~90 knots / ~10k residuals, and separately verifying the
per-residual weight range was unremarkable, 10-50, nothing pathological)
ruled out both a scale/conditioning issue and a data/logic bug in this
file's own residual construction. Switching the linear solver to
`DENSE_QR` made the crash disappear entirely (confirmed clean under gdb
too) -- strong evidence the bug is specific to Eigen's sparse-Cholesky
code path IN THIS BINARY, not a general ceres/Eigen problem, since
`ct_refine`'s own separate executable runs the identical
`SPARSE_NORMAL_CHOLESKY`/`EIGEN_SPARSE` setup routinely without issue. The
leading hypothesis (not independently confirmed further, given time) is
an allocator conflict from this target's extra Torch/CUDA/`livo_cuda`
link dependencies (needed for the LIO/VoxelMap/tracker machinery this
tool reuses) interfering with Eigen's sparse-matrix heap allocations --
`ct_refine`'s binary has none of those. `DENSE_QR` itself was not kept:
it's O(rows x cols^2) in memory, infeasible at this tool's default
~500-knot/~250k-residual window size. Final fix: `ceres::CGNR` (iterative,
Jacobi-preconditioned) -- avoids both the crashing code path and the
dense blowup, confirmed stable on the 30s window AND the full 188s run.

**Full-bag (188s) run, end to end**: Phase 1 (1883 LIO poses, ~10Hz) ->
Phase 2 (7527 refined poses, 40Hz, `window_s=25` default, 6 outer
iterations, ~4 minutes wall time) -- clean, zero NaN/inf, point-to-plane
rms converges smoothly (17.52mm -> 17.11mm, matched 1877062/1883000
points at convergence).

**Ripple diagnostic result (the actual acceptance test): MIXED, not a
clean win.** Compared against li_init's baseline (`/tmp/
az_incidence_run36_lidar_odom.txt`, same `ripple_stats()` methodology used
throughout this investigation):
- **8-12Hz FFT peak amplitude: 4444.5 -> 1610.7 (-63.8%)**, same frequency
  (9.90Hz) -- strong evidence the dominant NARROWBAND ringing component
  this whole investigation characterized is substantially reduced by
  adaptive per-point weighting, supporting the original hypothesis
  (li_init's flat `R_inv=1000` weight drives this artifact).
- **8-12Hz envelope: median -11.6% (2.52->2.23mm/s^2, modest improvement),
  but p90 +31.5% and max +65.3% (4.85->6.38, 9.26->15.30) -- WORSE.**
  Traced the top 10 envelope peaks (all at t=156-159s and t=65s, out of
  188s) and ruled out two obvious explanations: NOT at a windowed-solve
  boundary (boundaries are at 12.5s multiples; peaks are 2.5-3.5s away,
  too far for a solve-boundary artifact at this knot spacing), and NOT
  shared real motion (li_init's own envelope in the SAME 150-165s/60-70s
  windows is unremarkable -- max 7.28/6.21, close to its own overall
  baseline of 9.26 max -- while this pipeline's local max there, 15.30,
  is roughly 2x higher than li_init's in the same window). This is a
  genuine, localized artifact specific to this new pipeline, not yet
  root-caused.

**Net assessment**: the core hypothesis (flat weighting drives the
narrowband 9.9Hz ringing) now has real quantitative support from an
independent pipeline, which is a meaningful result on its own. But a new,
different, localized high-envelope artifact appeared at two specific
times, large enough that the pipeline cannot yet be called a clean
replacement for li_init's output. Candidate causes, not yet
investigated (next steps for a future continuation): (1) the per-scan
random voxel-subsample point selection (`selectPointsForRefine()`,
`per_scan_cap=1000` default) is non-deterministic scan-to-scan and could
introduce extra jitter li_init's deterministic 4-way split doesn't have
-- check whether a larger cap or deterministic (not random) selection
changes this; (2) CGNR is iterative and may be under-converging in a
specific window that happens to cover 156-159s/65s (check `ceres::Solver::
Summary::termination_type` per window, not currently inspected); (3) a
genuine VoxelMap coverage gap in that specific region (few/no planes
available, or geometry the voxel-plane PCA fit handles poorly) that
`findPlaneResidual()` handles differently than li_init's own kd-tree+PCA
association -- check `n_miss`/match-rate specifically in that time window
rather than only the whole-run aggregate already logged.

## Phase 1 architecture change: EKF correction replaced with pure ICP (2026-10-29, same session)

User objection (raised independently twice this investigation, worded
differently each time but the same point): using an EKF to produce the
reference trajectory that calibrates that EKF's own process-noise
constants is circular. This was already understood to apply to li_init's
EKF -- the whole reason this restructure exists -- but on reflection it
ALSO applied to Phase 1 as first implemented here: `LioProcIekf::
processLIO()` is itself an iterated EKF, and its correction step computes
a Kalman gain from the propagated prior covariance, which depends on
`imu/process_noise/motion/{acc,gyro}_scale` -- the very constants this
pipeline calibrates. The IMU's MEAN propagation (used for deskewing and
as the inter-scan pose guess) does NOT depend on those scales -- only
their COVARIANCE does -- so routing the seed trajectory through an EKF
correction step still made it soft-dependent on an unvalidated prior.

**Fix**: new `include/livo_recon/calib/icp_odometry.h` + `src/calib/
icp_odometry.cpp` (`calib::icpRegister()`) -- plain iterated point-to-
plane Gauss-Newton registration against the VoxelMap, with NO Kalman
gain and NO prior covariance anywhere in the solve. Derived as the exact
zero-information-prior limit of `EkfUpdate::applyMeanUpdate()` (as
`prior_cov -> infinity`, that function's own solution formula reduces to
`-(HtH)^-1 * Htz`, which is what `icpRegister()` computes directly) --
confirmed by first-principles Jacobian derivation that this is standard
point-to-plane ICP, using the SAME `accumulateLioResiduals()` helper the
real EKF already uses to build `HtH`/`Htz` (so the per-point weighting,
Jacobian convention, and row-ordering are all identical/already-validated
code, not reimplemented). `qscale_refine.cpp`'s main loop no longer calls
`lio_proc.processLIO(mg)` at all: it reads `ctx.state->rot()/pos()`
immediately after `imu_proc.processIMU(mg)` (the IMU's own mean-propagated
prior, written via its own internal `state_->setPropagatedState()` call),
refines that guess via `icpRegister()`, and writes the result back with
`setPropagatedState()` -- never touching `state_->cov()` at all, so there
is no process-noise dependency anywhere in Phase 1's pose solve now.
Velocity for the next scan's IMU propagation is a finite difference of
consecutive ICP-corrected positions (not the raw IMU-integrated velocity,
which would otherwise drift unboundedly over 188s with nothing correcting
it, now that there's no EKF update feeding back into `vel_`).

**Validated**: 30s window -- 294 Phase-1 poses, Phase 2 converges to
17.21mm (vs. 17.17mm for the EKF-seeded version -- within noise, as
expected, since both use the same VoxelMap/association/weighting; the
difference is only in what seeds Phase 2's initial guess). A full 188s
run was launched next to regenerate the ripple-diagnostic numbers above
under the EKF-free architecture -- the comparison needs to be redone from
this version, since the previous Phase-2 ripple result (FFT peak -63.8%,
envelope p90/max worse) was generated from the EKF-seeded Phase 1 and
should not be treated as this architecture's result.

**Full-bag (188s) ICP-seeded result, regenerated**: Phase 1 (1883 poses,
ICP-only, no EKF) -> Phase 2 (7527 refined poses, 40Hz, 6 outer
iterations, ~3.5 minutes wall time, rms 17.55mm -> 17.14mm, matched
1876973/1883000 points at convergence -- both figures consistent with
the EKF-seeded run to within solve noise). Ripple diagnostic vs. li_init:
- 8-12Hz FFT peak amplitude: 4444.5 -> **1709.3 (-61.5%)** (EKF-seeded
  run: -63.8%) -- essentially unchanged.
- 8-12Hz envelope: median -10.2% (EKF-seeded: -11.6%), p90 **+31.5%**
  (identical to EKF-seeded to 1 decimal), max **+69.4%** (EKF-seeded:
  +65.3%) -- essentially unchanged.

**Conclusion: the EKF-vs-ICP choice in Phase 1 does not materially
change Phase 2's output.** This makes sense structurally (Phase 2
re-associates against the frozen VoxelMap and re-solves from scratch
every outer iteration, regardless of what seeded its first iterate) and
is a useful cross-check in its own right: it confirms the mixed finding
above (narrowband 9.9Hz ringing substantially reduced; a new, different,
localized high-envelope artifact near t=156-159s/t=65s) is a property of
Phase 2's own association/weighting, not an artifact of Phase 1's
EKF-circularity bug that has now been fixed. The architecture fix was
still correct and necessary (the user's objection was valid on its own
terms, independent of whether it moved this particular number), but it
does not explain or resolve the open envelope-artifact question -- the
candidate causes already listed above (per-scan random point selection,
CGNR under-convergence in a specific window, a genuine VoxelMap coverage
gap at that time) remain the live leads for a future continuation.

## Map-cleanliness check (2026-10-29, same session): confirmed clean, no fix needed

Follow-up objection: if Phase 1's VoxelMap is itself built from an EKF
run, the circularity problem just moves from the pose solve into the
map. Checked directly rather than assumed, two separate points:

1. **Map insertion uses the ICP pose, not an EKF pose.** `qscale_refine.
   cpp`'s main loop calls `ctx.state->setPropagatedState(R, p, new_vel)`
   (the `icpRegister()` result) BEFORE `ctx.voxel_map->updateMap(mg)`.
   `VoxelMap::updateMapInternal()` (`voxelmap.cpp:574`,
   `pts_world_[i] = state_->toWorld(map_pts[i])`) reads `state_` live at
   call time, so every inserted point uses that scan's ICP pose.
2. **The one place EKF covariance COULD still leak in is already dead
   code.** `ImuProc::processIMU()` does propagate `state_->cov()` every
   scan via `acc_scale`/`gyro_scale`-dependent process noise
   (`imu_processing.cpp:482`) -- unavoidable, since this tool still calls
   `processIMU()` for deskewing. `VoxelMap::updateMapInternal()` then
   computes `pts_world_[i].pos_cov = state_->poseCovAt(...)`
   (`voxelmap.cpp:592`), which would carry that forward -- BUT
   `voxelnode.cpp:152` explicitly zeros `pos_cov` before the binned plane
   fit ("the binned map keeps the sensor covariance only"), and
   `voxelnode.cpp:131-132` only ever extracts `point`/`sensor_cov` when
   adding to the bin accumulator. `pos_cov` is computed in `voxelmap.cpp`
   but never consumed anywhere in the actual map representation. Binned
   is the ONLY `MapBackend` in this codebase (`voxelmap.h`'s own comment:
   "the only MapBackend since R97A"), not a config option that could
   silently differ in some other run.

**Conclusion: no fix needed.** The map Phase 1's ICP (and Phase 2's
association) reads from has no dependency on `acc_scale`/`gyro_scale` or
any EKF covariance, by construction in the current codebase. Worth
having verified explicitly against the actual code rather than assumed.

## Envelope-artifact investigation continued: per-point azimuth/range/phase correlation (in progress)

Added `TrajectoryRefineOptions::debug_dump_path` (CSV: `t range_m
azimuth_deg residual_m weight`, one row per matched point from the LAST
outer iteration, written in `refineTrajectory()`'s association loop) to
check whether the t=60-70s/150-165s envelope anomaly correlates with any
per-point signal, per the user's question (azimuth/phase/range/
reflectance -- the same signals this investigation checked against
li_init's systematic ripple earlier in the session, now checked against
THIS pipeline's localized anomaly instead).

**Raw-bag checks (before the per-point dump), both come back
unremarkable**: extracted `/livox/lidar` points directly for both
anomalous windows plus two baseline windows (via the pure-python
`rosbags` library, no ROS needed) -- range/azimuth coverage and
reflectivity distributions are comparable across all four windows (one
notable but not alarming exception: median reflectivity in the 150-165s
window is lower, 14 vs 37-47 elsewhere, and its range p10 is also lower,
1.01m vs 1.7-1.8m -- consistent with being nearer a darker surface at
that point in the recording, not an obviously broken scan). Raw `/livox/
imu` gyro/accel magnitude in both windows is also unremarkable (if
anything slightly LOWER variance than the baseline window checked).
Neither raw signal explains the anomaly on its own -- next, checking
whether the SOLVE's own per-point residuals/weights (not just the raw
input data) show a pattern specific to these windows.

NOTE: livo_recon's own point pipeline (`PointXYZCov`) does not carry
reflectivity through preprocessing at all (confirmed: no such field),
unlike li_init's `preprocess.cpp` (`intensity = msg->points[i].
reflectivity`) -- the raw-bag reflectivity check above had to go around
livo_recon entirely, directly to the bag. If reflectivity needs to be a
real-time, in-pipeline per-point weighting term in a future continuation
(not just a one-off diagnostic correlation check), it would need to be
added to `PointXYZCov` and threaded through deskew/downsample.

**No azimuth/range/cos-incidence intercept found either (user's follow-up
question, applying the earlier li_init az-incidence methodology to this
pipeline's own residuals, added `cos_incidence` to the dump)**: overall
mean residual -0.0045mm (weighted -0.0198mm) -- no constant offset.
Global OLS `res ~ a + b*cos_inc`: a=0.022mm, b=0.34mm/unit, **R^2=0.0002**.
`res ~ a + b*range`: R^2=0.00002. Joint fit: R^2=0.00024. Per-azimuth-
bucket (36 buckets) intercepts have std=0.21mm, all within +-0.5mm. In
sharp contrast to li_init's residuals (where this same bias was large
enough to need its own correction scheme earlier in this investigation),
`VoxelPlane::gate()`'s adaptive per-point weighting does not reproduce
this specific failure mode -- a genuine positive result supporting the
restructure's core thesis, though it does not explain the envelope
artifact (which remains open at this point in the investigation).

## MAJOR FINDING: a real, pre-existing timestamp bug in livo_recon's own Livox point extraction (2026-10-29, same session)

User's hypothesis, independent of everything above: maybe this isn't a
weighting/association problem at all, but a timestamp-assignment bug --
"are we 100% certain the device is correctly modeling the offset at
which it receives the points? or it could be the lidar message headers
are slightly off?" Checked directly, found a real, confirmed,
deterministic bug, **not specific to this calibration tool -- it affects
`livo_recon_node` (the real, deployed pipeline) too**, for every mid360
dataset.

**The bug**: `pointsFromLivoxMsg()` (`src/utils/lidar/lidar.cpp`) does not
use each point's raw `offset_time` field directly. It smooths/clamps the
per-point time increment against a hardcoded constant, `LIDAR_HZ = 1./
240000.` (`utils/algo/math.h`) -- described in its own (pre-existing) doc
comment as "a guard against rare out-of-order values." 240000 Hz is the
Livox **Avia's** point rate, not the **mid360's**. Measured directly on
`calib_10_28.bag` (all 3777 scan messages, zero exceptions): the mid360's
real per-point `offset_time` step is a **rock-solid 4.947us (202142 Hz,
zero variance across the entire bag)**, and the raw `offset_time` field
is monotonically non-decreasing in 100% of messages checked (zero
negative diffs anywhere) -- i.e. there was never a "rare out-of-order
value" for this guard to actually catch. Since the real 4.947us step is
LARGER than the clamp's 4.1667us (240kHz) threshold, **the clamp fires on
every single point, every single scan, for the entire bag** -- not a
rare-value guard at all in this configuration, but a permanent,
deterministic distortion: the assigned per-point time increments by
exactly 4.1667us regardless of the real elapsed time, so by the end of
each ~50ms-long `CustomMsg` (confirmed: `/livox/lidar` publishes at
~20.09Hz, 3777 msgs/188s, each with ~9984-10080 points), the assigned
time has fallen behind the TRUE time by **a consistent ~8.3ms (median
8.314ms, range 8.309-8.434ms across the whole bag)** -- then resets to 0
at the start of the next scan message. **li_init's own `preprocess.cpp`
has no such clamp at all** (`pl_full[i].curvature = msg->points[i].
offset_time / float(1000000);` -- raw value, no smoothing), confirmed by
direct comparison, so this bug is specific to livo_recon and does not
explain li_init's own earlier-diagnosed ripple; it is a separate,
independently-confirmed problem in the NEW pipeline this session built.

This systematically mis-times every single LiDAR point fed into deskewing
AND this tool's continuous-time spline fit (which trusts `pt.t` directly)
by up to 8.3ms in a sawtooth pattern resetting at the scan-message rate
(~20Hz) -- a very plausible contributor to spurious high-frequency
content when the resulting trajectory is differentiated into
acceleration, independent of (and probably more fundamental than) any of
the per-point weighting/association questions this investigation spent
most of its time on.

**Fix (per user's explicit direction: make the rate a config parameter,
not a second hardcoded constant)**: `CbkProcOptions::lidar_point_rate_hz`
(new, default 240000.0 -- unchanged behavior for any other dataset
config unless it opts in), threaded through `pointsFromLivoxMsg()`'s new
`point_rate_hz` parameter (replacing the `LIDAR_HZ` macro reference) and
`CbkProc::loadParameters()`'s new `cbk/lidar/point_rate_hz` key.
`config/mid360.yaml` now sets this explicitly to `200000` (the mid360's
documented nominal rate -- close enough to the measured 202142 that the
guard still never trips on real data, while erring toward the documented
spec value over one bag's own measurement). This is a real code+config
change, live in both `qscale_refine_livo_recon` AND `livo_recon_node` --
rebuilt and confirmed `cbk/lidar/point_rate_hz: 200000` loads correctly
via the param dump.

**Status: fix implemented and built; full-bag revalidation in progress**
(re-running Phase 1+2 and the ripple diagnostic under the corrected
timestamps -- results to follow). This is the most likely single highest-
leverage finding of this entire continued investigation, independent of
whether it turns out to explain the specific localized envelope artifact
those sections were chasing -- it is a real, silent, ~8.3ms systematic
mis-timing bug that was affecting every mid360 run of the real production
node, not just this calibration tool, until today.

**Revalidation result (full 188s bag, timing fix applied): real
improvement in the typical/aggregate statistics, but a tail-spike issue
remains, now relocated.**

vs. li_init baseline:
- envelope median: **-22.3%** (was -10.2%/-11.6% pre-fix) -- genuinely
  better, not just noise.
- envelope p90: **+11.1%** (was +31.5% pre-fix, identical both pre-fix
  runs) -- much closer to li_init's own value, real improvement.
- 8-12Hz FFT peak amplitude: **-58.0%** (was -61.5%/-63.8% pre-fix) --
  still a strong, consistent reduction, slightly smaller than before but
  within the same range.
- envelope max: **+109.8%** (was +65.3%/+69.4% pre-fix) -- WORSE.

Traced the new max to a **third, different** location (t=174.7s, not
t=65s or t=150-165s as in both pre-fix runs) -- checked li_init's own
envelope in that same window (170-180s): unremarkable, max=6.03,
consistent with its own overall baseline (9.26) -- ruling out shared
real motion there too, same as the previous two spike locations. So the
timing fix is a genuine, confirmed improvement to the typical/median
behavior and the dominant narrowband peak, but did NOT eliminate
occasional large localized spikes -- it just moved where the worst one
happens to land. This is consistent with a separate, still-unexplained
mechanism (the CGNR-under-convergence/sparse-knot-support hypotheses
from the earlier envelope-artifact section, never actually tested) which
the timing fix does not address.

**Honest overall assessment at this point in the investigation**: three
independent, confirmed, real findings this session (EKF-circularity in
both the original problem and this pipeline's own Phase 1; the map
insertion already being clean; the Avia-vs-mid360 LIDAR_HZ timestamp
bug) all moved the aggregate/typical-case numbers in the right direction
without fully eliminating a tail of occasional large localized
excursions that keep reappearing at different times across different
runs/fixes. Whether that tail is itself a real artifact worth chasing
further, or an acceptable residual given how much the typical-case
numbers have already improved, is a judgment call -- not yet resolved.
