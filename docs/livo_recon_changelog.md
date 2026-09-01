# livo_recon changelog

Dated historical/narrative comments migrated out of source on 2026-09-01, per the "historical narrative belongs in docs, not code" convention (see CLAUDE.md). Each entry is the original comment block, verbatim, grouped by file in original line order. Code left behind a one-line `// History (...): see docs/livo_recon_changelog.md#<anchor>` pointer at each site.


## include/livo_recon/common_lib.h

### include/livo_recon/common_lib.h:3-12

<a id="include-livo_recon-common_lib.h-3"></a>

```
// Legacy "everything" aggregator. As of 2026-08-14, NodeContext (and hence
// every processing/*.h) no longer includes this transitively -- see
// node_context.h's doc comment. This header still exists for the handful
// of files that genuinely want the full breadth (ROS/CV/PCL message types
// AND all of livo_recon's own internal utility types together): map/
// voxelmap.h, vio/tracker.h, tools/cache_tracker_output.cpp. New code
// should prefer including exactly the specific internal headers it needs
// (state.h, ekf.h, measures.h, etc.) plus common_ros.h only if it also
// needs ROS/CV/PCL message types -- not this file -- so an unrelated
// header change doesn't force it to rebuild.
```


## include/livo_recon/common_ros.h

### include/livo_recon/common_ros.h:3-10

<a id="include-livo_recon-common_ros.h-3"></a>

```
// ROS/CV/PCL message-type and transport includes, split out of the old
// common_lib.h (2026-08-14) -- these are stable EXTERNAL headers (not
// livo_recon's own churny internal types like ekf.h/state.h), so bundling
// them together has no rebuild-cascade cost. Only livo_recon/common_lib.h
// (the internal-utility-header aggregator) was the actual problem; this
// file is unaffected by that split and can still be included wherever
// broad ROS/CV/PCL access is genuinely needed (cbk_processing, pub_
// processing, evo_processing, livo_recon_node -- the ROS-I/O-facing files).
```


## include/livo_recon/lio/deskew.h

### include/livo_recon/lio/deskew.h:9-22

<a id="include-livo_recon-lio-deskew.h-9"></a>

```
// Per-point LiDAR deskewing. Only ONE strategy lives here now:
// deskewPoints(), a one-shot deskew against the piecewise-IMU-integrated
// trajectory `poses` (ImuProc::propagate()'s output), using the IMU-only
// propagated state. Never re-evaluated once the IEKF loop starts
// correcting state_ -- only the single scan-end pose ever gets corrected.
//
// 2026-08-24: REMOVED an experimental "iterative deskew" alternative
// (deskewAndSelect()/evalSpline(), a re-evaluated-every-IEKF-iteration
// cubic Hermite spline over the whole scan) -- never enabled in any
// production config, and its one validated result came with a known,
// unresolved regression on 3 HILTI sequences. Full design, math, and
// removal rationale preserved at
// docs/removed_livo_recon_spline_deskew_2026aug24.md; literal pre-removal
// code at refactor_snapshots/remove_livo_recon_spline_deskew_2026aug24/.
```


## include/livo_recon/lio/lio_accumulator.h

### include/livo_recon/lio/lio_accumulator.h:9-16

<a id="include-livo_recon-lio-lio_accumulator.h-9"></a>

```
// LIO's point-to-plane residual -> HtH/Htz accumulation, lifted out of
// LioProc (2026-08-14) so both LioProc's own sequential solve AND
// CombinedProc's joint LIO+VIO solve can call the exact same math without
// CombinedProc depending on LioProc's private internals. Deliberately only
// depends on Residual/EkfUpdate/LioCudaBuffers -- not on LioProc, NodeContext,
// or anything ROS -- so touching this file only rebuilds its few actual
// callers, not the whole package (see node_context.h's doc comment for the
// broader rebuild-cascade context this follows).
```

### include/livo_recon/lio/lio_accumulator.h:20-32

<a id="include-livo_recon-lio-lio_accumulator.h-20"></a>

```
// CPU/OMP accumulation: fills out.HtH/Htz/n_meas from the weighted normal
// equations (w = 1/sigma_squared) over every residual. Does NOT call
// out.applyMeanUpdate()/applyCovarianceUpdate() -- callers own the EKF
// solve step (LioProc's own sequential path calls applyMeanUpdate right
// after; CombinedProc sums this with VIO's own accumulation first).
// Resets `out` (HtH/Htz/n_meas only -- dtheta/dt are untouched, they're
// only ever written by applyMeanUpdate) before accumulating. Fixed 6-dim
// (R,P) system.
//
// 2026-08-24: REMOVED the `wide` (12-dim R,P,V,W) accumulation mode --
// existed solely to support the removed iterative-deskew mechanism's
// wide_jacobian_vw option (confirmed regressed, never usable). See
// docs/removed_livo_recon_spline_deskew_2026aug24.md.
```


## include/livo_recon/lio/voxelnode.h

### include/livo_recon/lio/voxelnode.h:123-130

<a id="include-livo_recon-lio-voxelnode.h-123"></a>

```
  // Was `convergence_mode == "always_update"` until 2026-08-30 (T0-B-4):
  // that conflated binning with never-locking, so every experiment
  // comparing a binned vs. "unbinned" arm via convergence_mode also
  // changed whether the voxel could converge and freeze -- a confound
  // T0-B-2/T0-B-3/T3-0's "unbounded, no binning" arm (convergence_mode:
  // "normal", max_points: 100000) never separated from binning itself.
  // Decoupled onto its own opts_->use_bins flag so binning and
  // convergence-locking can be varied independently.
```


## include/livo_recon/lio/voxelplane.h

### include/livo_recon/lio/voxelplane.h:36-65

<a id="include-livo_recon-lio-voxelplane.h-36"></a>

```
  // total_count (default -1 -> use points.size()): the TRUE number of raw
  // points `points` summarizes, for isInit()/isFull() gating -- lets a
  // caller pass a much smaller set of pre-binned representative points
  // (see VoxelNode::always_update's bin accumulator) while still gating
  // convergence against the real accumulated point count, not the
  // (much smaller) representative-point count.
  //
  // weights (default nullptr): per-entry weight, parallel to `points`.
  // When null, falls back to today's behavior exactly (unweighted) --
  // unchanged for callers that still pass raw points (normal/frame_gated
  // modes, where points_ stays bounded by max_points anyway). When
  // provided, points/weights are used directly with NO internal binning:
  // mean = sum(w_i*p_i)/sum(w_i), covariance = sum(w_i*(p_i-mean)(p_i-mean)^T)/sum(w_i),
  // and the plane_var_ Jacobian uses w_i/sum(w_i) per entry. Passing
  // weight_i = each bin's own raw point count gives an EXACT reconstruction
  // of the unweighted PCA over every raw point the bins summarize (see
  // VoxelNode::buildBinReps()).
  //
  // running (default nullptr): when non-null (VoxelNode's default
  // equal-weight points_ path only -- see RunningMoments' docs), mean/
  // covariance are derived from it in O(1) instead of the loops above --
  // exact, not an approximation. `points` is still required alongside it
  // for the per-point plane_var_ Jacobian loop, which this does not
  // affect. Must not be passed together with `weights`.
  // `var_weights`: separate weight vector for the plane_var_ Jacobian
  // loop only (see T3-0c) -- lets a caller change the fitted plane
  // (center/covariance/normal, via `weights`) independently of the
  // uncertainty scaling attached to it. Defaults to `weights` itself
  // when null, matching pre-2026-08-30 behavior (one weight vector for
  // both fit and uncertainty).
```

### include/livo_recon/lio/voxelplane.h:71-102

<a id="include-livo_recon-lio-voxelplane.h-71"></a>

```
  // opts_->plane_fit_mode == "debiased" path: folds `points` into this
  // VoxelPlane's own PERSISTENT O(1) accumulators (never cleared/replayed)
  // and refits normal_/plane_/plane_var_ from them directly -- see
  // docs/debiased_voxel_plane_fit_2026aug24.md for the full derivation.
  // Unlike update(), safe (and intended) to call on every incoming batch
  // forever; never needs the caller to retain raw points.
  //
  // distinct_frames (default -1 -> treated as 1, i.e. maximally
  // conservative/no between-frame correction): VoxelNode's count of
  // distinct frames that have contributed to this voxel so far. Needed for
  // the shared-pose-noise shrinkage correction in refitDebiased() -- see
  // docs/debiased_voxel_plane_fit_2026aug24.md's "single-frame
  // over-subtraction" section. Cheap to pass every call; only the latest
  // value is kept.
  //
  // trust_sensor_noise (default true): false for points from the
  // calibration/bootstrap window (VoxelNode's g_current_frame_idx == 0) --
  // that window aggregates many scans from a STATIONARY sensor looking at
  // the same geometry from the same vantage point repeatedly, so per-point
  // sensor noise is correlated across those points (shared range/
  // incidence-angle-driven error), not independent the way it is across
  // genuinely different frames/viewpoints once the sensor is moving.
  // Subtracting the full per-point sensor_cov sum as if independent
  // over-corrects there the same way naively subtracting shared pose_cov
  // per point did (see refitDebiased()'s pose_shrink) -- confirmed
  // empirically 2026-08-24 (docs/debiased_voxel_plane_fit_2026aug24.md).
  // Points with trust_sensor_noise=false still contribute to Sp_/Spp_ (the
  // fit itself) but not to Scov_sensor_ (what gets subtracted) -- a voxel
  // fed only by bootstrap points gets plain, uncorrected-for-sensor-noise
  // scatter (safe/conservative, matches pca-mode's own untouched
  // behavior), and the correction phases in normally once real,
  // independent-viewpoint frames start contributing.
```

### include/livo_recon/lio/voxelplane.h:122-138

<a id="include-livo_recon-lio-voxelplane.h-122"></a>

```
  // sigma_diag_squared: independent (sensor+pose) noise VARIANCE only.
  // plane_var_term: this plane's own fit-uncertainty contribution, returned
  // separately so the caller (VoxelPlane::computeResidual()) can add it
  // into Residual::sigma_squared. The accept/reject gate still uses the
  // full (sigma_diag_squared + plane_var_term) combined variance.
  // is_candidate (default nullptr): set true as soon as this point clears
  // the purely-geometric checks (finite r, within max_radius*radius_ of
  // this plane's center) -- i.e. "this is a real correspondence, whatever
  // the chi2 test below decides" -- regardless of gate()'s own return
  // value (which still means the full accept/reject decision, unchanged).
  // T0-D's corr.csv wants exactly this population (pre-outlier-gate), see
  // computeResidual().
  // dropped_by_ablation (default nullptr): set true when T3-0e's
  // occ_aniso_drop_mode rejected this correspondence (distinct from the
  // ordinary sigma-gate rejection) -- set AFTER *is_candidate, so a drop
  // still reaches corr.csv instead of silently vanishing (code-review
  // fix, 2026-08-31 -- T0-D's own "log before the gate" guardrail).
```

### include/livo_recon/lio/voxelplane.h:180-185

<a id="include-livo_recon-lio-voxelplane.h-180"></a>

```
  // T3-0e (2026-08-31): 8x8 tangent-frame occupancy bitmask, cell size
  // opts_->voxel_size/8 -- see updateOccupancy()'s doc comment (in the
  // .cpp) for the full design, anchoring, and reset rule. Same structure
  // T8-b's card independently proposes reusing later (J_eff = popcount,
  // the gap detector, the m2 leverage term); built here first because
  // T3-0e needs it now and T8-b is not yet scheduled.
```

### include/livo_recon/lio/voxelplane.h:193-201

<a id="include-livo_recon-lio-voxelplane.h-193"></a>

```
  // Code-review fix, 2026-08-31: occupancyAnisotropy() used to heap-
  // allocate a std::vector and scan all 64 cells on EVERY call, including
  // from computeResidual() inside LioProc::buildResiduals()'s OMP
  // parallel-for -- once per correspondence, twice when covariates
  // logging is on. Cached here instead (recomputed once per fit, in
  // recomputeOccupancyCache(), called at the end of update()/addPoints()
  // after the occupancy-update loop) -- same pattern last_fit_j_ already
  // uses. -2.0 is "not yet computed this fit" (distinct from -1.0,
  // occupancyAnisotropy()'s own "undefined, <3 cells" sentinel).
```

### include/livo_recon/lio/voxelplane.h:238-248

<a id="include-livo_recon-lio-voxelplane.h-238"></a>

```
// T0-F-2b (2026-08-31): per-frame aggregates across every VoxelPlane fit
// this frame, for LioProc's per-frame diagnostic log -- file-scope atomics
// in voxelplane.cpp rather than threading a stats object through every
// VoxelPlane constructor call site (this diagnostic is opt-in and rare;
// changing the constructor signature would touch every caller for a
// feature most runs never use). Call voxelPlaneFrameStatsReset() once at
// the start of each frame (before any VoxelPlane::update()/refitDebiased()
// calls), then voxelPlaneFrameStatsRead() at the end of that same frame.
// Thread-safe (the accumulation calls run inside VoxelMap's OMP
// parallel-for over voxels); NOT re-entrant across overlapping frames --
// this codebase processes one frame at a time, so that never happens.
```

### include/livo_recon/lio/voxelplane.h:252-256

<a id="include-livo_recon-lio-voxelplane.h-252"></a>

```
// 14b (2026-09-01): corr_scan.csv's per-scan accumulator flushes itself on
// every scan_id change (see debugAccumConsistencyCorr() in voxelplane.cpp),
// but the LAST scan of a run never sees a subsequent scan_id to trigger that
// -- call this once, at shutdown, to flush it. No-op if
// log_consistency_corr_en was never on.
```


## include/livo_recon/map/voxelmap.h

### include/livo_recon/map/voxelmap.h:34-45

<a id="include-livo_recon-map-voxelmap.h-34"></a>

```
  // T0-D (2026-08-31): gates whether findPlaneResidual() passes a real
  // scan_id (enabling corr.csv logging, see VoxelOpts::
  // log_consistency_corr_en) or -1 (no-op) down to VoxelPlane::
  // computeResidual() -- NOT part of the MapBackend virtual interface,
  // since only LioProc's own IEKF loop knows which call is the frame's
  // FIRST buildResiduals() (the pre-update, un-relinearized innovation
  // the register's nu/S spec wants) versus a later re-linearizing
  // iteration or a throwaway runDryRunShadowPass() call, neither of which
  // should be logged. Defaults true (log allowed) so any call site that
  // doesn't explicitly manage this (there are none as of this writing,
  // but a future one would silently get the safe default rather than
  // silently losing all logging) still works.
```


## include/livo_recon/node_context.h

### include/livo_recon/node_context.h:8-20

<a id="include-livo_recon-node_context.h-8"></a>

```
// Forward-declares only -- NO full type definitions here. This is the
// single highest-leverage fix for livo_recon's rebuild-cascade problem
// (2026-08-14): every processing/*.h includes this file, so it used to
// transitively pull in common_lib.h (ROS/OpenCV/PCL/Eigen plus ALL of
// livo_recon's own utility headers: ekf.h, state.h, measures.h, etc.) plus
// the full voxelmap.h/tracker.h -- meaning a change to e.g. ekf.h alone
// forced a rebuild of the whole package. Since NodeContext only ever
// stores these as shared_ptr (never by value), the pointee types don't
// need to be complete here -- shared_ptr<Incomplete> is fine as a class
// member (its destructor doesn't need T complete, since make_shared bakes
// a type-erased deleter into the control block at construction time).
// The constructor body (the only place that actually needs the complete
// types, for the make_shared<T>() calls) moved to node_context.cpp.
```


## include/livo_recon/processing/cbk_processing.h

### include/livo_recon/processing/cbk_processing.h:54-65

<a id="include-livo_recon-processing-cbk_processing.h-54"></a>

```
  // 2026-08-24: opt-in LIO shadow dry-run diagnostic (see LioProcOptions::
  // dry_run_point_filter_num, lio_processing.h, and MeasureGroup::
  // dry_run_lidar_points/dry_run_points, measures.h). 0 (default) = no-op,
  // zero cost -- the raw scan is decimated only once (at point_filter_num
  // above), exactly as before this feature existed. > 0 = decimate the
  // SAME raw scan a SECOND time at this point_filter_num and store it in
  // MeasureGroup::dry_run_lidar_points, purely for LioProc's own shadow
  // IEKF pass to consume -- never inserted into the voxel map, never
  // affects the real trajectory. Loaded from the SAME rosparam key as
  // LioProcOptions::dry_run_point_filter_num ("lio/dry_run_point_filter_
  // num") -- two Options structs reading one key, to avoid a
  // cbk_processing<->lio_processing header dependency for a single int.
```

### include/livo_recon/processing/cbk_processing.h:86-98

<a id="include-livo_recon-processing-cbk_processing.h-86"></a>

```
  // Generalized frame-rate subsampling -- ported from FAST-LIVO2's
  // preprocess/image_subsample_n (LIVMapper::img_cbk()). 1 (default)
  // processes every image frame, unchanged from prior behavior. N > 1
  // processes only every Nth frame (frame_counter % N == 0), dropping the
  // rest before any other work (tracker feed, queue push) happens on them.
  // Motivated by the 2026-08-12 sync investigation: a rotating lidar
  // publishing full scans at ~10Hz alongside a 40Hz camera means 3 of
  // every 4 image frames can never pair with a lidar scan anyway (see
  // DataQueues::ready()'s doc comment) -- image_subsample_n=4 on such a
  // dataset (e.g. HILTI slam_2022/2023) matches the camera to the lidar's
  // own natural cadence instead of feeding the tracker 3x more frames than
  // syncMeasures() can ever actually use. NTU_VIRAL's camera/lidar rates
  // don't have this mismatch, so it stays at 1 (no skipping) there.
```


## include/livo_recon/processing/combined_processing.h

### include/livo_recon/processing/combined_processing.h:16-26

<a id="include-livo_recon-processing-combined_processing.h-16"></a>

```
  // Master switch -- when false (default), LivoReconNode::estimateState()
  // takes the original, unchanged sequential LIO-then-VIO path and never
  // touches CombinedProc at all. Opt-in per the user's explicit request
  // (2026-08-14): fuses HtH_lio+HtH_vio into ONE joint EKF update per
  // iteration instead of running LIO to convergence and only then starting
  // VIO's own separate iterated update from that posterior -- intended to
  // better balance translational/rotational correction from both
  // modalities at once. See lio/lio_accumulator.h and vio/vio_accumulator.h
  // for the shared accumulation code both this class and the ORIGINAL
  // sequential LioProc/VioProc now call -- this class adds no new residual
  // math of its own, it only sums the two existing accumulators' output.
```

### include/livo_recon/processing/combined_processing.h:45-50

<a id="include-livo_recon-processing-combined_processing.h-45"></a>

```
  // G6 (2026-09-01). Logs both channels' NLL (nll.txt, channel="lio"/"vio")
  // for the SAME frame via the shared consistency_log.h writers -- see that
  // header's doc comment for why this was previously unaskable in combined
  // mode. Also logs a "rollback" row when the whole-frame VIO rejection
  // fires, so a rejected frame is recorded as one that happened rather than
  // silently vanishing from the log.
```


## include/livo_recon/processing/imu_processing.h

### include/livo_recon/processing/imu_processing.h:12-15

<a id="include-livo_recon-processing-imu_processing.h-12"></a>

```
  // Gates the per-propagation /tmp/imu.txt dump (debugLogImu), which
  // includes a full flattened state covariance matrix every call --
  // previously unconditional (2026-08-09 cleanup, task #149). Off by
  // default.
```

### include/livo_recon/processing/imu_processing.h:17-20

<a id="include-livo_recon-processing-imu_processing.h-17"></a>

```
  // T7-a (2026-09-01). Arms the per-frame Myers-Tapley accumulators below.
  // Pure instrumentation: it changes no state, no covariance and no
  // trajectory, so it is pinned ON for the whole sweep rather than being a
  // factor in it.
```

### include/livo_recon/processing/imu_processing.h:24-44

<a id="include-livo_recon-processing-imu_processing.h-24"></a>

```
  // T0-E (2026-08-31), split per T7 step 2 (2026-08-31): Q = per-block
  // alpha * Q0 -- each scales one part of the process-noise matrix
  // (cov_w) built each IMU propagation substep, before it's added into
  // state_->cov():
  //   q_alpha_gyr  -- rotation block (from varGyr)
  //   q_alpha_acc  -- velocity block, and (second_order only) the
  //                   position block and position/velocity cross terms
  //                   (all derived from varAcc via acc_noise_world)
  //   q_alpha_bias -- both bias random-walk blocks (BG and BA) together;
  //                   not split further, since T7-a/T7-b's per-block
  //                   estimators only distinguish acc/gyr/bias, not
  //                   bias_acc vs bias_gyr separately
  // All default 1.0 -- bit-identical to pre-split behavior when equal.
  // Setting all three to the same value reproduces the old single-scalar
  // q_alpha sweep (T0-E-2's 240 jobs used the pre-split scalar this way,
  // uniformly); this option exists so scripts/analysis/{consistency.py's
  // NLL panel, qsens.py's secant driver} can externally re-run a sequence
  // at different alpha (uniform or per-block) and compare total NLL (see
  // LioProcOptions::log_nll_en) to find the alpha that maximizes filter
  // consistency -- now with per-term weighting available, per T7-a/T7-b's
  // own per-block Q estimates.
```

### include/livo_recon/processing/imu_processing.h:78-82

<a id="include-livo_recon-processing-imu_processing.h-78"></a>

```
  // Just propagate() -- deskewing/downsampling moved to LioProc (2026-08-18),
  // since it's now intrinsically a LIO concern, not a one-shot IMU-side
  // step. This is a thin wrapper rather than callers just calling
  // propagate() directly so the TimedScope("imu/propagate") profiler tag
  // stays intact.
```


## include/livo_recon/processing/lio_processing.h

### include/livo_recon/processing/lio_processing.h:24-27

<a id="include-livo_recon-processing-lio_processing.h-24"></a>

```
  // Gates the per-update /tmp/lio.txt dump (debugLogLio, includes a full
  // flattened state covariance matrix) and the per-iteration
  // /tmp/iter_error.txt dump (debugLogIterError) -- both previously
  // unconditional (2026-08-09 cleanup, task #149). Off by default.
```

### include/livo_recon/processing/lio_processing.h:30-50

<a id="include-livo_recon-processing-lio_processing.h-30"></a>

```
  // 2026-08-24: opt-in shadow dry-run diagnostic, built to isolate WHY
  // point_filter_num=1 (undecimated) catastrophically diverges NTU_VIRAL's
  // eee_01 compared to the production point_filter_num=3 default (see
  // docs/ -- the residual_weighting investigation above and the voxel-map
  // premature-plane-convergence mechanism in voxelmap_utils.h both bear on
  // this, but neither's fix, when tested directly, actually resolved the
  // CURRENT regression -- motivating this tool). 0 (default) = disabled,
  // zero cost: no extra decimation, no extra deskew/downsample, no shadow
  // IEKF pass. > 0 = for every frame, in addition to the REAL frame's own
  // normal processing (always using the production point_filter_num, real
  // state_/voxel_map_ updates, completely unaffected), ALSO run one full
  // non-committing shadow IEKF convergence pass using a point set decimated
  // at THIS filter number instead, against the SAME starting state and the
  // SAME current voxel map the real frame is about to use -- then discard
  // the shadow result entirely (see LioProc::runDryRunShadowPass()). This
  // isolates whether the denser point set's own residual structure (H_pp/
  // H_rr eigenstructure, frac_weak/frac_strong) explains the divergence
  // independent of any map differences that build up over time when the
  // denser filter number is used for REAL processing instead. Logged (when
  // log_debug_en is also on) as `[lio_dryrun]` lines to /tmp/lio_dryrun.txt
  // -- see runDryRunShadowPass()'s doc comment for the exact fields.
```

### include/livo_recon/processing/lio_processing.h:115-132

<a id="include-livo_recon-processing-lio_processing.h-115"></a>

```
  // density_sigma_ref (0.0 = disabled): a DIFFERENT mechanism from the
  // abandoned per-plane/per-residual weighting above -- applies a single
  // UNIFORM sigma_squared multiplier to every residual in a frame,
  // max(1, n_residuals/density_sigma_ref), instead of discounting
  // individual residuals by their own plane's group size. Motivated by the
  // 2026-08-03 posterior-confidence analysis: pfn1_ds000 vs pfn1_ds010
  // (same point_filter_num, only ds_leaf_size differs) showed nearly
  // IDENTICAL per-frame Hessian eigenvalues (H_pp_min/H_rr_min within
  // ~3-5%) but ds000's converged trace(P_PP)/trace(P_RR) were ~1.8x
  // smaller -- i.e. overconfidence wasn't from any one direction/plane
  // being over-weighted (which is what sank plane_averaged/count_weighted/
  // info_gain, each of which starved a specific position/rotation
  // direction), but from a small, uniform ~15% higher residual COUNT every
  // single frame compounding over thousands of frames. A uniform per-frame
  // scalar can't reproduce that direction-specific starvation failure mode
  // since every residual (and hence every direction) is scaled identically
  // -- it only damps the OVERALL correction strength as density rises,
  // never redistributes weight between planes/directions.
```

### include/livo_recon/processing/lio_processing.h:151-159

<a id="include-livo_recon-processing-lio_processing.h-151"></a>

```
  // 2026-08-24: REMOVED iterative_deskew/spline_k/fine_shape_offset/
  // wide_jacobian_vw/process_noise_update_mode -- the "iterative deskew"
  // Hermite-spline LIO mechanism and its three follow-on experiments.
  // Never enabled in any production config; its one validated result
  // (gyro-tangent iterative_deskew alone, eee_01 ATE 0.0261m) came with a
  // known, never-fixed regression (catastrophic divergence on 3 HILTI
  // sequences), and wide_jacobian_vw was independently confirmed
  // regressed (eee_01 0.0261m -> 0.858m). Full design, math, and removal
  // rationale: docs/removed_livo_recon_spline_deskew_2026aug24.md.
```

### include/livo_recon/processing/lio_processing.h:161-165

<a id="include-livo_recon-processing-lio_processing.h-161"></a>

```
  // Deskew params (mirrors ImuProcOptions' old sigma_r2/sigma_a2/
  // time_based_process_noise field names exactly -- loaded from the SAME
  // rosparam keys in LioProc::loadParameters(); deskewing/downsampling
  // moved here from ImuProc entirely on 2026-08-18, see
  // include/livo_recon/lio/deskew.h's module doc comment).
```

### include/livo_recon/processing/lio_processing.h:168-172

<a id="include-livo_recon-processing-lio_processing.h-168"></a>

```
  // Voxel-downsample params for the Stage-1 (deskewPoints() output)
  // downsample -- mirrors ImuProcOptions' old ds_leaf_size/ds_mode field
  // names exactly, same rosparam keys, moved here 2026-08-18 alongside
  // deskewing itself (downsample always ran ON already-deskewed points,
  // so it was never really an "IMU" concern either).
```

### include/livo_recon/processing/lio_processing.h:176-191

<a id="include-livo_recon-processing-lio_processing.h-176"></a>

```
  // T0-D (2026-08-31): scan.csv -- one row per LIO frame, for
  // scripts/analysis/consistency.py's scan-level covariates (the
  // corr.csv-only core already answers the calibration/whiteness
  // questions; this is the extra panel). Columns: scan_id (=VoxelMap::
  // frame_idx_, matching corr.csv's scan_id exactly), t (=t_abs), dt
  // (time since the previous logged scan), trP_pos/trP_vel/trP_att
  // (trace of this frame's POSTERIOR P_PP/P_VV/P_RR blocks), omega_norm/
  // acc_norm (mean |gyr|/|0.5*(acc_head+acc_tail)| over this frame's
  // mg.poses -- mg.imu_samples is already cleared by the time processLIO()
  // runs, see the .cpp call site).
  // trQdt_* (process-noise-authority columns) are NOT logged -- Q is
  // applied during ImuProc's propagation, a different module than this
  // one, and threading it through was scoped out the same way range/
  // incidence was for corr.csv (see VoxelOpts::log_consistency_covariates_
  // en's docs) -- consistency.py's Q-authority panel simply skips when
  // the column is absent. Off by default.
```

### include/livo_recon/processing/lio_processing.h:194-222

<a id="include-livo_recon-processing-lio_processing.h-194"></a>

```
  // T0-E (2026-08-31): nll.txt -- one line per frame, EVERY frame
  // (including n_residuals=0 ones -- 2026-08-31 code-audit fix: silently
  // skipping empty frames made the line count alpha-dependent, since
  // alpha changes the trajectory and hence which frames match), first
  // IEKF iteration only, same "un-relinearized prior" scope as T0-D's
  // corr.csv/scan.csv. Columns: t_abs, nll = 0.5*(n_residuals*log(2*pi) +
  // sum_log_sigma2 + sum_chi2 + EkfUpdate::nllQuadraticAndLogdet(
  // prior_cov_)) [0 when n_residuals==0], n_residuals, pivot_ratio
  // (T0-E-4, 2026-08-31: EkfUpdate::pivotRatio(), max/min |LDLT diagonal|
  // of the actual solve this frame -- NaN when n_residuals==0, same
  // staleness caveat), kalman_gain_norm (T0-F-2b, 2026-08-31:
  // EkfUpdate::kalmanGainNorm(), same staleness caveat). The log(2*pi)
  // term and the n_residuals column are BOTH 2026-08-31 code-audit fixes
  // (the term was previously dropped, safe only when every compared run
  // has the same N per frame, which is false across a q_alpha sweep since
  // alpha changes N) -- an external driver (scripts/analysis/qsens.py's
  // eventual real-EKF port) re-runs the same sequence at several
  // ImuProcOptions::q_alpha values and compares sum(nll)/sum(n_residuals)
  // (per-correspondence-normalized, NOT a raw sum(nll)) to find the alpha
  // that minimizes NLL per correspondence, instead of differentiating
  // through the update analytically.
  //
  // Known gaps, not yet fixed: dead in CombinedProc (which calls
  // processLIO() only as its own fallback, so a q_alpha sweep on a
  // combined LIO+VIO config silently yields a near-empty nll.txt); the
  // CUDA residual-accumulation path (accumulateLioResidualsCuda) returns
  // float-precision HtH/Htz, so nll.txt is not bit-reproducible between
  // CPU and GPU builds -- hold the build fixed across any alpha sweep
  // until this is addressed.
```

### include/livo_recon/processing/lio_processing.h:270-279

<a id="include-livo_recon-processing-lio_processing.h-270"></a>

```
  // Combined-mode entry point (CombinedProc, 2026-08-14): builds this
  // frame's residuals (same buildResiduals()/density_sigma_ref path as the
  // sequential estimateStateCorrection() above) and accumulates them into
  // `out` via the shared lio_accumulator -- but does NOT call
  // applyMeanUpdate/applyCovarianceUpdate itself (CombinedProc sums this
  // with VioProc::accumulateForCombined()'s output first). Returns false
  // if the voxel map is empty or no residuals were found this frame (out
  // is left untouched in that case). avg_res is the same mean-|r|
  // diagnostic estimateStateCorrection() returns, for logging/convergence
  // use by the caller.
```


## include/livo_recon/processing/vio_processing.h

### include/livo_recon/processing/vio_processing.h:22-29

<a id="include-livo_recon-processing-vio_processing.h-22"></a>

```
  // 2026-08-12: defaulted to 0.0 (never activates -- ekf_.dtheta.norm()/
  // dt.norm() are >=0 and essentially never EXACTLY 0.0 in floating point,
  // so this branch of estimateStateCorrection()'s stop condition becomes
  // unreachable dead weight at these defaults) rather than removed
  // outright -- FAST-LIVO2's current myvio has no equivalent convergence
  // criterion at all (min_diff_error alone gates iteration count there),
  // and this codebase's own field is kept only so a caller can still
  // opt back into it explicitly via config if ever wanted.
```

### include/livo_recon/processing/vio_processing.h:46-68

<a id="include-livo_recon-processing-vio_processing.h-46"></a>

```
  // Per-point WEIGHT (2026-08-04, PORTED FROM FAST-LIVO2's livo_vio A/B
  // scaffold -- see that repo's vio_proc.h/.cpp for the full ablation
  // history this default reflects) for the epipolar-line residual on a
  // point whose free (unconstrained) anchor-ray depth s0 is >=
  // anchor_split_depth (a normal, "in front of the camera" point), applied
  // DIRECTLY (weight = pos_epipolar_residual_var, not a variance requiring
  // a runtime 1/var division) unless distortion_weight_on multiplies it by
  // a per-point 1/predicted-variance factor -- see that option's doc
  // comment. Fixed base value -- NO state/pose-covariance term folded in
  // (that was this function's own bug before this port: weight =
  // 1/(J*P66*J^T + epipolar_residual_var), which (a) double-counts the
  // same pose uncertainty already accounted for via prior_cov_.inverse()
  // inside EkfUpdate::applyMeanUpdate()'s A = H_full + prior_cov.inverse()
  // -- exactly the pose_cov_in_sigma double-count LioProc's voxel residual
  // already had fixed on the LIO side, never ported to VIO until now --
  // and (b) creates a self-reinforcing feedback loop, P shrinks -> state_var
  // shrinks -> weight rises -> P shrinks further, plus the per-point
  // information Jr^T*weight*Jr = J^2/(J*P*J^T+var) cancels J's own scale
  // for near-degenerate points instead of vanishing the way a
  // no-information point should. Confirmed this session: running the OLD
  // state_var-coupled weight on eee_01 (pfn3_baseline defaults) diverged to
  // 10.93m ATE; this fixed-weight port is what's being validated to
  // replace it.
```

### include/livo_recon/processing/vio_processing.h:71-83

<a id="include-livo_recon-processing-vio_processing.h-71"></a>

```
  // Per-point WEIGHT for the epipolar-line residual on a point whose free
  // anchor-ray depth s0 is < anchor_split_depth (it would triangulate
  // behind or too close to the anchor camera) -- kept in the solve anyway,
  // at this separately-tunable (normally looser) weight, instead of being
  // dropped outright: still real epipolar-line information even though its
  // depth is untrustworthy. This fallback is unconditional (2026-08-05 --
  // see anchor_split_depth's doc comment, which replaced the old
  // use_behind_camera_penalty_epipolar toggle): set equal to
  // pos_epipolar_residual_var to make depth-clamped points
  // indistinguishable from normal ones; set to 0.0 to zero out their
  // contribution entirely (equivalent to dropping them, since a zero
  // weight contributes nothing to HtH/Htz either way). Ported default:
  // 1/32.
```

### include/livo_recon/processing/vio_processing.h:92-108

<a id="include-livo_recon-processing-vio_processing.h-92"></a>

```
  // 2026-08-12: consolidates the old, separately-tunable clamp_depth
  // (pos_/neg_epipolar_residual_var split threshold) and ray_min_depth/
  // use_cone_deadzone_epipolar/cone_radius_per_depth/cone_deadzone_hard
  // (a rejection-cone mechanism around the epipolar line, gated on a
  // SEPARATE min-depth floor) into ONE shared threshold, matching
  // FAST-LIVO2's own consolidation (see that codebase's
  // EpipolarOptions::anchor_split_depth doc comment) -- the cone-deadzone
  // mechanism itself was found strictly worse than the plain clamp_depth
  // fall-through weighting there (NTU_VIRAL production sweep:
  // bidir_cone_only 0.0297m vs bidir_clamp_bce 0.0292m) and removed
  // entirely rather than ported. A point whose free (unconstrained)
  // anchor-ray depth s0 falls below this floor is NOT excluded -- it still
  // contributes the same line residual, just at neg_epipolar_residual_
  // var's weight instead of pos_epipolar_residual_var's (see
  // computeDirectionalEpipolar). Default 0.0 (was 0.5 for the old
  // ray_min_depth) -- matches the request to have this floor alone drive
  // the pos_/neg_ split with no separate depth-validity gate underneath it.
```

### include/livo_recon/processing/vio_processing.h:116-133

<a id="include-livo_recon-processing-vio_processing.h-116"></a>

```
  // 2026-08-12: rigorous per-point distortion-weighting, ported from
  // FAST-LIVO2's EpipolarOptions::distortion_weight_mode -- simplified to
  // a plain on/off toggle here (FAST-LIVO2 also has a "cheap_bz2" middle
  // option this codebase has no equivalent bookkeeping for, so it's not
  // offered). When true, MULTIPLIES the base pos_/neg_epipolar_residual_
  // var weight by 1/Var(r), an error-propagation estimate of how much the
  // epipolar-line residual r is expected to move around from an assumed
  // weight_pixel_noise_px pixel-space Gaussian measurement noise alone
  // (propagated through both the anchor-seed and current-tracked pixels'
  // local undistortion Jacobians, AnchorPoint::uv_jacobian_seed/
  // TrackedFrame::uv_curr_uv_jacobian, into the residual's own analytic
  // Jacobian w.r.t. those two pixels) -- see epipolarLineWeightVar(). This
  // is a genuinely different signal from the depth-based pos_/neg_ split:
  // a point near the epipole (small norm_l) gets a large predicted
  // variance regardless of its depth, so it's downweighted here even when
  // its depth looks perfectly normal. false (default) reproduces the
  // existing fixed-weight behavior exactly (weight = pos_/neg_
  // epipolar_residual_var, unchanged).
```

### include/livo_recon/processing/vio_processing.h:149-159

<a id="include-livo_recon-processing-vio_processing.h-149"></a>

```
  // ── Diagnostic logging (2026-08-12, ported from FAST-LIVO2's myvio --
  // see VioProcOptions there for the originals). Replaces the old
  // log_debug_en/debugLogVio mechanism (a compile-time-gated dump to
  // /tmp/vio.txt that was permanently dead -- kDebugVioLoggingEnabled was
  // hardcoded false -- since task #149's cleanup) with properly runtime-
  // toggleable options sharing one log_path. NOT ported:
  // log_parallax_histogram -- livo_recon's epipolar residual has no
  // parallax-gating/parallax-angle computation at all (unlike FAST-LIVO2's
  // ParallaxGateOptions machinery), so there is no parallax angle to
  // histogram here; porting the flag alone with nothing behind it would
  // be a no-op that looks configurable but silently isn't. ──────────────
```

### include/livo_recon/processing/vio_processing.h:200-203

<a id="include-livo_recon-processing-vio_processing.h-200"></a>

```
  // G6 (2026-09-01): VIO-side correspondence log (vio_corr.csv), mirroring
  // voxelplane.cpp's corr.csv nu/S columns so both channels are scored by
  // the same statistic -- see consistency_log.h's doc comment for why
  // combined mode had no VIO-side consistency instrument before this.
```

### include/livo_recon/processing/vio_processing.h:217-225

<a id="include-livo_recon-processing-vio_processing.h-217"></a>

```
  // Tried and REMOVED (2026-08-05): a soft, continuous per-point
  // downweighting by 3D ray-consistency (anchor ray through uv_seed vs.
  // current-camera ray through uv_tracked, weight scaled by
  // 1/(1+(ray_gap/ray_gap_ref)^2)). Consistently mildly harmful once every
  // other option was correctly re-ablated post-undistortion-fix, both on
  // single-sequence (eee_01) and full 9-sequence-suite testing (see
  // FAST-LIVO2's NTU_VIRAL.yaml livo_vio: promotion comment for the
  // confirming numbers) -- removed from the code entirely rather than left
  // off by default.
```

### include/livo_recon/processing/vio_processing.h:228-257

<a id="include-livo_recon-processing-vio_processing.h-228"></a>

```
// Correspondence-driven VIO update: maintains a persistent set of tracked
// points (see vio/tracker.h, ctx_.tracker) via whichever native backend
// (CoTracker, TAPNext, or Track-On) is configured, anchored to a fixed grid
// of Shi-Tomasi seed pixels (see AnchorPoint) rather than requiring LIDAR
// depth. Every tracked point contributes a 1D epipolar-line residual (see
// computeEpipolarResidualAndJacobian) between its seed-time pixel and its
// currently-tracked pixel, via the same EkfUpdate machinery.
//
// This replaces an earlier design that instead anchored each tracked point
// to a real LIDAR 3D point and ran a metric-scale reprojection residual
// against it -- ported from FAST-LIVO2 (livo_vio/vio_proc.{h,cpp}), where a
// point-to-point reprojection residual against LIDAR-anchored 3D points was
// tried and removed: on eee_01 its per-iteration corrections were
// uncorrelated with (mixed config) or anti-correlated with (LIDAR-only
// config) actually moving the state closer to ground truth, unlike the
// epipolar residual's clearly corrective behavior.
//
// 2026-08-04: the residual/weighting model (everything in VioProcOptions
// above except max_iterations/min_norm_dtheta/min_norm_dt/min_diff_error/
// max_avg_error, which were already correct and unchanged) was replaced
// wholesale, ported from FAST-LIVO2's livo_vio A/B scaffold's own
// extensively-ablated production default -- see each option's doc comment
// above for what specifically changed and why. The EKF core (EkfUpdate,
// ekf.h) was NOT touched -- it's shared with LioProc, already carries this
// session's earlier validated fixes, and was confirmed correct; only the
// per-point residual/weight computation feeding into it was replaced.
// Motivated by: running the pre-port code on eee_01 (pfn3_baseline and
// pfn1_ds000 production defaults, VIO enabled) diverged to 10.93m and
// 1.51m ATE respectively -- see git history/session notes for the pre-port
// version if a reference is ever needed.
```

### include/livo_recon/processing/vio_processing.h:274-282

<a id="include-livo_recon-processing-vio_processing.h-274"></a>

```
  // Combined-mode entry point (CombinedProc, 2026-08-14): builds this
  // frame's residuals via the shared accumulator_ (same options/geometry
  // path as the sequential solveSystem() below) and fills `out` -- does
  // NOT call out.applyMeanUpdate()/applyCovarianceUpdate() itself
  // (CombinedProc sums this with LioProc::accumulateForCombined()'s output
  // first). Returns false if there's no image/tracker/valid frame this
  // call, or no valid residuals were found (out/avg_error untouched).
  // avg_error is the same per-frame diagnostic solveSystem() has always
  // computed, for CombinedProc's own VIO-fit-rejection check.
```

### include/livo_recon/processing/vio_processing.h:313-317

<a id="include-livo_recon-processing-vio_processing.h-313"></a>

```
  // Owns the actual residual/Jacobian/HtH-Htz math (2026-08-14, extracted
  // to vio/vio_accumulator.h so CombinedProc can call the exact same code
  // -- see that header's doc comment). solveSystem()/accumulateForCombined()
  // both just adapt VioProcOptions into a VioAccumulateOptions and call
  // accumulator_.accumulate().
```


## include/livo_recon/utils/algo/ekf.h

### include/livo_recon/utils/algo/ekf.h:9-13

<a id="include-livo_recon-utils-algo-ekf.h-9"></a>

```
    // 2026-08-24: reverted to fixed 6x6/6x1 (R,P) -- was Eigen::MatrixXd/
    // VectorXd dynamically sized to also support a 12-dim (R,P,V,W) "wide"
    // mode, which existed solely for the now-removed iterative-deskew
    // mechanism's wide_jacobian_vw option (confirmed regressed, never
    // usable). See docs/removed_livo_recon_spline_deskew_2026aug24.md.
```

### include/livo_recon/utils/algo/ekf.h:114-150

<a id="include-livo_recon-utils-algo-ekf.h-114"></a>

```
    // T0-E (2026-08-31): the quadratic+log-determinant terms of this
    // frame's batch-update NLL, i.e. everything EXCEPT the purely
    // residual-level pieces (sum_i log(sigma_i^2), sum_i r_i^2/sigma_i^2,
    // and n_residuals*log(2*pi) -- the Gaussian normalization term,
    // 2026-08-31 code-audit fix: dropping it is harmless comparing runs
    // with the same n_residuals per frame, but n_residuals varies both
    // per-frame and with q_alpha, so it must be included when comparing
    // totals ACROSS runs) the caller already has from iterating
    // residuals_ -- see LioProc::estimateStateCorrection()'s log_nll_en
    // block for how the pieces combine into the full per-frame NLL, and
    // LioProcOptions::log_nll_en's doc comment for the frame-accounting
    // fix (log every frame including n_residuals=0 ones, never silently
    // skip) that makes runs at different q_alpha comparable at all.
    //
    // Derivation: for a linear-Gaussian batch update with prior x~N(x0,P0)
    // and independent per-residual noise r_i~N(0,sigma_i^2), the marginal
    // (prior-predictive) NLL of the batch is NLL = 1/2*[log det(S_full) +
    // r^T S_full^-1 r] where S_full = H_full P0 H_full^T + R_full
    // (R_full = diag(sigma_i^2)) -- an N_CORR x N_CORR matrix, infeasible
    // to form directly. Two standard identities avoid ever forming it:
    //   det(S_full) = det(R_full) * det(P0) * det(A),  A = P0^-1 + HtH
    //   r^T S_full^-1 r = r^T R_full^-1 r  -  Htz^T A^-1 Htz
    // (HtH = H_full^T R_full^-1 H_full, Htz = H_full^T R_full^-1 r -- both
    // exactly this class's own HtH/Htz accumulators, and A is exactly
    // applyMeanUpdate()'s own A). So NLL = 1/2*[ sum_i log(sigma_i^2) +
    // sum_i r_i^2/sigma_i^2 + log det(P0) + log det(A) - Htz^T A^-1 Htz ]
    // -- this function returns log det(P0) + log det(A) - Htz^T A^-1 Htz;
    // the caller adds the two residual-level sums and halves the total.
    //
    // Computed from a FRESH LDLT of A/prior_cov, independent of
    // applyMeanUpdate()'s own (called separately, doesn't require this
    // to run before/after it, doesn't touch last_H_full_/last_K1_/
    // state_ at all -- purely read-only over HtH/Htz/prior_cov). "Frozen
    // Jacobian" scope note: this is the NLL of the CURRENT frame's prior
    // (state_->cov() before this frame's own correction) against its
    // OWN residuals -- exactly the T0-D-style "first-iteration,
    // un-relinearized" quantity, not a converged-update NLL.
```

### include/livo_recon/utils/algo/ekf.h:159-163

<a id="include-livo_recon-utils-algo-ekf.h-159"></a>

```
      // 2026-08-31 code-audit fix: a numerically non-PD prior/A (e.g. a
      // pathological q_alpha sweep value) previously wrote a silent NaN
      // into nll.txt instead of failing loudly -- NaN is easy to miss in
      // a summed column and would masquerade as "this alpha is somehow
      // infinitely good/bad" rather than "the covariance broke".
```

### include/livo_recon/utils/algo/ekf.h:180-191

<a id="include-livo_recon-utils-algo-ekf.h-180"></a>

```
    // T0-E-4 (2026-08-31): condition-number proxy for the LAST applyMeanUpdate()/
    // applyCovarianceUpdate() solve (A = H_full + prior_cov^-1), i.e. the actual
    // EKF solve path -- NOT nllQuadraticAndLogdet()'s own separate fresh LDLT,
    // which is a diagnostic-only decomposition. max(|D_ii|)/min(|D_ii|) from the
    // LDLT's diagonal is a cheap proxy for cond(A) (exact for a diagonal A;
    // an underestimate in general, but tracks the same order of magnitude and
    // needs no extra decomposition). Same staleness caveat as HtH/Htz: only
    // meaningful when applyMeanUpdate() actually ran this iteration (n_res>0).
    // T0-F-2b (2026-08-31): ||K|| for the LAST applyMeanUpdate() call --
    // Frobenius norm of K1_cols (last_K1_'s R-column block), the gain
    // that pre-multiplies Htz in applyMeanUpdate()'s solution formula
    // (see its own comment). Same staleness caveat as pivotRatio().
```


## include/livo_recon/utils/algo/omp_utils.h

### include/livo_recon/utils/algo/omp_utils.h:24-55

<a id="include-livo_recon-utils-algo-omp_utils.h-24"></a>

```
// Call ONCE, as the very first statement in main() -- before ros::init(),
// before constructing anything that might touch LibTorch/AOTInductor (the
// CoTracker backend's model loading, even in cache-replay mode where the
// model is loaded but never invoked) -- to fix a confirmed run-to-run
// non-determinism (2026-08-16): with OMP_NUM_THREADS left unset,
// omp_get_max_threads() (and therefore cappedOmpThreads(), and therefore
// the actual thread count a "#pragma omp parallel for" runs with) was
// observed to vary BETWEEN otherwise-identical runs of the same
// offline+cache-replay config on the same host -- most likely LibTorch's
// own OpenMP thread-pool initialization racing with or overriding the
// effective global thread count before this codebase's own parallel
// regions ever run.
//
// Two independent fixes are combined here, deliberately:
// 1. VioAccumulator::accumulate() and accumulateLioResiduals() (2026-08-16)
//    were changed to reduce HtH/Htz/err in fixed POINT/residual-index order
//    (not per-thread-partial-sum order) -- this makes their own numeric
//    result independent of thread count entirely, confirmed: reruns at
//    OMP_NUM_THREADS=4 vs 16 now produce byte-identical HtH/Htz for the
//    same frame. This is the real fix for the two call sites that matter
//    most (the ones feeding the EKF solve).
// 2. This function additionally PINS the thread count itself, as defense
//    in depth for any other OMP-parallel region in this codebase that
//    hasn't been (or in the future isn't) given the same point-indexed
//    treatment -- an unset OMP_NUM_THREADS otherwise leaves the actual
//    thread count to whatever LibTorch/the OS picked at that moment, which
//    is its own source of nondeterminism even where the reduction order no
//    longer matters (e.g. wall-clock-timing-sensitive code, if any exists).
//    Respects OMP_NUM_THREADS if the environment explicitly sets it (so a
//    deliberate override, e.g. for a benchmark or this fix's own
//    verification testing, still works); defaults to 16 (matching
//    cappedOmpThreads()'s own cap) otherwise.
```


## include/livo_recon/utils/data/data_queues.h

### include/livo_recon/utils/data/data_queues.h:60-69

<a id="include-livo_recon-utils-data-data_queues.h-60"></a>

```
  // 2026-08-24: opt-in LIO shadow dry-run diagnostic (CbkProcOptions::
  // dry_run_point_filter_num) -- a SECOND lidar queue, populated in
  // parallel by CbkProc's lidar callback from the SAME raw scan decimated
  // at a different point_filter_num, popped in lockstep (same max_time
  // window) by the same MeasureGroup-assembly call that pops the primary
  // queue. Entirely independent state (its own mutex/latest-time
  // bookkeeping) from lidar_queue above -- correctness only depends on
  // both being popped with the same `max_time`, not on any cross-queue
  // synchronization. Empty/unused (zero cost) unless dry_run_point_filter_
  // num > 0.
```


## include/livo_recon/utils/data/measures.h

### include/livo_recon/utils/data/measures.h:31-39

<a id="include-livo_recon-utils-data-measures.h-31"></a>

```
  // LIO Processing -- the single point set used for both residual matching
  // and voxel-map insertion (VoxelMap::updateMap()). Built from
  // lidar_points via point_filter_num (keep-every-Nth, applied at
  // ingestion -- CbkProc) then ds_leaf_size (voxel-grid downsample,
  // applied in LioProc::processLIO() after deskewing -- moved from
  // ImuProc 2026-08-18) -- both no-ops at their extreme settings
  // (point_filter_num=1, ds_leaf_size=0), so there's one point-processing
  // pipeline, not a separate decimated/full-resolution pair. Set once
  // (LioProc::deskewAndDownsample()) and never touched again this frame.
```

### include/livo_recon/utils/data/measures.h:42-44

<a id="include-livo_recon-utils-data-measures.h-42"></a>

```
  // 2026-08-24: REMOVED points_orig/points_orig_offset/spline_anchor --
  // state for the removed iterative-deskew Hermite-spline mechanism. See
  // docs/removed_livo_recon_spline_deskew_2026aug24.md.
```

### include/livo_recon/utils/data/measures.h:46-53

<a id="include-livo_recon-utils-data-measures.h-46"></a>

```
  // 2026-08-24: opt-in shadow dry-run diagnostic (LioProcOptions::
  // dry_run_point_filter_num) -- a SECOND raw point set, decimated at a
  // different point_filter_num than the primary `points` above (see
  // CbkProc's ingestion), used ONLY by LioProc::runDryRunShadowPass() to
  // run a non-committing shadow IEKF pass for direct comparison against
  // the real frame's residual structure. Empty (no-op, zero cost) unless
  // dry_run_point_filter_num > 0. Never touched by anything else --
  // never inserted into the voxel map, never affects `points`/state_.
```


## include/livo_recon/utils/log/consistency_log.h

### include/livo_recon/utils/log/consistency_log.h:2-17

<a id="include-livo_recon-utils-log-consistency_log.h-2"></a>

```
// G6 blocker (2026-09-01). nll.txt and scan.csv are written by two static
// helpers in the anonymous namespace of lio_processing.cpp, so they are
// reachable ONLY from LioProc::processLIO(). CombinedProc::processCombined()
// never calls processLIO() except on the VIO-rejection rollback path, and
// never references log_nll_en or log_consistency_scan_en at all.
//
// Consequence, stated plainly: with combined/enable true, this project has NO
// consistency instrument. Every NIS number in the register is LIO-only. The
// question "are the LiDAR and visual channels calibrated consistently
// relative to each other" -- which is the measured argument for or against
// collapsing the two filters -- has not been askable, and unifying the
// filters cannot be evaluated until it is.
//
// This header hoists the two writers out of that anonymous namespace so both
// processors can call them. No behaviour change on the LIO path: same
// filenames, same columns, same first-call truncate semantics.
```


## include/livo_recon/utils/log/param_warn.h

### include/livo_recon/utils/log/param_warn.h:8-22

<a id="include-livo_recon-utils-log-param_warn.h-8"></a>

```
// Every *ProcOptions field in this codebase is read via pnh.param<T>(key,
// out, default) -- if `key` was never set (typo'd config key, a rosparam
// that got renamed in code but not in every checked-in yaml, an option
// that's simply missing from a particular dataset's config), pnh.param()
// silently falls back to its C++ default with zero indication anything
// was missing. Confirmed this session (2026-08-16): a config-loading
// override placed under the wrong yaml section (e.g. common: instead of
// outputs:) silently no-op'd with no error, and separately, a hardcoded
// yaml value shadowing a just-changed C++ default silently kept the old
// behavior -- both would have been caught immediately by a fallback
// warning. paramWarn<T>() is a drop-in replacement for pnh.param<T>() that
// additionally logs a ROS_WARN when the key isn't found on the param
// server at all (i.e. every *.yaml actually loaded by this launch didn't
// set it), before falling back to `default_value` exactly like pnh.param()
// always has.
```


## include/livo_recon/utils/map/voxelmap_utils.h

### include/livo_recon/utils/map/voxelmap_utils.h:66-87

<a id="include-livo_recon-utils-map-voxelmap_utils.h-66"></a>

```
  // T0-D (2026-08-31): the frame-level context needed to compute this
  // correspondence's full innovation covariance S = H P- H^T + R for
  // corr.csv (see VoxelOpts::log_consistency_corr_en) -- NOT used by any
  // real accept/reject or EKF-weighting logic, which never needed the
  // H P- H^T term (this codebase's IEKF is a batch-WLS normal-equation
  // solve, not a per-correspondence sequential Kalman update -- see
  // ekf.h's applyMeanUpdate()). Populated unconditionally (cheap: a V3D
  // copy plus two small matrices set ONCE per frame and copied per point)
  // by LioProc::buildResiduals() so VoxelPlane::computeResidual() can
  // reconstruct this point's Jacobian and reach into the frame's prior
  // covariance -- without touching VoxelPlane's own constructor/call
  // chain, since WorldPointCov already flows unmodified through the
  // whole findPlaneResidual() chain. body_point: this point in
  // SENSOR/body frame (pre-toWorld) -- combined with plane_.normal (only
  // known inside VoxelPlane) via rot_transpose to reconstruct
  // point_cross_normal exactly as LioProc::buildResiduals() does for its
  // own HtH accumulation. rot_transpose: state_->rot().transpose() at
  // this frame's prior. prior_cov_rp: the frame's FIXED prior_cov_'s
  // [R,P]x[R,P] 6x6 block (R,P are contiguous starting at
  // StateGroup::idxR(), see ekf.h) -- the only sub-block a point-to-plane
  // residual's Jacobian (nonzero only in R,P columns) can have any
  // quadratic-form interaction with.
```

### include/livo_recon/utils/map/voxelmap_utils.h:177-181

<a id="include-livo_recon-utils-map-voxelmap_utils.h-177"></a>

```
  // Gates VoxelNode::debugLogPlaneInit()'s per-voxel-first-plane-fit dump
  // (/tmp/plane_init.txt) and VoxelPlane's noise-floor dump
  // (/tmp/noise_floor.txt) -- both were previously unconditional (2026-08-09
  // cleanup, task #149: unlike most of this codebase's debug logs, these
  // had no gate at all). Off by default.
```

### include/livo_recon/utils/map/voxelmap_utils.h:184-195

<a id="include-livo_recon-utils-map-voxelmap_utils.h-184"></a>

```
  // T3-0d (2026-08-30): does plane_var_ have enough authority in the
  // residual variance budget to move ATE at all? Gates two debug dumps,
  // off by default. /tmp/variance_shares.txt (one line per accepted
  // correspondence, from VoxelPlane::computeResidual()): sigma_diag_squared,
  // plane_var_term, and plane_var_term's share of their sum -- the direct
  // measurement T3-0/T3-0b/T0-B-2/T0-B-3 never took, all four having
  // turned knobs that act on plane_var_ and reported only ATE.
  // /tmp/plane_fit_stats.txt (one line per VoxelPlane::update() call that
  // used weights): N (points/bins seen), effective sample size N_eff =
  // (sum w)^2 / sum(w^2), and trace(plane_var_) -- lets a bin_size_fraction
  // or bin_weight_mode sweep be read against how much N_eff actually moved,
  // instead of inferring it from ATE alone.
```

### include/livo_recon/utils/map/voxelmap_utils.h:198-213

<a id="include-livo_recon-utils-map-voxelmap_utils.h-198"></a>

```
  // use_disc_distance (2026-08-04, tried and REMOVED): treated each
  // VoxelPlane as a finite disc (radius_ around center) rather than an
  // infinite plane, computing r as the true 3D point-to-disc distance
  // (Pythagorean combination of perpendicular + excess in-plane distance
  // beyond radius_, with the residual's gradient direction rotating
  // toward the in-plane radial direction for points past the edge) instead
  // of today's plane-only r gated by a hard max_radius*radius_ cutoff.
  // Motivated by nya_02's cov_acc collapse investigation (a genuine
  // physical argument -- the flat-plane assumption is shakiest right at a
  // patch's edge) but regressed badly on nya_02 across every cov_acc
  // tested, including ones that were otherwise clean without it: 0.01
  // (0.81m -> 8498.7m), 0.1 (0.033m -> 0.200m), gradually-compounding not
  // an instant blowup. Consistent with this session's broader pattern --
  // every attempt to make residual weighting more "statistically honest"
  // (info_gain, Woodbury, cross_frame_reuse, this) has hurt real-world
  // tracking robustness rather than helped it.
```

### include/livo_recon/utils/map/voxelmap_utils.h:215-229

<a id="include-livo_recon-utils-map-voxelmap_utils.h-215"></a>

```
  // cross_frame_reuse_ref (2026-08-03, tried and REMOVED): a per-plane
  // discount based on VoxelPlane::match_streak_ (how many consecutive
  // frames had matched at least one residual against a plane), motivated
  // by trace(P_PP)'s anti-correlation with error persisting over a ~1-2s
  // (many-frame) window rather than a single-frame spike -- unlike every
  // other mechanism tried this investigation (density_sigma_ref, the
  // abandoned plane_averaged/count_weighted/info_gain/
  // woodbury_plane_correction), this was the only one that looked at
  // cross-frame (not within-frame) correlation. It DID change the
  // qualitative oscillation character (at cross_frame_reuse_ref=2, eee_01:
  // mean-crossings of trace(P_PP) dropped ~60%, 113->45, and coefficient
  // of variation dropped ~23%, 6.09->4.67) but produced no meaningful ATE
  // improvement across ref values 2/5/10/20 (0.0256/0.0254/0.0254/0.0252m
  // vs 0.0254m baseline, all within noise) -- removed as not worth the
  // added complexity given the effect size.
```

### include/livo_recon/utils/map/voxelmap_utils.h:348-361

<a id="include-livo_recon-utils-map-voxelmap_utils.h-348"></a>

```
  // Controls plane_var_'s own residual-weighting propagation -- a
  // DIFFERENT quantity from pose_cov_in_sigma above (which controls only
  // gate()'s sigma_diag_squared). Applies to BOTH plane_fit_mode paths:
  // the PCA Jmin loop (VoxelPlane::update()) and, since 2026-08-24 pass4
  // (see docs/debiased_voxel_plane_fit_2026aug24.md), the debiased path's
  // own Scov_/V_/W_ accumulators too -- initially scoped PCA-only, but
  // live diagnostics showed plane_var_ blowups (traces >900,000) driven
  // by pos_cov's own magnitude for fresh/small-N voxels, independent of
  // the denom1/denom2 eigengap guard, so debiased mode needs this
  // exclusion option too. "combined" (default, unchanged) -- sensor_cov+
  // pos_cov, the full noise budget. "sensor_only" -- excludes pos_cov
  // from this propagation entirely (pos_cov, poseCovAt()'s range^2
  // lever-arm term, is documented to "dwarf true sensor noise for far
  // points" -- the same reason pose_cov_in_sigma defaults false).
```

### include/livo_recon/utils/map/voxelmap_utils.h:364-384

<a id="include-livo_recon-utils-map-voxelmap_utils.h-364"></a>

```
  // Controls VoxelNode::buildBinReps()'s two per-bin weight vectors,
  // consumed by VoxelPlane::update() as w_i/weight_sum (always self-
  // normalizing, so only the RELATIVE weighting across bins matters, not
  // the absolute scale). "count" -- weight = bin.count, i.e. the sum is
  // the inverse participation ratio over bins, an effective sample size
  // bounded above by the bin count J but shaped by point density within
  // bins (the Kish effective-N). "uniform" -- weight = 1 for every
  // occupied bin regardless of how many points landed in it, i.e. true
  // occupancy counting (effective sample size is exactly J). Only takes
  // effect when use_bins is true (see VoxelNode::useBins()).
  //
  // Split into _fit and _var 2026-08-30 (T3-0c): before this, one
  // "bin_weight_mode" fed BOTH the plane fit (center/covariance/normal,
  // in VoxelPlane::update()'s weighted mean/covariance) and the
  // plane_var_ Jacobian in the same call -- so T3-0's "uniform" arm moved
  // the ESTIMATE and its UNCERTAINTY together, the identical confound T3-0
  // was built to strip out of T0-B-2. T3's actual proposal only reweights
  // the uncertainty term (its directional M_cov substitutes for this
  // scalar case) -- (fit=count, var=count) reproduces pre-split behavior;
  // (fit=count, var=uniform) is the arm that actually tests T3's premise
  // in isolation.
```

### include/livo_recon/utils/map/voxelmap_utils.h:388-396

<a id="include-livo_recon-utils-map-voxelmap_utils.h-388"></a>

```
  // VoxelNode's PointBin accumulator bin size, as a fraction of that
  // node's own voxel extent (opts->voxel_size / 2^layer) -- see
  // VoxelNode::density_weight_leaf_. Was a hard-coded
  // kDensityWeightLeafFraction constant chosen only to bound refit cost;
  // exposed as a config option 2026-08-30 (T3-0b) since it also sets the
  // effective sample size (occupied-bin count J) that bin_weight_mode
  // above operates over, and had never been swept. Smaller -> finer bins,
  // more of them (J closer to N, less aggregation, higher refit cost).
  // Larger -> coarser bins, fewer of them (J smaller, more aggregation).
```

### include/livo_recon/utils/map/voxelmap_utils.h:399-409

<a id="include-livo_recon-utils-map-voxelmap_utils.h-399"></a>

```
  // Whether VoxelNode uses the PointBin accumulator (bins_) instead of
  // raw points_ storage -- see VoxelNode::useBins(). Split out from
  // convergence_mode 2026-08-30 (T0-B-4): before this, "always_update"
  // implied binning as a side effect (binning exists to bound refit cost
  // for a voxel that never locks), which meant every prior experiment
  // that varied convergence_mode to turn binning "on/off" also changed
  // whether the voxel could converge and freeze -- two effects in one
  // knob. Defaults false (raw points_, matching every non-always_update
  // config's actual historical behavior); a config that wants the OLD
  // "always_update implies binning" behavior must set this explicitly
  // alongside convergence_mode: "always_update".
```

### include/livo_recon/utils/map/voxelmap_utils.h:412-425

<a id="include-livo_recon-utils-map-voxelmap_utils.h-412"></a>

```
  // T0-D (2026-08-31): corr.csv -- one row per point-to-plane
  // correspondence, BEFORE the outlier (sigma_num_squared) gate, for
  // offline filter-consistency diagnosis (scripts/analysis/consistency.py
  // -- NIS/whiteness/Q-vs-R triage, no ground truth needed). Core columns
  // only: scan_id (=this frame's VoxelMap::frame_idx_), nu (=r), S (=full
  // residual variance, sensor[+pose]+plane_var_term), gated (0/1 -- did
  // this correspondence pass the chi2 test, logged either way, unlike
  // variance_shares.txt which only ever sees accepted ones -- the
  // register's own note is that gating biases NIS DOWN by truncating the
  // upper tail, so scoring must see the pre-gate population). Deliberately
  // split from log_variance_shares_en (a different, narrower log already
  // in production use) and from log_consistency_covariates_en below, so a
  // run can log the cheap core NIS columns without paying for the extra
  // per-point breakdown. Off by default.
```

### include/livo_recon/utils/map/voxelmap_utils.h:428-445

<a id="include-livo_recon-utils-map-voxelmap_utils.h-428"></a>

```
  // Extra covariate columns appended to log_consistency_corr_en's core
  // row: S_sensor, S_pose (gate()'s two independent variance sources,
  // logged separately regardless of pose_cov_in_sigma), S_plane_tilt,
  // S_plane_d (plane_var_term's quadratic form split by J_nq's tilt vs. d
  // components, off-diagonal tilt/d cross-terms divided evenly between the
  // two -- S_plane_tilt+S_plane_d reconstructs plane_var_term exactly), N
  // (this plane's total accumulated point count), J (occupied bin count if
  // this plane is bin-fit, else 0), aniso (eigen_values_(2)/
  // eigen_values_(1), in-plane coverage anisotropy -- T3's premise
  // directly: consistency.py's panel 5 bins NIS by this), lambda0
  // (eigen_values_(0), T0-G 2026-08-31 -- the plane fit's own is_plane_
  // threshold quantity; T8-0b's/T8-d's outcome variable). range/incidence
  // are NOT logged -- computing them needs the sensor's world position,
  // which isn't available at VoxelPlane::computeResidual()'s call depth
  // without threading it through the whole MapBackend virtual interface;
  // consistency.py degrades gracefully (skips that one covariate) when the
  // column is absent, so this was scoped out rather than done partially.
  // No-op unless log_consistency_corr_en is also true. Off by default.
```

### include/livo_recon/utils/map/voxelmap_utils.h:448-456

<a id="include-livo_recon-utils-map-voxelmap_utils.h-448"></a>

```
  // 14c (2026-09-01): 1 = a corr.csv row for every candidate (today's
  // behaviour, ~1GB/job with covariates on). N = every Nth. Level
  // statistics (mean NIS, accept fraction, dropped-by-ablation count) are
  // UNAFFECTED by this -- they come from corr_scan.csv, an exact per-scan
  // aggregate updated for every candidate regardless of stride (see
  // debugAccumConsistencyCorr()). corr.csv itself is for distributional
  // questions only (percentiles, decile cuts), which tolerate subsampling.
  // Keep this PRIME: LiDAR returns arrive in ring/azimuth order, so a round
  // stride (50, 64, 100) can alias with the beam count.
```

### include/livo_recon/utils/map/voxelmap_utils.h:459-476

<a id="include-livo_recon-utils-map-voxelmap_utils.h-459"></a>

```
  // T8-a (2026-08-31): "disc" (default) -- today's isotropic admission
  // test, range_dis > max_radius*radius_ with radius_ = sqrt(lambda2) (the
  // LARGEST in-plane eigenvalue) -- a sliver sampled 1m along one axis and
  // 3cm along the other admits correspondences 3m out along the axis it
  // barely observed, since the disc radius is set by the axis with the
  // MOST support. "ellipse" -- Mahalanobis ellipse of the fit's own
  // sampling, using the SAME x_normal_/y_normal_ basis J_nq already uses
  // (no new geometry): m2 = u1^2/lambda2 + u2^2/lambda1, gated against
  // max_radius^2. Bit-identical to "disc" only in the degenerate case
  // lambda1==lambda2; otherwise a real behavior change, so it must be
  // opted into explicitly.
  // "ellipse_area_matched" (2026-09-01): "ellipse" above is strictly
  // CONTAINED in the disc (admits only sqrt(lambda1/lambda2) of its area),
  // so an ellipse-vs-disc comparison confounds shape with admitted-area --
  // this mode rescales the ellipse's threshold by sqrt(lambda2/lambda1)
  // (capped at 4x, see gate()'s AUDIT comment) so the admitted AREA matches
  // the disc's, isolating the shape change the coverage hypothesis is
  // actually about.
```

### include/livo_recon/utils/map/voxelmap_utils.h:479-499

<a id="include-livo_recon-utils-map-voxelmap_utils.h-479"></a>

```
  // T3-0e (2026-08-31): test T3's DIRECTIONAL premise directly -- drop
  // correspondences from planes whose occupancy anisotropy (see
  // VoxelPlane::occupancyAnisotropy()) is judged "bad" by one of two
  // rules, applied at gate() time (a genuine drop from the residual, not
  // just a logged diagnostic):
  //   "none" (default): no dropping, bit-identical to today.
  //   "top": drop if occupancyAnisotropy() > occ_aniso_drop_threshold.
  //     The threshold must be supplied externally (occ_aniso_drop_threshold
  //     has no sensible default) -- computed from a baseline pass's own
  //     occ_aniso distribution (e.g. its 90th percentile, for a "top
  //     decile" cut). Leaving it at -1 (a value no real anisotropy can be
  //     below) makes "top" mode a no-op, matching "none" -- a safe default
  //     rather than an easy-to-miss misconfiguration.
  //   "random": drop with probability occ_aniso_drop_fraction, decided
  //     once per PLANE (not per correspondence) via a fixed-seed hash of
  //     plane_.center -- deterministic and reproducible, but only
  //     approximately size-matched to "top" mode's actual drop count (in
  //     EXPECTATION, not exact count) since this codebase has no
  //     first-class per-plane identity to do an exact two-pass match
  //     against. Document this approximation plainly if this arm's own
  //     result is ever quoted.
```

### include/livo_recon/utils/map/voxelmap_utils.h:505-516

<a id="include-livo_recon-utils-map-voxelmap_utils.h-505"></a>

```
  // Code-review fix, 2026-08-31: occupancyAnisotropy() returns -1.0
  // (undefined) when fewer than 3 cells are occupied -- typically a plane
  // whose normal/anchor is still settling. Under "top" mode, -1.0 used to
  // silently never exceed occ_aniso_drop_threshold, so undefined planes
  // were ALWAYS kept -- systematically protecting the least-converged
  // planes, plausibly the ones the ablation most wants to be able to
  // remove. false (default): undefined counts as "keep" (conservative,
  // matches the old accidental behavior, now a deliberate choice). true:
  // undefined counts as "top decile" (drop). Either way, the count of
  // undefined-and-therefore-policy-decided planes is now logged (corr.csv
  // gains a dropped_by_ablation column) so this bucket's size is visible,
  // not just its existence.
```

### include/livo_recon/utils/map/voxelmap_utils.h:519-537

<a id="include-livo_recon-utils-map-voxelmap_utils.h-519"></a>

```
  // ------------------------------------------------------------------
  // T8-b (2026-09-01): the three plane-confidence terms, each on its own
  // switch so the grid can turn them on and off independently. A
  // point-to-plane update can be wrong for three separable reasons --
  // the points were noisy, the points were redundant, the points did not
  // cover the plane -- and livo_recon has had a partial answer to each
  // that could not compose. These two flags supply the missing two.
  //
  // Both are EXACTLY behaviour-preserving when off, and the coverage term
  // is additionally an identity when coverage is isotropic, so a null
  // result is distinguishable from an inert switch: see the occ_cells
  // and plane_conf_factor columns in corr.csv.
  //
  // redundancy: plane_var_ scales as 1/N. On the pca path the bin
  // weights already discount repeated returns; on the debiased path
  // N_acc_ counts every redundant return, so debiased is over-confident
  // by the redundancy factor precisely where binning exists to prevent
  // it. Replace the raw count with a coverage-derived effective sample
  // size (occupied 8x8 cells) and inflate by the ratio.
```

### include/livo_recon/utils/map/voxelmap_utils.h:550-554

<a id="include-livo_recon-utils-map-voxelmap_utils.h-550"></a>

```
  // T1 (2026-09-01): refitDebiased() rejects a fit whose eigengap
  // denominators are within eps of zero; update() -- the pca path -- has
  // NO such guard and will happily divide plane_var_ by an arbitrarily
  // small denom1. That asymmetry sits underneath every pca-vs-debiased
  // comparison in the register. On: apply the identical guard to both.
```

### include/livo_recon/utils/map/voxelmap_utils.h:557-567

<a id="include-livo_recon-utils-map-voxelmap_utils.h-557"></a>

```
  // T0-G (2026-08-31): diagnostic-only. 0 (default) -- no-op, today's
  // behavior unchanged. Nonzero -- deterministically shuffle each frame's
  // point order (std::mt19937 seeded from this value XOR'd with the frame
  // index, so every frame gets a distinct but reproducible permutation)
  // immediately before VoxelMap::updateMap()'s bucket-build step. Point
  // order affects nothing SEMANTICALLY (a voxel's accumulated sums are
  // order-independent in exact arithmetic), but floating-point summation
  // is order-dependent -- this exists purely to measure how much that
  // last-bit-level nondeterminism can move ATE, i.e. this register's first
  // error bar. See VoxelMap::updateMap()'s call site for the actual
  // shuffle.
```

### include/livo_recon/utils/map/voxelmap_utils.h:570-575

<a id="include-livo_recon-utils-map-voxelmap_utils.h-570"></a>

```
  // T0-F-2b (2026-08-31): logs frame_stats.txt (t, frame_idx,
  // denom_rejected_count, max_plane_var_trace) bracketing each
  // VoxelMap::updateMap() call -- see voxelPlaneFrameStats{Reset,Read}()
  // in voxelplane.h for what's aggregated and why here specifically
  // (ordering-independent w.r.t. common/insert_map_after_lio). Off by
  // default.
```


## include/livo_recon/utils/state/state.h

### include/livo_recon/utils/state/state.h:22-29

<a id="include-livo_recon-utils-state-state.h-22"></a>

```
  // ── Index functions ───────────────────────────────────────────────────────
  // R, P, V are always estimated and always first in the state vector.
  // BG, BA, G are optional; return -1 when not estimated.
  //
  // 2026-08-24: REMOVED idxW()/W (angular-velocity) state estimation --
  // existed solely for the removed iterative-deskew mechanism's Hermite-
  // spline rotation tangent. See
  // docs/removed_livo_recon_spline_deskew_2026aug24.md.
```


## include/livo_recon/vio/tracker.h

### include/livo_recon/vio/tracker.h:198-214

<a id="include-livo_recon-vio-tracker.h-198"></a>

```
  // Pose-history mechanism (2026-08-14, ported from FAST-LIVO2's
  // LivoVioManager::anchor_Rt_) -- replaces the previous caller-driven
  // "commit now" pattern (Tracker used to own mutable rot_anchor_/
  // pos_anchor_, updated via a commitAnchorPose() call every caller had to
  // remember to make at exactly the right time; a NEW caller (CombinedProc)
  // forgetting to do so was a real bug this session). Instead:
  //   - recordFramePose() is called ONCE per frame by LivoReconNode::
  //     estimateState(), AFTER that frame's correction (sequential or
  //     combined, whichever path ran) has fully finished -- pushes
  //     (timestamp, state_->rot(), state_->pos()) onto a FIFO history.
  //   - resolveAnchorPose() is called ONCE per frame, BEFORE either
  //     correction path runs, and looks up frame.prev_timestamp in that
  //     same history (FIFO front -- frames are consumed and recorded in
  //     strict timestamp order) to fill frame.rot_anchor/pos_anchor.
  // No caller-side "did I remember to commit" contract exists anymore --
  // any consumer of a TrackedFrame just reads rot_anchor/pos_anchor,
  // already resolved.
```

### include/livo_recon/vio/tracker.h:275-297

<a id="include-livo_recon-vio-tracker.h-275"></a>

```
  // Maps a RAW (distorted) observed pixel to its distortion-corrected
  // pixel-space equivalent (2026-08-04): (u-cx)/fx, (v-cy)/fy is only the
  // correct inverse-pinhole projection for an IDEAL (distortion-free)
  // pixel -- VioProc::computeEpipolarResidualAndJacobian applies exactly
  // that projection to whatever pixel it's given, so feeding it the raw
  // observed pixel directly (as this codebase always had, a bug shared
  // with FAST-LIVO2's livo_vio scaffold this residual model was ported
  // from) bakes in a real geometric bias that grows with distance from
  // the principal point whenever the lens has non-negligible distortion
  // (NTU_VIRAL's cam_d0 is -0.288, not negligible). Fixed HERE, once per
  // point at the source (seed selection / tracked-position snapshot)
  // rather than repeatedly inside the residual function every EKF
  // iteration -- cv::undistortPoints with P=K (the camera matrix) maps
  // the distorted pixel to its true undistorted position and
  // re-projects it back into pixel units via P, so the OUTPUT stays a
  // plain pixel coordinate and every downstream consumer (the epipolar
  // residual's own (u-cx)/fx, drawUvTracking's visualization, etc.)
  // needs no changes at all. Deliberately NOT applied to uv_out in
  // selectAnchors() or to uv_prev_/uv_curr_ (the backend's raw I/O) --
  // those must stay in true (distorted) image pixel space since that's
  // what the actual image data is and what the tracking backend
  // operates against; only the values that end up feeding the epipolar
  // residual (AnchorPoint::uv_seed, TrackedFrame::uv_curr) are corrected.
```


## include/livo_recon/vio/vio_accumulator.h

### include/livo_recon/vio/vio_accumulator.h:7-18

<a id="include-livo_recon-vio-vio_accumulator.h-7"></a>

```
// VIO's epipolar-line residual -> HtH/Htz accumulation, lifted out of
// VioProc (2026-08-14) so both VioProc's own sequential solve AND
// CombinedProc's joint LIO+VIO solve can call the exact same math without
// CombinedProc depending on VioProc's private internals -- mirrors
// lio/lio_accumulator.h's split for the LIO side. Needs more per-frame
// context than the LIO accumulator (camera intrinsics, tracked anchors),
// so it's a small stateful class (holding state_, like VioProc itself
// did) rather than a single free function. Deliberately does NOT hold a
// TrackerPtr (2026-08-14) -- frame.rot_anchor/pos_anchor are already
// resolved by Tracker::resolveAnchorPose() before accumulate() is ever
// called (see Tracker's own doc comment for that mechanism), so this
// class has no need to reach back into Tracker itself.
```

### include/livo_recon/vio/vio_accumulator.h:60-78

<a id="include-livo_recon-vio-vio_accumulator.h-60"></a>

```
  // DOF-split weighting (2026-08-15): a single epipolar residual's weight
  // used to land identically on BOTH the rotation and position columns of
  // J when forming HtH/Htz (thread_HtH += J^T * weight * J), even though
  // epipolar geometry is fundamentally scale-ambiguous -- it constrains
  // translation DIRECTION and rotation well, but has no way to observe
  // absolute translation MAGNITUDE the way LIO's metric point-to-plane
  // residuals can. Letting VIO's position-column contribution carry the
  // same confidence as its rotation-column contribution lets a single
  // marginal-quality frame's position "vote" get baked permanently into
  // the shared voxel map (see vio_processing.cpp/livo_recon_node.cpp's
  // updateMaps()-after-estimateState() ordering), corrupting LIO's own
  // future residuals against a map LIO never agreed to.
  //
  // pos_weight_scale multiplies ONLY the position columns of J before
  // forming HtH/Htz -- rotation columns always keep the full computed
  // weight. 1.0 (default) reproduces the old behavior exactly (no DOF
  // split). Set < 1.0 to let VIO keep correcting rotation at full
  // confidence while trusting it less for position, deferring position
  // consistency to LIO's global map.
```

### include/livo_recon/vio/vio_accumulator.h:81-94

<a id="include-livo_recon-vio-vio_accumulator.h-81"></a>

```
  // residual_mode (2026-08-16, ported from FAST-LIVO2's livo_vio ResidualMode
  // ::kMixed -- see vio_proc.cpp's own accumulation loop): "epipolar"
  // (default) is the original single epipolar-line residual above, whose
  // pos_weight_scale knob is a blunt way to distrust its position-column
  // contribution. "mixed" instead runs TWO independently-weighted residuals
  // per point -- the same epipolar-line residual with its POSITION columns
  // zeroed (rotation-only), plus a new ray-gap residual (perpendicular
  // distance between the anchor and current rays' closest-approach points,
  // computed via the SAME solveClosestRayPoints() already used for the
  // epipolar depth-clamp decision) with its ROTATION columns zeroed
  // (position-only) -- a principled decoupling rather than a scale-down.
  // pos_weight_scale is ignored in "mixed" mode (the split is already
  // structural, not a weight knob). A point is accepted only if BOTH
  // residuals compute successfully (matches FAST-LIVO2's own behavior).
```

### include/livo_recon/vio/vio_accumulator.h:106-107

<a id="include-livo_recon-vio-vio_accumulator.h-106"></a>

```
  // G6 (2026-09-01): see VioProcOptions::log_consistency_corr_en's doc
  // comment.
```


## src/lio/voxelnode.cpp

### src/lio/voxelnode.cpp:135-153

<a id="src-lio-voxelnode.cpp-135"></a>

```
      // Bug found 2026-08-24 (plane_fit_mode=="debiased" divergence
      // investigation): unlike the pca branch below, which never evaluates
      // is_plane_/acts on a rejection until update_count_ has reached
      // min_init_points (see that branch's early return just below), this
      // branch used to act on plane_ptr_->isPlane() immediately after
      // EVERY addPoints() call, starting from n_points=1 -- refitDebiased()
      // rejects (N<3, or any real fit that fails plane_threshold on a
      // tiny/noisy sample) almost always at that size, and a rejection
      // here triggers immediate subdivision (see !now_plane && layer_ <
      // max_layer below), so voxels were being shattered into 8 children
      // long before they'd ever accumulated a fair sample -- confirmed via
      // plane_init.txt: 117955/118073 first-fit attempts on an 8s eee_01
      // window were rejected, 114599 of those at n_points=1, vs pca mode's
      // matching min_init_points=5 histogram peak. Fixed by gating on
      // isInit() (points_size_ > min_init_points, cheap since points_size_
      // is already tracked) before treating this voxel's plane status as
      // conclusive -- addPoints() itself still runs unconditionally every
      // call (cheap, O(1), and the whole point of this mode), only the
      // downstream accept/reject/subdivide DECISION is gated.
```


## src/lio/voxelplane.cpp

### src/lio/voxelplane.cpp:70-79

<a id="src-lio-voxelplane.cpp-70"></a>

```
// T0-D (2026-08-31): corr.csv for scripts/analysis/consistency.py --
// header written once on first call (trunc), core-vs-full row shape
// decided by whether the covariate fields are finite/non-empty (see
// computeResidual()'s call site: covariates are all left at their default
// -1.0/-1 sentinel when log_consistency_covariates_en is off, and this
// function omits those columns from the header AND every row in that
// case, since consistency.py keys off column presence, not sentinel
// values). Same per-call mutex pattern as the two functions above --
// computeResidual() runs inside LioProc::buildResiduals()'s OMP
// parallel-for.
```

### src/lio/voxelplane.cpp:87-94

<a id="src-lio-voxelplane.cpp-87"></a>

```
  // 14a (2026-09-01): one stream, opened once, held open, with a large
  // buffer. The previous form opened and closed the file PER CORRESPONDENCE
  // (~12M times per job at full stride) under a global lock inside the
  // residual loop -- confirmed live to add minutes of wall-clock per job and,
  // more importantly, an UNVERIFIED determinism risk (the lock serializes a
  // section that's otherwise threaded/CUDA-dispatched, which can perturb
  // floating-point accumulation order -- T0-E-4 found this surface sensitive
  // to ~1e-5 relative changes). Same first-call truncate semantics as before.
```

### src/lio/voxelplane.cpp:103-111

<a id="src-lio-voxelplane.cpp-103"></a>

```
    // dropped_by_ablation (2026-08-31, code-review fix): unconditional,
    // not gated behind with_covariates -- T3-0e's ablation arms need this
    // even in a plain (non-covariates) NIS run, to know which rows were
    // excluded from the residual by the ablation itself vs by the
    // ordinary sigma-gate (gated=1 without this =0 means the LATTER).
    // When this is 1, S/s_sensor/.../s_prior_pose are all the -1.0
    // sentinel (gate() does not finish computing them for a dropped
    // correspondence) -- only nu, N/J/aniso/lambda0/occ_aniso/occ_cells
    // (plane-level, valid regardless) are meaningful on that row.
```

### src/lio/voxelplane.cpp:113-120

<a id="src-lio-voxelplane.cpp-113"></a>

```
    // T0-G (2026-08-31): lambda0 added -- eigen_values_(0), the plane fit's
    // smallest eigenvalue (its own is_plane_ = eigen_values_(0) <
    // plane_threshold test uses this directly). T8-0b/T8-d's outcome
    // variable ("median lambda0"); previously computed but never logged
    // anywhere. T3-0e (2026-08-31): occ_aniso/occ_cells added --
    // VoxelPlane::occupancyAnisotropy()/occupiedCellCount(), the OCCUPANCY-
    // based (density-independent) in-plane coverage anisotropy, distinct
    // from `aniso` (point-scatter/density-weighted).
```

### src/lio/voxelplane.cpp:134-140

<a id="src-lio-voxelplane.cpp-134"></a>

```
// 14b (2026-09-01): exact per-scan aggregates, so a level statistic (mean
// NIS, accept fraction, dropped-by-ablation count) never depends on the
// corr.csv row stride below -- every candidate updates this accumulator
// regardless of whether it also gets a full corr.csv row. corr_scan.csv is
// therefore the source of truth for anything a cell is actually judged on;
// corr.csv (now strided, see 14c) exists only for distributional questions
// (percentiles, decile cuts) that genuinely need individual rows.
```

### src/lio/voxelplane.cpp:259-272

<a id="src-lio-voxelplane.cpp-259"></a>

```
    // AUDIT, 2026-09-01. "ellipse" is not a shape-only change. Its region
    // has semi-axes max_radius*sqrt(l2) and max_radius*sqrt(l1) against the
    // disc's radius max_radius*sqrt(l2), so it is strictly CONTAINED in the
    // disc and admits sqrt(l1/l2) of its area. On a sliver at aniso 1000
    // that is 1/32 of the correspondences. An ellipse-vs-disc comparison
    // therefore confounds "the gate is the wrong shape" with "the gate
    // admitted far fewer points", and the size of the confound scales with
    // the very anisotropy under test -- exactly the population-matching
    // defect the T3-0e review found in the drop ablation.
    //
    // ellipse area = pi * T * sqrt(l1*l2);  disc area = pi * R^2 * l2
    // equal  =>  T = R^2 * sqrt(l2/l1)
    // "ellipse_area_matched" is the shape change at matched admitted area,
    // and is the arm the coverage hypothesis is actually about.
```

### src/lio/voxelplane.cpp:274-278

<a id="src-lio-voxelplane.cpp-274"></a>

```
      // Capped at 4x. Beyond aniso ~16 the areas are only PARTIALLY matched
      // and the arm drifts back toward the plain ellipse -- which is the
      // safe direction to fail, since the alternative is admitting points
      // arbitrarily far from a barely-sampled axis just to hit exact area
      // parity (audit finding A2, 2026-09-01).
```

### src/lio/voxelplane.cpp:288-295

<a id="src-lio-voxelplane.cpp-288"></a>

```
  // T0-D's "log before the gate" guardrail: mark candidacy (this point
  // cleared the purely-geometric test) BEFORE any T3-0e ablation drop
  // below, so a dropped correspondence still reaches corr.csv (as
  // dropped_by_ablation=1) instead of vanishing from the population
  // entirely -- code-review fix, 2026-08-31 (the original version
  // returned false above *is_candidate=true, so drops were unlogged and
  // any NIS/count comparison against baseline was over a silently
  // truncated population).
```

### src/lio/voxelplane.cpp:318-333

<a id="src-lio-voxelplane.cpp-318"></a>

```
    // Code-review fix: hash on occ_anchor_center_ (only reassigned on a
    // >10-degree occupancy reset, see updateOccupancy()) instead of
    // plane_.center (reassigned on EVERY refitDebiased()/update() call --
    // the original version redrew this "random" decision at scan rate,
    // ~10Hz, instead of once per plane; see the bug ledger entry this
    // fixes). Still only APPROXIMATELY stable (a rare reset changes the
    // draw), not a first-class per-plane identity, but no longer redrawn
    // every single fit.
    // POPULATION MATCH (T3-0e blocking fix, 2026-09-01). The "top" arm can
    // only ever select planes whose occ_aniso is DEFINED -- >= 3 occupied
    // cells -- unless occ_aniso_undefined_as_top says otherwise. Drawing
    // "random" from every plane makes arm 3 a draw from a strictly larger
    // population than arm 2, so the two arms remove different KINDS of
    // plane and the comparison has no control. Gate eligibility on the same
    // predicate "top" uses, so the two arms stay matched under either
    // setting of occ_aniso_undefined_as_top rather than only under one.
```

### src/lio/voxelplane.cpp:403-409

<a id="src-lio-voxelplane.cpp-403"></a>

```
    // s_sensor/s_tilt/s_d/s_pose/S/s_prior_pose all depend on J_nq/
    // sigma_diag_squared/plane_var_term, which gate() does NOT finish
    // computing when it returns early on a T3-0e drop (they're zero-
    // initialized above, not meaningful) -- skip computing/logging these
    // for a dropped row rather than logging garbage-looking zeros dressed
    // up as real numbers. S itself logs as the -1.0 sentinel for a
    // dropped row (code-review fix, 2026-08-31).
```

### src/lio/voxelplane.cpp:513-518

<a id="src-lio-voxelplane.cpp-513"></a>

```
  // Independent weighting for the plane_var_ Jacobian loop below (see
  // T3-0c) -- falls back to `weights`/`weight_sum` when not supplied, so
  // a caller that only passes `weights` gets the pre-2026-08-30 behavior
  // (fit and uncertainty share one weighting) unchanged. weight_sum is
  // finalized by the fit block below, so var_weight_sum's fallback is
  // resolved after it, not here.
```

### src/lio/voxelplane.cpp:734-755

<a id="src-lio-voxelplane.cpp-734"></a>

```
    // Combined sensor+pose covariance -- for plane_var_'s accumulators
    // only (Scov_/V_/W_), which legitimately want the full residual-noise
    // budget, same as the PCA path's Jmin loop above. sensor_cov alone
    // (independent per point) is what M_debiased's own fit-bias
    // correction uses, via Scov_sensor_ below -- see refitDebiased() for
    // why the shared pose component can't be folded in there the same
    // naive way.
    // 2026-08-24 pass4 diagnostic finding: plane_var_ blowups (traces up
    // to ~965,882 seen live) persist even when the denom1/denom2 guard
    // below passes comfortably (e.g. denom1=-0.003, eps=0.00025) -- the
    // real driver is pos_cov's own magnitude, not the eigengap. pos_cov
    // (poseCovAt()'s range^2 lever-arm term) is documented elsewhere in
    // this codebase to "dwarf true sensor noise for far points"
    // (VoxelOpts::pose_cov_in_sigma's docs, which is why THAT flag
    // defaults false) -- every blown-up sample here was F=1, N=4-6: a
    // fresh, barely-initialized voxel whose Scov_/V_/W_ accumulators are
    // raw sums of only a handful of points, not yet averaged down, so one
    // large pos_cov contribution dominates. Respect the same
    // plane_fit_pose_cov_mode ablation used for the PCA path's Jmin loop
    // here too (deviating from the original spec, which scoped it
    // PCA-only, based on this evidence -- see
    // docs/debiased_voxel_plane_fit_2026aug24.md).
```

### src/lio/voxelplane.cpp:808-812

<a id="src-lio-voxelplane.cpp-808"></a>

```
  // Bug ledger 2026-08-31: this was never set in debiased mode, leaving
  // the J column of corr.csv stale/zero on every debiased run. Reset here
  // and set to N_acc_ (this fit's effective point count) only on a
  // committed success below, mirroring update()'s PCA-path convention
  // (last_fit_j_ = use_weights ? N : 0).
```

### src/lio/voxelplane.cpp:848-864

<a id="src-lio-voxelplane.cpp-848"></a>

```
  // A population covariance's eigenvalues can never be negative -- a
  // negative SAMPLE eigenvalue here is definitionally sampling noise (the
  // "smallest of three correlated noisy quantities" selection is a biased
  // estimator of the true smallest eigenvalue, especially at low N/F), not
  // evidence the true scatter is negative. Clamping (rather than the old
  // reject-on-any-negative behavior) lets a mildly-negative result --
  // exactly what a genuinely flat surface's corrected estimate looks like
  // some fraction of the time -- correctly read as "flat" instead of being
  // thrown out. Confirmed empirically 2026-08-24: rejected candidates at
  // N=8-24, F=2 had raw (uncorrected) eig0 of 1e-8 to 1e-4 -- genuinely
  // near-flat -- pushed to ~-0.0002 by a sensor-noise correction of
  // comparable magnitude to its own sampling noise, not by a real
  // over-subtraction. See docs/debiased_voxel_plane_fit_2026aug24.md.
  // Guarding against the SEPARATE, larger-magnitude over-subtraction
  // failure mode (correlated noise from non-independent samples) is now
  // handled at the source, via trust_sensor_noise excluding bootstrap
  // points from the correction entirely, rather than by this clamp.
```

### src/lio/voxelplane.cpp:960-979

<a id="src-lio-voxelplane.cpp-960"></a>

```
  // Sanity ceiling on plane_var_ itself, not just the denom1/denom2 gap
  // that feeds it -- confirmed live (2026-08-24) that denom1/denom2 can sit
  // comfortably outside eps_denom (e.g. denom1=-0.003, eps=0.001) while
  // plane_var_'s trace still reaches ~965,882: a barely-initialized voxel
  // (F=1, N=4-6) has Scov_/V_/W_ built from only a handful of points, so
  // one large pos_cov contribution (documented elsewhere in this codebase
  // to "dwarf true sensor noise for far points", see pose_cov_in_sigma's
  // docs) dominates a still-small denominator's amplification. Rather than
  // trying to characterize every combination of small-N/large-pos_cov/
  // small-denom that can produce this, cap the actual output directly --
  // this is the quantity that widens gate()'s acceptance
  // (sigma_gate_squared = sigma_diag_squared + plane_var_term) and lets
  // garbage-magnitude residuals through, which is the direct divergence
  // mechanism. 1.0 is generous (orientation variance in [theta1,theta2] is
  // radians^2-scale and d's variance is meters^2-scale; both should be
  // small fractions for anything worth trusting as a residual source) --
  // reject (not clamp) so an under-converged voxel just waits for more
  // points/frames rather than silently supplying an overconfident-looking
  // but wrong uncertainty. addPoints() keeps accumulating regardless, so
  // this voxel can still become usable once F/N grow past this point.
```

### src/lio/voxelplane.cpp:1006-1016

<a id="src-lio-voxelplane.cpp-1006"></a>

```
// T3-0e (2026-08-31): 8x8 tangent-frame occupancy bitmask. Cell size
// opts_->voxel_size/8, grid spans [-4,4) cells each axis (i.e. +/-
// voxel_size/2 centered on the anchor). The basis is ANCHORED (frozen at
// the first call after a successful fit) rather than re-derived from the
// current x_normal_/y_normal_ every call, for two reasons: (1) a tangent
// frame that shifts slightly every refit would make "which cell is this"
// answer differently for the same physical point across calls, corrupting
// the occupancy pattern instead of accumulating it; (2) it matches T8-b's
// own card, which anchors for the same reason and resets only when the
// normal has rotated >~10 degrees (a reset reads as low coverage, which
// inflates uncertainty -- the safe direction, not a correctness bug).
```

### src/lio/voxelplane.cpp:1041-1046

<a id="src-lio-voxelplane.cpp-1041"></a>

```
// Code-review fix, 2026-08-31: recomputes both cached_occ_cells_ and
// cached_occ_aniso_ ONCE per fit (called from update()/addPoints() right
// after the occupancy-update loop), rather than on every
// occupancyAnisotropy()/occupiedCellCount() call -- see those two
// methods' header doc comment. Logic is otherwise unchanged from the
// original per-call implementation.
```

### src/lio/voxelplane.cpp:1093-1105

<a id="src-lio-voxelplane.cpp-1093"></a>

```
// T8-b (2026-09-01). The two plane-confidence terms livo_recon did not have.
//
// Both are multiplicative on plane_var_ and both are the IDENTITY when their
// switch is off. The coverage term is additionally the identity when coverage
// is isotropic (var_u == var_v), which is what makes a null result on this
// switch distinguishable from an inert switch: if occ_cells is healthy and
// plane_conf_factor is still 1.0 across a run, the geometry -- not the code --
// is what made it inert.
//
// Applied AFTER the fit and AFTER recomputeOccupancyCache(), never inside the
// fit itself, so the eigen-decomposition, the plane_threshold acceptance test
// and (on the debiased path) the ceiling rejection all still see the RAW fit.
// A switch that could change which planes exist would not be ablatable.
```


## src/livo_recon_node.cpp

### src/livo_recon_node.cpp:249-252

<a id="src-livo_recon_node.cpp-249"></a>

```
  // 14b (2026-09-01): corr_scan.csv's per-scan accumulator only flushes on a
  // scan_id CHANGE -- the last scan of a run never sees one, so it would sit
  // unflushed in memory forever without this. No-op if
  // log_consistency_corr_en was never on.
```

### src/livo_recon_node.cpp:320-328

<a id="src-livo_recon_node.cpp-320"></a>

```
  // Ground-truth topic dispatch -- see CbkProcOptions::evo_enable's doc
  // comment and CbkProc::gtCallback(). Only relevant for gt_source=="topic"
  // (e.g. NTU_VIRAL's live Leica stream): runOffline() never runs rosbag
  // play or any real ROS transport, so a plain topic subscriber would
  // otherwise NEVER fire in offline mode -- confirmed 2026-08-12: all 9
  // NTU_VIRAL jobs in a livo_recon offline-mode sweep failed (0 "[evo"
  // lines each) for exactly this reason, while file-GT (HILTI) jobs worked
  // fine since gt_source=="file" reads a static file directly, no topic
  // delivery involved -- doesn't need gt_topic added to the bag view.
```


## src/map/voxelmap.cpp

### src/map/voxelmap.cpp:17-24

<a id="src-map-voxelmap.cpp-17"></a>

```
// T0-F-2b (2026-08-31): denom_rejected count + max plane_var_ trace,
// bracketing THIS updateMap() call -- see voxelPlaneFrameStats{Reset,Read}()
// in voxelplane.h. Logged here (not lio_processing.cpp) because updateMap()
// -- and hence every VoxelPlane::update()/refitDebiased() call that can
// touch these counters -- can run either before or after processLIO(),
// depending on common/insert_map_after_lio; bracketing tightly around this
// one call is the only ordering-independent way to attribute the stats to
// the right frame.
```

### src/map/voxelmap.cpp:165-172

<a id="src-map-voxelmap.cpp-165"></a>

```
      // T0-G (2026-08-31): diagnostic-only permutation of the PROCESSING
      // ORDER (which point lands in which OMP static-schedule chunk, and
      // in what order within a chunk's thread-local key list) -- see
      // VoxelOpts::shuffle_insertion_seed's doc comment. proc_order[oi] is
      // the original point index processed at loop position oi; identity
      // (no-op) when the seed is 0. Distinct per frame (seed XOR
      // frame_idx_) so a multi-frame run doesn't apply the same
      // permutation to every frame's point count.
```


## src/processing/combined_processing.cpp

### src/processing/combined_processing.cpp:58-71

<a id="src-processing-combined_processing.cpp-58"></a>

```
  // Per-iteration monotonic-improvement guard (2026-08-15) -- mirrors
  // VioProc::solveSystem()'s own `if (avg_error >= last_error) return
  // false;` check, which CombinedProc never had. Without this, EVERY
  // iteration's joint update gets applied via applyMeanUpdate() below
  // unconditionally, even one where either side's fit at the CURRENT
  // (not-yet-updated-this-iteration) state has already gotten WORSE than
  // its own best-seen-so-far -- e.g. a single frame with a degenerate/
  // near-zero-baseline epipolar point can spike avg_err_vio by orders of
  // magnitude for one iteration, and that iteration's HtH/Htz still got
  // summed with LIO's and applied, permanently contaminating state (and,
  // via updateMaps() running after estimateState(), the shared voxel map
  // too). Tracking LIO's and VIO's own quality metrics independently
  // (rather than one combined scalar) avoids conflating two residuals on
  // different scales/units.
```

### src/processing/combined_processing.cpp:74-78

<a id="src-processing-combined_processing.cpp-74"></a>

```
  // G6 (2026-09-01): mirrors best_avg_res_lio/best_avg_err_vio's own
  // "last applied iteration's value" semantics, just for n_meas -- needed by
  // logConsistencyNll() below, which wants the joint update's ACTUAL last
  // solve, not a per-iteration-loop-local value gone out of scope by the
  // time the loop ends.
```


## src/processing/imu_processing.cpp

### src/processing/imu_processing.cpp:32-34

<a id="src-processing-imu_processing.cpp-32"></a>

```
// T7-a (2026-09-01): Myers-Tapley process-noise-estimator accumulators.
// See imu_processing.h's imuProcQhatRead() doc comment for the recursion
// this implements and why it's a recursion, not a sum, across substeps.
```

### src/processing/imu_processing.cpp:61-65

<a id="src-processing-imu_processing.cpp-61"></a>

```
  // sensor/range_err, sensor/angle_err_deg, ds/*, undistort/
  // time_based_process_noise moved to LioProc::loadParameters()
  // (2026-08-18) -- deskewing/downsampling now live there, see
  // LioProcOptions::deskew/ds_leaf_size/ds_mode's doc comments. Still the
  // SAME rosparam keys, just read by a different class now.
```


## src/processing/lio_processing.cpp

### src/processing/lio_processing.cpp:40-42

<a id="src-processing-lio_processing.cpp-40"></a>

```
// Temporary: per-iteration avg|r| trace within a single frame's IEKF loop,
// for the 2026-08-03 early-stopping/convergence-shape question. Remove
// once done debugging.
```

### src/processing/lio_processing.cpp:51-54

<a id="src-processing-lio_processing.cpp-51"></a>

```
// 2026-08-24: shadow dry-run diagnostic trace -- see LioProcOptions::
// dry_run_point_filter_num's doc comment and LioProc::runDryRunShadowPass().
// Same truncate-on-first-call/append convention as debugLogLio() above, own
// file so it never interleaves with the real per-frame [lio] lines.
```

### src/processing/lio_processing.cpp:63-70

<a id="src-processing-lio_processing.cpp-63"></a>

```
// T0-D (2026-08-31): scan.csv, T0-E (2026-08-31): nll.txt -- see
// LioProcOptions::log_consistency_scan_en/log_nll_en's doc comments for the
// column lists/formula. G6 (2026-09-01): the writers themselves moved to
// utils/log/consistency_log.h/.cpp (logConsistencyScan()/logConsistencyNll())
// so CombinedProc can also call them -- this anonymous namespace previously
// made both reachable ONLY from LioProc::processLIO(), which meant
// combined/LIVO mode had no consistency instrument at all. Call sites below
// pass channel="lio".
```

### src/processing/lio_processing.cpp:72-75

<a id="src-processing-lio_processing.cpp-72"></a>

```
// 2026-08-24: REMOVED debugLogSplineIter()/debugLogSplineFit()/
// debugLogSplineFitPoints()/debugLogSplineKinFit() -- temporary debug
// logging for the removed iterative-deskew Hermite-spline mechanism. See
// docs/removed_livo_recon_spline_deskew_2026aug24.md.
```

### src/processing/lio_processing.cpp:77-89

<a id="src-processing-lio_processing.cpp-77"></a>

```
// T7-a (2026-09-01). One row per frame. The estimator is formed OFFLINE from
// these columns -- deliberately not online.
//
// T0-E-3(C) measured NIS lowest at the WORST alpha on site1_handheld_4, so an
// online scheme watching a consistency statistic would have steered the filter
// onto that boundary and reported an improvement. The honest form of adaptive
// Q on this evidence is offline estimation with a stability margin, which is
// what this logger supports and what an in-loop adaptation would not.
// acw_diag is the accumulated cov_w's diagonal (the frame's process noise as
// actually built at the run's q_alpha_* values) -- kept as its own column
// block so an offline reducer (scripts/analysis/qhat.py) can recover the
// at-alpha=1 per-block reference the estimator's alpha ratio is defined
// against, without re-deriving IMU noise propagation in Python.
```

### src/processing/lio_processing.cpp:134-136

<a id="src-processing-lio_processing.cpp-134"></a>

```
  // Same rosparam keys the (removed) ImuProc deskew/downsample code used
  // to read -- see LioProcOptions::deskew/ds_leaf_size/ds_mode's doc
  // comments (deskewing/downsampling moved here from ImuProc 2026-08-18).
```

### src/processing/lio_processing.cpp:242-246

<a id="src-processing-lio_processing.cpp-242"></a>

```
// Both paths now just call the shared accumulator (lio/lio_accumulator.h,
// 2026-08-14 -- extracted so CombinedProc can build the exact same HtH/Htz
// without depending on LioProc's internals, per the plan for an opt-in
// combined LIO+VIO EKF step) and then apply the update, unchanged from
// before the extraction.
```

### src/processing/lio_processing.cpp:590-607

<a id="src-processing-lio_processing.cpp-590"></a>

```
      // T0-E: same first-iteration-only scope as T0-D's corr.csv/scan.csv
      // (see estimateStateCorrection()'s allow_consistency_log doc
      // comment) -- this frame's prior_cov_/ekf_.HtH/ekf_.Htz are exactly
      // what this iteration's solveSystem() just accumulated.
      // 2026-08-31 code-audit fixes (see LioProcOptions::log_nll_en's doc
      // comment): (1) log EVERY frame, including zero-residual ones
      // (n_residuals=0, nll=0.5*(logdet(P0)+logdet(A)) is still
      // well-defined there since HtH/Htz vanish) -- line count is now
      // frame-count-invariant, not correspondence-count-invariant, so
      // runs at different alpha stay row-comparable by index/timestamp.
      // (2) n_residuals logged explicitly so callers can normalize
      // per-correspondence (nll/n_residuals, or sum(nll)/sum(n_residuals)
      // across a run) instead of summing a quantity whose per-frame N
      // varies with both frame content AND alpha. (3) the previously-
      // dropped n*log(2*pi) Gaussian normalization term is now included
      // -- omitting it is harmless for a FIXED N but not when N varies
      // between the runs being compared, which is exactly this
      // objective's use case.
```

### src/processing/lio_processing.cpp:774-785

<a id="src-processing-lio_processing.cpp-774"></a>

```
      // Post-fit normalized-residual consistency check (2026-08-03, NIS-
      // style -- see the Woodbury/Sherman-Morrison-per-plane discussion):
      // unlike trace(P_PP)/trace(P_RR) above (which only reflect the
      // filter's own INTERNAL belief about its uncertainty), this checks
      // that belief against what the ACTUAL post-fit residuals look like.
      // sum_chi2 = Sum(r_i^2/sigma_squared_i) over this frame's residuals;
      // reduced_chi2 = sum_chi2/n_residuals should average ~1 for a
      // correctly-calibrated noise model (each residual's squared error
      // should be on the same scale as its own claimed variance). >>1 means
      // sigma_squared is set too small somewhere (overconfident -- actual
      // errors are bigger than the model expects); <<1 means it's set too
      // large (overly conservative).
```

### src/processing/lio_processing.cpp:837-854

<a id="src-processing-lio_processing.cpp-837"></a>

```
      // Per-residual eigen-projection diagnostic (2026-08-03, temporary --
      // see the pfn1_ds000-vs-pfn1_ds010 confidence-analysis finding that
      // H_pp/H_rr's AGGREGATE eigenvalues were nearly identical between the
      // two despite ds000 having ~15% more residuals every frame). Tests
      // whether that extra residual mass is structurally redundant --
      // concentrated on the ALREADY-well-constrained (largest-eigenvalue)
      // direction -- rather than contributing to the weakest direction,
      // by projecting each residual's own Jacobian (unweighted by its own
      // sigma_squared, a purely geometric/structural quantity) onto H_pp/
      // H_rr's eigenvectors and averaging the normalized squared-projection
      // fraction across all residuals this frame. A LOW mean_frac_weak
      // means most residuals barely touch the weakest direction at all
      // (piling onto the strong one instead) -- direct, per-residual
      // evidence for or against the "extra residuals mostly redundant"
      // hypothesis this whole investigation has been built on, complementing
      // (not replacing) the aggregate H_pp_eig/H_rr_eig numbers already
      // logged above. Eigen::SelfAdjointEigenSolver returns ascending
      // eigenvalues, so col(0)=weakest, col(2)=strongest.
```


## src/processing/vio_processing.cpp

### src/processing/vio_processing.cpp:107-111

<a id="src-processing-vio_processing.cpp-107"></a>

```
// ─────────────────────────────────────────────
//  EKF update -- accumulation delegated to VioAccumulator (vio/
//  vio_accumulator.h, 2026-08-14), this class now just adapts opts_ and
//  owns the iteration/rejection/apply/logging policy.
// ─────────────────────────────────────────────
```

### src/processing/vio_processing.cpp:147-172

<a id="src-processing-vio_processing.cpp-147"></a>

```
  // 2026-08-16: investigated replacing this strict, zero-tolerance check
  // (dominant stop reason on 69.6% of frames in a site1_handheld_3
  // diagnostic, usually firing right after genuine convergence) with
  // either full removal or a small relative tolerance, on the theory that
  // the stop point was a coin flip decided by sub-ULP floating-point
  // noise. Both tested empirically on site1_handheld_3:
  //  - full removal (matching fastlivo's unconditional-apply myvio solve):
  //    REGRESSED badly, 6.37m -> 2378.8m ATE. Not every "worse" iteration
  //    this check catches is noise-level -- some are genuine overshoots
  //    (livo_recon's HtH is less well-conditioned than fastlivo's, per the
  //    anchor_split_depth/distortion_weight_on gap -- see
  //    VIO_METHODOLOGY_DIFF.md), and applying them unconditionally bakes a
  //    genuinely-bad overshoot into the state before rel_diff ever gets a
  //    chance to stop the loop.
  //  - 1e-6 relative tolerance: bit-identical ATE to the strict check
  //    (6.37m, both n=3 and n=4 evo scores matched exactly). This is a
  //    no-op because the OMP accumulation-order fix (see
  //    feedback_livo_recon_omp_nondeterminism memory) already made a
  //    single run's HtH/Htz accumulation, and therefore every avg_error
  //    value and accept/reject decision here, fully deterministic --
  //    there's no more run-to-run floating-point noise left for a
  //    tolerance to absorb. The original "coin flip" framing applied to
  //    comparing DIFFERENT thread-count runs pre-fix, not to a single
  //    run's own reproducibility.
  // Conclusion: this check is not broken. Reverted to the original strict
  // form.
```


## src/utils/data/data_queues.cpp

### src/utils/data/data_queues.cpp:21-32

<a id="src-utils-data-data_queues.cpp-21"></a>

```
    // See dry_run_lidar_queue's doc comment (data_queues.h) -- missed in
    // the original addition (2026-08-24): any dry-run scans pushed before
    // setStartTime() runs (calibration can consume several lidar/imu
    // callbacks before the first image arrives) kept absolute timestamps
    // forever, since only lidar_queue was corrected here. popDryRunLidar()
    // compares its front scan's (absolute) timestamp against max_time
    // (always relative, since it's derived from an already-adjusted image
    // timestamp) and `break`s immediately when front > max_time -- so once
    // even one such stale entry sat at the front, EVERY subsequent
    // popDryRunLidar() call returned nothing, permanently, for the entire
    // run. Confirmed via an eee_01 sanity run: 3981 real [lio] frames
    // logged, zero [lio_dryrun] frames.
```


## src/utils/log/consistency_log.cpp

### src/utils/log/consistency_log.cpp:9-13

<a id="src-utils-log-consistency_log.cpp-9"></a>

```
// T0-D (2026-08-31): scan.csv for scripts/analysis/consistency.py -- see
// LioProcOptions::log_consistency_scan_en's doc comment for the column
// list. G6 (2026-09-01): hoisted out of lio_processing.cpp's anonymous
// namespace so CombinedProc can also log its own scan rows, distinguished
// by the new leading `channel` column ("lio"/"vio"/"rollback").
```

### src/utils/log/consistency_log.cpp:29-33

<a id="src-utils-log-consistency_log.cpp-29"></a>

```
// T0-E (2026-08-31): nll.txt -- see LioProcOptions::log_nll_en's doc
// comment for the column/formula. G6 (2026-09-01): hoisted out of
// lio_processing.cpp's anonymous namespace, same reasoning as
// logConsistencyScan() above -- CombinedProc logs "lio"/"vio"/"rollback"
// rows for the same frame, so the two channels can finally be compared.
```

### src/utils/log/consistency_log.cpp:48-53

<a id="src-utils-log-consistency_log.cpp-48"></a>

```
// G6 (2026-09-01): VIO-side correspondence log, mirroring corr.csv's
// nu/S columns so both channels are scored by the same statistic. Logged
// BEFORE any further rejection at the call site (vio_accumulator.cpp),
// matching corr.csv's own "log candidates before the gate" convention --
// T0-D measured a NIS computed over only the survivors of its own gate as
// biased low (2.59 -> 1.85 on a fixture) when this was done wrong.
```


## src/utils/state/state.cpp

### src/utils/state/state.cpp:156-158

<a id="src-utils-state-state.cpp-156"></a>

```
// 2026-08-24: REMOVED setEstAngVel()/resetAngVelCov() -- existed solely
// for the removed iterative-deskew mechanism's W (angular-velocity) state
// estimation. See docs/removed_livo_recon_spline_deskew_2026aug24.md.
```


## src/vio/cotracker/cotracker_backend.cpp

### src/vio/cotracker/cotracker_backend.cpp:287-293

<a id="src-vio-cotracker-cotracker_backend.cpp-287"></a>

```
    // Confidence gate (2026-08-15, matching FAST-LIVO2's myvio dual vis+conf
    // gate -- see cotracker_backend.h's doc comment on outputs_conf being
    // "read but discarded"). Only the AOT path's 3-output package actually
    // carries a real confidence tensor -- runTracker()'s JIT branch leaves
    // out_conf default-constructed (undefined) since pre-AOT tracker
    // exports only ever had 2 outputs, so guard on that rather than
    // unconditionally reading outputs_conf.
```


## src/vio/vio_accumulator.cpp

### src/vio/vio_accumulator.cpp:335-350

<a id="src-vio-vio_accumulator.cpp-335"></a>

```
  // Per-POINT (not per-thread) accumulators for HtH/Htz/err -- see the doc
  // comment on the final reduction loop below for why: this makes the
  // floating-point summation order depend only on the point count (fixed
  // by this frame's own tracked-point data), never on how many OMP threads
  // happened to run this call. Fixes a confirmed run-to-run non-
  // determinism (2026-08-16): varying thread counts produced different
  // summation groupings and therefore different LSB-level rounding in
  // double-precision HtH/Htz, which the iterative IEKF solve then
  // amplified into materially, occasionally catastrophically different
  // results for byte-identical input on ill-conditioned frames -- see
  // pinOmpThreadsForDeterminism()'s doc comment (utils/algo/omp_utils.h)
  // for the original investigation. That fix pins the thread count so a
  // given machine is internally reproducible; THIS fix makes the actual
  // numeric result independent of thread count entirely, so two machines
  // (or the same machine before/after a hardware/OMP_NUM_THREADS change)
  // computing the identical frame get the identical HtH/Htz too.
```

