# BASE baseline reference

Authority for BASE-0 / BASE-B / BASE-W's configuration (rule 39). This doc, not a
path on h2 and not the register card's own prose, is what a pass loads. Written
2026-09-03 after pass 2 found and fixed two config-reconstruction errors and one
compiled-default drift; see the register's BASE card for the round-by-round
narrative, not repeated here.

## Part 1 — the three configs

All three: `eee_01`, `dataset=ntu_viral`, LIO-only (`vio.enable: false`),
`tracker.track_cache_path=/root/catkin_ws/track_cache_livo_recon/eee_01.rlivtrackcache`,
offline+cache mode. Build `a8f420c` or later (P-A..P-E landed there).

### BASE-0 — L0, recorded ATE 0.0281 m

Source: `livo_recon_results/dx1r/dx1r_l0_eee_01/config.yaml`, used **verbatim**.
Reproduces exactly at `a8f420c` (confirmed 2026-09-03, pass 2: ATE 0.0281 m to
the printed digit). No additional key needed.

```yaml
evo:
  enable: true
tracker:
  track_cache_path: /root/catkin_ws/track_cache_livo_recon/eee_01.rlivtrackcache
vio:
  enable: false
voxel_map:
  plane:
    plane_fit_pose_cov_mode: sensor_only
    log_consistency_mode: corr+covariates
    plane_fit_mode: pca
  map:
    log_frame_stats_en: true
lio:
  log_consistency_scan_en: true
  log_nll_en: true
spline:
  enable: false
outputs:
  odom:
    export: true
```

`spline/enable: false` here means `spline/*` and `adaptive_q/*` must be set
NO OTHER KEY — `lio_processing.cpp:277`'s `refuseUnclaimed({"spline","adaptive_q"})`
aborts the run (SIGABRT) if one is set with the scope off.

### BASE-B — L5, recorded ATE 0.0253 m

Source: `livo_recon_results/dx1r/dx1r_l5_eee_01/config.yaml`, **verbatim, PLUS
`spline/boundary_anchor_mode: "none"` pinned explicitly.**

`boundary_anchor_mode` is absent from the archived file. The compiled-in
default (`include/livo_recon/lio/spline.h:397`) is `"exact"` in the current
build, but the archived run — and per DX-3, all 276 retained cells — ran under
`"none"`, a default that has since drifted. Running the archived file verbatim
WITHOUT this pin reproduces 0.0287 m, a +3.4 mm miss. WITH the pin, reproduces
exactly (confirmed 2026-09-03, pass 2: ATE 0.0253 m to the printed digit).

```yaml
evo:
  enable: true
tracker:
  track_cache_path: /root/catkin_ws/track_cache_livo_recon/eee_01.rlivtrackcache
vio:
  enable: false
voxel_map:
  plane:
    plane_fit_pose_cov_mode: sensor_only
    log_consistency_mode: corr+covariates
    plane_fit_mode: pca
    plane_var_mode: information_directional
    weight_floor:
      mode: roughness
  map:
    log_frame_stats_en: true
lio:
  log_consistency_scan_en: true
  log_nll_en: true
spline:
  enable: true
  boundary_anchor_mode: "none"        # PINNED -- see note above, do not omit
  control_points:
    mode: hz
    hz: 100.0
  per_iteration:
    mode: redeskew+refine+reintegrate
    refine:
      prior_w: 0.1
      max_step: 0.02
adaptive_q:
  enable: true
outputs:
  odom:
    export: true
```

### BASE-W — L6, recorded ATE 0.0349 m

BASE-B's block above, with EXACTLY these keys changed:

```
cbk/lidar/point_filter_num:  3 -> 1
imu/ds/mode:                 "first" -> "off"
```

`imu/ds/ds_leaf_size` (0.1, unconditionally set by `config/ntu_viral.yaml`) is
DEAD once `imu/ds/mode` is off and must be deleted from the base document, not
merely left unset in the override — `gen_jobs.py`'s `split_deletions()`/
`apply_deletions()` (a `None`-valued override) does this; setting the override
layer's `mode: off` alone triggers a `[config/REFUSED]` SIGABRT (dead-scope
key set explicitly). Do NOT re-add `ds_leaf_size` anywhere in the override.

`spline/boundary_anchor_mode: "none"` must ALSO be pinned here. **CONFIRMED
2026-09-03 (pass 3): reproduces exactly, ATE 0.0349 m to the printed digit**,
matching the prediction from DX-3's exact-vs-none delta for L6 exactly.

## FR-5 — the two config readouts that finish Part 1 (2026-09-04, pass 5)

**(a) `voxel_map/plane/weight_floor/mode` AS RESOLVED.** Confirmed by direct
grep of `[config/effective]` in both BASE-B's and BASE-W's `run.log`:
`weight_floor/mode = roughness` on **both** cells (matches the archived L5
config, which sets it explicitly — see BASE-B's block above, already correct
in this doc since pass 3). BASE-0 (eigengap, no floor-mode key set) resolves
to the code default `sensor_range` — legal there since `eigengap` is not an
`information*` mode.

**(b) `[config/effective]` verbatim + resolved `cbk`/`imu`/`outputs` values —
present for all three cells, captured via `emit_batch_blocks.py`'s
`[BASE-META v3]` block (2026-09-04) and cross-checked directly against each
cell's own `run.log`.** Rule 39b's claim that `cbk`/`imu`/`outputs` are a
blind spot is corrected (pass 3): they print resolved values via
`[params/cbk]`/`[params/imu]`/`[params/pub]` — a different block than
`[config/effective]`, equally authoritative, present in every run.log. Full
verbatim blocks are in the `[BASE-META]` output
(`scratch_base_smoke/base_pass3_blocks.txt`, not reproduced here — this doc
records the fact of their presence and the one load-bearing value (a) above,
not every echoed key). **This closes FR-5 and un-blocks TQ-4 (the
configuration grid) per FR-5(a)'s own gating condition** — Part 1 of this doc
is no longer PARTIAL.

## Provenance

- `dx1r_l0_eee_01` / `dx1r_l5_eee_01` archived at commit `77574f3`; `dx1`'s
  equivalents (same cells, same harness ATE) at `f3e372b`.
- BASE (this doc) runs at `a8f420c` or later.
- Known compiled-default drift between those commits and `a8f420c`+:
  `spline/boundary_anchor_mode` (`none` → `exact`) — see BASE-B above. No
  other drift confirmed yet; rule 39's two-way `[config/effective]` diff only
  covers `lio_processing.cpp`/`voxelmap.cpp`-resolved keys (`voxel_map/plane/*`,
  `spline/*`, some `imu/ds/*`) — `cbk/*`, `imu/sensor/*`, `outputs/*`,
  `tracker/*` are NOT covered and have not been checked for drift.
