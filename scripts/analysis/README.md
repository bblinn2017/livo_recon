# T0-D/T0-E/T0-F analysis scripts

`consistency.py` is the analyser: reads `corr.csv`/`scan.csv` (see
`VoxelOpts::log_consistency_corr_en`/`log_consistency_covariates_en` and
`LioProcOptions::log_consistency_scan_en`) and reports NIS calibration,
whiteness, the variance budget, and NIS-vs-covariate structure. `synth.py`
generates filter runs with *known* Q/R mis-tunings so the analyser can be
checked before it's pointed at real logs. `qsens.py` is the scalar
correctness proof for the Q-by-gradient idea (see T0-E's card and note
for why the real port used a finite-difference multi-run search instead).

`sweep_scoring.py` is `consistency.py`'s statistics repackaged as a quiet,
per-job data extractor rather than a human-readable report -- built for a
sweep's score-then-delete pass (see the register's SP-4a"/4b" protocol
card), which needs the same numbers (block-bootstrap NIS, lag-1 ACF/
Ljung-Box, the plane-share-vs-full-S percentiles, occ_aniso/occ_cells
quintile-mean NIS, the 1e-3-floor histogram) computed and retained per
job, before that job's multi-GB `corr.csv` is deleted. It imports
`block_bootstrap_ci`/`acf` from `consistency.py` directly rather than
reimplementing them. `fast_ws/scripts/parallel/sp4_sweep_score_job.py`
imports this module by path (it is not itself part of this git repo) --
see that script's own comment for why the split is there: the sweep
orchestration is disposable/session-specific, the statistics are not.

`sm1_engagement_check.py` is the per-flag existence-proof check for the
register's SM-1 card -- not a statistics module, since SM-1 is ~10-15
short one-flag-at-a-time runs, not a population. Reads a job's own
`spline_q.csv` (retained in full, no score-then-delete pass needed) and
reports the CSV's own effective `per_iteration` value against what was
requested, per-frame refine/reintegrate/redeskew counters and
magnitudes, and an `odometry.txt` md5 comparison against a baseline job.
Found a real bug while being built: a bare YAML `mode: off` parses as
the boolean `False` under YAML 1.1's "Norway problem," so the C++ side's
string-typed read silently fell back to its default rather than
disabling the mechanism SM-1 was checking -- fixed in `fast_ws`'s
`gen_jobs.py` (`_quote_if_yaml11_ambiguous()`), not in this repo.

## Step 0 — validate the analyser before trusting a word it says

```
for c in cal q_low q_high r_low; do python3 synth.py $c; done
for c in cal q_low q_high r_low; do python3 consistency.py --corr corr_$c.csv --scan scan_$c.csv; done
```

Expected output (1500 scans x 400 correspondences per fixture):

| fixture | true fault | mean per-scan NIS [95% block-CI] | lag-1 ACF (band ±0.051) | verdict |
|---|---|---|---|---|
| cal | none | 1.006 [0.991, 1.023] | -0.039 | calibrated — correct |
| q_low | Q 25x too small | 1.305 [1.254, 1.362] | +0.087 | "Q IS TOO LOW" — correct |
| r_low | R 4x too small | 2.590 [2.438, 2.735] | -0.044 | "R IS TOO LOW, not Q" — correct |
| q_high | Q 25x too large | 0.229 [0.194, 0.261] | -0.043 | under-confident, cause NOT attributed (known gap — see consistency.py's own panel 3 caveat) |

If a future change to `consistency.py` doesn't reproduce this table
(within noise), the analyser is broken — find out from the fixture, not
from a real result that looks odd.

## Known gaps (2026-08-31)

T0-E's real-EKF NLL objective (`LioProcOptions::log_nll_en`,
`EkfUpdate::nllQuadraticAndLogdet()`) has two accounting bugs an external
audit found: `nll.txt`'s line count is alpha-dependent (zero-residual/
empty-map frames are skipped, and this changes with alpha), and the
`(N/2)*log(2*pi)` normalization term is dropped while N varies per frame
and with alpha. Both need fixing (or the objective needs replacing with
mean NIS, which is already correspondence-normalized and immune to
both) before any q_alpha sweep's totals are compared across runs again.
See the register's T0-E/T0-E-2 cards for the full writeup.
