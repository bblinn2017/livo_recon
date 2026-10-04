# R73 smoke table (22 jobs)

| arm | state | ATE | engagement verdict |
|---|---|---|---|
| r73_00_id_baseline_60s | DONE | 0.0548m | identity reference (digest 6cd2d75482662f38) |
| r73_00_id_baseline_full | DONE | 0.0309m | within 0.005mm of R72 |
| r73_00_id_motionq_full | DONE | 0.0260m | within 0.005mm of R72 |
| r73_id_diag_on_60s | DONE | 0.0548m | byte-identical to baseline |
| r73_id_dump_lastiter_60s | DONE | 0.0548m | byte-identical to baseline |
| r73_id_nisgate_dead_60s | DONE | 0.0548m | byte-identical to baseline; gain never left 1 |
| r73_id_scanlog_60s | DONE | 0.0548m | byte-identical to baseline |
| r73_id_shadow_same_60s | DONE | 0.0548m | byte-identical to baseline -- shadow does not disturb driver |
| r73_id_stepcap_wide_60s | DONE | 0.0548m | byte-identical to baseline; cap never bound |
| r73_id_zupt_detect_60s | DONE | 0.0548m | byte-identical to baseline |
| r73_smoke_co_ipscale_60s | DONE | **215834.197m** | clean exit (exit_code=0), but catastrophic divergence under estimator.mode=coupled, residual_evaluation_time=measurement, imu_prior_info_scale=0.1 -- reported as a fact (see report.md S7), not interpreted |
| r73_smoke_co_measurement_rec_60s | DONE | 4.4022m | record files present, no failure dump |
| r73_smoke_co_tail_rec_60s | DONE | 0.0248m | record files present, no failure dump |
| r73_smoke_freeze_60s | DONE | 0.2691m | map digest constant after frame 1 |
| r73_smoke_innov_scale_60s | DONE | 0.0369m | gain moved, not inert |
| r73_smoke_local_poly_60s | DONE | 1.1038m | not inert |
| r73_smoke_nis_gate_60s | DONE | 0.0285m | not inert |
| r73_smoke_shadow_fixed_mq_60s | DONE | 0.0292m | shadow rows == driver rows, map digest unchanged |
| r73_smoke_shadow_iso_60s | **FAILED** | n/a | **crashed: SIGABRT, "motion-dependent Q requires positive stationary calibration floors" -- see report.md S2/S7 for the traced root cause (shadow's per-scan state rollback losing the calibration floor fields, only exercised by the isotropic process-noise-model's dynamic_q check)** |
| r73_smoke_stepcap_fq_60s | DONE | 0.0404m | not inert, 251 bound frames |
| r73_smoke_wnoa_60s | DONE | 0.0325m | not inert |
| r73_smoke_zupt_apply_60s | DONE | 0.0372m | applied > 0 |

21/22 DONE, 1/22 FAILED (r73_smoke_shadow_iso_60s -- a genuine runtime crash, not a sentinel misclassification; exit_status.txt reads `exit_code=-6 signal=SIGABRT`). Full gate results in `R73_STAGE1B/r73_gates.txt`: PASS 78, FAIL 0, MISSING 2 (both MISSING rows are this arm's SHADOW checks, correctly not a stop condition per the instructions' MISSING-vs-FAIL distinction, since the job ran and crashed rather than never running).
