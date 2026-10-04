#!/usr/bin/env python3
"""R75: build jobs_ln.json, arm_table.tsv, listener_spec_ln.json, required_outputs.json and r75_file_size_ledger.json from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r75_manifest.py --jobs r69_inputs/r66_jobs.json --out r75_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (run only in the selftest-style check on the shipped r66_jobs.json: counts and key checks).
Reuses the R74 helpers (make_r74_manifest as M): the same R66 base jobs, the same EF cell grid, the same overrides, so an R75 arm differs from its R74 counterpart ONLY in the line variant.

Bryce 2026-10-02: line mode detects lines properly (hard absolute); hard relative is inert (e0 ~ 0) and needs a redesign; rerun ONLY the redesigned hard-relative on the full grid,
keep the R74 data of the other arms; "Run both as separate arms": noise-floor variant (hardrelnoise) and shape variant (hardshape, tau 0.1); floor 4e-4, k 3; no variance floor on hard rows; no new outputs.
  hardrelnoise  line_residual.mode hard, threshold relative_noise, rel_sigmas 3.0   (e1 <= (1 + 3 sqrt(2/(N-1))) * max(e0, sensor_noise_var 4e-4))
  hardshape     line_residual.mode hard, threshold shape, shape_ratio 0.1            (e1 <= 0.1 * e2)
Arms: 2 variants x 2 voxel sizes (0.425, 1.0; max_layer 2) x 40 EF cells = 160, plus ONE determinism control (rule 3.7) = 161, one queue (r75_ln).
"""
import argparse, copy, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r74_manifest as M  # noqa: E402

LINE75 = {"hardrelnoise": {"voxel_map.plane.line_residual.mode": "hard", "voxel_map.plane.line_residual.threshold": "relative_noise", "voxel_map.plane.line_residual.rel_sigmas": 3.0},
          "hardshape": {"voxel_map.plane.line_residual.mode": "hard", "voxel_map.plane.line_residual.threshold": "shape", "voxel_map.plane.line_residual.shape_ratio": 0.1}}
DESC = {"hardrelnoise": "hard relative with a noise floor (e1 <= (1+3*sqrt(2/(N-1)))*max(e0, 4e-4))", "hardshape": "hard shape (e1 <= 0.1*e2)"}
DET_ID = "r75_own_det_hardshape_v1p0_n1600_dsoff_pfn3_fq"
DET_ORIG = "r75_line_hardshape_v1p0_n1600_dsoff_pfn3_fq"
ASKED = ("Bryce (R74 return, 2026-10-02): 'the hard relative does need a redesign ... we can just change and repeat hard relative line, no need to repeat the others' ; 'Run both as separate arms' ; "
         "'just rerun hard relative only full grid and add these on to the existing data and recompute the table' ; floor 4e-4, k 3 ; shape tau 0.1 ; 'No, leave as is' (no variance floor on hard rows)")
# measured in the R74 return (median / max over 322 arms): residual_dump_all 405 KB / 2.4 MB, r73_f417_hth_frames 2 KB, pos_trace_all 4.3 KB / 146 KB, r73_f417_residuals (n200 arms) 54 KB / 284 KB
EXPECTED = {"residual_dump_all.csv.gz": 450000, "r73_f417_hth_frames.csv": 2000, "pos_trace_all.csv.gz": 5000, "r73_f417_residuals.csv.gz": 60000}
MAXFILE = {"residual_dump_all.csv.gz": 2800000, "r73_f417_hth_frames.csv": 1000000, "pos_trace_all.csv.gz": 200000, "r73_f417_residuals.csv.gz": 330000}   # per-file ceilings (R74 maxima + 15 %)
COLS_F426 = {}


def arms():
    A = []
    for var in ("hardrelnoise", "hardshape"):
        for v in (0.425, 1.0):
            for ns, ds, pfn, q, dl, dur in _cells():
                A.append(dict(id="r75_line_%s_v%s_n%d_ds%s_pfn%d_%s" % (var, M.tag(v), ns, ds, pfn, q), base=M.base_of("eee_01", q), group="LN", tier="EF", dur=dur,
                              delta=dict(M.VOX(v, 2), **LINE75[var], **M.F1DUMP, **dl), f417=1, plane_check=0,
                              asked="%s: variant %s, %s; root voxel %s, max_layer 2, EF cell n%d ds %s pfn %d %s (the R74 cell, only the line variant is new)" % (ASKED, var, DESC[var], v, ns, ds, pfn, q),
                              purpose="R75 line variant %s at voxel %s / max_layer 2, EF cell n%d ds %s pfn %d %s, frame-1 dump" % (var, v, ns, ds, pfn, q)))
    dl = {"calib.stationary.num_samples": 1600, "imu.ds.mode": "off", "cbk.lidar.point_filter_num": 3, "imu.ds.ds_leaf_size": None}
    A.append(dict(id=DET_ID, base=M.base_of("eee_01", "fq"), group="DET", tier="EF", dur=int(math.ceil(1600 / 197.5 + 8.0)),
                  delta=dict(M.VOX(1.0, 2), **LINE75["hardshape"], **M.F1DUMP, **dl), f417=1, plane_check=0,
                  asked="OWN (rule 3.7, determinism control in every batch): an exact repeat of %s; its state-trace digest must equal that arm's. Needs Bryce's sight: not asked for in his words" % DET_ORIG,
                  purpose="OWN determinism control: repeat of %s" % DET_ORIG))
    return A


def _cells():
    for ns in M.EF_N:
        for ds in ("first", "off"):
            for pfn in (1, 3):
                for q in ("fq", "mq"):
                    dl = {"calib.stationary.num_samples": ns, "imu.ds.mode": ds, "cbk.lidar.point_filter_num": pfn}
                    if ds == "off":
                        dl["imu.ds.ds_leaf_size"] = None
                    yield ns, ds, pfn, q, dl, int(math.ceil(ns / 197.5 + 8.0))


def required_outputs(table):
    ids = lambda grp: [t["arm"] for t in table if t["group"] in grp]
    ln = ids({"LN", "DET"})
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm"):
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={},
                      expected_bytes=EXPECTED.get(file, 50000) * max(1, len(arms_)) if where == "return" else 0, max_bytes_per_file=MAXFILE.get(file, 400000)))
    add("F-436", "per-residual dump of frame 1 with row_kind (0 plane, 1-2 hard line u0 u1) for every R75 arm", ln, "residual_dump_all.csv.gz", M.DUMP_COLS + ["row_kind"], 100)
    add("F-436", "residual report of frame 1 (HtH / Htz metrics) for every R75 arm", ln, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1)
    add("F-436", "per-scan posterior position and yaw (t_rel_s) of every R75 arm", ln, "pos_trace_all.csv.gz", M.POS_COLS, 20)
    add("F-436", "F-417 per-residual table of frame 1, only the arms with calibration samples 200 (as R74)", [a for a in ln if "_n200_" in a], "r73_f417_residuals.csv.gz", ["frame", "iter", "weight", "cls"], 100)
    add("F-435", "F-426 part 1 per-frame shadow-vs-driver step difference (existing R73 shadow files, r75_step0_f426.py, no new run)", [], "r74_f426_part1_frames.csv", ["pair", "frame", "step_diff_mm", "at_or_after_onset"], 200, source="generated")
    add("F-435", "F-426 part 1 summary", [], "r74_f426_part1_summary.csv", ["pair", "onset", "max_before_mm", "max_from_onset_mm"], 4, source="generated")
    add("F-435", "F-426 part 2 per-frame alignment-free comparison (R74 switch arms, no new run)", [], "r74_f426_part2_frames.csv", ["pair", "frame", "dA_mm", "dB_mm", "dAB_mm"], 200, source="generated")
    add("F-435", "F-426 part 2 summary", [], "r74_f426_part2_summary.csv", ["pair", "switch_arm", "onset", "dA_before_max_mm", "dB_from_onset_max_mm"], 4, source="generated")
    add("F-434", "gates: identity not applicable; determinism digest, effective line config, row kinds, engagement", [], "r75_gates.txt", [], 1, source="generated")
    add("F-434", "determinism control and its original", [], "identity_controls.txt", [], 1, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated", [], "engagement_all.txt", [], 50, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 100, source="generated")
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm", ln, "engagement_imu.txt", [], 1, where="archive")
    return R


def ledger(RO):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"]:
            continue
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * 3 if e["file"] != "residual_dump_all.csv.gz" else e["expected_bytes"] * 2),
                    "basis": "measured R74 return (median per file, expected_bytes) ; per-file ceiling = R74 maximum + 15 %%; %d arms" % len(e["arms"]), "note": e["followup"]})
    for pat, mx, tot in (("r73_f417_classes.csv", 20000, 4000000), ("r73_f417_consistency.csv", 5000, 1000000), ("r75_gates.txt", 400000, 400000), ("identity_controls.txt", 150000, 150000),
                         ("engagement_all.txt", 400000, 400000), ("completeness.csv", 200000, 200000), ("r74_f426_part*", 200000, 800000), ("step0_f426_log.txt", 20000, 20000),
                         ("phase_summary.csv", 3000000, 3000000), ("arm_scan_all.csv.gz", 30000000, 30000000), ("raw_sha256.txt", 20000, 20000), ("R75_tree.tar.gz", 80000000, 80000000),
                         ("agent_edits.patch", 2000000, 2000000), ("*.md", 1000000, 3000000), ("*.json", 2000000, 10000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000),
                         ("*.tsv", 1000000, 3000000), ("*.csv", 1000000, 5000000), ("*.sha256", 100000, 100000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": tot, "basis": "generic ceiling (R74 ledger values or a round figure); not a measurement", "note": "non-arm file"})
    seen = set(); res = []
    for x in out:
        if x["pattern"] not in seen:
            seen.add(x["pattern"]); res.append(x)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r75_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out, table, seen = [], [], set()
    for r in arms():
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v   # None for imu.ds.ds_leaf_size: gen_jobs.py's null-deletion strips the key the base config sets (as R72 to R74)
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r75_ln/%s" % (a.results_root, r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        j["duration_secs"] = r["dur"]
        out.append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j["duration_secs"]), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="", r66_id=r["base"],
                          group=r["group"], batch="r75_ln", sampling=sampling, requires=" ".join(M.requires_of(o, "ln")), f417=str(r["f417"]), plane_check=str(r["plane_check"]), asked=r["asked"]))
    json.dump(out, open(os.path.join(a.out, "jobs_ln.json"), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    sp = M.listener_spec("R75-ln", [{"dir": "ablations/_job_queue/r75_ln", "results_root": "%s/r75_ln" % a.results_root, "expected_jobs": len(out)}], a.results_root, a.min_free_gb, a.workers)
    for blk in sp["stages"] + sp["verify"]:
        blk["cmd"] = [c.replace("%s", "listener_spec_ln.json").replace("r74_manifest", "r75_manifest") for c in blk["cmd"]]
    json.dump(sp, open(os.path.join(a.out, "listener_spec_ln.json"), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(ledger(RO), open(os.path.join(a.out, "r75_file_size_ledger.json"), "w"), indent=1)
    print("ln %d arms (160 + 1 determinism control)" % len(out))


if __name__ == "__main__":
    main()
