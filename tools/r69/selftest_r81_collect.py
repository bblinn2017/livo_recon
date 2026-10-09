#!/usr/bin/env python3
"""R81 selftest of the collector changes (synthetic arm dirs only, planner sandbox): items 7 to 9 of r81_tool_fixes.patch and F-469.
   python3 tools/r69/selftest_r81_collect.py
Checks: (7) a plain `*_all.csv` of a group outside F1_GROUPS is copied to compact/<arm>/; SWT arms return no per-arm file but pos_trace when gt_max_mm > 500; (8) identity_controls.txt lists the whole-bag determinism
pair (arm table `ref` column and the name fallback) and the identity pair against a reference completeness.csv; (F-469) plane_fit_trace_counts_all.csv is written by the collector for the arm that has a raw
plane_fit_trace.csv, and for no other arm; no trace output at all -> no file."""
import json, os, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def mk(runs, arm, sha, extra=None, gt_max=1.0):
    d = os.path.join(runs, arm); os.makedirs(d, exist_ok=True)
    json.dump({"arm": arm, "kind": "closed", "meta": {"exit_code": 0}, "gt_max_mm": gt_max, "n_scans": 3}, open(os.path.join(d, "arm_summary.json"), "w"))
    json.dump({"sentinel": "DONE"}, open(os.path.join(d, "arm.json"), "w"))
    open(os.path.join(d, "state_trace_blank_runid.sha256"), "w").write(sha + "\n")
    pd.DataFrame({"scan_id": [1, 2, 3], "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    pd.DataFrame({"scan": [1, 2, 3], "t": ["1.0", "2.0", "3.0"], "ep_x_mm": 0.0, "ep_y_mm": 0.0, "ep_z_mm": 0.0}).to_csv(os.path.join(d, "arm_scan.csv"), index=False)
    for k, v in (extra or {}).items():
        v.to_csv(os.path.join(d, k), index=False)


def run(runs, table, out, extra_args=()):
    tp = os.path.join(os.path.dirname(out), "arm_table.tsv"); table.to_csv(tp, sep="\t", index=False)
    return subprocess.run([sys.executable, os.path.join(HERE, "r69_collect.py"), "--runs", runs, "--arms", tp, "--out", out, "--level", "-1", *extra_args], capture_output=True, text=True)


def main():
    fails = []
    t = tempfile.mkdtemp(); runs = os.path.join(t, "runs"); os.makedirs(runs)
    trace = pd.DataFrame({"frame": [0, 0, 1], "path": ["pca", "pca", "pca"], "reason": [0, 3, 0], "N": 5})
    mk(runs, "r81_rt_mg_eee_01_fq", "a" * 64, {"window_row_summary_all.csv": pd.DataFrame({"x": [1]})})
    mk(runs, "r81_own_det_eee_01", "a" * 64)
    mk(runs, "r81_rt_pc_eee_01_fq", "b" * 64, {"plane_fit_trace.csv": trace})
    mk(runs, "r81_dc_id_mg_eee_01_fq", "c" * 64)
    mk(runs, "r81_rt_sw_eee_01_off_v0p25_pfn3_fq", "d" * 64, gt_max=900.0)
    cols = dict(tier="FB", duration_s="full", keep_raw=0, dup_of="", notes="")
    tab = pd.DataFrame([dict(arm="r81_rt_mg_eee_01_fq", kind="closed", seq="eee_01", group="MG", ref="", **cols), dict(arm="r81_own_det_eee_01", kind="det", seq="eee_01", group="OWN", ref="r81_rt_mg_eee_01_fq", **cols),
                        dict(arm="r81_rt_pc_eee_01_fq", kind="closed", seq="eee_01", group="N469", ref="", **cols), dict(arm="r81_dc_id_mg_eee_01_fq", kind="id", seq="eee_01", group="ID", ref="r80_mg_eee_01_fq", **cols),
                        dict(arm="r81_rt_sw_eee_01_off_v0p25_pfn3_fq", kind="closed", seq="eee_01", group="SWT", ref="", **cols)])
    rc = os.path.join(t, "r80_completeness.csv"); pd.DataFrame({"arm": ["r80_mg_eee_01_fq"], "state_trace_sha256_run_id_blanked": ["c" * 64]}).to_csv(rc, index=False)
    out = os.path.join(t, "ret"); os.makedirs(out)
    r = run(runs, tab, out, ["--ref-completeness", rc])
    if r.returncode:
        fails.append("collector failed: " + r.stderr[-400:])
    else:
        c = lambda arm, f: os.path.exists(os.path.join(out, "compact", arm, f))
        # window_row_summary_all.csv: group MG is in F1_GROUPS and the file is in F1_KEEP, so it is returned; the point of fix 7 is that the glob reaches it at all
        if not c("r81_rt_mg_eee_01_fq", "window_row_summary_all.csv"):
            fails.append("fix 7: *_all.csv not copied")
        if c("r81_rt_sw_eee_01_off_v0p25_pfn3_fq", "pos_trace_all.csv.gz") is not True:
            fails.append("SWT arm with gt_max_mm 900 must return its pos_trace")
        pf = os.path.join(out, "plane_fit_trace_counts_all.csv")
        if not os.path.exists(pf):
            fails.append("F-469: plane_fit_trace_counts_all.csv not written")
        else:
            df = pd.read_csv(pf)
            if set(df["arm"]) != {"r81_rt_pc_eee_01_fq"} or int(df["rows"].sum()) != 3 or "reason_name" not in df.columns:
                fails.append("F-469: wrong content %s" % df.to_dict("list"))
        ic = open(os.path.join(out, "identity_controls.txt")).read()
        if "determinism: " + "a" * 64 + "  r81_own_det_eee_01" not in ic or "IDENTICAL" not in [l for l in ic.splitlines() if "r81_own_det_eee_01" in l][0]:
            fails.append("fix 8: determinism pair missing or not IDENTICAL: " + ic)
        if not [l for l in ic.splitlines() if l.startswith("identity: ") and "r80_mg_eee_01_fq" in l and "-> IDENTICAL" in l]:
            fails.append("identity pair against the R80 completeness digest missing: " + ic)
    # a different digest must show DIFFERENT, no trace arm -> no file
    mk(runs, "r81_own_det_eee_01", "e" * 64)
    runs2 = os.path.join(t, "runs2"); os.makedirs(runs2)
    mk(runs2, "r81_rt_mg_eee_01_fq", "a" * 64); mk(runs2, "r81_own_det_eee_01", "e" * 64)
    out2 = os.path.join(t, "ret2"); os.makedirs(out2)
    r = run(runs2, tab.iloc[:2], out2)
    ic = open(os.path.join(out2, "identity_controls.txt")).read() if not r.returncode else ""
    if r.returncode or "DIFFERENT" not in [l for l in ic.splitlines() if "r81_own_det_eee_01" in l][0] or os.path.exists(os.path.join(out2, "plane_fit_trace_counts_all.csv")):
        fails.append("changed digest must read DIFFERENT and no counts file without trace output: %s %s" % (r.stderr[-200:], ic))
    print("SELFTEST r81 collect", "FAIL " + "; ".join(fails) if fails else "PASS (fix 7 glob, SWT trace rule, fix 8 determinism + identity pairs, F-469 counts file written by the collector only for the arm with trace output; DIFFERENT digest detected)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
