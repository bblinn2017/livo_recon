#!/usr/bin/env python3
"""R84 selftest of the collector changes (synthetic arm dirs only, planner sandbox, not evidence about the node).
   python3 -I tools/r69/selftest_r84_collect.py
Checks: the R84 groups T3, L2R, CMP are known groups; T3 and L2R are sweep groups (summaries only, pos_trace_all.csv.gz only when gt_max_mm > 500, a collapsed arm's trace is returned);
CMP and OWN arms return the F1_KEEP files; the new arm_table columns (fit lines edge layer sw_ms tbpn l2_ref l2_source) pass through the collector; an identity arm and a determinism arm
reach identity_controls.txt; a refuse arm (arm.json only) still gets a phase_summary row; R83 extras (point_cov_window, peak_rss, motion-Q) are inert by default; every group of the real
r84 arm_table (when /tmp/claude-0/r84h/H/r84_manifest/arm_table.tsv exists) is a known group."""
import json, os, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r69_collect as C  # noqa: E402
REAL_TABLE = "/tmp/claude-0/r84h/H/r84_manifest/arm_table.tsv"
SHA = "ab" * 32


def mk(runs, arm, files=None, gt_max=1.0, summary=True, sha=SHA):
    d = os.path.join(runs, arm); os.makedirs(d, exist_ok=True)
    if summary:
        json.dump({"arm": arm, "kind": "closed", "meta": {"exit_code": 0}, "gt_max_mm": gt_max, "gt_ate_mm": 3.0, "n_scans": 3}, open(os.path.join(d, "arm_summary.json"), "w"))
        pd.DataFrame({"scan": [1, 2, 3], "t": ["1.0", "2.0", "3.0"], "ep_x_mm": 0.0, "ep_y_mm": 0.0, "ep_z_mm": 0.0}).to_csv(os.path.join(d, "arm_scan.csv"), index=False)
        pd.DataFrame({"scan_id": [1, 2, 3], "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    json.dump({"sentinel": "DONE", "exit_code": 0}, open(os.path.join(d, "arm.json"), "w"))
    if sha:
        open(os.path.join(d, "state_trace_blank_runid.sha256"), "w").write(sha + "\n")
    for k, v in (files or {}).items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(d, k), index=False, compression="gzip" if k.endswith(".gz") else None)
        else:
            open(os.path.join(d, k), "w").write(v)


def collect(runs, tp, out, *extra):
    return subprocess.run([sys.executable, "-I", os.path.join(HERE, "r69_collect.py"), "--runs", runs, "--arms", tp, "--out", out, "--level", "-1", *extra], capture_output=True, text=True)


def main():
    fails = []; n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            fails.append(m)
    # unit level
    ck({"T3", "L2R", "CMP"} <= C.F1_GROUPS, "T3, L2R, CMP are known groups")
    ck({"T3", "L2R"} <= C.SW_GROUPS and "CMP" not in C.SW_GROUPS, "T3 and L2R are sweep groups, CMP is not")
    for g_ in ("T3", "L2R"):
        ck(not C.return_file("residual_dump_all.csv.gz", "a", g_) and not C.return_file("lidar_msg_log_all.csv.gz", "a", g_) and not C.return_file("line_rows_per_scan_all.csv.gz", "a", g_), "%s: summaries only" % g_)
        ck(C.return_file("pos_trace_all.csv.gz", "a", g_, 900.0) and not C.return_file("pos_trace_all.csv.gz", "a", g_, 500.0) and not C.return_file("pos_trace_all.csv.gz", "a", g_, 10.0) and C.return_file("pos_trace_all.csv.gz", "a", g_, None), "%s: trace only when gt_max_mm > 500 (500.0 itself no; missing yes)" % g_)
    ck(all(C.return_file(f, "r84_rt_cmp_eee_01_w5ms_vacc_fq", "CMP") for f in ("pos_trace_all.csv.gz", "lidar_msg_log_all.csv.gz", "line_rows_per_scan_all.csv.gz", "state_reference.txt")), "CMP returns the F1_KEEP files")
    ck(not C.return_file("point_cov_window.csv.gz", "a", "CMP") and not C.return_file("peak_rss.txt", "a", "OWN") and not C.return_file("motion_q_scan_all.csv.gz", "r84_x_mqs_arm", "CMP"), "R83 extras are inert by default")
    ck(C.return_file("pos_trace_all.csv.gz", "a", "SW", 900.0) and not C.return_file("pos_trace_all.csv.gz", "a", "SW", 10.0) and C.return_file("pos_trace_all.csv.gz", "a", "VSW", 900.0), "older sweep groups unchanged")
    # real manifest groups, when present
    if os.path.exists(REAL_TABLE):
        rt = pd.read_csv(REAL_TABLE, sep="\t")
        ck(set(rt["group"].dropna().astype(str)) <= C.F1_GROUPS and {"fit", "lines", "edge", "layer", "sw_ms", "tbpn", "l2_ref", "l2_source"} <= set(rt.columns), "every group of the real arm_table is known, new columns present")
    # collector run on synthetic arms
    t = tempfile.mkdtemp(); runs = os.path.join(t, "runs"); out = os.path.join(t, "out")
    pcw = "scan_index,t_abs\n1,1\n"
    msg = pd.DataFrame({"msg_index": [1, 2], "seq": [1, 2]})
    rowsps = pd.DataFrame({"scan": [1, 2], "n": [3, 4]})
    arms = [  # arm, group, extra columns, files, gt_max, summary
        ("r84_rt_t3_eee_01_pca_off_v0p3_fq", "T3", dict(fit="pca", lines="off", edge=0.3, layer=0, l2_ref="r81_x", l2_source="R81"), {"residual_dump_all.csv.gz": pd.DataFrame({"frame": [1]})}, 10.0, True),
        ("r84_rt_t3_eee_01_deb_on_v0p7_fq", "T3", dict(fit="deb", lines="on", edge=0.7, layer=0, l2_ref="r84_rt_l2_eee_01_deb_on_v0p7_fq", l2_source="R84"), {}, 900.0, True),
        ("r84_rt_l2_eee_01_pca_off_v0p3_fq", "L2R", dict(fit="pca", lines="off", edge=0.3, layer=2), {"lidar_msg_log_all.csv.gz": msg}, 20.0, True),
        ("r84_rt_l2_eee_01_deb_off_v0p6_fq", "L2R", dict(fit="deb", lines="off", edge=0.6, layer=2), {}, 750.0, True),
        ("r84_rt_cmp_eee_01_w5ms_vacc_fq", "CMP", dict(fit="pca", lines="off", edge=0.425, layer=2, sw_ms=5.0, tbpn="var_acc", l2_ref="r83_rt_ml_eee_01_l2_fq", l2_source="R83", same_batch_control="r84_rt_cmp_eee_01_w0ms_vacc_fq", control="r82_rt_ds_eee_01_first_fq"),
         {"lidar_msg_log_all.csv.gz": msg, "line_rows_per_scan_all.csv.gz": rowsps, "state_reference.txt": "ref\n", "residual_dump_all.csv.gz": pd.DataFrame({"frame": [1]}), "point_cov_window.csv": pcw, "peak_rss.txt": "peak_rss_kb 1\n",
          "motion_q_scan_all.csv.gz": pd.DataFrame({"scan_index": [1]})}, 4.0, True),
        ("r84_own_det_x_eee_01", "OWN", dict(arm_kind="det_fb", ref="r84_rt_cmp_eee_01_w5ms_vacc_fq"), {}, 4.0, True),
        ("r84_rt_id_eee_01_fq", "OWN", dict(arm_kind="id", ref="r82_rt_ds_eee_01_first_fq"), {}, 4.0, True),
        ("r84_dc_refuse_sample_window_big", "OWN", dict(arm_kind="refuse", needles="sample_window_s"), {}, None, False),
    ]
    recs = []
    for arm, g_, ex, files, gt, summ in arms:
        mk(runs, arm, files, gt_max=gt, summary=summ, sha=(SHA if arm != "r84_dc_refuse_sample_window_big" else None))
        r_ = dict(arm=arm, tier="FB", kind="closed", duration_s=("5" if "refuse" in arm else "full"), seq="eee_01", keep_raw=0, group=g_, dup_of="", notes="")
        r_.update(ex); recs.append(r_)
    tab = pd.DataFrame(recs)
    tp = os.path.join(t, "arm_table.tsv"); tab.to_csv(tp, sep="\t", index=False)
    refc = os.path.join(t, "ref_completeness.csv")
    pd.DataFrame([dict(arm="r82_rt_ds_eee_01_first_fq", state_trace_sha256_run_id_blanked=SHA)]).to_csv(refc, index=False)
    r = collect(runs, tp, out, "--ref-completeness", refc)
    ck(r.returncode == 0, "collector exit %s: %s" % (r.returncode, r.stderr[-400:]))
    has = lambda a, f: os.path.exists(os.path.join(out, "compact", a, f))
    ck(not has("r84_rt_t3_eee_01_pca_off_v0p3_fq", "pos_trace_all.csv.gz") and not has("r84_rt_t3_eee_01_pca_off_v0p3_fq", "residual_dump_all.csv.gz"), "T3 gt_max 10: no per-arm file returned")
    ck(has("r84_rt_t3_eee_01_deb_on_v0p7_fq", "pos_trace_all.csv.gz"), "T3 collapsed arm (gt_max 900): trace returned")
    ck(not has("r84_rt_l2_eee_01_pca_off_v0p3_fq", "pos_trace_all.csv.gz") and not has("r84_rt_l2_eee_01_pca_off_v0p3_fq", "lidar_msg_log_all.csv.gz"), "L2R gt_max 20: summaries only")
    ck(has("r84_rt_l2_eee_01_deb_off_v0p6_fq", "pos_trace_all.csv.gz"), "L2R collapsed arm (gt_max 750): trace returned")
    c = "r84_rt_cmp_eee_01_w5ms_vacc_fq"
    ck(all(has(c, f) for f in ("pos_trace_all.csv.gz", "lidar_msg_log_all.csv.gz", "line_rows_per_scan_all.csv.gz", "state_reference.txt")), "CMP arm: F1_KEEP files returned")
    ck(not has(c, "point_cov_window.csv.gz") and not has(c, "peak_rss.txt") and not has(c, "motion_q_scan_all.csv.gz"), "CMP arm: R83 extras not returned")
    ps = pd.read_csv(os.path.join(out, "phase_summary.csv"))
    ck(set(ps["arm"]) == {a[0] for a in arms}, "phase_summary.csv has a row for every arm incl. the refuse arm (arm.json only): %s" % sorted(ps["arm"]))
    sc = os.path.join(out, "arm_scan_all.csv.gz")
    if os.path.exists(sc):
        sa = set(pd.read_csv(sc)["arm"])
        ck(not any("_t3_" in a or "_l2_" in a for a in sa) and c in sa, "arm_scan_all.csv.gz: no sweep (T3/L2R) arm, CMP arm present")
    else:
        ck(False, "arm_scan_all.csv.gz not written")
    ic = open(os.path.join(out, "identity_controls.txt")).read()
    ck("IDENTICAL" in ic and "determinism: " + SHA in ic and "r84_own_det_x_eee_01" in ic and "identity: " + SHA in ic and "r84_rt_id_eee_01_fq" in ic, "identity_controls.txt lists the determinism pair and the identity arm as IDENTICAL: %s" % ic[-500:])
    cp = pd.read_csv(os.path.join(out, "completeness.csv"))
    ck(bool(cp.set_index("arm").loc["r84_dc_refuse_sample_window_big", "has_dir"]) and not bool(cp.set_index("arm").loc["r84_dc_refuse_sample_window_big", "has_summary"]), "refuse arm: dir present, no summary, listed in completeness.csv")
    ck(all(os.path.exists(os.path.join(out, f)) for f in ("completeness.csv", "summaries_all.json", "compress_manifests.json", "sensor_stamps_summaries.json")), "the usual collector outputs")
    print("SELFTEST r84_collect", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (n[0], len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
